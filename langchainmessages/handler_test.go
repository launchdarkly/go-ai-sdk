package langchainmessages

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/anthropic"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/langchainwire"
)

var bg = context.Background()

func cfg(extra func(*aiserver.AiConfig)) aiserver.AiConfig {
	c := aiserver.AiConfig{Model: aiserver.Model{Name: "test-model"}}
	c.Provider.Name = "Anthropic"
	if extra != nil {
		extra(&c)
	}
	return c
}

func instr(s string) aiserver.AiConfig {
	return cfg(func(c *aiserver.AiConfig) { c.Instructions = s })
}

func text(m llms.MessageContent) string {
	var b strings.Builder
	for _, p := range m.Parts {
		if t, ok := p.(llms.TextContent); ok {
			b.WriteString(t.Text)
		}
	}
	return b.String()
}

func roles(msgs []llms.MessageContent) []llms.ChatMessageType {
	out := make([]llms.ChatMessageType, 0, len(msgs))
	for _, m := range msgs {
		out = append(out, m.Role)
	}
	return out
}

const (
	sys   = llms.ChatMessageTypeSystem
	human = llms.ChatMessageTypeHuman
	ai    = llms.ChatMessageTypeAI
	tool  = llms.ChatMessageTypeTool
)

func spanAttrs(s sdktrace.ReadOnlySpan) map[string]string {
	out := map[string]string{}
	for _, a := range s.Attributes() {
		out[string(a.Key)] = a.Value.Emit()
	}
	return out
}

func spansByName(exp *tracetest.InMemoryExporter) map[string][]sdktrace.ReadOnlySpan {
	out := map[string][]sdktrace.ReadOnlySpan{}
	for _, s := range exp.GetSpans().Snapshots() {
		out[s.Name()] = append(out[s.Name()], s)
	}
	return out
}

func hasEvent(s sdktrace.ReadOnlySpan, name, contains string) bool {
	for _, e := range s.Events() {
		if e.Name != name {
			continue
		}
		for _, a := range e.Attributes {
			if strings.Contains(a.Value.Emit(), contains) {
				return true
			}
		}
	}
	return false
}

// factory metadata: providesFor, captureContent default, independent instances.
func TestFactoryMetadata(t *testing.T) {
	a, b := newHandler(Options{}), newHandler(Options{})
	if a.ProvidesFor != (aiserver.ProvidesFor{Provider: "*", Mode: aiserver.ModeMessages}) {
		t.Errorf("providesFor = %+v", a.ProvidesFor)
	}
	if a.CaptureContent != aiserver.CaptureDefault || a == b || a.Invoke == nil || a.Stream == nil {
		t.Errorf("factory metadata wrong: %+v", a)
	}
	if newHandler(Options{CaptureContent: aiserver.CaptureOn}).CaptureContent != aiserver.CaptureOn {
		t.Error("captureContent not passed through")
	}
}

// instructions path; an unresolved placeholder is left as written.
func TestPromptPathA(t *testing.T) {
	h, f, _ := setup(t, textTurn("a", 1, 1), textTurn("b", 1, 1))
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: instr("Hi {{name}} {{missing}}"), UserInput: "q",
		Variables: map[string]any{"name": "Al"}}); err != nil {
		t.Fatal(err)
	}
	m := f.calls[0].msgs
	if !reflect.DeepEqual(roles(m), []llms.ChatMessageType{sys, human}) || text(m[0]) != "Hi Al {{missing}}" || text(m[1]) != "q" {
		t.Errorf("path A messages = %+v", m)
	}
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: instr("Hello {{missing}}")}); err != nil {
		t.Fatal(err)
	}
	if got := text(f.calls[1].msgs[0]); got != "Hello {{missing}}" {
		t.Errorf("unresolved placeholder: %q", got)
	}
}

// messages path: the last user message or the appended input closes the prompt.
func TestPromptPathB(t *testing.T) {
	h, f, _ := setup(t, textTurn("a", 1, 1), textTurn("b", 1, 1))
	vars := map[string]any{"x": "1"}
	lastUser := cfg(func(c *aiserver.AiConfig) {
		c.Messages = []aiserver.Message{{Role: "system", Text: "sys {{x}}"}, {Role: "system", Text: "more"},
			{Role: "assistant", Text: "hi"}, {Role: "user", Text: "u {{x}}"}}
	})
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: lastUser, UserInput: "ignored", Variables: vars}); err != nil {
		t.Fatal(err)
	}
	m := f.calls[0].msgs
	if !reflect.DeepEqual(roles(m), []llms.ChatMessageType{sys, ai, human}) || text(m[0]) != "sys 1\nmore" || text(m[2]) != "u 1" {
		t.Errorf("last-user branch = %+v", m)
	}
	lastAsst := cfg(func(c *aiserver.AiConfig) {
		c.Messages = []aiserver.Message{{Role: "user", Text: "u"}, {Role: "assistant", Text: "a"}}
	})
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: lastAsst, UserInput: "q"}); err != nil {
		t.Fatal(err)
	}
	m = f.calls[1].msgs
	if !reflect.DeepEqual(roles(m), []llms.ChatMessageType{human, ai, human}) || text(m[2]) != "q" {
		t.Errorf("last-assistant branch = %+v", m)
	}
}

// empty input and instructions-versus-messages precedence.
func TestPromptPathC(t *testing.T) {
	h, f, _ := setup(t, textTurn("a", 1, 1), textTurn("b", 1, 1), textTurn("c", 1, 1))
	// Instructions path with empty input: the user turn is still appended, as "".
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: instr("x")}); err != nil {
		t.Fatal(err)
	}
	m := f.calls[0].msgs
	if !reflect.DeepEqual(roles(m), []llms.ChatMessageType{sys, human}) || text(m[1]) != "" {
		t.Errorf("empty input on instructions path = %+v", m)
	}
	// Both present: messages win, instructions are not injected.
	both := cfg(func(c *aiserver.AiConfig) {
		c.Instructions = "IGNORED"
		c.Messages = []aiserver.Message{{Role: "system", Text: "from messages"}, {Role: "user", Text: "u"}}
	})
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: both}); err != nil {
		t.Fatal(err)
	}
	m = f.calls[1].msgs
	if !reflect.DeepEqual(roles(m), []llms.ChatMessageType{sys, human}) || text(m[0]) != "from messages" {
		t.Errorf("both present = %+v", m)
	}
	// Messages path, last non-system message assistant, empty input: "" appended.
	asst := cfg(func(c *aiserver.AiConfig) { c.Messages = []aiserver.Message{{Role: "assistant", Text: "a"}} })
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: asst}); err != nil {
		t.Fatal(err)
	}
	m = f.calls[2].msgs
	if !reflect.DeepEqual(roles(m), []llms.ChatMessageType{ai, human}) || text(m[1]) != "" {
		t.Errorf("empty input after assistant = %+v", m)
	}
}

// history as structured messages, including images.
func TestHistory(t *testing.T) {
	h, f, _ := setup(t, textTurn("a", 1, 1), textTurn("b", 1, 1), textTurn("c", 1, 1), textTurn("d", 1, 1))
	base := cfg(func(c *aiserver.AiConfig) {
		c.Messages = []aiserver.Message{{Role: "system", Text: "s"}, {Role: "user", Text: "cfg-u"}, {Role: "assistant", Text: "cfg-a"}}
	})
	hist := []aiserver.Message{
		{Role: "system", Text: "drop me"},
		{Role: "user", Blocks: []aiserver.ContentBlock{
			{Type: "text", Text: "look"},
			{Type: "image", Source: &aiserver.ImageSource{Type: "base64", MediaType: "image/png", Data: "AAEC"}},
			{Type: "image", Source: &aiserver.ImageSource{Type: "url", URL: "https://x/y.png"}},
		}},
		{Role: "assistant", Text: "h-a"},
	}
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: base, UserInput: "q", History: hist}); err != nil {
		t.Fatal(err)
	}
	m := f.calls[0].msgs
	if !reflect.DeepEqual(roles(m), []llms.ChatMessageType{sys, human, ai, human, ai, human}) {
		t.Fatalf("history order = %v", roles(m))
	}
	if text(m[1]) != "cfg-u" || text(m[4]) != "h-a" || text(m[5]) != "q" || text(m[0]) != "s" {
		t.Errorf("history content = %+v", m)
	}
	parts := m[3].Parts
	bin, ok1 := parts[1].(llms.BinaryContent)
	url, ok2 := parts[2].(llms.ImageURLContent)
	if !ok1 || bin.MIMEType != "image/png" || !reflect.DeepEqual(bin.Data, []byte{0, 1, 2}) || !ok2 || url.URL != "https://x/y.png" {
		t.Errorf("image blocks not mapped natively: %#v", parts)
	}

	// Instructions path: history before userInput.
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: instr("x"), UserInput: "q", History: []aiserver.Message{{Role: "user", Text: "h"}}}); err != nil {
		t.Fatal(err)
	}
	if got := roles(f.calls[1].msgs); !reflect.DeepEqual(got, []llms.ChatMessageType{sys, human, human}) || text(f.calls[1].msgs[1]) != "h" {
		t.Errorf("instructions + history = %v", got)
	}

	// Empty history is identical to no history.
	_, _ = h.Invoke(bg, aiserver.HandlerRequest{Config: base, UserInput: "q", History: []aiserver.Message{}})
	_, _ = h.Invoke(bg, aiserver.HandlerRequest{Config: base, UserInput: "q"})
	if !reflect.DeepEqual(f.calls[2].msgs, f.calls[3].msgs) {
		t.Errorf("empty history differs: %v vs %v", f.calls[2].msgs, f.calls[3].msgs)
	}
}

func TestHistoryEmptyInputEndingInUser(t *testing.T) {
	h, f, _ := setup(t, textTurn("a", 1, 1))
	hist := []aiserver.Message{{Role: "user", Blocks: []aiserver.ContentBlock{{Type: "text", Text: "already here"}}}}
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: instr("x"), History: hist}); err != nil {
		t.Fatal(err)
	}
	if got := roles(f.calls[0].msgs); !reflect.DeepEqual(got, []llms.ChatMessageType{sys, human}) {
		t.Errorf("extra user turn appended: %v", got)
	}
}

func okTool(name string, calls *[]string, gotArgs *map[string]any, result any) aiserver.ToolFunc {
	return func(_ context.Context, args map[string]any) (any, error) {
		*calls = append(*calls, name)
		if gotArgs != nil {
			*gotArgs = args
		}
		return result, nil
	}
}

var objSchema = map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}, "required": []any{"city"}}

func threeTools(c *aiserver.AiConfig) {
	c.Instructions = "x"
	c.Tools = map[string]aiserver.Tool{
		"a": {Name: "a", Type: "function", Description: "tool a", Parameters: objSchema},
		"b": {Name: "b", Type: "function", Description: "tool b", Parameters: map[string]any{"type": "object"}},
		"c": {Name: "c", Type: "function", Description: "tool c", Parameters: map[string]any{"type": "object"}},
	}
}

// tool conversion and filtering, including no tools in config.
func TestToolConversion(t *testing.T) {
	var calls []string
	h, f, _ := setup(t, textTurn("1", 1, 1), textTurn("2", 1, 1), textTurn("3", 1, 1), textTurn("4", 1, 1))
	all := map[string]aiserver.ToolHandler{"a": okTool("a", &calls, nil, ""), "b": okTool("b", &calls, nil, ""), "c": okTool("c", &calls, nil, "")}
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: cfg(threeTools), ToolHandlers: all}); err != nil {
		t.Fatal(err)
	}
	sent := f.calls[0].opts.Tools
	if len(sent) != 3 {
		t.Fatalf("want 3 tools, got %d", len(sent))
	}
	for i, name := range []string{"a", "b", "c"} {
		fn := sent[i].Function
		if sent[i].Type != "function" || fn.Name != name || fn.Description != "tool "+name {
			t.Errorf("tool %d = %+v", i, fn)
		}
	}
	if !reflect.DeepEqual(sent[0].Function.Parameters, objSchema) {
		t.Errorf("parameters modified: %v", sent[0].Function.Parameters)
	}

	two := map[string]aiserver.ToolHandler{"a": all["a"], "b": all["b"], "c": &aiserver.NativeTool{ToolName: "c"}}
	_, _ = h.Invoke(bg, aiserver.HandlerRequest{Config: cfg(threeTools), ToolHandlers: two})
	if n := len(f.calls[1].opts.Tools); n != 2 {
		t.Errorf("3 config tools, 2 handlers: sent %d", n)
	}
	one := cfg(func(c *aiserver.AiConfig) {
		c.Instructions = "x"
		c.Tools = map[string]aiserver.Tool{"z": {Name: "z", Type: "function", Parameters: map[string]any{}}}
	})
	_, _ = h.Invoke(bg, aiserver.HandlerRequest{Config: one})
	if f.calls[2].opts.Tools != nil {
		t.Errorf("tool without handler forwarded: %v", f.calls[2].opts.Tools)
	}
	_, _ = h.Invoke(bg, aiserver.HandlerRequest{Config: instr("x"), ToolHandlers: all})
	if f.calls[3].opts.Tools != nil || len(calls) != 0 || len(f.calls) != 4 {
		t.Errorf("no config tools: tools=%v calls=%v modelCalls=%d", f.calls[3].opts.Tools, calls, len(f.calls))
	}
}

// tool loop and blocking-path telemetry.
func TestToolLoopAndTelemetry(t *testing.T) {
	h, f, exp := setup(t, toolTurn("a", map[string]any{"city": "Paris"}), toolTurn("b", map[string]any{}), textTurn("final", 2, 3))
	var calls []string
	var gotArgs map[string]any
	tools := map[string]aiserver.ToolHandler{
		"a": okTool("a", &calls, &gotArgs, "sunny"),
		"b": okTool("b", &calls, nil, map[string]any{"ok": true}),
	}
	res, err := h.Invoke(bg, aiserver.HandlerRequest{Config: cfg(threeTools), UserInput: "q", ToolHandlers: tools,
		Variables: map[string]any{"__ld": aiserver.TrackData{ConfigKey: "test-config", VariationKey: "variation-a", RunID: "run-123",
			Version: 1, ModelName: "test-model", ProviderName: "TestProvider"}}})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(calls, []string{"a", "b"}) || gotArgs["city"] != "Paris" || res.Output != "final" {
		t.Errorf("calls=%v args=%v out=%v", calls, gotArgs, res.Output)
	}
	if u := aiserver.ParseUsage(res.Usage); u.Input != 4 || u.Output != 5 {
		t.Errorf("usage = %+v", u)
	}
	// Turn 3 sees: system, user, AI(tool call a), tool result a, AI(tool call b), tool result b.
	m := f.calls[2].msgs
	if !reflect.DeepEqual(roles(m), []llms.ChatMessageType{sys, human, ai, tool, ai, tool}) {
		t.Fatalf("loop messages = %v", roles(m))
	}
	if tr := m[3].Parts[0].(llms.ToolCallResponse); tr.ToolCallID != "call_a" || tr.Content != "sunny" || tr.Name != "a" {
		t.Errorf("tool result a = %+v", tr)
	}
	if tr := m[5].Parts[0].(llms.ToolCallResponse); tr.Content != `{"ok":true}` {
		t.Errorf("tool result b = %+v", tr)
	}
	if tc, ok := m[2].Parts[0].(llms.ToolCall); !ok || tc.ID != "call_a" {
		t.Errorf("assistant tool call not replayed: %#v", m[2].Parts)
	}

	spans := spansByName(exp)
	if len(spans[SpanName]) != 1 || len(spans["chat test-model"]) != 3 || len(spans["execute_tool a"]) != 1 || len(spans["execute_tool b"]) != 1 {
		t.Fatalf("span tree = %v", spans)
	}
	root := spans[SpanName][0]
	attrs := spanAttrs(root)
	for k, want := range map[string]string{
		"gen_ai.system": "anthropic", "gen_ai.operation.name": "chat", "gen_ai.request.model": "test-model",
		"gen_ai.response.model": "test-model", "gen_ai.usage.input_tokens": "4", "gen_ai.usage.output_tokens": "5",
		"gen_ai.usage.total_tokens": "9", "gen_ai.usage.prompt_tokens": "4", "gen_ai.usage.completion_tokens": "5",
		"gen_ai.prompt.0.role": "system", "gen_ai.prompt.0.content": "x", "gen_ai.prompt.1.role": "user",
		"gen_ai.prompt.1.content": "q", "gen_ai.completion.0.role": "assistant", "gen_ai.completion.0.content": "final",
		"launchdarkly.operation.type": "gen_ai", "launchdarkly.config.key": "test-config",
		"launchdarkly.variation.key": "variation-a", "launchdarkly.run.id": "run-123",
	} {
		if attrs[k] != want {
			t.Errorf("root %s = %q, want %q", k, attrs[k], want)
		}
	}
	if _, has := attrs["launchdarkly.graph.key"]; has {
		t.Error("graph key set without graphKey")
	}
	if root.Status().Code != codes.Ok || !hasEvent(root, "gen_ai.content.prompt", "q") || !hasEvent(root, "gen_ai.content.completion", "final") {
		t.Errorf("root status %v / content events missing", root.Status())
	}
	for name, ss := range spans {
		for _, s := range ss {
			if name == SpanName {
				continue
			}
			if s.Parent().SpanID() != root.SpanContext().SpanID() {
				t.Errorf("%s is not a child of the root", name)
			}
			for k := range spanAttrs(s) {
				if strings.HasPrefix(k, "launchdarkly.config") || strings.HasPrefix(k, "context.") {
					t.Errorf("child %s carries %s", name, k)
				}
			}
		}
	}
}

func TestGraphKeyAttribute(t *testing.T) {
	h, _, exp := setup(t, textTurn("ok", 1, 1))
	_, _ = h.Invoke(bg, aiserver.HandlerRequest{Config: instr("x"), Variables: map[string]any{"__ld": aiserver.TrackData{ConfigKey: "k", GraphKey: "g"}}})
	if got := spanAttrs(spansByName(exp)[SpanName][0])["launchdarkly.graph.key"]; got != "g" {
		t.Errorf("graph key = %q", got)
	}
}

// tool errors and provider errors propagate.
func TestErrors(t *testing.T) {
	boom := errors.New("tool boom")
	h, _, exp := setup(t, toolTurn("a", nil))
	_, err := h.Invoke(bg, aiserver.HandlerRequest{Config: cfg(threeTools), ToolHandlers: map[string]aiserver.ToolHandler{
		"a": aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) { return nil, boom })}})
	if !errors.Is(err, boom) {
		t.Errorf("tool error not propagated: %v", err)
	}
	if s := spansByName(exp)["execute_tool a"]; len(s) != 1 || s[0].Status().Code != codes.Error {
		t.Errorf("tool span = %v", s)
	}

	provider := errors.New("provider down")
	h, _, exp = setup(t, scripted{err: provider})
	_, err = h.Invoke(bg, aiserver.HandlerRequest{Config: instr("x")})
	if !errors.Is(err, provider) {
		t.Fatalf("want provider error, got %v", err)
	}
	snaps := exp.GetSpans().Snapshots()
	if len(snaps) != 2 {
		t.Fatalf("want root + chat ended, got %d spans", len(snaps))
	}
	for _, s := range snaps {
		if s.Status().Code != codes.Error || len(s.Events()) == 0 || s.EndTime().IsZero() {
			t.Errorf("%s: status %v events %d", s.Name(), s.Status().Code, len(s.Events()))
		}
	}

	h, _, _ = setup(t, toolTurn("zzz", nil))
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: cfg(threeTools), ToolHandlers: map[string]aiserver.ToolHandler{
		"a": aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) { return "", nil })}}); !errors.Is(err, ErrNoToolHandler) {
		t.Errorf("unknown tool: %v", err)
	}

	loop := make([]scripted, maxSteps+2)
	for i := range loop {
		loop[i] = toolTurn("a", nil)
	}
	h, _, _ = setup(t, loop...)
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: cfg(threeTools), ToolHandlers: map[string]aiserver.ToolHandler{
		"a": aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) { return "", nil })}}); !errors.Is(err, ErrMaxSteps) {
		t.Errorf("runaway loop: %v", err)
	}
}

// streaming: deltas, the tool loop and done.
func TestStreamToolLoop(t *testing.T) {
	toolChunk := `[{"id":"call_a","type":"function","function":{"name":"a","arguments":""}}]`
	h, _, exp := setup(t,
		streamTurn([]string{"Let me ", "check. ", toolChunk}, 3, 2, toolCall("a", map[string]any{"city": "Paris"})),
		streamTurn([]string{"It is ", "sunny."}, 5, 4))
	var gotArgs map[string]any
	var calls []string
	var events []aiserver.HandlerStreamEvent
	var chunksAtTool int
	tools := map[string]aiserver.ToolHandler{"a": aiserver.ToolFunc(func(_ context.Context, args map[string]any) (any, error) {
		calls = append(calls, "a")
		gotArgs = args
		chunksAtTool = len(events)
		return "sunny", nil
	})}
	for ev, err := range h.Stream(bg, aiserver.HandlerRequest{Config: cfg(threeTools), UserInput: "q", ToolHandlers: tools,
		Variables: map[string]any{"__ld": aiserver.TrackData{ConfigKey: "test-config", VariationKey: "variation-a", RunID: "run-123"}}}) {
		if err != nil {
			t.Fatal(err)
		}
		events = append(events, ev)
	}
	var chunks []string
	doneAt := -1
	for i, ev := range events {
		switch ev.Type {
		case "chunk":
			if doneAt >= 0 {
				t.Error("chunk after done")
			}
			chunks = append(chunks, ev.Text)
		case "done":
			if doneAt >= 0 {
				t.Error("two done events")
			}
			doneAt = i
		}
	}
	if !reflect.DeepEqual(chunks, []string{"Let me ", "check. ", "It is ", "sunny."}) || doneAt != len(events)-1 {
		t.Fatalf("events = %+v", events)
	}
	if chunksAtTool != 2 || !reflect.DeepEqual(calls, []string{"a"}) || gotArgs["city"] != "Paris" {
		t.Errorf("tool ran at %d with %v", chunksAtTool, gotArgs)
	}
	done := events[doneAt]
	if *done.Output != "Let me check. It is sunny." {
		t.Errorf("done.output = %q", *done.Output)
	}
	if done.Usage["input_tokens"] != float64(8) || done.Usage["output_tokens"] != float64(6) {
		t.Errorf("done.usage = %v", done.Usage)
	}

	spans := spansByName(exp)
	if len(spans[StreamSpanName]) != 1 || len(spans[SpanName]) != 0 || len(spans["chat test-model"]) != 2 || len(spans["execute_tool a"]) != 1 {
		t.Fatalf("stream span tree = %v", spans)
	}
	root := spans[StreamSpanName][0]
	attrs := spanAttrs(root)
	for k, want := range map[string]string{
		"gen_ai.system": "anthropic", "gen_ai.operation.name": "chat", "gen_ai.request.model": "test-model",
		"gen_ai.response.model": "test-model", "gen_ai.usage.input_tokens": "8", "gen_ai.usage.output_tokens": "6",
		"gen_ai.usage.total_tokens": "14", "gen_ai.usage.prompt_tokens": "8", "gen_ai.usage.completion_tokens": "6",
		"gen_ai.prompt.1.content": "q", "gen_ai.completion.0.content": "Let me check. It is sunny.",
		"launchdarkly.operation.type": "gen_ai", "launchdarkly.config.key": "test-config",
		"launchdarkly.variation.key": "variation-a", "launchdarkly.run.id": "run-123",
	} {
		if attrs[k] != want {
			t.Errorf("stream root %s = %q, want %q", k, attrs[k], want)
		}
	}
	if root.Status().Code != codes.Ok || root.EndTime().IsZero() || !hasEvent(root, "gen_ai.content.prompt", "q") || !hasEvent(root, "gen_ai.content.completion", "sunny") {
		t.Errorf("stream root status/events wrong")
	}
}

func TestStreamErrorAndAbandon(t *testing.T) {
	provider := errors.New("stream broke")
	h, _, exp := setup(t, scripted{err: provider})
	var got error
	for _, err := range h.Stream(bg, aiserver.HandlerRequest{Config: instr("x")}) {
		got = err
	}
	if !errors.Is(got, provider) {
		t.Fatalf("stream error = %v", got)
	}
	for _, s := range exp.GetSpans().Snapshots() {
		if s.Status().Code != codes.Error || len(s.Events()) == 0 || s.EndTime().IsZero() {
			t.Errorf("%s: status %v", s.Name(), s.Status().Code)
		}
	}

	h, f, exp := setup(t, streamTurn([]string{"a", "b", "c"}, 1, 1))
	for range h.Stream(bg, aiserver.HandlerRequest{Config: instr("x")}) {
		break
	}
	if len(f.calls) != 1 {
		t.Errorf("model called %d times after abandonment", len(f.calls))
	}
	snaps := exp.GetSpans().Snapshots()
	if len(snaps) != 2 {
		t.Fatalf("abandoned spans = %d", len(snaps))
	}
	for _, s := range snaps {
		if spanAttrs(s)["launchdarkly.stream.abandoned"] != "true" || s.Status().Code != codes.Unset {
			t.Errorf("%s: attrs=%v status=%v", s.Name(), spanAttrs(s), s.Status().Code)
		}
	}
}

// A model that ignores StreamingFunc still yields its text as one chunk.
func TestStreamNonStreamingModel(t *testing.T) {
	h, _, _ := setup(t, textTurn("whole", 1, 1))
	var evs []aiserver.HandlerStreamEvent
	for ev, err := range h.Stream(bg, aiserver.HandlerRequest{Config: instr("x")}) {
		if err != nil {
			t.Fatal(err)
		}
		evs = append(evs, ev)
	}
	if len(evs) != 2 || evs[0].Text != "whole" || *evs[1].Output != "whole" {
		t.Errorf("events = %+v", evs)
	}
}

// streaming ignores outputFormat.
func TestStreamIgnoresOutputFormat(t *testing.T) {
	h, f, _ := setup(t, streamTurn([]string{`{"a":`, `1}`}, 1, 1))
	c := instr("base")
	c.OutputFormat = map[string]any{"type": "object"}
	for _, err := range h.Stream(bg, aiserver.HandlerRequest{Config: c}) {
		if err != nil {
			t.Fatal(err)
		}
	}
	if f.calls[0].opts.JSONMode || text(f.calls[0].msgs[0]) != "base" {
		t.Errorf("streaming applied outputFormat: json=%v system=%q", f.calls[0].opts.JSONMode, text(f.calls[0].msgs[0]))
	}
}

// langchain-messages.
func TestStructuredOutput(t *testing.T) {
	schema := map[string]any{"type": "object", "properties": map[string]any{"name": map[string]any{"type": "string"}}}

	// Absent: no structured configuration, plain string.
	h, f, _ := setup(t, textTurn("plain", 1, 1))
	res, _ := h.Invoke(bg, aiserver.HandlerRequest{Config: instr("base")})
	if f.calls[0].opts.JSONMode || text(f.calls[0].msgs[0]) != "base" || res.Output != "plain" {
		t.Errorf("absent: json=%v out=%v", f.calls[0].opts.JSONMode, res.Output)
	}

	// Set, no tools: JSON mode + schema, parsed object returned, real usage.
	h, f, _ = setup(t, textTurn("```json\n{\"name\":\"Al\"}\n```", 11, 7))
	c := instr("base")
	c.OutputFormat = schema
	res, err := h.Invoke(bg, aiserver.HandlerRequest{Config: c})
	if err != nil {
		t.Fatal(err)
	}
	sysText := text(f.calls[0].msgs[0])
	if !f.calls[0].opts.JSONMode || !strings.HasPrefix(sysText, "base\n\n") || !strings.Contains(sysText, `"name"`) {
		t.Errorf("structured call: json=%v system=%q", f.calls[0].opts.JSONMode, sysText)
	}
	if !reflect.DeepEqual(res.Output, map[string]any{"name": "Al"}) {
		t.Errorf("output = %#v", res.Output)
	}
	if res.Usage["input_tokens"] != float64(11) || res.Usage["output_tokens"] != float64(7) {
		t.Errorf("structured usage = %v", res.Usage)
	}

	// Set with tools, non-OpenAI: tool loop without enforcement, string output.
	h, f, _ = setup(t, toolTurn("a", nil), textTurn(`{"name":"x"}`, 1, 1))
	tc := cfg(threeTools)
	tc.OutputFormat = schema
	tools := map[string]aiserver.ToolHandler{"a": aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) { return "r", nil })}
	res, err = h.Invoke(bg, aiserver.HandlerRequest{Config: tc, ToolHandlers: tools})
	if err != nil || f.calls[0].opts.JSONMode || res.Output != `{"name":"x"}` || len(f.calls[0].opts.Tools) != 1 {
		t.Errorf("tools+schema (anthropic): err=%v json=%v out=%#v", err, f.calls[0].opts.JSONMode, res.Output)
	}

	// Set with tools, OpenAI: response format bound on every turn, string output.
	h, f, _ = setup(t, toolTurn("a", nil), textTurn(`{"name":"x"}`, 1, 1))
	tc.Provider.Name = "OpenAI"
	res, err = h.Invoke(bg, aiserver.HandlerRequest{Config: tc, ToolHandlers: tools})
	if err != nil || !f.calls[0].opts.JSONMode || !f.calls[1].opts.JSONMode || res.Output != `{"name":"x"}` {
		t.Errorf("tools+schema (openai): err=%v out=%#v", err, res.Output)
	}
}

// LangChain rule: gen_ai.system is the lowercased provider.
func TestGenAISystemFromProvider(t *testing.T) {
	for provider, want := range map[string]string{"Anthropic": "anthropic", "OpenAI": "openai", "": "openai"} {
		h, _, exp := setup(t, textTurn("ok", 1, 1))
		c := instr("x")
		c.Provider.Name = provider
		_, _ = h.Invoke(bg, aiserver.HandlerRequest{Config: c})
		for _, s := range exp.GetSpans().Snapshots() {
			if got := spanAttrs(s)["gen_ai.system"]; got != want {
				t.Errorf("provider %q: %s gen_ai.system = %q", provider, s.Name(), got)
			}
		}
	}
}

func TestModelParametersAndFactory(t *testing.T) {
	f := &fakeModel{turns: []scripted{textTurn("ok", 1, 1)}}
	var seen aiserver.AiConfig
	h := newHandler(Options{ModelFactory: func(c aiserver.AiConfig) (llms.Model, error) { seen = c; return f, nil }})
	withTracer(t)
	c := cfg(func(c *aiserver.AiConfig) {
		c.Instructions = "x"
		c.Provider.Name = "Bedrock"
		c.Model = aiserver.Model{Name: "anthropic.claude", Region: "us",
			Parameters: map[string]any{"temperature": 0.2, "max_tokens": float64(99), "stop": []any{"END"}},
			Custom:     map[string]any{"temperature": 0.9, "seed": 7}}
	})
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: c}); err != nil {
		t.Fatal(err)
	}
	o := f.calls[0].opts
	if o.Temperature != 0.2 || o.MaxTokens != 99 || !reflect.DeepEqual(o.StopWords, []string{"END"}) || o.Seed != 0 {
		t.Errorf("call options = %+v", o)
	}
	if seen.Model.Name != "us.anthropic.claude" {
		t.Errorf("factory saw model %q", seen.Model.Name)
	}
	if got := langchainwire.ModelConfig(cfg(func(c *aiserver.AiConfig) {
		c.Provider.Name = "Bedrock"
		c.Model = aiserver.Model{Name: "us.anthropic.claude", Region: "us"}
	})).Model.Name; got != "us.anthropic.claude" {
		t.Errorf("prefix not idempotent: %q", got)
	}

	bad := errors.New("no creds")
	h = newHandler(Options{ModelFactory: func(aiserver.AiConfig) (llms.Model, error) { return nil, bad }})
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: instr("x")}); !errors.Is(err, bad) {
		t.Errorf("factory error = %v", err)
	}
}

// Built-in routing: provider.name picks the langchaingo model.
func TestDefaultModelRouting(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "test")
	m, err := langchainwire.DefaultModel(bg, cfg(nil), nil)
	if _, ok := m.(*anthropic.LLM); err != nil || !ok {
		t.Errorf("anthropic route: %T %v", m, err)
	}

	var body string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		body = string(b)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"c1","object":"chat.completion","created":1,"model":"gpt-4o","choices":[{"index":0,
"message":{"role":"assistant","content":"{\"name\":\"Al\"}"},"finish_reason":"stop"}],
"usage":{"prompt_tokens":7,"completion_tokens":3,"total_tokens":10,"prompt_tokens_details":{"cached_tokens":2}}}`)
	}))
	t.Cleanup(srv.Close)
	t.Setenv("OPENAI_BASE_URL", srv.URL)
	t.Setenv("OPENAI_API_KEY", "test")
	exp := withTracer(t)
	c := instr("base")
	c.Provider.Name = "OpenAI"
	c.Model.Name = "gpt-4o"
	c.OutputFormat = map[string]any{"type": "object", "properties": map[string]any{"name": map[string]any{"type": "string"}}}
	res, err := newHandler(Options{}).Invoke(bg, aiserver.HandlerRequest{Config: c, UserInput: "q"})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(body, `"response_format":{"type":"json_schema"`) || !strings.Contains(body, `"model":"gpt-4o"`) {
		t.Errorf("openai request body = %s", body)
	}
	if !reflect.DeepEqual(res.Output, map[string]any{"name": "Al"}) || res.Usage["input_tokens"] != float64(7) || res.Usage["output_tokens"] != float64(3) {
		t.Errorf("openai result = %#v usage %v", res.Output, res.Usage)
	}
	chat := spansByName(exp)["chat gpt-4o"][0]
	if a := spanAttrs(chat); a["gen_ai.usage.cache_read.input_tokens"] != "2" || a["gen_ai.response.finish_reasons"] != `["stop"]` {
		t.Errorf("openai chat span = %v", a)
	}
}

// the wrapper passes userInput, context and variables
// to Config(...).Invoke; ldContext reaches only the root span.
func TestLangChainMessagesWrapper(t *testing.T) {
	exp := withTracer(t)
	ld := &fakeLD{variation: map[string]any{
		"model": map[string]any{"name": "test-model"}, "provider": map[string]any{"name": "Anthropic"},
		"instructions": "Hi {{name}}", "_ldMeta": map[string]any{"enabled": true, "variationKey": "v"},
	}}
	if _, err := aiserver.InitWithClient(bg, ld, aiserver.InitOptions{OTLPEndpoint: "http://127.0.0.1:1/v1/traces"}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = aiserver.Shutdown(bg) })
	otel.SetTracerProvider(sdktrace.NewTracerProvider(sdktrace.WithSyncer(exp))) // InitWithClient installed its own

	f := &fakeModel{turns: []scripted{textTurn("one", 1, 1), textTurn("two", 1, 1)}}
	multi := ldcontext.NewMulti(ldcontext.New("u-1"), ldcontext.NewWithKind("org", "o-1"))
	res, err := LangChainMessages(bg, "cfg", "hello", multi, &InvokeOptions{Variables: map[string]any{"name": "Al"}, Handler: Options{Model: f}})
	if err != nil {
		t.Fatal(err)
	}
	if res.Response != "one" || text(f.calls[0].msgs[0]) != "Hi Al" || text(f.calls[0].msgs[1]) != "hello" {
		t.Errorf("wrapper call: %v / %v", res.Response, f.calls[0].msgs)
	}
	if got := ld.requestedKeys(); !reflect.DeepEqual(got, []string{"cfg"}) {
		t.Errorf("wrapper evaluated flag keys %v, want [cfg]", got)
	}
	roots := 0
	for _, s := range exp.GetSpans().Snapshots() {
		a, ev := spanAttrs(s), map[string]string{}
		for _, e := range s.Events() {
			if e.Name == "feature_flag" {
				for _, kv := range e.Attributes {
					ev[string(kv.Key)] = kv.Value.Emit()
				}
			}
		}
		if s.Name() != SpanName {
			for k := range a {
				if strings.HasPrefix(k, "context.") {
					t.Errorf("child %s carries %s", s.Name(), k)
				}
			}
			if len(ev) > 0 {
				t.Errorf("child %s has a feature_flag event %v", s.Name(), ev)
			}
			continue
		}
		roots++
		if ev["feature_flag.context.id"] != multi.FullyQualifiedKey() || ev["feature_flag.contextKeys"] != `{"org":"o-1","user":"u-1"}` {
			t.Errorf("root feature_flag event = %v, want context.id %q", ev, multi.FullyQualifiedKey())
		}
		if a["context.contextKeys.user"] != "u-1" || a["context.contextKeys.org"] != "o-1" {
			t.Errorf("root context keys = user %q org %q", a["context.contextKeys.user"], a["context.contextKeys.org"])
		}
	}
	if roots != 1 {
		t.Errorf("root spans = %d", roots)
	}

	// Without variables (nil options is allowed elsewhere): placeholders stay.
	if _, err := LangChainMessages(bg, "cfg", "", multi, &InvokeOptions{Handler: Options{Model: f}}); err != nil {
		t.Fatal(err)
	}
	if got := text(f.calls[1].msgs[0]); got != "Hi {{name}}" {
		t.Errorf("variables leaked or invented: %q", got)
	}
}

// the wrapper works with nil options, building the default langchaingo
// model for the variation's provider (OpenAI here, against OPENAI_BASE_URL).
func TestLangChainMessagesNilOptions(t *testing.T) {
	var body string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		body = string(b)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"c1","object":"chat.completion","created":1,"model":"gpt-4o","choices":[{"index":0,
"message":{"role":"assistant","content":"hi there"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`)
	}))
	t.Cleanup(srv.Close)
	t.Setenv("OPENAI_BASE_URL", srv.URL)
	t.Setenv("OPENAI_API_KEY", "test")
	exp := withTracer(t)
	ld := &fakeLD{variation: map[string]any{
		"model": map[string]any{"name": "gpt-4o"}, "provider": map[string]any{"name": "OpenAI"},
		"instructions": "be brief", "_ldMeta": map[string]any{"enabled": true, "variationKey": "v"},
	}}
	if _, err := aiserver.InitWithClient(bg, ld, aiserver.InitOptions{OTLPEndpoint: "http://127.0.0.1:1/v1/traces"}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = aiserver.Shutdown(bg) })
	otel.SetTracerProvider(sdktrace.NewTracerProvider(sdktrace.WithSyncer(exp))) // InitWithClient installed its own

	res, err := LangChainMessages(bg, "cfg", "q", ldcontext.New("u-1"), nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.Response != "hi there" || !strings.Contains(body, `"model":"gpt-4o"`) || !strings.Contains(body, "be brief") || !strings.Contains(body, `"q"`) {
		t.Errorf("response %q, request %s", res.Response, body)
	}
	if got := ld.requestedKeys(); !reflect.DeepEqual(got, []string{"cfg"}) {
		t.Errorf("evaluated flag keys %v, want [cfg]", got)
	}
	if a := spanAttrs(spansByName(exp)[SpanName][0]); a["gen_ai.system"] != "openai" {
		t.Errorf("gen_ai.system = %q", a["gen_ai.system"])
	}
}

// helper usage strings.
func TestHelperUsage(t *testing.T) {
	withTracer(t)
	ld := &fakeLD{variation: map[string]any{
		"model": map[string]any{"name": "m"}, "provider": map[string]any{"name": "Anthropic"},
		"instructions": "x", "_ldMeta": map[string]any{"enabled": true},
	}}
	start := func() {
		if _, err := aiserver.InitWithClient(bg, ld, aiserver.InitOptions{OTLPEndpoint: "http://127.0.0.1:1/v1/traces"}); err != nil {
			t.Fatal(err)
		}
	}
	start()
	t.Cleanup(func() { _ = aiserver.Shutdown(bg) })
	helpers := func() []string {
		var out []string
		for _, d := range ld.usage() {
			out = append(out, d.GetByKey("helper").StringValue())
			if pkg, ver := d.GetByKey("helperPackageName").StringValue(), d.GetByKey("helperPackageVersion").StringValue(); pkg != PackageName || ver != PackageVersion {
				t.Errorf("helper package = %s@%s", pkg, ver)
			}
			if d.GetByKey("aiSdkName").StringValue() != aiserver.PackageName || d.GetByKey("aiSdkVersion").StringValue() != aiserver.PackageVersion ||
				d.GetByKey("aiSdkLanguage").StringValue() != "go" {
				t.Errorf("core identity = %v", d)
			}
		}
		return out
	}

	NewHandler(Options{})
	NewHandler(Options{})
	if got := helpers(); !reflect.DeepEqual(got, []string{"langchain-messages.createLangChainHandler"}) {
		t.Errorf("factory usage = %v", got)
	}
	f := &fakeModel{turns: []scripted{textTurn("a", 1, 1), textTurn("b", 1, 1)}}
	_, _ = LangChainMessages(bg, "k", "q", ldcontext.New("u"), &InvokeOptions{Handler: Options{Model: f}})
	_, _ = LangChainMessages(bg, "k", "q", ldcontext.New("u"), &InvokeOptions{Handler: Options{Model: f}})
	if got := helpers(); !reflect.DeepEqual(got, []string{"langchain-messages.createLangChainHandler", "langchain-messages.langchainMessages"}) {
		t.Errorf("wrapper usage = %v", got)
	}

	// Shutdown re-arms the report.
	_ = aiserver.Shutdown(bg)
	ld.events = nil
	start()
	NewHandler(Options{})
	if got := helpers(); !reflect.DeepEqual(got, []string{"langchain-messages.createLangChainHandler"}) {
		t.Errorf("after shutdown = %v", got)
	}
}
