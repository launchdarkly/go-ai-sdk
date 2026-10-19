package langchainagents

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/stategraph"
	"github.com/launchdarkly/go-ai-sdk/llms"
)

const (
	sys   = llms.ChatMessageTypeSystem
	human = llms.ChatMessageTypeHuman
	ai    = llms.ChatMessageTypeAI
	tool  = llms.ChatMessageTypeTool
)

func cfg(extra func(*aiserver.AiConfig)) aiserver.AiConfig {
	c := aiserver.AiConfig{Model: aiserver.Model{Name: "test-model"}}
	c.Provider.Name = "Anthropic"
	if extra != nil {
		extra(&c)
	}
	return c
}

func instr(s string) aiserver.AiConfig { return cfg(func(c *aiserver.AiConfig) { c.Instructions = s }) }

func text(m llms.MessageContent) string {
	var b strings.Builder
	for _, p := range m.Parts {
		if t, ok := p.(llms.TextContent); ok {
			b.WriteString(t.Text)
		}
	}
	return b.String()
}

type turn struct {
	role llms.ChatMessageType
	text string
}

func turns(msgs []llms.MessageContent) []turn {
	out := make([]turn, 0, len(msgs))
	for _, m := range msgs {
		out = append(out, turn{m.Role, text(m)})
	}
	return out
}

func toolNames(o llms.CallOptions) []string {
	var out []string
	for _, t := range o.Tools {
		out = append(out, t.Function.Name)
	}
	return out
}

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

func fn(f func(args map[string]any) (any, error)) aiserver.ToolFunc {
	return func(_ context.Context, args map[string]any) (any, error) { return f(args) }
}

func collect(t *testing.T, h *aiserver.ProviderHandler, req aiserver.HandlerRequest) ([]aiserver.HandlerStreamEvent, error) {
	t.Helper()
	var evs []aiserver.HandlerStreamEvent
	for ev, err := range h.Stream(bg, req) {
		if err != nil {
			return evs, err
		}
		evs = append(evs, ev)
	}
	return evs, nil
}

// factory and metadata.
func TestFactory(t *testing.T) {
	a, b := NewHandler(Options{}), NewHandler(Options{CaptureContent: true})
	if a.ProvidesFor != (aiserver.ProvidesFor{Provider: "*", Mode: aiserver.ModeAgent}) {
		t.Errorf("providesFor = %+v", a.ProvidesFor)
	}
	if a == b || a.Invoke == nil || a.Stream == nil || a.CaptureContent || !b.CaptureContent {
		t.Errorf("handlers: %p %p capture %v %v", a, b, a.CaptureContent, b.CaptureContent)
	}
}

// both paths build the agent through the ReAct factory with
// the model, tools and system prompt.
func TestAgentCreationAPI(t *testing.T) {
	type seen struct {
		model  llms.Model
		tools  []string
		system string
	}
	var got []seen
	orig := createAgent
	createAgent = func(a *reactAgent) (*stategraph.Runnable, error) {
		var names []string
		for _, t := range a.tools {
			names = append(names, t.Function.Name)
		}
		got = append(got, seen{a.model, names, a.system})
		return orig(a)
	}
	t.Cleanup(func() { createAgent = orig })

	c := instr("Be terse")
	c.Tools = map[string]aiserver.Tool{"lookup": {Name: "lookup", Parameters: map[string]any{"type": "object"}}}
	tools := map[string]aiserver.ToolHandler{"lookup": fn(func(map[string]any) (any, error) { return "x", nil })}
	h, f, _ := setup(t, textTurn("ok", 1, 1), streamTurn([]string{"ok"}, 1, 1))
	req := aiserver.HandlerRequest{Config: c, UserInput: "hi", ToolHandlers: tools}
	if _, err := h.Invoke(bg, req); err != nil {
		t.Fatal(err)
	}
	if _, err := collect(t, h, req); err != nil {
		t.Fatal(err)
	}
	want := seen{f, []string{"lookup"}, "Be terse"}
	if len(got) != 2 || !reflect.DeepEqual(got[0], want) || !reflect.DeepEqual(got[1], want) {
		t.Errorf("createAgent calls = %+v", got)
	}
}

// prompt construction, agent mode.
func TestPromptConstruction(t *testing.T) {
	msgs := func(ms ...aiserver.Message) func(*aiserver.AiConfig) {
		return func(c *aiserver.AiConfig) { c.Messages = ms }
	}
	m := func(role, s string) aiserver.Message { return aiserver.Message{Role: role, Text: s} }
	cases := []struct {
		name  string
		cfg   aiserver.AiConfig
		input string
		want  []turn
	}{
		{"A: instructions, placeholders", instr("Hi {{name}}, {{missing}}"), "q",
			[]turn{{sys, "Hi Al, {{missing}}"}, {human, "q"}}},
		{"B: system messages combined, last user not duplicated",
			cfg(msgs(m("system", "S1 {{name}}"), m("system", "S2"), m("assistant", "a"), m("user", "u {{name}}"))), "q",
			[]turn{{sys, "S1 Al\nS2"}, {ai, "a"}, {human, "u Al"}}},
		{"B: last assistant gets userInput", cfg(msgs(m("user", "u"), m("assistant", "a"))), "q",
			[]turn{{human, "u"}, {ai, "a"}, {human, "q"}}},
		{"C: empty input still appended as \"\"", instr("I"), "", []turn{{sys, "I"}, {human, ""}}},
		{"C: instructions win over messages", cfg(func(c *aiserver.AiConfig) {
			c.Instructions = "I"
			c.Messages = []aiserver.Message{m("system", "S"), m("assistant", "a")}
		}), "q", []turn{{sys, "I"}, {ai, "a"}, {human, "q"}}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			h, f, _ := setup(t, textTurn("ok", 1, 1))
			if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: c.cfg, UserInput: c.input, Variables: map[string]any{"name": "Al"}}); err != nil {
				t.Fatal(err)
			}
			if got := turns(f.call(0).msgs); !reflect.DeepEqual(got, c.want) {
				t.Errorf("messages = %v, want %v", got, c.want)
			}
		})
	}
}

func threeTools() map[string]aiserver.Tool {
	ts := map[string]aiserver.Tool{}
	for _, n := range []string{"a", "b", "c"} {
		ts[n] = aiserver.Tool{Name: n, Description: n + " desc", Parameters: map[string]any{"type": "object", "properties": map[string]any{n: map[string]any{"type": "string"}}}}
	}
	return ts
}

// tool conversion and filtering.
func TestToolConversion(t *testing.T) {
	c := instr("I")
	c.Tools = threeTools()
	h, f, _ := setup(t, textTurn("ok", 1, 1))
	ok := fn(func(map[string]any) (any, error) { return "", nil })
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: c, ToolHandlers: map[string]aiserver.ToolHandler{"a": ok, "c": ok, "x": ok}}); err != nil {
		t.Fatal(err)
	}
	got := f.call(0).opts.Tools
	if len(got) != 2 || got[0].Function.Name != "a" || got[1].Function.Name != "c" || got[0].Function.Description != "a desc" ||
		!reflect.DeepEqual(got[0].Function.Parameters, c.Tools["a"].Parameters) {
		t.Errorf("tools = %+v", got)
	}

	h, f, _ = setup(t, textTurn("ok", 1, 1))
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: c}); err != nil {
		t.Fatal(err)
	}
	if n := len(f.call(0).opts.Tools); n != 0 {
		t.Errorf("no handlers: %d tools sent", n)
	}
}

// tool loop.
func TestToolLoop(t *testing.T) {
	c := instr("I")
	c.Tools = threeTools()
	var order []string
	var args []map[string]any
	rec := func(name string) aiserver.ToolFunc {
		return fn(func(a map[string]any) (any, error) {
			order = append(order, name)
			args = append(args, a)
			return map[string]any{"from": name}, nil
		})
	}
	h, f, _ := setup(t, toolTurn("a", map[string]any{"a": "1"}), toolTurn("b", nil), textTurn("final", 2, 3))
	res, err := h.Invoke(bg, aiserver.HandlerRequest{Config: c, UserInput: "q", ToolHandlers: map[string]aiserver.ToolHandler{"a": rec("a"), "b": rec("b")}})
	if err != nil {
		t.Fatal(err)
	}
	if res.Output != "final" || !reflect.DeepEqual(order, []string{"a", "b"}) || args[0]["a"] != "1" {
		t.Errorf("output %v order %v args %v", res.Output, order, args)
	}
	if !reflect.DeepEqual(res.Usage, map[string]any{"input_tokens": 4.0, "output_tokens": 5.0}) {
		t.Errorf("usage = %v", res.Usage)
	}
	// The third call sees both tool results, appended by the AddMessages reducer.
	var roles []llms.ChatMessageType
	for _, m := range f.call(2).msgs {
		roles = append(roles, m.Role)
	}
	if !reflect.DeepEqual(roles, []llms.ChatMessageType{sys, human, ai, tool, ai, tool}) {
		t.Errorf("third call roles = %v", roles)
	}
	if tr := f.call(1).msgs[3].Parts[0].(llms.ToolCallResponse); tr.Content != `{"from":"a"}` || tr.ToolCallID != "call_a" {
		t.Errorf("tool response = %+v", tr)
	}

	boom := errors.New("boom")
	h, _, _ = setup(t, toolTurn("a", nil))
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: c, ToolHandlers: map[string]aiserver.ToolHandler{
		"a": fn(func(map[string]any) (any, error) { return nil, boom })}}); !errors.Is(err, boom) {
		t.Errorf("tool error = %v", err)
	}
	h, _, _ = setup(t, toolTurn("b", nil))
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: c, ToolHandlers: map[string]aiserver.ToolHandler{"a": rec("a")}}); !errors.Is(err, ErrNoToolHandler) {
		t.Errorf("unoffered tool = %v", err)
	}

	called := false
	h, f, _ = setup(t, textTurn("x", 1, 1))
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: instr("I"), ToolHandlers: map[string]aiserver.ToolHandler{
		"a": fn(func(map[string]any) (any, error) { called = true; return nil, nil })}}); err != nil || called || f.count() != 1 {
		t.Errorf("no tools: err %v called %v calls %d", err, called, f.count())
	}
}

// LangGraph's recursion limit: a model that never stops calling tools.
func TestRecursionLimit(t *testing.T) {
	c := instr("I")
	c.Tools = threeTools()
	h, f, exp := setup(t, toolTurn("a", nil))
	f.repeat = true
	_, err := h.Invoke(bg, aiserver.HandlerRequest{Config: c, ToolHandlers: map[string]aiserver.ToolHandler{"a": fn(func(map[string]any) (any, error) { return "", nil })}})
	if !errors.Is(err, ErrRecursionLimit) || f.count() != 13 {
		t.Errorf("err = %v after %d calls", err, f.count())
	}
	if root := spansByName(exp)[SpanName]; len(root) != 1 || root[0].Status().Code != codes.Error {
		t.Errorf("root span = %v", root)
	}
}

var ldFixture = aiserver.TrackData{ConfigKey: "test-config", VariationKey: "variation-a", RunID: "run-123", Version: 1, ModelName: "test-model", ProviderName: "TestProvider"}

// telemetry on the blocking path, with chat and execute_tool children.
func TestTelemetry(t *testing.T) {
	c := instr("Sys")
	c.Tools = threeTools()
	h, _, exp := setup(t, toolTurn("a", nil), textTurn("answer", 3, 4))
	_, err := h.Invoke(bg, aiserver.HandlerRequest{Config: c, UserInput: "hello",
		ToolHandlers: map[string]aiserver.ToolHandler{"a": fn(func(map[string]any) (any, error) { return "r", nil })},
		Variables:    map[string]any{aiserver.VarLD: ldFixture}})
	if err != nil {
		t.Fatal(err)
	}
	spans := spansByName(exp)
	root := spans[SpanName]
	if len(root) != 1 {
		t.Fatalf("spans = %v", spans)
	}
	a := spanAttrs(root[0])
	want := map[string]string{
		"gen_ai.system": "anthropic", "gen_ai.operation.name": "invoke_agent", "gen_ai.request.model": "test-model",
		"gen_ai.response.model": "test-model", "gen_ai.usage.input_tokens": "4", "gen_ai.usage.output_tokens": "5",
		"gen_ai.usage.total_tokens": "9", "gen_ai.usage.prompt_tokens": "4", "gen_ai.usage.completion_tokens": "5",
		"gen_ai.prompt.0.role": "system", "gen_ai.prompt.0.content": "Sys", "gen_ai.prompt.1.role": "user",
		"gen_ai.prompt.1.content": "hello", "gen_ai.completion.0.role": "assistant", "gen_ai.completion.0.content": "answer",
		"launchdarkly.operation.type": "gen_ai", "launchdarkly.config.key": "test-config",
		"launchdarkly.variation.key": "variation-a", "launchdarkly.run.id": "run-123",
	}
	for k, v := range want {
		if a[k] != v {
			t.Errorf("%s = %q, want %q", k, a[k], v)
		}
	}
	if _, ok := a["launchdarkly.graph.key"]; ok {
		t.Error("graph key set without graphKey")
	}
	if !hasEvent(root[0], "gen_ai.content.prompt", "hello") || !hasEvent(root[0], "gen_ai.content.completion", "answer") {
		t.Error("content events missing")
	}
	if root[0].Status().Code != codes.Ok || root[0].EndTime().IsZero() {
		t.Errorf("status %v", root[0].Status())
	}
	if len(spans["chat test-model"]) != 2 || len(spans["execute_tool a"]) != 1 {
		t.Errorf("children = %v", spans)
	}
	for _, s := range append(spans["chat test-model"], spans["execute_tool a"]...) {
		if s.Parent().SpanID() != root[0].SpanContext().SpanID() {
			t.Errorf("%s is not a child of the root", s.Name())
		}
	}
}

// provider error.
func TestProviderError(t *testing.T) {
	boom := errors.New("provider down")
	h, _, exp := setup(t, scripted{err: boom})
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: instr("I")}); !errors.Is(err, boom) {
		t.Fatalf("err = %v", err)
	}
	root := spansByName(exp)[SpanName]
	if len(root) != 1 || root[0].Status().Code != codes.Error || len(root[0].Events()) == 0 || root[0].EndTime().IsZero() {
		t.Fatalf("root = %v", root)
	}
	recorded := false
	for _, e := range root[0].Events() {
		recorded = recorded || e.Name == "exception"
	}
	if !recorded {
		t.Error("exception not recorded")
	}
	if chat := spansByName(exp)["chat test-model"]; len(chat) != 1 || chat[0].Status().Code != codes.Error {
		t.Errorf("chat span = %v", chat)
	}
}

// streaming: deltas, done, tool loop, telemetry.
func TestStreaming(t *testing.T) {
	c := instr("I")
	c.Tools = threeTools()
	var got map[string]any
	h, _, exp := setup(t,
		streamTurn([]string{"Let me ", "check. "}, 1, 1, toolCall("a", map[string]any{"a": "v"})),
		streamTurn([]string{"Done", "!"}, 2, 3))
	evs, err := collect(t, h, aiserver.HandlerRequest{Config: c, UserInput: "q", Variables: map[string]any{aiserver.VarLD: ldFixture},
		ToolHandlers: map[string]aiserver.ToolHandler{"a": fn(func(a map[string]any) (any, error) { got = a; return "r", nil })}})
	if err != nil {
		t.Fatal(err)
	}
	var chunks []string
	for _, e := range evs[:len(evs)-1] {
		if e.Type != aiserver.EventChunk {
			t.Fatalf("non-chunk before done: %+v", e)
		}
		chunks = append(chunks, e.Text)
	}
	done := evs[len(evs)-1]
	if !reflect.DeepEqual(chunks, []string{"Let me ", "check. ", "Done", "!"}) || done.Type != aiserver.EventDone ||
		*done.Output != "Let me check. Done!" || done.Usage["input_tokens"] != 3.0 || done.Usage["output_tokens"] != 4.0 || got["a"] != "v" {
		t.Errorf("chunks %q done %+v args %v", chunks, done, got)
	}
	root := spansByName(exp)[SpanName]
	if len(root) != 1 {
		t.Fatal("no streaming root span")
	}
	a := spanAttrs(root[0])
	if a["gen_ai.system"] != "anthropic" || a["gen_ai.usage.total_tokens"] != "7" || a["launchdarkly.run.id"] != "run-123" ||
		a["gen_ai.completion.0.content"] != "Let me check. Done!" || a["gen_ai.prompt.0.content"] != "I" || root[0].Status().Code != codes.Ok ||
		!hasEvent(root[0], "gen_ai.content.prompt", "q") || !hasEvent(root[0], "gen_ai.content.completion", "Done!") {
		t.Errorf("stream root = %v", a)
	}

	boom := errors.New("mid-stream")
	h, _, exp = setup(t, scripted{err: boom})
	if _, err := collect(t, h, aiserver.HandlerRequest{Config: instr("I")}); !errors.Is(err, boom) {
		t.Errorf("stream err = %v", err)
	}
	if r := spansByName(exp)[SpanName]; len(r) != 1 || r[0].Status().Code != codes.Error || r[0].EndTime().IsZero() {
		t.Errorf("error root = %v", r)
	}
}

// A consumer that stops early ends every span as abandoned.
func TestStreamAbandoned(t *testing.T) {
	h, _, exp := setup(t, streamTurn([]string{"a", "b", "c"}, 1, 1))
	n := 0
	for range h.Stream(bg, aiserver.HandlerRequest{Config: instr("I")}) {
		n++
		break
	}
	spans := exp.GetSpans().Snapshots()
	if n != 1 || len(spans) != 2 {
		t.Fatalf("events %d spans %d", n, len(spans))
	}
	for _, s := range spans {
		if spanAttrs(s)["launchdarkly.stream.abandoned"] != "true" || s.Status().Code != codes.Unset {
			t.Errorf("%s: %v %v", s.Name(), spanAttrs(s), s.Status())
		}
	}
}

// langchain-agents: best-effort schema instruction.
func TestOutputFormat(t *testing.T) {
	schema := map[string]any{"type": "object", "properties": map[string]any{"name": map[string]any{"type": "string"}}}
	c := instr("Extract")
	c.OutputFormat = schema
	h, f, _ := setup(t, textTurn(`{"name":"x"}`, 1, 1), streamTurn([]string{"x"}, 1, 1), textTurn("x", 1, 1))
	res, err := h.Invoke(bg, aiserver.HandlerRequest{Config: c})
	if err != nil {
		t.Fatal(err)
	}
	if s := text(f.call(0).msgs[0]); !strings.HasPrefix(s, "Extract\n\n") || !strings.Contains(s, `{"properties":{"name":{"type":"string"}},"type":"object"}`) || res.Output != `{"name":"x"}` {
		t.Errorf("system = %q output %v", s, res.Output)
	}
	if _, err := collect(t, h, aiserver.HandlerRequest{Config: c}); err != nil {
		t.Fatal(err)
	}
	if s := text(f.call(1).msgs[0]); s != "Extract" {
		t.Errorf("streaming system = %q", s)
	}
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: instr("Extract")}); err != nil {
		t.Fatal(err)
	}
	if s := text(f.call(2).msgs[0]); s != "Extract" {
		t.Errorf("no outputFormat: system = %q", s)
	}
}

func image() aiserver.Message {
	return aiserver.Message{Role: "user", Blocks: []aiserver.ContentBlock{{Type: "image", Source: &aiserver.ImageSource{Type: "base64", MediaType: "image/png", Data: "aGk="}}}}
}

// agent mode: history as structured messages.
func TestHistory(t *testing.T) {
	hist := []aiserver.Message{{Role: "system", Text: "ignore me"}, {Role: "user", Text: "earlier"}, {Role: "assistant", Text: "reply"}}
	run := func(t *testing.T, input string, history []aiserver.Message) []llms.MessageContent {
		t.Helper()
		h, f, _ := setup(t, textTurn("ok", 1, 1))
		if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: instr("Sys"), UserInput: input, History: history}); err != nil {
			t.Fatal(err)
		}
		return f.call(0).msgs
	}
	got := run(t, "now", hist)
	if want := []turn{{sys, "Sys"}, {human, "earlier"}, {ai, "reply"}, {human, "now"}}; !reflect.DeepEqual(turns(got), want) {
		t.Errorf("history = %v", turns(got))
	}
	if strings.Contains(text(got[0]), "Conversation History") || strings.Contains(text(got[0]), "earlier") {
		t.Error("history leaked into the system prompt")
	}
	if !reflect.DeepEqual(turns(run(t, "q", []aiserver.Message{})), turns(run(t, "q", nil))) {
		t.Error("empty history differs from none")
	}
	got = run(t, "", []aiserver.Message{image()})
	if len(got) != 2 || got[1].Role != human {
		t.Fatalf("empty input after user history appended a turn: %v", turns(got))
	}
	if b, ok := got[1].Parts[0].(llms.BinaryContent); !ok || b.MIMEType != "image/png" || string(b.Data) != "hi" {
		t.Errorf("image part = %#v", got[1].Parts[0])
	}
	got = run(t, "what is this?", []aiserver.Message{image()})
	if len(got) != 3 || got[2].Role != human || text(got[2]) != "what is this?" {
		t.Errorf("image-only history + input = %v", turns(got))
	}
}

// convenience export, plus context identity on the root only.
func TestLangChainAgents(t *testing.T) {
	ld := &fakeLD{fallback: map[string]any{
		"instructions": "Hi {{name}}", "model": map[string]any{"name": "m"}, "provider": map[string]any{"name": "OpenAI"},
		"tools":   map[string]any{"a": map[string]any{"name": "a", "type": "function", "parameters": map[string]any{"type": "object"}}},
		"_ldMeta": map[string]any{"enabled": true, "variationKey": "v", "mode": "agent"},
	}}
	exp := installClient(t, ld)
	f := &fakeModel{turns: []scripted{toolTurn("a", nil), textTurn("one", 1, 1), textTurn("two", 1, 1)}}
	multi := ldcontext.NewMulti(ldcontext.New("u-1"), ldcontext.NewWithKind("org", "o-1"))
	res, err := LangChainAgents(bg, "cfg", "hello", multi, &InvokeOptions{Variables: map[string]any{"name": "Al"}, Handler: Options{Model: f},
		ToolHandlers: map[string]aiserver.ToolHandler{"a": fn(func(map[string]any) (any, error) { return "r", nil })}})
	if err != nil {
		t.Fatal(err)
	}
	if res.Response != "one" || ld.requestedKeys()[0] != "cfg" || text(f.call(0).msgs[0]) != "Hi Al" || text(f.call(0).msgs[1]) != "hello" {
		t.Errorf("res %+v keys %v msgs %v", res, ld.requestedKeys(), turns(f.call(0).msgs))
	}
	spans := spansByName(exp)
	if a := spanAttrs(spans[SpanName][0]); a["context.contextKeys.user"] != "u-1" || a["context.contextKeys.org"] != "o-1" || a["gen_ai.system"] != "openai" {
		t.Errorf("root identity = %v", a)
	}
	for _, s := range append(spans["chat m"], spans["execute_tool a"]...) {
		for k := range spanAttrs(s) {
			if strings.HasPrefix(k, "context.") || strings.HasPrefix(k, "launchdarkly.") {
				t.Errorf("%s carries %s", s.Name(), k)
			}
		}
	}

	if _, err := LangChainAgents(bg, "cfg", "", ldcontext.New("u"), &InvokeOptions{Handler: Options{Model: f}}); err != nil {
		t.Fatal(err) // nil variables leave {{name}} unresolved
	}
	if text(f.call(2).msgs[0]) != "Hi {{name}}" {
		t.Errorf("unresolved = %q", text(f.call(2).msgs[0]))
	}
	if _, err := LangChainAgents(bg, "cfg", "", ldcontext.New("u"), nil); err == nil {
		t.Error("nil options with no model credentials should fail at model construction, not panic")
	}
}

// model routing: Bedrock prefix and an explicit model factory.
func TestModelResolution(t *testing.T) {
	var seen aiserver.AiConfig
	f := &fakeModel{turns: []scripted{textTurn("ok", 1, 1)}}
	h := newHandler(Options{ModelFactory: func(c aiserver.AiConfig) (llms.Model, error) { seen = c; return f, nil }})
	withTracer(t)
	c := instr("I")
	c.Provider.Name = "Bedrock"
	c.Model = aiserver.Model{Name: "anthropic.claude", Region: "us"}
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: c}); err != nil || seen.Model.Name != "us.anthropic.claude" {
		t.Errorf("err %v model %q", err, seen.Model.Name)
	}
	bad := errors.New("no creds")
	h = newHandler(Options{ModelFactory: func(aiserver.AiConfig) (llms.Model, error) { return nil, bad }})
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: instr("x")}); !errors.Is(err, bad) {
		t.Errorf("factory error = %v", err)
	}
}

// helper usage strings for the handler and convenience export.
func TestSdkUsageHandler(t *testing.T) {
	ld := &fakeLD{fallback: map[string]any{"instructions": "I", "model": map[string]any{"name": "m"}, "provider": map[string]any{"name": "OpenAI"},
		"_ldMeta": map[string]any{"enabled": true, "mode": "agent"}}}
	installClient(t, ld)
	NewHandler(Options{})
	if got := ld.usageHelpers(); !reflect.DeepEqual(got, []string{"langchain-agents.createLangChainAgentsHandler"}) {
		t.Errorf("factory helpers = %v", got)
	}
	if pkg := ld.events("$ld:ai:sdk:usage")[0].data["helperPackageName"]; pkg != PackageName {
		t.Errorf("package = %v", pkg)
	}
	ld.resetTracks()
	_ = aiserver.Shutdown(bg) // re-arm the once-per-process report
	installClient(t, ld)
	if _, err := LangChainAgents(bg, "cfg", "q", ldcontext.New("u"), &InvokeOptions{Handler: Options{Model: &fakeModel{turns: []scripted{textTurn("x", 1, 1)}}}}); err != nil {
		t.Fatal(err)
	}
	if got := ld.usageHelpers(); !reflect.DeepEqual(got, []string{"langchain-agents.langchainAgents"}) {
		t.Errorf("wrapper helpers = %v", got)
	}
}

// Claude can return no content right after a tool result (seen live after a
// handoff). The turn is retried once with a "Please continue." user turn that
// is not kept in the conversation; an empty reply anywhere else still fails.
func TestEmptyResponseAfterToolResultRetries(t *testing.T) {
	c := instr("I")
	c.Tools = threeTools()
	tools := map[string]aiserver.ToolHandler{"a": fn(func(map[string]any) (any, error) { return "ok", nil })}
	h, f, _ := setup(t, toolTurn("a", nil), scripted{err: llms.ErrEmptyResponse}, textTurn("final", 1, 1))
	res, err := h.Invoke(bg, aiserver.HandlerRequest{Config: c, UserInput: "q", ToolHandlers: tools})
	if err != nil || res.Output != "final" {
		t.Fatalf("got %q %v", res.Output, err)
	}
	retry := f.call(2).msgs
	if last := retry[len(retry)-1]; last.Role != human || last.Parts[0] != (llms.TextContent{Text: "Please continue."}) {
		t.Errorf("retry ends with %v", last)
	}

	h, f, _ = setup(t, scripted{err: llms.ErrEmptyResponse}, textTurn("never", 1, 1))
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: c, UserInput: "q"}); err == nil || f.count() != 1 {
		t.Errorf("empty first turn: err %v, %d calls", err, f.count())
	}
}

// init() registers this package, so a new client gets its
// `$ld:ai:sdk:info` event under the package's own name and version.
func TestSdkInfoRegistered(t *testing.T) {
	ld := &fakeLD{}
	installClient(t, ld)
	for _, e := range ld.events("$ld:ai:sdk:info") {
		if e.data["aiSdkName"] == PackageName && e.data["aiSdkVersion"] == PackageVersion && e.data["aiSdkLanguage"] == "go" {
			return
		}
	}
	t.Errorf("no $ld:ai:sdk:info for %s: %v", PackageName, ld.events("$ld:ai:sdk:info"))
}

// A panicking tool ends the run's root and tool spans ERROR. stategraph
// recovers a node panic as the node's error, so the call returns an error.
func TestToolPanicEndsSpans(t *testing.T) {
	c := instr("I")
	c.Tools = threeTools()
	h, _, exp := setup(t, toolTurn("a", nil), textTurn("never", 1, 1))
	var err error
	func() {
		defer func() { _ = recover() }()
		_, err = h.Invoke(bg, aiserver.HandlerRequest{Config: c, UserInput: "q",
			ToolHandlers: map[string]aiserver.ToolHandler{"a": fn(func(map[string]any) (any, error) { panic("boom") })}})
	}()
	if err == nil {
		t.Log("the panic propagated instead of returning an error")
	}
	ended := map[string]codes.Code{}
	for _, s := range exp.GetSpans().Snapshots() {
		ended[s.Name()] = s.Status().Code
	}
	for _, name := range []string{SpanName, "execute_tool a"} {
		if code, ok := ended[name]; !ok || code != codes.Error {
			t.Errorf("%s: ended=%v status=%v", name, ok, code)
		}
	}
}
