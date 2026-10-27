package openaiagents

import (
	"context"
	"encoding/json"
	"errors"
	"maps"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/openai/openai-go/v3"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

func cfg(extra func(*aiserver.AiConfig)) aiserver.AiConfig {
	c := aiserver.AiConfig{Model: aiserver.Model{Name: "test-model"}}
	c.Provider.Name = "OpenAI"
	if extra != nil {
		extra(&c)
	}
	return c
}

func instructions(s string) func(*aiserver.AiConfig) {
	return func(c *aiserver.AiConfig) { c.Instructions = s }
}

// turn is one outbound input item, flattened: content is the string
// content, the JSON of a content-part array, or a function output.
type turn struct{ role, content string }

func inputTurns(req map[string]any) []turn {
	var out []turn
	items, _ := req["input"].([]any)
	for _, it := range items {
		m := it.(map[string]any)
		role, _ := m["role"].(string)
		if role == "" {
			role = m["type"].(string)
		}
		content, ok := m["content"].(string)
		if !ok {
			if o, isOut := m["output"].(string); isOut {
				content = o
			} else {
				b, _ := json.Marshal(m["content"])
				content = string(b)
			}
		}
		out = append(out, turn{role, content})
	}
	return out
}

func toolNames(req map[string]any) []string {
	var out []string
	tools, _ := req["tools"].([]any)
	for _, t := range tools {
		m := t.(map[string]any)
		name, _ := m["name"].(string)
		if name == "" {
			name = "hosted:" + m["type"].(string)
		}
		out = append(out, name)
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

func attrs(s sdktrace.ReadOnlySpan) map[string]string {
	out := map[string]string{}
	for _, a := range s.Attributes() {
		out[string(a.Key)] = a.Value.Emit()
	}
	return out
}

func eventNames(s sdktrace.ReadOnlySpan) []string {
	var out []string
	for _, e := range s.Events() {
		out = append(out, e.Name)
	}
	return out
}

// eventAttr returns attribute key of the first event named name on s.
func eventAttr(s sdktrace.ReadOnlySpan, name, key string) (string, bool) {
	for _, e := range s.Events() {
		if e.Name != name {
			continue
		}
		for _, a := range e.Attributes {
			if string(a.Key) == key {
				return a.Value.AsString(), true
			}
		}
	}
	return "", false
}

func tool(fn func(map[string]any) (any, error)) aiserver.ToolFunc {
	return func(_ context.Context, args map[string]any) (any, error) { return fn(args) }
}

func funcTool(desc string, params map[string]any) aiserver.Tool {
	return aiserver.Tool{Type: "function", Description: desc, Parameters: params}
}

func invoke(t *testing.T, h *aiserver.ProviderHandler, req aiserver.HandlerRequest) aiserver.HandlerResult {
	t.Helper()
	res, err := h.Invoke(bg, req)
	if err != nil {
		t.Fatal(err)
	}
	return res
}

func TestFactoryMetadata(t *testing.T) {
	a, b := newHandler(Options{}), newHandler(Options{CaptureContent: true})
	if a.ProvidesFor != (aiserver.ProvidesFor{Provider: "OpenAI", Mode: aiserver.ModeAgent}) || a.ProvidesFor != ProvidesFor {
		t.Errorf("providesFor = %+v", a.ProvidesFor)
	}
	if a.CaptureContent || !b.CaptureContent {
		t.Errorf("captureContent default/passthrough wrong: %v %v", a.CaptureContent, b.CaptureContent)
	}
	if a == b || a.Invoke == nil || a.Stream == nil {
		t.Errorf("factory must return independent callable instances")
	}
}

// Prompt construction: the agent's system prompt is the Responses
// `instructions` field; turns are input items.
func TestPromptConstruction(t *testing.T) {
	vars := map[string]any{"name": "Ada", "topic": "math"}
	msgs := func(ms ...aiserver.Message) func(*aiserver.AiConfig) {
		return func(c *aiserver.AiConfig) { c.Messages = ms }
	}
	for _, tc := range []struct {
		name         string
		config       aiserver.AiConfig
		userInput    string
		instructions any // nil means the field must be absent
		turns        []turn
	}{
		{"instructions with resolved and unresolved placeholders", cfg(instructions("Help {{name}} with {{missing}}")), "hi",
			"Help Ada with {{missing}}", []turn{{"user", "hi"}}},
		{"messages: system messages combined, turns in order, user last suppresses userInput",
			cfg(msgs(aiserver.Message{Role: "system", Text: "S1 {{topic}}"}, aiserver.Message{Role: "system", Text: "S2"},
				aiserver.Message{Role: "assistant", Text: "hello"}, aiserver.Message{Role: "user", Text: "about {{topic}}"})), "ignored",
			"S1 math\nS2", []turn{{"assistant", "hello"}, {"user", "about math"}}},
		{"messages: assistant last appends userInput", cfg(msgs(aiserver.Message{Role: "user", Text: "q"}, aiserver.Message{Role: "assistant", Text: "a"})), "next",
			nil, []turn{{"user", "q"}, {"assistant", "a"}, {"user", "next"}}},
		{"empty userInput on instructions path is an empty user turn", cfg(instructions("I")), "", "I", []turn{{"user", ""}}},
		{"precedence: instructions win over messages system entries",
			cfg(func(c *aiserver.AiConfig) {
				c.Instructions = "Use {{name}}"
				c.Messages = []aiserver.Message{{Role: "system", Text: "FROM MESSAGES"}, {Role: "user", Text: "u"}, {Role: "assistant", Text: "a"}}
			}), "x", "Use Ada", []turn{{"user", "u"}, {"assistant", "a"}, {"user", "x"}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h, f, _ := setup(t, textResp("ok", 1, 1))
			invoke(t, h, aiserver.HandlerRequest{Config: tc.config, UserInput: tc.userInput, Variables: vars})
			req := f.request(0)
			if req["instructions"] != tc.instructions {
				t.Errorf("instructions = %#v, want %#v", req["instructions"], tc.instructions)
			}
			if got := inputTurns(req); !reflect.DeepEqual(got, tc.turns) {
				t.Errorf("input = %v, want %v", got, tc.turns)
			}
			if strings.Contains(toJSON(req["input"]), "FROM MESSAGES") {
				t.Error("messages system entry leaked into input")
			}
		})
	}
}

func toJSON(v any) string {
	b, _ := json.Marshal(v)
	return string(b)
}

// only tools with a ToolFunc become function tools; a
// NativeTool becomes a hosted tool with its customParameters; no tools
// without config tools.
func TestToolConversion(t *testing.T) {
	schema := map[string]any{"type": "object", "properties": map[string]any{"q": map[string]any{"type": "string"}}}
	noop := tool(func(map[string]any) (any, error) { return "", nil })
	config := cfg(func(c *aiserver.AiConfig) {
		c.Tools = map[string]aiserver.Tool{
			"a": funcTool("tool a", schema), "b": funcTool("tool b", schema), "missing": funcTool("no handler", schema),
			"search": {Type: "function", CustomParameters: map[string]any{"search_context_size": "low"}},
		}
	})

	h, f, _ := setup(t, textResp("ok", 1, 1))
	invoke(t, h, aiserver.HandlerRequest{Config: config, ToolHandlers: map[string]aiserver.ToolHandler{
		"a": noop, "b": noop, "search": &aiserver.NativeTool{ToolName: "web_search"},
	}})
	req := f.request(0)
	if got := toolNames(req); !reflect.DeepEqual(got, []string{"a", "b", "hosted:web_search"}) {
		t.Fatalf("tools = %v", got)
	}
	a := req["tools"].([]any)[0].(map[string]any)
	if a["type"] != "function" || a["description"] != "tool a" || !reflect.DeepEqual(a["parameters"], schema) || a["strict"] != false {
		t.Errorf("function tool = %v", a)
	}
	if ws := req["tools"].([]any)[2].(map[string]any); ws["search_context_size"] != "low" {
		t.Errorf("hosted tool customParameters not forwarded: %v", ws)
	}

	h, f, _ = setup(t, textResp("ok", 1, 1))
	invoke(t, h, aiserver.HandlerRequest{Config: cfg(func(c *aiserver.AiConfig) {
		c.Tools = map[string]aiserver.Tool{"missing": funcTool("x", schema)}
	}), ToolHandlers: map[string]aiserver.ToolHandler{"other": noop}})
	if _, ok := f.request(0)["tools"]; ok {
		t.Errorf("tools sent with no handler: %v", f.request(0)["tools"])
	}

	h, f, _ = setup(t, textResp("ok", 1, 1))
	invoke(t, h, aiserver.HandlerRequest{Config: cfg(nil), ToolHandlers: map[string]aiserver.ToolHandler{"a": noop}})
	if _, ok := f.request(0)["tools"]; ok {
		t.Errorf("tools sent without config tools: %v", f.request(0)["tools"])
	}
}

func TestToolLoop(t *testing.T) {
	schema := map[string]any{"type": "object"}
	twoTools := cfg(func(c *aiserver.AiConfig) {
		c.Tools = map[string]aiserver.Tool{"a": funcTool("", schema), "b": funcTool("", schema)}
	})

	t.Run("single call then done, consecutive calls in order", func(t *testing.T) {
		h, f, exp := setup(t, toolResp("r1", "a", `{"x":1}`), toolResp("r2", "b", `{"y":2}`), textResp("final", 5, 7))
		var order []string
		res := invoke(t, h, aiserver.HandlerRequest{Config: twoTools, ToolHandlers: map[string]aiserver.ToolHandler{
			"a": tool(func(args map[string]any) (any, error) {
				order = append(order, "a")
				return map[string]any{"got": args["x"]}, nil
			}),
			"b": tool(func(args map[string]any) (any, error) { order = append(order, "b"); return "B", nil }),
		}})
		if res.Output != "final" || !reflect.DeepEqual(order, []string{"a", "b"}) {
			t.Errorf("output %v order %v", res.Output, order)
		}
		if res.Usage["input_tokens"] != 7.0 || res.Usage["output_tokens"] != 9.0 {
			t.Errorf("usage = %v", res.Usage)
		}
		second := f.request(1)
		if second["previous_response_id"] != "r1" || !reflect.DeepEqual(inputTurns(second), []turn{{"function_call_output", `{"got":1}`}}) {
			t.Errorf("follow-up = %v", second)
		}
		if got := toolNames(second); !reflect.DeepEqual(got, []string{"a", "b"}) {
			t.Errorf("tools must be re-sent each turn (the agent may change): %v", got)
		}
		spans := spansByName(exp)
		if len(spans["chat test-model"]) != 3 || len(spans["execute_tool a"]) != 1 || len(spans["execute_tool b"]) != 1 {
			t.Errorf("spans = %v", spans)
		}
	})

	t.Run("handler throws propagates", func(t *testing.T) {
		boom := errors.New("boom")
		h, _, exp := setup(t, toolResp("r1", "a", `{}`))
		_, err := h.Invoke(bg, aiserver.HandlerRequest{Config: twoTools, ToolHandlers: map[string]aiserver.ToolHandler{
			"a": tool(func(map[string]any) (any, error) { return nil, boom }),
		}})
		if !errors.Is(err, boom) {
			t.Fatalf("err = %v", err)
		}
		if s := spansByName(exp)["execute_tool a"]; len(s) != 1 || s[0].Status().Code != codes.Error {
			t.Errorf("tool span must fail: %v", s)
		}
	})

	t.Run("no config tools: one call, handlers never invoked", func(t *testing.T) {
		h, f, _ := setup(t, textResp("ok", 1, 1))
		called := false
		invoke(t, h, aiserver.HandlerRequest{Config: cfg(nil), ToolHandlers: map[string]aiserver.ToolHandler{
			"a": tool(func(map[string]any) (any, error) { called = true; return "", nil }),
		}})
		if called || f.count() != 1 {
			t.Errorf("called=%v requests=%d", called, f.count())
		}
	})

	t.Run("a call for a tool that was not offered fails without reaching the raw handler map", func(t *testing.T) {
		h, _, _ := setup(t, toolResp("r1", "hidden", `{}`))
		reached := false
		_, err := h.Invoke(bg, aiserver.HandlerRequest{Config: cfg(nil), ToolHandlers: map[string]aiserver.ToolHandler{
			"hidden": tool(func(map[string]any) (any, error) { reached = true; return "", nil }),
		}})
		if !errors.Is(err, ErrToolNotOffered) || !strings.Contains(err.Error(), "hidden") || reached {
			t.Errorf("err=%v reached=%v", err, reached)
		}
	})

	t.Run("max turns", func(t *testing.T) {
		h, f, _ := setup(t, toolResp("r1", "a", `{}`), toolResp("r2", "a", `{}`), toolResp("r3", "a", `{}`))
		_, err := h.Invoke(bg, aiserver.HandlerRequest{Config: cfg(func(c *aiserver.AiConfig) {
			c.Tools = twoTools.Tools
			c.Model.Parameters = map[string]any{"max_turns": 2.0}
		}), ToolHandlers: map[string]aiserver.ToolHandler{"a": tool(func(map[string]any) (any, error) { return "", nil })}})
		if !errors.Is(err, ErrMaxTurns) || f.count() != 2 {
			t.Errorf("err=%v requests=%d", err, f.count())
		}
	})
}

// A hosted tool call is tracked as `$ld:ai:tool_call` with the LD tool key
// through the core's wrapped NativeTool.
func TestNativeToolTracked(t *testing.T) {
	ld := &fakeLD{fallback: map[string]any{
		"model": map[string]any{"name": "test-model"}, "provider": map[string]any{"name": "OpenAI"},
		"instructions": "I", "tools": map[string]any{"ld-search": map[string]any{"name": "ld-search", "type": "function", "parameters": map[string]any{}}},
		"_ldMeta": map[string]any{"enabled": true, "variationKey": "v", "mode": "agent"},
	}}
	installClient(t, ld)
	_, c := fakeClient(t, response("r", `[{"type":"web_search_call","id":"ws","status":"completed","action":{"type":"search","query":"q"}},`+textOutput("found")[1:], 1, 0, 1))
	res, err := aiserver.Config(aiserver.ConfigArgs{Key: "cfg", Handlers: []*aiserver.ProviderHandler{newHandler(Options{Client: c})},
		ToolHandlers: map[string]aiserver.ToolHandler{"ld-search": &aiserver.NativeTool{ToolName: "web_search"}}}).Invoke(bg, "q", ldcontext.New("u"))
	if err != nil {
		t.Fatal(err)
	}
	calls := ld.events("$ld:ai:tool_call")
	if res.Response != "found" || len(calls) != 1 || calls[0].data["toolKey"] != "ld-search" {
		t.Errorf("response %v tool calls %+v", res.Response, calls)
	}
}

var ldFixture = aiserver.TrackData{ConfigKey: "test-config", VariationKey: "variation-a", RunID: "run-123", Version: 1, ModelName: "test-model", ProviderName: "TestProvider"}

func assertRoot(t *testing.T, exp *tracetest.InMemoryExporter, name string, in, out, cached int, completion string) sdktrace.ReadOnlySpan {
	t.Helper()
	roots := spansByName(exp)[name]
	if len(roots) != 1 {
		t.Fatalf("want one %q span, got %v", name, spansByName(exp))
	}
	root := roots[0]
	a := attrs(root)
	want := map[string]string{
		"gen_ai.system": "openai", "gen_ai.operation.name": "invoke_agent", "gen_ai.request.model": "test-model",
		"gen_ai.response.model": servedModel, "gen_ai.usage.input_tokens": itoa(in), "gen_ai.usage.output_tokens": itoa(out),
		"gen_ai.usage.total_tokens": itoa(in + out), "gen_ai.usage.prompt_tokens": itoa(in), "gen_ai.usage.completion_tokens": itoa(out),
		"gen_ai.usage.cache_read.input_tokens": itoa(cached),
		"gen_ai.prompt.0.role":                 "system", "gen_ai.prompt.0.content": "Be terse", "gen_ai.prompt.1.role": "user", "gen_ai.prompt.1.content": "hi",
		"gen_ai.completion.0.role": "assistant", "gen_ai.completion.0.content": completion,
		"launchdarkly.operation.type": "gen_ai", "launchdarkly.config.key": "test-config", "launchdarkly.variation.key": "variation-a",
		"launchdarkly.run.id": "run-123",
	}
	for k, v := range want {
		if a[k] != v {
			t.Errorf("%s = %q, want %q", k, a[k], v)
		}
	}
	if _, ok := a["launchdarkly.graph.key"]; ok {
		t.Error("launchdarkly.graph.key set without graphKey")
	}
	ev := eventNames(root)
	if !slices.Contains(ev, "gen_ai.content.prompt") || !slices.Contains(ev, "gen_ai.content.completion") {
		t.Errorf("events = %v", ev)
	}
	if p, _ := eventAttr(root, "gen_ai.content.prompt", "gen_ai.prompt"); !strings.Contains(p, "Be terse") || !strings.Contains(p, "hi") {
		t.Errorf("gen_ai.content.prompt gen_ai.prompt = %q, want the system prompt and user input", p)
	}
	if c, ok := eventAttr(root, "gen_ai.content.completion", "gen_ai.completion"); !ok || c != completion {
		t.Errorf("gen_ai.content.completion gen_ai.completion = %q, want %q", c, completion)
	}
	if root.Status().Code != codes.Ok {
		t.Errorf("status = %v", root.Status())
	}
	for _, s := range exp.GetSpans().Snapshots() {
		if s.Name() != name && s.Parent().SpanID() != root.SpanContext().SpanID() {
			t.Errorf("%s is not a child of the root", s.Name())
		}
	}
	return root
}

func itoa(n int) string { return strings.TrimSpace(toJSON(n)) }

func TestTelemetry(t *testing.T) {
	h, _, exp := setup(t, response("r", textOutput("done"), 10, 4, 3))
	invoke(t, h, aiserver.HandlerRequest{Config: cfg(instructions("Be terse")), UserInput: "hi",
		Variables: map[string]any{aiserver.VarLD: ldFixture}})
	assertRoot(t, exp, "openai.agent.run", 10, 3, 4, "done")
	if SpanName != "openai.agent.run" || StreamSpanName != "openai.agent.run.stream" {
		t.Error("span name constants changed")
	}
	chat := spansByName(exp)["chat test-model"]
	if len(chat) != 1 || attrs(chat[0])["gen_ai.response.finish_reasons"] != `["stop"]` {
		t.Errorf("chat span = %v", chat)
	}
	if _, ok := attrs(chat[0])["launchdarkly.config.key"]; ok {
		t.Error("child span carries LD association")
	}

	t.Run("graph key when present", func(t *testing.T) {
		h, _, exp := setup(t, textResp("ok", 1, 1))
		td := ldFixture
		td.GraphKey = "g-1"
		invoke(t, h, aiserver.HandlerRequest{Config: cfg(instructions("sys")), Variables: map[string]any{aiserver.VarLD: td}})
		if attrs(spansByName(exp)[SpanName][0])["launchdarkly.graph.key"] != "g-1" {
			t.Error("graph key missing")
		}
	})
}

func TestProviderError(t *testing.T) {
	h, _, exp := setup(t, "STATUS400")
	_, err := h.Invoke(bg, aiserver.HandlerRequest{Config: cfg(nil), UserInput: "hi"})
	var apiErr *openai.Error
	if !errors.As(err, &apiErr) || apiErr.StatusCode != 400 {
		t.Fatalf("err = %v", err)
	}
	if root := spansByName(exp)[SpanName]; len(root) != 1 {
		t.Fatalf("spans = %v", spansByName(exp))
	}
	for _, s := range exp.GetSpans().Snapshots() {
		if s.Status().Code != codes.Error || !slices.Contains(eventNames(s), "exception") || s.EndTime().IsZero() {
			t.Errorf("%s: status %v events %v", s.Name(), s.Status().Code, eventNames(s))
		}
	}
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

func TestStream(t *testing.T) {
	t.Run("chunks, one done with usage and output, telemetry", func(t *testing.T) {
		h, _, exp := setup(t, sse([]string{"do", "ne"}, response("r", textOutput("done"), 10, 4, 3)))
		evs, err := collect(t, h, aiserver.HandlerRequest{Config: cfg(func(c *aiserver.AiConfig) {
			c.Instructions = "Be terse"
			c.OutputFormat = map[string]any{"properties": map[string]any{"a": map[string]any{"type": "string"}}}
		}), UserInput: "hi", Variables: map[string]any{aiserver.VarLD: ldFixture}})
		if err != nil {
			t.Fatal(err)
		}
		if len(evs) != 3 || evs[0].Text != "do" || evs[1].Text != "ne" || evs[2].Type != "done" || *evs[2].Output != "done" ||
			evs[2].Usage["input_tokens"] != 10.0 || evs[2].Usage["output_tokens"] != 3.0 {
			t.Fatalf("events = %+v", evs)
		}
		assertRoot(t, exp, StreamSpanName, 10, 3, 4, "done")
	})

	t.Run("tool loop: chunks from both turns, done output concatenated, args passed", func(t *testing.T) {
		first := response("r1", `[{"type":"message","id":"m","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Let me check. ","annotations":[]}]},`+callItem("a", `{"q":"x"}`)+`]`, 1, 0, 1)
		h, f, _ := setup(t, sse([]string{"Let me check. "}, first), sse([]string{"It is ", "42"}, response("r2", textOutput("It is 42"), 2, 0, 2)))
		var got map[string]any
		var order []string // chunks and the tool call, in the order the consumer saw them
		var done aiserver.HandlerStreamEvent
		for ev, err := range h.Stream(bg, aiserver.HandlerRequest{Config: cfg(func(c *aiserver.AiConfig) {
			c.Tools = map[string]aiserver.Tool{"a": funcTool("", map[string]any{"type": "object"})}
		}), ToolHandlers: map[string]aiserver.ToolHandler{"a": tool(func(args map[string]any) (any, error) {
			order = append(order, "tool")
			got = args
			return "ok", nil
		})}}) {
			if err != nil {
				t.Fatal(err)
			}
			if ev.Type == aiserver.EventChunk {
				order = append(order, ev.Text)
			} else {
				done = ev
			}
		}
		if !reflect.DeepEqual(order, []string{"Let me check. ", "tool", "It is ", "42"}) || done.Type != aiserver.EventDone ||
			*done.Output != "Let me check. It is 42" || got["q"] != "x" {
			t.Errorf("order %q done %+v args %v", order, done, got)
		}
		if f.request(0)["stream"] != true || f.request(1)["previous_response_id"] != "r1" {
			t.Errorf("requests = %v", f.requests)
		}
		if _, ok := f.request(0)["text"]; ok {
			t.Error("streaming must ignore outputFormat")
		}
	})

	t.Run("provider error ends the span with ERROR", func(t *testing.T) {
		h, _, exp := setup(t, "STATUS400")
		if _, err := collect(t, h, aiserver.HandlerRequest{Config: cfg(nil)}); err == nil {
			t.Fatal("want error")
		}
		if s := spansByName(exp)[StreamSpanName]; len(s) != 1 {
			t.Fatalf("spans = %v", spansByName(exp))
		}
		for _, s := range exp.GetSpans().Snapshots() {
			if s.Status().Code != codes.Error || !slices.Contains(eventNames(s), "exception") || s.EndTime().IsZero() {
				t.Errorf("%s: status %v events %v", s.Name(), s.Status().Code, eventNames(s))
			}
		}
	})

	t.Run("consumer break abandons and ends every span", func(t *testing.T) {
		h, _, exp := setup(t, sse([]string{"a", "b"}, response("r", textOutput("ab"), 1, 0, 1)))
		for range h.Stream(bg, aiserver.HandlerRequest{Config: cfg(nil)}) {
			break
		}
		spans := exp.GetSpans().Snapshots()
		if len(spans) != 2 {
			t.Fatalf("want root + chat ended, got %d", len(spans))
		}
		for _, s := range spans {
			if attrs(s)["launchdarkly.stream.abandoned"] != "true" {
				t.Errorf("%s not marked abandoned", s.Name())
			}
		}
	})
}

// openai-agents: outputType is the buildOutputType envelope, sent as
// text.format; the parsed object is returned directly.
func TestStructuredOutput(t *testing.T) {
	format := map[string]any{"properties": map[string]any{"score": map[string]any{"type": "number"}}}
	h, f, _ := setup(t, textResp(`{"score":7}`, 1, 1))
	res := invoke(t, h, aiserver.HandlerRequest{Config: cfg(func(c *aiserver.AiConfig) { c.OutputFormat = format })})
	if !reflect.DeepEqual(res.Output, map[string]any{"score": 7.0}) {
		t.Errorf("output = %#v", res.Output)
	}
	got := f.request(0)["text"].(map[string]any)["format"]
	want := map[string]any{"type": "json_schema", "name": "output", "strict": false, "schema": map[string]any{
		"type": "object", "properties": format["properties"], "required": []any{"score"},
	}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("text.format = %v", got)
	}

	h, f, _ = setup(t, textResp("plain", 1, 1))
	if res := invoke(t, h, aiserver.HandlerRequest{Config: cfg(nil)}); res.Output != "plain" {
		t.Errorf("output = %#v", res.Output)
	}
	if _, ok := f.request(0)["text"]; ok {
		t.Error("text sent without outputFormat")
	}

	h, _, _ = setup(t, textResp("not json", 1, 1))
	if res := invoke(t, h, aiserver.HandlerRequest{Config: cfg(func(c *aiserver.AiConfig) { c.OutputFormat = format })}); res.Output != "not json" {
		t.Errorf("unparseable output must be returned raw: %#v", res.Output)
	}
}

func TestBuildOutputType(t *testing.T) {
	if buildOutputType(nil) != nil || buildOutputType(map[string]any{}) != nil {
		t.Error("nil / empty outputFormat must give nil")
	}
	in := map[string]any{"properties": map[string]any{"field2": map[string]any{}, "field1": map[string]any{}}, "required": []any{"field1"}}
	ot := buildOutputType(in)
	if ot.Type != "json_schema" || ot.Name != "output" || ot.Strict {
		t.Errorf("envelope = %+v", ot)
	}
	if ot.Schema["type"] != "object" || !reflect.DeepEqual(ot.Schema["required"], []string{"field1", "field2"}) {
		t.Errorf("schema = %v", ot.Schema)
	}
	if _, mutated := in["type"]; mutated {
		t.Error("input mutated")
	}
	b, _ := json.Marshal(buildOutputType(map[string]any{"type": "object", "properties": map[string]any{}, "required": []any{}}))
	if string(b) != `{"type":"json_schema","name":"output","strict":false,"schema":{"properties":{},"type":"object"}}` {
		t.Errorf("empty properties: %s", b)
	}
}

// History in agent mode, including image blocks.
func TestHistory(t *testing.T) {
	img := aiserver.ContentBlock{Type: "image", Source: &aiserver.ImageSource{Type: "base64", MediaType: "image/png", Data: "AAA"}}
	urlImg := aiserver.ContentBlock{Type: "image", Source: &aiserver.ImageSource{Type: "url", URL: "https://x/y.png"}}
	history := []aiserver.Message{{Role: "system", Text: "SYS"}, {Role: "user", Text: "earlier"}, {Role: "assistant", Text: "reply"}}
	for _, tc := range []struct {
		name    string
		history []aiserver.Message
		input   string
		want    []turn
	}{
		{"structured, system filtered, before userInput", history, "now",
			[]turn{{"user", "earlier"}, {"assistant", "reply"}, {"user", "now"}}},
		{"empty history matches none", []aiserver.Message{}, "now", []turn{{"user", "now"}}},
		{"multimodal native input_image", []aiserver.Message{{Role: "user", Blocks: []aiserver.ContentBlock{{Type: "text", Text: "look"}, img, urlImg}}}, "",
			[]turn{{"user", `[{"text":"look","type":"input_text"},{"detail":"auto","image_url":"data:image/png;base64,AAA","type":"input_image"},{"detail":"auto","image_url":"https://x/y.png","type":"input_image"}]`}}},
		{"image-only history then userInput", []aiserver.Message{{Role: "user", Blocks: []aiserver.ContentBlock{img}}}, "what is it?",
			[]turn{{"user", `[{"detail":"auto","image_url":"data:image/png;base64,AAA","type":"input_image"}]`}, {"user", "what is it?"}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h, f, _ := setup(t, textResp("ok", 1, 1))
			invoke(t, h, aiserver.HandlerRequest{Config: cfg(instructions("INSTR")), UserInput: tc.input, History: tc.history})
			req := f.request(0)
			if got := inputTurns(req); !reflect.DeepEqual(got, tc.want) {
				t.Errorf("input = %v\nwant    %v", got, tc.want)
			}
			if req["instructions"] != "INSTR" || strings.Contains(toJSON(req), "Conversation History") || strings.Contains(toJSON(req), "SYS") {
				t.Errorf("history leaked into instructions: %v", req["instructions"])
			}
		})
	}
}

// Allowlisted model parameters are forwarded; max_tokens is renamed,
// verbosity becomes text.verbosity, max_turns caps the run, others drop.
func TestModelParameters(t *testing.T) {
	h, f, _ := setup(t, textResp("ok", 1, 1))
	invoke(t, h, aiserver.HandlerRequest{Config: cfg(func(c *aiserver.AiConfig) {
		c.Model.Parameters = map[string]any{"temperature": 0.2, "max_tokens": 50.0, "verbosity": "low", "max_turns": 3.0,
			"frequency_penalty": 1.0, "model": "evil", "instructions": "evil", "stream": true}
	})})
	req := f.request(0)
	if req["temperature"] != 0.2 || req["max_output_tokens"] != 50.0 || req["model"] != "test-model" {
		t.Errorf("req = %v", req)
	}
	if !reflect.DeepEqual(req["text"], map[string]any{"verbosity": "low"}) {
		t.Errorf("text = %v", req["text"])
	}
	for _, k := range []string{"max_turns", "frequency_penalty", "instructions", "stream", "verbosity", "max_tokens"} {
		if _, ok := req[k]; ok {
			t.Errorf("%s forwarded", k)
		}
	}
}

// camelCase keys (the JavaScript ModelSettings spelling) forward like their
// snake_case names, snake_case wins, and an explicit text.verbosity wins.
func TestModelParameterSpellings(t *testing.T) {
	h, f, _ := setup(t, textResp("ok", 1, 1))
	invoke(t, h, aiserver.HandlerRequest{Config: cfg(func(c *aiserver.AiConfig) {
		c.Model.Parameters = map[string]any{"topP": 0.5, "maxTokens": 40.0, "parallelToolCalls": false,
			"temperature": 0.1, "toolChoice": "none", "tool_choice": "auto",
			"verbosity": "low", "text": map[string]any{"verbosity": "high", "format": "evil"}}
	})})
	req := f.request(0)
	want := map[string]any{"top_p": 0.5, "max_output_tokens": 40.0, "parallel_tool_calls": false,
		"temperature": 0.1, "tool_choice": "auto", "text": map[string]any{"verbosity": "high"}}
	for k, v := range want {
		if !reflect.DeepEqual(req[k], v) {
			t.Errorf("%s = %v, want %v", k, req[k], v)
		}
	}
	for _, k := range []string{"topP", "maxTokens", "parallelToolCalls", "toolChoice"} {
		if _, ok := req[k]; ok {
			t.Errorf("%s forwarded verbatim", k)
		}
	}
}

var variation = map[string]any{
	"model": map[string]any{"name": "test-model"}, "provider": map[string]any{"name": "OpenAI"},
	"instructions": "say {{user_input}}",
	"_ldMeta":      map[string]any{"enabled": true, "variationKey": "v", "mode": "agent"},
}

func TestConvenienceExport(t *testing.T) {
	f, client := fakeClient(t, textResp("one", 1, 1), textResp("two", 1, 1))
	ld := &fakeLD{fallback: variation}
	installClient(t, ld)

	res, err := OpenAIAgents(bg, "cfg", "hi", ldcontext.New("u-1"), &InvokeOptions{
		Variables: map[string]any{"user_input": "hello"}, Handler: Options{Client: client},
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.Response != "one" || res.TrackData.ConfigKey != "cfg" || !slices.Equal(ld.keys, []string{"cfg"}) {
		t.Errorf("response = %+v keys %v", res, ld.keys)
	}
	if f.request(0)["instructions"] != "say hello" {
		t.Errorf("variables not forwarded to invoke: %v", f.request(0)["instructions"])
	}
	if got := ld.usageHelpers(); !reflect.DeepEqual(got, []string{"openai-agents.openaiAgents"}) {
		t.Errorf("usage helpers = %v", got)
	}
	if _, err := OpenAIAgents(bg, "cfg", "hi", ldcontext.New("u-1"), &InvokeOptions{Handler: Options{Client: client}}); err != nil {
		t.Fatal(err)
	}
	if f.request(1)["instructions"] != "say {{user_input}}" {
		t.Errorf("unexpected variables: %v", f.request(1)["instructions"])
	}
}

func TestConvenienceExportNilOptions(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "test")
	t.Setenv("OPENAI_BASE_URL", "http://127.0.0.1:1") // unreachable: no network
	installClient(t, &fakeLD{fallback: variation})
	_, err := OpenAIAgents(bg, "cfg", "hi", ldcontext.New("u-1"), nil)
	var apiErr *openai.Error
	if err == nil || errors.As(err, &apiErr) {
		t.Errorf("want a connection error from the default client, got %v", err)
	}
}

// TestThroughConfig drives the handler via aiserver.Config with a tool turn:
// ldContext reaches the root span as context identity, and no chat or
// execute_tool span carries identity or a feature_flag event.
func TestThroughConfig(t *testing.T) {
	_, client := fakeClient(t, toolResp("r1", "a", `{}`), textResp("ok", 1, 1))
	toolVariation := maps.Clone(variation)
	toolVariation["tools"] = map[string]any{"a": map[string]any{"name": "a", "type": "function", "parameters": map[string]any{"type": "object"}}}
	exp := installClient(t, &fakeLD{fallback: toolVariation})

	ran := false
	multi := ldcontext.NewMulti(ldcontext.New("u-1"), ldcontext.NewWithKind("org", "o-1"))
	if _, err := aiserver.Config(aiserver.ConfigArgs{Key: "cfg", Handlers: []*aiserver.ProviderHandler{newHandler(Options{Client: client})},
		ToolHandlers: map[string]aiserver.ToolHandler{"a": tool(func(map[string]any) (any, error) { ran = true; return "r", nil })},
	}).Invoke(bg, "q", multi); err != nil {
		t.Fatal(err)
	}
	spans := spansByName(exp)
	if !ran || len(spans[SpanName]) != 1 || len(spans["chat test-model"]) != 2 || len(spans["execute_tool a"]) != 1 {
		t.Fatalf("tool ran %v, span tree = %v", ran, spans)
	}
	for _, s := range exp.GetSpans().Snapshots() {
		var keys []string
		for k := range attrs(s) {
			if strings.HasPrefix(k, "context.") {
				keys = append(keys, k)
			}
		}
		if s.Name() != SpanName {
			if len(keys) != 0 || slices.Contains(eventNames(s), "feature_flag") {
				t.Errorf("child %s carries context identity %v or events %v", s.Name(), keys, eventNames(s))
			}
			continue
		}
		slices.Sort(keys)
		if !reflect.DeepEqual(keys, []string{"context.contextKeys.org", "context.contextKeys.user"}) {
			t.Errorf("root context keys = %v", keys)
		}
		contextKeys, _ := eventAttr(s, "feature_flag", "feature_flag.contextKeys")
		contextID, _ := eventAttr(s, "feature_flag", "feature_flag.context.id")
		if contextKeys != `{"org":"o-1","user":"u-1"}` || contextID == "" {
			t.Errorf("feature_flag event contextKeys=%q id=%q", contextKeys, contextID)
		}
	}
}

// each helper reports once, only itself, with this package's identity.
func TestSdkUsage(t *testing.T) {
	ld := &fakeLD{}
	installClient(t, ld)
	NewHandler(Options{})
	NewHandler(Options{})
	OpenAIGraph("g", aiserver.GraphOptions{}, Options{})
	ToOpenAIAgents(nil, nil)
	want := []string{"openai-agents.createOpenAIAgentHandler", "openai-agents.openaiGraph", "openai-agents.toOpenAIAgents"}
	if got := ld.usageHelpers(); !reflect.DeepEqual(got, want) {
		t.Fatalf("usage helpers = %v (no client.createHandler / client.graph.*, no repeats)", got)
	}
	for _, tr := range ld.events("$ld:ai:sdk:usage") {
		if tr.data["helperPackageName"] != PackageName || tr.data["helperPackageVersion"] != PackageVersion ||
			tr.data["aiSdkName"] != aiserver.PackageName || tr.data["aiSdkLanguage"] != "go" || tr.value != 1 ||
			tr.ctx.Kind() != "ld_ai" || !tr.ctx.Anonymous() {
			t.Errorf("usage event = %+v", tr)
		}
	}
	var info bool
	for _, tr := range ld.events("$ld:ai:sdk:info") {
		info = info || (tr.data["aiSdkName"] == PackageName && tr.data["aiSdkVersion"] == PackageVersion)
	}
	if !info {
		t.Error("init() must register the package for $ld:ai:sdk:info")
	}
}
