package openaimessages

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"net/http"
	"net/http/httptest"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/openaiwire"
)

// fakeOpenAI replays canned responses and records request bodies.
type fakeOpenAI struct {
	mu        sync.Mutex
	responses []string // JSON bodies; SSE streams start with "event:"; "STATUS400" is an API error
	requests  []map[string]any
}

func (f *fakeOpenAI) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	b, _ := io.ReadAll(r.Body)
	var body map[string]any
	_ = json.Unmarshal(b, &body)
	f.mu.Lock()
	f.requests = append(f.requests, body)
	i := len(f.requests) - 1
	f.mu.Unlock()
	if i >= len(f.responses) {
		http.Error(w, `{"error":{"message":"no more responses","type":"server_error"}}`, 500)
		return
	}
	resp := f.responses[i]
	switch {
	case resp == "STATUS400":
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(400)
		resp = `{"error":{"message":"bad request","type":"invalid_request_error"}}`
	case strings.HasPrefix(resp, "event:"):
		w.Header().Set("Content-Type", "text/event-stream")
	default:
		w.Header().Set("Content-Type", "application/json")
	}
	_, _ = io.WriteString(w, resp)
}

const servedModel = "gpt-test-2026-01-01" // what the API reports; requests use "test-model"

func textOutput(text string) string {
	return fmt.Sprintf(`[{"type":"message","id":"msg","role":"assistant","status":"completed","content":[{"type":"output_text","text":%q,"annotations":[]}]}]`, text)
}

func callOutput(name, args string) string {
	return fmt.Sprintf(`[{"type":"function_call","id":"fc_%s","call_id":"call_%s","name":%q,"arguments":%q,"status":"completed"}]`, name, name, name, args)
}

func response(id, output string, in, cached, out int) string {
	return fmt.Sprintf(`{"id":%q,"object":"response","created_at":0,"status":"completed","model":%q,"output":%s,"usage":{"input_tokens":%d,"input_tokens_details":{"cached_tokens":%d},"output_tokens":%d,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":%d}}`,
		id, servedModel, output, in, cached, out, in+out)
}

func textResp(text string, in, out int) string {
	return response("resp_text", textOutput(text), in, 0, out)
}

func toolResp(id, name, args string) string { return response(id, callOutput(name, args), 1, 0, 1) }

// sse streams deltas and ends with response.completed carrying final.
func sse(deltas []string, final string) string {
	var b strings.Builder
	ev := func(name, data string) { fmt.Fprintf(&b, "event: %s\ndata: %s\n\n", name, data) }
	ev("response.created", `{"type":"response.created","sequence_number":0,"response":{"id":"r","status":"in_progress","output":[]}}`)
	for i, d := range deltas {
		ev("response.output_text.delta", fmt.Sprintf(`{"type":"response.output_text.delta","sequence_number":%d,"item_id":"msg","output_index":0,"content_index":0,"delta":%q}`, i+1, d))
	}
	ev("response.completed", fmt.Sprintf(`{"type":"response.completed","sequence_number":99,"response":%s}`, final))
	return b.String()
}

func setup(t *testing.T, responses ...string) (*aiserver.ProviderHandler, *fakeOpenAI, *tracetest.InMemoryExporter) {
	t.Helper()
	f, c := fakeClient(t, responses...)
	exp := tracetest.NewInMemoryExporter()
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(sdktrace.NewTracerProvider(sdktrace.WithSyncer(exp)))
	t.Cleanup(func() { otel.SetTracerProvider(prev) })
	return newHandler(Options{Client: c}), f, exp
}

func fakeClient(t *testing.T, responses ...string) (*fakeOpenAI, *openai.Client) {
	t.Helper()
	f := &fakeOpenAI{responses: responses}
	srv := httptest.NewServer(f)
	t.Cleanup(srv.Close)
	c := openai.NewClient(option.WithBaseURL(srv.URL), option.WithAPIKey("test"), option.WithMaxRetries(0))
	return f, &c
}

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

var bg = context.Background()

// turn is one outbound input message, flattened for comparison: content is
// the string content, or the JSON of a content-part array.
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
			b, _ := json.Marshal(m["content"])
			content = string(b)
		}
		out = append(out, turn{role, content})
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

func TestFactoryMetadata(t *testing.T) {
	a, b := newHandler(Options{}), newHandler(Options{CaptureContent: aiserver.CaptureOn})
	if a.ProvidesFor != (aiserver.ProvidesFor{Provider: "OpenAI", Mode: aiserver.ModeMessages}) || a.ProvidesFor != ProvidesFor {
		t.Errorf("providesFor = %+v", a.ProvidesFor)
	}
	if a.CaptureContent != aiserver.CaptureDefault || b.CaptureContent != aiserver.CaptureOn {
		t.Errorf("captureContent default/passthrough wrong: %v %v", a.CaptureContent, b.CaptureContent)
	}
	if a == b || a.Invoke == nil || a.Stream == nil {
		t.Errorf("factory must return independent callable instances")
	}
}

// Prompt construction: instructions path, messages path, and precedence.
func TestPromptConstruction(t *testing.T) {
	for _, tc := range []struct {
		name      string
		config    aiserver.AiConfig
		userInput string
		vars      map[string]any
		want      []turn
	}{
		{"instructions path: with resolved and unresolved placeholders",
			cfg(instructions("Hi {{name}} {{missing}}")), "q", map[string]any{"name": "Al"},
			[]turn{{"system", "Hi Al {{missing}}"}, {"user", "q"}}},
		{"instructions path: empty userInput still appends an empty user turn",
			cfg(instructions("sys")), "", nil,
			[]turn{{"system", "sys"}, {"user", ""}}},
		{"messages path: system extracted, placeholders resolved, last user not duplicated",
			cfg(func(c *aiserver.AiConfig) {
				c.Messages = []aiserver.Message{{Role: "system", Text: "sys {{x}}"}, {Role: "user", Text: "u {{x}}"}}
			}), "ignored", map[string]any{"x": "1"},
			[]turn{{"system", "sys 1"}, {"user", "u 1"}}},
		{"messages path: last assistant appends userInput",
			cfg(func(c *aiserver.AiConfig) {
				c.Messages = []aiserver.Message{{Role: "user", Text: "hello"}, {Role: "assistant", Text: "hi"}}
			}), "next", nil,
			[]turn{{"user", "hello"}, {"assistant", "hi"}, {"user", "next"}}},
		{"messages path: last assistant with empty userInput appends empty string",
			cfg(func(c *aiserver.AiConfig) { c.Messages = []aiserver.Message{{Role: "assistant", Text: "hi"}} }), "", nil,
			[]turn{{"assistant", "hi"}, {"user", ""}}},
		{"precedence: messages win over instructions in messages mode",
			cfg(func(c *aiserver.AiConfig) {
				c.Instructions = "IGNORED"
				c.Messages = []aiserver.Message{{Role: "system", Text: "from messages"}}
			}), "q", nil,
			[]turn{{"system", "from messages"}, {"user", "q"}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h, f, _ := setup(t, textResp("ok", 1, 1))
			if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: tc.config, UserInput: tc.userInput, Variables: tc.vars}); err != nil {
				t.Fatal(err)
			}
			if got := inputTurns(f.requests[0]); !reflect.DeepEqual(got, tc.want) {
				t.Errorf("input = %v, want %v", got, tc.want)
			}
			if f.requests[0]["model"] != "test-model" {
				t.Errorf("model = %v", f.requests[0]["model"])
			}
		})
	}
}

func TestToolConversion(t *testing.T) {
	params := map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}, "required": []any{"city"}, "additionalProperties": false}
	noop := tool(func(map[string]any) (any, error) { return nil, nil })
	for _, tc := range []struct {
		name     string
		tools    map[string]aiserver.Tool
		handlers map[string]aiserver.ToolHandler
		want     []string
	}{
		{"three config tools, two handlers", map[string]aiserver.Tool{
			"a": funcTool("tool a", params), "b": funcTool("", map[string]any{"type": "object"}), "c": funcTool("", map[string]any{}),
		}, map[string]aiserver.ToolHandler{"a": noop, "b": noop}, []string{"a", "b"}},
		{"tool without handler excluded", map[string]aiserver.Tool{"a": funcTool("", params)}, nil, nil},
		{"native tool is not a local function", map[string]aiserver.Tool{"a": funcTool("", params)},
			map[string]aiserver.ToolHandler{"a": &aiserver.NativeTool{ToolName: "web_search"}}, nil},
		{"no config tools", nil, map[string]aiserver.ToolHandler{"a": noop}, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h, f, _ := setup(t, textResp("ok", 1, 1))
			_, err := h.Invoke(bg, aiserver.HandlerRequest{Config: cfg(func(c *aiserver.AiConfig) { c.Tools = tc.tools }), UserInput: "q", ToolHandlers: tc.handlers})
			if err != nil {
				t.Fatal(err)
			}
			sent, present := f.requests[0]["tools"].([]any)
			if len(tc.want) == 0 {
				if present {
					t.Errorf("tools must be absent, got %v", sent)
				}
				return
			}
			var names []string
			for _, s := range sent {
				st := s.(map[string]any)
				names = append(names, st["name"].(string))
				if st["type"] != "function" || st["strict"] != false {
					t.Errorf("tool shape: %v", st)
				}
				if st["name"] == "a" && (!reflect.DeepEqual(st["parameters"], params) || st["description"] != "tool a") {
					t.Errorf("tool a modified: %v", st)
				}
			}
			if strings.Join(slices.Sorted(slices.Values(names)), ",") != strings.Join(tc.want, ",") {
				t.Errorf("tools sent = %v, want %v", names, tc.want)
			}
		})
	}
}

// (no text.format or tools on follow-ups).
func TestToolLoop(t *testing.T) {
	toolCfg := cfg(func(c *aiserver.AiConfig) {
		c.Instructions = "x"
		c.OutputFormat = map[string]any{"type": "object"}
		c.Tools = map[string]aiserver.Tool{"a": funcTool("", map[string]any{"type": "object"}), "b": funcTool("", map[string]any{"type": "object"})}
	})
	t.Run("consecutive calls then done", func(t *testing.T) {
		h, f, _ := setup(t, toolResp("resp_1", "a", `{"city":"Paris"}`), toolResp("resp_2", "b", `{}`), textResp(`{"done":true}`, 2, 3))
		var calls []string
		var gotArgs map[string]any
		res, err := h.Invoke(bg, aiserver.HandlerRequest{Config: toolCfg, UserInput: "q", ToolHandlers: map[string]aiserver.ToolHandler{
			"a": tool(func(a map[string]any) (any, error) { calls, gotArgs = append(calls, "a"), a; return "sunny", nil }),
			"b": tool(func(map[string]any) (any, error) { calls = append(calls, "b"); return map[string]any{"ok": true}, nil }),
		}})
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(calls, []string{"a", "b"}) || gotArgs["city"] != "Paris" || res.Output != `{"done":true}` {
			t.Errorf("calls=%v args=%v out=%v", calls, gotArgs, res.Output)
		}
		if u := aiserver.ParseUsage(res.Usage); u.Input != 4 || u.Output != 5 {
			t.Errorf("usage summed across turns = %+v", u)
		}
		if len(f.requests) != 3 {
			t.Fatalf("requests = %d", len(f.requests))
		}
		if _, ok := f.requests[0]["text"]; !ok {
			t.Error("first call must carry text.format")
		}
		for i, want := range []struct{ prev, output string }{{"resp_1", "sunny"}, {"resp_2", `{"ok":true}`}} {
			req := f.requests[i+1]
			_, hasTools := req["tools"]
			_, hasText := req["text"]
			// previous_response_id does not carry tools or text.format, so each follow-up resends both.
			if req["previous_response_id"] != want.prev || !hasTools || !hasText || !reflect.DeepEqual(req["tools"], f.requests[0]["tools"]) {
				t.Errorf("follow-up %d: %v", i, req)
			}
			items := req["input"].([]any)
			out := items[0].(map[string]any)
			if len(items) != 1 || out["type"] != "function_call_output" || out["output"] != want.output || !strings.HasPrefix(out["call_id"].(string), "call_") {
				t.Errorf("follow-up %d input = %v", i, items)
			}
		}
	})
	t.Run("tool error propagates", func(t *testing.T) {
		h, _, exp := setup(t, toolResp("resp_1", "a", `{}`))
		boom := errors.New("tool boom")
		_, err := h.Invoke(bg, aiserver.HandlerRequest{Config: toolCfg, ToolHandlers: map[string]aiserver.ToolHandler{
			"a": tool(func(map[string]any) (any, error) { return nil, boom })}})
		if !errors.Is(err, boom) {
			t.Fatalf("tool error not propagated: %v", err)
		}
		if s := spansByName(exp)["execute_tool a"]; len(s) != 1 || s[0].Status().Code != codes.Error {
			t.Errorf("tool span not failed: %v", s)
		}
	})
	t.Run("no tools in config calls provider once", func(t *testing.T) {
		h, f, _ := setup(t, textResp("ok", 1, 1))
		called := false
		_, err := h.Invoke(bg, aiserver.HandlerRequest{Config: cfg(instructions("x")), ToolHandlers: map[string]aiserver.ToolHandler{
			"a": tool(func(map[string]any) (any, error) { called = true; return nil, nil })}})
		if err != nil || called || len(f.requests) != 1 {
			t.Errorf("err=%v called=%v requests=%d", err, called, len(f.requests))
		}
	})
	t.Run("call for a tool that was not offered is rejected", func(t *testing.T) {
		h, f, _ := setup(t, toolResp("resp_1", "dangerous", `{}`))
		called := false
		_, err := h.Invoke(bg, aiserver.HandlerRequest{Config: toolCfg, ToolHandlers: map[string]aiserver.ToolHandler{
			"dangerous": tool(func(map[string]any) (any, error) { called = true; return nil, nil })}})
		if err == nil || !strings.Contains(err.Error(), "dangerous") || called || len(f.requests) != 1 {
			t.Errorf("err=%v called=%v requests=%d", err, called, len(f.requests))
		}
	})
	t.Run("max steps", func(t *testing.T) {
		loop := func(n int, final bool) ([]string, int) {
			var rs []string
			for i := range n {
				rs = append(rs, toolResp(fmt.Sprint("resp_", i), "a", `{}`))
			}
			if final {
				rs = append(rs, textResp("done", 1, 1))
			}
			return rs, n
		}
		ok, _ := loop(maxSteps, true)
		h, _, _ := setup(t, ok...)
		handlers := map[string]aiserver.ToolHandler{"a": tool(func(map[string]any) (any, error) { return "r", nil })}
		if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: toolCfg, ToolHandlers: handlers}); err != nil {
			t.Errorf("exactly maxSteps tool turns must succeed: %v", err)
		}
		over, _ := loop(maxSteps+1, false)
		h, _, _ = setup(t, over...)
		if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: toolCfg, ToolHandlers: handlers}); !errors.Is(err, ErrMaxSteps) {
			t.Errorf("want ErrMaxSteps, got %v", err)
		}
	})
}

var ldFixture = aiserver.TrackData{ConfigKey: "test-config", VariationKey: "variation-a", RunID: "run-123", Version: 1, ModelName: "test-model", ProviderName: "TestProvider"}

// assertRootTelemetry checks the root span attributes shared by both paths.
func assertRootTelemetry(t *testing.T, exp *tracetest.InMemoryExporter, wantIn, wantOut, wantCache, chats int, completion string) {
	t.Helper()
	spans := spansByName(exp)
	if len(spans[SpanName]) != 1 || len(spans["chat test-model"]) != chats {
		t.Fatalf("span tree = %v", spans)
	}
	root := spans[SpanName][0]
	a := attrs(root)
	for k, want := range map[string]string{
		"gen_ai.system": "openai", "gen_ai.operation.name": "chat", "gen_ai.request.model": "test-model",
		"gen_ai.response.model":                servedModel,
		"gen_ai.usage.input_tokens":            fmt.Sprint(wantIn),
		"gen_ai.usage.prompt_tokens":           fmt.Sprint(wantIn),
		"gen_ai.usage.output_tokens":           fmt.Sprint(wantOut),
		"gen_ai.usage.total_tokens":            fmt.Sprint(wantIn + wantOut),
		"gen_ai.usage.completion_tokens":       fmt.Sprint(wantOut),
		"gen_ai.usage.cache_read.input_tokens": fmt.Sprint(wantCache),
		"gen_ai.prompt.0.role":                 "system", "gen_ai.prompt.0.content": "sys",
		"gen_ai.prompt.1.role": "user", "gen_ai.prompt.1.content": "q",
		"gen_ai.completion.0.role": "assistant", "gen_ai.completion.0.content": completion,
		"launchdarkly.operation.type": "gen_ai", "launchdarkly.config.key": "test-config",
		"launchdarkly.variation.key": "variation-a", "launchdarkly.run.id": "run-123",
	} {
		if a[k] != want {
			t.Errorf("root %s = %q, want %q", k, a[k], want)
		}
	}
	if _, has := a["launchdarkly.graph.key"]; has {
		t.Error("graph key set without graphKey")
	}
	if ev := eventNames(root); !reflect.DeepEqual(ev[len(ev)-2:], []string{"gen_ai.content.prompt", "gen_ai.content.completion"}) {
		t.Errorf("root events = %v", ev)
	}
	if p, _ := eventAttr(root, "gen_ai.content.prompt", "gen_ai.prompt"); !strings.Contains(p, "sys") || !strings.Contains(p, "q") {
		t.Errorf("gen_ai.content.prompt gen_ai.prompt = %q, want the system prompt and user input", p)
	}
	if c, ok := eventAttr(root, "gen_ai.content.completion", "gen_ai.completion"); !ok || c != completion {
		t.Errorf("gen_ai.content.completion gen_ai.completion = %q, want %q", c, completion)
	}
	if root.Status().Code != codes.Ok || root.EndTime().IsZero() {
		t.Errorf("root status %v ended %v", root.Status().Code, !root.EndTime().IsZero())
	}
	for name, ss := range spans {
		for _, s := range ss {
			if name == SpanName {
				continue
			}
			if s.Parent().SpanID() != root.SpanContext().SpanID() {
				t.Errorf("%s is not a direct child of the root", name)
			}
			for k := range attrs(s) {
				if strings.HasPrefix(k, "launchdarkly.") || strings.HasPrefix(k, "context.") {
					t.Errorf("child %s carries %s", name, k)
				}
			}
		}
	}
}

// context identity is covered in TestThroughConfig.
func TestTelemetry(t *testing.T) {
	h, _, exp := setup(t, toolResp("resp_1", "a", `{}`), response("resp_2", textOutput("final"), 10, 4, 3))
	c := cfg(func(c *aiserver.AiConfig) {
		c.Instructions = "sys"
		c.Tools = map[string]aiserver.Tool{"a": funcTool("", map[string]any{"type": "object"})}
	})
	_, err := h.Invoke(bg, aiserver.HandlerRequest{Config: c, UserInput: "q",
		ToolHandlers: map[string]aiserver.ToolHandler{"a": tool(func(map[string]any) (any, error) { return "r", nil })},
		Variables:    map[string]any{"__ld": ldFixture}})
	if err != nil {
		t.Fatal(err)
	}
	// Cached tokens are inside input_tokens: 1 + 10, not 1 + 10 + 4.
	assertRootTelemetry(t, exp, 11, 4, 4, 2, "final")
	spans := spansByName(exp)
	if s := spans["execute_tool a"]; len(s) != 1 || attrs(s[0])["gen_ai.tool.call.id"] != "call_a" ||
		attrs(s[0])["gen_ai.tool.call.arguments"] != "{}" || attrs(s[0])["gen_ai.tool.call.result"] != "r" {
		t.Errorf("tool span = %v", s)
	}
	chat := attrs(spans["chat test-model"][1])
	if chat["gen_ai.response.model"] != servedModel || chat["gen_ai.usage.input_tokens"] != "10" || chat["gen_ai.response.finish_reasons"] != `["stop"]` {
		t.Errorf("chat span = %v", chat)
	}
	if fr := attrs(spans["chat test-model"][0])["gen_ai.response.finish_reasons"]; fr != `["tool_calls"]` {
		t.Errorf("tool turn finish reason = %s", fr)
	}

	t.Run("graph key when present", func(t *testing.T) {
		h, _, exp := setup(t, textResp("ok", 1, 1))
		td := ldFixture
		td.GraphKey = "g-1"
		if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: cfg(instructions("sys")), Variables: map[string]any{"__ld": td}}); err != nil {
			t.Fatal(err)
		}
		if attrs(spansByName(exp)[SpanName][0])["launchdarkly.graph.key"] != "g-1" {
			t.Error("graph key missing")
		}
	})
	t.Run("response model falls back to requested", func(t *testing.T) {
		h, _, exp := setup(t, strings.Replace(textResp("ok", 1, 1), servedModel, "", 1))
		if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: cfg(instructions("sys"))}); err != nil {
			t.Fatal(err)
		}
		if m := attrs(spansByName(exp)[SpanName][0])["gen_ai.response.model"]; m != "test-model" {
			t.Errorf("response model = %q", m)
		}
	})
}

func TestProviderError(t *testing.T) {
	h, _, exp := setup(t, "STATUS400")
	_, err := h.Invoke(bg, aiserver.HandlerRequest{Config: cfg(instructions("x"))})
	var apiErr *openai.Error
	if !errors.As(err, &apiErr) || apiErr.StatusCode != 400 {
		t.Fatalf("want the provider error, got %v", err)
	}
	spans := exp.GetSpans().Snapshots()
	if len(spans) != 2 {
		t.Fatalf("spans = %d", len(spans))
	}
	for _, s := range spans {
		if s.Status().Code != codes.Error || !slices.Contains(eventNames(s), "exception") || s.EndTime().IsZero() {
			t.Errorf("%s: status %v events %v", s.Name(), s.Status().Code, eventNames(s))
		}
		if s.Name() == SpanName {
			if _, has := attrs(s)["gen_ai.usage.input_tokens"]; has {
				t.Error("root must not report usage when no turn completed")
			}
		}
	}
}

func TestStream(t *testing.T) {
	t.Run("chunks, done, usage, telemetry", func(t *testing.T) {
		h, _, exp := setup(t, sse([]string{"Hel", "lo"}, response("r1", textOutput("Hello"), 3, 1, 2)))
		var events []aiserver.HandlerStreamEvent
		for ev, err := range h.Stream(bg, aiserver.HandlerRequest{Config: cfg(instructions("sys")), UserInput: "q", Variables: map[string]any{"__ld": ldFixture}}) {
			if err != nil {
				t.Fatal(err)
			}
			events = append(events, ev)
		}
		if len(events) != 3 || events[0].Text != "Hel" || events[1].Text != "lo" || events[2].Type != "done" || *events[2].Output != "Hello" {
			t.Fatalf("events = %+v", events)
		}
		if u := aiserver.ParseUsage(events[2].Usage); u.Input != 3 || u.Output != 2 {
			t.Errorf("usage = %+v", u)
		}
		assertRootTelemetry(t, exp, 3, 2, 1, 1, "Hello")
	})
	t.Run("tool loop across streamed turns", func(t *testing.T) {
		h, f, exp := setup(t,
			sse([]string{"Let me ", "check. "}, response("r1", callOutput("a", `{"city":"Paris"}`), 2, 0, 1)),
			sse([]string{"Sunny."}, response("r2", textOutput("Sunny."), 3, 0, 2)))
		var order []string
		var gotArgs map[string]any
		c := cfg(func(c *aiserver.AiConfig) {
			c.Instructions = "x"
			c.OutputFormat = map[string]any{"type": "object"} // streaming ignores it
			c.Tools = map[string]aiserver.Tool{"a": funcTool("", map[string]any{"type": "object"})}
		})
		var done aiserver.HandlerStreamEvent
		for ev, err := range h.Stream(bg, aiserver.HandlerRequest{Config: c, UserInput: "q", ToolHandlers: map[string]aiserver.ToolHandler{
			"a": tool(func(a map[string]any) (any, error) { order = append(order, "tool"); gotArgs = a; return "sunny", nil }),
		}}) {
			if err != nil {
				t.Fatal(err)
			}
			if ev.Type == "chunk" {
				order = append(order, ev.Text)
			} else {
				done = ev
			}
		}
		if !reflect.DeepEqual(order, []string{"Let me ", "check. ", "tool", "Sunny."}) || gotArgs["city"] != "Paris" {
			t.Errorf("order = %v args = %v", order, gotArgs)
		}
		if *done.Output != "Let me check. Sunny." || aiserver.ParseUsage(done.Usage).Input != 5 {
			t.Errorf("done = %+v", done)
		}
		if f.requests[0]["stream"] != true || f.requests[0]["tools"] == nil {
			t.Errorf("first streamed call: %v", f.requests[0])
		}
		if _, has := f.requests[0]["text"]; has {
			t.Error("streaming must ignore outputFormat")
		}
		if _, has := f.requests[1]["tools"]; !has || f.requests[1]["previous_response_id"] != "r1" {
			t.Errorf("follow-up streamed call: %v", f.requests[1])
		}
		if n := len(spansByName(exp)["chat test-model"]); n != 2 {
			t.Errorf("chat spans = %d", n)
		}
	})
	t.Run("provider error", func(t *testing.T) {
		h, _, exp := setup(t, "STATUS400")
		var gotErr error
		n := 0
		for _, err := range h.Stream(bg, aiserver.HandlerRequest{Config: cfg(instructions("x"))}) {
			n++
			gotErr = err
		}
		var apiErr *openai.Error
		if n != 1 || !errors.As(gotErr, &apiErr) {
			t.Fatalf("want one provider error, got %d items, %v", n, gotErr)
		}
		spans := exp.GetSpans().Snapshots()
		if len(spans) != 2 {
			t.Fatalf("spans = %d", len(spans))
		}
		for _, s := range spans {
			if s.Status().Code != codes.Error || !slices.Contains(eventNames(s), "exception") || s.EndTime().IsZero() {
				t.Errorf("%s: status %v events %v", s.Name(), s.Status().Code, eventNames(s))
			}
		}
	})
	t.Run("response.failed event", func(t *testing.T) {
		h, _, _ := setup(t, "event: response.failed\ndata: {\"type\":\"response.failed\",\"response\":{\"id\":\"r\",\"status\":\"failed\",\"error\":{\"code\":\"server_error\",\"message\":\"exploded\"}}}\n\n")
		for _, err := range h.Stream(bg, aiserver.HandlerRequest{Config: cfg(instructions("x"))}) {
			if err == nil || !strings.Contains(err.Error(), "exploded") {
				t.Errorf("err = %v", err)
			}
		}
	})
	t.Run("response.incomplete ends the turn as the final response", func(t *testing.T) {
		final := strings.Replace(response("r1", textOutput("ab"), 1, 0, 1), `"status":"completed","model"`,
			`"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"model"`, 1)
		s := strings.Replace(sse([]string{"a", "b"}, final), "response.completed", "response.incomplete", 2)
		h, _, exp := setup(t, s)
		var done *aiserver.HandlerStreamEvent
		for ev, err := range h.Stream(bg, aiserver.HandlerRequest{Config: cfg(instructions("x"))}) {
			if err != nil {
				t.Fatal(err)
			}
			if ev.Type == aiserver.EventDone {
				done = &ev
			}
		}
		if done == nil || *done.Output != "ab" {
			t.Fatalf("done = %+v", done)
		}
		chat := spansByName(exp)["chat test-model"]
		if len(chat) != 1 || attrs(chat[0])["gen_ai.response.finish_reasons"] != `["length"]` {
			t.Errorf("chat span attrs = %v", attrs(chat[0]))
		}
	})
	t.Run("error event and a stream without a final response fail", func(t *testing.T) {
		for name, body := range map[string]string{
			"error event": "event: error\ndata: {\"type\":\"error\",\"code\":\"rate_limit\",\"message\":\"slow down\",\"sequence_number\":1}\n\n",
			"no final":    strings.SplitAfter(sse([]string{"a"}, textResp("a", 1, 1)), "\n\n")[0],
		} {
			h, _, exp := setup(t, body)
			var gotErr error
			for _, err := range h.Stream(bg, aiserver.HandlerRequest{Config: cfg(instructions("x"))}) {
				if err != nil {
					gotErr = err
				}
			}
			if gotErr == nil || (name == "no final") != errors.Is(gotErr, openaiwire.ErrNoFinalResponse) ||
				(name == "error event" && !strings.Contains(gotErr.Error(), "slow down")) {
				t.Errorf("%s: err = %v", name, gotErr)
			}
			if root := spansByName(exp)[SpanName]; len(root) != 1 || root[0].Status().Code != codes.Error {
				t.Errorf("%s: root span not ERROR", name)
			}
		}
	})
	t.Run("abandoned", func(t *testing.T) {
		h, _, exp := setup(t, sse([]string{"a", "b"}, response("r1", textOutput("ab"), 1, 0, 1)))
		for range h.Stream(bg, aiserver.HandlerRequest{Config: cfg(instructions("x"))}) {
			break
		}
		spans := exp.GetSpans().Snapshots()
		if len(spans) != 2 {
			t.Fatalf("spans = %d", len(spans))
		}
		for _, s := range spans {
			if attrs(s)["launchdarkly.stream.abandoned"] != "true" || s.Status().Code != codes.Unset {
				t.Errorf("%s: attrs %v status %v", s.Name(), attrs(s), s.Status().Code)
			}
		}
	})
}

func TestStructuredOutput(t *testing.T) {
	schema := map[string]any{"type": "object", "properties": map[string]any{"name": map[string]any{"type": "string"}}}
	h, f, _ := setup(t, textResp(`{"name":"x"}`, 1, 1), textResp("plain", 1, 1))
	res, err := h.Invoke(bg, aiserver.HandlerRequest{Config: cfg(func(c *aiserver.AiConfig) { c.Instructions = "sys"; c.OutputFormat = schema })})
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]any{"format": map[string]any{"type": "json_schema", "name": "output", "schema": schema, "strict": false}}
	if !reflect.DeepEqual(f.requests[0]["text"], want) {
		t.Errorf("text = %v", f.requests[0]["text"])
	}
	if got := inputTurns(f.requests[0]); got[0].content != "sys" {
		t.Errorf("schema must not be injected into the system prompt: %v", got)
	}
	if res.Output != `{"name":"x"}` {
		t.Errorf("output = %v", res.Output)
	}
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: cfg(instructions("sys"))}); err != nil {
		t.Fatal(err)
	}
	if _, has := f.requests[1]["text"]; has {
		t.Error("text must be absent without outputFormat")
	}
}

// History, including image blocks.
func TestHistory(t *testing.T) {
	img := aiserver.Message{Role: "user", Blocks: []aiserver.ContentBlock{
		{Type: "text", Text: "what is this"},
		{Type: "image", Source: &aiserver.ImageSource{Type: "base64", MediaType: "image/png", Data: "AAA"}},
		{Type: "image", Source: &aiserver.ImageSource{Type: "url", URL: "https://x/y.png"}},
	}}
	imgJSON := `[{"text":"what is this","type":"input_text"},{"detail":"auto","image_url":"data:image/png;base64,AAA","type":"input_image"},{"detail":"auto","image_url":"https://x/y.png","type":"input_image"}]`
	msgs := cfg(func(c *aiserver.AiConfig) {
		c.Messages = []aiserver.Message{{Role: "system", Text: "sys"}, {Role: "user", Text: "cfg-u"}, {Role: "assistant", Text: "cfg-a"}}
	})
	hist := []aiserver.Message{{Role: "user", Text: "h-u"}, {Role: "system", Text: "drop me"}, {Role: "assistant", Text: "h-a"}}
	for _, tc := range []struct {
		name      string
		config    aiserver.AiConfig
		history   []aiserver.Message
		userInput string
		want      []turn
	}{
		{"between config messages and userInput, system filtered", msgs, hist, "q",
			[]turn{{"system", "sys"}, {"user", "cfg-u"}, {"assistant", "cfg-a"}, {"user", "h-u"}, {"assistant", "h-a"}, {"user", "q"}}},
		{"instructions path", cfg(instructions("sys")), hist, "q",
			[]turn{{"system", "sys"}, {"user", "h-u"}, {"assistant", "h-a"}, {"user", "q"}}},
		{"empty history equals none", cfg(instructions("sys")), []aiserver.Message{}, "q",
			[]turn{{"system", "sys"}, {"user", "q"}}},
		{"multimodal mapped to input_image", cfg(instructions("sys")), []aiserver.Message{img}, "",
			[]turn{{"system", "sys"}, {"user", imgJSON}}},
		{"non-empty userInput after image-only history", cfg(instructions("sys")), []aiserver.Message{img}, "and this?",
			[]turn{{"system", "sys"}, {"user", imgJSON}, {"user", "and this?"}}},
		{"assistant blocks flatten to text", cfg(instructions("sys")),
			[]aiserver.Message{{Role: "assistant", Blocks: []aiserver.ContentBlock{{Type: "text", Text: "a"}, {Type: "text", Text: "b"}}}}, "q",
			[]turn{{"system", "sys"}, {"assistant", "ab"}, {"user", "q"}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h, f, _ := setup(t, textResp("ok", 1, 1))
			if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: tc.config, UserInput: tc.userInput, History: tc.history}); err != nil {
				t.Fatal(err)
			}
			if got := inputTurns(f.requests[0]); !reflect.DeepEqual(got, tc.want) {
				t.Errorf("input =\n %v\nwant\n %v", got, tc.want)
			}
		})
	}
}

// Spec-silent; follows the JavaScript handler's model.parameters allowlist.
func TestModelParameters(t *testing.T) {
	for _, tc := range []struct {
		name   string
		params map[string]any
		want   map[string]any
		absent []string
	}{
		{"allowlisted forwarded, unknown and unsafe dropped",
			map[string]any{"temperature": 0.2, "store": false, "frequency_penalty": 1.0, "background": true, "stream": true, "prompt": "p", "base_url": "http://evil"},
			map[string]any{"temperature": 0.2, "store": false},
			[]string{"frequency_penalty", "background", "prompt", "base_url"}},
		{"max_tokens renamed", map[string]any{"max_tokens": 100.0}, map[string]any{"max_output_tokens": 100.0}, []string{"max_tokens"}},
		{"max_output_tokens wins", map[string]any{"max_tokens": 1.0, "max_completion_tokens": 2.0, "max_output_tokens": 3.0},
			map[string]any{"max_output_tokens": 3.0}, []string{"max_tokens", "max_completion_tokens"}},
		{"max_completion_tokens beats max_tokens", map[string]any{"max_tokens": 1.0, "max_completion_tokens": 2.0},
			map[string]any{"max_output_tokens": 2.0}, nil},
		{"cannot override handler-owned fields", map[string]any{"model": "other", "input": "x", "tools": []any{}, "previous_response_id": "p"},
			map[string]any{"model": "test-model"}, []string{"tools", "previous_response_id"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h, f, _ := setup(t, textResp("ok", 1, 1))
			c := cfg(func(c *aiserver.AiConfig) { c.Instructions = "x"; c.Model.Parameters = tc.params })
			if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: c}); err != nil {
				t.Fatal(err)
			}
			req := f.requests[0]
			for k, v := range tc.want {
				if !reflect.DeepEqual(req[k], v) {
					t.Errorf("%s = %v, want %v", k, req[k], v)
				}
			}
			for _, k := range tc.absent {
				if _, has := req[k]; has {
					t.Errorf("%s must not be forwarded", k)
				}
			}
			if _, isList := req["input"].([]any); !isList {
				t.Errorf("input overridden: %v", req["input"])
			}
		})
	}
}

// installClient makes fake the LD client with a fresh reported-usage set and
// routes spans to exp.
func installClient(t *testing.T, fake *fakeLD, exp *tracetest.InMemoryExporter) {
	t.Helper()
	_ = aiserver.Shutdown(bg)
	if _, err := aiserver.InitWithClient(bg, fake, aiserver.InitOptions{OTLPEndpoint: "http://127.0.0.1:1/v1/traces"}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = aiserver.Shutdown(bg) })
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(sdktrace.NewTracerProvider(sdktrace.WithSyncer(exp))) // InitWithClient installed its own
	t.Cleanup(func() { otel.SetTracerProvider(prev) })
}

var variation = map[string]any{
	"model": map[string]any{"name": "test-model"}, "provider": map[string]any{"name": "OpenAI"},
	"messages": []any{map[string]any{"role": "user", "content": "say {{user_input}}"}},
	"_ldMeta":  map[string]any{"enabled": true, "variationKey": "v"},
}

// the convenience export routes through config().invoke with
// this package's handler, forwards variables, and reports only itself.
func TestConvenienceExport(t *testing.T) {
	f, client := fakeClient(t, textResp("one", 1, 1), textResp("two", 1, 1))
	ld := &fakeLD{variation: variation}
	installClient(t, ld, tracetest.NewInMemoryExporter())

	res, err := OpenAIMessages(bg, "cfg", "hi", ldcontext.New("u-1"), &InvokeOptions{
		Variables: map[string]any{"user_input": "hello"}, Handler: Options{Client: client},
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.Response != "one" || res.TrackData.ConfigKey != "cfg" || res.TrackData.ProviderName != "OpenAI" {
		t.Errorf("response = %+v", res)
	}
	if got := inputTurns(f.requests[0]); !reflect.DeepEqual(got, []turn{{"user", "say hello"}}) {
		t.Errorf("variables not forwarded to invoke: %v", got)
	}
	if got := ld.usageHelpers(); !reflect.DeepEqual(got, []string{"openai-messages.openaiMessages"}) {
		t.Errorf("usage helpers = %v", got)
	}

	// Nil options: variables absent, so the placeholder stays unresolved.
	if _, err := OpenAIMessages(bg, "cfg", "hi", ldcontext.New("u-1"), &InvokeOptions{Handler: Options{Client: client}}); err != nil {
		t.Fatal(err)
	}
	if got := inputTurns(f.requests[1]); got[0].content != "say {{user_input}}" {
		t.Errorf("unexpected variables: %v", got)
	}
	if n := len(ld.usageHelpers()); n != 1 {
		t.Errorf("repeat call must not report again: %d events", n)
	}
}

func TestConvenienceExportNilOptions(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "test")
	t.Setenv("OPENAI_BASE_URL", "http://127.0.0.1:1") // unreachable: no network
	installClient(t, &fakeLD{variation: variation}, tracetest.NewInMemoryExporter())
	_, err := OpenAIMessages(bg, "cfg", "hi", ldcontext.New("u-1"), nil)
	var apiErr *openai.Error
	if err == nil || errors.As(err, &apiErr) {
		t.Errorf("want a connection error from the default client, got %v", err)
	}
}

func TestSdkUsage(t *testing.T) {
	ld := &fakeLD{}
	installClient(t, ld, tracetest.NewInMemoryExporter())
	NewHandler(Options{})
	NewHandler(Options{})
	if got := ld.usageHelpers(); !reflect.DeepEqual(got, []string{"openai-messages.createOpenAIMessagesHandler"}) {
		t.Fatalf("usage helpers = %v (factory must not emit client.createHandler, repeat must not re-emit)", got)
	}
	for _, tr := range ld.tracks {
		if tr.event != "$ld:ai:sdk:usage" {
			continue
		}
		if tr.data["helperPackageName"] != PackageName || tr.data["helperPackageVersion"] != PackageVersion ||
			tr.data["aiSdkName"] != aiserver.PackageName || tr.data["aiSdkLanguage"] != "go" || tr.value != 1 ||
			tr.ctx.Kind() != "ld_ai" || !tr.ctx.Anonymous() {
			t.Errorf("usage event = %+v", tr)
		}
	}
	var info bool
	for _, tr := range ld.tracks {
		info = info || (tr.event == "$ld:ai:sdk:info" && tr.data["aiSdkName"] == PackageName && tr.data["aiSdkVersion"] == PackageVersion)
	}
	if !info {
		t.Error("init() must register the package for $ld:ai:sdk:info")
	}
}

// TestThroughConfig drives the handler via aiserver.Config with a tool turn:
// ldContext reaches the root span as context identity, and no chat or
// execute_tool span carries identity or a feature_flag event.
func TestThroughConfig(t *testing.T) {
	_, client := fakeClient(t,
		sse(nil, response("r1", callOutput("a", `{}`), 1, 0, 1)),
		sse([]string{"o", "k"}, response("r2", textOutput("ok"), 1, 0, 1)))
	exp := tracetest.NewInMemoryExporter()
	toolVariation := maps.Clone(variation)
	toolVariation["tools"] = map[string]any{"a": map[string]any{"name": "a", "type": "function", "parameters": map[string]any{"type": "object"}}}
	installClient(t, &fakeLD{variation: toolVariation}, exp)
	h := newHandler(Options{Client: client})

	ran := false
	multi := ldcontext.NewMulti(ldcontext.New("u-1"), ldcontext.NewWithKind("org", "o-1"))
	for _, err := range aiserver.Config(aiserver.ConfigArgs{Key: "cfg", Handlers: []*aiserver.ProviderHandler{h},
		ToolHandlers: map[string]aiserver.ToolHandler{"a": tool(func(map[string]any) (any, error) { ran = true; return "r", nil })},
	}).Stream(bg, "q", multi) {
		if err != nil {
			t.Fatal(err)
		}
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
