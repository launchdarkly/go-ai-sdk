package claudemessages

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

// fakeAnthropic replays canned responses and records request bodies.
type fakeAnthropic struct {
	mu        sync.Mutex
	responses []string // JSON bodies, or SSE streams when prefixed "event:"
	requests  []map[string]any
}

func (f *fakeAnthropic) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	b, _ := io.ReadAll(r.Body)
	var body map[string]any
	_ = json.Unmarshal(b, &body)
	f.mu.Lock()
	f.requests = append(f.requests, body)
	i := len(f.requests) - 1
	f.mu.Unlock()
	if i >= len(f.responses) {
		http.Error(w, `{"type":"error","error":{"type":"api_error","message":"no more"}}`, 500)
		return
	}
	resp := f.responses[i]
	if strings.HasPrefix(resp, "event:") {
		w.Header().Set("Content-Type", "text/event-stream")
	} else {
		w.Header().Set("Content-Type", "application/json")
	}
	if strings.HasPrefix(resp, "STATUS400") {
		w.WriteHeader(400)
		resp = `{"type":"error","error":{"type":"invalid_request_error","message":"bad"}}`
	}
	_, _ = io.WriteString(w, resp)
}

func msgJSON(stop, content string, in, out int) string {
	return fmt.Sprintf(`{"id":"m","type":"message","role":"assistant","model":"test-model","stop_reason":%q,
"content":[%s],"usage":{"input_tokens":%d,"output_tokens":%d,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}`, stop, content, in, out)
}

func textResp(text string, in, out int) string {
	return msgJSON("end_turn", fmt.Sprintf(`{"type":"text","text":%q}`, text), in, out)
}

func toolUse(name, input string) string {
	return fmt.Sprintf(`{"type":"tool_use","id":"tu_%s","name":%q,"input":%s}`, name, name, input)
}

func toolResp(name string, input string) string {
	return msgJSON("tool_use", toolUse(name, input), 1, 1)
}

// sse streams text deltas; tool, when set, appends a tool_use block whose
// input arrives as an input_json_delta, and stops for tools.
func sse(deltas []string, tool string, in, out int) string {
	var b strings.Builder
	ev := func(name, data string) { fmt.Fprintf(&b, "event: %s\ndata: %s\n\n", name, data) }
	ev("message_start", fmt.Sprintf(`{"type":"message_start","message":{"id":"m","type":"message","role":"assistant","model":"test-model","content":[],"stop_reason":null,"usage":{"input_tokens":%d,"output_tokens":0}}}`, in))
	ev("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`)
	for _, d := range deltas {
		ev("content_block_delta", fmt.Sprintf(`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":%q}}`, d))
	}
	ev("content_block_stop", `{"type":"content_block_stop","index":0}`)
	stop := "end_turn"
	if tool != "" {
		stop = "tool_use"
		ev("content_block_start", fmt.Sprintf(`{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"tu_1","name":%q,"input":{}}}`, tool))
		ev("content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"city\":\"Paris\"}"}}`)
		ev("content_block_stop", `{"type":"content_block_stop","index":1}`)
	}
	ev("message_delta", fmt.Sprintf(`{"type":"message_delta","delta":{"stop_reason":%q},"usage":{"output_tokens":%d}}`, stop, out))
	ev("message_stop", `{"type":"message_stop"}`)
	return b.String()
}

func setup(t *testing.T, responses ...string) (*aiserver.ProviderHandler, *fakeAnthropic, *tracetest.InMemoryExporter) {
	t.Helper()
	opts, f, exp := setupClient(t, responses...)
	return newHandler(opts), f, exp
}

// setupClient starts the fake Anthropic server and an in-memory span exporter
// and returns handler options pointing at them.
func setupClient(t *testing.T, responses ...string) (Options, *fakeAnthropic, *tracetest.InMemoryExporter) {
	t.Helper()
	f := &fakeAnthropic{responses: responses}
	srv := httptest.NewServer(f)
	t.Cleanup(srv.Close)
	c := anthropic.NewClient(option.WithBaseURL(srv.URL), option.WithAPIKey("test"), option.WithMaxRetries(0))
	exp := tracetest.NewInMemoryExporter()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exp))
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(tp)
	t.Cleanup(func() { otel.SetTracerProvider(prev) })
	return Options{Client: &c, CaptureContent: aiserver.CaptureOn}, f, exp
}

func cfg(extra func(*aiserver.AiConfig)) aiserver.AiConfig {
	c := aiserver.AiConfig{Model: aiserver.Model{Name: "test-model"}}
	c.Provider.Name = "Anthropic"
	if extra != nil {
		extra(&c)
	}
	return c
}

var bg = context.Background()

func roles(req map[string]any) []string {
	var out []string
	for _, m := range req["messages"].([]any) {
		out = append(out, m.(map[string]any)["role"].(string))
	}
	return out
}

func system(req map[string]any) string {
	s, _ := req["system"].([]any)
	if len(s) == 0 {
		return ""
	}
	return s[0].(map[string]any)["text"].(string)
}

func lastText(req map[string]any) string {
	msgs := req["messages"].([]any)
	content := msgs[len(msgs)-1].(map[string]any)["content"].([]any)
	return content[0].(map[string]any)["text"].(string)
}

func TestFactoryMetadata(t *testing.T) {
	a, b := newHandler(Options{}), newHandler(Options{})
	if a.ProvidesFor != (aiserver.ProvidesFor{Provider: "Anthropic", Mode: aiserver.ModeMessages}) || a.CaptureContent != aiserver.CaptureDefault || a == b || a.Stream == nil {
		t.Errorf("factory metadata wrong: %+v", a)
	}
	if newHandler(Options{CaptureContent: aiserver.CaptureOn}).CaptureContent != aiserver.CaptureOn {
		t.Error("captureContent not passed through")
	}
}

func TestPromptPaths(t *testing.T) {
	h, f, _ := setup(t, textResp("a", 1, 1), textResp("b", 1, 1), textResp("c", 1, 1), textResp("d", 1, 1))
	// Instructions path.
	_, err := h.Invoke(bg, aiserver.HandlerRequest{
		Config:    cfg(func(c *aiserver.AiConfig) { c.Instructions = "Hi {{name}} {{missing}}" }),
		UserInput: "q", Variables: map[string]any{"name": "Al"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if system(f.requests[0]) != "Hi Al {{missing}}" || lastText(f.requests[0]) != "q" {
		t.Errorf("instructions path: %v", f.requests[0])
	}
	// Instructions and messages both present: messages win, last config msg is user and input empty → no append.
	both := cfg(func(c *aiserver.AiConfig) {
		c.Instructions = "IGNORED"
		c.Messages = []aiserver.Message{{Role: "system", Text: "sys {{x}}"}, {Role: "user", Text: "u {{x}}"}}
	})
	_, _ = h.Invoke(bg, aiserver.HandlerRequest{Config: both, Variables: map[string]any{"x": "1"}})
	if system(f.requests[1]) != "sys 1" || !reflect.DeepEqual(roles(f.requests[1]), []string{"user"}) || lastText(f.requests[1]) != "u 1" {
		t.Errorf("messages path: %v", f.requests[1])
	}
	// Last config message assistant → userInput appended; history sits between.
	asst := cfg(func(c *aiserver.AiConfig) { c.Messages = []aiserver.Message{{Role: "assistant", Text: "hello"}} })
	hist := []aiserver.Message{{Role: "system", Text: "drop"}, {Role: "user", Blocks: []aiserver.ContentBlock{
		{Type: "image", Source: &aiserver.ImageSource{Type: "base64", MediaType: "image/png", Data: "AAA"}}}}}
	_, _ = h.Invoke(bg, aiserver.HandlerRequest{Config: asst, UserInput: "", History: hist})
	if got := roles(f.requests[2]); !reflect.DeepEqual(got, []string{"assistant", "user"}) {
		t.Errorf("history ending in user with empty input must not append: %v", got)
	}
	img := f.requests[2]["messages"].([]any)[1].(map[string]any)["content"].([]any)[0].(map[string]any)
	if img["type"] != "image" || img["source"].(map[string]any)["data"] != "AAA" {
		t.Errorf("image not mapped natively: %v", img)
	}
	// outputFormat → schema appended to system on the blocking path.
	of := cfg(func(c *aiserver.AiConfig) {
		c.Instructions = "base"
		c.OutputFormat = map[string]any{"type": "object"}
	})
	_, _ = h.Invoke(bg, aiserver.HandlerRequest{Config: of, UserInput: "q"})
	if s := system(f.requests[3]); !strings.HasPrefix(s, "base\n\n") || !strings.Contains(s, `{"type":"object"}`) {
		t.Errorf("outputFormat system = %q", s)
	}
}

func TestToolFilteringAndLoop(t *testing.T) {
	h, f, exp := setup(t, toolResp("a", `{"city":"Paris"}`), toolResp("b", `{}`), textResp("final", 2, 3))
	params := map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}, "required": []any{"city"}, "additionalProperties": false}
	c := cfg(func(c *aiserver.AiConfig) {
		c.Instructions = "x"
		c.Tools = map[string]aiserver.Tool{
			"a": {Name: "a", Type: "function", Description: "tool a", Parameters: params},
			"b": {Name: "b", Type: "function", Parameters: map[string]any{"type": "object"}},
			"c": {Name: "c", Type: "function", Parameters: map[string]any{"type": "object"}},
		}
	})
	var calls []string
	var gotArgs map[string]any
	tools := map[string]aiserver.ToolHandler{
		"a": aiserver.ToolFunc(func(_ context.Context, args map[string]any) (any, error) {
			calls = append(calls, "a")
			gotArgs = args
			return "sunny", nil
		}),
		"b": aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) {
			calls = append(calls, "b")
			return map[string]any{"ok": true}, nil
		}),
	}
	res, err := h.Invoke(bg, aiserver.HandlerRequest{Config: c, UserInput: "q", ToolHandlers: tools,
		Variables: map[string]any{"__ld": aiserver.TrackData{ConfigKey: "test-config", VariationKey: "variation-a", RunID: "run-123"}}})
	if err != nil {
		t.Fatal(err)
	}
	sent := f.requests[0]["tools"].([]any)
	if len(sent) != 2 {
		t.Fatalf("want 2 tools sent, got %d", len(sent))
	}
	for _, s := range sent {
		tool := s.(map[string]any)
		if tool["name"] == "a" && (!reflect.DeepEqual(tool["input_schema"], params) || tool["description"] != "tool a") {
			t.Errorf("schema modified: %v", tool["input_schema"])
		}
	}
	if !reflect.DeepEqual(calls, []string{"a", "b"}) || gotArgs["city"] != "Paris" || res.Output != "final" {
		t.Errorf("calls=%v args=%v out=%v", calls, gotArgs, res.Output)
	}
	if u := aiserver.ParseUsage(res.Usage); u.Input != 4 || u.Output != 5 {
		t.Errorf("usage = %+v", u)
	}

	names := map[string]int{}
	var root sdktrace.ReadOnlySpan
	for _, s := range exp.GetSpans().Snapshots() {
		names[s.Name()]++
		if s.Name() == "claude.messages" {
			root = s
		}
		for _, a := range s.Attributes() {
			if s.Name() != "claude.messages" && strings.HasPrefix(string(a.Key), "launchdarkly.config") {
				t.Errorf("child %s carries %s", s.Name(), a.Key)
			}
		}
	}
	if names["claude.messages"] != 1 || names["chat test-model"] != 3 || names["execute_tool a"] != 1 || names["execute_tool b"] != 1 {
		t.Errorf("span tree = %v", names)
	}
	attrs := map[string]string{}
	for _, a := range root.Attributes() {
		attrs[string(a.Key)] = a.Value.Emit()
	}
	for k, want := range map[string]string{
		"gen_ai.operation.name": "chat", "gen_ai.system": "anthropic", "gen_ai.request.model": "test-model",
		"gen_ai.response.model":     "test-model",
		"gen_ai.usage.input_tokens": "4", "gen_ai.usage.prompt_tokens": "4", "gen_ai.usage.total_tokens": "9",
		"gen_ai.usage.output_tokens": "5", "gen_ai.usage.completion_tokens": "5",
		"launchdarkly.operation.type": "gen_ai", "launchdarkly.config.key": "test-config", "launchdarkly.run.id": "run-123",
		"launchdarkly.variation.key": "variation-a",
		"gen_ai.prompt.0.role":       "system", "gen_ai.prompt.0.content": "x", "gen_ai.prompt.1.role": "user", "gen_ai.prompt.1.content": "q",
		"gen_ai.completion.0.role": "assistant", "gen_ai.completion.0.content": "final",
	} {
		if attrs[k] != want {
			t.Errorf("root %s = %q, want %q", k, attrs[k], want)
		}
	}
	if _, has := attrs["launchdarkly.graph.key"]; has || root.Status().Code != codes.Ok {
		t.Error("graph key set or status not OK")
	}
	events := map[string]string{}
	for _, ev := range root.Events() {
		for _, a := range ev.Attributes {
			events[ev.Name] += a.Value.Emit()
		}
	}
	if !strings.Contains(events["gen_ai.content.prompt"], "q") || events["gen_ai.content.completion"] != "final" {
		t.Errorf("content events = %v", events)
	}
}

// launchdarkly.graph.key is set when __ld carries a graph key.
func TestGraphKey(t *testing.T) {
	h, _, exp := setup(t, textResp("ok", 1, 1))
	td := aiserver.TrackData{ConfigKey: "test-config", VariationKey: "variation-a", RunID: "run-123", GraphKey: "g1"}
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: cfg(nil), UserInput: "q", Variables: map[string]any{"__ld": td}}); err != nil {
		t.Fatal(err)
	}
	for _, s := range exp.GetSpans().Snapshots() {
		if s.Name() != SpanName {
			continue
		}
		for _, a := range s.Attributes() {
			if a.Key == "launchdarkly.graph.key" && a.Value.AsString() == "g1" {
				return
			}
		}
	}
	t.Error("launchdarkly.graph.key not set on the root span")
}

// the blocking output is the last response's text, not every turn's.
func TestFinalTurnOutput(t *testing.T) {
	h, _, exp := setup(t, msgJSON("tool_use", `{"type":"text","text":"thinking "},`+toolUse("a", `{}`), 1, 1), textResp("final", 1, 1))
	c := cfg(func(c *aiserver.AiConfig) {
		c.Tools = map[string]aiserver.Tool{"a": {Name: "a", Type: "function", Parameters: map[string]any{"type": "object"}}}
	})
	res, err := h.Invoke(bg, aiserver.HandlerRequest{Config: c, UserInput: "q", ToolHandlers: map[string]aiserver.ToolHandler{
		"a": aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) { return "ok", nil })}})
	if err != nil {
		t.Fatal(err)
	}
	if res.Output != "final" {
		t.Errorf("output = %q, want %q", res.Output, "final")
	}
	for _, s := range exp.GetSpans().Snapshots() {
		for _, a := range s.Attributes() {
			if s.Name() == SpanName && a.Key == "gen_ai.completion.0.content" && a.Value.AsString() != "final" {
				t.Errorf("completion content = %q", a.Value.AsString())
			}
		}
	}
}

// Tool filtering edge cases and no tools in config.
func TestNoToolsAndExcluded(t *testing.T) {
	h, f, _ := setup(t, textResp("x", 1, 1), textResp("y", 1, 1))
	called := false
	fn := aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) { called = true; return nil, nil })
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: cfg(nil), UserInput: "q", ToolHandlers: map[string]aiserver.ToolHandler{"a": fn}}); err != nil {
		t.Fatal(err)
	}
	only := cfg(func(c *aiserver.AiConfig) {
		c.Tools = map[string]aiserver.Tool{"ghost": {Name: "ghost", Type: "function", Parameters: map[string]any{}}}
	})
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: only, UserInput: "q"}); err != nil {
		t.Fatal(err)
	}
	if len(f.requests) != 2 || f.requests[0]["tools"] != nil || f.requests[1]["tools"] != nil || called {
		t.Errorf("calls=%d tools=%v / %v called=%v", len(f.requests), f.requests[0]["tools"], f.requests[1]["tools"], called)
	}
}

func TestErrors(t *testing.T) {
	h, _, exp := setup(t, "STATUS400")
	_, err := h.Invoke(bg, aiserver.HandlerRequest{Config: cfg(func(c *aiserver.AiConfig) { c.Instructions = "x" })})
	var apiErr *anthropic.Error
	if !errors.As(err, &apiErr) {
		t.Fatalf("want provider error, got %v", err)
	}
	for _, s := range exp.GetSpans().Snapshots() {
		if s.Status().Code != codes.Error || s.EndTime().IsZero() || !hasEvent(s, "exception") {
			t.Errorf("%s: status %v events %v", s.Name(), s.Status().Code, s.Events())
		}
	}

	h, _, _ = setup(t, toolResp("a", `{}`))
	boom := errors.New("tool boom")
	c := cfg(func(c *aiserver.AiConfig) {
		c.Instructions = "x"
		c.Tools = map[string]aiserver.Tool{"a": {Name: "a", Type: "function", Parameters: map[string]any{}}}
	})
	_, err = h.Invoke(bg, aiserver.HandlerRequest{Config: c, ToolHandlers: map[string]aiserver.ToolHandler{
		"a": aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) { return nil, boom })}})
	if !errors.Is(err, boom) {
		t.Errorf("tool error not propagated: %v", err)
	}

	// The model calls a tool it was never offered: the run fails naming it.
	h, _, exp = setup(t, toolResp("ghost", `{}`))
	_, err = h.Invoke(bg, aiserver.HandlerRequest{Config: c, ToolHandlers: map[string]aiserver.ToolHandler{
		"a": aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) { return "ok", nil })}})
	if !errors.Is(err, ErrToolNotFound) || !strings.Contains(err.Error(), "ghost") {
		t.Errorf("tool-not-found error = %v", err)
	}
	for _, s := range exp.GetSpans().Snapshots() {
		if s.Name() == SpanName && s.Status().Code != codes.Error {
			t.Errorf("root status = %v", s.Status().Code)
		}
	}
}

func hasEvent(s sdktrace.ReadOnlySpan, name string) bool {
	for _, e := range s.Events() {
		if e.Name == name {
			return true
		}
	}
	return false
}

func TestStream(t *testing.T) {
	h, _, exp := setup(t, sse([]string{"Hel", "lo"}, "", 3, 2), sse([]string{"never"}, "", 1, 1))
	var events []aiserver.HandlerStreamEvent
	for ev, err := range h.Stream(bg, aiserver.HandlerRequest{Config: cfg(func(c *aiserver.AiConfig) { c.Instructions = "x" }), UserInput: "q"}) {
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

	// Abandonment: break on the first chunk → root marked abandoned, status unset.
	exp.Reset()
	for range h.Stream(bg, aiserver.HandlerRequest{Config: cfg(func(c *aiserver.AiConfig) { c.Instructions = "x" })}) {
		break
	}
	for _, s := range exp.GetSpans().Snapshots() {
		abandoned := false
		for _, a := range s.Attributes() {
			abandoned = abandoned || (a.Key == "launchdarkly.stream.abandoned" && a.Value.AsBool())
		}
		if !abandoned || s.Status().Code != codes.Unset {
			t.Errorf("%s: abandoned=%v status=%v", s.Name(), abandoned, s.Status().Code)
		}
	}
}

// tool loop with streaming, and a provider error mid-stream.
func TestStreamToolLoop(t *testing.T) {
	h, _, exp := setup(t, sse([]string{"Hel", "lo "}, "a", 2, 1), sse([]string{"wor", "ld"}, "", 3, 2), "STATUS400")
	var gotArgs map[string]any
	c := cfg(func(c *aiserver.AiConfig) {
		c.Tools = map[string]aiserver.Tool{"a": {Name: "a", Type: "function", Parameters: map[string]any{"type": "object"}}}
	})
	tools := map[string]aiserver.ToolHandler{"a": aiserver.ToolFunc(func(_ context.Context, args map[string]any) (any, error) {
		gotArgs = args
		return "ok", nil
	})}
	var chunks []string
	var done *aiserver.HandlerStreamEvent
	for ev, err := range h.Stream(bg, aiserver.HandlerRequest{Config: c, UserInput: "q", ToolHandlers: tools}) {
		if err != nil {
			t.Fatal(err)
		}
		if done != nil {
			t.Fatal("event after done")
		}
		switch ev.Type {
		case aiserver.EventChunk:
			chunks = append(chunks, ev.Text)
		case aiserver.EventDone:
			done = &ev
		}
	}
	if !reflect.DeepEqual(chunks, []string{"Hel", "lo ", "wor", "ld"}) || done == nil || *done.Output != "Hello world" || gotArgs["city"] != "Paris" {
		t.Fatalf("chunks=%v done=%+v args=%v", chunks, done, gotArgs)
	}

	exp.Reset()
	var gotErr error
	for _, err := range h.Stream(bg, aiserver.HandlerRequest{Config: cfg(nil), UserInput: "q"}) {
		gotErr = err
	}
	var apiErr *anthropic.Error
	if !errors.As(gotErr, &apiErr) {
		t.Errorf("stream error not propagated: %v", gotErr)
	}
	for _, s := range exp.GetSpans().Snapshots() {
		if s.Name() == SpanName && (s.Status().Code != codes.Error || s.EndTime().IsZero() || !hasEvent(s, "exception")) {
			t.Errorf("stream error root: status=%v events=%v", s.Status().Code, s.Events())
		}
	}
}

// TestThroughConfig drives the handler via aiserver.Config with a fake LD
// client: ldContext reaches the root span as context identity.
func TestThroughConfig(t *testing.T) {
	h, _, exp := setup(t, textResp("ok", 1, 1))
	withLD(t, exp, map[string]any{"cfg": variation("x")})

	multi := ldcontext.NewMulti(ldcontext.New("u-1"), ldcontext.NewWithKind("org", "o-1"))
	if _, err := aiserver.Config(aiserver.ConfigArgs{Key: "cfg", Handlers: []*aiserver.ProviderHandler{h}}).Invoke(bg, "q", multi); err != nil {
		t.Fatal(err)
	}
	for _, s := range exp.GetSpans().Snapshots() {
		var keys []string
		for _, a := range s.Attributes() {
			if strings.HasPrefix(string(a.Key), "context.contextKeys.") {
				keys = append(keys, string(a.Key))
			}
		}
		if (s.Name() == "claude.messages") != (len(keys) == 2) {
			t.Errorf("%s context keys = %v", s.Name(), keys)
		}
		if s.Name() == "claude.messages" {
			got := map[string]string{}
			for _, e := range s.Events() {
				for _, a := range e.Attributes {
					got[string(a.Key)] = a.Value.AsString()
				}
			}
			if got["feature_flag.contextKeys"] != `{"org":"o-1","user":"u-1"}` || got["feature_flag.context.id"] != multi.FullyQualifiedKey() {
				t.Errorf("feature_flag event = %v", got)
			}
		}
	}
}

// convenience export.
func TestClaudeMessagesConvenience(t *testing.T) {
	opts, f, exp := setupClient(t, textResp("hi there", 1, 1))
	withLD(t, exp, map[string]any{"cfg": variation("Answer {{user_input}}")})
	res, err := ClaudeMessages(bg, "cfg", "hi", ldcontext.New("u"), &InvokeOptions{Handler: opts, Variables: map[string]any{"user_input": "hello"}})
	if err != nil || res.Response != "hi there" || res.TrackData.ConfigKey != "cfg" {
		t.Fatalf("res=%+v err=%v", res, err)
	}
	if s := system(f.requests[0]); s != "Answer hello" {
		t.Errorf("variables not forwarded: %q", s)
	}
	// opts may be nil; an unknown key fails before any provider call.
	if _, err := ClaudeMessages(bg, "missing", "hi", ldcontext.New("u"), nil); err == nil || len(f.requests) != 1 {
		t.Errorf("nil opts: err=%v calls=%d", err, len(f.requests))
	}
}

// Usage events: each helper reports once with this package's identity.
func TestSDKUsage(t *testing.T) {
	cases := []struct {
		helper string
		call   func(Options)
	}{
		{"claude-messages.createClaudeMessagesHandler", func(Options) { NewHandler(Options{}) }},
		{"claude-messages.claudeMessages", func(o Options) {
			_, _ = ClaudeMessages(bg, "cfg", "q", ldcontext.New("u"), &InvokeOptions{Handler: o})
		}},
	}
	for _, tc := range cases {
		t.Run(tc.helper, func(t *testing.T) {
			opts, _, exp := setupClient(t, textResp("a", 1, 1), textResp("b", 1, 1))
			ld := withLD(t, exp, map[string]any{"cfg": variation("x")})
			tc.call(opts)
			tc.call(opts) // a repeat sends nothing
			if got := ld.helpers(); !reflect.DeepEqual(got, []string{tc.helper}) {
				t.Errorf("usage helpers = %v", got)
			}
			ev := ld.events("$ld:ai:sdk:usage")[0]
			if ev.Data["helperPackageName"] != PackageName || ev.Data["helperPackageVersion"] != PackageVersion ||
				ev.Data["aiSdkName"] != aiserver.PackageName || ev.Data["aiSdkVersion"] != aiserver.PackageVersion ||
				ev.Data["aiSdkLanguage"] != "go" || ev.Metric != 1 {
				t.Errorf("usage payload = %v", ev.Data)
			}
			if tc.helper == "claude-messages.claudeMessages" && len(ld.events("$ld:ai:generation:success")) == 0 {
				t.Error("wrapper did not reach config().invoke")
			}
		})
	}
}

// model.parameters reach the request the way the JavaScript handler sends
// them: the allowlisted keys verbatim (a zero temperature included), effort
// folded into output_config, unknown keys dropped, max_tokens handled apart.
func TestModelParametersForwarded(t *testing.T) {
	h, f, _ := setup(t, textResp("a", 1, 1), textResp("b", 1, 1), textResp("c", 1, 1))
	invoke := func(params map[string]any) map[string]any {
		t.Helper()
		c := cfg(func(c *aiserver.AiConfig) { c.Instructions = "sys"; c.Model.Parameters = params })
		if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: c, UserInput: "q"}); err != nil {
			t.Fatal(err)
		}
		return f.requests[len(f.requests)-1]
	}
	got := invoke(map[string]any{"temperature": 0.5, "top_k": 5.0, "top_p": 0.9, "stop_sequences": []any{"END"},
		"max_tokens": 100.0, "frequency_penalty": 1.0, "stream": true, "model": "other"})
	if got["temperature"] != 0.5 || got["top_k"] != 5.0 || got["top_p"] != 0.9 || !reflect.DeepEqual(got["stop_sequences"], []any{"END"}) {
		t.Errorf("forwarded parameters missing: %v", got)
	}
	if got["max_tokens"] != 100.0 || got["model"] != "test-model" {
		t.Errorf("handler-owned fields overridden: max_tokens %v model %v", got["max_tokens"], got["model"])
	}
	for _, k := range []string{"frequency_penalty", "stream"} {
		if _, ok := got[k]; ok {
			t.Errorf("%s forwarded", k)
		}
	}
	if zero := invoke(map[string]any{"temperature": 0.0}); zero["temperature"] != 0.0 {
		t.Errorf("a zero temperature was dropped: %v", zero["temperature"])
	}
	effort := invoke(map[string]any{"effort": "low", "output_config": map[string]any{"effort": "high", "x": 1.0}})
	if !reflect.DeepEqual(effort["output_config"], map[string]any{"effort": "high", "x": 1.0}) {
		t.Errorf("output_config = %v", effort["output_config"])
	}
	if _, ok := effort["effort"]; ok {
		t.Error("effort forwarded at the top level")
	}
}

// TestMaxSteps: a model that keeps asking for tools past the 10-turn bound
// fails the call instead of returning an empty answer (JS throws the same).
func TestMaxSteps(t *testing.T) {
	var replies []string
	for range maxSteps + 1 {
		replies = append(replies, toolResp("a", `{}`))
	}
	h, api, exp := setup(t, replies...)
	runs := 0
	c := cfg(func(c *aiserver.AiConfig) {
		c.Instructions = "x"
		c.Tools = map[string]aiserver.Tool{"a": {Name: "a", Type: "function", Parameters: map[string]any{}}}
	})
	_, err := h.Invoke(bg, aiserver.HandlerRequest{Config: c, ToolHandlers: map[string]aiserver.ToolHandler{
		"a": aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) { runs++; return "r", nil })}})
	if !errors.Is(err, ErrMaxSteps) {
		t.Fatalf("err = %v, want ErrMaxSteps", err)
	}
	// As JS: maxSteps tool turns run; the next call's tools do not.
	if n := len(api.requests); n != maxSteps+1 || runs != maxSteps {
		t.Errorf("%d calls, %d tool runs; want %d and %d", n, runs, maxSteps+1, maxSteps)
	}
	for _, s := range exp.GetSpans().Snapshots() {
		if s.Name() == SpanName && s.Status().Code != codes.Error {
			t.Errorf("root status = %v", s.Status())
		}
	}
}
