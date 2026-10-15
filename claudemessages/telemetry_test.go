package claudemessages

import (
	"context"
	"errors"
	"testing"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

func attrsOf(s sdktrace.ReadOnlySpan) map[string]string {
	out := map[string]string{}
	for _, a := range s.Attributes() {
		out[string(a.Key)] = a.Value.Emit()
	}
	return out
}

func spansNamed(exp *tracetest.InMemoryExporter, name string) []sdktrace.ReadOnlySpan {
	var out []sdktrace.ReadOnlySpan
	for _, s := range exp.GetSpans().Snapshots() {
		if s.Name() == name {
			out = append(out, s)
		}
	}
	return out
}

func wantAttrs(t *testing.T, label string, got map[string]string, want map[string]string) {
	t.Helper()
	for k, v := range want {
		if got[k] != v {
			t.Errorf("%s %s = %q, want %q", label, k, got[k], v)
		}
	}
}

var toolA = func(c *aiserver.AiConfig) {
	c.Instructions = "x"
	c.Tools = map[string]aiserver.Tool{"a": {Name: "a", Type: "function", Parameters: map[string]any{"type": "object"}}}
}

var okA = map[string]aiserver.ToolHandler{"a": aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) { return "r", nil })}

// chat spans carry finish reasons and cache-inclusive input; the root
// carries the run's cache buckets and gen_ai.provider.name.
func TestChatSpanTelemetry(t *testing.T) {
	cached := `{"id":"m","type":"message","role":"assistant","model":"test-model","stop_reason":"end_turn",
"content":[{"type":"text","text":"done"}],"usage":{"input_tokens":10,"output_tokens":5,"cache_read_input_tokens":3,"cache_creation_input_tokens":2}}`
	h, _, exp := setup(t, toolResp("a", `{}`), cached)
	res, err := h.Invoke(bg, aiserver.HandlerRequest{Config: cfg(toolA), UserInput: "q", ToolHandlers: okA})
	if err != nil {
		t.Fatal(err)
	}
	if u := aiserver.ParseUsage(res.Usage); u.Input != 16 || u.Output != 6 || u.InputDetails == nil || u.InputDetails.CacheRead != 3 || u.InputDetails.CacheCreation != 2 {
		t.Errorf("usage = %+v", u)
	}
	chats := spansNamed(exp, "chat test-model")
	if len(chats) != 2 {
		t.Fatalf("chat spans = %d", len(chats))
	}
	wantAttrs(t, "tool turn", attrsOf(chats[0]), map[string]string{
		"gen_ai.operation.name": "chat", "gen_ai.system": "anthropic", "gen_ai.provider.name": "anthropic",
		"gen_ai.request.model": "test-model", "gen_ai.response.model": "test-model",
		"gen_ai.response.finish_reasons": `["tool_calls"]`, "gen_ai.usage.input_tokens": "1",
	})
	wantAttrs(t, "final turn", attrsOf(chats[1]), map[string]string{
		"gen_ai.response.finish_reasons": `["stop"]`, "gen_ai.usage.input_tokens": "15", "gen_ai.usage.output_tokens": "5",
		"gen_ai.usage.cache_read.input_tokens": "3", "gen_ai.usage.cache_creation.input_tokens": "2",
	})
	wantAttrs(t, "root", attrsOf(spansNamed(exp, SpanName)[0]), map[string]string{
		"gen_ai.provider.name": "anthropic", "gen_ai.usage.input_tokens": "16", "gen_ai.usage.total_tokens": "22",
		"gen_ai.usage.cache_read.input_tokens": "3", "gen_ai.usage.cache_creation.input_tokens": "2",
	})
	tool := spansNamed(exp, "execute_tool a")
	if len(tool) != 1 || attrsOf(tool[0])["gen_ai.tool.call.id"] != "tu_a" || tool[0].Status().Code != codes.Ok {
		t.Errorf("tool span = %v", tool)
	}
}

// a zero usage report is still written, not dropped.
func TestZeroUsage(t *testing.T) {
	h, _, exp := setup(t, textResp("ok", 0, 0))
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: cfg(nil), UserInput: "q"}); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{SpanName, "chat test-model"} {
		wantAttrs(t, name, attrsOf(spansNamed(exp, name)[0]), map[string]string{
			"gen_ai.usage.input_tokens": "0", "gen_ai.usage.output_tokens": "0", "gen_ai.usage.total_tokens": "0",
			"gen_ai.usage.prompt_tokens": "0", "gen_ai.usage.completion_tokens": "0",
		})
	}
}

// OTel: the streaming root carries the blocking path's attributes and
// content, ends OK, and its chat span reports the turn.
func TestStreamTelemetry(t *testing.T) {
	h, _, exp := setup(t, sse([]string{"Hel", "lo"}, "", 3, 2))
	for _, err := range h.Stream(bg, aiserver.HandlerRequest{Config: cfg(func(c *aiserver.AiConfig) { c.Instructions = "sys" }), UserInput: "q",
		Variables: map[string]any{"__ld": aiserver.TrackData{ConfigKey: "test-config", VariationKey: "variation-a", RunID: "run-123"}}}) {
		if err != nil {
			t.Fatal(err)
		}
	}
	roots := spansNamed(exp, SpanName)
	if len(roots) != 1 {
		t.Fatalf("roots = %d", len(roots))
	}
	root := roots[0]
	wantAttrs(t, "stream root", attrsOf(root), map[string]string{
		"gen_ai.system": "anthropic", "gen_ai.operation.name": "chat", "gen_ai.request.model": "test-model",
		"gen_ai.response.model": "test-model", "gen_ai.usage.input_tokens": "3", "gen_ai.usage.output_tokens": "2",
		"gen_ai.usage.total_tokens": "5", "gen_ai.usage.prompt_tokens": "3", "gen_ai.usage.completion_tokens": "2",
		"gen_ai.prompt.0.role": "system", "gen_ai.prompt.0.content": "sys", "gen_ai.prompt.1.role": "user", "gen_ai.prompt.1.content": "q",
		"gen_ai.completion.0.role": "assistant", "gen_ai.completion.0.content": "Hello",
		"launchdarkly.operation.type": "gen_ai", "launchdarkly.config.key": "test-config",
		"launchdarkly.variation.key": "variation-a", "launchdarkly.run.id": "run-123",
	})
	if root.Status().Code != codes.Ok || root.EndTime().IsZero() || !hasEvent(root, "gen_ai.content.prompt") || !hasEvent(root, "gen_ai.content.completion") {
		t.Errorf("stream root status %v events %v", root.Status(), root.Events())
	}
	chat := spansNamed(exp, "chat test-model")
	if len(chat) != 1 || attrsOf(chat[0])["gen_ai.response.finish_reasons"] != `["stop"]` || attrsOf(chat[0])["gen_ai.usage.input_tokens"] != "3" {
		t.Errorf("stream chat span = %v", chat)
	}
}

// a tool the model names but was not offered fails its own
// execute_tool span as well as the root, as in JavaScript.
func TestUnofferedToolSpan(t *testing.T) {
	h, _, exp := setup(t, toolResp("ghost", `{}`))
	_, err := h.Invoke(bg, aiserver.HandlerRequest{Config: cfg(toolA), UserInput: "q", ToolHandlers: okA})
	if !errors.Is(err, ErrToolNotFound) {
		t.Fatalf("err = %v", err)
	}
	ghost := spansNamed(exp, "execute_tool ghost")
	if len(ghost) != 1 || ghost[0].Status().Code != codes.Error || !hasEvent(ghost[0], "exception") || ghost[0].EndTime().IsZero() {
		t.Errorf("ghost tool span = %v", ghost)
	}
}

// every handler span inside a bound scope carries the caller's
// conversation id; none is invented without one.
func TestConversationID(t *testing.T) {
	h, _, exp := setup(t, toolResp("a", `{}`), textResp("ok", 1, 1), textResp("ok", 1, 1))
	otel.SetTracerProvider(sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(aiserver.ConversationIDSpanProcessor{}), sdktrace.WithSyncer(exp)))
	if _, err := h.Invoke(aiserver.WithConversationID(bg, "conv-1"), aiserver.HandlerRequest{Config: cfg(toolA), UserInput: "q", ToolHandlers: okA}); err != nil {
		t.Fatal(err)
	}
	spans := exp.GetSpans().Snapshots()
	if len(spans) != 4 {
		t.Fatalf("spans = %d", len(spans))
	}
	for _, s := range spans {
		if attrsOf(s)["gen_ai.conversation.id"] != "conv-1" {
			t.Errorf("%s lacks the conversation id", s.Name())
		}
	}
	exp.Reset()
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: cfg(nil), UserInput: "q"}); err != nil {
		t.Fatal(err)
	}
	if n := len(exp.GetSpans().Snapshots()); n != 2 {
		t.Fatalf("unbound spans = %d", n)
	}
	for _, s := range exp.GetSpans().Snapshots() {
		if _, has := attrsOf(s)["gen_ai.conversation.id"]; has {
			t.Errorf("%s has an invented conversation id", s.Name())
		}
	}
}

// A panicking tool ends every open span ERROR (none is left unended) and
// still panics; the chat turn that finished before it stays OK.
func TestPanicEndsSpans(t *testing.T) {
	h, _, exp := setup(t, toolResp("a", `{}`))
	c := cfg(func(c *aiserver.AiConfig) {
		c.Instructions = "x"
		c.Tools = map[string]aiserver.Tool{"a": {Name: "a", Type: "function", Parameters: map[string]any{}}}
	})
	func() {
		defer func() {
			if recover() == nil {
				t.Error("panic did not propagate")
			}
		}()
		_, _ = h.Invoke(bg, aiserver.HandlerRequest{Config: c, ToolHandlers: map[string]aiserver.ToolHandler{
			"a": aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) { panic("boom") })}})
	}()
	got := map[string]bool{}
	for _, s := range exp.GetSpans().Snapshots() {
		if s.Name() != "chat test-model" && s.Status().Code != codes.Error {
			t.Errorf("%s status = %v", s.Name(), s.Status())
		}
		got[s.Name()] = true
	}
	for _, want := range []string{SpanName, "execute_tool a"} {
		if !got[want] {
			t.Errorf("span %q not ended (ended: %v)", want, got)
		}
	}
}
