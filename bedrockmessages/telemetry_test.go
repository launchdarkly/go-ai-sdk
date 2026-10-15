package bedrockmessages

import (
	"context"
	"errors"
	"testing"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

var toolA = func(c *aiserver.AiConfig) {
	c.Instructions = "x"
	c.Tools = map[string]aiserver.Tool{"a": {Name: "a", Type: "function", Parameters: map[string]any{"type": "object"}}}
}

func okTools(names ...string) map[string]aiserver.ToolHandler {
	out := map[string]aiserver.ToolHandler{}
	for _, n := range names {
		out[n] = aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) { return "r", nil })
	}
	return out
}

// each chat span carries its turn's semconv finish reason, and a
// zero usage report is still written.
func TestChatSpanFinishReasonsAndZeroUsage(t *testing.T) {
	h, _, exp := setup(t, Options{}, toolReply("a", `{}`), textReply("done", 0, 0))
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: cfg(toolA), UserInput: "q", ToolHandlers: okTools("a")}); err != nil {
		t.Fatal(err)
	}
	var reasons []string
	for _, s := range exp.GetSpans().Snapshots() {
		if s.Name() == "chat anthropic.claude-test" {
			reasons = append(reasons, attrs(s)["gen_ai.response.finish_reasons"])
		}
	}
	if len(reasons) != 2 || reasons[0] != `["tool_calls"]` || reasons[1] != `["stop"]` {
		t.Errorf("finish reasons = %v", reasons)
	}

	h, _, exp = setup(t, Options{}, textReply("ok", 0, 0))
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: cfg(instructions("x")), UserInput: "q"}); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{SpanName, "chat anthropic.claude-test"} {
		a := attrs(spanNamed(t, exp, name))
		for _, k := range []string{"gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens", "gen_ai.usage.total_tokens",
			"gen_ai.usage.prompt_tokens", "gen_ai.usage.completion_tokens"} {
			if a[k] != "0" {
				t.Errorf("%s %s = %q, want 0", name, k, a[k])
			}
		}
	}
}

// a toolUse naming a tool the config did not offer fails, even
// when ToolHandlers has a function for it, and its execute_tool span fails
// with the root, as in JavaScript.
func TestUnofferedToolFails(t *testing.T) {
	h, _, exp := setup(t, Options{}, toolReply("extra", `{}`), textReply("never", 1, 1))
	ran := false
	tools := okTools("a")
	tools["extra"] = aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) { ran = true; return "r", nil })
	_, err := h.Invoke(bg, aiserver.HandlerRequest{Config: cfg(toolA), UserInput: "q", ToolHandlers: tools})
	if !errors.Is(err, ErrToolNotFound) || ran {
		t.Fatalf("err = %v ran = %v", err, ran)
	}
	s := spanNamed(t, exp, "execute_tool extra")
	if s.Status().Code != codes.Error || s.EndTime().IsZero() || len(s.Events()) == 0 {
		t.Errorf("tool span status %v events %v", s.Status(), s.Events())
	}
}

// every handler span inside a bound scope carries the caller's
// conversation id; none is invented without one.
func TestConversationID(t *testing.T) {
	h, _, exp := setup(t, Options{}, toolReply("a", `{}`), textReply("ok", 1, 1), textReply("ok", 1, 1))
	otel.SetTracerProvider(sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(aiserver.ConversationIDSpanProcessor{}), sdktrace.WithSyncer(exp)))
	if _, err := h.Invoke(aiserver.WithConversationID(bg, "conv-1"), aiserver.HandlerRequest{Config: cfg(toolA), UserInput: "q", ToolHandlers: okTools("a")}); err != nil {
		t.Fatal(err)
	}
	spans := exp.GetSpans().Snapshots()
	if len(spans) != 4 {
		t.Fatalf("spans = %v", spanCounts(exp))
	}
	for _, s := range spans {
		if attrs(s)["gen_ai.conversation.id"] != "conv-1" {
			t.Errorf("%s lacks the conversation id", s.Name())
		}
	}
	exp.Reset()
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: cfg(instructions("x")), UserInput: "q"}); err != nil {
		t.Fatal(err)
	}
	if n := len(exp.GetSpans().Snapshots()); n != 2 {
		t.Fatalf("unbound spans = %d", n)
	}
	for _, s := range exp.GetSpans().Snapshots() {
		if _, has := attrs(s)["gen_ai.conversation.id"]; has {
			t.Errorf("%s has an invented conversation id", s.Name())
		}
	}
}
