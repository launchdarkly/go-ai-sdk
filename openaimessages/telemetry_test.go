package openaimessages

import (
	"testing"

	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

// Every handler span inside a bound scope carries the caller's
// conversation id; none is invented without one. Zero usage is
// written, not dropped.
func TestConversationIDAndZeroUsage(t *testing.T) {
	h, _, exp := setup(t, toolResp("resp_1", "a", `{}`), textResp("ok", 1, 1), textResp("ok", 0, 0))
	otel.SetTracerProvider(sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(aiserver.ConversationIDSpanProcessor{}), sdktrace.WithSyncer(exp)))
	c := cfg(func(c *aiserver.AiConfig) {
		c.Instructions = "x"
		c.Tools = map[string]aiserver.Tool{"a": funcTool("", map[string]any{"type": "object"})}
	})
	_, err := h.Invoke(aiserver.WithConversationID(bg, "conv-1"), aiserver.HandlerRequest{Config: c, UserInput: "q",
		ToolHandlers: map[string]aiserver.ToolHandler{"a": tool(func(map[string]any) (any, error) { return "r", nil })}})
	if err != nil {
		t.Fatal(err)
	}
	spans := exp.GetSpans().Snapshots()
	if len(spans) != 4 {
		t.Fatalf("spans = %d", len(spans))
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
		t.Fatalf("spans = %d", n)
	}
	for _, s := range exp.GetSpans().Snapshots() {
		a := attrs(s)
		if _, has := a["gen_ai.conversation.id"]; has {
			t.Errorf("%s has an invented conversation id", s.Name())
		}
		for _, k := range []string{"gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens", "gen_ai.usage.total_tokens",
			"gen_ai.usage.prompt_tokens", "gen_ai.usage.completion_tokens"} {
			if a[k] != "0" {
				t.Errorf("%s %s = %q, want 0", s.Name(), k, a[k])
			}
		}
	}
}
