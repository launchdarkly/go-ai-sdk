package langchainmessages

import (
	"context"
	"errors"
	"testing"

	"github.com/tmc/langchaingo/llms"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

func okA() map[string]aiserver.ToolHandler {
	return map[string]aiserver.ToolHandler{"a": aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) { return "r", nil })}
}

// chat spans carry semconv finish reasons (a refusal included) and
// cache-inclusive input; a zero usage report is still written.
func TestChatSpanFinishReasonsAndUsage(t *testing.T) {
	refusal := scripted{resp: &llms.ContentResponse{Choices: []*llms.ContentChoice{{
		Content: "no", StopReason: "refusal",
		GenerationInfo: map[string]any{"InputTokens": 10, "OutputTokens": 0, "CacheReadInputTokens": 3, "CacheCreationInputTokens": 2},
	}}}}
	h, _, exp := setup(t, toolTurn("a", nil), refusal)
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: cfg(threeTools), UserInput: "q", ToolHandlers: okA()}); err != nil {
		t.Fatal(err)
	}
	chats := spansByName(exp)["chat test-model"]
	if len(chats) != 2 {
		t.Fatalf("chat spans = %d", len(chats))
	}
	if fr := spanAttrs(chats[0])["gen_ai.response.finish_reasons"]; fr != `["tool_calls"]` {
		t.Errorf("tool turn finish reason = %s", fr)
	}
	last := spanAttrs(chats[1])
	for k, want := range map[string]string{
		"gen_ai.response.finish_reasons": `["content_filter"]`, "gen_ai.usage.input_tokens": "15", "gen_ai.usage.output_tokens": "0",
		"gen_ai.usage.completion_tokens": "0", "gen_ai.usage.cache_read.input_tokens": "3", "gen_ai.usage.cache_creation.input_tokens": "2",
		"gen_ai.system": "anthropic", "gen_ai.provider.name": "anthropic",
	} {
		if last[k] != want {
			t.Errorf("final chat %s = %q, want %q", k, last[k], want)
		}
	}
	if root := spanAttrs(spansByName(exp)[SpanName][0]); root["gen_ai.usage.input_tokens"] != "16" || root["gen_ai.usage.cache_read.input_tokens"] != "3" {
		t.Errorf("root usage = %v", root)
	}
}

// a tool the model names without a handler fails its own
// execute_tool span as well as the root, as in JavaScript.
func TestUnknownToolSpan(t *testing.T) {
	h, _, exp := setup(t, toolTurn("zzz", nil))
	_, err := h.Invoke(bg, aiserver.HandlerRequest{Config: cfg(threeTools), UserInput: "q", ToolHandlers: okA()})
	if !errors.Is(err, ErrNoToolHandler) {
		t.Fatalf("err = %v", err)
	}
	s := spansByName(exp)["execute_tool zzz"]
	if len(s) != 1 || s[0].Status().Code != codes.Error || s[0].EndTime().IsZero() || len(s[0].Events()) == 0 {
		t.Errorf("tool span = %v", s)
	}
	if root := spansByName(exp)[SpanName]; len(root) != 1 || root[0].Status().Code != codes.Error {
		t.Errorf("root = %v", root)
	}
}

// every handler span inside a bound scope carries the caller's
// conversation id, blocking and streaming; none is invented without one.
func TestConversationID(t *testing.T) {
	h, _, exp := setup(t, toolTurn("a", nil), textTurn("ok", 1, 1), streamTurn([]string{"o", "k"}, 1, 1), textTurn("ok", 1, 1))
	otel.SetTracerProvider(sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(aiserver.ConversationIDSpanProcessor{}), sdktrace.WithSyncer(exp)))
	ctx := aiserver.WithConversationID(bg, "conv-1")
	if _, err := h.Invoke(ctx, aiserver.HandlerRequest{Config: cfg(threeTools), UserInput: "q", ToolHandlers: okA()}); err != nil {
		t.Fatal(err)
	}
	for _, err := range h.Stream(ctx, aiserver.HandlerRequest{Config: instr("x"), UserInput: "q"}) {
		if err != nil {
			t.Fatal(err)
		}
	}
	spans := exp.GetSpans().Snapshots()
	if len(spans) != 6 {
		t.Fatalf("spans = %d", len(spans))
	}
	for _, s := range spans {
		if spanAttrs(s)["gen_ai.conversation.id"] != "conv-1" {
			t.Errorf("%s lacks the conversation id", s.Name())
		}
	}
	exp.Reset()
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: instr("x"), UserInput: "q"}); err != nil {
		t.Fatal(err)
	}
	if n := len(exp.GetSpans().Snapshots()); n != 2 {
		t.Fatalf("unbound spans = %d", n)
	}
	for _, s := range exp.GetSpans().Snapshots() {
		if _, has := spanAttrs(s)["gen_ai.conversation.id"]; has {
			t.Errorf("%s has an invented conversation id", s.Name())
		}
	}
}
