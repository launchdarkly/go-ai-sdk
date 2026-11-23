package langchainwire

import (
	"reflect"
	"testing"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/llms"
)

func TestTextParts(t *testing.T) {
	got := TextParts(llms.ChatMessageTypeSystem, "be ", "brief")
	want := llms.MessageContent{Role: llms.ChatMessageTypeSystem, Parts: []llms.ContentPart{llms.TextContent{Text: "be "}, llms.TextContent{Text: "brief"}}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("TextParts = %+v", got)
	}
}

func TestAssistantMessage(t *testing.T) {
	call := llms.ToolCall{ID: "c1", FunctionCall: &llms.FunctionCall{Name: "f", Arguments: "{}"}}
	text := llms.TextContent{Text: "Let me look."}
	for name, c := range map[string]struct {
		resp *llms.ContentResponse
		want []llms.ContentPart
	}{
		"nil":   {nil, nil},
		"empty": {&llms.ContentResponse{}, nil},
		// every choice's Parts in order win over Content and ToolCalls
		"parts": {&llms.ContentResponse{Choices: []*llms.ContentChoice{
			{Content: "ignored", Parts: []llms.ContentPart{text}}, nil, {Parts: []llms.ContentPart{call}},
		}}, []llms.ContentPart{text, call}},
		// without Parts: the first choice's text, then its tool calls
		"fallback": {&llms.ContentResponse{Choices: []*llms.ContentChoice{
			{Content: "Let me look.", ToolCalls: []llms.ToolCall{call}}, {Content: "second"},
		}}, []llms.ContentPart{text, call}},
		"no text": {&llms.ContentResponse{Choices: []*llms.ContentChoice{{ToolCalls: []llms.ToolCall{call}}}}, []llms.ContentPart{call}},
	} {
		got := AssistantMessage(c.resp)
		if got.Role != llms.ChatMessageTypeAI || !reflect.DeepEqual(got.Parts, c.want) {
			t.Errorf("%s: AssistantMessage = %+v, want parts %+v", name, got, c.want)
		}
	}
}

func TestSpanProviderName(t *testing.T) {
	for configured, want := range map[string]string{
		"Anthropic": "anthropic", "OpenAI": "openai", "": "openai",
		"Bedrock": "aws.bedrock", "bedrock": "aws.bedrock",
		"Gemini": "gcp.gemini", "gemini": "gcp.gemini",
	} {
		var cfg aiserver.AiConfig
		cfg.Provider.Name = configured
		if got := SpanProviderName(cfg); got != want {
			t.Errorf("SpanProviderName(%q) = %q, want %q", configured, got, want)
		}
	}
}

// Gemini shouts its enum, so STOP must land on the same value as Anthropic's end_turn.
func TestFinishReason(t *testing.T) {
	for raw, want := range map[string]string{
		"STOP": "stop", "end_turn": "stop", "stop": "stop", "MAX_TOKENS": "length", "length": "length",
		"tool_use": "tool_calls", "tool_calls": "tool_calls", "SAFETY": "content_filter", "pause_turn": "pause_turn",
	} {
		resp := &llms.ContentResponse{Choices: []*llms.ContentChoice{{StopReason: raw}}}
		if got := FinishReason(resp); got != want {
			t.Errorf("FinishReason(%q) = %q, want %q", raw, got, want)
		}
	}
}
