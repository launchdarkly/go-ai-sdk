package langchainwire

import (
	"reflect"
	"testing"

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
