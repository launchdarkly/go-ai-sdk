package langchainagents

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/llms"
)

// converseModel fails like Bedrock Converse when a request carries tool
// blocks but declares no tools.
type converseModel struct{ fakeModel }

func (m *converseModel) GenerateContent(ctx context.Context, msgs []llms.MessageContent, options ...llms.CallOption) (*llms.ContentResponse, error) {
	var o llms.CallOptions
	for _, opt := range options {
		opt(&o)
	}
	if len(o.Tools) == 0 {
		for _, msg := range msgs {
			for _, p := range msg.Parts {
				switch p.(type) {
				case llms.ToolCall, llms.ToolCallResponse:
					return nil, errors.New("ValidationException: The toolConfig field must be defined when using toolUse and toolResult content blocks.")
				}
			}
		}
	}
	return m.fakeModel.GenerateContent(ctx, msgs, options...)
}

// A hand-off target with no tools receives the router's tool call and its
// result as text, so a Bedrock target accepts the request.
func TestToLangGraphHandoffToToolLessNode(t *testing.T) {
	installClient(t, &fakeLD{variations: map[string]map[string]any{
		"g":        graphFlag("desk", map[string][]any{"desk": {edge("gardener")}}),
		"desk":     node("Desk", "desk-model"),
		"gardener": node("Garden", "gardener-model"),
	}})
	routing := streamTurn([]string{"Routing."}, 1, 1, toolCall("transfer_to_gardener", nil))
	desk := &fakeModel{turns: []scripted{routing}}
	gardener := &converseModel{fakeModel{turns: []scripted{textTurn("The roses were cut at dawn.", 1, 1)}}}
	res, err := must(ToLangGraph(resolve("g"), &NativeGraphOptions{ModelFactory: func(n *aiserver.GraphNode) (llms.Model, error) {
		if n.Key == "gardener" {
			return gardener, nil
		}
		return desk, nil
	}})).Invoke(bg, "who cut the roses?")
	if err != nil {
		t.Fatal(err)
	}
	if res.Response != "The roses were cut at dawn." {
		t.Errorf("response = %q", res.Response)
	}
	want := []turn{{sys, "Garden"}, {human, "who cut the roses?"},
		{ai, "Routing.\n[Tool call] transfer_to_gardener {}"},
		{human, "[Tool result] transfer_to_gardener: Transferred to gardener."}}
	if got := turns(gardener.call(0).msgs); !reflect.DeepEqual(got, want) {
		t.Errorf("gardener input =\n%v\nwant\n%v", got, want)
	}
}

func TestNeutralHistory(t *testing.T) {
	call := func(id, name, args string) llms.ToolCall {
		return llms.ToolCall{ID: id, Type: "function", FunctionCall: &llms.FunctionCall{Name: name, Arguments: args}}
	}
	result := func(id, name, content string) llms.MessageContent {
		return llms.MessageContent{Role: tool, Parts: []llms.ContentPart{llms.ToolCallResponse{ToolCallID: id, Name: name, Content: content}}}
	}
	img := llms.MessageContent{Role: human, Parts: []llms.ContentPart{llms.BinaryContent{MIMEType: "image/png", Data: []byte{1}}}}
	in := []llms.MessageContent{
		{Role: sys, Parts: []llms.ContentPart{llms.TextContent{Text: "s"}}},
		img,
		{Role: ai, Parts: []llms.ContentPart{llms.TextContent{Text: "a"}, llms.TextContent{Text: "b"},
			call("1", "first", ` {"q": "café", "n": {"x": [1, 2]}} `), call("2", "second", "")}},
		result("x", "orphan", "lost"), // no matching call: goes last
		result("2", "ignored-name", "two"),
		result("1", "", "one"),
		{Role: ai, Parts: []llms.ContentPart{img.Parts[0]}}, // no text left: dropped
		{Role: human, Parts: []llms.ContentPart{llms.TextContent{Text: "next"}}},
		{Role: ai, Parts: []llms.ContentPart{call("3", "raw", "not json")}},
	}
	got := neutralHistory(in)
	want := []turn{{sys, "s"}, {human, ""},
		{ai, "ab\n[Tool call] first {\"q\":\"café\",\"n\":{\"x\":[1,2]}}\n[Tool call] second {}"},
		{human, "[Tool result] first: one\n[Tool result] second: two\n[Tool result] orphan: lost"},
		{human, "next"},
		{ai, "[Tool call] raw not json"}}
	if !reflect.DeepEqual(turns(got), want) {
		t.Errorf("got\n%v\nwant\n%v", turns(got), want)
	}
	if _, ok := got[1].Parts[0].(llms.BinaryContent); !ok {
		t.Error("human image not kept")
	}
	for _, m := range got {
		for _, p := range m.Parts {
			if _, ok := p.(llms.TextContent); !ok && m.Role != human {
				t.Errorf("non-text part %T left on %s", p, m.Role)
			}
		}
	}
}
