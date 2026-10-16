package llms

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
)

const anthropicReply = `{"id":"m1","type":"message","role":"assistant","model":"claude-sonnet-4-6",
"content":[{"type":"text","text":"Let me look."},{"type":"tool_use","id":"tu_1","name":"lookup","input":{"z":1,"a":"x"}}],
"stop_reason":"tool_use","usage":{"input_tokens":10,"output_tokens":5,"cache_read_input_tokens":2,"cache_creation_input_tokens":1}}`

func newTestAnthropic(t *testing.T, url string) *Anthropic {
	t.Helper()
	m, err := NewAnthropic(AnthropicOptions{APIKey: "k", BaseURL: url})
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func TestAnthropicRequest(t *testing.T) {
	f, url := newFake(t, "application/json", anthropicReply)
	msgs := []MessageContent{
		TextParts(ChatMessageTypeSystem, "be ", "brief"),
		{Role: ChatMessageTypeHuman, Parts: []ContentPart{TextContent{Text: "look"}, BinaryContent{MIMEType: "image/png", Data: []byte{1, 2}},
			ImageURLContent{URL: "data:image/jpeg;base64,AAA="}, ImageURLContent{URL: "https://example.com/a.png"}}},
		{Role: ChatMessageTypeAI, Parts: []ContentPart{ToolCall{ID: "tu_1", FunctionCall: &FunctionCall{Name: "lookup", Arguments: ""}}}},
		{Role: ChatMessageTypeTool, Parts: []ContentPart{ToolCallResponse{ToolCallID: "tu_1", Content: "found"}}},
		TextParts(ChatMessageTypeHuman, "thanks"),
	}
	_, err := newTestAnthropic(t, url).GenerateContent(bg, msgs, WithTemperature(0), WithTopK(5), WithStopWords([]string{"END"}),
		WithTools([]Tool{{Type: "function", Function: &FunctionDefinition{Name: "lookup", Description: "d",
			Parameters: map[string]any{"type": "object", "properties": map[string]any{"q": map[string]any{"type": "string"}}, "required": []any{"q"}}}}}))
	if err != nil {
		t.Fatal(err)
	}
	if f.paths[0] != "/v1/messages" || f.headers[0].Get("X-Api-Key") != "k" {
		t.Errorf("path %q key %q", f.paths[0], f.headers[0].Get("X-Api-Key"))
	}
	b := f.bodies[0]
	// Defaults, explicit zero and top_k; unset top_p is not sent.
	if b["model"] != DefaultAnthropicModel || b["max_tokens"] != 2048.0 || b["temperature"] != 0.0 || b["top_k"] != 5.0 {
		t.Errorf("model %v max_tokens %v temperature %v top_k %v", b["model"], b["max_tokens"], b["temperature"], b["top_k"])
	}
	if _, ok := b["top_p"]; ok {
		t.Error("unset top_p was sent")
	}
	// System parts are concatenated with no separator.
	if sys, _ := json.Marshal(b["system"]); !strings.Contains(string(sys), `"text":"be brief"`) {
		t.Errorf("system = %s", sys)
	}
	got := b["messages"].([]any)
	if len(got) != 3 {
		t.Fatalf("messages = %d, want 3 (the tool result and the next human turn share one user message)", len(got))
	}
	user := got[0].(map[string]any)["content"].([]any)
	if len(user) != 4 || user[1].(map[string]any)["source"].(map[string]any)["data"] != "AQI=" ||
		user[2].(map[string]any)["source"].(map[string]any)["media_type"] != "image/jpeg" ||
		user[3].(map[string]any)["source"].(map[string]any)["url"] != "https://example.com/a.png" {
		ub, _ := json.Marshal(user)
		t.Errorf("user blocks = %s", ub)
	}
	tu := got[1].(map[string]any)["content"].([]any)[0].(map[string]any)
	if got[1].(map[string]any)["role"] != "assistant" || tu["type"] != "tool_use" || !reflect.DeepEqual(tu["input"], map[string]any{}) {
		t.Errorf("tool_use block = %v (empty arguments must be {})", tu)
	}
	tr := got[2].(map[string]any)
	trBlocks := tr["content"].([]any)
	if tr["role"] != "user" || len(trBlocks) != 2 || trBlocks[0].(map[string]any)["tool_use_id"] != "tu_1" ||
		trBlocks[1].(map[string]any)["text"] != "thanks" {
		t.Errorf("tool result message = %v (tool_result first, then the human text)", tr)
	}
	tool := b["tools"].([]any)[0].(map[string]any)
	if tool["name"] != "lookup" || tool["description"] != "d" || !reflect.DeepEqual(tool["input_schema"].(map[string]any)["required"], []any{"q"}) {
		t.Errorf("tool = %v", tool)
	}
	if !reflect.DeepEqual(b["stop_sequences"], []any{"END"}) {
		t.Errorf("stop_sequences = %v", b["stop_sequences"])
	}
}

// Parallel tool calls: the handlers append one tool message per call, and
// Anthropic requires every tool_use id to be answered in the single user
// message that follows the assistant turn.
func TestAnthropicMergesParallelToolResults(t *testing.T) {
	f, url := newFake(t, "application/json", anthropicReply)
	msgs := []MessageContent{
		TextParts(ChatMessageTypeHuman, "weather in Paris and Rome?"),
		{Role: ChatMessageTypeAI, Parts: []ContentPart{
			ToolCall{ID: "tu_1", FunctionCall: &FunctionCall{Name: "w", Arguments: `{"c":"Paris"}`}},
			ToolCall{ID: "tu_2", FunctionCall: &FunctionCall{Name: "w", Arguments: `{"c":"Rome"}`}},
		}},
		{Role: ChatMessageTypeTool, Parts: []ContentPart{ToolCallResponse{ToolCallID: "tu_1", Content: "sunny"}}},
		{Role: ChatMessageTypeTool, Parts: []ContentPart{ToolCallResponse{ToolCallID: "tu_2", Content: "rain"}}},
	}
	if _, err := newTestAnthropic(t, url).GenerateContent(bg, msgs); err != nil {
		t.Fatal(err)
	}
	got := f.bodies[0]["messages"].([]any)
	if len(got) != 3 {
		mb, _ := json.Marshal(got)
		t.Fatalf("messages = %s, want 3: both tool results in one user message", mb)
	}
	last := got[2].(map[string]any)
	blocks := last["content"].([]any)
	if last["role"] != "user" || len(blocks) != 2 ||
		blocks[0].(map[string]any)["tool_use_id"] != "tu_1" || blocks[1].(map[string]any)["tool_use_id"] != "tu_2" {
		t.Errorf("tool results message = %v", last)
	}
}

func TestAnthropicResponse(t *testing.T) {
	_, url := newFake(t, "application/json", anthropicReply)
	resp, err := newTestAnthropic(t, url).GenerateContent(bg, []MessageContent{TextParts(ChatMessageTypeHuman, "q")})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Choices) != 2 {
		t.Fatalf("choices = %d, want one per content block", len(resp.Choices))
	}
	text, tool := resp.Choices[0], resp.Choices[1]
	if text.Content != "Let me look." || text.StopReason != "tool_use" {
		t.Errorf("text choice = %+v", text)
	}
	// Arguments are re-marshalled: compact, keys sorted.
	if len(tool.ToolCalls) != 1 || tool.ToolCalls[0].FunctionCall.Arguments != `{"a":"x","z":1}` || tool.ToolCalls[0].Type != "" {
		t.Errorf("tool choice = %+v", tool.ToolCalls)
	}
	for _, c := range resp.Choices {
		if c.GenerationInfo["InputTokens"] != 10 || c.GenerationInfo["OutputTokens"] != 5 ||
			c.GenerationInfo["CacheReadInputTokens"] != 2 || c.GenerationInfo["CacheCreationInputTokens"] != 1 {
			t.Errorf("usage = %v", c.GenerationInfo)
		}
	}
	am := resp.AssistantMessage()
	if len(am.Parts) != 2 {
		t.Errorf("assistant message = %+v", am)
	}
}

func TestAnthropicEmptyContent(t *testing.T) {
	_, url := newFake(t, "application/json", `{"id":"m","type":"message","role":"assistant","content":[],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":0}}`)
	if _, err := newTestAnthropic(t, url).GenerateContent(bg, []MessageContent{TextParts(ChatMessageTypeHuman, "q")}); !errors.Is(err, ErrEmptyResponse) {
		t.Fatalf("err = %v, want ErrEmptyResponse", err)
	}
}

func anthropicSSE(events ...string) string {
	var b strings.Builder
	for _, e := range events {
		var head struct{ Type string }
		_ = json.Unmarshal([]byte(e), &head)
		b.WriteString("event: " + head.Type + "\ndata: " + e + "\n\n")
	}
	return b.String()
}

func TestAnthropicStreaming(t *testing.T) {
	body := anthropicSSE(
		`{"type":"message_start","message":{"id":"m","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-6","usage":{"input_tokens":9,"output_tokens":1}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"tu_1","name":"lookup","input":{}}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"q\":"}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"\"x\"}"}}`,
		`{"type":"content_block_stop","index":1}`,
		`{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":6}}`,
		`{"type":"message_stop"}`,
	)
	f, url := newFake(t, "text/event-stream", body)
	var chunks []string
	resp, err := newTestAnthropic(t, url).GenerateContent(bg, []MessageContent{TextParts(ChatMessageTypeHuman, "q")},
		WithStreamingFunc(func(_ context.Context, c []byte) error { chunks = append(chunks, string(c)); return nil }))
	if err != nil {
		t.Fatal(err)
	}
	if f.bodies[0]["stream"] != true {
		t.Errorf("stream = %v", f.bodies[0]["stream"])
	}
	if !reflect.DeepEqual(chunks, []string{"Hel", "lo"}) {
		t.Errorf("chunks = %q", chunks)
	}
	if len(resp.Choices) != 2 || resp.Choices[0].Content != "Hello" || resp.Choices[1].ToolCalls[0].FunctionCall.Arguments != `{"q":"x"}` {
		rb, _ := json.Marshal(resp)
		t.Errorf("response = %s", rb)
	}
	if resp.Choices[0].StopReason != "tool_use" || resp.Choices[0].GenerationInfo["InputTokens"] != 9 || resp.Choices[0].GenerationInfo["OutputTokens"] != 6 {
		t.Errorf("stop %q usage %v", resp.Choices[0].StopReason, resp.Choices[0].GenerationInfo)
	}
}

func TestAnthropicRejectsUnsupportedParts(t *testing.T) {
	_, url := newFake(t, "application/json", anthropicReply)
	m := newTestAnthropic(t, url)
	if _, err := m.GenerateContent(bg, []MessageContent{{Role: ChatMessageTypeSystem, Parts: []ContentPart{BinaryContent{}}}}); err == nil {
		t.Error("a non-text system part must fail")
	}
	if _, err := m.GenerateContent(bg, []MessageContent{{Role: ChatMessageTypeHuman}}); err == nil {
		t.Error("an empty message must fail")
	}
}
