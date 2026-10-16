package llms

import (
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

const bedrockReply = `{"output":{"message":{"role":"assistant","content":[{"text":"Let me look."},
{"toolUse":{"toolUseId":"tu_1","name":"lookup","input":{"z":1,"a":"x"}}}]}},
"stopReason":"tool_use","usage":{"inputTokens":10,"outputTokens":5,"totalTokens":15},"metrics":{"latencyMs":1}}`

func newTestBedrock(t *testing.T, url, model string) *Bedrock {
	t.Helper()
	for k, v := range map[string]string{
		"AWS_ENDPOINT_URL_BEDROCK_RUNTIME": url, "AWS_REGION": "us-east-1", "AWS_MAX_ATTEMPTS": "1",
		"AWS_ACCESS_KEY_ID": "test", "AWS_SECRET_ACCESS_KEY": "test", "AWS_SESSION_TOKEN": "",
		"AWS_BEARER_TOKEN_BEDROCK": "", "AWS_PROFILE": "", "AWS_CONFIG_FILE": "/nonexistent", "AWS_SHARED_CREDENTIALS_FILE": "/nonexistent",
	} {
		t.Setenv(k, v)
	}
	m, err := NewBedrock(BedrockOptions{Model: model})
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func TestBedrockRequest(t *testing.T) {
	f, url := newFake(t, "application/json", bedrockReply)
	msgs := []MessageContent{
		TextParts(ChatMessageTypeSystem, "be brief"),
		TextParts(ChatMessageTypeSystem, "   "), // blank system text is dropped
		{Role: ChatMessageTypeHuman, Parts: []ContentPart{TextContent{Text: "look"}, TextContent{Text: " "}, BinaryContent{MIMEType: "image/png", Data: []byte{1, 2}}}},
		{Role: ChatMessageTypeAI, Parts: []ContentPart{ToolCall{ID: "tu_1", FunctionCall: &FunctionCall{Name: "lookup", Arguments: "not json"}}}},
		{Role: ChatMessageTypeTool, Parts: []ContentPart{ToolCallResponse{ToolCallID: "tu_1", Content: "found"}}},
		TextParts(ChatMessageTypeHuman, "thanks"), // merged into the tool message's user turn
	}
	_, err := newTestBedrock(t, url, "us.anthropic.claude-sonnet-4-5").GenerateContent(bg, msgs, WithTemperature(0), WithTopK(3),
		WithStopWords([]string{"END"}),
		WithTools([]Tool{{Type: "function", Function: &FunctionDefinition{Name: "lookup", Description: "d", Parameters: map[string]any{"type": "object"}}}}))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(f.paths[0], "/model/us.anthropic.claude-sonnet-4-5/converse") {
		t.Errorf("path = %q", f.paths[0])
	}
	b := f.bodies[0]
	if !reflect.DeepEqual(b["system"], []any{map[string]any{"text": "be brief"}}) {
		t.Errorf("system = %v", b["system"])
	}
	inf := b["inferenceConfig"].(map[string]any)
	if inf["temperature"] != 0.0 || inf["maxTokens"] != 2048.0 || !reflect.DeepEqual(inf["stopSequences"], []any{"END"}) {
		t.Errorf("inferenceConfig = %v", inf)
	}
	if _, ok := inf["topP"]; ok {
		t.Error("unset topP was sent")
	}
	if !reflect.DeepEqual(b["additionalModelRequestFields"], map[string]any{"top_k": 3.0}) {
		t.Errorf("additionalModelRequestFields = %v", b["additionalModelRequestFields"])
	}
	got := b["messages"].([]any)
	if len(got) != 3 {
		mb, _ := json.Marshal(got)
		t.Fatalf("messages = %s, want 3 (same-role turns merged)", mb)
	}
	user := got[0].(map[string]any)["content"].([]any)
	if len(user) != 2 || user[0].(map[string]any)["text"] != "look" || user[1].(map[string]any)["image"].(map[string]any)["format"] != "png" {
		t.Errorf("user blocks = %v", user)
	}
	tu := got[1].(map[string]any)["content"].([]any)[0].(map[string]any)["toolUse"].(map[string]any)
	if !reflect.DeepEqual(tu["input"], map[string]any{"arguments": "not json"}) {
		t.Errorf("toolUse input = %v (invalid JSON is kept under arguments)", tu["input"])
	}
	last := got[2].(map[string]any)["content"].([]any)
	if got[2].(map[string]any)["role"] != "user" || len(last) != 2 || last[0].(map[string]any)["toolResult"] == nil || last[1].(map[string]any)["text"] != "thanks" {
		t.Errorf("merged user turn = %v", last)
	}
	spec := b["toolConfig"].(map[string]any)["tools"].([]any)[0].(map[string]any)["toolSpec"].(map[string]any)
	if spec["name"] != "lookup" || spec["description"] != "d" {
		t.Errorf("toolSpec = %v", spec)
	}
}

func TestBedrockTopKOnlyForAnthropic(t *testing.T) {
	f, url := newFake(t, "application/json", bedrockReply)
	if _, err := newTestBedrock(t, url, "us.amazon.nova-pro-v1:0").GenerateContent(bg, []MessageContent{TextParts(ChatMessageTypeHuman, "q")}, WithTopK(3)); err != nil {
		t.Fatal(err)
	}
	if _, ok := f.bodies[0]["additionalModelRequestFields"]; ok {
		t.Error("top_k sent to a non-Anthropic model")
	}
}

func TestBedrockResponse(t *testing.T) {
	_, url := newFake(t, "application/json", bedrockReply)
	resp, err := newTestBedrock(t, url, "anthropic.claude").GenerateContent(bg, []MessageContent{TextParts(ChatMessageTypeHuman, "q")})
	if err != nil {
		t.Fatal(err)
	}
	c := resp.Choices[0]
	if len(resp.Choices) != 1 || c.Content != "Let me look." || c.StopReason != "tool_use" {
		t.Errorf("choice = %+v", c)
	}
	if len(c.ToolCalls) != 1 || c.ToolCalls[0].Type != "function" || c.ToolCalls[0].FunctionCall.Arguments != `{"a":"x","z":1}` {
		t.Errorf("tool calls = %+v", c.ToolCalls)
	}
	if c.GenerationInfo["input_tokens"] != 10 || c.GenerationInfo["output_tokens"] != 5 {
		t.Errorf("usage = %v", c.GenerationInfo)
	}
}

func TestBedrockEmptyAndMaxTokens(t *testing.T) {
	_, url := newFake(t, "application/json",
		`{"output":{"message":{"role":"assistant","content":[]}},"stopReason":"end_turn","usage":{"inputTokens":1,"outputTokens":0,"totalTokens":1}}`,
		`{"output":{"message":{"role":"assistant","content":[{"text":"trunc"}]}},"stopReason":"max_tokens","usage":{"inputTokens":1,"outputTokens":9,"totalTokens":10}}`)
	m := newTestBedrock(t, url, "anthropic.claude")
	if _, err := m.GenerateContent(bg, []MessageContent{TextParts(ChatMessageTypeHuman, "q")}); !errors.Is(err, ErrEmptyResponse) {
		t.Fatalf("err = %v, want ErrEmptyResponse", err)
	}
	// A max_tokens stop returns the truncated answer rather than an error.
	resp, err := m.GenerateContent(bg, []MessageContent{TextParts(ChatMessageTypeHuman, "q")})
	if err != nil || resp.Choices[0].Content != "trunc" || resp.Choices[0].StopReason != "max_tokens" {
		t.Fatalf("resp %+v err %v", resp, err)
	}
}

func TestBedrockImages(t *testing.T) {
	_, url := newFake(t, "application/json", bedrockReply)
	m := newTestBedrock(t, url, "anthropic.claude")
	if _, err := m.GenerateContent(bg, []MessageContent{{Role: ChatMessageTypeHuman, Parts: []ContentPart{ImageURLContent{URL: "https://example.com/a.png"}}}}); err == nil {
		t.Error("an http image URL must fail on Converse")
	}
	if _, err := m.GenerateContent(bg, []MessageContent{{Role: ChatMessageTypeHuman, Parts: []ContentPart{BinaryContent{MIMEType: "image/tiff", Data: []byte{1}}}}}); err == nil {
		t.Error("an unsupported image type must fail")
	}
	if _, err := m.GenerateContent(bg, []MessageContent{{Role: ChatMessageTypeHuman, Parts: []ContentPart{ImageURLContent{URL: "data:image/png;base64,AQI="}}}}); err != nil {
		t.Errorf("data: URL image: %v", err)
	}
}

func TestBedrockStreamAccumulator(t *testing.T) {
	acc := bedrockStream{blocks: map[int32]*bedrockBlock{}}
	events := []types.ConverseStreamOutput{
		&types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{ContentBlockIndex: aws.Int32(0), Delta: &types.ContentBlockDeltaMemberText{Value: "Hel"}}},
		&types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{ContentBlockIndex: aws.Int32(0), Delta: &types.ContentBlockDeltaMemberText{Value: "lo"}}},
		&types.ConverseStreamOutputMemberContentBlockStart{Value: types.ContentBlockStartEvent{ContentBlockIndex: aws.Int32(1),
			Start: &types.ContentBlockStartMemberToolUse{Value: types.ToolUseBlockStart{ToolUseId: aws.String("tu_1"), Name: aws.String("lookup")}}}},
		&types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{ContentBlockIndex: aws.Int32(1), Delta: &types.ContentBlockDeltaMemberToolUse{Value: types.ToolUseBlockDelta{Input: aws.String(`{"q":`)}}}},
		&types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{ContentBlockIndex: aws.Int32(1), Delta: &types.ContentBlockDeltaMemberToolUse{Value: types.ToolUseBlockDelta{Input: aws.String(`"x"}`)}}}},
		&types.ConverseStreamOutputMemberContentBlockStop{Value: types.ContentBlockStopEvent{ContentBlockIndex: aws.Int32(1)}},
		&types.ConverseStreamOutputMemberMessageStop{Value: types.MessageStopEvent{StopReason: types.StopReasonToolUse}},
		&types.ConverseStreamOutputMemberMetadata{Value: types.ConverseStreamMetadataEvent{Usage: &types.TokenUsage{InputTokens: aws.Int32(4), OutputTokens: aws.Int32(2)}}},
	}
	var text []string
	for _, ev := range events {
		d, err := acc.add(ev)
		if err != nil {
			t.Fatal(err)
		}
		if d != "" {
			text = append(text, d)
		}
	}
	if !reflect.DeepEqual(text, []string{"Hel", "lo"}) {
		t.Errorf("text deltas = %q", text)
	}
	resp, err := bedrockResponse(acc.message(), acc.stop, acc.usage)
	if err != nil {
		t.Fatal(err)
	}
	c := resp.Choices[0]
	// Streaming keeps tool calls (langchaingo's Bedrock stream dropped them).
	if c.Content != "Hello" || c.StopReason != "tool_use" || len(c.ToolCalls) != 1 || c.ToolCalls[0].FunctionCall.Arguments != `{"q":"x"}` {
		t.Errorf("choice = %+v tool calls %+v", c, c.ToolCalls)
	}
	if c.GenerationInfo["input_tokens"] != 4 || c.GenerationInfo["output_tokens"] != 2 {
		t.Errorf("usage = %v", c.GenerationInfo)
	}
}

func TestBedrockStreamRejectsInvalidToolJSON(t *testing.T) {
	acc := bedrockStream{blocks: map[int32]*bedrockBlock{}}
	_, _ = acc.add(&types.ConverseStreamOutputMemberContentBlockStart{Value: types.ContentBlockStartEvent{ContentBlockIndex: aws.Int32(0),
		Start: &types.ContentBlockStartMemberToolUse{Value: types.ToolUseBlockStart{ToolUseId: aws.String("t"), Name: aws.String("n")}}}})
	_, _ = acc.add(&types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{ContentBlockIndex: aws.Int32(0),
		Delta: &types.ContentBlockDeltaMemberToolUse{Value: types.ToolUseBlockDelta{Input: aws.String(`{"broken`)}}}})
	if _, err := acc.add(&types.ConverseStreamOutputMemberContentBlockStop{Value: types.ContentBlockStopEvent{ContentBlockIndex: aws.Int32(0)}}); err == nil {
		t.Fatal("invalid streamed tool JSON must fail")
	}
}
