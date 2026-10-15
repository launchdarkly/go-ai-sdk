package bedrockwire

import (
	"errors"
	"reflect"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

// Every ConverseInput field must reach ConverseStream: guards future SDK
// fields that toStreamInput copies by name.
func TestStreamInputCoversConverseInput(t *testing.T) {
	src, dst := reflect.TypeFor[bedrockruntime.ConverseInput](), reflect.TypeFor[bedrockruntime.ConverseStreamInput]()
	for f := range src.Fields() {
		if !f.IsExported() {
			continue
		}
		df, ok := dst.FieldByName(f.Name)
		if !ok || (!f.Type.AssignableTo(df.Type) && f.Name != "GuardrailConfig") {
			t.Errorf("ConverseInput.%s has no ConverseStreamInput counterpart", f.Name)
		}
	}
}

// Both spellings of each documented parameter map to inferenceConfig;
// integers are accepted.
func TestInferenceConfigSpellings(t *testing.T) {
	want := &types.InferenceConfiguration{MaxTokens: aws.Int32(7), Temperature: aws.Float32(0.5), TopP: aws.Float32(0.25), StopSequences: []string{"END"}}
	for _, params := range []map[string]any{
		{"maxTokens": 7, "temperature": 0.5, "topP": 0.25, "stopSequences": []any{"END"}},
		{"max_tokens": 7.0, "temperature": 0.5, "top_p": 0.25, "stop_sequences": []any{"END"}},
	} {
		if got := inferenceConfig(params); !reflect.DeepEqual(got, want) {
			t.Errorf("inferenceConfig(%v) = %+v", params, got)
		}
	}
	if got := inferenceConfig(map[string]any{"top_k": 5}); got != nil {
		t.Errorf("top_k must not map: %+v", got)
	}
}

// Images Converse cannot carry as bytes fail before any provider call.
func TestUnsupportedImages(t *testing.T) {
	for name, src := range map[string]*aiserver.ImageSource{
		"media type": {Type: "base64", MediaType: "image/tiff", Data: "AAAA"},
		"not base64": {Type: "base64", MediaType: "image/png", Data: "%%%"},
		"url":        {Type: "url", URL: "https://x/y.png"},
	} {
		_, err := toContent(aiserver.Message{Role: "user", Blocks: []aiserver.ContentBlock{{Type: "image", Source: src}}})
		if !errors.Is(err, ErrUnsupportedContent) {
			t.Errorf("%s: err = %v", name, err)
		}
	}
}

// A streamed tool input is parsed at contentBlockStop: invalid JSON fails
// the turn, and an empty input becomes the empty object.
func TestStreamToolInput(t *testing.T) {
	start := func(i int32, id string) types.ConverseStreamOutput {
		return &types.ConverseStreamOutputMemberContentBlockStart{Value: types.ContentBlockStartEvent{
			ContentBlockIndex: aws.Int32(i), Start: &types.ContentBlockStartMemberToolUse{Value: types.ToolUseBlockStart{ToolUseId: aws.String(id), Name: aws.String("t")}},
		}}
	}
	stop := func(i int32) types.ConverseStreamOutput {
		return &types.ConverseStreamOutputMemberContentBlockStop{Value: types.ContentBlockStopEvent{ContentBlockIndex: aws.Int32(i)}}
	}
	acc := newStreamAccumulator("p")
	for _, ev := range []types.ConverseStreamOutput{start(0, "empty"), stop(0)} {
		if _, err := acc.add(ev); err != nil {
			t.Fatal(err)
		}
	}
	use := acc.result().msg.Content[0].(*types.ContentBlockMemberToolUse).Value
	if args, err := documentArgs(use.Input); err != nil || len(args) != 0 || args == nil {
		t.Errorf("empty input = %v, %v", args, err)
	}
	if _, err := acc.add(start(1, "bad")); err != nil {
		t.Fatal(err)
	}
	if _, err := acc.add(&types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{
		ContentBlockIndex: aws.Int32(1), Delta: &types.ContentBlockDeltaMemberToolUse{Value: types.ToolUseBlockDelta{Input: aws.String(`{"a":`)}},
	}}); err != nil {
		t.Fatal(err)
	}
	if _, err := acc.add(stop(1)); err == nil {
		t.Error("invalid streamed tool JSON must fail the turn")
	}
}

// Converse rejects blank text ("" and whitespace, verified live), so blank text
// is dropped and an otherwise empty turn carries EmptyText.
func TestBlankTextBlocks(t *testing.T) {
	for _, m := range []aiserver.Message{
		{Role: "user", Text: ""},
		{Role: "user", Text: " \n"},
		{Role: "user", Blocks: []aiserver.ContentBlock{{Type: "text", Text: " "}}},
	} {
		got, err := toContent(m)
		if err != nil || len(got) != 1 || got[0].(*types.ContentBlockMemberText).Value != EmptyText {
			t.Errorf("%+v: %v %v", m, got, err)
		}
	}
	got, _ := toContent(aiserver.Message{Role: "user", Blocks: []aiserver.ContentBlock{{Type: "text", Text: ""}, {Type: "text", Text: "hi"}}})
	if len(got) != 1 || got[0].(*types.ContentBlockMemberText).Value != "hi" {
		t.Errorf("blank block kept beside real text: %v", got)
	}
}
