package bedrockwire

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"reflect"
	"slices"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

// ModelID builds the Bedrock modelId: model.region is a short
// inference-profile prefix ("us", "eu", "global"), prepended once.
func ModelID(m aiserver.Model) string {
	if m.Region == "" || strings.HasPrefix(m.Name, m.Region+".") {
		return m.Name
	}
	return m.Region + "." + m.Name
}

var imageFormats = map[string]types.ImageFormat{
	"image/png": types.ImageFormatPng, "image/jpeg": types.ImageFormatJpeg, "image/jpg": types.ImageFormatJpeg,
	"image/gif": types.ImageFormatGif, "image/webp": types.ImageFormatWebp,
}

// toContent maps a turn to Converse content blocks. Unknown block
// types are skipped, as in the other handlers; images that Converse cannot
// carry as bytes are an error rather than silently dropped or fetched.
//
// Converse rejects blank text blocks, "" and whitespace alike (verified live),
// so blank text is dropped and a turn left empty carries EmptyText: the
// provider default for an empty user input.
func toContent(m aiserver.Message) ([]types.ContentBlock, error) {
	if m.Blocks == nil {
		return []types.ContentBlock{textBlock(m.Text)}, nil
	}
	out := make([]types.ContentBlock, 0, len(m.Blocks))
	for _, b := range m.Blocks {
		switch {
		case b.Type == "text" && strings.TrimSpace(b.Text) == "":
		case b.Type == "text":
			out = append(out, &types.ContentBlockMemberText{Value: b.Text})
		case b.Type == "image" && b.Source != nil && b.Source.Type == "base64":
			format, ok := imageFormats[strings.ToLower(b.Source.MediaType)]
			if !ok {
				return nil, fmt.Errorf("%w: image media type %q", ErrUnsupportedContent, b.Source.MediaType)
			}
			data, err := base64.StdEncoding.DecodeString(b.Source.Data)
			if err != nil {
				return nil, fmt.Errorf("%w: image data is not base64: %v", ErrUnsupportedContent, err)
			}
			out = append(out, &types.ContentBlockMemberImage{Value: types.ImageBlock{
				Format: format, Source: &types.ImageSourceMemberBytes{Value: data},
			}})
		case b.Type == "image" && b.Source != nil && b.Source.Type == "url":
			return nil, fmt.Errorf("%w: URL image sources are not fetched; supply base64 data", ErrUnsupportedContent)
		}
	}
	if len(out) == 0 {
		out = append(out, textBlock(""))
	}
	return out, nil
}

// EmptyText stands in for blank text, which Converse rejects.
const EmptyText = "."

func textBlock(s string) types.ContentBlock {
	if strings.TrimSpace(s) == "" {
		s = EmptyText
	}
	return &types.ContentBlockMemberText{Value: s}
}

func toMessages(turns []aiserver.Message) ([]types.Message, error) {
	out := make([]types.Message, 0, len(turns))
	for _, t := range turns {
		content, err := toContent(t)
		if err != nil {
			return nil, err
		}
		role := types.ConversationRoleUser
		if t.Role == "assistant" {
			role = types.ConversationRoleAssistant
		}
		out = append(out, types.Message{Role: role, Content: content})
	}
	return out, nil
}

// buildTools forwards config tools that have a local ToolFunc as
// toolSpecs whose JSON Schema is sent unmodified. nil means no toolConfig.
func buildTools(cfg aiserver.AiConfig, handlers map[string]aiserver.ToolHandler) *types.ToolConfiguration {
	callable := aiserver.CallableTools(cfg, handlers)
	if len(callable) == 0 {
		return nil
	}
	keys := make([]string, 0, len(callable))
	for k := range callable {
		keys = append(keys, k)
	}
	slices.Sort(keys) // deterministic requests
	tc := &types.ToolConfiguration{}
	for _, key := range keys {
		t := callable[key]
		schema := t.Parameters
		if schema == nil {
			schema = map[string]any{} // a nil map would serialise as JSON null
		}
		spec := types.ToolSpecification{Name: aws.String(key), InputSchema: &types.ToolInputSchemaMemberJson{Value: document.NewLazyDocument(schema)}}
		if t.Description != "" {
			spec.Description = aws.String(t.Description)
		}
		tc.Tools = append(tc.Tools, &types.ToolMemberToolSpec{Value: spec})
	}
	return tc
}

func number(params map[string]any, keys ...string) (float64, bool) {
	for _, k := range keys {
		switch v := params[k].(type) {
		case float64:
			return v, true
		case int:
			return float64(v), true
		}
	}
	return 0, false
}

// inferenceConfig maps the documented common inference parameters (Converse
// InferenceConfiguration fields, camelCase or snake_case) and nothing else. nil means no inferenceConfig.
func inferenceConfig(params map[string]any) *types.InferenceConfiguration {
	var ic types.InferenceConfiguration
	set := false
	if v, ok := number(params, "maxTokens", "max_tokens"); ok {
		ic.MaxTokens, set = aws.Int32(int32(v)), true
	}
	if v, ok := number(params, "temperature"); ok {
		ic.Temperature, set = aws.Float32(float32(v)), true
	}
	if v, ok := number(params, "topP", "top_p"); ok {
		ic.TopP, set = aws.Float32(float32(v)), true
	}
	for _, k := range []string{"stopSequences", "stop_sequences"} {
		raw, ok := params[k].([]any)
		if !ok {
			continue
		}
		for _, s := range raw {
			if str, ok := s.(string); ok {
				ic.StopSequences = append(ic.StopSequences, str)
			}
		}
		set = true
		break
	}
	if !set {
		return nil
	}
	return &ic
}

var finishReasons = map[types.StopReason]string{
	types.StopReasonEndTurn: "stop", types.StopReasonStopSequence: "stop", types.StopReasonMaxTokens: "length",
	types.StopReasonToolUse: "tool_calls", types.StopReasonGuardrailIntervened: "content_filter",
	types.StopReasonContentFiltered: "content_filter",
}

func finishReason(r types.StopReason) string {
	if m, ok := finishReasons[r]; ok {
		return m
	}
	return string(r)
}

// protectedFields are handler-generated and never taken from ConverseOptions. ToolConfig is merged separately.
var protectedFields = map[string]bool{"ModelId": true, "Messages": true, "System": true, "ToolConfig": true}

// applyConverseOptions merges the caller's options o over the standard
// fields. A caller ToolConfig is used as-is when the handler sends no tools;
// otherwise only its ToolChoice applies, so the tool-result conversation
// stays valid. A caller OutputConfig without a TextFormat keeps the
// handler's structured-output format.
func applyConverseOptions(in, o *bedrockruntime.ConverseInput) {
	if o == nil {
		return
	}
	handlerOutput := in.OutputConfig
	copyFields(in, o, protectedFields)
	switch {
	case o.ToolConfig == nil:
	case in.ToolConfig == nil:
		in.ToolConfig = o.ToolConfig
	case o.ToolConfig.ToolChoice != nil:
		tc := *in.ToolConfig
		tc.ToolChoice = o.ToolConfig.ToolChoice
		in.ToolConfig = &tc
	}
	if handlerOutput != nil && in.OutputConfig != handlerOutput && in.OutputConfig.TextFormat == nil {
		oc := *in.OutputConfig
		oc.TextFormat = handlerOutput.TextFormat
		in.OutputConfig = &oc
	}
}

// outputConfig is Converse's native structured output for a schema.
// nil when schema is nil.
func outputConfig(schema map[string]any) (*types.OutputConfig, error) {
	if schema == nil {
		return nil, nil
	}
	b, err := json.Marshal(schema)
	if err != nil {
		return nil, err
	}
	return &types.OutputConfig{TextFormat: &types.OutputFormat{
		Type: types.OutputFormatTypeJsonSchema,
		Structure: &types.OutputFormatStructureMemberJsonSchema{Value: types.JsonSchemaDefinition{
			Name: aws.String("output"), Schema: aws.String(string(b)),
		}},
	}}, nil
}

// copyFields assigns each non-zero exported field of *src to the same-named
// field of *dst when the types are assignable. Reflection keeps fields added
// by future SDK versions flowing without a code change. dst and src must be
// non-nil pointers to structs (internal callers only).
func copyFields(dst, src any, skip map[string]bool) {
	d, s := reflect.ValueOf(dst).Elem(), reflect.ValueOf(src).Elem()
	st := s.Type()
	for i := range st.NumField() {
		f := st.Field(i)
		if !f.IsExported() || skip[f.Name] || s.Field(i).IsZero() {
			continue
		}
		df := d.FieldByName(f.Name)
		if df.IsValid() && df.CanSet() && f.Type.AssignableTo(df.Type()) {
			df.Set(s.Field(i))
		}
	}
}

// toStreamInput converts a Converse request to its ConverseStream form. The
// two inputs share every field except GuardrailConfig's type.
func toStreamInput(in *bedrockruntime.ConverseInput) *bedrockruntime.ConverseStreamInput {
	out := &bedrockruntime.ConverseStreamInput{}
	copyFields(out, in, nil)
	if g := in.GuardrailConfig; g != nil {
		out.GuardrailConfig = &types.GuardrailStreamConfiguration{
			GuardrailIdentifier: g.GuardrailIdentifier, GuardrailVersion: g.GuardrailVersion, Trace: g.Trace,
		}
	}
	return out
}

// streamBlock accumulates one content block of a ConverseStream turn.
type streamBlock struct {
	text           strings.Builder
	tool           bool
	toolID, name   string
	input          strings.Builder
	args           map[string]any
	hasTextContent bool
}

// streamAccumulator rebuilds the assistant message, stop reason and usage
// from ConverseStream events, keeping tool-use blocks for the follow-up turn.
type streamAccumulator struct {
	prefix string
	blocks map[int32]*streamBlock
	stop   types.StopReason
	usage  *types.TokenUsage
}

func newStreamAccumulator(prefix string) *streamAccumulator {
	return &streamAccumulator{prefix: prefix, blocks: map[int32]*streamBlock{}}
}

func (a *streamAccumulator) block(i *int32) *streamBlock {
	idx := aws.ToInt32(i)
	b := a.blocks[idx]
	if b == nil {
		b = &streamBlock{}
		a.blocks[idx] = b
	}
	return b
}

// add folds one event in and returns its text delta, if any.
func (a *streamAccumulator) add(ev types.ConverseStreamOutput) (string, error) {
	switch v := ev.(type) {
	case *types.ConverseStreamOutputMemberContentBlockStart:
		if t, ok := v.Value.Start.(*types.ContentBlockStartMemberToolUse); ok {
			b := a.block(v.Value.ContentBlockIndex)
			b.tool, b.toolID, b.name = true, aws.ToString(t.Value.ToolUseId), aws.ToString(t.Value.Name)
		}
	case *types.ConverseStreamOutputMemberContentBlockDelta:
		b := a.block(v.Value.ContentBlockIndex)
		switch d := v.Value.Delta.(type) {
		case *types.ContentBlockDeltaMemberText:
			b.text.WriteString(d.Value)
			b.hasTextContent = true
			return d.Value, nil
		case *types.ContentBlockDeltaMemberToolUse:
			b.input.WriteString(aws.ToString(d.Value.Input))
		}
	case *types.ConverseStreamOutputMemberContentBlockStop:
		b := a.block(v.Value.ContentBlockIndex)
		if b.tool && b.input.Len() > 0 {
			if err := json.Unmarshal([]byte(b.input.String()), &b.args); err != nil {
				return "", fmt.Errorf("%s: tool %q streamed invalid input JSON: %w", a.prefix, b.name, err)
			}
		}
	case *types.ConverseStreamOutputMemberMessageStop:
		a.stop = v.Value.StopReason
	case *types.ConverseStreamOutputMemberMetadata:
		a.usage = v.Value.Usage
	}
	return "", nil
}

// result returns the turn so far. A tool block's input is the JSON parsed at
// its contentBlockStop; an empty input is the empty object.
func (a *streamAccumulator) result() turnResult {
	idx := make([]int32, 0, len(a.blocks))
	for i := range a.blocks {
		idx = append(idx, i)
	}
	slices.Sort(idx)
	msg := types.Message{Role: types.ConversationRoleAssistant}
	for _, i := range idx {
		b := a.blocks[i]
		switch {
		case b.tool:
			args := b.args
			if args == nil {
				args = map[string]any{}
			}
			msg.Content = append(msg.Content, &types.ContentBlockMemberToolUse{Value: types.ToolUseBlock{
				ToolUseId: aws.String(b.toolID), Name: aws.String(b.name), Input: document.NewLazyDocument(args),
			}})
		case b.hasTextContent:
			msg.Content = append(msg.Content, &types.ContentBlockMemberText{Value: b.text.String()})
		}
	}
	return turnResult{msg: msg, stop: a.stop, usage: a.usage}
}
