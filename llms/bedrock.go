package llms

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"slices"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"

	"github.com/launchdarkly/go-ai-sdk/internal/bedrockwire"
)

// DefaultBedrockMaxTokens is sent when the call sets no max tokens.
const DefaultBedrockMaxTokens = 2048

// bedrockEmptyText stands in for a message with no content, which Converse
// rejects.
const bedrockEmptyText = "."

var bedrockImageFormats = map[string]types.ImageFormat{
	"image/png": types.ImageFormatPng, "image/jpeg": types.ImageFormatJpeg, "image/jpg": types.ImageFormatJpeg,
	"image/gif": types.ImageFormatGif, "image/webp": types.ImageFormatWebp,
}

// BedrockOptions configures NewBedrock. A nil Client is built from the
// default AWS configuration (environment, shared config, profile).
type BedrockOptions struct {
	Model  string
	Client *bedrockruntime.Client
}

// Bedrock is a Model on the Bedrock Converse API, which serves every model
// family with one request shape.
type Bedrock struct {
	client *bedrockruntime.Client
	model  string
}

// NewBedrock builds a Bedrock chat model.
func NewBedrock(o BedrockOptions) (*Bedrock, error) {
	c := o.Client
	if c == nil {
		var err error
		if c, err = bedrockwire.NewClient("", ""); err != nil {
			return nil, fmt.Errorf("llms: bedrock: %w", err)
		}
	}
	return &Bedrock{client: c, model: o.Model}, nil
}

// GenerateContent implements Model.
func (m *Bedrock) GenerateContent(ctx context.Context, messages []MessageContent, options ...CallOption) (*ContentResponse, error) {
	o := Apply(options...)
	in, err := m.input(messages, o)
	if err != nil {
		return nil, err
	}
	if o.StreamingFunc == nil {
		out, err := m.client.Converse(ctx, in)
		if err != nil {
			return nil, fmt.Errorf("llms: bedrock: %w", err)
		}
		msg, _ := out.Output.(*types.ConverseOutputMemberMessage)
		if msg == nil {
			return nil, ErrEmptyResponse
		}
		return bedrockResponse(msg.Value, out.StopReason, out.Usage)
	}
	out, err := m.client.ConverseStream(ctx, &bedrockruntime.ConverseStreamInput{
		ModelId: in.ModelId, Messages: in.Messages, System: in.System, InferenceConfig: in.InferenceConfig,
		ToolConfig: in.ToolConfig, AdditionalModelRequestFields: in.AdditionalModelRequestFields,
	})
	if err != nil {
		return nil, fmt.Errorf("llms: bedrock: %w", err)
	}
	stream := out.GetStream()
	defer stream.Close()
	acc := bedrockStream{blocks: map[int32]*bedrockBlock{}}
	for ev := range stream.Events() {
		text, err := acc.add(ev)
		if err != nil {
			return nil, err
		}
		if text != "" {
			if err := o.StreamingFunc(ctx, []byte(text)); err != nil {
				return nil, err
			}
		}
	}
	if err := stream.Err(); err != nil {
		return nil, fmt.Errorf("llms: bedrock: %w", err)
	}
	return bedrockResponse(acc.message(), acc.stop, acc.usage)
}

func (m *Bedrock) input(messages []MessageContent, o CallOptions) (*bedrockruntime.ConverseInput, error) {
	model := o.Model
	if model == "" {
		model = m.model
	}
	in := &bedrockruntime.ConverseInput{ModelId: aws.String(model)}
	for _, msg := range messages {
		if msg.Role == ChatMessageTypeSystem {
			if t := joinText(msg.Parts); strings.TrimSpace(t) != "" {
				in.System = append(in.System, &types.SystemContentBlockMemberText{Value: t})
			}
			continue
		}
		role := types.ConversationRoleUser
		switch msg.Role {
		case ChatMessageTypeAI:
			role = types.ConversationRoleAssistant
		case ChatMessageTypeHuman, ChatMessageTypeTool:
		default:
			return nil, fmt.Errorf("llms: bedrock: role %q not supported", msg.Role)
		}
		blocks, err := bedrockBlocks(msg.Parts)
		if err != nil {
			return nil, err
		}
		// Converse requires alternating roles: fold a message into the
		// previous one when both have the same role.
		if n := len(in.Messages); n > 0 && in.Messages[n-1].Role == role {
			in.Messages[n-1].Content = append(in.Messages[n-1].Content, blocks...)
			continue
		}
		in.Messages = append(in.Messages, types.Message{Role: role, Content: blocks})
	}
	for i := range in.Messages {
		if len(in.Messages[i].Content) == 0 {
			in.Messages[i].Content = []types.ContentBlock{&types.ContentBlockMemberText{Value: bedrockEmptyText}}
		}
	}
	cfg := &types.InferenceConfiguration{MaxTokens: aws.Int32(DefaultBedrockMaxTokens)}
	if o.MaxTokens > 0 {
		cfg.MaxTokens = aws.Int32(int32(o.MaxTokens))
	}
	if o.Temperature != nil {
		cfg.Temperature = aws.Float32(float32(*o.Temperature))
	}
	if o.TopP != nil {
		cfg.TopP = aws.Float32(float32(*o.TopP))
	}
	if len(o.StopWords) > 0 {
		cfg.StopSequences = o.StopWords
	}
	in.InferenceConfig = cfg
	// Converse has no top_k; Anthropic models take it as an extra field.
	if o.TopK != nil && strings.Contains(model, "anthropic") {
		in.AdditionalModelRequestFields = document.NewLazyDocument(map[string]any{"top_k": *o.TopK})
	}
	if len(o.Tools) > 0 {
		tc := &types.ToolConfiguration{}
		for _, t := range o.Tools {
			if t.Type != "function" || t.Function == nil {
				return nil, fmt.Errorf("llms: bedrock: tool type %q not supported", t.Type)
			}
			schema := t.Function.Parameters
			if schema == nil {
				schema = map[string]any{"type": "object", "properties": map[string]any{}}
			}
			spec := types.ToolSpecification{Name: aws.String(t.Function.Name), InputSchema: &types.ToolInputSchemaMemberJson{Value: document.NewLazyDocument(schema)}}
			if t.Function.Description != "" {
				spec.Description = aws.String(t.Function.Description)
			}
			tc.Tools = append(tc.Tools, &types.ToolMemberToolSpec{Value: spec})
		}
		in.ToolConfig = tc
	}
	return in, nil
}

func bedrockBlocks(parts []ContentPart) ([]types.ContentBlock, error) {
	var out []types.ContentBlock
	for _, part := range parts {
		switch v := part.(type) {
		case TextContent:
			if strings.TrimSpace(v.Text) != "" { // Converse rejects blank text blocks
				out = append(out, &types.ContentBlockMemberText{Value: v.Text})
			}
		case BinaryContent:
			b, err := bedrockImage(v.MIMEType, v.Data)
			if err != nil {
				return nil, err
			}
			out = append(out, b)
		case ImageURLContent:
			mime, data, ok := parseDataURL(v.URL)
			if !ok {
				return nil, fmt.Errorf("llms: bedrock: image URLs are not supported, only inline images: %q", v.URL)
			}
			raw, err := base64.StdEncoding.DecodeString(data)
			if err != nil {
				return nil, fmt.Errorf("llms: bedrock: decode image data URL: %w", err)
			}
			b, err := bedrockImage(mime, raw)
			if err != nil {
				return nil, err
			}
			out = append(out, b)
		case ToolCall:
			if v.FunctionCall == nil {
				return nil, fmt.Errorf("llms: bedrock: tool call %q has no function", v.ID)
			}
			out = append(out, &types.ContentBlockMemberToolUse{Value: types.ToolUseBlock{
				ToolUseId: aws.String(v.ID), Name: aws.String(v.FunctionCall.Name),
				Input: document.NewLazyDocument(toolInput(v.FunctionCall.Arguments)),
			}})
		case ToolCallResponse:
			out = append(out, &types.ContentBlockMemberToolResult{Value: types.ToolResultBlock{
				ToolUseId: aws.String(v.ToolCallID),
				Content:   []types.ToolResultContentBlock{&types.ToolResultContentBlockMemberText{Value: v.Content}},
			}})
		default:
			return nil, fmt.Errorf("llms: bedrock: unsupported message part %T", part)
		}
	}
	return out, nil
}

func bedrockImage(mime string, data []byte) (types.ContentBlock, error) {
	format, ok := bedrockImageFormats[strings.ToLower(mime)]
	if !ok {
		return nil, fmt.Errorf("llms: bedrock: unsupported image type %q", mime)
	}
	return &types.ContentBlockMemberImage{Value: types.ImageBlock{Format: format, Source: &types.ImageSourceMemberBytes{Value: data}}}, nil
}

func bedrockResponse(msg types.Message, stop types.StopReason, usage *types.TokenUsage) (*ContentResponse, error) {
	c := &ContentChoice{
		StopReason: string(stop),
		GenerationInfo: map[string]any{
			"input_tokens":  int(aws.ToInt32(usageField(usage, true))),
			"output_tokens": int(aws.ToInt32(usageField(usage, false))),
		},
	}
	var text strings.Builder
	for _, b := range msg.Content {
		switch v := b.(type) {
		case *types.ContentBlockMemberText:
			text.WriteString(v.Value)
		case *types.ContentBlockMemberToolUse:
			// Read the input through its JSON: a lazy document (the stream's)
			// can marshal but not unmarshal.
			input := map[string]any{}
			if v.Value.Input != nil {
				raw, err := v.Value.Input.MarshalSmithyDocument()
				if err == nil {
					err = json.Unmarshal(raw, &input)
				}
				if err != nil {
					return nil, fmt.Errorf("llms: bedrock: decode tool input: %w", err)
				}
			}
			args, err := json.Marshal(input)
			if err != nil {
				return nil, fmt.Errorf("llms: bedrock: encode tool input: %w", err)
			}
			c.ToolCalls = append(c.ToolCalls, ToolCall{
				ID: aws.ToString(v.Value.ToolUseId), Type: "function",
				FunctionCall: &FunctionCall{Name: aws.ToString(v.Value.Name), Arguments: string(args)},
			})
		}
	}
	c.Content = text.String()
	if c.Content == "" && len(c.ToolCalls) == 0 {
		return nil, ErrEmptyResponse
	}
	return &ContentResponse{Choices: []*ContentChoice{c}}, nil
}

func usageField(u *types.TokenUsage, input bool) *int32 {
	if u == nil {
		return nil
	}
	if input {
		return u.InputTokens
	}
	return u.OutputTokens
}

type bedrockBlock struct {
	tool         bool
	id, name     string
	text, input  strings.Builder
	sawText      bool
	completeArgs map[string]any
}

// bedrockStream rebuilds the assistant message, stop reason and usage from
// ConverseStream events.
type bedrockStream struct {
	blocks map[int32]*bedrockBlock
	stop   types.StopReason
	usage  *types.TokenUsage
}

func (a *bedrockStream) block(i *int32) *bedrockBlock {
	idx := aws.ToInt32(i)
	b := a.blocks[idx]
	if b == nil {
		b = &bedrockBlock{}
		a.blocks[idx] = b
	}
	return b
}

// add folds one event in and returns its text delta, if any.
func (a *bedrockStream) add(ev types.ConverseStreamOutput) (string, error) {
	switch v := ev.(type) {
	case *types.ConverseStreamOutputMemberContentBlockStart:
		if t, ok := v.Value.Start.(*types.ContentBlockStartMemberToolUse); ok {
			b := a.block(v.Value.ContentBlockIndex)
			b.tool, b.id, b.name = true, aws.ToString(t.Value.ToolUseId), aws.ToString(t.Value.Name)
		}
	case *types.ConverseStreamOutputMemberContentBlockDelta:
		b := a.block(v.Value.ContentBlockIndex)
		switch d := v.Value.Delta.(type) {
		case *types.ContentBlockDeltaMemberText:
			b.text.WriteString(d.Value)
			b.sawText = true
			return d.Value, nil
		case *types.ContentBlockDeltaMemberToolUse:
			b.input.WriteString(aws.ToString(d.Value.Input))
		}
	case *types.ConverseStreamOutputMemberContentBlockStop:
		b := a.block(v.Value.ContentBlockIndex)
		if b.tool && strings.TrimSpace(b.input.String()) != "" {
			if err := json.Unmarshal([]byte(b.input.String()), &b.completeArgs); err != nil {
				return "", fmt.Errorf("llms: bedrock: tool %q streamed invalid input JSON: %w", b.name, err)
			}
		}
	case *types.ConverseStreamOutputMemberMessageStop:
		a.stop = v.Value.StopReason
	case *types.ConverseStreamOutputMemberMetadata:
		a.usage = v.Value.Usage
	}
	return "", nil
}

func (a *bedrockStream) message() types.Message {
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
			args := b.completeArgs
			if args == nil {
				args = map[string]any{}
			}
			msg.Content = append(msg.Content, &types.ContentBlockMemberToolUse{Value: types.ToolUseBlock{
				ToolUseId: aws.String(b.id), Name: aws.String(b.name), Input: document.NewLazyDocument(args),
			}})
		case b.sawText:
			msg.Content = append(msg.Content, &types.ContentBlockMemberText{Value: b.text.String()})
		}
	}
	return msg
}
