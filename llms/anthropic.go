package llms

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// ErrMissingAnthropicKey is returned by NewAnthropic when no API key is set.
var ErrMissingAnthropicKey = errors.New("llms: missing the Anthropic API key, set it in the ANTHROPIC_API_KEY environment variable")

const (
	// DefaultAnthropicModel is used when neither the call nor the
	// constructor names a model.
	DefaultAnthropicModel = "claude-sonnet-4-6"
	// DefaultAnthropicMaxTokens is sent when the call sets no max tokens;
	// the Messages API requires the field.
	DefaultAnthropicMaxTokens = 2048
)

// AnthropicOptions configures NewAnthropic. An empty APIKey falls back to
// ANTHROPIC_API_KEY.
type AnthropicOptions struct {
	APIKey         string
	Model          string
	BaseURL        string
	RequestOptions []option.RequestOption
}

// Anthropic is a Model on the Anthropic Messages API.
type Anthropic struct {
	client anthropic.Client
	model  string
}

// NewAnthropic builds an Anthropic chat model.
func NewAnthropic(o AnthropicOptions) (*Anthropic, error) {
	key := o.APIKey
	if key == "" {
		key = os.Getenv("ANTHROPIC_API_KEY")
	}
	if key == "" {
		return nil, ErrMissingAnthropicKey
	}
	opts := []option.RequestOption{option.WithAPIKey(key)}
	if o.BaseURL != "" {
		opts = append(opts, option.WithBaseURL(o.BaseURL))
	}
	opts = append(opts, o.RequestOptions...)
	return &Anthropic{client: anthropic.NewClient(opts...), model: o.Model}, nil
}

// GenerateContent implements Model.
func (m *Anthropic) GenerateContent(ctx context.Context, messages []MessageContent, options ...CallOption) (*ContentResponse, error) {
	o := Apply(options...)
	params, err := m.params(messages, o)
	if err != nil {
		return nil, err
	}
	if o.StreamingFunc == nil {
		msg, err := m.client.Messages.New(ctx, params)
		if err != nil {
			return nil, fmt.Errorf("llms: anthropic: %w", err)
		}
		return anthropicResponse(msg)
	}
	stream := m.client.Messages.NewStreaming(ctx, params)
	defer stream.Close()
	var msg anthropic.Message
	for stream.Next() {
		ev := stream.Current()
		if err := msg.Accumulate(ev); err != nil {
			return nil, fmt.Errorf("llms: anthropic: %w", err)
		}
		if ev.Type == "content_block_delta" && ev.Delta.Type == "text_delta" && ev.Delta.Text != "" {
			if err := o.StreamingFunc(ctx, []byte(ev.Delta.Text)); err != nil {
				return nil, err
			}
		}
	}
	if err := stream.Err(); err != nil {
		return nil, fmt.Errorf("llms: anthropic: %w", err)
	}
	return anthropicResponse(&msg)
}

func (m *Anthropic) params(messages []MessageContent, o CallOptions) (anthropic.MessageNewParams, error) {
	model := o.Model
	if model == "" {
		model = m.model
	}
	if model == "" {
		model = DefaultAnthropicModel
	}
	max := int64(o.MaxTokens)
	if max <= 0 {
		max = DefaultAnthropicMaxTokens
	}
	p := anthropic.MessageNewParams{Model: anthropic.Model(model), MaxTokens: max}
	var system strings.Builder
	for _, msg := range messages {
		switch msg.Role {
		case ChatMessageTypeSystem:
			for _, part := range msg.Parts {
				t, ok := part.(TextContent)
				if !ok {
					return p, fmt.Errorf("llms: anthropic: system message part %T is not text", part)
				}
				system.WriteString(t.Text)
			}
		case ChatMessageTypeHuman, ChatMessageTypeTool:
			blocks, err := anthropicUserBlocks(msg.Parts)
			if err != nil {
				return p, err
			}
			p.Messages = appendAnthropic(p.Messages, anthropic.MessageParamRoleUser, blocks)
		case ChatMessageTypeAI:
			blocks, err := anthropicAssistantBlocks(msg.Parts)
			if err != nil {
				return p, err
			}
			p.Messages = appendAnthropic(p.Messages, anthropic.MessageParamRoleAssistant, blocks)
		default:
			return p, fmt.Errorf("llms: anthropic: role %q not supported", msg.Role)
		}
	}
	if system.Len() > 0 {
		p.System = []anthropic.TextBlockParam{{Text: system.String()}}
	}
	if o.Temperature != nil {
		p.Temperature = anthropic.Float(*o.Temperature)
	}
	if o.TopP != nil {
		p.TopP = anthropic.Float(*o.TopP)
	}
	if o.TopK != nil {
		p.TopK = anthropic.Int(int64(*o.TopK))
	}
	if len(o.StopWords) > 0 {
		p.StopSequences = o.StopWords
	}
	for _, t := range o.Tools {
		if t.Type != "function" || t.Function == nil {
			return p, fmt.Errorf("llms: anthropic: tool type %q not supported", t.Type)
		}
		p.Tools = append(p.Tools, anthropicTool(t.Function))
	}
	return p, nil
}

// appendAnthropic adds a turn, folding it into the previous one when both
// have the same role. Anthropic requires every tool_use id to be answered in
// the single user message that follows, and the handlers send one tool
// message per parallel call.
func appendAnthropic(msgs []anthropic.MessageParam, role anthropic.MessageParamRole, blocks []anthropic.ContentBlockParamUnion) []anthropic.MessageParam {
	if n := len(msgs); n > 0 && msgs[n-1].Role == role {
		msgs[n-1].Content = append(msgs[n-1].Content, blocks...)
		return msgs
	}
	return append(msgs, anthropic.MessageParam{Role: role, Content: blocks})
}

func anthropicTool(f *FunctionDefinition) anthropic.ToolUnionParam {
	schema := anthropic.ToolInputSchemaParam{ExtraFields: map[string]any{}}
	if params, ok := f.Parameters.(map[string]any); ok {
		for k, v := range params {
			switch k {
			case "type":
			case "properties":
				schema.Properties = v
			case "required":
				if req, ok := v.([]any); ok {
					for _, r := range req {
						if s, ok := r.(string); ok {
							schema.Required = append(schema.Required, s)
						}
					}
				} else if req, ok := v.([]string); ok {
					schema.Required = req
				}
			default:
				schema.ExtraFields[k] = v
			}
		}
	}
	tp := anthropic.ToolParam{Name: f.Name, InputSchema: schema}
	if f.Description != "" {
		tp.Description = anthropic.String(f.Description)
	}
	return anthropic.ToolUnionParam{OfTool: &tp}
}

func anthropicUserBlocks(parts []ContentPart) ([]anthropic.ContentBlockParamUnion, error) {
	var out []anthropic.ContentBlockParamUnion
	for _, part := range parts {
		switch v := part.(type) {
		case TextContent:
			out = append(out, anthropic.NewTextBlock(v.Text))
		case BinaryContent:
			out = append(out, anthropic.NewImageBlockBase64(v.MIMEType, base64.StdEncoding.EncodeToString(v.Data)))
		case ImageURLContent:
			if mime, data, ok := parseDataURL(v.URL); ok {
				out = append(out, anthropic.NewImageBlockBase64(mime, data))
			} else {
				out = append(out, anthropic.NewImageBlock(anthropic.URLImageSourceParam{URL: v.URL}))
			}
		case ToolCallResponse:
			out = append(out, anthropic.NewToolResultBlock(v.ToolCallID, v.Content, false))
		default:
			return nil, fmt.Errorf("llms: anthropic: unsupported user message part %T", part)
		}
	}
	if len(out) == 0 {
		return nil, errors.New("llms: anthropic: message has no content")
	}
	return out, nil
}

func anthropicAssistantBlocks(parts []ContentPart) ([]anthropic.ContentBlockParamUnion, error) {
	var out []anthropic.ContentBlockParamUnion
	for _, part := range parts {
		switch v := part.(type) {
		case TextContent:
			out = append(out, anthropic.NewTextBlock(v.Text))
		case ToolCall:
			if v.FunctionCall == nil {
				return nil, fmt.Errorf("llms: anthropic: tool call %q has no function", v.ID)
			}
			out = append(out, anthropic.NewToolUseBlock(v.ID, toolInput(v.FunctionCall.Arguments), v.FunctionCall.Name))
		default:
			return nil, fmt.Errorf("llms: anthropic: unsupported assistant message part %T", part)
		}
	}
	if len(out) == 0 {
		return nil, errors.New("llms: anthropic: message has no content")
	}
	return out, nil
}

// toolInput decodes a tool call's JSON arguments into its input object:
// empty arguments are {}, and arguments that are not a JSON object are
// kept under "arguments".
func toolInput(args string) map[string]any {
	in := map[string]any{}
	if a := strings.TrimSpace(args); a != "" {
		if json.Unmarshal([]byte(a), &in) != nil {
			return map[string]any{"arguments": args}
		}
	}
	return in
}

// parseDataURL splits a base64 data: URL into its media type and data.
func parseDataURL(u string) (mime, data string, ok bool) {
	rest, ok := strings.CutPrefix(u, "data:")
	if !ok {
		return "", "", false
	}
	meta, data, ok := strings.Cut(rest, ",")
	if !ok {
		return "", "", false
	}
	mime, enc, _ := strings.Cut(meta, ";")
	return mime, data, enc == "base64"
}

func anthropicResponse(msg *anthropic.Message) (*ContentResponse, error) {
	if msg == nil || len(msg.Content) == 0 {
		return nil, ErrEmptyResponse
	}
	info := func() map[string]any {
		return map[string]any{
			"InputTokens":              int(msg.Usage.InputTokens),
			"OutputTokens":             int(msg.Usage.OutputTokens),
			"CacheCreationInputTokens": int(msg.Usage.CacheCreationInputTokens),
			"CacheReadInputTokens":     int(msg.Usage.CacheReadInputTokens),
		}
	}
	out := &ContentResponse{}
	for _, b := range msg.Content {
		c := &ContentChoice{StopReason: string(msg.StopReason), GenerationInfo: info()}
		switch b.Type {
		case "text":
			c.Content = b.Text
			c.Parts = []ContentPart{TextContent{Text: b.Text}}
		case "tool_use":
			var input any
			if len(b.Input) > 0 {
				if err := json.Unmarshal(b.Input, &input); err != nil {
					return nil, fmt.Errorf("llms: anthropic: decode tool input: %w", err)
				}
			}
			// Re-marshal so the arguments are compact with sorted keys.
			args, err := json.Marshal(input)
			if err != nil {
				return nil, fmt.Errorf("llms: anthropic: encode tool input: %w", err)
			}
			tc := ToolCall{ID: b.ID, FunctionCall: &FunctionCall{Name: b.Name, Arguments: string(args)}}
			c.ToolCalls = []ToolCall{tc}
			c.Parts = []ContentPart{tc}
		default:
			continue // thinking and server-tool blocks are not part of this interface
		}
		out.Choices = append(out.Choices, c)
	}
	if len(out.Choices) == 0 {
		return nil, ErrEmptyResponse
	}
	return out, nil
}
