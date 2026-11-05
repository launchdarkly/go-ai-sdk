package llms

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/shared"
)

// ErrMissingOpenAIKey is returned by NewOpenAI when no API key is set.
var ErrMissingOpenAIKey = errors.New("llms: missing the OpenAI API key, set it in the OPENAI_API_KEY environment variable")

// DefaultOpenAIModel is used when neither the call, the constructor nor
// OPENAI_MODEL names a model.
const DefaultOpenAIModel = "gpt-4o"

// OpenAIOptions configures NewOpenAI. Empty fields fall back to the
// environment: OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL (then
// OPENAI_API_BASE) and OPENAI_ORGANIZATION.
type OpenAIOptions struct {
	APIKey       string
	Model        string
	BaseURL      string
	Organization string
	// JSONSchema, when set, is sent as the json_schema response format on
	// every call, verbatim.
	JSONSchema map[string]any
	// RequestOptions are passed to the openai-go client, after the above.
	RequestOptions []option.RequestOption
}

// OpenAI is a Model on the OpenAI Chat Completions API. It also serves any
// OpenAI-compatible endpoint through BaseURL / OPENAI_BASE_URL.
type OpenAI struct {
	client openai.Client
	model  string
	schema map[string]any
}

// NewOpenAI builds an OpenAI chat model.
func NewOpenAI(o OpenAIOptions) (*OpenAI, error) {
	key := o.APIKey
	if key == "" {
		key = os.Getenv("OPENAI_API_KEY")
	}
	if key == "" {
		return nil, ErrMissingOpenAIKey
	}
	base := o.BaseURL
	if base == "" {
		var ok bool
		if base, ok = os.LookupEnv("OPENAI_BASE_URL"); !ok {
			base = os.Getenv("OPENAI_API_BASE")
		}
	}
	org := o.Organization
	if org == "" {
		org = os.Getenv("OPENAI_ORGANIZATION")
	}
	model := o.Model
	if model == "" {
		model = os.Getenv("OPENAI_MODEL")
	}
	opts := []option.RequestOption{option.WithAPIKey(key)}
	if base = strings.TrimSuffix(base, "/"); base != "" {
		opts = append(opts, option.WithBaseURL(base+"/"))
	}
	if org != "" {
		opts = append(opts, option.WithOrganization(org))
	}
	opts = append(opts, o.RequestOptions...)
	return &OpenAI{client: openai.NewClient(opts...), model: model, schema: o.JSONSchema}, nil
}

// GenerateContent implements Model.
func (m *OpenAI) GenerateContent(ctx context.Context, messages []MessageContent, options ...CallOption) (*ContentResponse, error) {
	o := Apply(options...)
	params, err := m.params(messages, o)
	if err != nil {
		return nil, err
	}
	if o.StreamingFunc == nil {
		resp, err := m.client.Chat.Completions.New(ctx, params)
		if err != nil {
			return nil, fmt.Errorf("llms: openai: %w", err)
		}
		return openAIResponse(resp)
	}
	params.StreamOptions = openai.ChatCompletionStreamOptionsParam{IncludeUsage: openai.Bool(true)}
	stream := m.client.Chat.Completions.NewStreaming(ctx, params)
	defer stream.Close()
	var acc openai.ChatCompletionAccumulator
	for stream.Next() {
		chunk := stream.Current()
		acc.AddChunk(chunk)
		for _, c := range chunk.Choices {
			if c.Delta.Content == "" {
				continue
			}
			if err := o.StreamingFunc(ctx, []byte(c.Delta.Content)); err != nil {
				return nil, err
			}
		}
	}
	if err := stream.Err(); err != nil {
		return nil, fmt.Errorf("llms: openai: %w", err)
	}
	return openAIResponse(&acc.ChatCompletion)
}

func (m *OpenAI) params(messages []MessageContent, o CallOptions) (openai.ChatCompletionNewParams, error) {
	model := o.Model
	if model == "" {
		model = m.model
	}
	if model == "" {
		model = DefaultOpenAIModel
	}
	p := openai.ChatCompletionNewParams{Model: model}
	for _, msg := range messages {
		mp, err := openAIMessage(msg)
		if err != nil {
			return p, err
		}
		p.Messages = append(p.Messages, mp)
	}
	if o.MaxTokens > 0 {
		p.MaxCompletionTokens = openai.Int(int64(o.MaxTokens))
	}
	if o.Temperature != nil {
		p.Temperature = openai.Float(*o.Temperature)
	}
	if o.TopP != nil {
		p.TopP = openai.Float(*o.TopP)
	}
	if o.Seed != nil {
		p.Seed = openai.Int(int64(*o.Seed))
	}
	if o.FrequencyPenalty != nil {
		p.FrequencyPenalty = openai.Float(*o.FrequencyPenalty)
	}
	if o.PresencePenalty != nil {
		p.PresencePenalty = openai.Float(*o.PresencePenalty)
	}
	if len(o.StopWords) > 0 {
		p.Stop = openai.ChatCompletionNewParamsStopUnion{OfStringArray: o.StopWords}
	}
	for _, t := range o.Tools {
		if t.Type != "function" || t.Function == nil {
			return p, fmt.Errorf("llms: openai: tool type %q not supported", t.Type)
		}
		def := shared.FunctionDefinitionParam{Name: t.Function.Name, Parameters: functionParameters(t.Function.Parameters)}
		if t.Function.Description != "" {
			def.Description = openai.String(t.Function.Description)
		}
		p.Tools = append(p.Tools, openai.ChatCompletionToolUnionParam{OfFunction: &openai.ChatCompletionFunctionToolParam{Function: def}})
	}
	switch {
	case m.schema != nil:
		p.ResponseFormat = openai.ChatCompletionNewParamsResponseFormatUnion{OfJSONSchema: &shared.ResponseFormatJSONSchemaParam{
			JSONSchema: shared.ResponseFormatJSONSchemaJSONSchemaParam{Name: "output", Strict: openai.Bool(false), Schema: m.schema},
		}}
	case o.JSONMode:
		p.ResponseFormat = openai.ChatCompletionNewParamsResponseFormatUnion{OfJSONObject: &shared.ResponseFormatJSONObjectParam{}}
	}
	return p, nil
}

// functionParameters sends a schema as given, adding "properties": {} to an
// object schema without them, which the Chat Completions API requires.
func functionParameters(params any) shared.FunctionParameters {
	m, ok := params.(map[string]any)
	if !ok {
		return nil
	}
	if m["type"] == "object" {
		if _, has := m["properties"]; !has {
			c := make(map[string]any, len(m)+1)
			for k, v := range m {
				c[k] = v
			}
			c["properties"] = map[string]any{}
			return c
		}
	}
	return m
}

func openAIMessage(msg MessageContent) (openai.ChatCompletionMessageParamUnion, error) {
	switch msg.Role {
	case ChatMessageTypeSystem:
		return openai.ChatCompletionMessageParamUnion{OfSystem: &openai.ChatCompletionSystemMessageParam{
			Content: openai.ChatCompletionSystemMessageParamContentUnion{OfString: openai.String(joinText(msg.Parts))},
		}}, nil
	case ChatMessageTypeHuman:
		u := &openai.ChatCompletionUserMessageParam{}
		parts, err := openAIUserParts(msg.Parts)
		if err != nil {
			return openai.ChatCompletionMessageParamUnion{}, err
		}
		if len(parts) == 1 && parts[0].OfText != nil {
			u.Content.OfString = openai.String(parts[0].OfText.Text)
		} else if len(parts) > 0 {
			u.Content.OfArrayOfContentParts = parts
		} else {
			u.Content.OfString = openai.String("")
		}
		return openai.ChatCompletionMessageParamUnion{OfUser: u}, nil
	case ChatMessageTypeAI:
		a := &openai.ChatCompletionAssistantMessageParam{}
		a.Content.OfString = openai.String(joinText(msg.Parts))
		for _, p := range msg.Parts {
			tc, ok := p.(ToolCall)
			if !ok || tc.FunctionCall == nil {
				continue
			}
			a.ToolCalls = append(a.ToolCalls, openai.ChatCompletionMessageToolCallUnionParam{OfFunction: &openai.ChatCompletionMessageFunctionToolCallParam{
				ID:       tc.ID,
				Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{Name: tc.FunctionCall.Name, Arguments: tc.FunctionCall.Arguments},
			}})
		}
		return openai.ChatCompletionMessageParamUnion{OfAssistant: a}, nil
	case ChatMessageTypeTool:
		if len(msg.Parts) != 1 {
			return openai.ChatCompletionMessageParamUnion{}, fmt.Errorf("llms: openai: expected exactly one part for role tool, got %d", len(msg.Parts))
		}
		r, ok := msg.Parts[0].(ToolCallResponse)
		if !ok {
			return openai.ChatCompletionMessageParamUnion{}, fmt.Errorf("llms: openai: expected a ToolCallResponse for role tool, got %T", msg.Parts[0])
		}
		return openai.ChatCompletionMessageParamUnion{OfTool: &openai.ChatCompletionToolMessageParam{
			ToolCallID: r.ToolCallID,
			Content:    openai.ChatCompletionToolMessageParamContentUnion{OfString: openai.String(r.Content)},
		}}, nil
	}
	return openai.ChatCompletionMessageParamUnion{}, fmt.Errorf("llms: openai: role %q not supported", msg.Role)
}

func openAIUserParts(parts []ContentPart) ([]openai.ChatCompletionContentPartUnionParam, error) {
	var out []openai.ChatCompletionContentPartUnionParam
	for _, p := range parts {
		switch v := p.(type) {
		case TextContent:
			out = append(out, openai.ChatCompletionContentPartUnionParam{OfText: &openai.ChatCompletionContentPartTextParam{Text: v.Text}})
		case ImageURLContent:
			out = append(out, openAIImage(v.URL, v.Detail))
		case BinaryContent:
			out = append(out, openAIImage(dataURL(v), ""))
		}
	}
	return out, nil
}

func openAIImage(url, detail string) openai.ChatCompletionContentPartUnionParam {
	return openai.ChatCompletionContentPartUnionParam{OfImageURL: &openai.ChatCompletionContentPartImageParam{
		ImageURL: openai.ChatCompletionContentPartImageImageURLParam{URL: url, Detail: detail},
	}}
}

func dataURL(b BinaryContent) string {
	return "data:" + b.MIMEType + ";base64," + base64.StdEncoding.EncodeToString(b.Data)
}

func joinText(parts []ContentPart) string {
	var b strings.Builder
	for _, p := range parts {
		if t, ok := p.(TextContent); ok {
			b.WriteString(t.Text)
		}
	}
	return b.String()
}

func openAIResponse(resp *openai.ChatCompletion) (*ContentResponse, error) {
	if resp == nil || len(resp.Choices) == 0 {
		return nil, ErrEmptyResponse
	}
	u := resp.Usage
	out := &ContentResponse{}
	for _, c := range resp.Choices {
		choice := &ContentChoice{
			Content:    c.Message.Content,
			StopReason: c.FinishReason,
			GenerationInfo: map[string]any{
				"CompletionTokens":                   int(u.CompletionTokens),
				"PromptTokens":                       int(u.PromptTokens),
				"TotalTokens":                        int(u.TotalTokens),
				"ReasoningTokens":                    int(u.CompletionTokensDetails.ReasoningTokens),
				"ThinkingTokens":                     int(u.CompletionTokensDetails.ReasoningTokens),
				"CompletionReasoningTokens":          int(u.CompletionTokensDetails.ReasoningTokens),
				"PromptAudioTokens":                  int(u.PromptTokensDetails.AudioTokens),
				"PromptCachedTokens":                 int(u.PromptTokensDetails.CachedTokens),
				"CompletionAudioTokens":              int(u.CompletionTokensDetails.AudioTokens),
				"CompletionAcceptedPredictionTokens": int(u.CompletionTokensDetails.AcceptedPredictionTokens),
				"CompletionRejectedPredictionTokens": int(u.CompletionTokensDetails.RejectedPredictionTokens),
				"ThinkingContent":                    "",
			},
		}
		for _, tc := range c.Message.ToolCalls {
			choice.ToolCalls = append(choice.ToolCalls, ToolCall{
				ID: tc.ID, Type: tc.Type, FunctionCall: &FunctionCall{Name: tc.Function.Name, Arguments: tc.Function.Arguments},
			})
		}
		out.Choices = append(out.Choices, choice)
	}
	return out, nil
}
