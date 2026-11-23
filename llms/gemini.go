package llms

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"

	"google.golang.org/genai"
)

// ErrMissingGeminiKey is returned by NewGemini when no API key is set.
var ErrMissingGeminiKey = errors.New("llms: missing the Gemini API key, set it in the GOOGLE_API_KEY or GEMINI_API_KEY environment variable")

// DefaultGeminiModel is used when neither the call nor the constructor
// names a model.
const DefaultGeminiModel = "gemini-2.5-flash"

// geminiLocalIDPrefix marks a tool call id this package made up because the
// Gemini API returned none. It is never sent back to the API.
const geminiLocalIDPrefix = "gemini-call-"

// GeminiOptions configures NewGemini. An empty APIKey falls back to
// GOOGLE_API_KEY, then GEMINI_API_KEY. A non-nil JSONSchema asks for a JSON
// reply that matches it.
type GeminiOptions struct {
	APIKey      string
	Model       string
	BaseURL     string
	JSONSchema  map[string]any
	HTTPOptions genai.HTTPOptions
}

// Gemini is a Model on the Gemini API (generateContent).
type Gemini struct {
	client *genai.Client
	model  string
	schema map[string]any
}

// NewGemini builds a Gemini chat model.
func NewGemini(o GeminiOptions) (*Gemini, error) {
	key := o.APIKey
	if key == "" {
		key = os.Getenv("GOOGLE_API_KEY")
	}
	if key == "" {
		key = os.Getenv("GEMINI_API_KEY")
	}
	if key == "" {
		return nil, ErrMissingGeminiKey
	}
	httpOpts := o.HTTPOptions
	if o.BaseURL != "" {
		httpOpts.BaseURL = o.BaseURL
	}
	client, err := genai.NewClient(context.Background(), &genai.ClientConfig{
		APIKey:      key,
		Backend:     genai.BackendGeminiAPI,
		HTTPOptions: httpOpts,
	})
	if err != nil {
		return nil, fmt.Errorf("llms: gemini: %w", err)
	}
	return &Gemini{client: client, model: o.Model, schema: o.JSONSchema}, nil
}

// GenerateContent implements Model.
func (m *Gemini) GenerateContent(ctx context.Context, messages []MessageContent, options ...CallOption) (*ContentResponse, error) {
	o := Apply(options...)
	model := o.Model
	if model == "" {
		model = m.model
	}
	if model == "" {
		model = DefaultGeminiModel
	}
	contents, cfg, err := m.request(messages, o)
	if err != nil {
		return nil, err
	}
	if o.StreamingFunc == nil {
		resp, err := m.client.Models.GenerateContent(ctx, model, contents, cfg)
		if err != nil {
			return nil, fmt.Errorf("llms: gemini: %w", err)
		}
		return geminiResponse(resp)
	}
	var acc geminiStream
	for chunk, err := range m.client.Models.GenerateContentStream(ctx, model, contents, cfg) {
		if err != nil {
			return nil, fmt.Errorf("llms: gemini: %w", err)
		}
		delta := acc.add(chunk)
		if delta != "" {
			if err := o.StreamingFunc(ctx, []byte(delta)); err != nil {
				return nil, err
			}
		}
	}
	return geminiResponse(acc.response())
}

func (m *Gemini) request(messages []MessageContent, o CallOptions) ([]*genai.Content, *genai.GenerateContentConfig, error) {
	cfg := &genai.GenerateContentConfig{}
	var system strings.Builder
	var contents []*genai.Content
	for _, msg := range messages {
		switch msg.Role {
		case ChatMessageTypeSystem:
			for _, part := range msg.Parts {
				t, ok := part.(TextContent)
				if !ok {
					return nil, nil, fmt.Errorf("llms: gemini: system message part %T is not text", part)
				}
				system.WriteString(t.Text)
			}
		case ChatMessageTypeHuman, ChatMessageTypeTool:
			parts, err := geminiUserParts(msg.Parts)
			if err != nil {
				return nil, nil, err
			}
			contents = appendGemini(contents, genai.RoleUser, parts)
		case ChatMessageTypeAI:
			parts, err := geminiModelParts(msg.Parts)
			if err != nil {
				return nil, nil, err
			}
			contents = appendGemini(contents, genai.RoleModel, parts)
		default:
			return nil, nil, fmt.Errorf("llms: gemini: role %q not supported", msg.Role)
		}
	}
	if strings.TrimSpace(system.String()) != "" {
		cfg.SystemInstruction = &genai.Content{Parts: []*genai.Part{{Text: system.String()}}}
	}
	if o.MaxTokens > 0 {
		cfg.MaxOutputTokens = int32(o.MaxTokens)
	}
	if o.Temperature != nil {
		cfg.Temperature = genai.Ptr(float32(*o.Temperature))
	}
	if o.TopP != nil {
		cfg.TopP = genai.Ptr(float32(*o.TopP))
	}
	if o.TopK != nil {
		cfg.TopK = genai.Ptr(float32(*o.TopK))
	}
	if o.Seed != nil {
		cfg.Seed = genai.Ptr(int32(*o.Seed))
	}
	if o.FrequencyPenalty != nil {
		cfg.FrequencyPenalty = genai.Ptr(float32(*o.FrequencyPenalty))
	}
	if o.PresencePenalty != nil {
		cfg.PresencePenalty = genai.Ptr(float32(*o.PresencePenalty))
	}
	if len(o.StopWords) > 0 {
		cfg.StopSequences = o.StopWords
	}
	if m.schema != nil {
		cfg.ResponseMIMEType = "application/json"
		cfg.ResponseJsonSchema = m.schema
	} else if o.JSONMode {
		cfg.ResponseMIMEType = "application/json"
	}
	if len(o.Tools) > 0 {
		decls := make([]*genai.FunctionDeclaration, 0, len(o.Tools))
		for _, t := range o.Tools {
			if t.Type != "function" || t.Function == nil {
				return nil, nil, fmt.Errorf("llms: gemini: tool type %q not supported", t.Type)
			}
			decls = append(decls, &genai.FunctionDeclaration{
				Name:                 t.Function.Name,
				Description:          t.Function.Description,
				ParametersJsonSchema: t.Function.Parameters,
			})
		}
		cfg.Tools = []*genai.Tool{{FunctionDeclarations: decls}}
	}
	return contents, cfg, nil
}

// appendGemini adds a turn, folding it into the previous one when both have
// the same role. Gemini expects every function response for a model turn in
// the single user turn that follows, and the handlers send one tool message
// per parallel call.
func appendGemini(contents []*genai.Content, role string, parts []*genai.Part) []*genai.Content {
	if n := len(contents); n > 0 && contents[n-1].Role == role {
		contents[n-1].Parts = append(contents[n-1].Parts, parts...)
		return contents
	}
	return append(contents, &genai.Content{Role: role, Parts: parts})
}

func geminiUserParts(parts []ContentPart) ([]*genai.Part, error) {
	var out []*genai.Part
	for _, part := range parts {
		switch v := part.(type) {
		case TextContent:
			if strings.TrimSpace(v.Text) != "" {
				out = append(out, &genai.Part{Text: v.Text})
			}
		case BinaryContent:
			out = append(out, &genai.Part{InlineData: &genai.Blob{MIMEType: v.MIMEType, Data: v.Data}})
		case ImageURLContent:
			mime, data, ok := parseDataURL(v.URL)
			if !ok {
				return nil, fmt.Errorf("llms: gemini: image %q is not a base64 data: URL; the Gemini API takes inline images", v.URL)
			}
			raw, err := base64.StdEncoding.DecodeString(data)
			if err != nil {
				return nil, fmt.Errorf("llms: gemini: decode image: %w", err)
			}
			out = append(out, &genai.Part{InlineData: &genai.Blob{MIMEType: mime, Data: raw}})
		case ToolCallResponse:
			out = append(out, &genai.Part{FunctionResponse: &genai.FunctionResponse{
				ID:       geminiWireID(v.ToolCallID),
				Name:     v.Name,
				Response: geminiToolOutput(v.Content),
			}})
		default:
			return nil, fmt.Errorf("llms: gemini: unsupported user message part %T", part)
		}
	}
	if len(out) == 0 {
		return nil, errors.New("llms: gemini: message has no content")
	}
	return out, nil
}

func geminiModelParts(parts []ContentPart) ([]*genai.Part, error) {
	var out []*genai.Part
	for _, part := range parts {
		switch v := part.(type) {
		case TextContent:
			if strings.TrimSpace(v.Text) != "" {
				out = append(out, &genai.Part{Text: v.Text})
			}
		case ToolCall:
			if v.FunctionCall == nil {
				return nil, fmt.Errorf("llms: gemini: tool call %q has no function", v.ID)
			}
			out = append(out, &genai.Part{
				FunctionCall:     &genai.FunctionCall{ID: geminiWireID(v.ID), Name: v.FunctionCall.Name, Args: toolInput(v.FunctionCall.Arguments)},
				ThoughtSignature: v.Signature,
			})
		default:
			return nil, fmt.Errorf("llms: gemini: unsupported assistant message part %T", part)
		}
	}
	if len(out) == 0 {
		return nil, errors.New("llms: gemini: message has no content")
	}
	return out, nil
}

// geminiToolOutput is a tool result as a function response object: a JSON
// object result as is, anything else under "output".
func geminiToolOutput(content string) map[string]any {
	var obj map[string]any
	if json.Unmarshal([]byte(content), &obj) == nil && obj != nil {
		return obj
	}
	return map[string]any{"output": content}
}

func geminiWireID(id string) string {
	if strings.HasPrefix(id, geminiLocalIDPrefix) {
		return ""
	}
	return id
}

// geminiStream folds streamed chunks into one response: text deltas are
// joined, function calls kept in order, and the last usage and finish
// reason win.
type geminiStream struct {
	parts  []*genai.Part
	usage  *genai.GenerateContentResponseUsageMetadata
	finish genai.FinishReason
}

func (s *geminiStream) add(chunk *genai.GenerateContentResponse) string {
	if chunk == nil {
		return ""
	}
	if chunk.UsageMetadata != nil {
		s.usage = chunk.UsageMetadata
	}
	if len(chunk.Candidates) == 0 || chunk.Candidates[0] == nil {
		return ""
	}
	c := chunk.Candidates[0]
	if c.FinishReason != "" {
		s.finish = c.FinishReason
	}
	if c.Content == nil {
		return ""
	}
	var delta strings.Builder
	for _, p := range c.Content.Parts {
		if p == nil {
			continue
		}
		if p.Text != "" && !p.Thought {
			delta.WriteString(p.Text)
			if n := len(s.parts); n > 0 && s.parts[n-1].Text != "" && s.parts[n-1].FunctionCall == nil && !s.parts[n-1].Thought {
				s.parts[n-1].Text += p.Text
				continue
			}
		}
		cp := *p
		s.parts = append(s.parts, &cp)
	}
	return delta.String()
}

func (s *geminiStream) response() *genai.GenerateContentResponse {
	return &genai.GenerateContentResponse{
		Candidates:    []*genai.Candidate{{Content: &genai.Content{Role: genai.RoleModel, Parts: s.parts}, FinishReason: s.finish}},
		UsageMetadata: s.usage,
	}
}

func geminiResponse(resp *genai.GenerateContentResponse) (*ContentResponse, error) {
	if resp == nil || len(resp.Candidates) == 0 || resp.Candidates[0] == nil {
		return nil, ErrEmptyResponse
	}
	cand := resp.Candidates[0]
	choice := &ContentChoice{StopReason: string(cand.FinishReason), GenerationInfo: geminiUsage(resp.UsageMetadata)}
	var text strings.Builder
	if cand.Content != nil {
		for _, p := range cand.Content.Parts {
			switch {
			case p == nil || p.Thought:
				continue // thought summaries are not part of this interface
			case p.FunctionCall != nil:
				args, err := json.Marshal(p.FunctionCall.Args)
				if err != nil {
					return nil, fmt.Errorf("llms: gemini: encode tool input: %w", err)
				}
				if p.FunctionCall.Args == nil {
					args = []byte("{}")
				}
				id := p.FunctionCall.ID
				if id == "" {
					id = geminiLocalIDPrefix + strconv.Itoa(len(choice.ToolCalls))
				}
				tc := ToolCall{ID: id, Type: "function", FunctionCall: &FunctionCall{Name: p.FunctionCall.Name, Arguments: string(args)}, Signature: p.ThoughtSignature}
				choice.ToolCalls = append(choice.ToolCalls, tc)
				choice.Parts = append(choice.Parts, tc)
			case p.Text != "":
				text.WriteString(p.Text)
				choice.Parts = append(choice.Parts, TextContent{Text: p.Text})
			}
		}
	}
	choice.Content = text.String()
	if len(choice.Parts) == 0 {
		return nil, ErrEmptyResponse
	}
	return &ContentResponse{Choices: []*ContentChoice{choice}}, nil
}

// geminiUsage writes token counts under the OpenAI key family: Gemini's
// prompt count already includes cached tokens, and thinking tokens are billed
// as output.
func geminiUsage(u *genai.GenerateContentResponseUsageMetadata) map[string]any {
	if u == nil {
		return nil
	}
	return map[string]any{
		"PromptTokens":       int(u.PromptTokenCount + u.ToolUsePromptTokenCount),
		"CompletionTokens":   int(u.CandidatesTokenCount + u.ThoughtsTokenCount),
		"PromptCachedTokens": int(u.CachedContentTokenCount),
	}
}
