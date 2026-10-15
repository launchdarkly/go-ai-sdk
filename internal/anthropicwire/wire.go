// Package anthropicwire holds the Anthropic Messages API mapping shared by the
// claudemessages and claudeagents handlers: LaunchDarkly turns to message
// params, config tools to tool params, max_tokens, finish
// reasons, and usage folding.
package anthropicwire

import (
	"strings"

	"github.com/anthropics/anthropic-sdk-go"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

// DefaultMaxTokens is sent when the config sets no max_tokens; the Messages
// API requires the field.
const DefaultMaxTokens = 1024

// Blocks maps one LaunchDarkly message to Anthropic content blocks. The
// canonical image source already has Anthropic's shape, so base64 and URL
// images map across unchanged.
func Blocks(m aiserver.Message) []anthropic.ContentBlockParamUnion {
	if m.Blocks == nil {
		return []anthropic.ContentBlockParamUnion{anthropic.NewTextBlock(m.Text)}
	}
	out := make([]anthropic.ContentBlockParamUnion, 0, len(m.Blocks))
	for _, b := range m.Blocks {
		switch {
		case b.Type == "text":
			out = append(out, anthropic.NewTextBlock(b.Text))
		case b.Type == "image" && b.Source != nil && b.Source.Type == "base64":
			out = append(out, anthropic.NewImageBlockBase64(b.Source.MediaType, b.Source.Data))
		case b.Type == "image" && b.Source != nil && b.Source.Type == "url":
			out = append(out, anthropic.NewImageBlock(anthropic.URLImageSourceParam{URL: b.Source.URL}))
		}
	}
	return out
}

// Messages maps composed turns to Anthropic message params.
func Messages(turns []aiserver.Message) []anthropic.MessageParam {
	out := make([]anthropic.MessageParam, 0, len(turns))
	for _, t := range turns {
		if t.Role == "assistant" {
			out = append(out, anthropic.NewAssistantMessage(Blocks(t)...))
		} else {
			out = append(out, anthropic.NewUserMessage(Blocks(t)...))
		}
	}
	return out
}

// FunctionTool builds a custom tool param, rebuilding the JSON Schema
// field-for-field so it reaches the API unmodified.
func FunctionTool(name, description string, parameters map[string]any) anthropic.ToolUnionParam {
	schema := anthropic.ToolInputSchemaParam{ExtraFields: map[string]any{}}
	for k, v := range parameters {
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
			}
		default:
			schema.ExtraFields[k] = v
		}
	}
	tp := anthropic.ToolParam{Name: name, InputSchema: schema}
	if description != "" {
		tp.Description = anthropic.String(description)
	}
	return anthropic.ToolUnionParam{OfTool: &tp}
}

// Tools forwards config tools that have a local ToolFunc, named by
// their config map key.
func Tools(cfg aiserver.AiConfig, handlers map[string]aiserver.ToolHandler) []anthropic.ToolUnionParam {
	var out []anthropic.ToolUnionParam
	for key, t := range aiserver.CallableTools(cfg, handlers) {
		out = append(out, FunctionTool(key, t.Description, t.Parameters))
	}
	return out
}

// MaxTokens reads max_tokens / maxTokens from the model parameters.
func MaxTokens(cfg aiserver.AiConfig) int64 {
	for _, k := range []string{"max_tokens", "maxTokens"} {
		if v, ok := cfg.Model.Parameters[k].(float64); ok && v > 0 {
			return int64(v)
		}
	}
	return DefaultMaxTokens
}

var finishReasons = map[string]string{
	"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length", "tool_use": "tool_calls", "refusal": "content_filter",
}

// FinishReason maps an Anthropic stop reason to the OTel semconv value;
// unknown reasons pass through.
func FinishReason(r string) string {
	if m, ok := finishReasons[strings.ToLower(r)]; ok {
		return m
	}
	return r
}

// Usage keeps Anthropic's own field names with cache buckets unfolded: it is
// a handler's return value and aiserver.ParseUsage folds once.
type Usage struct{ Input, Output, CacheRead, CacheCreation int64 }

// Add accumulates one response's usage.
func (u *Usage) Add(x anthropic.Usage) {
	u.Input += x.InputTokens
	u.Output += x.OutputTokens
	u.CacheRead += x.CacheReadInputTokens
	u.CacheCreation += x.CacheCreationInputTokens
}

// Raw is the usage map a handler returns.
func (u Usage) Raw() map[string]any {
	return map[string]any{
		"input_tokens": float64(u.Input), "output_tokens": float64(u.Output),
		"cache_read_input_tokens": float64(u.CacheRead), "cache_creation_input_tokens": float64(u.CacheCreation),
	}
}

// SpanUsage is one response's span usage, cache-inclusive input.
func SpanUsage(x anthropic.Usage) aiserver.SpanUsage {
	return aiserver.SpanUsage{
		Input:     float64(x.InputTokens + x.CacheReadInputTokens + x.CacheCreationInputTokens),
		Output:    float64(x.OutputTokens),
		CacheRead: float64(x.CacheReadInputTokens), CacheCreation: float64(x.CacheCreationInputTokens),
	}
}
