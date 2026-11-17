// Package chatwire holds the OpenAI-compatible Chat Completions mapping shared
// by the handlers that talk to a Chat Completions proxy or gateway: messages,
// tool definitions, model-parameter forwarding, the structured-output format,
// usage, finish reasons, tool-call turns and one streamed turn.
package chatwire

import (
	"context"
	"errors"
	"maps"
	"slices"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/shared"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/handlerkit"
	"github.com/launchdarkly/go-ai-sdk/internal/kitbase"
)

// ErrNoChoices is returned when a response or stream carries no choice.
var ErrNoChoices = errors.New("response has no choices")

// Messages maps the system prompt and conversation turns to Chat Completions
// messages, keeping each turn's role. A non-empty system comes first. Text
// content stays a string; content blocks become text and image_url parts (a
// URL, or a data: URI for base64 data), and an assistant turn keeps only its
// text parts.
func Messages(system string, turns []aiserver.Message) []openai.ChatCompletionMessageParamUnion {
	var out []openai.ChatCompletionMessageParamUnion
	if system != "" {
		out = append(out, openai.SystemMessage(system))
	}
	for _, t := range turns {
		out = append(out, message(t))
	}
	return out
}

func message(t aiserver.Message) openai.ChatCompletionMessageParamUnion {
	switch t.Role {
	case "assistant":
		if t.Blocks == nil {
			return openai.AssistantMessage(t.Text)
		}
		var parts []openai.ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion
		for _, b := range t.Blocks {
			if b.Type == "text" {
				parts = append(parts, openai.ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion{
					OfText: &openai.ChatCompletionContentPartTextParam{Text: b.Text},
				})
			}
		}
		return openai.AssistantMessage(parts)
	case "system":
		if t.Blocks == nil {
			return openai.SystemMessage(t.Text)
		}
		var parts []openai.ChatCompletionContentPartTextParam
		for _, b := range t.Blocks {
			if b.Type == "text" {
				parts = append(parts, openai.ChatCompletionContentPartTextParam{Text: b.Text})
			}
		}
		return openai.SystemMessage(parts)
	default:
		if t.Blocks == nil {
			return openai.UserMessage(t.Text)
		}
		var parts []openai.ChatCompletionContentPartUnionParam
		for _, b := range t.Blocks {
			switch {
			case b.Type == "text":
				parts = append(parts, openai.TextContentPart(b.Text))
			case b.Type == "image" && b.Source != nil:
				parts = append(parts, openai.ImageContentPart(openai.ChatCompletionContentPartImageImageURLParam{URL: handlerkit.ImageURL(*b.Source)}))
			}
		}
		return openai.UserMessage(parts)
	}
}

// FunctionTools converts tools to function tools, in key order, with their
// JSON Schema unmodified. No tools map to nil, so the request omits tools.
func FunctionTools(tools map[string]aiserver.Tool) []openai.ChatCompletionToolUnionParam {
	var out []openai.ChatCompletionToolUnionParam
	for _, name := range slices.Sorted(maps.Keys(tools)) {
		out = append(out, FunctionTool(name, tools[name].Description, tools[name].Parameters))
	}
	return out
}

// FunctionTool is one function tool definition. The description is always
// sent, "" when unset, and strict is not.
func FunctionTool(name, description string, parameters map[string]any) openai.ChatCompletionToolUnionParam {
	return openai.ChatCompletionFunctionTool(shared.FunctionDefinitionParam{
		Name:        name,
		Description: param.NewOpt(description),
		Parameters:  parameters,
	})
}

// ModelParameters picks the allowed, non-nil keys from config.model.parameters
// unchanged. allow must not contain handler-owned or connection keys.
func ModelParameters(in map[string]any, allow []string) map[string]any {
	out := map[string]any{}
	for _, k := range allow {
		if v, ok := in[k]; ok && v != nil {
			out[k] = v
		}
	}
	return out
}

// ResponseFormat is the structured-output format for schema:
// {type: json_schema, json_schema: {name: "output", strict: false, schema}}.
func ResponseFormat(schema map[string]any) openai.ChatCompletionNewParamsResponseFormatUnion {
	return openai.ChatCompletionNewParamsResponseFormatUnion{
		OfJSONSchema: &shared.ResponseFormatJSONSchemaParam{
			JSONSchema: shared.ResponseFormatJSONSchemaJSONSchemaParam{
				Name:   "output",
				Strict: param.NewOpt(false),
				Schema: schema,
			},
		},
	}
}

// ToolCall is one function call the model requested. Arguments is the raw
// JSON string the model sent.
type ToolCall struct {
	ID, Name, Arguments string
}

// Turn is one model call's result, from choices[0].
type Turn struct {
	Text      string
	ToolCalls []ToolCall
	Usage     kitbase.SpanUsage
	// Model is the model the API reported, or "" when it reported none.
	Model string
	// Finish is the finish reason the API reported.
	Finish string
}

// TurnOf reads a blocking response. A response with no choice returns
// ErrNoChoices.
func TurnOf(resp *openai.ChatCompletion) (Turn, error) {
	if resp == nil || len(resp.Choices) == 0 {
		return Turn{}, ErrNoChoices
	}
	c := resp.Choices[0]
	t := Turn{Text: c.Message.Content, Usage: Usage(resp.Usage), Model: resp.Model, Finish: c.FinishReason}
	for _, tc := range c.Message.ToolCalls {
		if tc.Type == "function" {
			t.ToolCalls = append(t.ToolCalls, ToolCall{ID: tc.ID, Name: tc.Function.Name, Arguments: tc.Function.Arguments})
		}
	}
	return t, nil
}

// Usage reports one call's usage for a span. prompt_tokens already includes
// cached tokens, so cache reads are detail and are not added again. A
// response without usage is Unreported.
func Usage(u openai.CompletionUsage) kitbase.SpanUsage {
	return kitbase.SpanUsage{
		Input: float64(u.PromptTokens), Output: float64(u.CompletionTokens),
		CacheRead:  float64(u.PromptTokensDetails.CachedTokens),
		Unreported: !u.JSON.PromptTokens.Valid(),
	}
}

// AssistantMessage is the assistant message that replays t: its text (null
// when empty) and one tool_calls entry per requested call.
func AssistantMessage(t Turn) openai.ChatCompletionMessageParamUnion {
	m := openai.ChatCompletionAssistantMessageParam{}
	if t.Text != "" {
		m.Content.OfString = param.NewOpt(t.Text)
	} else {
		m.Content.OfString = param.Null[string]()
	}
	for _, c := range t.ToolCalls {
		m.ToolCalls = append(m.ToolCalls, openai.ChatCompletionMessageToolCallUnionParam{
			OfFunction: &openai.ChatCompletionMessageFunctionToolCallParam{
				ID:       c.ID,
				Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{Name: c.Name, Arguments: c.Arguments},
			},
		})
	}
	return openai.ChatCompletionMessageParamUnion{OfAssistant: &m}
}

// ToolMessage is the role: tool message answering the call callID.
func ToolMessage(callID, output string) openai.ChatCompletionMessageParamUnion {
	return openai.ToolMessage(output, callID)
}

// StreamTurn performs one streamed call with usage requested in the final
// chunk, passing each text delta to onText. Tool-call fragments are joined by
// index. stopped reports that onText returned false; the HTTP body is
// released either way. Errors are not prefixed; callers wrap them.
func StreamTurn(ctx context.Context, client *openai.Client, params openai.ChatCompletionNewParams, onText func(string) bool) (turn Turn, stopped bool, err error) {
	params.StreamOptions = openai.ChatCompletionStreamOptionsParam{IncludeUsage: param.NewOpt(true)}
	s := client.Chat.Completions.NewStreaming(ctx, params)
	defer s.Close()
	var acc openai.ChatCompletionAccumulator
	// The accumulator keeps usage counts but not whether usage was sent.
	var usage openai.CompletionUsage
	for s.Next() {
		ch := s.Current()
		acc.AddChunk(ch)
		if ch.JSON.Usage.Valid() {
			usage = ch.Usage
		}
		for _, c := range ch.Choices {
			if c.Index == 0 && c.Delta.Content != "" && !onText(c.Delta.Content) {
				return Turn{}, true, nil
			}
		}
	}
	if err := s.Err(); err != nil {
		return Turn{}, false, err
	}
	if turn, err = TurnOf(&acc.ChatCompletion); err != nil {
		return Turn{}, false, err
	}
	turn.Usage = Usage(usage)
	return turn, false, nil
}
