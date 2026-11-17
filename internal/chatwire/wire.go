// Package chatwire holds the OpenAI-compatible Chat Completions mapping shared
// by the handlers that talk to a Chat Completions proxy or gateway: messages,
// tool definitions, model-parameter forwarding, the structured-output format,
// usage, finish reasons, tool-call turns and one streamed turn.
package chatwire

import (
	"context"
	"errors"

	"github.com/openai/openai-go/v3"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/kitbase"
)

// ErrNoChoices is returned when a response or stream carries no choice.
var ErrNoChoices = errors.New("response has no choices")

var errNotImplemented = errors.New("chatwire: not implemented yet")

// Messages maps the system prompt and conversation turns to Chat Completions
// messages, keeping each turn's role. A non-empty system comes first. Text
// content stays a string; content blocks become text and image_url parts (a
// URL, or a data: URI for base64 data), and an assistant turn keeps only its
// text parts.
func Messages(system string, turns []aiserver.Message) []openai.ChatCompletionMessageParamUnion {
	return nil
}

// FunctionTools converts tools to function tools, in key order, with their
// JSON Schema unmodified. No tools map to nil, so the request omits tools.
func FunctionTools(tools map[string]aiserver.Tool) []openai.ChatCompletionToolUnionParam {
	return nil
}

// FunctionTool is one function tool definition. The description is always
// sent, "" when unset, and strict is not.
func FunctionTool(name, description string, parameters map[string]any) openai.ChatCompletionToolUnionParam {
	return openai.ChatCompletionToolUnionParam{}
}

// ModelParameters picks the allowed, non-nil keys from config.model.parameters
// unchanged. allow must not contain handler-owned or connection keys.
func ModelParameters(in map[string]any, allow []string) map[string]any {
	return nil
}

// ResponseFormat is the structured-output format for schema:
// {type: json_schema, json_schema: {name: "output", strict: false, schema}}.
func ResponseFormat(schema map[string]any) openai.ChatCompletionNewParamsResponseFormatUnion {
	return openai.ChatCompletionNewParamsResponseFormatUnion{}
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
	return Turn{}, errNotImplemented
}

// Usage reports one call's usage for a span. prompt_tokens already includes
// cached tokens, so cache reads are detail and are not added again. A
// response without usage is Unreported.
func Usage(u openai.CompletionUsage) kitbase.SpanUsage {
	return kitbase.SpanUsage{}
}

// AssistantMessage is the assistant message that replays t: its text (null
// when empty) and one tool_calls entry per requested call.
func AssistantMessage(t Turn) openai.ChatCompletionMessageParamUnion {
	return openai.ChatCompletionMessageParamUnion{}
}

// ToolMessage is the role: tool message answering the call callID.
func ToolMessage(callID, output string) openai.ChatCompletionMessageParamUnion {
	return openai.ChatCompletionMessageParamUnion{}
}

// StreamTurn performs one streamed call with usage requested in the final
// chunk, passing each text delta to onText. Tool-call fragments are joined by
// index. stopped reports that onText returned false; the HTTP body is
// released either way. Errors are not prefixed; callers wrap them.
func StreamTurn(ctx context.Context, client *openai.Client, params openai.ChatCompletionNewParams, onText func(string) bool) (turn Turn, stopped bool, err error) {
	return Turn{}, false, errNotImplemented
}
