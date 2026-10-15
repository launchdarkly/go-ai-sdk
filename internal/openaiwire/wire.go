// Package openaiwire holds the OpenAI Responses API mapping shared by the
// openaimessages and openaiagents handlers: input items, tool definitions,
// model-parameter forwarding, usage, finish reasons, and one streamed turn.
package openaiwire

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

// ErrNoFinalResponse is returned when a stream ends without a terminal
// response event.
var ErrNoFinalResponse = errors.New("stream ended without a final response")

// Turns maps conversation turns to Responses input messages. User
// content blocks become input_text / input_image parts; an assistant turn
// flattens to text, because input parts are user-only.
func Turns(turns []aiserver.Message) responses.ResponseInputParam {
	var out responses.ResponseInputParam
	for _, t := range turns {
		switch {
		case t.Role == "assistant":
			out = append(out, responses.ResponseInputItemParamOfMessage(aiserver.ContentText(t), responses.EasyInputMessageRoleAssistant))
		case t.Blocks == nil:
			out = append(out, responses.ResponseInputItemParamOfMessage(t.Text, responses.EasyInputMessageRoleUser))
		default:
			var parts responses.ResponseInputMessageContentListParam
			for _, b := range t.Blocks {
				switch {
				case b.Type == "text":
					parts = append(parts, responses.ResponseInputContentParamOfInputText(b.Text))
				case b.Type == "image" && b.Source != nil:
					img := responses.ResponseInputContentParamOfInputImage(responses.ResponseInputImageDetailAuto)
					img.OfInputImage.ImageURL = param.NewOpt(aiserver.ImageURL(*b.Source))
					parts = append(parts, img)
				}
			}
			out = append(out, responses.ResponseInputItemParamOfMessage(parts, responses.EasyInputMessageRoleUser))
		}
	}
	return out
}

// FunctionTools converts tools to non-strict function tools, in key order,
// with their JSON Schema unmodified.
func FunctionTools(tools map[string]aiserver.Tool) []responses.ToolUnionParam {
	var out []responses.ToolUnionParam
	for _, key := range slices.Sorted(maps.Keys(tools)) {
		t := tools[key]
		out = append(out, FunctionTool(key, t.Description, t.Parameters))
	}
	return out
}

// FunctionTool is one non-strict function tool definition.
func FunctionTool(name, description string, parameters map[string]any) responses.ToolUnionParam {
	ft := responses.FunctionToolParam{Name: name, Parameters: parameters, Strict: param.NewOpt(false)}
	if description != "" {
		ft.Description = param.NewOpt(description)
	}
	return responses.ToolUnionParam{OfFunction: &ft}
}

// HostedTool is a provider built-in tool `{type: toolType, ...extra}`, such
// as `web_search` or `file_search` with its `vector_store_ids`.
func HostedTool(toolType string, extra map[string]any) responses.ToolUnionParam {
	body := maps.Clone(extra)
	if body == nil {
		body = map[string]any{}
	}
	body["type"] = toolType
	// The union marshals whichever member is set; the override replaces that
	// member's JSON with body, so any hosted tool type can be sent.
	t := param.Override[responses.WebSearchToolParam](body)
	return responses.ToolUnionParam{OfWebSearch: &t}
}

// ModelParameters picks the allowed keys from config.model.parameters,
// renaming the Chat Completions token limits the LaunchDarkly UI offers.
// Precedence: max_output_tokens, then max_completion_tokens, then max_tokens.
// allow must not contain handler-owned fields.
func ModelParameters(in map[string]any, allow []string) map[string]any {
	out := map[string]any{}
	for _, k := range allow {
		if v, ok := in[k]; ok && v != nil {
			out[k] = v
		}
	}
	for _, k := range []string{"max_tokens", "max_completion_tokens", "max_output_tokens"} {
		if v, ok := in[k]; ok && v != nil {
			out["max_output_tokens"] = v
		}
	}
	return out
}

// Usage reports one response's usage for a span. input_tokens already
// includes cached tokens, so cache reads are detail and are not added again.
func Usage(u responses.ResponseUsage) aiserver.SpanUsage {
	return aiserver.SpanUsage{
		Input: float64(u.InputTokens), Output: float64(u.OutputTokens),
		CacheRead: float64(u.InputTokensDetails.CachedTokens),
	}
}

// RunUsage sums token counts over a run's responses.
type RunUsage struct{ Input, Output int64 }

// Add accumulates one response's usage.
func (u *RunUsage) Add(x responses.ResponseUsage) {
	u.Input += x.InputTokens
	u.Output += x.OutputTokens
}

// Raw is the handler-result usage. It omits cache keys: OpenAI input already
// includes them and aiserver.ParseUsage would fold them in a second time.
func (u RunUsage) Raw() map[string]any {
	return map[string]any{"input_tokens": float64(u.Input), "output_tokens": float64(u.Output)}
}

// FinishReason derives a semconv finish reason; the Responses API reports a
// run status rather than a per-message reason.
func FinishReason(r *responses.Response) string {
	for _, item := range r.Output {
		if item.Type == "function_call" {
			return "tool_calls"
		}
	}
	switch r.Status {
	case responses.ResponseStatusIncomplete:
		if r.IncompleteDetails.Reason == "max_output_tokens" {
			return "length"
		}
		return "content_filter"
	case responses.ResponseStatusCompleted:
		return "stop"
	}
	return ""
}

// ResponseModel is the model the API reports, or requested when absent.
func ResponseModel(r *responses.Response, requested string) string {
	if r.Model != "" {
		return r.Model
	}
	return requested
}

// FunctionCallOutput is the input item answering one function call.
func FunctionCallOutput(callID, output string) responses.ResponseInputItemUnionParam {
	out := responses.ResponseInputItemParamOfFunctionCallOutput(output)
	out.OfFunctionCallOutput.CallID = param.NewOpt(callID)
	return out
}

// ParseArguments decodes a function call's JSON argument object; blank
// arguments decode to nil.
func ParseArguments(raw string) (map[string]any, error) {
	if strings.TrimSpace(raw) == "" {
		return nil, nil
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(raw), &args); err != nil {
		return nil, err
	}
	return args, nil
}

// StreamTurn performs one streamed provider call, passing each text delta to
// onText. stopped reports that onText returned false; the HTTP body is
// released either way. Errors are not prefixed; callers wrap them.
func StreamTurn(ctx context.Context, client *openai.Client, params responses.ResponseNewParams, onText func(string) bool) (resp *responses.Response, stopped bool, err error) {
	s := client.Responses.NewStreaming(ctx, params)
	defer s.Close()
	var final *responses.Response
	for s.Next() {
		ev := s.Current()
		switch ev.Type {
		case "response.output_text.delta":
			if !onText(ev.Delta) {
				return nil, true, nil
			}
		case "response.completed", "response.incomplete":
			r := ev.Response
			final = &r
		case "response.failed":
			return nil, false, fmt.Errorf("response failed: %s", ev.Response.Error.Message)
		case "error":
			return nil, false, fmt.Errorf("%s %s", ev.Code, ev.Message)
		}
	}
	if err := s.Err(); err != nil {
		return nil, false, err
	}
	if final == nil {
		return nil, false, ErrNoFinalResponse
	}
	return final, false, nil
}
