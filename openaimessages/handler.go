// Package openaimessages is the OpenAI Responses API handler for aiserver
// (providesFor OpenAI/messages).
package openaimessages

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"strings"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/openaiwire"
	"github.com/launchdarkly/go-ai-sdk/internal/sdkapi"
)

const (
	// PackageName is the Go import path, registered for `$ld:ai:sdk:info`.
	PackageName = "github.com/launchdarkly/go-ai-sdk/openaimessages"
	// PackageVersion is this package's release version.
	PackageVersion = "0.1.0" // x-release-please-version
	// SpanName is the root span name for both the blocking and streaming paths.
	SpanName = "openai.messages"
	// maxSteps bounds the tool loop, matching the JavaScript and Python handlers.
	maxSteps = 10
)

// ErrMaxSteps is returned when the model keeps requesting tools past the
// tool-loop bound.
var ErrMaxSteps = errors.New("openaimessages: tool loop exceeded the maximum number of steps")

// ProvidesFor is the variation provider/mode this handler serves.
var ProvidesFor = aiserver.ProvidesFor{Provider: "OpenAI", Mode: aiserver.ModeMessages}

func init() { aiserver.RegisterAiSdkPackage(PackageName, PackageVersion) }

// Options configure NewHandler.
type Options struct {
	// Client is used for every call when set. Otherwise a client is built from
	// APIKey, falling back to the OPENAI_API_KEY environment variable.
	Client *openai.Client
	// APIKey is used only when Client is nil.
	APIKey string
	// CaptureContent gates prompt, completion, tool and judge-explanation
	// content on spans. CaptureDefault follows aiserver.Configure, else on.
	CaptureContent aiserver.ContentCapture
}

// NewHandler returns the OpenAI messages handler (createOpenAIMessagesHandler).
func NewHandler(opts Options) *aiserver.ProviderHandler {
	aiserver.ReportUsage("openai-messages.createOpenAIMessagesHandler", PackageName, PackageVersion)
	return newHandler(opts)
}

func newHandler(opts Options) *aiserver.ProviderHandler {
	c := opts.Client
	if c == nil {
		var reqOpts []option.RequestOption
		if opts.APIKey != "" {
			reqOpts = append(reqOpts, option.WithAPIKey(opts.APIKey))
		}
		nc := openai.NewClient(reqOpts...)
		c = &nc
	}
	h := &handler{client: c}
	return sdkapi.CreateHandler(ProvidesFor, h.invoke, h.stream, opts.CaptureContent)
}

// InvokeOptions are the optional arguments to OpenAIMessages.
type InvokeOptions struct {
	ToolHandlers map[string]aiserver.ToolHandler
	Registry     *aiserver.Registry
	SkipJudges   bool
	// Variables are passed to Invoke, not to the config.
	Variables map[string]any
	// Handler configures the handler built for this call.
	Handler Options
}

// OpenAIMessages invokes configKey through a fresh OpenAI messages handler.
// opts may be nil.
func OpenAIMessages(ctx context.Context, configKey, userInput string, userCtx ldcontext.Context, opts *InvokeOptions) (aiserver.ProviderResponse, error) {
	aiserver.ReportUsage("openai-messages.openaiMessages", PackageName, PackageVersion)
	if opts == nil {
		opts = &InvokeOptions{}
	}
	return sdkapi.Config(aiserver.ConfigArgs{
		Key: configKey, Handlers: []*aiserver.ProviderHandler{newHandler(opts.Handler)},
		ToolHandlers: opts.ToolHandlers, Registry: opts.Registry, SkipJudges: opts.SkipJudges,
	}).Invoke(ctx, userInput, userCtx, aiserver.WithVariables(opts.Variables))
}

type handler struct{ client *openai.Client }

// toInput maps the composed prompt to Responses input items. The system
// prompt is a leading system message, as in the JavaScript handler.
func toInput(p aiserver.Prompt) responses.ResponseInputParam {
	var out responses.ResponseInputParam
	if p.System != "" {
		out = append(out, responses.ResponseInputItemParamOfMessage(p.System, responses.EasyInputMessageRoleSystem))
	}
	return append(out, openaiwire.Turns(p.Turns)...)
}

// forwardedParameters are the config.model.parameters keys sent verbatim
// (the JavaScript handler's allowlist). Handler-owned fields (model, input,
// tools, previous_response_id, text) and ones that would break the call
// (stream, background, conversation, prompt) are deliberately absent.
var forwardedParameters = []string{
	"context_management", "include", "instructions", "max_output_tokens", "metadata", "moderation",
	"parallel_tool_calls", "prompt_cache_key", "prompt_cache_retention", "reasoning", "safety_identifier",
	"service_tier", "store", "temperature", "tool_choice", "top_logprobs", "top_p", "truncation", "user",
}

func newParams(cfg aiserver.AiConfig, input responses.ResponseInputParam) responses.ResponseNewParams {
	p := responses.ResponseNewParams{
		Model: cfg.Model.Name,
		Input: responses.ResponseNewParamsInputUnion{OfInputItemList: input},
	}
	if extra := openaiwire.ModelParameters(cfg.Model.Parameters, forwardedParameters); len(extra) > 0 {
		p.SetExtraFields(extra)
	}
	return p
}

// runTools executes the function calls in r. A call for a tool that was not
// offered is an error, never a lookup in the raw handler map: the model must
// not reach a handler the active config excluded.
func runTools(r *aiserver.HandlerRun, resp *responses.Response, callable map[string]aiserver.Tool, handlers map[string]aiserver.ToolHandler) (responses.ResponseInputParam, error) {
	var outputs responses.ResponseInputParam
	for _, item := range resp.Output {
		if item.Type != "function_call" {
			continue
		}
		call := item.AsFunctionCall()
		span := r.StartTool(call.Name, call.CallID)
		r.ToolArguments(span, call.Arguments)
		res, err := callTool(r.Context(), call, callable, handlers)
		if err != nil {
			return nil, err // Fail ends the open tool span
		}
		r.ToolResult(span, res)
		r.EndTool(span, nil)
		outputs = append(outputs, openaiwire.FunctionCallOutput(call.CallID, aiserver.ToolResultString(res)))
	}
	return outputs, nil
}

func callTool(ctx context.Context, call responses.ResponseFunctionToolCall, callable map[string]aiserver.Tool, handlers map[string]aiserver.ToolHandler) (any, error) {
	fn, _ := handlers[call.Name].(aiserver.ToolFunc)
	if _, offered := callable[call.Name]; !offered || fn == nil {
		return nil, fmt.Errorf("openaimessages: no handler registered for tool %q", call.Name)
	}
	args, err := openaiwire.ParseArguments(call.Arguments)
	if err != nil {
		return nil, fmt.Errorf("openaimessages: tool %q arguments: %w", call.Name, err)
	}
	res, err := fn(ctx, args)
	if err != nil {
		return nil, fmt.Errorf("openaimessages: tool %q: %w", call.Name, err) // propagate
	}
	return res, nil
}

// run is the tool loop shared by both paths; onText is nil when blocking.
// abandoned reports that the stream consumer stopped iterating.
func (h *handler) run(ctx context.Context, req aiserver.HandlerRequest, onText func(string) bool) (out string, usage openaiwire.RunUsage, abandoned bool, err error) {
	model := req.Config.Model.Name
	p := aiserver.ComposePrompt(req, aiserver.ModeMessages)
	r := aiserver.StartHandlerRun(ctx, aiserver.HandlerSpanOptions{
		Tracer: PackageName, Name: SpanName, System: "openai", RequestModel: model,
	}, req, p)
	defer r.EndOnPanic()

	callable := aiserver.CallableTools(req.Config, req.ToolHandlers)
	params := newParams(req.Config, toInput(p))
	params.Tools = openaiwire.FunctionTools(callable)
	if onText == nil && req.Config.OutputFormat != nil { // streaming ignores outputFormat
		params.Text = responses.ResponseTextConfigParam{Format: responses.ResponseFormatTextConfigUnionParam{
			OfJSONSchema: &responses.ResponseFormatTextJSONSchemaConfigParam{
				Name: "output", Schema: req.Config.OutputFormat, Strict: param.NewOpt(false),
			},
		}}
	}

	var streamed strings.Builder
	lastModel := model
	for step := 0; ; step++ {
		chat := r.StartChat(model)
		resp, stopped, terr := h.turn(r.Context(), params, onText, &streamed)
		if stopped {
			r.Abandon()
			return "", usage, true, nil
		}
		if terr != nil {
			r.Fail(terr)
			return "", usage, false, terr
		}
		usage.Add(resp.Usage)
		lastModel = openaiwire.ResponseModel(resp, model)
		r.EndChat(chat, lastModel, openaiwire.Usage(resp.Usage), openaiwire.FinishReason(resp))

		if openaiwire.FinishReason(resp) != "tool_calls" {
			out = resp.OutputText()
			break
		}
		if step >= maxSteps {
			terr = fmt.Errorf("%w (%d)", ErrMaxSteps, maxSteps)
			r.Fail(terr)
			return "", usage, false, terr
		}
		outputs, terr := runTools(r, resp, callable, req.ToolHandlers)
		if terr != nil {
			r.Fail(terr)
			return "", usage, false, terr
		}
		// previous_response_id carries the conversation but not tools or
		// text.format (verified live: the follow-up response lists no tools and
		// the model writes tool calls as text), so both are sent again.
		tools, text := params.Tools, params.Text
		params = newParams(req.Config, outputs)
		params.PreviousResponseID = param.NewOpt(resp.ID)
		params.Tools, params.Text = tools, text
	}
	if onText != nil {
		out = streamed.String() // done.output spans every streamed turn
	}
	r.Finish(lastModel, out)
	return out, usage, false, nil
}

// turn performs one provider call. When onText is set it streams, appending
// each text delta to streamed and passing it to onText; stopped reports that
// onText returned false.
func (h *handler) turn(ctx context.Context, params responses.ResponseNewParams, onText func(string) bool, streamed *strings.Builder) (resp *responses.Response, stopped bool, err error) {
	if onText == nil {
		resp, err := h.client.Responses.New(ctx, params)
		if err != nil {
			return nil, false, fmt.Errorf("openaimessages: responses.create: %w", err)
		}
		return resp, false, nil
	}
	resp, stopped, err = openaiwire.StreamTurn(ctx, h.client, params, func(t string) bool {
		streamed.WriteString(t)
		return onText(t)
	})
	if err != nil {
		return nil, false, fmt.Errorf("openaimessages: responses.stream: %w", err)
	}
	return resp, stopped, nil
}

func (h *handler) invoke(ctx context.Context, req aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
	out, usage, _, err := h.run(ctx, req, nil)
	if err != nil {
		return aiserver.HandlerResult{}, err
	}
	return aiserver.HandlerResult{Output: out, Usage: usage.Raw()}, nil
}

func (h *handler) stream(ctx context.Context, req aiserver.HandlerRequest) iter.Seq2[aiserver.HandlerStreamEvent, error] {
	return func(yield func(aiserver.HandlerStreamEvent, error) bool) {
		out, usage, abandoned, err := h.run(ctx, req, func(t string) bool {
			return yield(aiserver.HandlerStreamEvent{Type: aiserver.EventChunk, Text: t}, nil)
		})
		if abandoned {
			return
		}
		if err != nil {
			yield(aiserver.HandlerStreamEvent{}, err)
			return
		}
		yield(aiserver.HandlerStreamEvent{Type: aiserver.EventDone, Output: &out, Usage: usage.Raw()}, nil)
	}
}
