// Package claudemessages is the Anthropic Messages API handler for aiserver
// (providesFor Anthropic/messages).
package claudemessages

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/anthropicwire"
	"github.com/launchdarkly/go-ai-sdk/internal/sdkapi"
)

const (
	PackageName    = "github.com/launchdarkly/go-ai-sdk/claudemessages"
	PackageVersion = "0.1.0" // x-release-please-version
	// SpanName is the documented root span name.
	SpanName = "claude.messages"
	// maxSteps bounds the tool loop. 10 matches the
	// JavaScript and Python handlers.
	maxSteps = 10
)

// ProvidesFor is the variation provider/mode this handler serves.
var ProvidesFor = aiserver.ProvidesFor{Provider: "Anthropic", Mode: aiserver.ModeMessages}

// ErrToolNotFound is returned when the model calls a tool the handler did not
// offer (no config tool with a ToolFunc under that name). The error message
// names the tool.
var ErrToolNotFound = errors.New("claudemessages: no handler registered for tool")

// ErrMaxSteps is returned when the model still asks for tools after the
// tool-loop bound (10 turns, as in the JavaScript and Python handlers).
var ErrMaxSteps = errors.New("claudemessages: tool loop exceeded the maximum number of steps")

func init() { aiserver.RegisterAiSdkPackage(PackageName, PackageVersion) }

// Options configure NewHandler.
type Options struct {
	// Client is used for every call when set. Otherwise a client is built from
	// APIKey, falling back to the ANTHROPIC_API_KEY environment variable.
	Client *anthropic.Client
	APIKey string
	// CaptureContent is attached to the handler so the client core can gate
	// judge explanations on it.
	CaptureContent bool
}

// NewHandler returns the Anthropic messages handler (createClaudeMessagesHandler).
func NewHandler(opts Options) *aiserver.ProviderHandler {
	aiserver.ReportUsage("claude-messages.createClaudeMessagesHandler", PackageName, PackageVersion)
	return newHandler(opts)
}

func newHandler(opts Options) *aiserver.ProviderHandler {
	c := opts.Client
	if c == nil {
		var reqOpts []option.RequestOption
		if opts.APIKey != "" {
			reqOpts = append(reqOpts, option.WithAPIKey(opts.APIKey))
		}
		nc := anthropic.NewClient(reqOpts...)
		c = &nc
	}
	h := &handler{client: c}
	return sdkapi.CreateHandler(ProvidesFor, h.invoke, h.stream, opts.CaptureContent)
}

// InvokeOptions are the optional arguments to ClaudeMessages.
type InvokeOptions struct {
	ToolHandlers map[string]aiserver.ToolHandler
	Registry     *aiserver.Registry
	SkipJudges   bool
	Variables    map[string]any
	Handler      Options
}

// ClaudeMessages invokes configKey through a fresh Anthropic messages handler.
// opts may be nil.
func ClaudeMessages(ctx context.Context, configKey, userInput string, userCtx ldcontext.Context, opts *InvokeOptions) (aiserver.ProviderResponse, error) {
	aiserver.ReportUsage("claude-messages.claudeMessages", PackageName, PackageVersion)
	if opts == nil {
		opts = &InvokeOptions{}
	}
	return sdkapi.Config(aiserver.ConfigArgs{
		Key: configKey, Handlers: []*aiserver.ProviderHandler{newHandler(opts.Handler)},
		ToolHandlers: opts.ToolHandlers, Registry: opts.Registry, SkipJudges: opts.SkipJudges,
	}).Invoke(ctx, userInput, userCtx, aiserver.WithVariables(opts.Variables))
}

// forwardedParameters are the config.model.parameters keys sent verbatim: the
// JavaScript handler's allowlist (claude-messages FORWARDED_MODEL_PARAMETER_KEYS).
// Handler-owned fields (model, messages, system, tools, max_tokens) and stream,
// which would fight the method the handler calls, are deliberately absent.
var forwardedParameters = []string{
	"cache_control", "container", "inference_geo", "metadata", "output_config", "service_tier",
	"stop_sequences", "temperature", "thinking", "tool_choice", "top_k", "top_p",
}

// modelParameters picks the forwarded parameters, after moving a top-level
// effort (the LaunchDarkly UI's shape) into output_config.effort (the Messages
// API's). An explicit output_config.effort wins over the moved value.
func modelParameters(params map[string]any) map[string]any {
	in := params
	if effort, ok := params["effort"]; ok && effort != nil {
		in = make(map[string]any, len(params))
		for k, v := range params {
			in[k] = v
		}
		merged := map[string]any{"effort": effort}
		if oc, ok := params["output_config"].(map[string]any); ok {
			for k, v := range oc {
				merged[k] = v
			}
		}
		in["output_config"] = merged
	}
	out := map[string]any{}
	for _, k := range forwardedParameters {
		if v, ok := in[k]; ok && v != nil {
			out[k] = v
		}
	}
	return out
}

type handler struct{ client *anthropic.Client }

// run is the tool loop shared by both paths; onText is nil when blocking.
// abandoned reports that the stream consumer stopped iterating. out is the
// last response's text when blocking and every turn's text when
// streaming.
func (h *handler) run(ctx context.Context, req aiserver.HandlerRequest, onText func(string) bool) (out string, usage anthropicwire.Usage, abandoned bool, err error) {
	model := req.Config.Model.Name
	p := aiserver.ComposePrompt(req, aiserver.ModeMessages)
	if onText == nil { // streaming ignores outputFormat
		p.System = aiserver.WithOutputFormat(p.System, req.Config.OutputFormat)
	}
	r := aiserver.StartHandlerRun(ctx, aiserver.HandlerSpanOptions{
		Tracer: PackageName, Name: SpanName, System: "anthropic", RequestModel: model,
	}, req, p)
	defer r.EndOnPanic()

	params := anthropic.MessageNewParams{
		Model: anthropic.Model(model), MaxTokens: anthropicwire.MaxTokens(req.Config),
		Messages: anthropicwire.Messages(p.Turns), Tools: anthropicwire.Tools(req.Config, req.ToolHandlers),
	}
	if extra := modelParameters(req.Config.Model.Parameters); len(extra) > 0 {
		params.SetExtraFields(extra)
	}
	if p.System != "" {
		params.System = []anthropic.TextBlockParam{{Text: p.System}}
	}

	var final, all strings.Builder
	finished := false
	for step := range maxSteps + 1 {
		chat := r.StartChat(model)
		msg, stopped, terr := h.turn(r.Context(), params, onText)
		if stopped {
			usage.Add(msg.Usage)
			r.Abandon()
			return "", usage, true, nil
		}
		if terr != nil {
			r.Fail(terr)
			return "", usage, false, terr
		}
		usage.Add(msg.Usage)
		r.EndChat(chat, model, anthropicwire.SpanUsage(msg.Usage), anthropicwire.FinishReason(string(msg.StopReason)))
		if step == maxSteps && msg.StopReason == anthropic.StopReasonToolUse {
			break // maxSteps tool turns ran; fail before running more, as JS does
		}

		var results []anthropic.ContentBlockParamUnion
		for _, b := range msg.Content {
			switch b.Type {
			case "text":
				final.WriteString(b.Text)
				all.WriteString(b.Text)
			case "tool_use":
				tool := r.StartTool(b.Name, b.ID) // opened first so a missing tool shows as a failed call, as in JS
				fn, _ := req.ToolHandlers[b.Name].(aiserver.ToolFunc)
				if _, offered := req.Config.Tools[b.Name]; !offered || fn == nil {
					err := fmt.Errorf("%w: %q", ErrToolNotFound, b.Name)
					r.Fail(err)
					return "", usage, false, err
				}
				var args map[string]any
				_ = json.Unmarshal(b.Input, &args)
				res, ferr := fn(r.Context(), args)
				if ferr != nil {
					r.Fail(ferr)
					return "", usage, false, ferr // tool errors propagate
				}
				r.EndTool(tool, nil)
				results = append(results, anthropic.NewToolResultBlock(b.ID, aiserver.ToolResultString(res), false))
			}
		}
		if msg.StopReason != anthropic.StopReasonToolUse || len(results) == 0 {
			finished = true
			break
		}
		params.Messages = append(params.Messages, msg.ToParam(), anthropic.NewUserMessage(results...))
		final.Reset() // the blocking result is the last response's text
	}
	if !finished {
		r.Fail(ErrMaxSteps)
		return "", usage, false, ErrMaxSteps
	}
	out = final.String()
	if onText != nil {
		out = all.String()
	}
	r.Finish(model, out)
	return out, usage, false, nil
}

// turn performs one provider call. When onText is set it streams, calling
// onText per text delta; stopped reports that onText returned false.
func (h *handler) turn(ctx context.Context, params anthropic.MessageNewParams, onText func(string) bool) (msg anthropic.Message, stopped bool, err error) {
	if onText == nil {
		m, err := h.client.Messages.New(ctx, params)
		if err != nil {
			return anthropic.Message{}, false, err
		}
		return *m, false, nil
	}
	s := h.client.Messages.NewStreaming(ctx, params)
	defer s.Close() // releases the HTTP body on early exit
	for s.Next() {
		ev := s.Current()
		if err := msg.Accumulate(ev); err != nil {
			return msg, false, err
		}
		if ev.Type == "content_block_delta" && ev.Delta.Type == "text_delta" && !onText(ev.Delta.Text) {
			return msg, true, nil
		}
	}
	return msg, false, s.Err()
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
