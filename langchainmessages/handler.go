// Package langchainmessages is the LangChain messages handler for aiserver
// (providesFor */messages), built on the SDK's llms package. One handler
// serves any provider: the config's provider.name picks a built-in llms model
// (Anthropic, Bedrock, or OpenAI Chat Completions for anything else), or the
// caller injects one.
package langchainmessages

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"strings"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/langchainwire"
	"github.com/launchdarkly/go-ai-sdk/llms"
)

const (
	// PackageName is this package's identity for $ld:ai:sdk:info and
	// $ld:ai:sdk:usage.
	PackageName = "github.com/launchdarkly/go-ai-sdk/langchainmessages"
	// PackageVersion is this package's release version.
	PackageVersion = "0.1.0" // x-release-please-version
	// SpanName is the blocking root span name.
	SpanName = "langchain.invoke"
	// StreamSpanName is the streaming root span name.
	StreamSpanName = "langchain.stream"
	// maxSteps bounds the tool loop; 10 matches the JavaScript handler.
	maxSteps = 10
)

// ProvidesFor is the variation provider/mode this handler serves: the '*'
// wildcard, because LangChain routes to any provider.
var ProvidesFor = aiserver.ProvidesFor{Provider: aiserver.WildcardProvider, Mode: aiserver.ModeMessages}

var (
	// ErrMaxSteps is returned when the model keeps requesting tools past the
	// loop bound.
	ErrMaxSteps = errors.New("langchainmessages: tool loop exceeded the maximum number of steps")
	// ErrNoToolHandler is returned when the model requests a tool that has no
	// registered ToolFunc.
	ErrNoToolHandler = langchainwire.ErrNoToolHandler
	// ErrEmptyResponse is returned when the model returns no choices.
	ErrEmptyResponse = langchainwire.ErrEmptyResponse
)

func init() { aiserver.RegisterAiSdkPackage(PackageName, PackageVersion) }

// ModelFactory builds the llms model for one call from the evaluated
// AI Config, so the model can see config.model.parameters and name.
type ModelFactory func(cfg aiserver.AiConfig) (llms.Model, error)

// Options configure NewHandler.
type Options struct {
	// Model is used for every call when set.
	Model llms.Model
	// ModelFactory is called per request when Model is nil. When both are nil
	// the handler builds a built-in llms model from config.provider.name:
	// "anthropic", "bedrock", or OpenAI for anything else.
	ModelFactory ModelFactory
	// CaptureContent is attached to the handler so the client core can gate
	// judge explanations on it.
	CaptureContent bool
}

// NewHandler returns the LangChain messages handler (createLangChainHandler).
func NewHandler(opts Options) *aiserver.ProviderHandler {
	aiserver.ReportUsage("langchain-messages.createLangChainHandler", PackageName, PackageVersion)
	return newHandler(opts)
}

func newHandler(opts Options) *aiserver.ProviderHandler {
	h := &handler{opts: opts}
	return aiserver.CreateHandlerInternal(ProvidesFor, h.invoke, h.stream, opts.CaptureContent)
}

// InvokeOptions are the optional arguments to LangChainMessages.
type InvokeOptions struct {
	ToolHandlers map[string]aiserver.ToolHandler
	Registry     *aiserver.Registry
	SkipJudges   bool
	// Variables are passed to Invoke, not to Config.
	Variables map[string]any
	// Handler configures the handler the wrapper builds.
	Handler Options
}

// LangChainMessages invokes configKey through a fresh LangChain messages
// handler. opts may be nil.
func LangChainMessages(ctx context.Context, configKey, userInput string, userCtx ldcontext.Context, opts *InvokeOptions) (aiserver.ProviderResponse, error) {
	aiserver.ReportUsage("langchain-messages.langchainMessages", PackageName, PackageVersion)
	if opts == nil {
		opts = &InvokeOptions{}
	}
	return aiserver.ConfigInternal(aiserver.ConfigArgs{
		Key: configKey, Handlers: []*aiserver.ProviderHandler{newHandler(opts.Handler)},
		ToolHandlers: opts.ToolHandlers, Registry: opts.Registry, SkipJudges: opts.SkipJudges,
	}).Invoke(ctx, userInput, userCtx, aiserver.WithVariables(opts.Variables))
}

type handler struct{ opts Options }

// normalizeSchema maps the OpenAI format descriptor `type: "json_schema"` to
// a JSON Schema object type, as the JavaScript handler does.
func normalizeSchema(schema map[string]any) map[string]any {
	if schema["type"] == "json_schema" && schema["properties"] != nil {
		out := make(map[string]any, len(schema))
		for k, v := range schema {
			out[k] = v
		}
		out["type"] = "object"
		return out
	}
	return schema
}

type runResult struct {
	text      string // final output text (blocking: last turn; streaming: all turns)
	parsed    any    // decoded structured output, nil when not applicable
	usage     aiserver.SpanUsage
	abandoned bool
}

func (r runResult) rawUsage() map[string]any {
	return map[string]any{"input_tokens": r.usage.Input, "output_tokens": r.usage.Output}
}

// run is the tool loop shared by both paths; onText is nil when blocking.
func (h *handler) run(ctx context.Context, req aiserver.HandlerRequest, onText func(string) bool) (runResult, error) {
	req.Config = langchainwire.ModelConfig(req.Config)
	cfg := req.Config
	streaming := onText != nil
	var schema map[string]any
	if !streaming && cfg.OutputFormat != nil { // streaming ignores outputFormat
		schema = normalizeSchema(cfg.OutputFormat)
	}
	tools, fns := langchainwire.BuildTools(cfg, req.ToolHandlers)
	provider := langchainwire.ProviderName(cfg)
	// schema alone is enforced for every provider; with tools, only
	// OpenAI binds a response format and the client parses the string.
	enforce := schema != nil && (len(tools) == 0 || provider == "openai")
	parse := schema != nil && len(tools) == 0

	p := aiserver.ComposePrompt(req, aiserver.ModeMessages)
	if enforce {
		p.System = aiserver.WithOutputFormat(p.System, schema)
	}
	name := SpanName
	if streaming {
		name = StreamSpanName
	}
	r := aiserver.StartHandlerRun(ctx, aiserver.HandlerSpanOptions{
		Tracer: PackageName, Name: name, System: provider, RequestModel: cfg.Model.Name,
	}, req, p)
	defer r.EndOnPanic()

	var res runResult
	var bindSchema map[string]any
	if enforce {
		bindSchema = schema
	}
	model, err := langchainwire.ResolveModel(r.Context(), h.opts.Model, h.opts.ModelFactory, cfg, bindSchema)
	if err != nil {
		r.Fail(err)
		return res, err
	}
	opts := langchainwire.CallOptions(cfg.Model.Parameters)
	if len(tools) > 0 {
		opts = append(opts, llms.WithTools(tools))
	}
	if enforce {
		opts = append(opts, llms.WithJSONMode())
	}

	msgs := langchainwire.ToMessages(p)
	var all strings.Builder
	for step := 0; ; step++ {
		chat := r.StartChat(cfg.Model.Name)
		resp, text, stopped, terr := langchainwire.Turn(r.Context(), model, msgs, opts, onText)
		if stopped {
			r.Abandon()
			res.abandoned = true
			return res, nil
		}
		if terr != nil {
			r.Fail(terr)
			return res, terr
		}
		u := langchainwire.ResponseUsage(resp)
		res.usage.Add(u)
		r.EndChat(chat, cfg.Model.Name, u, langchainwire.FinishReason(resp))
		all.WriteString(text)
		res.text = text

		calls := langchainwire.ResponseToolCalls(resp)
		if len(calls) == 0 {
			break
		}
		if step >= maxSteps {
			err := fmt.Errorf("%w (%d)", ErrMaxSteps, maxSteps)
			r.Fail(err)
			return res, err
		}
		msgs = append(msgs, resp.AssistantMessage())
		for _, tc := range calls {
			out, terr := langchainwire.RunTool(r.Context(), r, fns, tc)
			if terr != nil {
				r.Fail(terr)
				return res, terr
			}
			msgs = append(msgs, out)
		}
	}
	if streaming {
		res.text = all.String()
	}
	if parse {
		if v, ok := aiserver.ParseJSONWithPossibleFences(res.text); ok {
			res.parsed = v
		}
	}
	r.Finish(cfg.Model.Name, res.text)
	return res, nil
}

func (h *handler) invoke(ctx context.Context, req aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
	res, err := h.run(ctx, req, nil)
	if err != nil {
		return aiserver.HandlerResult{}, err
	}
	var out any = res.text
	if res.parsed != nil {
		out = res.parsed
	}
	return aiserver.HandlerResult{Output: out, Usage: res.rawUsage()}, nil
}

func (h *handler) stream(ctx context.Context, req aiserver.HandlerRequest) iter.Seq2[aiserver.HandlerStreamEvent, error] {
	return func(yield func(aiserver.HandlerStreamEvent, error) bool) {
		res, err := h.run(ctx, req, func(t string) bool {
			return yield(aiserver.HandlerStreamEvent{Type: aiserver.EventChunk, Text: t}, nil)
		})
		if res.abandoned {
			return
		}
		if err != nil {
			yield(aiserver.HandlerStreamEvent{}, err)
			return
		}
		yield(aiserver.HandlerStreamEvent{Type: aiserver.EventDone, Output: &res.text, Usage: res.rawUsage()}, nil)
	}
}
