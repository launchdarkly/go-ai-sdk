// Package langchainagents is the LangChain agent-mode handler for aiserver
// (providesFor */agent), plus the LangChainGraph wrapper and the ToLangGraph
// native graph adapter.
//
// Models, messages and tool calling come from github.com/tmc/langchaingo.
// The agent is a ReAct loop on internal/stategraph, the Go equivalent of
// create_react_agent.
package langchainagents

import (
	"context"
	"errors"
	"iter"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/tmc/langchaingo/llms"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/langchainwire"
	"github.com/launchdarkly/go-ai-sdk/internal/stategraph"
)

const (
	// PackageName is this package's identity for $ld:ai:sdk:info and
	// $ld:ai:sdk:usage.
	PackageName = "github.com/launchdarkly/go-ai-sdk/langchainagents"
	// PackageVersion is this package's release version.
	PackageVersion = "0.1.0" // x-release-please-version
	// SpanName is the handler's root span. There is no separate
	// streaming span name, so streaming uses the same name.
	SpanName = "langchain.agent"
	// operation is gen_ai.operation.name on the root span.
	operation = "invoke_agent"
)

// ProvidesFor is the variation provider/mode this handler serves: the '*'
// wildcard, because LangChain routes to any provider.
var ProvidesFor = aiserver.ProvidesFor{Provider: aiserver.WildcardProvider, Mode: aiserver.ModeAgent}

var (
	// ErrRecursionLimit is returned when an agent or graph run executes more
	// than 25 nodes (LangGraph's default recursion limit) without finishing.
	ErrRecursionLimit = errors.New("langchainagents: recursion limit reached without hitting a stop condition")
	// ErrNoToolHandler is returned when the model requests a tool that has no
	// registered handler.
	ErrNoToolHandler = langchainwire.ErrNoToolHandler
	// ErrEmptyResponse is returned when the model returns no choices.
	ErrEmptyResponse = langchainwire.ErrEmptyResponse
	// ErrNoRoot is returned by the native graph runner for a graph without a
	// root node.
	ErrNoRoot = errors.New("langchainagents: agent graph has no root node")
	// ErrNilGraph is returned by the native graph runner for a nil definition.
	ErrNilGraph = errors.New("langchainagents: agent graph definition is nil")
	// ErrNodeName is returned by the native graph runner when a node key,
	// sanitized for stategraph, is END or matches another node's name.
	ErrNodeName = errors.New("langchainagents: node name collision")
)

func init() { aiserver.RegisterAiSdkPackage(PackageName, PackageVersion) }

// ModelFactory builds the langchaingo model for one call from the evaluated
// AI Config, so the model can see config.model.parameters and name.
type ModelFactory func(cfg aiserver.AiConfig) (llms.Model, error)

// Options configure NewHandler (the model to call plus
// captureContent).
type Options struct {
	// Model is used for every call when set.
	Model llms.Model
	// ModelFactory is called per request when Model is nil. When both are nil
	// the handler builds a langchaingo model from config.provider.name:
	// "anthropic", "bedrock", or OpenAI for anything else.
	ModelFactory ModelFactory
	// CaptureContent is attached to the handler so the client core can gate
	// judge explanations on it.
	CaptureContent bool
}

// NewHandler returns the LangChain agent handler
// (createLangChainAgentsHandler).
func NewHandler(opts Options) *aiserver.ProviderHandler {
	aiserver.ReportUsage("langchain-agents.createLangChainAgentsHandler", PackageName, PackageVersion)
	return newHandler(opts)
}

func newHandler(opts Options) *aiserver.ProviderHandler {
	h := &handler{opts: opts}
	return aiserver.CreateHandlerInternal(ProvidesFor, h.invoke, h.stream, opts.CaptureContent)
}

// InvokeOptions are the optional arguments to LangChainAgents.
type InvokeOptions struct {
	ToolHandlers map[string]aiserver.ToolHandler
	Registry     *aiserver.Registry
	SkipJudges   bool
	// Variables are passed to Invoke, not to Config.
	Variables map[string]any
	// Handler configures the handler the wrapper builds.
	Handler Options
}

// LangChainAgents invokes configKey through a fresh LangChain agent handler
// (langchainAgents). opts may be nil.
func LangChainAgents(ctx context.Context, configKey, userInput string, userCtx ldcontext.Context, opts *InvokeOptions) (aiserver.ProviderResponse, error) {
	aiserver.ReportUsage("langchain-agents.langchainAgents", PackageName, PackageVersion)
	if opts == nil {
		opts = &InvokeOptions{}
	}
	return aiserver.ConfigInternal(aiserver.ConfigArgs{
		Key: configKey, Handlers: []*aiserver.ProviderHandler{newHandler(opts.Handler)},
		ToolHandlers: opts.ToolHandlers, Registry: opts.Registry, SkipJudges: opts.SkipJudges,
	}).Invoke(ctx, userInput, userCtx, aiserver.WithVariables(opts.Variables))
}

type handler struct{ opts Options }

type runResult struct {
	text      string // blocking: the last AI message; streaming: every turn's text
	usage     aiserver.SpanUsage
	abandoned bool
}

func (r runResult) rawUsage() map[string]any {
	return map[string]any{"input_tokens": r.usage.Input, "output_tokens": r.usage.Output}
}

// run builds the ReAct agent for req and runs it; onText is nil when
// blocking.
func (h *handler) run(ctx context.Context, req aiserver.HandlerRequest, onText func(string) bool) (runResult, error) {
	// Mark an explicit temperature of 0 before any span or model context is
	// derived from ctx, so the Bedrock client middleware sees it.
	ctx = langchainwire.MarkExplicitZeroTemperature(ctx, req.Config.Model.Parameters)
	req.Config = langchainwire.ModelConfig(req.Config)
	cfg := req.Config

	// Agent mode: instructions win; history and userInput become messages,
	// never system-prompt text.
	p := aiserver.ComposePrompt(req, aiserver.ModeAgent)
	if onText == nil { // streaming ignores outputFormat
		p.System = aiserver.WithOutputFormat(p.System, cfg.OutputFormat)
	}
	r := aiserver.StartHandlerRun(ctx, aiserver.HandlerSpanOptions{
		Tracer: PackageName, Name: SpanName, System: langchainwire.ProviderName(cfg), Operation: operation,
		RequestModel: cfg.Model.Name,
	}, req, p)
	defer r.EndOnPanic()

	var res runResult
	model, err := langchainwire.ResolveModel(r.Context(), h.opts.Model, h.opts.ModelFactory, cfg, nil)
	if err != nil {
		r.Fail(err)
		return res, err
	}
	tools, fns := langchainwire.BuildTools(cfg, req.ToolHandlers)
	a := &reactAgent{
		model: model, tools: tools, system: p.System, opts: langchainwire.CallOptions(cfg.Model.Parameters),
		modelName: cfg.Model.Name, run: r, onText: onText,
		exec: func(ctx context.Context, tc llms.ToolCall) (llms.MessageContent, *stategraph.Command, error) {
			m, err := langchainwire.RunTool(ctx, r, fns, tc)
			return m, nil, err
		},
	}
	out, err := a.invoke(r.Context(), langchainwire.ToMessages(aiserver.Prompt{Turns: p.Turns}))
	res.usage = a.usage
	if errors.Is(err, errStopped) {
		r.Abandon()
		res.abandoned = true
		return res, nil
	}
	if err != nil {
		r.Fail(err)
		return res, err
	}
	res.text = lastAIText(out.messages)
	if onText != nil {
		res.text = a.streamed.String()
	}
	r.Finish(cfg.Model.Name, res.text)
	return res, nil
}

func (h *handler) invoke(ctx context.Context, req aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
	res, err := h.run(ctx, req, nil)
	if err != nil {
		return aiserver.HandlerResult{}, err
	}
	return aiserver.HandlerResult{Output: res.text, Usage: res.rawUsage()}, nil
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
