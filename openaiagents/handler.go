// Package openaiagents is the OpenAI agent-mode handler for aiserver
// (providesFor OpenAI/agent), plus the openaiGraph wrapper and the
// toOpenAIAgents native graph adapter.
//
// Go has no official OpenAI Agents SDK, so this package runs its own agent
// loop on the official Responses API client: an agent has instructions,
// function and hosted tools, handoffs and an optional output type; a run
// executes tools, follows handoffs, and stops at the first turn with no
// function calls, reproducing the Agents SDK's observable contract.
package openaiagents

import (
	"context"
	"errors"
	"iter"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/openaiwire"
	"github.com/launchdarkly/go-ai-sdk/internal/sdkapi"
)

const (
	// PackageName is the Go import path, registered for `$ld:ai:sdk:info`.
	PackageName = "github.com/launchdarkly/go-ai-sdk/openaiagents"
	// PackageVersion is this package's release version.
	PackageVersion = "0.1.0" // x-release-please-version
	// SpanName is the blocking handler's root span.
	SpanName = "openai.agent.run"
	// StreamSpanName is the streaming handler's root span.
	StreamSpanName = "openai.agent.run.stream"
	// operation is gen_ai.operation.name on the root span.
	operation = "invoke_agent"
)

var (
	// ErrMaxTurns is returned when a run needs more model turns than allowed.
	ErrMaxTurns = errors.New("openaiagents: run exceeded the maximum number of turns")
	// ErrToolNotOffered is returned when the model calls a tool the agent
	// was not given.
	ErrToolNotOffered = errors.New("openaiagents: model called a tool that was not offered")
	// ErrNoRoot is returned by the native graph runner for a graph without a
	// root node.
	ErrNoRoot = errors.New("openaiagents: agent graph has no root node")
	// ErrNilGraph is returned by the native graph runner for a nil definition.
	ErrNilGraph = errors.New("openaiagents: agent graph definition is nil")
)

// ProvidesFor is the variation provider/mode this handler serves.
var ProvidesFor = aiserver.ProvidesFor{Provider: "OpenAI", Mode: aiserver.ModeAgent}

func init() { aiserver.RegisterAiSdkPackage(PackageName, PackageVersion) }

// Options configure NewHandler and the client used by the graph helpers.
type Options struct {
	// Client is used for every call when set. Otherwise a client is built from
	// APIKey, falling back to the OPENAI_API_KEY environment variable.
	Client *openai.Client
	// APIKey is used only when Client is nil.
	APIKey string
	// CaptureContent is attached to the handler so the client core can gate
	// judge explanations on it.
	CaptureContent bool
	// MaxTurns caps model turns per run; 0 means 10, the Agents SDK default.
	// A variation's `max_turns` model parameter takes precedence.
	MaxTurns int
}

func (o Options) client() *openai.Client {
	if o.Client != nil {
		return o.Client
	}
	var reqOpts []option.RequestOption
	if o.APIKey != "" {
		reqOpts = append(reqOpts, option.WithAPIKey(o.APIKey))
	}
	c := openai.NewClient(reqOpts...)
	return &c
}

// NewHandler returns the OpenAI agent handler (createOpenAIAgentsHandler).
func NewHandler(opts Options) *aiserver.ProviderHandler {
	aiserver.ReportUsage("openai-agents.createOpenAIAgentsHandler", PackageName, PackageVersion)
	return newHandler(opts)
}

func newHandler(opts Options) *aiserver.ProviderHandler {
	h := &handler{client: opts.client(), maxTurns: opts.MaxTurns}
	return sdkapi.CreateHandler(ProvidesFor, h.invoke, h.stream, opts.CaptureContent)
}

// InvokeOptions are the optional arguments to OpenAIAgents.
type InvokeOptions struct {
	ToolHandlers map[string]aiserver.ToolHandler
	Registry     *aiserver.Registry
	SkipJudges   bool
	// Variables are passed to Invoke, not to the config.
	Variables map[string]any
	// Handler configures the handler built for this call.
	Handler Options
}

// OpenAIAgents invokes configKey through a fresh OpenAI agent handler
// (openaiAgents). opts may be nil.
func OpenAIAgents(ctx context.Context, configKey, userInput string, userCtx ldcontext.Context, opts *InvokeOptions) (aiserver.ProviderResponse, error) {
	aiserver.ReportUsage("openai-agents.openaiAgents", PackageName, PackageVersion)
	if opts == nil {
		opts = &InvokeOptions{}
	}
	return sdkapi.Config(aiserver.ConfigArgs{
		Key: configKey, Handlers: []*aiserver.ProviderHandler{newHandler(opts.Handler)},
		ToolHandlers: opts.ToolHandlers, Registry: opts.Registry, SkipJudges: opts.SkipJudges,
	}).Invoke(ctx, userInput, userCtx, aiserver.WithVariables(opts.Variables))
}

type handler struct {
	client   *openai.Client
	maxTurns int
}

// run builds the single agent for req and runs it; onText is nil when
// blocking. abandoned reports that the stream consumer stopped iterating.
func (h *handler) run(ctx context.Context, req aiserver.HandlerRequest, onText func(string) bool) (res runResult, abandoned bool, err error) {
	cfg := req.Config
	// Agent mode: instructions win over config system messages;
	// history and userInput become structured input items.
	p := aiserver.ComposePrompt(req, aiserver.ModeAgent)
	name := SpanName
	if onText != nil {
		name = StreamSpanName
	}
	span := aiserver.StartHandlerRun(ctx, aiserver.HandlerSpanOptions{
		Tracer: PackageName, Name: name, System: "openai", Operation: operation, RequestModel: cfg.Model.Name,
	}, req, p)
	defer span.EndOnPanic()

	a := newAgent("assistant", p.System, cfg, req.ToolHandlers)
	if onText == nil { // streaming ignores outputFormat
		a.outputType = buildOutputType(cfg.OutputFormat)
	}
	r := &runner{client: h.client, maxTurns: maxTurns(cfg.Model.Parameters, h.maxTurns), span: span, onText: onText}
	res, stopped, err := r.run(span.Context(), a, openaiwire.Turns(p.Turns))
	switch {
	case stopped:
		span.Abandon()
		return res, true, nil
	case err != nil:
		span.Fail(err)
		return res, false, err
	}
	span.Finish(res.model, aiserver.ToolResultString(res.output))
	return res, false, nil
}

func (h *handler) invoke(ctx context.Context, req aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
	res, _, err := h.run(ctx, req, nil)
	if err != nil {
		return aiserver.HandlerResult{}, err
	}
	return aiserver.HandlerResult{Output: res.output, Usage: res.usage.Raw()}, nil
}

func (h *handler) stream(ctx context.Context, req aiserver.HandlerRequest) iter.Seq2[aiserver.HandlerStreamEvent, error] {
	return func(yield func(aiserver.HandlerStreamEvent, error) bool) {
		res, abandoned, err := h.run(ctx, req, func(t string) bool {
			return yield(aiserver.HandlerStreamEvent{Type: aiserver.EventChunk, Text: t}, nil)
		})
		if abandoned {
			return
		}
		if err != nil {
			yield(aiserver.HandlerStreamEvent{}, err)
			return
		}
		yield(aiserver.HandlerStreamEvent{Type: aiserver.EventDone, Output: &res.text, Usage: res.usage.Raw()}, nil)
	}
}
