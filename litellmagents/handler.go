// Package litellmagents is the LiteLLM agent-mode handler for aiserver
// (providesFor */agent), plus the LiteLLMGraph wrapper and the
// ToLiteLLMAgents native graph adapter.
//
// Go has no OpenAI Agents SDK, so this package runs the same agent loop as
// openaiagents, with a Chat Completions model whose openai-go client points
// at a LiteLLM proxy. The proxy routes each model name to its upstream, so
// one handler serves every provider the proxy is configured for.
package litellmagents

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"iter"
	"os"
	"strings"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/agentloop"
	"github.com/launchdarkly/go-ai-sdk/internal/handlerkit"
	"github.com/launchdarkly/go-ai-sdk/internal/kitbase"
	"github.com/launchdarkly/go-ai-sdk/internal/sdkapi"
	"github.com/launchdarkly/go-ai-sdk/internal/telemetry"
)

const (
	// PackageName is the Go import path, registered for `$ld:ai:sdk:info`.
	PackageName = "github.com/launchdarkly/go-ai-sdk/litellmagents"
	// PackageVersion is this package's release version.
	PackageVersion = "1.0.0" // x-release-please-version
	// SpanName is the root span name for both the blocking and streaming paths.
	SpanName = "invoke_agent"
	// operation is gen_ai.operation.name on the root span.
	operation = "invoke_agent"
	// system is gen_ai.system, and gen_ai.provider.name for a config
	// without a provider.
	system = "litellm"
)

var (
	// ErrNoBaseURL is returned when no proxy base URL is configured: no
	// Options.BaseURL, no LITELLM_BASE_URL, and no Client or ClientFactory.
	ErrNoBaseURL = errors.New("litellmagents: no LiteLLM proxy base URL (set Options.BaseURL or LITELLM_BASE_URL)")
	// ErrMaxTurns is returned when a run needs more model turns than allowed.
	ErrMaxTurns = errors.New("litellmagents: run exceeded the maximum number of turns")
	// ErrToolNotOffered is returned when the model calls a tool the agent
	// was not given.
	ErrToolNotOffered = errors.New("litellmagents: model called a tool that was not offered")
	// ErrNoRoot is returned by the native graph runner for a graph without a
	// root node.
	ErrNoRoot = errors.New("litellmagents: agent graph has no root node")
	// ErrNilGraph is returned by the native graph runner for a nil definition.
	ErrNilGraph = errors.New("litellmagents: agent graph definition is nil")
)

// providesFor is the variation provider/mode this handler serves: any
// provider, since the proxy routes the model name.
var providesFor = aiserver.ProvidesFor{Provider: kitbase.WildcardProvider, Mode: aiserver.ModeAgent}

func init() { handlerkit.RegisterAiSdkPackage(PackageName, PackageVersion) }

// Options configure NewHandler and the client used by the graph helpers.
type Options struct {
	// BaseURL is the LiteLLM proxy URL, falling back to LITELLM_BASE_URL.
	// The OPENAI_* environment variables are never read.
	BaseURL string
	// APIKey is the proxy key, falling back to LITELLM_API_KEY. A proxy
	// without authentication needs none.
	APIKey string
	// Client is used for every call when set, instead of BaseURL / APIKey.
	// It must point at the LiteLLM proxy.
	Client *openai.Client
	// ClientFactory, when set, returns the client for each call from the
	// evaluated config. It wins over Client.
	ClientFactory func(aiserver.AiConfig) (*openai.Client, error)
	// CaptureContent gates prompt, completion and tool content on spans.
	// CaptureDefault follows aiserver.Configure, else on.
	CaptureContent aiserver.ContentCapture
	// MaxTurns caps model turns per run; 0 means 10, the Agents SDK default.
	// A variation's `max_turns` model parameter takes precedence.
	MaxTurns int
}

// clientFunc returns the proxy client for one call's evaluated config.
type clientFunc func(aiserver.AiConfig) (*openai.Client, error)

// connect resolves where calls go, in order: ClientFactory, Client, then a
// client built from BaseURL / LITELLM_BASE_URL and APIKey / LITELLM_API_KEY.
// Without any of them it returns ErrNoBaseURL.
func (o Options) connect() (clientFunc, error) {
	switch {
	case o.ClientFactory != nil:
		return func(cfg aiserver.AiConfig) (*openai.Client, error) {
			c, err := o.ClientFactory(cfg)
			if err != nil {
				return nil, fmt.Errorf("litellmagents: client factory: %w", err)
			}
			if c == nil {
				return nil, errors.New("litellmagents: client factory returned a nil client")
			}
			return c, nil
		}, nil
	case o.Client != nil:
		return func(aiserver.AiConfig) (*openai.Client, error) { return o.Client, nil }, nil
	}
	base := cmp.Or(o.BaseURL, os.Getenv("LITELLM_BASE_URL"))
	if strings.TrimSpace(base) == "" {
		return nil, ErrNoBaseURL
	}
	// Later options override the OPENAI_* environment defaults openai-go
	// applies first: the proxy never sees OpenAI credentials or headers.
	reqOpts := []option.RequestOption{
		option.WithBaseURL(base),
		option.WithAPIKey(cmp.Or(o.APIKey, os.Getenv("LITELLM_API_KEY"))),
		option.WithAdminAPIKey(""),
		option.WithHeaderDel("OpenAI-Organization"),
		option.WithHeaderDel("OpenAI-Project"),
	}
	for _, line := range strings.Split(os.Getenv("OPENAI_CUSTOM_HEADERS"), "\n") {
		if name, _, ok := strings.Cut(line, ":"); ok {
			reqOpts = append(reqOpts, option.WithHeaderDel(strings.TrimSpace(name)))
		}
	}
	c := openai.NewClient(reqOpts...)
	return func(aiserver.AiConfig) (*openai.Client, error) { return &c, nil }, nil
}

// NewHandler returns the LiteLLM agent handler (createLiteLLMAgentsHandler).
// It returns ErrNoBaseURL when no proxy is configured.
func NewHandler(opts Options) (*aiserver.ProviderHandler, error) {
	handlerkit.ReportUsage("litellm-agents.createLiteLLMAgentsHandler", PackageName, PackageVersion)
	return newHandler(opts)
}

// newHandler is NewHandler without the usage report.
func newHandler(opts Options) (*aiserver.ProviderHandler, error) {
	client, err := opts.connect()
	if err != nil {
		return nil, err
	}
	h := &agentHandler{client: client, maxTurns: opts.MaxTurns}
	return sdkapi.CreateHandler(providesFor, h.invoke, h.stream, opts.CaptureContent), nil
}

// InvokeOptions are the optional arguments to LiteLLMAgents.
type InvokeOptions struct {
	ToolHandlers map[string]aiserver.ToolHandler
	Registry     *aiserver.Registry
	SkipJudges   bool
	// Variables are passed to Invoke, not to the config.
	Variables map[string]any
	// Handler configures the handler built for this call.
	Handler Options
}

// LiteLLMAgents invokes configKey through a fresh LiteLLM agent handler
// (litellmAgents). opts may be nil. A NewHandler error is returned here.
func LiteLLMAgents(ctx context.Context, configKey, userInput string, userCtx ldcontext.Context, opts *InvokeOptions) (aiserver.ProviderResponse, error) {
	handlerkit.ReportUsage("litellm-agents.litellmAgents", PackageName, PackageVersion)
	if opts == nil {
		opts = &InvokeOptions{}
	}
	h, err := newHandler(opts.Handler)
	if err != nil {
		return aiserver.ProviderResponse{}, err
	}
	return sdkapi.Config(aiserver.ConfigArgs{
		Key: configKey, Handlers: []*aiserver.ProviderHandler{h},
		ToolHandlers: opts.ToolHandlers, Registry: opts.Registry, SkipJudges: opts.SkipJudges,
	}).Invoke(ctx, userInput, userCtx, aiserver.WithVariables(opts.Variables))
}

type agentHandler struct {
	client   clientFunc
	maxTurns int
}

// spanOptions is the root span identity: the configured provider as
// gen_ai.provider.name ("aws.bedrock" for Bedrock, else lowercased, "litellm"
// when unset) and the model id sent to the proxy.
func spanOptions(cfg aiserver.AiConfig) kitbase.HandlerSpanOptions {
	return kitbase.HandlerSpanOptions{
		Tracer: PackageName, Name: SpanName, System: system, Operation: operation,
		RequestModel: requestModel(cfg), ProviderName: telemetry.ProviderName(cmp.Or(cfg.Provider.Name, system)),
	}
}

// run builds the single agent for req and runs it; onText is nil when
// blocking. abandoned reports that the stream consumer stopped iterating.
func (h *agentHandler) run(ctx context.Context, req aiserver.HandlerRequest, onText func(string) bool) (res agentloop.Result, abandoned bool, err error) {
	cfg := req.Config
	p := handlerkit.ComposePrompt(req, aiserver.ModeAgent)
	span := handlerkit.StartHandlerRun(ctx, spanOptions(cfg), req, p)
	defer span.EndOnPanic()

	client, err := h.client(cfg)
	if err != nil {
		span.Fail(err)
		return res, false, err
	}
	a := newAgent("assistant", p.System, cfg, req.ToolHandlers)
	if onText == nil { // streaming ignores outputFormat
		a.schema = outputSchema(cfg.OutputFormat)
	}
	r := newRunner(client, agentloop.MaxTurns(cfg.Model.Parameters, h.maxTurns))
	r.Span, r.OnText = span, onText
	res, stopped, err := r.Run(span.Context(), a, p.Turns)
	switch {
	case stopped:
		span.Abandon()
		return res, true, nil
	case err != nil:
		span.Fail(err)
		return res, false, err
	}
	span.Finish(res.Model, aiserver.ToolResultString(res.Output))
	return res, false, nil
}

func (h *agentHandler) invoke(ctx context.Context, req aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
	res, _, err := h.run(ctx, req, nil)
	if err != nil {
		return aiserver.HandlerResult{}, err
	}
	return aiserver.HandlerResult{Output: res.Output, Usage: res.Tokens.Raw()}, nil
}

func (h *agentHandler) stream(ctx context.Context, req aiserver.HandlerRequest) iter.Seq2[aiserver.HandlerStreamEvent, error] {
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
		yield(aiserver.HandlerStreamEvent{Type: aiserver.EventDone, Output: &res.Text, Usage: res.Tokens.Raw()}, nil)
	}
}
