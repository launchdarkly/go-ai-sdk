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
	"context"
	"errors"
	"iter"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/openai/openai-go/v3"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/handlerkit"
	"github.com/launchdarkly/go-ai-sdk/internal/sdkapi"
)

const (
	// PackageName is the Go import path, registered for `$ld:ai:sdk:info`.
	PackageName = "github.com/launchdarkly/go-ai-sdk/litellmagents"
	// PackageVersion is this package's release version.
	PackageVersion = "1.0.0" // x-release-please-version
	// SpanName is the root span name for both the blocking and streaming paths.
	SpanName = "invoke_agent"
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

var errNotImplemented = errors.New("litellmagents: not implemented")

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

// NewHandler returns the LiteLLM agent handler (createLiteLLMAgentsHandler).
// It returns ErrNoBaseURL when no proxy is configured.
func NewHandler(opts Options) (*aiserver.ProviderHandler, error) {
	return sdkapi.CreateHandler(aiserver.ProvidesFor{}, notImplemented, notImplementedStream, opts.CaptureContent), nil
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
	return aiserver.ProviderResponse{}, errNotImplemented
}

func notImplemented(context.Context, aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
	return aiserver.HandlerResult{}, errNotImplemented
}

func notImplementedStream(context.Context, aiserver.HandlerRequest) iter.Seq2[aiserver.HandlerStreamEvent, error] {
	return func(yield func(aiserver.HandlerStreamEvent, error) bool) {
		yield(aiserver.HandlerStreamEvent{}, errNotImplemented)
	}
}
