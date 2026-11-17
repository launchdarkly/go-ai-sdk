// Package litellmmessages is the LiteLLM messages handler for aiserver
// (providesFor */messages). It sends every call to a LiteLLM proxy over the
// OpenAI-compatible Chat Completions API, so one handler serves any model the
// proxy routes: provider-qualified names such as anthropic/claude-sonnet-4,
// bedrock/..., gemini/..., and proxy aliases.
package litellmmessages

import (
	"context"
	"errors"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/openai/openai-go/v3"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/handlerkit"
)

const (
	// PackageName is the Go import path, registered for `$ld:ai:sdk:info`.
	PackageName = "github.com/launchdarkly/go-ai-sdk/litellmmessages"
	// PackageVersion is this package's release version.
	PackageVersion = "1.0.0" // x-release-please-version
	// SpanName is the root span name for both the blocking and streaming paths.
	SpanName = "invoke_agent"
)

var (
	// ErrNoBaseURL is returned by NewHandler when no proxy base URL is set
	// (Options.BaseURL or LITELLM_BASE_URL) and no client is injected.
	ErrNoBaseURL = errors.New("litellmmessages: LiteLLM proxy base URL is required (set Options.BaseURL or LITELLM_BASE_URL)")
	// ErrMaxSteps is returned when the model keeps requesting tools past the
	// tool-loop bound.
	ErrMaxSteps = errors.New("litellmmessages: tool loop exceeded the maximum number of steps")
)

var errNotImplemented = errors.New("litellmmessages: not implemented yet")

func init() { handlerkit.RegisterAiSdkPackage(PackageName, PackageVersion) }

// ClientFactory returns the client for one call from the evaluated AI
// Config.
type ClientFactory func(cfg aiserver.AiConfig) (*openai.Client, error)

// Options configure NewHandler.
type Options struct {
	// BaseURL is the LiteLLM proxy base URL. "" falls back to LITELLM_BASE_URL.
	// The OPENAI_* environment variables are never read.
	BaseURL string
	// APIKey is the proxy key. "" falls back to LITELLM_API_KEY; an
	// unauthenticated proxy needs neither.
	APIKey string
	// Client is used for every call when set, instead of a client built from
	// BaseURL and APIKey. It must point at a LiteLLM proxy.
	Client *openai.Client
	// ClientFactory is called per call, after the AI Config is evaluated,
	// when set. It wins over Client.
	ClientFactory ClientFactory
	// MaxRetries is the retry count on the client NewHandler builds. nil keeps
	// the provider SDK's default; 0 means a single attempt. A negative count, or
	// MaxRetries together with Client or ClientFactory, is refused by NewHandler.
	MaxRetries *int
	// CaptureContent gates prompt, completion, tool and judge-explanation
	// content on spans. CaptureDefault follows aiserver.Configure, else on.
	CaptureContent aiserver.ContentCapture
}

// NewHandler returns the LiteLLM messages handler
// (createLiteLLMMessagesHandler). It returns ErrNoBaseURL without a proxy
// base URL or an injected client, and an error for an invalid
// Options.MaxRetries.
func NewHandler(opts Options) (*aiserver.ProviderHandler, error) {
	return nil, errNotImplemented
}

// InvokeOptions are the optional arguments to LiteLLMMessages.
type InvokeOptions struct {
	ToolHandlers map[string]aiserver.ToolHandler
	Registry     *aiserver.Registry
	SkipJudges   bool
	// Variables are passed to Invoke, not to the config.
	Variables map[string]any
	// Handler configures the handler built for this call.
	Handler Options
}

// LiteLLMMessages invokes configKey through a fresh LiteLLM messages handler
// (litellmMessages). A NewHandler error is returned as this call's error.
// opts may be nil.
func LiteLLMMessages(ctx context.Context, configKey, userInput string, userCtx ldcontext.Context, opts *InvokeOptions) (aiserver.ProviderResponse, error) {
	return aiserver.ProviderResponse{}, errNotImplemented
}
