// Package opencodeagents is the OpenCode agent handler. Each call starts its
// own `opencode serve` child on loopback, runs one session over its HTTP API
// and stops the child when the call ends.
package opencodeagents

import (
	"context"
	"errors"
	"time"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

const (
	// PackageName is the identity this package registers for $ld:ai:sdk:info
	// and $ld:ai:sdk:usage.
	PackageName = "github.com/launchdarkly/go-ai-sdk/opencodeagents"
	// PackageVersion is this package's release version.
	PackageVersion = "1.0.0" // x-release-please-version
	// SpanName is the root span name for both the blocking and streaming paths.
	SpanName = "invoke_agent"
)

// errNotImplemented marks the API skeleton the tests are written against.
var errNotImplemented = errors.New("opencodeagents: not implemented")

// Timings the tests shorten.
var (
	startupTimeout    = 45 * time.Second
	shutdownGrace     = 5 * time.Second
	drainFallback     = 5 * time.Second
	keepaliveInterval = 15 * time.Second
)

// Options configure NewHandler.
type Options struct {
	// WorkDir is the child's working directory. Required.
	WorkDir string
	// ConfigPath is the OpenCode config file, passed as OPENCODE_CONFIG.
	// Required.
	ConfigPath string
	// Env is the child's whole environment, as "KEY=value" entries. Required:
	// a nil Env is refused, because a nil exec.Cmd.Env inherits the parent's.
	Env []string
	// Command is the executable and leading arguments. Empty means
	// {"opencode"}.
	Command []string
	// Model maps a config onto an OpenCode provider id and model id. Nil
	// means the default mapping.
	Model func(cfg *aiserver.AiConfig) (providerID, modelID string, err error)
	// Agent is the OpenCode agent name sent with each message, if set.
	Agent string
	// CaptureContent gates prompt, completion and tool content on spans.
	CaptureContent aiserver.ContentCapture
}

// NewHandler returns the OpenCode agent handler (createOpenCodeHandler).
func NewHandler(opts Options) (*aiserver.ProviderHandler, error) {
	return nil, errNotImplemented
}

// InvokeOptions are the optional arguments to OpenCodeAgents.
type InvokeOptions struct {
	// ToolHandlers are the config's tool executors and NativeTool sentinels.
	ToolHandlers map[string]aiserver.ToolHandler
	// Registry supplies further tools and handlers.
	Registry *aiserver.Registry
	// SkipJudges returns judge tasks instead of running judges.
	SkipJudges bool
	// Variables are forwarded to Invoke, not to the config.
	Variables map[string]any
	// Handler configures the handler built for this call.
	Handler Options
}

// OpenCodeAgents invokes configKey through a fresh OpenCode handler.
func OpenCodeAgents(ctx context.Context, configKey, userInput string, userCtx ldcontext.Context, opts *InvokeOptions) (aiserver.ProviderResponse, error) {
	return aiserver.ProviderResponse{}, errNotImplemented
}

// SessionError reports that the OpenCode session failed on its event bus.
// It carries the error name, never the provider's error body.
type SessionError struct {
	// Name is the bus error's name, or "" when it had none.
	Name string
}

func (e *SessionError) Error() string { return "opencodeagents: not implemented" }

// defaultModel is the default Options.Model.
func defaultModel(cfg *aiserver.AiConfig) (providerID, modelID string, err error) {
	return "", "", errNotImplemented
}
