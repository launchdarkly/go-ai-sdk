// Package bridge holds the internal hooks between the core client and the
// packages built on it (the experimental entry point, skills, handlers). It
// imports nothing from the SDK, so both sides can depend on it without an
// import cycle.
package bridge

import (
	"context"
	"log/slog"
	"slices"
	"sync"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/launchdarkly/go-sdk-common/v3/ldvalue"
	"go.opentelemetry.io/otel/attribute"
)

// ExtractVariationRaw resolves a flag variation as decoded JSON and returns
// an error when it is disabled. Set by the core package at init; experimental
// code parses the result with handlerkit.ParseAiConfig.
var ExtractVariationRaw func(ctx context.Context, key string, c ldcontext.Context) (map[string]any, error)

// VariationClient is the part of the LaunchDarkly client a Variation gets.
type VariationClient interface {
	JSONVariation(key string, c ldcontext.Context, def ldvalue.Value) (ldvalue.Value, error)
}

// Variation evaluates a config's flag in place of the core's JSONVariation call.
type Variation func(client VariationClient, key string, c ldcontext.Context, def ldvalue.Value) (ldvalue.Value, error)

// RunOptions change one config's runs. The zero value is a plain Config run.
type RunOptions struct {
	Variation      Variation            // nil: JSONVariation
	RunID          string               // "": a new run id per run
	TrackData      map[string]any       // merged into every event's data
	SpanAttributes []attribute.KeyValue // added to the handler's root span
	ExistingClient bool                 // true: use the initialized client and never start one
	NoFlagEvent    bool                 // true: no feature_flag event on the handler's root span
	NoUsage        bool                 // true: no $ld:ai:sdk:usage for the call (SDK packages report their own)
	// WrapHandler, when set, wraps the handler that runs the config. It is
	// func(*aiserver.ProviderHandler) *aiserver.ProviderHandler. The config's
	// judges get the unwrapped handlers.
	WrapHandler any
	// LogJudgeErrors true: a judge that fails is logged and skipped, and the
	// run still succeeds with the other judges' results.
	LogJudgeErrors bool
}

// ConversationID returns the conversation id bound on ctx, or "". Set by the
// core package at init.
var ConversationID func(ctx context.Context) string

// WithExactConversationID binds exactly id on ctx: "" unbinds a bound id.
// Set by the core package at init.
var WithExactConversationID func(ctx context.Context, id string) context.Context

// TrackDataFor returns the event data of one run of config (an
// aiserver.AiConfig) with meta (an aiserver.VariationMeta), as its events
// carry it, with extra merged in. Set by the core package at init.
var TrackDataFor func(configKey string, config, meta any, runID string, extra map[string]any) map[string]any

// ConfigWithOptions returns an *aiserver.AIConfig for args (an
// aiserver.ConfigArgs) that runs with opts. Set by the core package at init.
var ConfigWithOptions func(args any, opts RunOptions) any

// GraphWithOptions returns an *aiserver.AgentGraph for key and opts (an
// aiserver.GraphOptions). Only run.NoUsage applies to a graph. Set by the
// core package at init.
var GraphWithOptions func(key string, opts any, run RunOptions) any

var (
	mu    sync.Mutex
	hooks []func(context.Context)
)

// OnShutdown registers package state cleanup to run inside
// aiserver.Shutdown. A panicking hook is recovered and logged by core so it
// never fails the core call.
func OnShutdown(fn func(context.Context)) {
	mu.Lock()
	defer mu.Unlock()
	hooks = append(hooks, fn)
}

// ShutdownHooks returns the registered hooks in registration order.
func ShutdownHooks() []func(context.Context) {
	mu.Lock()
	defer mu.Unlock()
	return slices.Clone(hooks)
}

// Logger returns the SDK logger set by aiserver.SetLogger, so package
// code logs where the application pointed the core. The core assigns it once
// in init, before any goroutine can read it.
var Logger = slog.Default

// The handler kit. The core package sets each variable at init to a function
// of the type its comment gives; internal/handlerkit asserts each one once.
// Package names in the comments: aiserver, kitbase (internal/kitbase), trace
// (OpenTelemetry), ldcontext.

// StartHandlerRun is func(context.Context, kitbase.HandlerSpanOptions, aiserver.HandlerRequest, string, []aiserver.Message) kitbase.Run.
// The string and the messages are the prompt's system text and turns.
var StartHandlerRun any

// ComposePrompt is func(aiserver.HandlerRequest, aiserver.Mode) (string, []aiserver.Message).
// It returns the prompt's system text and turns.
var ComposePrompt any

// CallableTools is func(aiserver.AiConfig, map[string]aiserver.ToolHandler) map[string]aiserver.Tool.
var CallableTools any

// ReportUsage is func(helper, pkg, version string).
var ReportUsage any

// RegisterAiSdkPackage is func(name, version string).
var RegisterAiSdkPackage any

// SetLDSpanAttributesFor is func(trace.Span, aiserver.TrackData, ldcontext.Context).
var SetLDSpanAttributesFor any

// WrapToolHandlers is func(map[string]aiserver.ToolHandler, ldcontext.Context, aiserver.TrackData) map[string]aiserver.ToolHandler.
var WrapToolHandlers any

// ParseUsage is func(map[string]any) aiserver.TokenUsage.
var ParseUsage any

// NormalizeMode is func(string) aiserver.Mode.
var NormalizeMode any

// ParseAiConfig is func(any) (aiserver.AiConfig, error).
var ParseAiConfig any

// TrackGraphEvent is func(event string, userCtx ldcontext.Context, td aiserver.TrackData, extra map[string]any, metric float64).
var TrackGraphEvent any

// SelectHandler is func([]*aiserver.ProviderHandler, string, aiserver.Mode) (*aiserver.ProviderHandler, error).
var SelectHandler any

// ImageURL is func(aiserver.ImageSource) string.
var ImageURL any

// ContentText is func(aiserver.Message) string.
var ContentText any

// NodeTrackData is func(n *aiserver.GraphNode, graphKey, runID string) aiserver.TrackData.
var NodeTrackData any

// GraphTrackData is func(*aiserver.GraphDefinition) aiserver.TrackData.
var GraphTrackData any

// GraphUserContext is func(*aiserver.GraphDefinition) ldcontext.Context.
var GraphUserContext any

// TrackNativeTool is func(*aiserver.NativeTool).
var TrackNativeTool any
