// Package bridge holds the internal hooks between the core client and the
// experimental entry point. It imports nothing from the SDK,
// so both sides can depend on it without an import cycle.
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
// code parses the result with aiserver.ParseAiConfig.
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

var (
	mu    sync.Mutex
	hooks []func(context.Context)
)

// OnShutdown registers experimental state cleanup to run inside
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

// Logger returns the SDK logger set by aiserver.SetLogger, so experimental
// code logs where the application pointed the core. The core assigns it once
// in init, before any goroutine can read it.
var Logger = slog.Default
