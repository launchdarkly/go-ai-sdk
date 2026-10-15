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
)

// ExtractVariationRaw resolves a flag variation as decoded JSON and returns
// an error when it is disabled. Set by the core package at init; experimental
// code parses the result with aiserver.ParseAiConfig.
var ExtractVariationRaw func(ctx context.Context, key string, c ldcontext.Context) (map[string]any, error)

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
