// Package sdkapi is the handler-package entry to aiserver. It does not report
// $ld:ai:sdk:usage; the handler package reports its own helper. Applications
// call aiserver.Config, aiserver.Graph, and aiserver.CreateHandler instead.
package sdkapi

import (
	"context"
	"iter"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/bridge"
	"github.com/launchdarkly/go-ai-sdk/internal/telemetry"
)

// Config is aiserver.Config without a usage report.
func Config(args aiserver.ConfigArgs) *aiserver.AIConfig {
	return bridge.ConfigWithOptions(args, bridge.RunOptions{NoUsage: true}).(*aiserver.AIConfig)
}

// Graph is aiserver.Graph without a usage report.
func Graph(key string, opts aiserver.GraphOptions) *aiserver.AgentGraph {
	return bridge.GraphWithOptions(key, opts, bridge.RunOptions{NoUsage: true}).(*aiserver.AgentGraph)
}

// CreateHandler is aiserver.CreateHandler without a usage report. As there,
// a direct Invoke or Stream call applies the handler's capture setting, and a
// call through the core keeps the core's binding (from the handler value it
// routed to, which may be a copy).
func CreateHandler(pf aiserver.ProvidesFor, invoke aiserver.InvokeFunc, stream aiserver.StreamFunc, captureContent aiserver.ContentCapture) *aiserver.ProviderHandler {
	h := &aiserver.ProviderHandler{ProvidesFor: pf, CaptureContent: captureContent}
	if invoke != nil {
		h.Invoke = func(ctx context.Context, req aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
			return invoke(telemetry.Enter(ctx, uint8(h.CaptureContent)), req)
		}
	}
	if stream != nil {
		h.Stream = func(ctx context.Context, req aiserver.HandlerRequest) iter.Seq2[aiserver.HandlerStreamEvent, error] {
			return stream(telemetry.Enter(ctx, uint8(h.CaptureContent)), req)
		}
	}
	return h
}

// StartInvocation marks the start of a run that does not go through a
// ProviderHandler, such as a native graph runner: it latches
// aiserver.Configure and binds capture on ctx for every span under it.
func StartInvocation(ctx context.Context, capture aiserver.ContentCapture) context.Context {
	return telemetry.StartInvocation(ctx, uint8(capture))
}
