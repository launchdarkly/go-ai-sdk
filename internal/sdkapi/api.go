// Package sdkapi is the handler-package entry to aiserver. It does not report
// $ld:ai:sdk:usage; the handler package reports its own helper. Applications
// call aiserver.Config, aiserver.Graph, and aiserver.CreateHandler instead.
package sdkapi

import (
	"context"
	"iter"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/bridge"
	"github.com/launchdarkly/go-ai-sdk/internal/sdkcall"
)

// Config is aiserver.Config without a usage report.
func Config(args aiserver.ConfigArgs) *aiserver.AIConfig {
	args.SDKCall = sdkcall.Mark()
	return aiserver.Config(args)
}

// Graph is aiserver.Graph without a usage report.
func Graph(key string, opts aiserver.GraphOptions) *aiserver.AgentGraph {
	opts.SDKCall = sdkcall.Mark()
	return aiserver.Graph(key, opts)
}

// CreateHandler is aiserver.CreateHandler without a usage report. As there,
// a direct Invoke or Stream call applies the handler's capture setting.
func CreateHandler(pf aiserver.ProvidesFor, invoke aiserver.InvokeFunc, stream aiserver.StreamFunc, captureContent aiserver.ContentCapture) *aiserver.ProviderHandler {
	h := &aiserver.ProviderHandler{ProvidesFor: pf, CaptureContent: captureContent}
	if invoke != nil {
		h.Invoke = func(ctx context.Context, req aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
			return invoke(bridge.StartInvocation(ctx, uint8(h.CaptureContent)), req)
		}
	}
	if stream != nil {
		h.Stream = func(ctx context.Context, req aiserver.HandlerRequest) iter.Seq2[aiserver.HandlerStreamEvent, error] {
			return stream(bridge.StartInvocation(ctx, uint8(h.CaptureContent)), req)
		}
	}
	return h
}
