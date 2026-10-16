// Package sdkapi is the handler-package entry to aiserver. It does not report
// $ld:ai:sdk:usage; the handler package reports its own helper. Applications
// call aiserver.Config, aiserver.Graph, and aiserver.CreateHandler instead.
package sdkapi

import (
	"github.com/launchdarkly/go-ai-sdk/aiserver"
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

// CreateHandler is aiserver.CreateHandler without a usage report.
func CreateHandler(pf aiserver.ProvidesFor, invoke aiserver.InvokeFunc, stream aiserver.StreamFunc, captureContent bool) *aiserver.ProviderHandler {
	return &aiserver.ProviderHandler{ProvidesFor: pf, CaptureContent: captureContent, Invoke: invoke, Stream: stream}
}
