package aiserver

import (
	"context"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"go.opentelemetry.io/otel/trace"

	"github.com/launchdarkly/go-ai-sdk/internal/bridge"
	"github.com/launchdarkly/go-ai-sdk/internal/kitbase"
)

var _ kitbase.Run = (*kitHandlerRun)(nil)

// RootSpan is the handler's root span.
func (r *kitHandlerRun) RootSpan() trace.Span { return r.Root }

// The handler kit stays in this package, unexported. Sibling packages reach it
// through internal/handlerkit, which asserts each bridge variable to its type.
func init() {
	bridge.StartHandlerRun = func(ctx context.Context, opts kitbase.HandlerSpanOptions, req HandlerRequest, system string, turns []Message) kitbase.Run {
		return kitStartHandlerRun(ctx, opts, req, prompt{System: system, Turns: turns})
	}
	bridge.ComposePrompt = func(req HandlerRequest, mode Mode) (string, []Message) {
		p := kitComposePrompt(req, mode)
		return p.System, p.Turns
	}
	bridge.CallableTools = kitCallableTools
	bridge.ReportUsage = kitReportUsage
	bridge.RegisterAiSdkPackage = kitRegisterAiSdkPackage
	bridge.SetLDSpanAttributesFor = kitSetLDSpanAttributesFor
	bridge.WrapToolHandlers = kitWrapToolHandlers
	bridge.ParseUsage = kitParseUsage
	bridge.NormalizeMode = kitNormalizeMode
	bridge.ParseAiConfig = kitParseAiConfig
	bridge.TrackGraphEvent = kitTrackGraphEvent
	bridge.SelectHandler = kitSelectHandler
	bridge.ImageURL = kitImageURL
	bridge.ContentText = kitContentText
	bridge.NodeTrackData = (*GraphNode).trackData
	bridge.GraphTrackData = func(d *GraphDefinition) TrackData { return d.trackData }
	bridge.GraphUserContext = func(d *GraphDefinition) ldcontext.Context { return d.userCtx }
	bridge.TrackNativeTool = (*NativeTool).trackCall
}
