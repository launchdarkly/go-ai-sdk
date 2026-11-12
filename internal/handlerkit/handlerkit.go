// Package handlerkit gives this module's handler and feature packages typed
// access to the handler kit, which stays unexported in aiserver. aiserver
// registers each function in internal/bridge at init; this package asserts each
// one once. aiserver never imports this package.
package handlerkit

import (
	"context"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"go.opentelemetry.io/otel/trace"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/bridge"
	"github.com/launchdarkly/go-ai-sdk/internal/kitbase"
)

// Prompt is a handler's provider-neutral request: a system prompt plus the
// ordered user/assistant turns. Handlers map it to their provider's wire format.
type Prompt struct {
	System string
	Turns  []aiserver.Message
}

// ComposePrompt applies the shared prompt-construction rules so every handler
// builds the same conversation: [system from config] → [config conversation
// messages] → [history] → [userInput?].
func ComposePrompt(req aiserver.HandlerRequest, mode aiserver.Mode) Prompt {
	system, turns := composePrompt(req, mode)
	return Prompt{System: system, Turns: turns}
}

// StartHandlerRun opens a handler's root span, writes model identity, the LD
// config-association and context-identity attributes (from req.Variables),
// and, when capture is enabled for ctx, the prompt content.
func StartHandlerRun(ctx context.Context, opts kitbase.HandlerSpanOptions, req aiserver.HandlerRequest, p Prompt) kitbase.Run {
	return startHandlerRun(ctx, opts, req, p.System, p.Turns)
}

// A missing registration or a type mismatch leaves a variable nil;
// TestEveryKitFunctionIsRegistered names it.
var (
	startHandlerRun, _ = bridge.StartHandlerRun.(func(context.Context, kitbase.HandlerSpanOptions, aiserver.HandlerRequest, string, []aiserver.Message) kitbase.Run)
	composePrompt, _   = bridge.ComposePrompt.(func(aiserver.HandlerRequest, aiserver.Mode) (string, []aiserver.Message))
)

var (
	// CallableTools returns the config tools that have a local ToolFunc,
	// keyed by the config map key: the name the model sees.
	CallableTools, _ = bridge.CallableTools.(func(aiserver.AiConfig, map[string]aiserver.ToolHandler) map[string]aiserver.Tool)
	// ReportUsage records one public helper call for $ld:ai:sdk:usage.
	ReportUsage, _ = bridge.ReportUsage.(func(helper, pkg, version string))
	// RegisterAiSdkPackage records an SDK package's name and version.
	RegisterAiSdkPackage, _ = bridge.RegisterAiSdkPackage.(func(name, version string))
	// SetLDSpanAttributesFor writes the LaunchDarkly config-association and
	// context-identity attributes on a span that has no handler variables.
	SetLDSpanAttributesFor, _ = bridge.SetLDSpanAttributesFor.(func(trace.Span, aiserver.TrackData, ldcontext.Context))
	// WrapToolHandlers wraps each tool so a call emits $ld:ai:tool_call.
	WrapToolHandlers, _ = bridge.WrapToolHandlers.(func(map[string]aiserver.ToolHandler, ldcontext.Context, aiserver.TrackData) map[string]aiserver.ToolHandler)
	// ParseUsage normalises provider token counts.
	ParseUsage, _ = bridge.ParseUsage.(func(map[string]any) aiserver.TokenUsage)
	// NormalizeMode maps a variation mode to a handler mode.
	NormalizeMode, _ = bridge.NormalizeMode.(func(string) aiserver.Mode)
	// ParseAiConfig validates and converts a decoded variation.
	ParseAiConfig, _ = bridge.ParseAiConfig.(func(any) (aiserver.AiConfig, error))
	// TrackGraphEvent sends a $ld:ai:graph:* event with td merged with extra.
	TrackGraphEvent, _ = bridge.TrackGraphEvent.(func(event string, userCtx ldcontext.Context, td aiserver.TrackData, extra map[string]any, metric float64))
	// SelectHandler picks the handler for a provider and mode.
	SelectHandler, _ = bridge.SelectHandler.(func([]*aiserver.ProviderHandler, string, aiserver.Mode) (*aiserver.ProviderHandler, error))
	// ImageURL returns a URL-sourced image's URL, or a data: URL for base64 data.
	ImageURL, _ = bridge.ImageURL.(func(aiserver.ImageSource) string)
	// ContentText flattens a message to text: Text, or the joined text blocks.
	ContentText, _ = bridge.ContentText.(func(aiserver.Message) string)
	// NodeTrackData is a node's tracking payload for one graph run.
	NodeTrackData, _ = bridge.NodeTrackData.(func(n *aiserver.GraphNode, graphKey, runID string) aiserver.TrackData)
	// GraphTrackData is a graph's graph-level track data (graphKey set).
	GraphTrackData, _ = bridge.GraphTrackData.(func(*aiserver.GraphDefinition) aiserver.TrackData)
	// GraphUserContext is the context a graph definition was resolved for.
	GraphUserContext, _ = bridge.GraphUserContext.(func(*aiserver.GraphDefinition) ldcontext.Context)
	// TrackNativeTool records a native tool call. It is a no-op on an
	// unwrapped NativeTool.
	TrackNativeTool, _ = bridge.TrackNativeTool.(func(*aiserver.NativeTool))
)
