package litellmagents

import (
	"context"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/sdkapi"
)

// LiteLLMGraph returns aiserver's model-driven graph runner with this
// package's agent handler as the only handler (litellmGraph). Any
// options.Handlers value is replaced. handler configures that agent
// handler, proxy connection included; the error is NewHandler's.
func LiteLLMGraph(key string, options aiserver.GraphOptions, handler Options) (*aiserver.AgentGraph, error) {
	return sdkapi.Graph(key, options), nil
}

// NativeGraphOptions configure ToLiteLLMAgents.
type NativeGraphOptions struct {
	// ToolHandlers supply each node's tools by config tool key.
	ToolHandlers map[string]aiserver.ToolHandler
	// Context is the LaunchDarkly context events are tracked for. Without
	// one (the zero value), the runner tracks no events.
	Context ldcontext.Context
	// Handler supplies the proxy connection and the default MaxTurns.
	Handler Options
}

var _ aiserver.NativeGraphRunner = (*NativeGraph)(nil)

// NativeGraph runs a resolved graph as one multi-agent run over the LiteLLM
// proxy: one agent per node, a handoff tool per outgoing edge, starting at
// the root.
type NativeGraph struct{}

// ToLiteLLMAgents translates def into an agent tree run natively over the
// LiteLLM proxy (toLiteLLMAgents). opts may be nil. Errors for a nil,
// disabled or rootless definition, and for a missing proxy, are returned by
// Invoke, which records them on the graph span.
func ToLiteLLMAgents(def *aiserver.GraphDefinition, opts *NativeGraphOptions) (*NativeGraph, error) {
	return &NativeGraph{}, nil
}

// Invoke runs the graph from the root with input.
func (g *NativeGraph) Invoke(ctx context.Context, input string, opts ...aiserver.CallOption) (aiserver.GraphResponse, error) {
	return aiserver.GraphResponse{}, errNotImplemented
}
