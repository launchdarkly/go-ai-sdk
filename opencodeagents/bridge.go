package opencodeagents

import (
	"context"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

// bridge is a per-call loopback MCP server for the config's ToolFunc tools.
type bridge struct {
	token string
}

func newBridge() *bridge { return &bridge{} }

// register serves fn under key, with the config's description and schema.
func (b *bridge) register(key, description string, parameters map[string]any, fn aiserver.ToolFunc) {}

// start binds a loopback port and returns the bridge's base URL. Tool calls
// run under ctx.
func (b *bridge) start(ctx context.Context) (string, error) { return "", errNotImplemented }

// stop closes the listener and ends open event streams.
func (b *bridge) stop() {}

// bridgeToolName is the name OpenCode reports for one bridged tool.
func bridgeToolName(server, key string) string { return "" }
