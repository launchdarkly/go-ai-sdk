package claudeagents

import (
	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/sdkapi"
)

// GraphOptions are aiserver.GraphOptions without Handlers: ClaudeGraph binds
// its own handler and the caller cannot override it.
type GraphOptions struct {
	// ToolHandlers are the tools offered to every node.
	ToolHandlers map[string]aiserver.ToolHandler
	// GraphJudge is an optional judge config key run against the final output.
	GraphJudge string
	// Registry supplies tools (and further handlers, tried after this one).
	Registry *aiserver.Registry
}

// ClaudeGraph returns aiserver's model-driven graph runner with exactly one
// handler, this package's agent handler, pre-bound (claudeGraph).
func ClaudeGraph(key string, opts GraphOptions, handler Options) *aiserver.AgentGraph {
	aiserver.ReportUsage("claude-agents.claudeGraph", PackageName, PackageVersion)
	return sdkapi.Graph(key, aiserver.GraphOptions{
		Handlers:     []*aiserver.ProviderHandler{newHandler(handler)},
		ToolHandlers: opts.ToolHandlers, GraphJudge: opts.GraphJudge, Registry: opts.Registry,
	})
}
