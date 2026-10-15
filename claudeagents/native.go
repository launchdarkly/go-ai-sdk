package claudeagents

import (
	"context"
	"fmt"
	"hash/fnv"
	"maps"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/anthropicwire"
)

// SubAgentToolPrefix prefixes each sub-agent tool name, matching the
// `mcp__subagents__` names the Claude Agent SDK adapter exposes.
const SubAgentToolPrefix = "mcp__subagents__"

var unsafeToolChars = regexp.MustCompile(`[^a-zA-Z0-9_-]`)

// AdapterOptions configure ToClaudeAgents.
type AdapterOptions struct {
	// ToolHandlers are the tools offered to every node. NativeTool values are
	// tracked under their map key.
	ToolHandlers map[string]aiserver.ToolHandler
	// Handler configures the Anthropic client the nodes run on.
	Handler Options
}

// GraphRunner runs a GraphDefinition as nested agents: every child node is a
// sub-agent tool of its parent, so the root drives the whole run.
type GraphRunner struct {
	def   *aiserver.GraphDefinition
	tools map[string]aiserver.ToolHandler
	agent *agent
}

// ToClaudeAgents adapts a resolved graph to nested Claude agents
// (toClaudeAgents). opts may be nil. The definition's user context is the
// context every tracking event is sent for.
func ToClaudeAgents(def *aiserver.GraphDefinition, opts *AdapterOptions) *GraphRunner {
	aiserver.ReportUsage("claude-agents.toClaudeAgents", PackageName, PackageVersion)
	if opts == nil {
		opts = &AdapterOptions{}
	}
	return &GraphRunner{def: def, tools: opts.ToolHandlers, agent: newAgent(opts.Handler)}
}

// subAgent is one node with the sub-agent tools for its children.
type subAgent struct {
	node  *aiserver.GraphNode
	tools []localTool
}

// graphRun is one Invoke's mutable state.
type graphRun struct {
	*GraphRunner
	userCtx   ldcontext.Context
	tracking  bool // a run without a context tracks no events
	runTD     aiserver.TrackData
	variables map[string]any

	mu    sync.Mutex
	path  []string
	usage aiserver.TokenUsage
}

// Invoke runs the root agent with input and returns its result as the
// graph response. variables and history may be nil; history reaches the
// root only.
func (g *GraphRunner) Invoke(ctx context.Context, input string, opts ...aiserver.CallOption) (aiserver.GraphResponse, error) {
	co := aiserver.NewCallOptions(opts...)
	variables, history := co.Variables, co.History
	if g.def == nil || !g.def.Enabled || g.def.Root == nil {
		key := ""
		if g.def != nil {
			key = g.def.Key
		}
		return aiserver.GraphResponse{}, fmt.Errorf("%w: %s", aiserver.ErrGraphDisabled, key)
	}
	userCtx := g.def.UserContext()
	run := &graphRun{GraphRunner: g, userCtx: userCtx, tracking: userCtx.IsDefined(), runTD: g.def.TrackData(), variables: variables}
	run.runTD.RunID = uuid.NewString()

	ctx, span := otel.Tracer(PackageName).Start(ctx, aiserver.GraphSpanName)
	defer span.End()
	span.SetAttributes(attribute.String("ld.ai.graph.key", g.def.Key))
	aiserver.SetLDSpanAttributesFor(span, run.runTD, ldcontext.Context{}) // identity belongs on handler root spans only
	start := time.Now()

	out, err := run.execute(ctx, input, history)
	run.graphEvent("$ld:ai:graph:duration:total", nil, float64(time.Since(start).Milliseconds()))
	if err != nil {
		run.graphEvent("$ld:ai:graph:invocation_failure", nil, 1)
		span.RecordError(err)
		span.SetStatus(codes.Error, err.Error())
		return aiserver.GraphResponse{}, err
	}
	run.mu.Lock()
	usage, path := run.usage, strings.Join(run.path, "->")
	run.mu.Unlock()
	if usage.Total > 0 {
		run.graphEvent("$ld:ai:graph:total_tokens", nil, usage.Total)
	}
	run.graphEvent("$ld:ai:graph:invocation_success", nil, 1)
	span.SetAttributes(attribute.String("launchdarkly.graph.path", path))
	aiserver.SetUsageSpanAttributes(span, aiserver.SpanUsage{Input: usage.Input, Output: usage.Output})
	span.SetStatus(codes.Ok, "")
	return aiserver.GraphResponse{Response: out, Usage: usage}, nil
}

// execute builds sub-agents leaf-first, then runs the root.
func (r *graphRun) execute(ctx context.Context, input string, history []aiserver.Message) (string, error) {
	v, err := r.def.ReverseTraverse(ctx, func(_ context.Context, n *aiserver.GraphNode, acc map[string]any) (any, error) {
		sa := &subAgent{node: n}
		taken := map[string]bool{}
		for _, e := range r.def.EdgesFrom(n.Key) {
			if child, ok := acc[e.TargetKey].(*subAgent); ok { // absent on a cycle back-edge
				name := subAgentToolName(child.node.Key, taken)
				taken[name] = true
				sa.tools = append(sa.tools, r.subAgentTool(n, e, child, name))
			}
		}
		return sa, nil
	})
	if err != nil {
		return "", err
	}
	root, ok := v.(*subAgent)
	if !ok { // no terminal node to start the reverse walk from
		root = &subAgent{node: r.def.Root}
	}
	r.markNode(root.node)
	return r.runNode(ctx, root, input, history)
}

// subAgentTool exposes child to parent as a tool taking one `input` string.
func (r *graphRun) subAgentTool(parent *aiserver.GraphNode, e aiserver.GraphEdge, child *subAgent, name string) localTool {
	desc, _ := e.Handoff["description"].(string)
	if desc == "" {
		desc = truncateRunes(child.node.Config.Instructions, 120)
	}
	if desc == "" {
		desc = child.node.Key
	}
	schema := map[string]any{
		"type":       "object",
		"properties": map[string]any{"input": map[string]any{"type": "string", "description": "Task or question for this agent"}},
		"required":   []any{"input"},
	}
	return localTool{name: name, param: anthropicwire.FunctionTool(name, desc, schema), fn: func(ctx context.Context, args map[string]any) (any, error) {
		r.markNode(child.node)
		in, _ := args["input"].(string)
		out, err := r.runNode(ctx, child, in, nil)
		// The handoff succeeded only once the child has answered.
		event := "$ld:ai:graph:handoff_success"
		if err != nil {
			event = "$ld:ai:graph:handoff_failure"
		}
		r.graphEvent(event, map[string]any{"sourceKey": parent.Key, "targetKey": child.node.Key}, 1)
		return out, err
	}}
}

// maxToolNameLen is the Anthropic tool name limit (^[a-zA-Z0-9_-]{1,64}$).
const maxToolNameLen = 64

// subAgentToolName is SubAgentToolPrefix plus key with unsafe characters
// replaced by "_". A name that is too long, or that another child of the
// same parent already uses (keys "a.b" and "a_b"), is cut and given a hash
// of the key so it stays valid and unique.
func subAgentToolName(key string, taken map[string]bool) string {
	name := SubAgentToolPrefix + unsafeToolChars.ReplaceAllString(key, "_")
	if len(name) <= maxToolNameLen && !taken[name] {
		return name
	}
	h := fnv.New32a()
	h.Write([]byte(key))
	suffix := fmt.Sprintf("_%08x", h.Sum32())
	return name[:min(len(name), maxToolNameLen-len(suffix))] + suffix
}

// markNode emits $ld:ai:graph:node the first time a node runs.
func (r *graphRun) markNode(n *aiserver.GraphNode) {
	r.mu.Lock()
	index := len(r.path)
	for _, k := range r.path {
		if k == n.Key {
			r.mu.Unlock()
			return
		}
	}
	r.path = append(r.path, n.Key)
	r.mu.Unlock()
	r.graphEvent("$ld:ai:graph:node", map[string]any{"nodeKey": n.Key, "index": index}, 1)
}

// graphEvent tracks a `$ld:ai:graph:*` event with the run's graph track data,
// as aiserver.Graph does.
func (r *graphRun) graphEvent(event string, extra map[string]any, metric float64) {
	if r.tracking {
		aiserver.TrackGraphEvent(event, r.userCtx, r.runTD, extra, metric)
	}
}

// nodeEvent tracks a per-node event with that node's track data.
func (r *graphRun) nodeEvent(event string, td aiserver.TrackData, metric float64) {
	if r.tracking {
		aiserver.Track(event, r.userCtx, td, metric)
	}
}

func (r *graphRun) addUsage(u aiserver.TokenUsage) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.usage.Input += u.Input
	r.usage.Output += u.Output
	r.usage.Total += u.Total
}

func isAnthropic(n *aiserver.GraphNode) bool {
	p := strings.ToLower(n.Config.Provider.Name)
	return p == "anthropic" || p == "claude"
}

// runNode runs one node: an Anthropic node through the agent loop with its
// sub-agent tools, any other provider through the definition's handlers.
func (r *graphRun) runNode(ctx context.Context, sa *subAgent, input string, history []aiserver.Message) (string, error) {
	n := sa.node
	if !isAnthropic(n) {
		res, err := r.def.RunNode(ctx, n, input, aiserver.RunNodeOptions{ToolHandlers: r.tools, Variables: r.variables, History: history})
		if err != nil {
			return "", err
		}
		r.addUsage(res.Usage)
		if s, ok := res.Response.(string); ok {
			return s, nil
		}
		return aiserver.ToolResultString(res.Response), nil
	}

	td := n.TrackData(r.def.Key, r.runTD.RunID)
	vars := maps.Clone(r.variables)
	if vars == nil {
		vars = map[string]any{}
	}
	vars[aiserver.VarLDContext] = aiserver.ContextToMap(r.userCtx)
	delete(vars, aiserver.VarLD) // the association is written from spanIdentity
	tools := r.tools
	if r.tracking {
		tools = aiserver.WrapToolHandlers(tools, r.userCtx, td)
	}
	start := time.Now()
	res, err := r.agent.run(ctx, aiserver.HandlerRequest{
		Config: n.Config, UserInput: input, Variables: vars, History: history, ToolHandlers: tools,
	}, sa.tools, &spanIdentity{td: td, userCtx: r.userCtx}, nil)
	r.nodeEvent("$ld:ai:duration:total", td, float64(time.Since(start).Milliseconds()))
	if err != nil {
		r.nodeEvent("$ld:ai:generation:error", td, 1)
		return "", fmt.Errorf("graph node %s: %w", n.Key, err)
	}
	u := aiserver.ParseUsage(res.usage.Raw())
	r.nodeEvent("$ld:ai:generation:success", td, 1)
	for event, v := range map[string]float64{"$ld:ai:tokens:total": u.Total, "$ld:ai:tokens:input": u.Input, "$ld:ai:tokens:output": u.Output} {
		if v > 0 {
			r.nodeEvent(event, td, v)
		}
	}
	r.addUsage(u)
	return res.final, nil
}

func truncateRunes(s string, n int) string {
	if r := []rune(s); len(r) > n {
		return string(r[:n])
	}
	return s
}
