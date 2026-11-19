package litellmagents

import (
	"cmp"
	"context"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/agentloop"
	"github.com/launchdarkly/go-ai-sdk/internal/handlerkit"
	"github.com/launchdarkly/go-ai-sdk/internal/kitbase"
	"github.com/launchdarkly/go-ai-sdk/internal/sdkapi"
)

// LiteLLMGraph returns aiserver's model-driven graph runner with this
// package's agent handler as the only handler (litellmGraph). Any
// options.Handlers value is replaced. handler configures that agent
// handler, proxy connection included; the error is NewHandler's.
func LiteLLMGraph(key string, options aiserver.GraphOptions, handler Options) (*aiserver.AgentGraph, error) {
	handlerkit.ReportUsage("litellm-agents.litellmGraph", PackageName, PackageVersion)
	h, err := newHandler(handler)
	if err != nil {
		return nil, err
	}
	options.Handlers = []*aiserver.ProviderHandler{h}
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
type NativeGraph struct {
	def  *aiserver.GraphDefinition
	opts NativeGraphOptions
}

// ToLiteLLMAgents translates def into an agent tree run natively over the
// LiteLLM proxy (toLiteLLMAgents). opts may be nil. Errors for a nil,
// disabled or rootless definition, and for a missing proxy, are returned by
// Invoke, which records them on the graph span. The error is for an
// invalid option, and nil for now.
func ToLiteLLMAgents(def *aiserver.GraphDefinition, opts *NativeGraphOptions) (*NativeGraph, error) {
	handlerkit.ReportUsage("litellm-agents.toLiteLLMAgents", PackageName, PackageVersion)
	if opts == nil {
		opts = &NativeGraphOptions{}
	}
	return &NativeGraph{def: def, opts: *opts}, nil
}

// buildAgents makes one agent per reachable node, then wires a handoff for
// each outgoing edge, so cycles need no build order. Instructions come from
// each node's config with variables applied; tools are tracked for the
// context when tracking is true.
func (g *NativeGraph) buildAgents(ctx context.Context, variables map[string]any, runID string, tracking bool) (map[string]*agent, error) {
	agents := map[string]*agent{}
	var order []*aiserver.GraphNode
	if _, err := g.def.Traverse(ctx, func(_ context.Context, n *aiserver.GraphNode, _ map[string]any) (any, error) {
		handlers := g.opts.ToolHandlers
		if tracking {
			handlers = handlerkit.WrapToolHandlers(handlers, g.opts.Context, handlerkit.NodeTrackData(n, g.def.Key, runID))
		}
		system := handlerkit.ComposePrompt(aiserver.HandlerRequest{Config: n.Config, Variables: variables}, aiserver.ModeAgent).System
		a := newAgent(sanitizeName(n.Key), system, n.Config, handlers)
		a.key = n.Key
		agents[n.Key] = a
		order = append(order, n)
		return nil, nil
	}); err != nil {
		return nil, err
	}
	for _, n := range order {
		for _, e := range g.def.EdgesFrom(n.Key) {
			target := agents[e.TargetKey]
			if target == nil {
				return nil, fmt.Errorf("litellmagents: graph %q: edge %s targets unknown node %q", g.def.Key, e.Key, e.TargetKey)
			}
			desc, _ := e.Handoff["description"].(string)
			agents[n.Key].addHandoff(target, e.TargetKey, desc)
		}
	}
	return agents, nil
}

// Invoke runs the graph from the root with input; variables fill each node's
// instruction templates and history precedes input at the root. It returns
// the final agent's output and the run's total usage.
func (g *NativeGraph) Invoke(ctx context.Context, input string, opts ...aiserver.CallOption) (aiserver.GraphResponse, error) {
	co := aiserver.NewCallOptions(opts...)
	def := g.def
	switch {
	case def == nil:
		return aiserver.GraphResponse{}, ErrNilGraph
	case !def.Enabled:
		return aiserver.GraphResponse{}, fmt.Errorf("%w: %q", aiserver.ErrGraphDisabled, def.Key)
	}
	userCtx := g.opts.Context
	runID := uuid.NewString()
	graphTD := handlerkit.GraphTrackData(def)
	graphTD.ConfigKey = cmp.Or(graphTD.ConfigKey, def.Key) // a definition not built by ResolveGraph has no track data
	graphTD.RunID = runID

	// Nodes run the agent loop directly, not through a handler, so the run
	// binds Handler.CaptureContent itself.
	ctx = sdkapi.StartInvocation(ctx, g.opts.Handler.CaptureContent)
	spanCtx, span := otel.Tracer(PackageName).Start(ctx, kitbase.GraphSpanName)
	defer span.End()
	span.SetAttributes(attribute.String("launchdarkly.graph.key", def.Key))
	// The graph flag is not an AI Config, so its span has no variation.
	spanTD := graphTD
	spanTD.VariationKey = ""
	handlerkit.SetLDSpanAttributesFor(span, spanTD, userCtx)
	start := time.Now()

	n := &nativeRun{g: g, ctx: spanCtx, userCtx: userCtx, tracking: userCtx.IsDefined(), graphTD: graphTD, runID: runID}
	defer func() { // node spans open in hooks, so a panic is caught here
		if p := recover(); p != nil {
			if n.seg != nil {
				n.seg.span.Fail(fmt.Errorf("panic: %v", p))
			}
			span.SetStatus(codes.Error, fmt.Sprint("panic: ", p))
			panic(p)
		}
	}()
	res, err := g.run(spanCtx, co.Variables, n, input, co.History)
	if err != nil {
		n.endSegment(err, "")
	}
	n.graphEvent("$ld:ai:graph:duration:total", nil, float64(time.Since(start).Milliseconds()))
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, err.Error())
		n.graphEvent("$ld:ai:graph:invocation_failure", nil, 1)
		return aiserver.GraphResponse{}, err
	}
	usage := aiserver.TokenUsage{Input: float64(res.Tokens.Input), Output: float64(res.Tokens.Output)}
	usage.Total = usage.Input + usage.Output
	span.SetAttributes(attribute.String("launchdarkly.graph.path", strings.Join(n.path, "->")))
	kitbase.SetUsageSpanAttributes(span, kitbase.SpanUsage{Input: usage.Input, Output: usage.Output})
	if usage.Total > 0 {
		n.graphEvent("$ld:ai:graph:total_tokens", nil, usage.Total)
	}
	n.graphEvent("$ld:ai:graph:invocation_success", nil, 1)
	span.SetStatus(codes.Ok, "")
	return aiserver.GraphResponse{Response: aiserver.ToolResultString(res.Output), Usage: usage}, nil
}

// nativeRun is one Invoke's tracking state. Each stretch of the run one agent
// is active (from agent_start to its handoff or its final answer) is a node
// run with its own child spans and per-node events.
type nativeRun struct {
	g         *NativeGraph
	ctx       context.Context // the graph span's; node spans start under it
	userCtx   ldcontext.Context
	tracking  bool // a run without a context tracks no events
	graphTD   aiserver.TrackData
	runID     string
	runner    *agentloop.Runner
	rootTurns []aiserver.Message // the root's input, for its span's prompt content

	path []string
	seg  *segment // the active agent's node run
	from *agent   // the agent that handed off to the next one
}

type segment struct {
	a     *agent
	from  *agent // nil for the root
	span  kitbase.Run
	start time.Time
	usage kitbase.SpanUsage
	model string
	text  string // the last turn's text
}

func (n *nativeRun) graphEvent(event string, extra map[string]any, metric float64) {
	if n.tracking {
		handlerkit.TrackGraphEvent(event, n.userCtx, n.graphTD, extra, metric)
	}
}

func (n *nativeRun) nodeEvent(event string, td aiserver.TrackData, metric float64) {
	if n.tracking {
		aiserver.Track(event, n.userCtx, td, metric)
	}
}

func (n *nativeRun) hooks() agentloop.Hooks {
	return agentloop.Hooks{
		AgentStart: func(a agentloop.Agent) { n.agentStart(a.(*agent)) },
		AgentEnd:   func(_ agentloop.Agent, output any) { n.endSegment(nil, aiserver.ToolResultString(output)) },
		Handoff: func(from, _ agentloop.Agent) {
			n.endSegment(nil, n.seg.text)
			n.from = from.(*agent)
		},
		TurnEnd: func(_ agentloop.Agent, t agentloop.Turn) {
			n.seg.usage.Add(t.Usage)
			n.seg.model = t.Model
			n.seg.text = t.Text
		},
	}
}

// agentStart records a node's first entry ($ld:ai:graph:node, once per node)
// and opens the node run.
func (n *nativeRun) agentStart(a *agent) {
	if !slices.Contains(n.path, a.key) {
		n.path = append(n.path, a.key)
		n.graphEvent("$ld:ai:graph:node", map[string]any{"nodeKey": a.key, "index": len(n.path) - 1}, 1)
	}
	p := handlerkit.Prompt{System: a.instructions}
	if n.from == nil {
		p.Turns = n.rootTurns
	}
	span := handlerkit.StartHandlerRun(n.ctx, kitbase.HandlerSpanOptions{
		Tracer: PackageName, System: system, Operation: operation, RequestModel: a.model,
		ProviderName: n.providerName(a), NoRoot: true,
	}, aiserver.HandlerRequest{}, p) // a node's spans nest under the graph span
	n.runner.Span = span
	n.seg = &segment{a: a, from: n.from, span: span, start: time.Now()}
}

// providerName is gen_ai.provider.name for a node, as on the handler's span.
func (n *nativeRun) providerName(a *agent) string {
	return spanOptions(n.g.def.GetNode(a.key).Config).ProviderName
}

// endSegment ends the active node run: its spans, its per-node events, and
// the handoff into it, which succeeded only if this node answered.
func (n *nativeRun) endSegment(err error, output string) {
	s := n.seg
	if s == nil {
		return
	}
	n.seg = nil
	def := n.g.def
	td := handlerkit.NodeTrackData(def.GetNode(s.a.key), def.Key, n.runID)
	n.nodeEvent("$ld:ai:duration:total", td, float64(time.Since(s.start).Milliseconds()))
	handoff := "$ld:ai:graph:handoff_success"
	if err != nil {
		s.span.Fail(err)
		n.nodeEvent("$ld:ai:generation:error", td, 1)
		handoff = "$ld:ai:graph:handoff_failure"
	} else {
		s.span.Finish(s.model, output)
		n.nodeEvent("$ld:ai:generation:success", td, 1)
		for event, v := range map[string]float64{"$ld:ai:tokens:total": s.usage.Input + s.usage.Output, "$ld:ai:tokens:input": s.usage.Input, "$ld:ai:tokens:output": s.usage.Output} {
			if v > 0 {
				n.nodeEvent(event, td, v)
			}
		}
	}
	if s.from != nil {
		n.graphEvent(handoff, map[string]any{"sourceKey": s.from.key, "targetKey": s.a.key}, 1)
	}
}

// run resolves the proxy client, builds the agents and runs the root. One
// client serves every node, picked for the root's config; one run spans
// every handoff, so the turn cap is the root node's.
func (g *NativeGraph) run(ctx context.Context, variables map[string]any, n *nativeRun, input string, history []aiserver.Message) (agentloop.Result, error) {
	root := g.def.Root
	if root == nil {
		return agentloop.Result{}, fmt.Errorf("%w: %q", ErrNoRoot, g.def.Key)
	}
	connect, err := g.opts.Handler.connect()
	if err != nil {
		return agentloop.Result{}, err
	}
	client, err := connect(root.Config)
	if err != nil {
		return agentloop.Result{}, err
	}
	agents, err := g.buildAgents(ctx, variables, n.runID, n.tracking)
	if err != nil {
		return agentloop.Result{}, err
	}
	// History seeds the root's turns. Chat Completions resends the whole
	// conversation every turn, so a handoff target sees it too.
	turns := handlerkit.ComposePrompt(aiserver.HandlerRequest{UserInput: input, History: history}, aiserver.ModeAgent).Turns
	n.rootTurns = turns
	r := newRunner(client, agentloop.MaxTurns(root.Config.Model.Parameters, g.opts.Handler.MaxTurns))
	n.runner, r.Hooks = r, n.hooks()
	res, _, err := r.Run(ctx, agents[root.Key], turns)
	return res, err
}
