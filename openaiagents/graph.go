package openaiagents

import (
	"context"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/responses"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/openaiwire"
	"github.com/launchdarkly/go-ai-sdk/internal/sdkapi"
)

// OpenAIGraph returns aiserver's model-driven graph runner with this
// package's agent handler as the only handler (openaiGraph). Any
// options.Handlers value is replaced, so the graph cannot be bound to a
// different handler. handler configures that agent handler.
func OpenAIGraph(key string, options aiserver.GraphOptions, handler Options) *aiserver.AgentGraph {
	aiserver.ReportUsage("openai-agents.openaiGraph", PackageName, PackageVersion)
	options.Handlers = []*aiserver.ProviderHandler{newHandler(handler)}
	return sdkapi.Graph(key, options)
}

// NativeGraphOptions configure ToOpenAIAgents.
type NativeGraphOptions struct {
	// ToolHandlers supply each node's tools by config tool key.
	ToolHandlers map[string]aiserver.ToolHandler
	// Context is the LaunchDarkly context events are tracked for. Without
	// one (the zero value), the runner tracks no events.
	Context ldcontext.Context
	// Handler supplies the OpenAI client and the default MaxTurns.
	Handler Options
}

// NativeGraph runs a resolved graph as one multi-agent run: one agent per
// node, a handoff tool per outgoing edge, starting at the root.
type NativeGraph struct {
	def    *aiserver.GraphDefinition
	opts   NativeGraphOptions
	client *openai.Client
}

// ToOpenAIAgents translates def into an agent tree run natively on the
// Responses API (toOpenAIAgents). opts may be nil. Errors for
// a nil, disabled or rootless definition are returned by Invoke.
func ToOpenAIAgents(def *aiserver.GraphDefinition, opts *NativeGraphOptions) *NativeGraph {
	aiserver.ReportUsage("openai-agents.toOpenAIAgents", PackageName, PackageVersion)
	if opts == nil {
		opts = &NativeGraphOptions{}
	}
	return &NativeGraph{def: def, opts: *opts, client: opts.Handler.client()}
}

// buildAgents makes one agent per reachable node, then wires a handoff for
// each outgoing edge, so cycles need no build order. Instructions come from
// each node's config with variables applied; tools are tracked for userCtx
// when tracking is true.
func (g *NativeGraph) buildAgents(ctx context.Context, variables map[string]any, runID string, tracking bool) (map[string]*agent, error) {
	agents := map[string]*agent{}
	var order []*aiserver.GraphNode
	if _, err := g.def.Traverse(ctx, func(_ context.Context, n *aiserver.GraphNode, _ map[string]any) (any, error) {
		handlers := g.opts.ToolHandlers
		if tracking {
			handlers = aiserver.WrapToolHandlers(handlers, g.opts.Context, n.TrackData(g.def.Key, runID))
		}
		system := aiserver.ComposePrompt(aiserver.HandlerRequest{Config: n.Config, Variables: variables}, aiserver.ModeAgent).System
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
				return nil, fmt.Errorf("openaiagents: graph %q: edge %s targets unknown node %q", g.def.Key, e.Key, e.TargetKey)
			}
			desc, _ := e.Handoff["description"].(string)
			agents[n.Key].addHandoff(target, e.TargetKey, desc)
		}
	}
	return agents, nil
}

// Invoke runs the graph from the root with input; variables fill each node's
// instruction templates and history (root only) precedes input as
// structured items. It returns the final agent's output and the
// run's total usage. variables and history may be nil.
func (g *NativeGraph) Invoke(ctx context.Context, input string, opts ...aiserver.CallOption) (aiserver.GraphResponse, error) {
	co := aiserver.NewCallOptions(opts...)
	variables, history := co.Variables, co.History
	def := g.def
	switch {
	case def == nil:
		return aiserver.GraphResponse{}, ErrNilGraph
	case !def.Enabled:
		return aiserver.GraphResponse{}, fmt.Errorf("%w: %q", aiserver.ErrGraphDisabled, def.Key)
	case def.Root == nil:
		return aiserver.GraphResponse{}, fmt.Errorf("%w: %q", ErrNoRoot, def.Key)
	}
	userCtx := g.opts.Context
	runID := uuid.NewString()
	graphTD := def.TrackData()
	graphTD.RunID = runID

	spanCtx, span := otel.Tracer(PackageName).Start(ctx, aiserver.GraphSpanName)
	defer span.End()
	span.SetAttributes(attribute.String("ld.ai.graph.key", def.Key))
	aiserver.SetLDSpanAttributesFor(span, graphTD, ldcontext.Context{}) // identity belongs on handler root spans only
	start := time.Now()

	n := &nativeRun{g: g, ctx: spanCtx, userCtx: userCtx, tracking: userCtx.IsDefined(), graphTD: graphTD, runID: runID}
	defer func() { // node spans open in hooks, so a panic is caught here, not at StartHandlerRun
		if p := recover(); p != nil {
			if n.seg != nil {
				n.seg.span.Fail(fmt.Errorf("panic: %v", p))
			}
			span.SetStatus(codes.Error, fmt.Sprint("panic: ", p))
			panic(p)
		}
	}()
	res, err := g.run(spanCtx, variables, n, input, history)
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
	usage := aiserver.TokenUsage{Input: float64(res.usage.Input), Output: float64(res.usage.Output)}
	usage.Total = usage.Input + usage.Output
	span.SetAttributes(attribute.String("launchdarkly.graph.path", strings.Join(n.path, "->")))
	aiserver.SetUsageSpanAttributes(span, aiserver.SpanUsage{Input: usage.Input, Output: usage.Output})
	if usage.Total > 0 {
		n.graphEvent("$ld:ai:graph:total_tokens", nil, usage.Total)
	}
	n.graphEvent("$ld:ai:graph:invocation_success", nil, 1)
	span.SetStatus(codes.Ok, "")
	return aiserver.GraphResponse{Response: aiserver.ToolResultString(res.output), Usage: usage}, nil
}

// nativeRun is one Invoke's tracking state. Each stretch of the run one agent
// is active (from agent_start to its handoff or its final answer) is a node
// run: it gets its own root span and its own per-node events, as a node run
// by aiserver.Graph does.
type nativeRun struct {
	g         *NativeGraph
	ctx       context.Context // the graph span's; node spans start under it
	userCtx   ldcontext.Context
	tracking  bool // a run without a context tracks no events
	graphTD   aiserver.TrackData
	runID     string
	runner    *runner
	rootTurns []aiserver.Message // the root's input, for its span's prompt content

	path []string
	seg  *segment // the active agent's node run
	from *agent   // the agent that handed off to the next one
}

type segment struct {
	a     *agent
	from  *agent // nil for the root
	span  *aiserver.HandlerRun
	start time.Time
	usage aiserver.SpanUsage
	model string
	text  string // the last turn's text
}

func (n *nativeRun) graphEvent(event string, extra map[string]any, metric float64) {
	if n.tracking {
		aiserver.TrackGraphEvent(event, n.userCtx, n.graphTD, extra, metric)
	}
}

func (n *nativeRun) nodeEvent(event string, td aiserver.TrackData, metric float64) {
	if n.tracking {
		aiserver.Track(event, n.userCtx, td, metric)
	}
}

func (n *nativeRun) hooks() hooks {
	return hooks{
		agentStart: n.agentStart,
		agentEnd:   func(a *agent, output any) { n.endSegment(nil, aiserver.ToolResultString(output)) },
		handoff: func(from, _ *agent) {
			n.endSegment(nil, n.seg.text)
			n.from = from
		},
		turnEnd: func(a *agent, resp *responses.Response) {
			n.seg.usage.Add(openaiwire.Usage(resp.Usage))
			n.seg.model = openaiwire.ResponseModel(resp, a.model)
			n.seg.text = resp.OutputText()
		},
	}
}

// agentStart records a node's first entry ($ld:ai:graph:node, once per node)
// and opens the node run.
func (n *nativeRun) agentStart(a *agent) {
	def := n.g.def
	if !slices.Contains(n.path, a.key) {
		n.path = append(n.path, a.key)
		n.graphEvent("$ld:ai:graph:node", map[string]any{"nodeKey": a.key, "index": len(n.path) - 1}, 1)
	}
	p := aiserver.Prompt{System: a.instructions}
	if n.from == nil {
		p.Turns = n.rootTurns // a handoff target continues the conversation through previous_response_id
	}
	span := aiserver.StartHandlerRun(n.ctx, aiserver.HandlerSpanOptions{
		Tracer: PackageName, Name: SpanName, System: "openai", Operation: operation, RequestModel: a.model,
	}, aiserver.HandlerRequest{}, p)
	aiserver.SetLDSpanAttributesFor(span.Root, def.GetNode(a.key).TrackData(def.Key, n.runID), n.userCtx)
	n.runner.span = span
	n.seg = &segment{a: a, from: n.from, span: span, start: time.Now()}
}

// endSegment ends the active node run: its span, its per-node events, and
// the handoff into it, which succeeded only if this node answered.
func (n *nativeRun) endSegment(err error, output string) {
	s := n.seg
	if s == nil {
		return
	}
	n.seg = nil
	def := n.g.def
	td := def.GetNode(s.a.key).TrackData(def.Key, n.runID)
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

// run builds the agents and runs the root; one run spans every handoff, so
// the turn cap is the root node's (as in the JavaScript and Python adapters).
func (g *NativeGraph) run(ctx context.Context, variables map[string]any, n *nativeRun, input string, history []aiserver.Message) (runResult, error) {
	agents, err := g.buildAgents(ctx, variables, n.runID, n.tracking)
	if err != nil {
		return runResult{}, err
	}
	root := g.def.Root
	// History seeds the root only; handoff targets see the conversation
	// through previous_response_id, never the original history array.
	turns := aiserver.ComposePrompt(aiserver.HandlerRequest{UserInput: input, History: history}, aiserver.ModeAgent).Turns
	n.rootTurns = turns
	r := &runner{client: g.client, maxTurns: maxTurns(root.Config.Model.Parameters, g.opts.Handler.MaxTurns)}
	n.runner, r.hooks = r, n.hooks()
	res, _, err := r.run(ctx, agents[root.Key], openaiwire.Turns(turns))
	return res, err
}
