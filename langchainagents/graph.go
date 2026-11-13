package langchainagents

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"regexp"
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/handlerkit"
	"github.com/launchdarkly/go-ai-sdk/internal/kitbase"
	"github.com/launchdarkly/go-ai-sdk/internal/langchainwire"
	"github.com/launchdarkly/go-ai-sdk/internal/sdkapi"
	"github.com/launchdarkly/go-ai-sdk/internal/stategraph"
	"github.com/launchdarkly/go-ai-sdk/llms"
)

// LangChainGraph returns aiserver's model-driven graph runner with this
// package's agent handler as the only handler. Any options.Handlers value is replaced, so the graph
// cannot be bound to a different handler. handler supplies the model: its Model or ModelFactory runs every node.
// The error is NewHandler's, and nil for now.
func LangChainGraph(key string, options aiserver.GraphOptions, handler Options) (*aiserver.AgentGraph, error) {
	handlerkit.ReportUsage("langchain-agents.langchainGraph", PackageName, PackageVersion)
	options.Handlers = []*aiserver.ProviderHandler{newHandler(handler)}
	return sdkapi.Graph(key, options), nil
}

// NativeGraphOptions configure ToLangGraph.
type NativeGraphOptions struct {
	// ToolHandlers supply each node's tools by config tool key.
	ToolHandlers map[string]aiserver.ToolHandler
	// ModelFactory builds a node's model (the JavaScript modelFactory(node)).
	// When nil, the model is chosen from the node's config.provider.name, as
	// the handler does.
	ModelFactory func(node *aiserver.GraphNode) (llms.Model, error)
	// Context is the LaunchDarkly context events are tracked for. Without
	// one (the zero value), the runner tracks no events.
	Context ldcontext.Context
}

var _ aiserver.NativeGraphRunner = (*LangGraph)(nil)

// LangGraph runs a resolved graph as one stategraph.Graph: one node
// per GraphNode, each a ReAct agent with its tools and a transfer_to_<key>
// handoff tool per outgoing edge.
type LangGraph struct {
	def  *aiserver.GraphDefinition
	opts NativeGraphOptions
}

// ToLangGraph translates def into a stategraph runner.
// opts may be nil. Errors for a nil, disabled or
// rootless definition are returned by Invoke.
// The error is for an invalid option, and nil for now.
func ToLangGraph(def *aiserver.GraphDefinition, opts *NativeGraphOptions) (*LangGraph, error) {
	handlerkit.ReportUsage("langchain-agents.toLangGraph", PackageName, PackageVersion)
	if opts == nil {
		opts = &NativeGraphOptions{}
	}
	return &LangGraph{def: def, opts: *opts}, nil
}

var unsafeNameChars = regexp.MustCompile(`[^a-zA-Z0-9_-]`)

// sanitizeName makes a config key a graph node and tool name, as the
// JavaScript adapter does.
func sanitizeName(key string) string { return unsafeNameChars.ReplaceAllString(key, "_") }

// handoffIgnored answers a hand-off call after the first one in a node run.
const handoffIgnored = "Ignored: only the first transfer in a turn routes."

func handoffToolName(targetKey string) string { return "transfer_to_" + sanitizeName(targetKey) }

// handoffCommand is a transfer_to_<target> tool's result: a Command routing
// the graph to the target node.
func handoffCommand(targetKey string) *stategraph.Command {
	return &stategraph.Command{Goto: sanitizeName(targetKey)}
}

// Invoke runs the graph from the root with input; variables fill each node's
// instruction templates and history (root only) precedes input as messages.
// It returns the last AI message's text and the run's total usage.
func (g *LangGraph) Invoke(ctx context.Context, input string, opts ...aiserver.CallOption) (aiserver.GraphResponse, error) {
	co := aiserver.NewCallOptions(opts...)
	def := g.def
	switch {
	case def == nil:
		return aiserver.GraphResponse{}, ErrNilGraph
	case !def.Enabled:
		return aiserver.GraphResponse{}, fmt.Errorf("%w: %q", aiserver.ErrGraphDisabled, def.Key)
	case def.Root == nil:
		return aiserver.GraphResponse{}, fmt.Errorf("%w: %q", ErrNoRoot, def.Key)
	}
	r := &graphRun{
		g: g, variables: co.Variables, runID: uuid.NewString(),
		userCtx: g.opts.Context, tracking: g.opts.Context.IsDefined(),
	}
	r.graphTD = handlerkit.GraphTrackData(def)
	r.graphTD.RunID = r.runID

	// Nodes run without a handler, so the run binds the process default
	// (Configure) itself, as an invocation.
	ctx = sdkapi.StartInvocation(ctx, aiserver.CaptureDefault)
	spanCtx, span := otel.Tracer(PackageName).Start(ctx, kitbase.GraphSpanName)
	defer span.End()
	span.SetAttributes(attribute.String("launchdarkly.graph.key", def.Key))
	// The graph flag is not an AI Config, so its span has no variation, as in the
	// other SDKs: no variation key and no feature_flag.result.* attributes.
	spanTD := r.graphTD
	spanTD.VariationKey = ""
	handlerkit.SetLDSpanAttributesFor(span, spanTD, r.userCtx) // a node opens no root span, so the run's context identity goes here
	start := time.Now()

	response, err := r.run(spanCtx, input, co.History)
	r.graphEvent("$ld:ai:graph:duration:total", nil, float64(time.Since(start).Milliseconds()))
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, err.Error())
		r.graphEvent("$ld:ai:graph:invocation_failure", nil, 1)
		return aiserver.GraphResponse{}, err
	}
	usage := aiserver.TokenUsage{Input: r.usage.Input, Output: r.usage.Output, Total: r.usage.Input + r.usage.Output}
	span.SetAttributes(attribute.String("launchdarkly.graph.path", strings.Join(r.path, "->")))
	kitbase.SetUsageSpanAttributes(span, r.usage)
	if usage.Total > 0 {
		r.graphEvent("$ld:ai:graph:total_tokens", nil, usage.Total)
	}
	r.graphEvent("$ld:ai:graph:invocation_success", nil, 1)
	span.SetStatus(codes.Ok, "")
	return aiserver.GraphResponse{Response: response, Usage: usage}, nil
}

// graphRun is one Invoke: node functions run one at a time and record the
// path, usage and the first error here.
type graphRun struct {
	g         *LangGraph
	variables map[string]any
	runID     string
	userCtx   ldcontext.Context
	tracking  bool
	graphTD   aiserver.TrackData

	// rootHistory is the caller's history, read only by the root node.
	rootHistory []llms.MessageContent

	path  []string
	usage kitbase.SpanUsage
	steps int
	err   error
	// prev is the node that ran last: the source of the next node's handoff.
	prev *aiserver.GraphNode
}

// graphEvent tracks a `$ld:ai:graph:*` event with the run's graph track data,
// as aiserver.Graph does.
func (r *graphRun) graphEvent(event string, extra map[string]any, metric float64) {
	if r.tracking {
		handlerkit.TrackGraphEvent(event, r.userCtx, r.graphTD, extra, metric)
	}
}

func (r *graphRun) fail(err error) (any, error) {
	if r.err == nil {
		r.err = err
	}
	return nil, err
}

// build registers every node reachable from the root, root first: one
// StateGraph node per GraphNode, entered at the root (START), each
// ending at END unless it has exactly one child or a handoff routes it.
func (r *graphRun) build(ctx context.Context) (stateGraph, error) {
	def := r.g.def
	b := newStateGraph()
	b.SetSchema(newMessagesSchema())
	names := map[string]string{} // graph node name → config key
	_, err := def.Traverse(ctx, func(_ context.Context, n *aiserver.GraphNode, _ map[string]any) (any, error) {
		outgoing := def.EdgesFrom(n.Key)
		for _, e := range outgoing {
			if def.GetNode(e.TargetKey) == nil {
				return nil, fmt.Errorf("langchainagents: graph %q: edge %s targets unknown node %q", def.Key, e.Key, e.TargetKey)
			}
		}
		name := sanitizeName(n.Key)
		// Graph.AddNode replaces a node of the same name, and END is the
		// stategraph stop sentinel, so either collision would misroute the run.
		if prev, ok := names[name]; ok || name == stategraph.END {
			return nil, fmt.Errorf("%w: graph %q: node %q is named %q, which is taken by %q", ErrNodeName, def.Key, n.Key, name, cmp.Or(prev, stategraph.END))
		}
		names[name] = n.Key
		b.AddNode(name, n.Key, r.nodeFunc(n))
		if len(outgoing) == 1 {
			// One child: an answer without a handoff still moves on, as in
			// the JavaScript and Python adapters.
			b.AddEdge(name, sanitizeName(outgoing[0].TargetKey))
		} else {
			b.AddEdge(name, stategraph.END)
		}
		return nil, nil
	})
	if err != nil {
		return nil, err
	}
	b.SetEntryPoint(sanitizeName(def.Root.Key)) // the entry point is START → root
	return b, nil
}

func (r *graphRun) run(ctx context.Context, input string, history []aiserver.Message) (string, error) {
	b, err := r.build(ctx)
	if err != nil {
		return "", err
	}
	compiled, err := b.Compile()
	if err != nil {
		return "", fmt.Errorf("langchainagents: compile graph %q: %w", r.g.def.Key, err)
	}
	// The graph state starts with the current user turn. History reaches
	// the root only: the root's agent reads it ahead of the state, and
	// it never enters the state other nodes see.
	turns := handlerkit.ComposePrompt(aiserver.HandlerRequest{UserInput: input, History: history}, aiserver.ModeAgent).Turns
	msgs := langchainwire.ToMessages(handlerkit.Prompt{Turns: turns})
	if len(msgs) == 0 { // history of system turns only, and no input
		msgs = []llms.MessageContent{langchainwire.TextParts(llms.ChatMessageTypeHuman, "")}
	}
	r.rootHistory = msgs[:len(msgs)-1]
	final, err := compiled.Invoke(ctx, map[string]any{messagesKey: msgs[len(msgs)-1:]})
	if r.err != nil {
		return "", r.err
	}
	if err != nil {
		return "", err
	}
	return lastAIText(messagesOf(final)), nil
}

// visit records a node's first run in the path and emits $ld:ai:graph:node.
func (r *graphRun) visit(n *aiserver.GraphNode) {
	if slices.Contains(r.path, n.Key) {
		return
	}
	r.path = append(r.path, n.Key)
	r.graphEvent("$ld:ai:graph:node", map[string]any{"nodeKey": n.Key, "index": len(r.path) - 1}, 1)
}

func (r *graphRun) model(ctx context.Context, n *aiserver.GraphNode, cfg aiserver.AiConfig) (llms.Model, error) {
	if f := r.g.opts.ModelFactory; f != nil {
		m, err := f(n)
		if err != nil {
			return nil, fmt.Errorf("langchainagents: model factory for node %q: %w", n.Key, err)
		}
		if m == nil {
			return nil, fmt.Errorf("langchainagents: model factory returned a nil model for node %q", n.Key)
		}
		return m, nil
	}
	return langchainwire.DefaultModel(ctx, cfg, nil, nil)
}

// nodeFunc is the function registered for n: it runs n as a ReAct agent over
// the graph's messages and returns the messages it added, as a Command when
// the model handed off.
func (r *graphRun) nodeFunc(n *aiserver.GraphNode) func(context.Context, any) (any, error) {
	return func(ctx context.Context, state any) (any, error) {
		r.steps++
		if r.steps > recursionLimit {
			return r.fail(fmt.Errorf("%w (%d)", ErrRecursionLimit, recursionLimit))
		}
		r.visit(n)
		td := handlerkit.NodeTrackData(n, r.g.def.Key, r.runID)
		start := time.Now()
		cfg := langchainwire.ModelConfig(n.Config)

		in := messagesOf(state)
		if n == r.g.def.Root {
			in = append(slices.Clip(r.rootHistory), in...)
		}
		// The node's own tool loop runs inside a.invoke, so everything in is
		// from before this entry: send it as text, not provider tool blocks.
		in = neutralHistory(in)
		system := handlerkit.ComposePrompt(aiserver.HandlerRequest{Config: n.Config, Variables: r.variables}, aiserver.ModeAgent).System
		// A node opens no root span: its spans nest under the graph span, and
		// its $ld:ai:graph:node event names it.
		span := handlerkit.StartHandlerRun(ctx, kitbase.HandlerSpanOptions{
			Tracer: PackageName, System: langchainwire.GenAISystem, ProviderName: langchainwire.SpanProviderName(cfg), Operation: operation,
			RequestModel: cfg.Model.Name, NoRoot: true,
		}, aiserver.HandlerRequest{}, promptOf(system, in))
		defer span.EndOnPanic()

		a, err := r.agent(span.Context(), n, cfg, td, system, span)
		var res agentResult
		if err == nil {
			res, err = a.invoke(span.Context(), in)
			r.usage.Add(a.usage)
		}
		if r.tracking {
			aiserver.Track("$ld:ai:duration:total", r.userCtx, td, float64(time.Since(start).Milliseconds()))
		}
		if err != nil {
			span.Fail(err)
			if r.tracking {
				aiserver.Track("$ld:ai:generation:error", r.userCtx, td, 1)
			}
			r.handoff(n, "$ld:ai:graph:handoff_failure")
			return r.fail(err)
		}
		span.Finish(cfg.Model.Name, lastAIText(res.messages))
		if r.tracking {
			trackNodeUsage(r.userCtx, td, a.usage)
		}
		r.handoff(n, "$ld:ai:graph:handoff_success")
		update := map[string]any{messagesKey: res.messages[len(in):]}
		if res.next != "" {
			return &stategraph.Command{Update: update, Goto: res.next}, nil
		}
		return update, nil
	}
}

// handoff tracks the hop from the previous node into n once n has run: a
// handoff succeeds only when its target answers.
func (r *graphRun) handoff(n *aiserver.GraphNode, event string) {
	if r.prev != nil {
		r.graphEvent(event, map[string]any{"sourceKey": r.prev.Key, "targetKey": n.Key}, 1)
	}
	r.prev = n
}

func trackNodeUsage(userCtx ldcontext.Context, td aiserver.TrackData, u kitbase.SpanUsage) {
	aiserver.Track("$ld:ai:generation:success", userCtx, td, 1)
	if total := u.Input + u.Output; total > 0 {
		aiserver.Track("$ld:ai:tokens:total", userCtx, td, total)
	}
	if u.Input > 0 {
		aiserver.Track("$ld:ai:tokens:input", userCtx, td, u.Input)
	}
	if u.Output > 0 {
		aiserver.Track("$ld:ai:tokens:output", userCtx, td, u.Output)
	}
}

// agent builds n's ReAct agent: its model, system prompt (instructions, or
// the config's system messages), regular tools and handoff tools.
func (r *graphRun) agent(ctx context.Context, n *aiserver.GraphNode, cfg aiserver.AiConfig, td aiserver.TrackData, system string, span kitbase.Run) (*reactAgent, error) {
	model, err := r.model(ctx, n, cfg)
	if err != nil {
		return nil, err
	}
	handlers := r.g.opts.ToolHandlers
	if r.tracking {
		handlers = handlerkit.WrapToolHandlers(handlers, r.userCtx, td)
	}
	tools, exec := r.tools(n, handlers, span)
	return &reactAgent{
		model: model, tools: tools, exec: exec, system: system, run: span,
		opts: langchainwire.CallOptions(cfg.Model.Parameters), modelName: cfg.Model.Name,
	}, nil
}

// tools lists n's tool definitions — config tools that have a handler
// (a ToolFunc, or a NativeTool answered with ""), then one
// transfer_to_<target> tool per outgoing edge — and the executor for them.
func (r *graphRun) tools(n *aiserver.GraphNode, handlers map[string]aiserver.ToolHandler, span kitbase.Run) ([]llms.Tool, toolExec) {
	var defs []llms.Tool
	fns := map[string]aiserver.ToolFunc{}
	natives := map[string]*aiserver.NativeTool{}
	keys := make([]string, 0, len(n.Config.Tools))
	for k := range n.Config.Tools {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		switch h := handlers[k].(type) {
		case aiserver.ToolFunc:
			fns[k] = h
		case *aiserver.NativeTool:
			natives[k] = h
		default:
			continue // a tool without a handler is not offered
		}
		t := n.Config.Tools[k]
		defs = append(defs, langchainwire.FunctionTool(k, t.Description, t.Parameters))
	}
	handoffs := map[string]string{} // tool name → target node key
	for _, e := range r.g.def.EdgesFrom(n.Key) {
		name := handoffToolName(e.TargetKey)
		desc := "Transfer control to the " + e.TargetKey + " agent."
		if d, _ := e.Handoff["description"].(string); d != "" {
			desc += " " + d
		}
		handoffs[name] = e.TargetKey
		defs = append(defs, langchainwire.FunctionTool(name, desc,
			map[string]any{"type": "object", "properties": map[string]any{}, "additionalProperties": false}))
	}

	chosen := "" // the first handoff in this node run wins
	exec := func(ctx context.Context, tc llms.ToolCall) (llms.MessageContent, *stategraph.Command, error) {
		name := ""
		if tc.FunctionCall != nil {
			name = tc.FunctionCall.Name
		}
		if target, ok := handoffs[name]; ok {
			markTool(span, name, tc.ID)
			if chosen != "" {
				return langchainwire.ToolResponse(tc, handoffIgnored), nil, nil
			}
			chosen = target // the handoff is tracked once the target has run
			return langchainwire.ToolResponse(tc, "Handed off to "+target+"."), handoffCommand(target), nil
		}
		if nt := natives[name]; nt != nil {
			markTool(span, name, tc.ID)
			handlerkit.TrackNativeTool(nt) // a provider built-in has no local implementation
			return langchainwire.ToolResponse(tc, ""), nil, nil
		}
		m, err := langchainwire.RunTool(ctx, span, fns, tc)
		return m, nil, err
	}
	return defs, exec
}

// markTool records a tool call that runs no local code as an execute_tool span.
func markTool(span kitbase.Run, name, callID string) {
	if span != nil {
		span.EndTool(span.StartTool(name, callID), nil)
	}
}

// promptOf is a node run's prompt for its root span: the system prompt and
// the text of each message the node receives.
func promptOf(system string, msgs []llms.MessageContent) handlerkit.Prompt {
	p := handlerkit.Prompt{System: system}
	for _, m := range msgs {
		role := map[llms.ChatMessageType]string{llms.ChatMessageTypeHuman: "user", llms.ChatMessageTypeAI: "assistant",
			llms.ChatMessageTypeSystem: "system", llms.ChatMessageTypeTool: "tool"}[m.Role]
		var b strings.Builder
		for _, part := range m.Parts {
			switch v := part.(type) {
			case llms.TextContent:
				b.WriteString(v.Text)
			case llms.ToolCallResponse:
				b.WriteString(v.Content)
			}
		}
		p.Turns = append(p.Turns, aiserver.Message{Role: cmp.Or(role, string(m.Role)), Text: b.String()})
	}
	return p
}

// neutralHistory rewrites the messages a node finds when it is entered as
// provider-neutral text, so a target on another provider never receives the
// previous node's tool blocks (Bedrock Converse rejects toolUse/toolResult
// blocks from a request that declares no tools). Human and system messages
// pass through. An AI message keeps its text and gains one
// "[Tool call] <name> <args>" line per call; the tool results after it become
// one human message of "[Tool result] <name>: <text>" lines, in call order.
func neutralHistory(msgs []llms.MessageContent) []llms.MessageContent {
	out := make([]llms.MessageContent, 0, len(msgs))
	var calls []llms.ToolCall
	var results []llms.ToolCallResponse
	flush := func() {
		if len(results) == 0 {
			return
		}
		order := func(r llms.ToolCallResponse) int {
			if i := slices.IndexFunc(calls, func(c llms.ToolCall) bool { return c.ID == r.ToolCallID }); i >= 0 {
				return i
			}
			return len(calls) // a result with no matching call goes last
		}
		slices.SortStableFunc(results, func(a, b llms.ToolCallResponse) int { return order(a) - order(b) })
		lines := make([]string, len(results))
		for i, r := range results {
			name := r.Name
			if j := order(r); j < len(calls) && calls[j].FunctionCall != nil {
				name = calls[j].FunctionCall.Name
			}
			lines[i] = "[Tool result] " + name + ": " + r.Content
		}
		out = append(out, langchainwire.TextParts(llms.ChatMessageTypeHuman, strings.Join(lines, "\n")))
		results = nil
	}
	for _, m := range msgs {
		if m.Role == llms.ChatMessageTypeTool {
			for _, p := range m.Parts {
				if r, ok := p.(llms.ToolCallResponse); ok {
					results = append(results, r)
				}
			}
			continue
		}
		flush()
		if m.Role != llms.ChatMessageTypeAI {
			out = append(out, m)
			continue
		}
		calls = nil
		var text strings.Builder
		for _, p := range m.Parts {
			switch v := p.(type) {
			case llms.TextContent:
				text.WriteString(v.Text)
			case llms.ToolCall:
				calls = append(calls, v)
			}
		}
		var lines []string
		if text.Len() > 0 {
			lines = append(lines, text.String())
		}
		for _, c := range calls {
			if c.FunctionCall != nil {
				lines = append(lines, "[Tool call] "+c.FunctionCall.Name+" "+compactArgs(c.FunctionCall.Arguments))
			}
		}
		if len(lines) > 0 {
			out = append(out, langchainwire.TextParts(llms.ChatMessageTypeAI, strings.Join(lines, "\n")))
		}
	}
	flush()
	return out
}

// compactArgs is a tool call's arguments as compact JSON ({} when empty),
// written by the standard encoder without HTML escaping, so "&" stays "&".
// Arguments that are not JSON pass through.
func compactArgs(args string) string {
	if strings.TrimSpace(args) == "" {
		return "{}"
	}
	d := json.NewDecoder(strings.NewReader(args))
	d.UseNumber() // keep numbers as written
	var v any
	if d.Decode(&v) != nil {
		return args
	}
	if _, err := d.Token(); err != io.EOF { // trailing data, even a stray } or ]
		return args
	}
	if v == nil {
		return "{}"
	}
	var b bytes.Buffer
	e := json.NewEncoder(&b)
	e.SetEscapeHTML(false)
	if e.Encode(v) != nil {
		return args
	}
	return strings.TrimSuffix(b.String(), "\n")
}
