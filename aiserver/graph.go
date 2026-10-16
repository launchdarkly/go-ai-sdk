package aiserver

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/launchdarkly/go-sdk-common/v3/ldvalue"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"

	"github.com/launchdarkly/go-ai-sdk/internal/trajectory"
)

const (
	// maxTraversalDepth bounds graph walks and BFS expansion (cycle guard).
	maxTraversalDepth = 100
	// MaxGraphCacheSize bounds the per-instance resolved-graph cache.
	MaxGraphCacheSize = 512
	// GraphSpanName is the graph-level span.
	GraphSpanName = "ld.ai.graph"
)

// GraphTopology is a graph flag variation: a root config key and each
// source key's outgoing edges.
type GraphTopology struct {
	Root  string                         `json:"root"`
	Edges map[string][]GraphTopologyEdge `json:"edges,omitempty"`
}

// GraphTopologyEdge is one outgoing edge in a GraphTopology.
type GraphTopologyEdge struct {
	Key     string         `json:"key"`
	Handoff map[string]any `json:"handoff,omitempty"`
}

// GraphEdge is a directed edge between two agent configs.
type GraphEdge struct {
	Key       string // "<sourceKey>-<targetKey>"
	SourceKey string
	TargetKey string
	Handoff   map[string]any
}

// GraphNode is an evaluated agent config plus its outgoing edges.
type GraphNode struct {
	Key    string
	Config AiConfig
	Meta   VariationMeta
	Edges  []GraphEdge
}

// IsTerminal reports whether the node has no outgoing edges.
func (n *GraphNode) IsTerminal() bool { return len(n.Edges) == 0 }

// TrackData is the node's tracking payload for one graph run: its config
// identity, graphKey, and runID. For framework-native graph runners, so they
// track nodes exactly as Graph does.
func (n *GraphNode) TrackData(graphKey, runID string) TrackData {
	td := newTrackData(n.Key, n.Config, n.Meta, graphKey)
	td.RunID = runID
	return td
}

// GraphOptions configure Graph and ResolveGraph.
type GraphOptions struct {
	// Handlers route each node by its provider + mode. Required for
	// Graph().Invoke/Stream and RunNode/Route; optional when the definition
	// is handed to a framework-native runner.
	Handlers     []*ProviderHandler
	ToolHandlers map[string]ToolHandler
	// GraphJudge is an optional judge config key evaluated against the final output.
	GraphJudge string
	// Registry supplies handlers and tools; local values take precedence.
	Registry *Registry
}

// RunNodeOptions are per-node options for RunNode and Route.
type RunNodeOptions struct {
	Variables    map[string]any
	ToolHandlers map[string]ToolHandler
	History      []Message
	// From is the node being handed off from; set it to emit handoff events.
	From *GraphNode
}

// RouteResult is a node response plus the edge the model chose (Next is nil
// when the node produced a terminal answer).
type RouteResult struct {
	ProviderResponse
	Next *GraphNode
	// tracked is true when the handoff to Next was already tracked (a
	// model-chosen edge), so the next node must not track it again.
	tracked bool
}

// GraphResponse is the aggregate result of a graph run.
type GraphResponse struct {
	Response     string                 `json:"response"`
	Usage        TokenUsage             `json:"usage"`
	JudgeResults map[string]JudgeResult `json:"judgeResults,omitempty"`
}

// GraphStreamEvent is one item from Graph(...).Stream. Type is
// node_start, chunk, node_done, handoff or done.
type GraphStreamEvent struct {
	Type         EventType
	NodeKey      string
	Text         string
	Response     string
	Usage        TokenUsage
	SourceKey    string
	TargetKey    string
	JudgeResults map[string]JudgeResult
}

// GraphDefinition is a resolved agent graph: topology accessors, tracked
// per-node execution, and ordered walks for framework adapters.
type GraphDefinition struct {
	Key     string
	Enabled bool
	Root    *GraphNode

	nodes     map[string]*GraphNode
	edges     []GraphEdge
	userCtx   ldcontext.Context
	opts      GraphOptions
	trackData TrackData
}

// GetNode returns the node for key, or nil.
func (d *GraphDefinition) GetNode(key string) *GraphNode { return d.nodes[key] }

// EdgesFrom returns key's outgoing edges.
func (d *GraphDefinition) EdgesFrom(key string) []GraphEdge {
	var out []GraphEdge
	for _, e := range d.edges {
		if e.SourceKey == key {
			out = append(out, e)
		}
	}
	return out
}

// GetChildNodes returns the targets of key's outgoing edges.
func (d *GraphDefinition) GetChildNodes(key string) []*GraphNode {
	var out []*GraphNode
	for _, e := range d.EdgesFrom(key) {
		if n := d.nodes[e.TargetKey]; n != nil {
			out = append(out, n)
		}
	}
	return out
}

// GetParentNodes returns the sources of edges into key.
func (d *GraphDefinition) GetParentNodes(key string) []*GraphNode {
	var out []*GraphNode
	for _, e := range d.edges {
		if e.TargetKey == key {
			if n := d.nodes[e.SourceKey]; n != nil {
				out = append(out, n)
			}
		}
	}
	return out
}

// TerminalNodes returns every node with no outgoing edges, in key order.
func (d *GraphDefinition) TerminalNodes() []*GraphNode {
	var out []*GraphNode
	for _, k := range d.sortedKeys() {
		if d.nodes[k].IsTerminal() {
			out = append(out, d.nodes[k])
		}
	}
	return out
}

func (d *GraphDefinition) sortedKeys() []string {
	keys := make([]string, 0, len(d.nodes))
	for k := range d.nodes {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// TraverseVisitor is called once per node; acc holds earlier results by key.
type TraverseVisitor func(ctx context.Context, node *GraphNode, acc map[string]any) (any, error)

// Traverse visits nodes root-first by depth (deepest last) and returns the
// root's result.
func (d *GraphDefinition) Traverse(ctx context.Context, fn TraverseVisitor) (any, error) {
	if d.Root == nil {
		return nil, nil
	}
	depths := map[string]int{d.Root.Key: 0}
	seen := map[string]bool{d.Root.Key: true}
	frontier := []string{d.Root.Key}
	for i := 0; len(frontier) > 0 && i < maxTraversalDepth; i++ {
		var next []string
		for _, k := range frontier {
			for _, c := range d.GetChildNodes(k) {
				if cd := depths[k] + 1; cd > depths[c.Key] || !seen[c.Key] {
					depths[c.Key] = cd
				}
				if !seen[c.Key] {
					seen[c.Key] = true
					next = append(next, c.Key)
				}
			}
		}
		frontier = next
	}
	order := make([]string, 0, len(depths))
	for k := range depths {
		order = append(order, k)
	}
	sort.SliceStable(order, func(i, j int) bool {
		if depths[order[i]] != depths[order[j]] {
			return depths[order[i]] < depths[order[j]]
		}
		return order[i] < order[j]
	})
	acc := map[string]any{}
	for _, k := range order {
		v, err := fn(ctx, d.nodes[k], acc)
		if err != nil {
			return nil, err
		}
		acc[k] = v
	}
	return acc[d.Root.Key], nil
}

// ReverseTraverse visits nodes leaf-first, the root always last, and
// returns the root's result.
func (d *GraphDefinition) ReverseTraverse(ctx context.Context, fn TraverseVisitor) (any, error) {
	if d.Root == nil {
		return nil, nil
	}
	var frontier []string
	for _, n := range d.TerminalNodes() {
		frontier = append(frontier, n.Key)
	}
	if len(frontier) == 0 {
		return nil, nil
	}
	acc := map[string]any{}
	visited := map[string]bool{}
	for i := 0; len(frontier) > 0 && i < maxTraversalDepth; i++ {
		var next []string
		for _, k := range frontier {
			if visited[k] {
				continue
			}
			visited[k] = true
			if k == d.Root.Key {
				continue
			}
			v, err := fn(ctx, d.nodes[k], acc)
			if err != nil {
				return nil, err
			}
			acc[k] = v
			for _, p := range d.GetParentNodes(k) {
				if !visited[p.Key] {
					next = append(next, p.Key)
				}
			}
		}
		frontier = next
	}
	v, err := fn(ctx, d.Root, acc)
	if err != nil {
		return nil, err
	}
	acc[d.Root.Key] = v
	return v, nil
}

// TrackData returns the graph-level track data (graphKey set) for adapters.
func (d *GraphDefinition) TrackData() TrackData { return d.trackData }

// UserContext is the context the definition was resolved for.
func (d *GraphDefinition) UserContext() ldcontext.Context { return d.userCtx }

func (d *GraphDefinition) handlers() []*ProviderHandler {
	return resolveHandlers(d.opts.Registry, d.opts.Handlers)
}

func (d *GraphDefinition) tools(override map[string]ToolHandler) map[string]ToolHandler {
	if override != nil {
		return override
	}
	return resolveTools(d.opts.Registry, d.opts.ToolHandlers)
}

// selectNodeHandler matches provider+mode, then wildcard+mode, then provider
// alone, then a sole handler.
func selectNodeHandler(n *GraphNode, handlers []*ProviderHandler) (*ProviderHandler, error) {
	provider := n.Config.Provider.Name
	if provider == "" {
		return nil, errorf("provider not found for graph node %s", n.Key)
	}
	mode := NormalizeMode(n.Meta.Mode)
	for _, want := range []ProvidesFor{{provider, mode}, {WildcardProvider, mode}} {
		for _, h := range handlers {
			if h.ProvidesFor.Provider == want.Provider && handlerMode(h) == want.Mode {
				return h, nil
			}
		}
	}
	for _, h := range handlers {
		if h.ProvidesFor.Provider == provider {
			return h, nil
		}
	}
	if len(handlers) == 1 {
		return handlers[0], nil
	}
	return nil, fmt.Errorf("%w %s", ErrNoHandler, provider)
}

var unsafeToolChars = regexp.MustCompile(`[^a-zA-Z0-9_]`)

const routingSuffix = "\n\nComplete your task using your available tools first. Only once you have your final answer, call exactly one transfer tool to route to the next agent."

// handoffRouting is the synthetic routing surface for a multi-edge node,
// shared by Invoke and Stream so they route identically.
type handoffRouting struct {
	config AiConfig
	tools  map[string]ToolHandler
	mu     sync.Mutex
	chosen string
}

func (d *GraphDefinition) buildRouting(n *GraphNode, outgoing []GraphEdge) *handoffRouting {
	r := &handoffRouting{tools: map[string]ToolHandler{}}
	tools := map[string]Tool{}
	for k, t := range n.Config.Tools {
		tools[k] = t
	}
	for _, e := range outgoing {
		name := trajectory.HandoffPrefix + unsafeToolChars.ReplaceAllString(e.TargetKey, "_")
		detail, _ := e.Handoff["description"].(string)
		if detail == "" {
			if t := d.nodes[e.TargetKey]; t != nil {
				detail = truncateRunes(t.Config.Instructions, 120)
			}
		}
		desc := "Transfer control to " + e.TargetKey + "."
		if detail != "" {
			desc += " " + detail
		}
		tools[name] = Tool{Name: name, Type: "function", Description: desc,
			Parameters: map[string]any{"type": "object", "properties": map[string]any{}, "additionalProperties": false}}
		target := e.TargetKey
		r.tools[name] = ToolFunc(func(context.Context, map[string]any) (any, error) {
			r.mu.Lock()
			if r.chosen == "" {
				r.chosen = target
			}
			r.mu.Unlock()
			return "Handoff to " + target + " recorded. Finish your own work and provide your final response.", nil
		})
	}
	r.config = n.Config
	r.config.Instructions = n.Config.Instructions + routingSuffix
	r.config.Tools = tools
	return r
}

func (r *handoffRouting) choice() string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.chosen
}

func truncateRunes(s string, n int) string {
	if r := []rune(s); len(r) > n {
		return string(r[:n])
	}
	return s
}

// trackGraph sends a graph event: td plus extra fields (nodeKey, index,
// sourceKey, targetKey).
func trackGraph(event string, userCtx ldcontext.Context, td TrackData, extra map[string]any, metric float64) {
	c, err := GetClient()
	if err != nil {
		return
	}
	b, _ := json.Marshal(td)
	var m map[string]any
	_ = json.Unmarshal(b, &m)
	for k, v := range extra {
		m[k] = v
	}
	b, _ = json.Marshal(m)
	safeTrack(c, event, userCtx, metric, ldvalue.Parse(b))
}

// TrackGraphEvent sends a `$ld:ai:graph:*` event with td merged with extra.
// For framework-native graph adapters.
func TrackGraphEvent(event string, userCtx ldcontext.Context, td TrackData, extra map[string]any, metric float64) {
	trackGraph(event, userCtx, td, extra, metric)
}

// streamRoute runs one node, emitting node_start / chunk / node_done. A
// multi-edge node gets handoff tools and the model's choice becomes Next.
func (d *GraphDefinition) streamRoute(ctx context.Context, n *GraphNode, input string, o RunNodeOptions, runTD TrackData, route bool, emit func(GraphStreamEvent) bool) (res RouteResult, abandoned bool, err error) {
	handlers := d.handlers()
	if len(handlers) == 0 {
		return RouteResult{}, false, errorf("graph node execution requires handlers; use a framework-native runner instead")
	}
	h, err := selectNodeHandler(n, handlers)
	if err != nil {
		return RouteResult{}, false, err
	}
	tools := d.tools(o.ToolHandlers)
	outgoing := d.EdgesFrom(n.Key)
	cfg := n.Config
	var routing *handoffRouting
	if route && len(outgoing) > 1 {
		routing = d.buildRouting(n, outgoing)
		cfg = routing.config
		merged := map[string]ToolHandler{}
		for k, v := range tools {
			merged[k] = v
		}
		for k, v := range routing.tools {
			merged[k] = v
		}
		tools = merged
	}
	handoff := func(event, target string) {
		trackGraph(event, d.userCtx, runTD, map[string]any{"sourceKey": n.Key, "targetKey": target}, 1)
	}
	failed := func(err error) (RouteResult, bool, error) {
		switch {
		case routing != nil && routing.choice() != "":
			handoff("$ld:ai:graph:handoff_failure", routing.choice())
		case o.From != nil:
			trackGraph("$ld:ai:graph:handoff_failure", d.userCtx, runTD, map[string]any{"sourceKey": o.From.Key, "targetKey": n.Key}, 1)
		}
		return RouteResult{}, false, err
	}

	if !emit(GraphStreamEvent{Type: EventNodeStart, NodeKey: n.Key}) {
		return RouteResult{}, true, nil
	}
	var done streamDone
	for ev, serr := range executeAndStream(ctx, execArgs{
		configKey: n.Key, config: cfg, meta: n.Meta, userCtx: d.userCtx, handler: h,
		userInput: input, tools: tools, variables: o.Variables, graphKey: d.Key, history: o.History,
	}, &done) {
		if serr != nil {
			return failed(serr)
		}
		if !emit(GraphStreamEvent{Type: EventChunk, Text: ev.Text, NodeKey: n.Key}) {
			return RouteResult{}, true, nil
		}
	}
	// Judges grade the node's original config, not the routed one.
	judgeResults, jerr := runJudges(ctx, judgeArgs{cfg: n.Config, userCtx: d.userCtx, handler: h, handlers: handlers,
		userInput: input, output: done.response, parentTD: done.td, graphKey: d.Key, trajectory: done.trajectory})
	if jerr != nil {
		return failed(jerr)
	}
	res = RouteResult{ProviderResponse: ProviderResponse{Response: done.response, Usage: done.usage, TrackData: done.td, JudgeResults: judgeResults}}
	switch {
	case routing != nil:
		if c := routing.choice(); c != "" {
			res.Next = d.nodes[c]
			res.tracked = true
			handoff("$ld:ai:graph:handoff_success", c)
		}
	case o.From != nil:
		trackGraph("$ld:ai:graph:handoff_success", d.userCtx, runTD, map[string]any{"sourceKey": o.From.Key, "targetKey": n.Key}, 1)
	}
	if route && routing == nil && len(outgoing) == 1 {
		res.Next = d.nodes[outgoing[0].TargetKey]
	}
	if !emit(GraphStreamEvent{Type: EventNodeDone, NodeKey: n.Key, Response: done.response, Usage: done.usage}) {
		return res, true, nil
	}
	return res, false, nil
}

// RunNode executes one node through the tracked path.
func (d *GraphDefinition) RunNode(ctx context.Context, n *GraphNode, input string, o RunNodeOptions) (ProviderResponse, error) {
	if !d.Enabled {
		return ProviderResponse{}, fmt.Errorf("%w: %s", ErrGraphDisabled, d.Key)
	}
	res, _, err := d.streamRoute(ctx, n, input, o, d.trackData, false, func(GraphStreamEvent) bool { return true })
	return res.ProviderResponse, err
}

// Route executes a node, offering its edges to the model as handoff tools,
// and returns the response plus the chosen next node.
func (d *GraphDefinition) Route(ctx context.Context, n *GraphNode, input string, o RunNodeOptions) (RouteResult, error) {
	if !d.Enabled {
		return RouteResult{}, fmt.Errorf("%w: %s", ErrGraphDisabled, d.Key)
	}
	res, _, err := d.streamRoute(ctx, n, input, o, d.trackData, true, func(GraphStreamEvent) bool { return true })
	return res, err
}

// resolveDefinition fetches the topology and every node config. A disabled
// node, a missing root, or an evaluation failure is an error.
func resolveDefinition(ctx context.Context, key string, userCtx ldcontext.Context, opts GraphOptions) (*GraphDefinition, error) {
	def := &GraphDefinition{Key: key, userCtx: userCtx, opts: opts, nodes: map[string]*GraphNode{}}
	c, err := InitClient(ctx, InitOptions{})
	if err != nil {
		return nil, err
	}
	v, evalErr := c.JSONVariation(key, userCtx, ldvalue.ObjectBuild().Build())
	if evalErr != nil {
		return nil, fmt.Errorf("graph %q: %w", key, evalErr)
	}
	raw, _ := variationToAny(v).(map[string]any)
	meta := &VariationMeta{}
	if m, ok := raw["_ldMeta"].(map[string]any); ok {
		meta = parseMeta(m)
	}
	def.trackData = newTrackData(key, AiConfig{}, *meta, key)
	var topo GraphTopology
	b, err := json.Marshal(raw)
	if err != nil {
		return nil, errorf("graph %q: malformed topology: %w", key, err)
	}
	if err := json.Unmarshal(b, &topo); err != nil {
		return nil, errorf("graph %q: malformed topology: %w", key, err)
	}
	if topo.Root == "" {
		return nil, errorf("graph %q has no root", key)
	}
	keys := []string{topo.Root}
	seen := map[string]bool{topo.Root: true}
	sources := make([]string, 0, len(topo.Edges))
	for s := range topo.Edges {
		sources = append(sources, s)
	}
	sort.Strings(sources)
	for _, s := range sources {
		for _, e := range topo.Edges[s] {
			def.edges = append(def.edges, GraphEdge{Key: s + "-" + e.Key, SourceKey: s, TargetKey: e.Key, Handoff: e.Handoff})
			for _, k := range []string{s, e.Key} {
				if !seen[k] {
					seen[k] = true
					keys = append(keys, k)
				}
			}
		}
	}
	for _, k := range keys {
		cfg, nmeta, err := extractVariation(ctx, k, userCtx)
		if err != nil {
			return nil, fmt.Errorf("graph %q node %q: %w", key, k, err)
		}
		def.nodes[k] = &GraphNode{Key: k, Config: cfg, Meta: nmeta}
	}
	for _, n := range def.nodes {
		n.Edges = def.EdgesFrom(n.Key)
	}
	def.Enabled = true
	def.Root = def.nodes[topo.Root]
	return def, nil
}

// ResolveGraph resolves a graph flag into a definition without executing it,
// for framework-native runners. A non-nil error means the graph did not resolve.
func ResolveGraph(ctx context.Context, key string, userCtx ldcontext.Context, opts GraphOptions) (*GraphDefinition, error) {
	reportUsage("client.resolveGraph", "", "")
	return resolveDefinition(ctx, key, userCtx, opts)
}

// AgentGraph is the value Graph returns: a model-driven router bound to a
// graph flag key.
type AgentGraph struct {
	key      string
	opts     GraphOptions
	internal bool

	mu    sync.Mutex
	cache map[string]*GraphDefinition
	order []string
}

// Graph returns a model-driven graph runner. It starts at the root, lets the
// model choose an outgoing edge at each step, and threads each node's output
// into the next. It stops on a terminal answer, a leaf, a revisited node, or
// the step cap.
func Graph(key string, opts GraphOptions) *AgentGraph {
	return &AgentGraph{key: key, opts: opts, cache: map[string]*GraphDefinition{}}
}

// GraphInternal is the non-reporting Graph for handler-package wrappers.
func GraphInternal(key string, opts GraphOptions) *AgentGraph {
	g := Graph(key, opts)
	g.internal = true
	return g
}

// definition resolves once per distinct context content (equal contexts
// share an entry), bounded by MaxGraphCacheSize. Only enabled definitions
// are cached.
func (g *AgentGraph) definition(ctx context.Context, userCtx ldcontext.Context) (*GraphDefinition, error) {
	b, _ := json.Marshal(userCtx)
	k := string(b)
	g.mu.Lock()
	if d, ok := g.cache[k]; ok {
		g.mu.Unlock()
		return d, nil
	}
	g.mu.Unlock()
	d, err := resolveDefinition(ctx, g.key, userCtx, g.opts)
	if err != nil {
		return nil, err // a failed resolution may be transient; resolve again next call
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if existing, ok := g.cache[k]; ok {
		return existing, nil
	}
	if len(g.order) >= MaxGraphCacheSize {
		delete(g.cache, g.order[0])
		g.order = g.order[1:]
	}
	g.cache[k] = d
	g.order = append(g.order, k)
	return d, nil
}

func (g *AgentGraph) cacheLen() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return len(g.cache)
}

// Invoke runs the graph and returns the final node's response and the
// aggregate usage. variables and history may be nil; history reaches the
// root node only.
func (g *AgentGraph) Invoke(ctx context.Context, input string, userCtx ldcontext.Context, opts ...CallOption) (GraphResponse, error) {
	co := NewCallOptions(opts...)
	variables, history := co.Variables, co.History
	if !g.internal {
		reportUsage("client.graph.invoke", "", "")
	}
	var out GraphResponse
	var got bool
	for ev, err := range g.stream(ctx, input, userCtx, variables, history) {
		if err != nil {
			return GraphResponse{}, err
		}
		if ev.Type == EventDone {
			out, got = GraphResponse{Response: ev.Response, Usage: ev.Usage, JudgeResults: ev.JudgeResults}, true
		}
	}
	if !got {
		return GraphResponse{}, errorf("agent graph %q ended without a result", g.key)
	}
	return out, nil
}

// Stream runs the graph, yielding node_start, chunk, node_done, handoff and
// a final done event. The conversation id and parent span are taken from
// ctx at call time.
func (g *AgentGraph) Stream(ctx context.Context, input string, userCtx ldcontext.Context, opts ...CallOption) iter.Seq2[GraphStreamEvent, error] {
	co := NewCallOptions(opts...)
	variables, history := co.Variables, co.History
	if !g.internal {
		reportUsage("client.graph.stream", "", "")
	}
	return g.stream(ctx, input, userCtx, variables, history)
}

func (g *AgentGraph) stream(ctx context.Context, input string, userCtx ldcontext.Context, variables map[string]any, history []Message) iter.Seq2[GraphStreamEvent, error] {
	return func(yield func(GraphStreamEvent, error) bool) {
		if len(resolveHandlers(g.opts.Registry, g.opts.Handlers)) == 0 {
			yield(GraphStreamEvent{}, errorf("graph %q requires handlers; pass Handlers or use ResolveGraph with a framework-native runner", g.key))
			return
		}
		def, err := g.definition(ctx, userCtx)
		if err != nil {
			yield(GraphStreamEvent{}, err)
			return
		}
		if !def.Enabled {
			yield(GraphStreamEvent{}, fmt.Errorf("%w: %s", ErrGraphDisabled, g.key))
			return
		}
		runTD := def.trackData
		runTD.RunID = uuid.NewString() // the definition is cached; each run gets its own id

		spanCtx, span := otel.Tracer(TracerPrefix+"/aiserver").Start(ctx, GraphSpanName)
		span.SetAttributes(attribute.String("ld.ai.graph.key", g.key))
		// Context identity goes on the root handler span only, not here.
		SetLDSpanAttributesFor(span, runTD, ldcontext.Context{})

		start := time.Now()
		usage := TokenUsage{}
		var path []string // node keys in visit order, as the native runners report it
		var last RouteResult
		finished := false
		// A panic from the consumer's loop body passes through emit and is
		// abandonment; any other panic is a failed run. Both re-panic.
		inYield := false
		emit := func(ev GraphStreamEvent) bool {
			inYield = true
			ok := yield(ev, nil)
			inYield = false
			return ok
		}
		failSpan := func(err error) {
			finished = true
			trackGraph("$ld:ai:graph:duration:total", userCtx, runTD, nil, float64(time.Since(start).Milliseconds()))
			trackGraph("$ld:ai:graph:invocation_failure", userCtx, runTD, nil, 1)
			span.RecordError(err)
			span.SetStatus(codes.Error, err.Error())
			span.End()
		}
		defer func() {
			if finished {
				return // recover is not called, so a panic continues
			}
			r := recover()
			if r != nil && !inYield {
				failSpan(fmt.Errorf("graph node panicked: %v", r))
				panic(r)
			}
			// consumer stopped: neither success nor failure
			span.SetAttributes(attribute.Bool("launchdarkly.stream.abandoned", true))
			span.End()
			if r != nil {
				panic(r)
			}
		}()
		fail := func(err error) {
			failSpan(err)
			yield(GraphStreamEvent{}, err)
		}

		current, input0 := def.Root, input
		var previous *GraphNode
		visited := map[string]bool{}
		for steps := 0; current != nil && steps < maxTraversalDepth; steps++ {
			trackGraph("$ld:ai:graph:node", userCtx, runTD, map[string]any{"nodeKey": current.Key, "index": steps}, 1)
			path = append(path, current.Key)
			o := RunNodeOptions{Variables: variables, From: previous}
			if steps == 0 {
				o.History = history // root only
			}
			res, abandoned, err := def.streamRoute(spanCtx, current, input, o, runTD, true, emit)
			if abandoned {
				return
			}
			if err != nil {
				fail(err)
				return
			}
			usage.Input += res.Usage.Input
			usage.Output += res.Usage.Output
			usage.Total += res.Usage.Total
			last = res
			if res.Next == nil || visited[res.Next.Key] {
				break
			}
			if !emit(GraphStreamEvent{Type: EventHandoff, SourceKey: current.Key, TargetKey: res.Next.Key}) {
				return
			}
			visited[current.Key] = true
			previous, current = current, res.Next
			if res.tracked {
				previous = nil // the model-chosen handoff was already tracked
			}
			input = "[Original request]\n" + input0 + "\n\n[Previous agent response]\n" + outputString(res.Response)
		}

		done := GraphStreamEvent{Type: EventDone, Response: outputString(last.Response), Usage: usage}
		if g.opts.GraphJudge != "" && def.Root != nil {
			handlers := def.handlers()
			h, err := selectNodeHandler(def.Root, handlers)
			if err != nil {
				fail(err)
				return
			}
			// A graph-level judge grades an answer built across nodes, so it
			// gets no trajectory.
			jr, err := runJudges(spanCtx, judgeArgs{
				cfg:     AiConfig{JudgeConfiguration: &JudgeConfiguration{Judges: []JudgeRef{{Key: g.opts.GraphJudge, SamplingRate: 1}}}},
				userCtx: userCtx, handler: h, handlers: handlers, userInput: input0, output: outputString(last.Response), parentTD: runTD, graphKey: g.key,
			})
			if err != nil {
				fail(err)
				return
			}
			if len(jr) > 0 {
				done.JudgeResults = jr
			}
		}
		// Success is tracked only once the graph judge has also finished, so a
		// judge failure never reports both success and failure.
		trackGraph("$ld:ai:graph:duration:total", userCtx, runTD, nil, float64(time.Since(start).Milliseconds()))
		if usage.Total > 0 {
			trackGraph("$ld:ai:graph:total_tokens", userCtx, runTD, nil, usage.Total)
		}
		trackGraph("$ld:ai:graph:invocation_success", userCtx, runTD, nil, 1)
		finished = true
		span.SetAttributes(attribute.String("launchdarkly.graph.path", strings.Join(path, "->")))
		SetUsageSpanAttributes(span, SpanUsage{Input: usage.Input, Output: usage.Output})
		span.SetStatus(codes.Ok, "")
		span.End()
		yield(done, nil)
	}
}
