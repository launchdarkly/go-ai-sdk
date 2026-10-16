package langchainagents

import (
	"context"
	"errors"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/tmc/langchaingo/llms"
	"go.opentelemetry.io/otel/codes"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/stategraph"
)

// spyGraph records the builder calls made on a real StateGraph and
// delegates every one, so a spied graph still compiles and runs.
type spyGraph struct {
	stateGraph
	nodes  []string
	fns    map[string]func(context.Context, any) (any, error)
	edges  [][2]string
	conds  []string
	entry  string
	schema bool
}

func (s *spyGraph) AddNode(name, desc string, fn func(context.Context, any) (any, error)) {
	s.nodes = append(s.nodes, name)
	s.fns[name] = fn
	s.stateGraph.AddNode(name, desc, fn)
}
func (s *spyGraph) AddEdge(from, to string) {
	s.edges = append(s.edges, [2]string{from, to})
	s.stateGraph.AddEdge(from, to)
}
func (s *spyGraph) AddConditionalEdge(from string, c func(context.Context, any) string) {
	s.conds = append(s.conds, from)
	s.stateGraph.AddConditionalEdge(from, c)
}
func (s *spyGraph) SetEntryPoint(n string) { s.entry = n; s.stateGraph.SetEntryPoint(n) }
func (s *spyGraph) SetSchema(sc stategraph.Schema) {
	_, s.schema = sc.(messagesSchema)
	s.stateGraph.SetSchema(sc)
}

// spyGraphs replaces newStateGraph; it returns every graph built, in order.
func spyGraphs(t *testing.T) func() []*spyGraph {
	t.Helper()
	var mu sync.Mutex
	var all []*spyGraph
	orig := newStateGraph
	newStateGraph = func() stateGraph {
		s := &spyGraph{stateGraph: orig(), fns: map[string]func(context.Context, any) (any, error){}}
		mu.Lock()
		all = append(all, s)
		mu.Unlock()
		return s
	}
	t.Cleanup(func() { newStateGraph = orig })
	return func() []*spyGraph { mu.Lock(); defer mu.Unlock(); return slices.Clone(all) }
}

func node(instr, model string, tools ...string) map[string]any {
	ts := map[string]any{}
	for _, t := range tools {
		ts[t] = map[string]any{"name": t, "type": "function", "description": t + " tool", "parameters": map[string]any{"type": "object"}}
	}
	return map[string]any{
		"model": map[string]any{"name": model}, "provider": map[string]any{"name": "OpenAI"},
		"instructions": instr, "tools": ts,
		"_ldMeta": map[string]any{"enabled": true, "variationKey": "v-" + model, "mode": "agent"},
	}
}

func graphFlag(root string, edges map[string][]any) map[string]any {
	return map[string]any{"root": root, "edges": edges, "_ldMeta": map[string]any{"enabled": true, "variationKey": "gv"}}
}

func edge(target string) any { return map[string]any{"key": target} }

// twoNode is triage → billing; triage has `lookup`, billing has `refund`.
func twoNode() *fakeLD {
	return &fakeLD{variations: map[string]map[string]any{
		"g":       graphFlag("triage", map[string][]any{"triage": {map[string]any{"key": "billing", "handoff": map[string]any{"description": "billing questions"}}}}),
		"triage":  node("Route {{tone}}", "triage-model", "lookup"),
		"billing": node("Bill", "billing-model", "refund"),
	}}
}

var user = ldcontext.New("u-1")

func resolve(key string) *aiserver.GraphDefinition {
	return aiserver.ResolveGraph(bg, key, user, aiserver.GraphOptions{})
}

// models returns a ModelFactory serving one fake model per node key.
func models(m map[string]*fakeModel) func(*aiserver.GraphNode) (llms.Model, error) {
	return func(n *aiserver.GraphNode) (llms.Model, error) { return m[n.Key], nil }
}

func handlersFor(calls *[]string) map[string]aiserver.ToolHandler {
	mk := func(name string) aiserver.ToolFunc {
		return fn(func(map[string]any) (any, error) { *calls = append(*calls, name); return name + " done", nil })
	}
	return map[string]aiserver.ToolHandler{"lookup": mk("lookup"), "refund": mk("refund")}
}

func newRun(def *aiserver.GraphDefinition, opts NativeGraphOptions) *graphRun {
	return &graphRun{g: ToLangGraph(def, &opts), runID: "run", userCtx: opts.Context, tracking: opts.Context.IsDefined()}
}

// topology: one StateGraph node per GraphNode, START → root,
// terminal → END, handoff tools only on non-terminal nodes.
func TestToLangGraphTopology(t *testing.T) {
	installClient(t, twoNode())
	graphs := spyGraphs(t)
	def := resolve("g")
	if _, err := newRun(def, NativeGraphOptions{}).build(bg); err != nil {
		t.Fatal(err)
	}
	all := graphs()
	if len(all) != 1 {
		t.Fatalf("graphs built = %d", len(all))
	}
	g := all[0]
	if !reflect.DeepEqual(g.nodes, []string{"triage", "billing"}) || g.entry != "triage" || !g.schema {
		t.Errorf("nodes %v entry %q schema %v", g.nodes, g.entry, g.schema)
	}
	if !slices.Contains(g.edges, [2]string{"billing", stategraph.END}) || !slices.Contains(g.edges, [2]string{"triage", "billing"}) {
		t.Errorf("edges = %v", g.edges)
	}

	var calls []string
	r := newRun(def, NativeGraphOptions{ToolHandlers: handlersFor(&calls)})
	names := func(key string) []string {
		defs, _ := r.tools(def.GetNode(key), r.g.opts.ToolHandlers, nil)
		var out []string
		for _, d := range defs {
			out = append(out, d.Function.Name)
		}
		return out
	}
	if got := names("triage"); !reflect.DeepEqual(got, []string{"lookup", "transfer_to_billing"}) {
		t.Errorf("root tools = %v", got)
	}
	if got := names("billing"); !reflect.DeepEqual(got, []string{"refund"}) {
		t.Errorf("terminal tools = %v", got)
	}
	defs, _ := r.tools(def.GetNode("triage"), nil, nil)
	if len(defs) != 1 || !strings.Contains(defs[0].Function.Description, "billing") || !strings.Contains(defs[0].Function.Description, "billing questions") {
		t.Errorf("handoff tool = %+v", defs)
	}
}

// A non-terminal node with two outgoing edges gets exactly two handoff
// tools, alongside its regular tool.
func TestToLangGraphTwoHandoffs(t *testing.T) {
	installClient(t, &fakeLD{variations: map[string]map[string]any{
		"g":    graphFlag("hub", map[string][]any{"hub": {edge("a"), edge("b.c")}}),
		"hub":  node("Hub", "m", "lookup"),
		"a":    node("A", "m"),
		"b.c":  node("B", "m"),
		"none": nil,
	}})
	def := resolve("g")
	var calls []string
	r := newRun(def, NativeGraphOptions{ToolHandlers: handlersFor(&calls)})
	defs, _ := r.tools(def.GetNode("hub"), r.g.opts.ToolHandlers, nil)
	var names []string
	for _, d := range defs {
		names = append(names, d.Function.Name)
	}
	if !reflect.DeepEqual(names, []string{"lookup", "transfer_to_a", "transfer_to_b_c"}) {
		t.Errorf("tools = %v", names)
	}
	graphs := spyGraphs(t)
	if _, err := r.build(bg); err != nil {
		t.Fatal(err)
	}
	if g := graphs()[0]; !slices.Contains(g.edges, [2]string{"hub", stategraph.END}) || len(g.nodes) != 3 {
		t.Errorf("multi-child edges %v nodes %v", g.edges, g.nodes)
	}
}

// the registered node function puts a SystemMessage first, from
// instructions, or from config system messages when instructions are absent.
func TestToLangGraphSystemMessage(t *testing.T) {
	billing := node("", "billing-model")
	delete(billing, "instructions")
	billing["messages"] = []any{map[string]any{"role": "system", "content": "From messages"}, map[string]any{"role": "user", "content": "ignored"}}
	ld := twoNode()
	ld.variations["billing"] = billing
	installClient(t, ld)
	graphs := spyGraphs(t)
	def := resolve("g")
	fm := map[string]*fakeModel{"triage": {turns: []scripted{textTurn("x", 1, 1)}}, "billing": {turns: []scripted{textTurn("y", 1, 1)}}}
	r := newRun(def, NativeGraphOptions{ModelFactory: models(fm)})
	r.variables = map[string]any{"tone": "calm"}
	if _, err := r.build(bg); err != nil {
		t.Fatal(err)
	}
	outer := graphs()[0]
	empty := map[string]any{messagesKey: []llms.MessageContent{}}
	for key, want := range map[string]string{"triage": "Route calm", "billing": "From messages"} {
		if _, err := outer.fns[key](bg, empty); err != nil {
			t.Fatal(err)
		}
		first := fm[key].call(0).msgs[0]
		if first.Role != sys || text(first) != want {
			t.Errorf("%s first message = %v %q", key, first.Role, text(first))
		}
	}
}

// ToolNode wiring: a node with config.tools runs a ReAct subgraph
// whose tools node executes the wired handler.
func TestToLangGraphToolNode(t *testing.T) {
	installClient(t, twoNode())
	graphs := spyGraphs(t)
	def := resolve("g")
	var calls []string
	fm := map[string]*fakeModel{
		"triage":  {turns: []scripted{toolTurn("lookup", map[string]any{"id": 1}), textTurn("routed", 1, 1)}},
		"billing": {turns: []scripted{textTurn("billed", 1, 1)}},
	}
	res, err := ToLangGraph(def, &NativeGraphOptions{ToolHandlers: handlersFor(&calls), ModelFactory: models(fm)}).Invoke(bg, "hi")
	if err != nil {
		t.Fatal(err)
	}
	if res.Response != "billed" || !slices.Equal(calls, []string{"lookup"}) {
		t.Errorf("res %+v calls %v", res, calls)
	}
	all := graphs()
	if len(all) != 3 {
		t.Fatalf("graphs = %d", len(all))
	}
	inner := all[1] // triage's agent
	if !reflect.DeepEqual(inner.nodes, []string{agentNode, toolsNode}) || !slices.Contains(inner.conds, agentNode) ||
		!slices.Contains(inner.edges, [2]string{toolsNode, agentNode}) || !inner.schema {
		t.Errorf("subgraph = %+v", inner)
	}
	if got := toolNames(fm["triage"].call(0).opts); !reflect.DeepEqual(got, []string{"lookup", "transfer_to_billing"}) {
		t.Errorf("triage tools = %v", got)
	}
	if tr := fm["triage"].call(1).msgs[3].Parts[0].(llms.ToolCallResponse); tr.Content != "lookup done" {
		t.Errorf("tool result = %+v", tr)
	}
}

// NativeTool → "" result, tracked under its LD key; the handoff
// executor returns a Command to the target node.
func TestToLangGraphExecutors(t *testing.T) {
	ld := twoNode()
	installClient(t, ld)
	def := resolve("g")
	ld.resetTracks()
	native := &aiserver.NativeTool{ToolName: "web_search"}
	r := newRun(def, NativeGraphOptions{Context: user})
	handlers := aiserver.WrapToolHandlers(map[string]aiserver.ToolHandler{"lookup": native}, user, def.GetNode("triage").TrackData("g", "run"))
	defs, exec := r.tools(def.GetNode("triage"), handlers, nil)
	if len(defs) != 2 || defs[0].Function.Name != "lookup" {
		t.Fatalf("defs = %+v", defs)
	}
	m, cmd, err := exec(bg, toolCall("lookup", nil))
	if err != nil || cmd != nil || m.Parts[0].(llms.ToolCallResponse).Content != "" {
		t.Errorf("native: %+v %v %v", m, cmd, err)
	}
	if tc := ld.events("$ld:ai:tool_call"); len(tc) != 1 || tc[0].data["toolKey"] != "lookup" {
		t.Errorf("tool_call events = %+v", tc)
	}

	m, cmd, err = exec(bg, toolCall("transfer_to_billing", nil))
	if err != nil || cmd == nil || cmd.Goto != "billing" || !strings.Contains(m.Parts[0].(llms.ToolCallResponse).Content, "billing") {
		t.Errorf("handoff: %+v %+v %v", m, cmd, err)
	}
	if _, cmd, _ = exec(bg, toolCall("transfer_to_billing", nil)); cmd != nil {
		t.Error("a second handoff in one node run must not route again")
	}
	if c := handoffCommand("a.b"); c.Goto != "a_b" {
		t.Errorf("handoffCommand = %+v", c)
	}
	if _, _, err := exec(bg, toolCall("nope", nil)); !errors.Is(err, ErrNoToolHandler) {
		t.Errorf("unknown tool = %v", err)
	}
}

// execution: root hands off to leaf; response, usage, events
// and the graph span.
func TestToLangGraphRun(t *testing.T) {
	ld := twoNode()
	exp := installClient(t, ld)
	def := resolve("g")
	ld.resetTracks()
	var calls []string
	fm := map[string]*fakeModel{
		"triage":  {turns: []scripted{{resp: &llms.ContentResponse{Choices: []*llms.ContentChoice{{StopReason: "tool_use", ToolCalls: []llms.ToolCall{toolCall("transfer_to_billing", nil)}, GenerationInfo: usageInfo(3, 1)}}}}}},
		"billing": {turns: []scripted{toolTurn("refund", map[string]any{"id": 1}), textTurn("refunded", 5, 2)}},
	}
	res, err := ToLangGraph(def, &NativeGraphOptions{ToolHandlers: handlersFor(&calls), ModelFactory: models(fm), Context: user}).
		Invoke(bg, "hello", aiserver.WithVariables(map[string]any{"tone": "calm"}))
	if err != nil {
		t.Fatal(err)
	}
	if res.Response != "refunded" || res.Usage != (aiserver.TokenUsage{Input: 9, Output: 4, Total: 13}) || !slices.Equal(calls, []string{"refund"}) {
		t.Errorf("res = %+v calls %v", res, calls)
	}
	// The leaf sees the conversation so far, under its own system prompt.
	if got := turns(fm["billing"].call(0).msgs); len(got) != 4 || got[0] != (turn{sys, "Bill"}) || got[1] != (turn{human, "hello"}) || got[3].role != tool {
		t.Errorf("leaf input = %v", got)
	}
	if got := toolNames(fm["billing"].call(0).opts); !reflect.DeepEqual(got, []string{"refund"}) {
		t.Errorf("leaf tools = %v", got)
	}

	// The handoff is tracked once its target has answered, never before (a
	// failing target is a handoff_failure).
	want := []string{"$ld:ai:graph:node", "$ld:ai:duration:total", "$ld:ai:generation:success",
		"$ld:ai:tokens:total", "$ld:ai:tokens:input", "$ld:ai:tokens:output",
		"$ld:ai:graph:node", "$ld:ai:tool_call", "$ld:ai:duration:total", "$ld:ai:generation:success",
		"$ld:ai:tokens:total", "$ld:ai:tokens:input", "$ld:ai:tokens:output", "$ld:ai:graph:handoff_success",
		"$ld:ai:graph:duration:total", "$ld:ai:graph:total_tokens", "$ld:ai:graph:invocation_success"}
	if got := ld.aiEvents(); !reflect.DeepEqual(got, want) {
		t.Errorf("events =\n%v\nwant\n%v", got, want)
	}
	nodes := ld.events("$ld:ai:graph:node")
	if nodes[0].data["nodeKey"] != "triage" || nodes[0].data["index"] != 0.0 || nodes[1].data["nodeKey"] != "billing" ||
		nodes[1].data["index"] != 1.0 || nodes[0].value != 1 || nodes[1].data["graphKey"] != "g" || nodes[0].data["runId"] != nodes[1].data["runId"] {
		t.Errorf("node events = %+v", nodes)
	}
	if tok := ld.events("$ld:ai:tokens:total"); tok[0].value != 4 || tok[1].value != 9 || tok[1].data["configKey"] != "billing" {
		t.Errorf("token events = %+v", tok)
	}
	if h := ld.events("$ld:ai:graph:handoff_success")[0]; h.data["sourceKey"] != "triage" || h.data["targetKey"] != "billing" {
		t.Errorf("handoff = %+v", h.data)
	}
	if s := ld.events("$ld:ai:graph:invocation_success")[0]; s.data["graphKey"] != "g" || s.data["configKey"] != "g" {
		t.Errorf("success = %+v", s.data)
	}
	if len(ld.events("$ld:ai:graph:path")) != 0 {
		t.Error("$ld:ai:graph:path tracked")
	}

	spans := spansByName(exp)[aiserver.GraphSpanName]
	if len(spans) != 1 {
		t.Fatalf("graph spans = %d", len(spans))
	}
	a := spanAttrs(spans[0])
	if a["ld.ai.graph.key"] != "g" || a["gen_ai.usage.total_tokens"] != "13" || a["launchdarkly.graph.path"] != "triage->billing" ||
		spans[0].Status().Code != codes.Ok || spans[0].EndTime().IsZero() {
		t.Errorf("graph span = %v %v", a, spans[0].Status())
	}
	for k := range a {
		if strings.HasPrefix(k, "context.") {
			t.Errorf("graph span carries identity %s", k)
		}
	}
}

// failure path, and no tracking without a context.
func TestToLangGraphFailureAndNoContext(t *testing.T) {
	ld := twoNode()
	exp := installClient(t, ld)
	def := resolve("g")
	ld.resetTracks()
	boom := errors.New("model down")
	fm := map[string]*fakeModel{"triage": {turns: []scripted{{err: boom}}}}
	_, err := ToLangGraph(def, &NativeGraphOptions{ModelFactory: models(fm), Context: user}).Invoke(bg, "hi")
	if !errors.Is(err, boom) {
		t.Fatalf("err = %v", err)
	}
	if f := ld.events("$ld:ai:graph:invocation_failure"); len(f) != 1 || len(ld.events("$ld:ai:graph:invocation_success")) != 0 ||
		len(ld.events("$ld:ai:generation:error")) != 1 {
		t.Errorf("events = %v", ld.aiEvents())
	}
	if s := spansByName(exp)[aiserver.GraphSpanName]; len(s) != 1 || s[0].Status().Code != codes.Error || s[0].EndTime().IsZero() {
		t.Errorf("graph span = %v", s)
	}

	ld.resetTracks()
	fm = map[string]*fakeModel{"triage": {turns: []scripted{textTurn("a", 1, 1)}}, "billing": {turns: []scripted{textTurn("b", 1, 1)}}}
	if _, err := ToLangGraph(def, &NativeGraphOptions{ModelFactory: models(fm)}).Invoke(bg, "hi"); err != nil {
		t.Fatal(err)
	}
	if got := ld.aiEvents(); len(got) != 0 {
		t.Errorf("tracked without a context: %v", got)
	}
	ld.resetTracks()
	if _, err := ToLangGraph(def, &NativeGraphOptions{ModelFactory: models(map[string]*fakeModel{"triage": {turns: []scripted{{err: boom}}}})}).Invoke(bg, "hi"); err == nil || len(ld.aiEvents()) != 0 {
		t.Errorf("failure without a context: err %v events %v", err, ld.aiEvents())
	}
}

// disabled / null root / nil definitions.
func TestToLangGraphInvalid(t *testing.T) {
	ld := &fakeLD{}
	installClient(t, ld)
	disabled := resolve("missing-graph")
	if _, err := ToLangGraph(disabled, nil).Invoke(bg, "x"); !errors.Is(err, aiserver.ErrGraphDisabled) || !strings.Contains(err.Error(), "missing-graph") {
		t.Errorf("disabled = %v", err)
	}
	if _, err := ToLangGraph(&aiserver.GraphDefinition{Key: "rootless", Enabled: true}, nil).Invoke(bg, "x"); !errors.Is(err, ErrNoRoot) || !strings.Contains(err.Error(), "rootless") {
		t.Errorf("null root = %v", err)
	}
	if _, err := ToLangGraph(nil, nil).Invoke(bg, "x"); !errors.Is(err, ErrNilGraph) {
		t.Errorf("nil = %v", err)
	}
}

// Node keys that sanitize to stategraph's END sentinel, or to another
// node's name, are rejected before any model runs.
func TestToLangGraphNodeNameCollision(t *testing.T) {
	for name, flags := range map[string]map[string]map[string]any{
		"END": {
			"g":    graphFlag("root", map[string][]any{"root": {edge("END")}}),
			"root": node("Root", "m"), "END": node("End", "m"),
		},
		"duplicate": {
			"g":    graphFlag("root", map[string][]any{"root": {edge("a.b"), edge("a_b")}}),
			"root": node("Root", "m"), "a.b": node("A", "m"), "a_b": node("B", "m"),
		},
	} {
		t.Run(name, func(t *testing.T) {
			installClient(t, &fakeLD{variations: flags})
			m := &fakeModel{}
			_, err := ToLangGraph(resolve("g"), &NativeGraphOptions{ModelFactory: models(map[string]*fakeModel{"root": m})}).Invoke(bg, "x")
			if !errors.Is(err, ErrNodeName) || len(m.calls) != 0 {
				t.Errorf("err = %v, model calls = %d", err, len(m.calls))
			}
		})
	}
}

// extractUsage zeros, array content concatenation.
func TestToLangGraphUsageAndContent(t *testing.T) {
	ld := &fakeLD{variations: map[string]map[string]any{"g": graphFlag("solo", nil), "solo": node("Solo", "m")}}
	installClient(t, ld)
	def := resolve("g")
	ld.resetTracks()
	multi := &llms.ContentResponse{Choices: []*llms.ContentChoice{
		{Content: "Hello, ", Parts: []llms.ContentPart{llms.TextContent{Text: "Hello, "}}},
		{Content: "world", Parts: []llms.ContentPart{llms.TextContent{Text: "world"}}},
	}}
	fm := map[string]*fakeModel{"solo": {turns: []scripted{{resp: multi}}}}
	res, err := ToLangGraph(def, &NativeGraphOptions{ModelFactory: models(fm), Context: user}).Invoke(bg, "hi")
	if err != nil {
		t.Fatal(err)
	}
	if res.Response != "Hello, world" || res.Usage != (aiserver.TokenUsage{}) {
		t.Errorf("res = %+v", res)
	}
	if len(ld.events("$ld:ai:tokens:total")) != 0 || len(ld.events("$ld:ai:graph:total_tokens")) != 0 || len(ld.events("$ld:ai:generation:success")) != 1 {
		t.Errorf("events = %v", ld.aiEvents())
	}
}

// history reaches the root only, in native (image) form.
func TestToLangGraphHistoryRootOnly(t *testing.T) {
	installClient(t, twoNode())
	def := resolve("g")
	fm := map[string]*fakeModel{
		"triage":  {turns: []scripted{toolTurn("transfer_to_billing", nil)}},
		"billing": {turns: []scripted{textTurn("blue", 1, 1)}},
	}
	hist := []aiserver.Message{image(), {Role: "assistant", Text: "an image"}}
	if _, err := ToLangGraph(def, &NativeGraphOptions{ModelFactory: models(fm)}).Invoke(bg, "what colour?", aiserver.WithHistory(hist)); err != nil {
		t.Fatal(err)
	}
	root := fm["triage"].call(0).msgs
	if _, ok := root[1].Parts[0].(llms.BinaryContent); !ok || len(root) != 4 || text(root[3]) != "what colour?" {
		t.Errorf("root input = %v", turns(root))
	}
	for _, m := range fm["billing"].call(0).msgs {
		for _, p := range m.Parts {
			if _, ok := p.(llms.BinaryContent); ok || text(m) == "an image" {
				t.Errorf("leaf received history: %v", turns(fm["billing"].call(0).msgs))
			}
		}
	}
}

// A handoff cycle stops at the recursion limit.
func TestToLangGraphCycle(t *testing.T) {
	installClient(t, &fakeLD{variations: map[string]map[string]any{
		"g": graphFlag("a", map[string][]any{"a": {edge("b")}, "b": {edge("a")}}),
		"a": node("A", "m"), "b": node("B", "m"),
	}})
	def := resolve("g")
	fm := map[string]*fakeModel{"a": {turns: []scripted{toolTurn("transfer_to_b", nil)}, repeat: true}, "b": {turns: []scripted{toolTurn("transfer_to_a", nil)}, repeat: true}}
	if _, err := ToLangGraph(def, &NativeGraphOptions{ModelFactory: models(fm)}).Invoke(bg, "x"); !errors.Is(err, ErrRecursionLimit) {
		t.Errorf("err = %v", err)
	}
}

// the real stategraph runner (no spy) with a fake
// model. The messages channel must merge through the messages reducer
// across nodes, or the second model turn would not see the tool result.
func TestToLangGraphRealStateGraph(t *testing.T) {
	installClient(t, &fakeLD{variations: map[string]map[string]any{"g": graphFlag("solo", nil), "solo": node("Solo", "m", "lookup")}})
	var calls []string
	fm := map[string]*fakeModel{"solo": {turns: []scripted{toolTurn("lookup", nil), textTurn("done", 1, 1)}}}
	res, err := ToLangGraph(resolve("g"), &NativeGraphOptions{ToolHandlers: handlersFor(&calls), ModelFactory: models(fm)}).Invoke(bg, "hello")
	if err != nil {
		t.Fatal(err)
	}
	second := fm["solo"].call(1).msgs
	if res.Response != "done" || len(second) != 4 || second[3].Role != tool || text(second[1]) != "hello" {
		t.Errorf("res %+v second call %v", res, turns(second))
	}
	if got, _ := (messagesSchema{}).Update(map[string]any{"x": 1}, nil); !reflect.DeepEqual(got, map[string]any{"x": 1}) {
		t.Errorf("a Command without Update must keep the state: %v", got)
	}
}

// LangChainGraph: key, one handler, options and llm forwarded.
func TestLangChainGraph(t *testing.T) {
	ld := &fakeLD{variations: map[string]map[string]any{"g": graphFlag("solo", nil), "solo": node("Solo", "test-model", "lookup")}}
	installClient(t, ld)
	f := &fakeModel{turns: []scripted{streamTurn(nil, 1, 1, toolCall("lookup", nil)), streamTurn([]string{"fine"}, 1, 1)}}
	bogusCalled := false
	bogus := &aiserver.ProviderHandler{ProvidesFor: ProvidesFor, Invoke: func(context.Context, aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
		bogusCalled = true
		return aiserver.HandlerResult{}, nil
	}}
	var calls []string
	res, err := LangChainGraph("g", aiserver.GraphOptions{Handlers: []*aiserver.ProviderHandler{bogus, bogus}, ToolHandlers: handlersFor(&calls)},
		Options{Model: f}).Invoke(bg, "q", user)
	if err != nil {
		t.Fatal(err)
	}
	if res.Response != "fine" || bogusCalled || f.count() != 2 || !slices.Equal(calls, []string{"lookup"}) || ld.requestedKeys()[0] != "g" {
		t.Errorf("res=%+v bogus=%v calls=%d tools=%v keys=%v", res, bogusCalled, f.count(), calls, ld.requestedKeys())
	}
	if got := ld.usageHelpers(); !reflect.DeepEqual(got, []string{"langchain-agents.langchainGraph"}) {
		t.Errorf("usage helpers = %v", got)
	}
}

// toLangGraph reports only itself.
func TestToLangGraphSdkUsage(t *testing.T) {
	ld := &fakeLD{variations: map[string]map[string]any{"g": graphFlag("solo", nil), "solo": node("Solo", "m")}}
	installClient(t, ld)
	def := resolve("g")
	ld.resetTracks()
	fm := map[string]*fakeModel{"solo": {turns: []scripted{textTurn("x", 1, 1)}}}
	if _, err := ToLangGraph(def, &NativeGraphOptions{ModelFactory: models(fm), Context: user}).Invoke(bg, "q"); err != nil {
		t.Fatal(err)
	}
	if got := ld.usageHelpers(); !reflect.DeepEqual(got, []string{"langchain-agents.toLangGraph"}) {
		t.Errorf("usage helpers = %v", got)
	}
}

// A handoff whose target fails is a handoff_failure, never a success; the
// graph duration is tracked on failure too.
func TestToLangGraphHandoffFailure(t *testing.T) {
	ld := twoNode()
	installClient(t, ld)
	def := resolve("g")
	ld.resetTracks()
	boom := errors.New("billing down")
	fm := map[string]*fakeModel{
		"triage":  {turns: []scripted{toolTurn("transfer_to_billing", nil)}},
		"billing": {turns: []scripted{{err: boom}}},
	}
	if _, err := ToLangGraph(def, &NativeGraphOptions{ModelFactory: models(fm), Context: user}).Invoke(bg, "hi"); !errors.Is(err, boom) {
		t.Fatalf("err = %v", err)
	}
	if hs := ld.events("$ld:ai:graph:handoff_success"); len(hs) != 0 {
		t.Errorf("handoff_success for a failed target: %+v", hs)
	}
	hf := ld.events("$ld:ai:graph:handoff_failure")
	if len(hf) != 1 || hf[0].data["sourceKey"] != "triage" || hf[0].data["targetKey"] != "billing" || hf[0].data["configKey"] != "g" {
		t.Errorf("handoff_failure = %+v", hf)
	}
	if d := ld.events("$ld:ai:graph:duration:total"); len(d) != 1 {
		t.Errorf("graph duration on failure = %+v", d)
	}
}

// Graph events carry the graph's track data, node events the node's; each
// node run has a langchain.agent span under the graph span with the node's
// association and the context identity, and its chat / execute_tool children
// carry neither.
func TestToLangGraphNodeSpansAndPayloads(t *testing.T) {
	ld := twoNode()
	exp := installClient(t, ld)
	def := resolve("g")
	ld.resetTracks()
	var calls []string
	fm := map[string]*fakeModel{
		"triage":  {turns: []scripted{toolTurn("transfer_to_billing", nil)}},
		"billing": {turns: []scripted{toolTurn("refund", map[string]any{"id": 1}), textTurn("refunded", 5, 2)}},
	}
	if _, err := ToLangGraph(def, &NativeGraphOptions{ToolHandlers: handlersFor(&calls), ModelFactory: models(fm), Context: user}).Invoke(bg, "hello"); err != nil {
		t.Fatal(err)
	}
	runID := ld.events("$ld:ai:graph:invocation_success")[0].data["runId"]
	for _, ev := range []string{"$ld:ai:graph:node", "$ld:ai:graph:handoff_success", "$ld:ai:graph:duration:total", "$ld:ai:graph:total_tokens"} {
		for _, c := range ld.events(ev) {
			if c.data["configKey"] != "g" || c.data["graphKey"] != "g" || c.data["runId"] != runID {
				t.Errorf("%s data = %v", ev, c.data)
			}
		}
	}
	for _, c := range ld.events("$ld:ai:generation:success") {
		if c.data["graphKey"] != "g" || c.data["runId"] != runID || c.data["variationKey"] != "v-"+c.data["modelName"].(string) || c.data["providerName"] != "OpenAI" {
			t.Errorf("node data = %v", c.data)
		}
	}

	byName := spansByName(exp)
	graphSpan := byName[aiserver.GraphSpanName][0]
	nodes := byName[SpanName]
	if len(nodes) != 2 {
		t.Fatalf("node spans = %d", len(nodes))
	}
	for i, key := range []string{"triage", "billing"} {
		s := nodes[i]
		a := spanAttrs(s)
		if s.Parent().SpanID() != graphSpan.SpanContext().SpanID() || a["launchdarkly.config.key"] != key || a["launchdarkly.graph.key"] != "g" ||
			a["context.contextKeys.user"] != "u-1" || a["gen_ai.operation.name"] != "invoke_agent" || s.Status().Code != codes.Ok || s.EndTime().IsZero() {
			t.Errorf("%s span = %v %v", key, a, s.Status())
		}
	}
	if a := spanAttrs(nodes[1]); a["gen_ai.usage.input_tokens"] != "6" || a["gen_ai.usage.total_tokens"] != "9" || a["gen_ai.completion.0.content"] != "refunded" {
		t.Errorf("leaf span usage/content = %v", a)
	}
	for _, name := range []string{"execute_tool transfer_to_billing", "execute_tool refund"} {
		if len(byName[name]) != 1 {
			t.Errorf("%s spans = %d", name, len(byName[name]))
		}
	}
	for name, ss := range byName {
		if name == SpanName || name == aiserver.GraphSpanName {
			continue
		}
		for k := range spanAttrs(ss[0]) {
			if strings.HasPrefix(k, "context.") || k == "launchdarkly.config.key" {
				t.Errorf("child %s carries %s", name, k)
			}
		}
	}
}
