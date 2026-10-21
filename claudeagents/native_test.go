package claudeagents

import (
	"bytes"
	"context"
	"errors"
	"log/slog"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"go.opentelemetry.io/otel/codes"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/sdkapi"
)

const serverSearch = `{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{"query":"x"}},
{"type":"web_search_tool_result","tool_use_id":"srvtoolu_1","content":[]},{"type":"text","text":"found"}`

func graphFlags(rootExtra map[string]any) map[string]any {
	return map[string]any{
		"g": map[string]any{
			"root":    "root",
			"edges":   map[string]any{"root": []any{map[string]any{"key": "leaf", "handoff": map[string]any{"description": "Ask the leaf"}}}},
			"_ldMeta": map[string]any{"enabled": true, "variationKey": "gv", "version": 1},
		},
		"root": agentVariation("root inst", rootExtra),
		"leaf": agentVariation("leaf inst", nil),
	}
}

func resolve(t *testing.T) *aiserver.GraphDefinition {
	t.Helper()
	def := aiserver.ResolveGraph(bg, "g", user, aiserver.GraphOptions{})
	if !def.Enabled {
		t.Fatal("graph did not resolve")
	}
	return def
}

func names(calls []trackCall, field string) []any {
	var out []any
	for _, c := range calls {
		out = append(out, c.Data[field])
	}
	return out
}

// toClaudeAgents: topology, handoff tools, events, result, span.
func TestToClaudeAgentsRun(t *testing.T) {
	e := setup(t, graphFlags(nil),
		toolResp("mcp__subagents__leaf", `{"input":"sub task"}`), textResp("leaf says hi", 2, 3), textResp("Final: leaf says hi", 4, 5))
	img := aiserver.ContentBlock{Type: "image", Source: &aiserver.ImageSource{Type: "url", URL: "https://x/y.png"}}
	runner := ToClaudeAgents(resolve(t), &AdapterOptions{Handler: e.opts})
	res, err := runner.Invoke(bg, "help me", aiserver.WithHistory([]aiserver.Message{{Role: "user", Blocks: []aiserver.ContentBlock{img}}}))
	if err != nil {
		t.Fatal(err)
	}
	if res.Response != "Final: leaf says hi" || res.Usage.Total != 1+1+2+3+4+5 {
		t.Errorf("response=%q usage=%+v", res.Response, res.Usage)
	}
	r := e.api.reqs()
	if len(r) != 3 {
		t.Fatalf("want 2 agents / 3 calls, got %d", len(r))
	}
	root, leaf := r[0], r[1]
	tool := root["tools"].([]any)[0].(map[string]any)
	if system(root) != "root inst" || tool["name"] != "mcp__subagents__leaf" || tool["description"] != "Ask the leaf" {
		t.Errorf("root: system=%q tool=%v", system(root), tool)
	}
	if system(leaf) != "leaf inst" || leaf["tools"] != nil || firstText(messages(leaf)[0]) != "sub task" || len(messages(leaf)) != 1 {
		t.Errorf("leaf: system=%q tools=%v msgs=%v", system(leaf), leaf["tools"], messages(leaf))
	}
	// history reaches the root only, as a native image block
	if m := messages(root); len(m) != 2 || m[0]["content"].([]any)[0].(map[string]any)["source"].(map[string]any)["url"] != "https://x/y.png" {
		t.Errorf("root history: %v", m)
	}

	nodes := e.ld.events("$ld:ai:graph:node")
	if !reflect.DeepEqual(names(nodes, "nodeKey"), []any{"root", "leaf"}) || !reflect.DeepEqual(names(nodes, "index"), []any{0.0, 1.0}) || nodes[0].Metric != 1 {
		t.Errorf("node events = %+v", nodes)
	}
	hs := e.ld.events("$ld:ai:graph:handoff_success")
	if len(hs) != 1 || hs[0].Data["sourceKey"] != "root" || hs[0].Data["targetKey"] != "leaf" || hs[0].Data["graphKey"] != "g" {
		t.Errorf("handoff events = %+v", hs)
	}
	if len(e.ld.events("$ld:ai:graph:invocation_success")) != 1 || len(e.ld.events("$ld:ai:graph:invocation_failure")) != 0 {
		t.Error("invocation events wrong")
	}
	if gen := e.ld.events("$ld:ai:generation:success"); !reflect.DeepEqual(names(gen, "configKey"), []any{"leaf", "root"}) {
		t.Errorf("per-node generation events = %v", names(gen, "configKey"))
	}
	if tt := e.ld.events("$ld:ai:graph:total_tokens"); len(tt) != 1 || tt[0].Metric != 16 {
		t.Errorf("total tokens = %+v", tt)
	}

	g := e.spans(aiserver.GraphSpanName)
	if len(g) != 1 || g[0].Status().Code != codes.Ok || g[0].EndTime().IsZero() {
		t.Fatal("graph span not ended OK")
	}
	ga := attrs(g[0])
	for k := range ga {
		if strings.HasPrefix(k, "context.contextKeys.") {
			t.Errorf("ld.ai.graph carries context identity %s; it belongs on handler root spans only", k)
		}
	}
	if ga["ld.ai.graph.key"] != "g" || ga["launchdarkly.graph.path"] != "root->leaf" || ga["gen_ai.usage.total_tokens"] != "16" {
		t.Errorf("graph span attrs = %v", ga)
	}
	q := e.spans(SpanName)
	if len(q) != 2 {
		t.Fatalf("want 2 claude.query spans, got %d", len(q))
	}
	// the leaf runs inside the root's execute_tool span
	tools := e.spans("execute_tool mcp__subagents__leaf")
	if len(tools) != 1 || q[0].Parent().SpanID() != tools[0].SpanContext().SpanID() {
		t.Error("leaf span not nested under the sub-agent tool span")
	}
	// each node's root span carries the graph key and the context
	// identity; child chat / execute_tool spans carry neither.
	for _, s := range q {
		got := attrs(s)
		ff := map[string]string{}
		for _, ev := range s.Events() {
			if ev.Name == "feature_flag" {
				for _, a := range ev.Attributes {
					ff[string(a.Key)] = a.Value.AsString()
				}
			}
		}
		if got["launchdarkly.graph.key"] != "g" || got["context.contextKeys.user"] != "user-123" ||
			ff["feature_flag.context.id"] != "user-123" || ff["feature_flag.contextKeys"] != `{"user":"user-123"}` {
			t.Errorf("node root attrs=%v feature_flag=%v", got, ff)
		}
	}
	for _, s := range e.exp.GetSpans().Snapshots() {
		if s.Name() == SpanName || s.Name() == aiserver.GraphSpanName {
			continue
		}
		for k := range attrs(s) {
			if strings.HasPrefix(k, "context.") || strings.HasPrefix(k, "feature_flag.") || k == "launchdarkly.graph.key" {
				t.Errorf("child %s carries %s", s.Name(), k)
			}
		}
		for _, ev := range s.Events() {
			if ev.Name == "feature_flag" {
				t.Errorf("child %s has a feature_flag event", s.Name())
			}
		}
	}
}

// tool forwarding: a function tool on the leaf reaches only the leaf,
// runs there, and is tracked under the leaf's config key.
func TestToClaudeAgentsLeafTool(t *testing.T) {
	flags := graphFlags(nil)
	flags["leaf"] = agentVariation("leaf inst", map[string]any{"tools": map[string]any{
		"lookup": map[string]any{"name": "lookup", "type": "function", "parameters": map[string]any{"type": "object"}}}})
	e := setup(t, flags, toolResp("mcp__subagents__leaf", `{"input":"find it"}`), toolResp("lookup", `{"id":"7"}`),
		textResp("leaf found 7", 1, 1), textResp("done", 1, 1))
	var gotArgs map[string]any
	tools := map[string]aiserver.ToolHandler{"lookup": aiserver.ToolFunc(func(_ context.Context, args map[string]any) (any, error) {
		gotArgs = args
		return "record 7", nil
	})}
	res, err := ToClaudeAgents(resolve(t), &AdapterOptions{Handler: e.opts, ToolHandlers: tools}).Invoke(bg, "q")
	if err != nil || res.Response != "done" {
		t.Fatalf("res=%+v err=%v", res, err)
	}
	r := e.api.reqs()
	if len(r) != 4 || !reflect.DeepEqual(toolNames(r[0]), []string{"mcp__subagents__leaf"}) || !reflect.DeepEqual(toolNames(r[1]), []string{"lookup"}) {
		t.Fatalf("calls=%d root tools=%v leaf tools=%v", len(r), toolNames(r[0]), toolNames(r[1]))
	}
	if gotArgs["id"] != "7" {
		t.Errorf("tool args = %v", gotArgs)
	}
	tc := e.ld.events("$ld:ai:tool_call")
	if len(tc) != 1 || tc[0].Data["toolKey"] != "lookup" || tc[0].Data["configKey"] != "leaf" {
		t.Errorf("tool_call = %+v", tc)
	}
}

// a node that runs twice emits $ld:ai:graph:node once; no graph:path event.
func TestToClaudeAgentsRepeatNode(t *testing.T) {
	e := setup(t, graphFlags(nil),
		toolResp("mcp__subagents__leaf", `{"input":"one"}`), textResp("first", 1, 1),
		toolResp("mcp__subagents__leaf", `{"input":"two"}`), textResp("second", 1, 1),
		textResp("done", 1, 1))
	if _, err := ToClaudeAgents(resolve(t), &AdapterOptions{Handler: e.opts}).Invoke(bg, "q"); err != nil {
		t.Fatal(err)
	}
	if got := names(e.ld.events("$ld:ai:graph:node"), "nodeKey"); !reflect.DeepEqual(got, []any{"root", "leaf"}) {
		t.Errorf("node events = %v", got)
	}
	if hs := e.ld.events("$ld:ai:graph:handoff_success"); len(hs) != 2 {
		t.Errorf("handoff events = %d, want 2", len(hs))
	}
	if p := e.ld.events("$ld:ai:graph:path"); len(p) != 0 {
		t.Errorf("graph:path tracked: %+v", p)
	}
}

// an empty history behaves like no history.
func TestToClaudeAgentsEmptyHistory(t *testing.T) {
	e := setup(t, graphFlags(nil), textResp("a", 1, 1), textResp("b", 1, 1))
	runner := ToClaudeAgents(resolve(t), &AdapterOptions{Handler: e.opts})
	if _, err := runner.Invoke(bg, "q"); err != nil {
		t.Fatal(err)
	}
	if _, err := runner.Invoke(bg, "q", aiserver.WithHistory([]aiserver.Message{})); err != nil {
		t.Fatal(err)
	}
	r := e.api.reqs()
	if len(r) != 2 || !reflect.DeepEqual(r[0], r[1]) {
		t.Errorf("empty history request differs:\n%v\n%v", r[0], r[len(r)-1])
	}
}

func TestToClaudeAgentsFailure(t *testing.T) {
	e := setup(t, graphFlags(nil), "STATUS400")
	_, err := ToClaudeAgents(resolve(t), &AdapterOptions{Handler: e.opts}).Invoke(bg, "q")
	if err == nil {
		t.Fatal("want error")
	}
	if len(e.ld.events("$ld:ai:graph:invocation_failure")) != 1 || len(e.ld.events("$ld:ai:graph:invocation_success")) != 0 {
		t.Error("failure not tracked")
	}
	g := e.spans(aiserver.GraphSpanName)
	if len(g) != 1 || g[0].Status().Code != codes.Error || g[0].EndTime().IsZero() {
		t.Error("graph span not ended ERROR")
	}
}

func TestToClaudeAgentsDisabled(t *testing.T) {
	setup(t, map[string]any{})
	def := aiserver.ResolveGraph(bg, "missing", user, aiserver.GraphOptions{})
	if _, err := ToClaudeAgents(def, nil).Invoke(bg, "q"); !errors.Is(err, aiserver.ErrGraphDisabled) {
		t.Errorf("want ErrGraphDisabled, got %v", err)
	}
	if _, err := ToClaudeAgents(nil, nil).Invoke(bg, "q"); !errors.Is(err, aiserver.ErrGraphDisabled) {
		t.Errorf("nil def: %v", err)
	}
}

// NativeTool: wired to the server tool, tracked under the LD map key.
func TestToClaudeAgentsNativeTool(t *testing.T) {
	tools := map[string]any{"web-search": map[string]any{"name": "web-search", "type": "function", "parameters": map[string]any{}}}
	e := setup(t, graphFlags(map[string]any{"tools": tools}), msgJSON("end_turn", serverSearch, 1, 1))
	res, err := ToClaudeAgents(resolve(t), &AdapterOptions{Handler: e.opts,
		ToolHandlers: map[string]aiserver.ToolHandler{"web-search": ClaudeWebSearch}}).Invoke(bg, "q")
	if err != nil || res.Response != "found" {
		t.Fatalf("%v %v", res, err)
	}
	if got := toolNames(e.api.reqs()[0]); !slices.Contains(got, "web_search") {
		t.Errorf("tools = %v", got)
	}
	tc := e.ld.events("$ld:ai:tool_call")
	if len(tc) != 1 || tc[0].Data["toolKey"] != "web-search" || tc[0].Data["configKey"] != "root" {
		t.Errorf("tool_call = %+v", tc)
	}
}

func TestClaudeGraph(t *testing.T) {
	flags := map[string]any{
		"g":    map[string]any{"root": "root", "_ldMeta": map[string]any{"enabled": true}},
		"root": agentVariation("root inst", map[string]any{"tools": map[string]any{"lookup": map[string]any{"name": "lookup", "type": "function", "parameters": map[string]any{}}}}),
	}
	e := setup(t, flags, sse(nil, "lookup", 1, 1), sse([]string{"ok"}, "", 1, 1)) // graph() streams each node
	stubCalled := false
	stub := sdkapi.CreateHandler(ProvidesFor, func(context.Context, aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
		stubCalled = true
		return aiserver.HandlerResult{Output: "stub"}, nil
	}, nil, false)
	looked := false
	g := ClaudeGraph("g", GraphOptions{
		Registry: aiserver.NewRegistry(aiserver.RegistryEntries{Handlers: []*aiserver.ProviderHandler{stub}}),
		ToolHandlers: map[string]aiserver.ToolHandler{"lookup": aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) {
			looked = true
			return "data", nil
		})},
	}, e.opts)
	res, err := g.Invoke(bg, "q", user)
	if err != nil || res.Response != "ok" || stubCalled || !looked || len(e.api.reqs()) != 2 {
		t.Errorf("res=%v err=%v stub=%v looked=%v calls=%d", res, err, stubCalled, looked, len(e.api.reqs()))
	}
}

// GraphJudge is forwarded to graph() and runs on this handler.
func TestClaudeGraphJudge(t *testing.T) {
	judge := agentVariation("Grade it", nil)
	judge["_ldMeta"] = map[string]any{"enabled": true, "variationKey": "jv", "version": 1, "mode": "judge"}
	flags := map[string]any{
		"g":    map[string]any{"root": "root", "_ldMeta": map[string]any{"enabled": true}},
		"root": agentVariation("root inst", nil),
		"gj":   judge,
	}
	e := setup(t, flags, sse([]string{"ok"}, "", 1, 1), textResp(`{"score":0.9,"reasoning":"fine"}`, 1, 1))
	res, err := ClaudeGraph("g", GraphOptions{GraphJudge: "gj"}, e.opts).Invoke(bg, "q", user)
	if err != nil || res.JudgeResults["gj"].Score != 0.9 || len(e.api.reqs()) != 2 {
		t.Fatalf("res=%+v err=%v calls=%d", res, err, len(e.api.reqs()))
	}
}

// conversation id on graph and handler spans; none invented.
func TestConversationID(t *testing.T) {
	e := setup(t, graphFlags(nil), textResp("a", 1, 1), textResp("b", 1, 1))
	def := resolve(t)
	if _, err := ToClaudeAgents(def, &AdapterOptions{Handler: e.opts}).Invoke(aiserver.WithConversationID(bg, "conv-1"), "q"); err != nil {
		t.Fatal(err)
	}
	for _, s := range e.exp.GetSpans().Snapshots() {
		if attrs(s)["gen_ai.conversation.id"] != "conv-1" {
			t.Errorf("%s missing conversation id", s.Name())
		}
	}
	e.exp.Reset()
	if _, err := ToClaudeAgents(def, &AdapterOptions{Handler: e.opts}).Invoke(bg, "q"); err != nil {
		t.Fatal(err)
	}
	for _, s := range e.exp.GetSpans().Snapshots() {
		if _, has := attrs(s)["gen_ai.conversation.id"]; has {
			t.Errorf("%s has an invented conversation id", s.Name())
		}
	}
}

// Sub-agent tool names stay within the Anthropic 64-character limit and
// unique per parent when two keys sanitize to the same name.
func TestSubAgentToolName(t *testing.T) {
	taken := map[string]bool{}
	first := subAgentToolName("a.b", taken)
	taken[first] = true
	second := subAgentToolName("a_b", taken)
	if first != "mcp__subagents__a_b" || second == first || !strings.HasPrefix(second, "mcp__subagents__a_b_") {
		t.Errorf("names = %q, %q", first, second)
	}
	long := strings.Repeat("k", 100)
	got := subAgentToolName(long, map[string]bool{})
	if len(got) != maxToolNameLen || !unsafeToolChars.MatchString(got+"!") || unsafeToolChars.MatchString(got) {
		t.Errorf("long name %q (%d chars)", got, len(got))
	}
	if other := subAgentToolName(long+"x", map[string]bool{}); other == got {
		t.Errorf("distinct long keys share name %q", got)
	}
}

// A child that fails is a failed handoff, tracked after the child ran and
// never as a success; graph events carry the graph's track data and per-node
// events the node's.
func TestToClaudeAgentsChildFailure(t *testing.T) {
	e := setup(t, graphFlags(nil), toolResp("mcp__subagents__leaf", `{"input":"sub"}`), "STATUS400")
	if _, err := ToClaudeAgents(resolve(t), &AdapterOptions{Handler: e.opts}).Invoke(bg, "q"); err == nil {
		t.Fatal("want error")
	}
	if hs := e.ld.events("$ld:ai:graph:handoff_success"); len(hs) != 0 {
		t.Errorf("handoff_success tracked for a failed child: %+v", hs)
	}
	hf := e.ld.events("$ld:ai:graph:handoff_failure")
	if len(hf) != 1 || hf[0].Data["sourceKey"] != "root" || hf[0].Data["targetKey"] != "leaf" || hf[0].Data["configKey"] != "g" {
		t.Errorf("handoff_failure = %+v", hf)
	}
	if ge := e.ld.events("$ld:ai:generation:error"); !reflect.DeepEqual(names(ge, "configKey"), []any{"leaf", "root"}) {
		t.Errorf("generation errors = %v", names(ge, "configKey"))
	}
	if d := e.ld.events("$ld:ai:graph:duration:total"); len(d) != 1 {
		t.Errorf("graph duration on failure = %+v", d)
	}
}

// Graph events use the graph's track data (configKey = graph key, the graph
// variation) as aiserver.Graph does; node events use the node's.
func TestToClaudeAgentsPayloads(t *testing.T) {
	e := setup(t, graphFlags(nil), toolResp("mcp__subagents__leaf", `{"input":"sub"}`), textResp("leaf", 1, 1), textResp("done", 1, 1))
	if _, err := ToClaudeAgents(resolve(t), &AdapterOptions{Handler: e.opts}).Invoke(bg, "q"); err != nil {
		t.Fatal(err)
	}
	var runID any
	for _, ev := range []string{"$ld:ai:graph:node", "$ld:ai:graph:handoff_success", "$ld:ai:graph:invocation_success", "$ld:ai:graph:duration:total", "$ld:ai:graph:total_tokens"} {
		for _, c := range e.ld.events(ev) {
			if runID == nil {
				runID = c.Data["runId"]
			}
			if c.Data["configKey"] != "g" || c.Data["graphKey"] != "g" || c.Data["variationKey"] != "gv" || c.Data["runId"] != runID || runID == "" {
				t.Errorf("%s data = %v", ev, c.Data)
			}
		}
	}
	for _, c := range e.ld.events("$ld:ai:generation:success") {
		if c.Data["graphKey"] != "g" || c.Data["runId"] != runID || c.Data["modelName"] != "test-model" || c.Data["providerName"] != "Anthropic" || c.Data["variationKey"] != "v1" || c.Data["version"] != 2.0 {
			t.Errorf("node data = %v", c.Data)
		}
	}
}

// a definition without a context tracks nothing.
func TestToClaudeAgentsNoContext(t *testing.T) {
	e := setup(t, graphFlags(nil), toolResp("mcp__subagents__leaf", `{"input":"sub"}`), textResp("leaf", 1, 1), textResp("done", 1, 1))
	def := aiserver.ResolveGraph(bg, "g", ldcontext.Context{}, aiserver.GraphOptions{})
	if _, err := ToClaudeAgents(def, &AdapterOptions{Handler: e.opts}).Invoke(bg, "q"); err != nil {
		t.Fatal(err)
	}
	e.ld.mu.Lock()
	defer e.ld.mu.Unlock()
	for _, c := range e.ld.tracks {
		if !strings.HasPrefix(c.Event, "$ld:ai:sdk:") {
			t.Errorf("tracked %s without a context", c.Event)
		}
	}
}

// A cycle back-edge (B -> A, with A still unbuilt) is dropped with a warning.
func TestToClaudeAgentsCycleBackEdgeDropped(t *testing.T) {
	var buf bytes.Buffer
	aiserver.SetLogger(slog.New(slog.NewTextHandler(&buf, nil)))
	t.Cleanup(func() { aiserver.SetLogger(nil) })
	edge := func(k string) []any { return []any{map[string]any{"key": k}} }
	e := setup(t, map[string]any{
		"g": map[string]any{
			"root":    "root",
			"edges":   map[string]any{"root": edge("A"), "A": edge("B"), "B": edge("A")},
			"_ldMeta": map[string]any{"enabled": true, "variationKey": "gv", "version": 1},
		},
		"root": agentVariation("root inst", nil), "A": agentVariation("a", nil), "B": agentVariation("b", nil),
	}, textResp("done", 1, 1))
	if _, err := ToClaudeAgents(resolve(t), &AdapterOptions{Handler: e.opts}).Invoke(bg, "q"); err != nil {
		t.Fatal(err)
	}
	if got := toolNames(e.api.reqs()[0]); !slices.Contains(got, "mcp__subagents__A") {
		t.Errorf("root tools = %v", got)
	}
	if out := buf.String(); !strings.Contains(out, "dropping cyclic edge") || !strings.Contains(out, "source=B target=A") {
		t.Errorf("log = %q", out)
	}
}
