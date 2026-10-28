package aiserver

import (
	"bytes"
	"context"
	"errors"
	"iter"
	"log/slog"
	"reflect"
	"strings"
	"sync"
	"testing"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

func nodeCfg(instructions string) map[string]any {
	c := enabledConfig("P", "agent", nil)
	c["instructions"] = instructions
	return c
}

// twoNode is root-node → leaf-node.
func twoNode(extra map[string]any) map[string]any {
	v := map[string]any{
		"g":         map[string]any{"root": "root-node", "edges": map[string]any{"root-node": []any{map[string]any{"key": "leaf-node"}}}},
		"root-node": nodeCfg("root work"),
		"leaf-node": nodeCfg("leaf work"),
	}
	for k, x := range extra {
		v[k] = x
	}
	return v
}

type nodeCall struct {
	Key     string
	Req     HandlerRequest
	TraceOK bool
}

// recordingHandler answers "<node> out" with 1/1 tokens, opens a span off ctx
// like a real handler, and can call a handoff tool or fail per node.
type recordingHandler struct {
	mu     sync.Mutex
	calls  []nodeCall
	pick   map[string]string // node → handoff target
	failOn string
	stream bool
}

func (r *recordingHandler) handler() *ProviderHandler {
	invoke := func(ctx context.Context, req HandlerRequest) (HandlerResult, error) {
		key := req.Variables[VarLD].(TrackData).ConfigKey
		_, span := otel.Tracer(TracerPrefix+"/test").Start(ctx, "handler "+key)
		defer span.End()
		r.mu.Lock()
		r.calls = append(r.calls, nodeCall{Key: key, Req: req})
		r.mu.Unlock()
		if key == r.failOn {
			return HandlerResult{}, errors.New("node failed: " + key)
		}
		if target := r.pick[key]; target != "" {
			if fn, ok := req.ToolHandlers["__handoff_"+target].(ToolFunc); ok {
				_, _ = fn(ctx, nil)
			}
		}
		return HandlerResult{Output: key + " out", Usage: map[string]any{"input": 1.0, "output": 1.0}}, nil
	}
	var stream StreamFunc
	if r.stream {
		stream = func(ctx context.Context, req HandlerRequest) iter.Seq2[HandlerStreamEvent, error] {
			return func(yield func(HandlerStreamEvent, error) bool) {
				res, err := invoke(ctx, req)
				if err != nil {
					yield(HandlerStreamEvent{}, err)
					return
				}
				out := res.Output.(string)
				for _, part := range strings.SplitAfter(out, " ") {
					if !yield(HandlerStreamEvent{Type: "chunk", Text: part}, nil) {
						return
					}
				}
				yield(HandlerStreamEvent{Type: "done", Usage: res.Usage}, nil)
			}
		}
	}
	return createHandler(ProvidesFor{"P", ModeAgent}, invoke, stream, false)
}

func (r *recordingHandler) keys() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	var out []string
	for _, c := range r.calls {
		out = append(out, c.Key)
	}
	return out
}

func TestResolveGraphTopology(t *testing.T) {
	withFake(t, map[string]any{
		"g": map[string]any{"root": "root", "edges": map[string]any{
			"root": []any{map[string]any{"key": "A"}}, "A": []any{map[string]any{"key": "B"}},
		}},
		"root": nodeCfg("r"), "A": nodeCfg("a"), "B": nodeCfg("b"),
		"no-root": map[string]any{"edges": map[string]any{}},
	})
	def := ResolveGraph(bg, "g", userCtx, GraphOptions{})
	if !def.Enabled || def.Root.Key != "root" || def.GetNode("A").Config.Instructions != "a" {
		t.Fatalf("def = %+v", def)
	}
	names := func(ns []*GraphNode) (out []string) {
		for _, n := range ns {
			out = append(out, n.Key)
		}
		return
	}
	if got := names(def.GetChildNodes("root")); !reflect.DeepEqual(got, []string{"A"}) {
		t.Errorf("children(root) = %v", got)
	}
	if got := names(def.GetChildNodes("A")); !reflect.DeepEqual(got, []string{"B"}) {
		t.Errorf("children(A) = %v", got)
	}
	if got := names(def.GetParentNodes("B")); !reflect.DeepEqual(got, []string{"A"}) {
		t.Errorf("parents(B) = %v", got)
	}
	if got := names(def.TerminalNodes()); !reflect.DeepEqual(got, []string{"B"}) {
		t.Errorf("terminal = %v", got)
	}
	if !def.GetNode("B").IsTerminal() || def.GetNode("root").IsTerminal() || def.GetNode("A").IsTerminal() {
		t.Error("IsTerminal wrong")
	}
	if e := def.EdgesFrom("root"); len(e) != 1 || e[0].TargetKey != "A" || e[0].Key != "root-A" {
		t.Errorf("edgesFrom = %+v", e)
	}
	var order []string
	visit := func(_ context.Context, n *GraphNode, _ map[string]any) (any, error) {
		order = append(order, n.Key)
		return n.Key, nil
	}
	if v, _ := def.Traverse(bg, visit); v != "root" || !reflect.DeepEqual(order, []string{"root", "A", "B"}) {
		t.Errorf("traverse order = %v (%v)", order, v)
	}
	order = nil
	if _, _ = def.ReverseTraverse(bg, visit); !reflect.DeepEqual(order, []string{"B", "A", "root"}) {
		t.Errorf("reverse order = %v", order)
	}
	if ResolveGraph(bg, "no-root", userCtx, GraphOptions{}).Enabled || ResolveGraph(bg, "missing", userCtx, GraphOptions{}).Enabled {
		t.Error("graph without root must be disabled")
	}
}

func TestReverseTraverseVisitsEveryChildBeforeItsParent(t *testing.T) {
	edges := func(m map[string][]string) map[string]any {
		out := map[string]any{}
		for k, ts := range m {
			var es []any
			for _, t := range ts {
				es = append(es, map[string]any{"key": t})
			}
			out[k] = es
		}
		return out
	}
	withFake(t, map[string]any{
		// Uneven depths: P has a terminal child and a deeper one.
		"uneven": map[string]any{"root": "root", "edges": edges(map[string][]string{"root": {"P"}, "P": {"T1", "Q"}, "Q": {"T2"}})},
		// Cycle: each node is visited exactly once.
		"cyclic": map[string]any{"root": "root", "edges": edges(map[string][]string{"root": {"A"}, "A": {"B"}, "B": {"A", "T"}})},
		"root":   nodeCfg("r"), "P": nodeCfg("p"), "Q": nodeCfg("q"), "T1": nodeCfg("t1"), "T2": nodeCfg("t2"),
		"A": nodeCfg("a"), "B": nodeCfg("b"), "T": nodeCfg("t"),
	})
	for key, want := range map[string][]string{
		"uneven": {"T1", "T2", "Q", "P", "root"},
		"cyclic": {"T", "B", "A", "root"},
	} {
		var order []string
		visit := func(_ context.Context, n *GraphNode, _ map[string]any) (any, error) {
			order = append(order, n.Key)
			return n.Key, nil
		}
		v, err := ResolveGraph(bg, key, userCtx, GraphOptions{}).ReverseTraverse(bg, visit)
		if err != nil || v != "root" || !reflect.DeepEqual(order, want) {
			t.Errorf("%s: reverse order = %v (%v, %v), want %v", key, order, v, err, want)
		}
	}
}

func TestDisabledNodeDisablesGraphAndLogs(t *testing.T) {
	var buf bytes.Buffer
	SetLogger(slog.New(slog.NewTextHandler(&buf, nil)))
	t.Cleanup(func() { SetLogger(slog.Default()) })
	withFake(t, twoNode(map[string]any{"leaf-node": map[string]any{"_ldMeta": map[string]any{"enabled": false}}}))
	if ResolveGraph(bg, "g", userCtx, GraphOptions{}).Enabled {
		t.Error("graph should be disabled")
	}
	if !strings.Contains(buf.String(), "level=ERROR") || !strings.Contains(buf.String(), "leaf-node") {
		t.Errorf("error not logged: %s", buf.String())
	}
}

func TestGraphInvoke(t *testing.T) {
	f := withFake(t, twoNode(nil))
	rh := &recordingHandler{}
	g := Graph("g", GraphOptions{Handlers: []*ProviderHandler{rh.handler()}})
	resp, err := g.Invoke(bg, "hi", userCtx)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Response != "leaf-node out" || resp.Usage.Total != 4 || !reflect.DeepEqual(rh.keys(), []string{"root-node", "leaf-node"}) {
		t.Errorf("resp=%+v calls=%v", resp, rh.keys())
	}
	if in := rh.calls[1].Req.UserInput; in != "[Original request]\nhi\n\n[Previous agent response]\nroot-node out" {
		t.Errorf("threaded input = %q", in)
	}
	nodes := f.events("$ld:ai:graph:node")
	if len(nodes) != 2 || nodes[0].Data["nodeKey"] != "root-node" || nodes[0].Data["index"] != 0.0 ||
		nodes[1].Data["nodeKey"] != "leaf-node" || nodes[1].Data["index"] != 1.0 ||
		nodes[0].Data["runId"] != nodes[1].Data["runId"] || nodes[0].Data["graphKey"] != "g" || nodes[0].Metric != 1 {
		t.Errorf("node events = %+v", nodes)
	}
	if _, has := nodes[0].Data["path"]; has || len(f.events("$ld:ai:graph:path")) != 0 {
		t.Error("path must not be tracked")
	}
	h := f.events("$ld:ai:graph:handoff_success")
	if len(h) != 1 || h[0].Data["sourceKey"] != "root-node" || h[0].Data["targetKey"] != "leaf-node" {
		t.Errorf("handoff = %+v", h)
	}
	for _, e := range []string{"$ld:ai:graph:duration:total", "$ld:ai:graph:invocation_success"} {
		if len(f.events(e)) != 1 {
			t.Errorf("missing %s", e)
		}
	}
	if gen := f.events("$ld:ai:generation:success"); len(gen) != 2 || gen[0].Data["graphKey"] != "g" {
		t.Errorf("per-node generation events = %+v", gen)
	}
}

func TestGraphInvokeErrors(t *testing.T) {
	f := withFake(t, twoNode(map[string]any{"off": map[string]any{"_ldMeta": map[string]any{"enabled": true}}}))
	if _, err := Graph("off", GraphOptions{Handlers: []*ProviderHandler{(&recordingHandler{}).handler()}}).Invoke(bg, "", userCtx); !errors.Is(err, ErrGraphDisabled) {
		t.Errorf("disabled graph err = %v", err)
	}
	if _, err := Graph("g", GraphOptions{}).Invoke(bg, "", userCtx); err == nil {
		t.Error("no handlers must error")
	}
	rh := &recordingHandler{failOn: "root-node"}
	_, err := Graph("g", GraphOptions{Handlers: []*ProviderHandler{rh.handler()}}).Invoke(bg, "", userCtx)
	if err == nil || !strings.Contains(err.Error(), "node failed") {
		t.Fatalf("err = %v", err)
	}
	if n := f.events("$ld:ai:graph:node"); len(n) != 1 || n[0].Data["nodeKey"] != "root-node" {
		t.Errorf("failing node not recorded: %+v", n)
	}
	if len(f.events("$ld:ai:graph:invocation_failure")) != 1 || len(f.events("$ld:ai:graph:path")) != 0 {
		t.Error("failure tracking wrong")
	}
}

func TestGraphCycleGuard(t *testing.T) {
	withFake(t, map[string]any{
		"loop": map[string]any{"root": "a", "edges": map[string]any{"a": []any{map[string]any{"key": "b"}}, "b": []any{map[string]any{"key": "a"}}}},
		"a":    nodeCfg("a"), "b": nodeCfg("b"),
	})
	rh := &recordingHandler{}
	resp, err := Graph("loop", GraphOptions{Handlers: []*ProviderHandler{rh.handler()}}).Invoke(bg, "", userCtx)
	if err != nil || resp.Response == "" || len(rh.keys()) > 3 {
		t.Errorf("resp=%+v err=%v calls=%v", resp, err, rh.keys())
	}
}

func TestGraphResolvesEachCallAndIsolation(t *testing.T) {
	f := withFake(t, twoNode(nil))
	rh := &recordingHandler{}
	g := Graph("g", GraphOptions{Handlers: []*ProviderHandler{rh.handler()}})
	if _, err := g.Invoke(bg, "", userCtx, WithVariables(map[string]any{"v": "first"})); err != nil {
		t.Fatal(err)
	}
	if _, err := g.Invoke(bg, "", userCtx); err != nil {
		t.Fatal(err)
	}
	if f.calls["root-node"] != 2 || f.calls["g"] != 2 {
		t.Errorf("flags not re-resolved per call: %v", f.calls)
	}
	if v := rh.calls[2].Req.Variables["v"]; v != nil {
		t.Errorf("variables leaked into second call: %v", v)
	}
}

// A reused Graph sees flag changes, including the kill switch, on the next call (TESTING.md §3.11).
func TestGraphSeesFlagChanges(t *testing.T) {
	f := withFake(t, map[string]any{"g": map[string]any{"root": "only"}, "only": nodeCfg("v1")})
	rh := &recordingHandler{}
	g := Graph("g", GraphOptions{Handlers: []*ProviderHandler{rh.handler()}})
	if _, err := g.Invoke(bg, "", userCtx); err != nil {
		t.Fatal(err)
	}
	f.variations["only"] = nodeCfg("v2")
	if _, err := g.Invoke(bg, "", userCtx); err != nil {
		t.Fatal(err)
	}
	if got := rh.calls[1].Req.Config.Instructions; got != "v2" {
		t.Errorf("second call instructions = %q, want v2", got)
	}
	f.variations["g"] = map[string]any{}
	if _, err := g.Invoke(bg, "", userCtx); !errors.Is(err, ErrGraphDisabled) {
		t.Errorf("disabled graph invoke err = %v", err)
	}
	for _, err := range g.Stream(bg, "", userCtx) {
		if !errors.Is(err, ErrGraphDisabled) {
			t.Errorf("disabled graph stream err = %v", err)
		}
	}
}

func TestGraphHistoryRootOnly(t *testing.T) {
	withFake(t, twoNode(nil))
	rh := &recordingHandler{}
	img := Message{Role: "user", Blocks: []ContentBlock{{Type: "image", Source: &ImageSource{Type: "base64", MediaType: "image/png", Data: "AAA"}}}}
	if _, err := Graph("g", GraphOptions{Handlers: []*ProviderHandler{rh.handler()}}).Invoke(bg, "q", userCtx, WithHistory([]Message{img})); err != nil {
		t.Fatal(err)
	}
	if h := rh.calls[0].Req.History; len(h) != 1 || h[0].Blocks[0].Source.Data != "AAA" {
		t.Errorf("root history = %+v", h)
	}
	if h := rh.calls[1].Req.History; h != nil {
		t.Errorf("leaf received history: %+v", h)
	}
}

func multiEdge() map[string]any {
	return map[string]any{
		"route": map[string]any{"root": "router", "edges": map[string]any{"router": []any{
			map[string]any{"key": "billing", "handoff": map[string]any{"description": "money questions"}},
			map[string]any{"key": "support"},
		}}},
		"router":  enabledConfig("P", "agent", judgedConfig(map[string]any{"instructions": "route it"})),
		"billing": nodeCfg("billing work"),
		"support": nodeCfg("support work"),
		"judge":   enabledConfig("P", "judge", nil),
	}
}

func TestGraphModelSelectedHandoff(t *testing.T) {
	f := withFake(t, multiEdge())
	// The router judge returns prose, so it is skipped; we only need the config it saw.
	invokeRH := &recordingHandler{pick: map[string]string{"router": "billing"}}
	resp, err := Graph("route", GraphOptions{Handlers: []*ProviderHandler{invokeRH.handler()}}).Invoke(bg, "q", userCtx)
	if err != nil || resp.Response != "billing out" {
		t.Fatalf("resp=%+v err=%v", resp, err)
	}
	h := f.events("$ld:ai:graph:handoff_success")
	if len(h) != 1 || h[0].Data["sourceKey"] != "router" || h[0].Data["targetKey"] != "billing" {
		t.Errorf("handoff = %+v", h)
	}
	routed := invokeRH.calls[0].Req
	if !strings.HasSuffix(routed.Config.Instructions, routingSuffix) || routed.Config.Tools["__handoff_billing"].Description != "Transfer control to billing. money questions" {
		t.Errorf("routing surface = %+v", routed.Config)
	}
	if len(f.events("$ld:ai:tool_call")) != 0 {
		t.Error("handoff tool emitted $ld:ai:tool_call")
	}
	judgeReq := invokeRH.calls[1] // the router's judge call
	if judgeReq.Key != "judge" || strings.Contains(judgeReq.Req.Variables["message_history"].(string), "__handoff_") {
		t.Errorf("judge saw routing tools: %+v", judgeReq)
	}

	// Stream routes byte-identically.
	streamRH := &recordingHandler{pick: map[string]string{"router": "billing"}, stream: true}
	for _, err := range Graph("route", GraphOptions{Handlers: []*ProviderHandler{streamRH.handler()}}).Stream(bg, "q", userCtx) {
		if err != nil {
			t.Fatal(err)
		}
	}
	if a, b := routed.Config, streamRH.calls[0].Req.Config; !reflect.DeepEqual(a, b) {
		t.Errorf("invoke and stream routed differently:\n%+v\n%+v", a, b)
	}
	ha, _ := routed.ToolHandlers["__handoff_billing"].(ToolFunc)(bg, nil)
	hb, _ := streamRH.calls[0].Req.ToolHandlers["__handoff_billing"].(ToolFunc)(bg, nil)
	if ha != hb {
		t.Errorf("handoff tool text differs: %v vs %v", ha, hb)
	}
}

// The routing text is pinned byte-for-byte by TESTING.md §3.15a and must match
// the TypeScript and Python SDKs (AIC-3601). The literals are spelled out here
// on purpose: building them from the production constants would not catch drift.
const (
	wantRoutingSuffix  = "\n\nComplete your task using your available tools first. Only once your own part is done, call exactly one transfer tool to route to the next agent. Do not answer the parts of the request that belong to the agent you route to."
	wantBillingHandoff = "Handoff to billing-team recorded. billing-team takes over when your turn ends. Finish your own part, then end your turn. Do not answer for billing-team."
	wantSupportHandoff = "Handoff to support recorded. support takes over when your turn ends. Finish your own part, then end your turn. Do not answer for support."
)

func TestGraphRoutingText(t *testing.T) {
	lookupTool := map[string]any{"lookup": map[string]any{"name": "lookup", "type": "function", "parameters": map[string]any{}}}
	for _, tc := range []struct {
		name  string
		tools map[string]any // the router's own tools; nil means none
	}{
		{"router with its own tool", lookupTool},
		{"router with no tools", nil},
	} {
		for _, mode := range []string{"invoke", "stream"} {
			t.Run(tc.name+"/"+mode, func(t *testing.T) {
				router := enabledConfig("P", "agent", map[string]any{"instructions": "route it"})
				if tc.tools != nil {
					router["tools"] = tc.tools
				}
				withFake(t, map[string]any{
					"route": map[string]any{"root": "router", "edges": map[string]any{"router": []any{
						map[string]any{"key": "billing-team"}, map[string]any{"key": "support"}}}},
					"router":       router,
					"billing-team": nodeCfg("billing work"),
					"support":      nodeCfg("support work"),
				})
				rh := &recordingHandler{pick: map[string]string{"router": "billing_team"}, stream: mode == "stream"}
				opts := GraphOptions{Handlers: []*ProviderHandler{rh.handler()},
					ToolHandlers: map[string]ToolHandler{"lookup": ToolFunc(func(context.Context, map[string]any) (any, error) { return "found", nil })}}
				g := Graph("route", opts)
				if mode == "stream" {
					for _, err := range g.Stream(bg, "q", userCtx) {
						if err != nil {
							t.Fatal(err)
						}
					}
				} else if _, err := g.Invoke(bg, "q", userCtx); err != nil {
					t.Fatal(err)
				}

				// The config and tool handlers the handler actually received.
				got := rh.calls[0].Req
				if got.Config.Instructions != "route it"+wantRoutingSuffix {
					t.Errorf("router instructions = %q", got.Config.Instructions)
				}
				for _, h := range []struct{ tool, want string }{
					{"__handoff_billing_team", wantBillingHandoff},
					{"__handoff_support", wantSupportHandoff},
				} {
					fn, ok := got.ToolHandlers[h.tool].(ToolFunc)
					if !ok {
						t.Fatalf("handler got no %s tool handler: %v", h.tool, got.ToolHandlers)
					}
					if res, err := fn(bg, nil); err != nil || res != h.want {
						t.Errorf("%s result = %q, %v; want %q", h.tool, res, err, h.want)
					}
				}
				// Real tools are left alone: present exactly when the node has them.
				_, hasLookup := got.Config.Tools["lookup"]
				_, hasLookupHandler := got.ToolHandlers["lookup"].(ToolFunc)
				if hasLookup != (tc.tools != nil) || (tc.tools != nil && !hasLookupHandler) {
					t.Errorf("lookup tool = %v, handler = %v, node tools = %v", hasLookup, hasLookupHandler, tc.tools != nil)
				}
				// Only the router routes; its target ran with its own instructions untouched.
				if keys := rh.keys(); !reflect.DeepEqual(keys, []string{"router", "billing-team"}) {
					t.Fatalf("nodes run = %v", keys)
				}
				if got := rh.calls[1].Req.Config.Instructions; got != "billing work" {
					t.Errorf("leaf instructions = %q", got)
				}
			})
		}
	}
}

func TestGraphHandoffFailure(t *testing.T) {
	f := withFake(t, multiEdge())
	h := createHandler(ProvidesFor{"P", ModeAgent}, func(ctx context.Context, req HandlerRequest) (HandlerResult, error) {
		if fn, ok := req.ToolHandlers["__handoff_support"].(ToolFunc); ok {
			_, _ = fn(ctx, nil)
		}
		return HandlerResult{}, errors.New("router crashed")
	}, nil, false)
	if _, err := Graph("route", GraphOptions{Handlers: []*ProviderHandler{h}}).Invoke(bg, "q", userCtx); err == nil {
		t.Fatal("expected error")
	}
	if hf := f.events("$ld:ai:graph:handoff_failure"); len(hf) != 1 || hf[0].Data["targetKey"] != "support" {
		t.Errorf("handoff_failure = %+v", hf)
	}
}

func TestGraphStream(t *testing.T) {
	f := withFake(t, twoNode(nil))
	exp := withSpans(t)
	rh := &recordingHandler{stream: true}
	var events []GraphStreamEvent
	for ev, err := range Graph("g", GraphOptions{Handlers: []*ProviderHandler{rh.handler()}}).Stream(bg, "q", userCtx) {
		if err != nil {
			t.Fatal(err)
		}
		events = append(events, ev)
	}
	var shape []string
	for _, e := range events {
		shape = append(shape, string(e.Type)+":"+e.NodeKey+e.SourceKey)
	}
	want := []string{"node_start:root-node", "chunk:root-node", "chunk:root-node", "node_done:root-node", "handoff:root-node",
		"node_start:leaf-node", "chunk:leaf-node", "chunk:leaf-node", "node_done:leaf-node", "done:"}
	if !reflect.DeepEqual(shape, want) {
		t.Fatalf("events = %v", shape)
	}
	last := events[len(events)-1]
	if last.Response != "leaf-node out" || last.Usage.Total != 4 || events[4].TargetKey != "leaf-node" || events[3].Response != "root-node out" {
		t.Errorf("done = %+v", last)
	}
	if len(f.events("$ld:ai:graph:invocation_success")) != 1 {
		t.Error("success not tracked")
	}
	// Every handler span descends from ld.ai.graph and shares its trace.
	var graphSpanID string
	spans := exp.GetSpans().Snapshots()
	for _, s := range spans {
		if s.Name() == GraphSpanName {
			graphSpanID = s.SpanContext().SpanID().String()
			if s.Status().Code != codes.Ok {
				t.Error("graph span not OK")
			}
			// Same route and usage attributes as the native graph runners.
			a := map[string]any{}
			for _, kv := range s.Attributes() {
				a[string(kv.Key)] = kv.Value.AsInterface()
			}
			in, _ := a["gen_ai.usage.input_tokens"].(int64)
			out, _ := a["gen_ai.usage.output_tokens"].(int64)
			if a["launchdarkly.graph.path"] != "root-node->leaf-node" || in+out != 4 {
				t.Errorf("graph span path/usage = %v, %d+%d", a["launchdarkly.graph.path"], in, out)
			}
		}
	}
	for _, s := range spans {
		if s.Name() != GraphSpanName && (s.Parent().SpanID().String() != graphSpanID || s.SpanContext().TraceID() != spans[len(spans)-1].SpanContext().TraceID()) {
			t.Errorf("%s not parented under the graph span", s.Name())
		}
	}
}

func TestGraphStreamAbandonAndErrors(t *testing.T) {
	f := withFake(t, twoNode(nil))
	exp := withSpans(t)
	rh := &recordingHandler{stream: true}
	g := Graph("g", GraphOptions{Handlers: []*ProviderHandler{rh.handler()}})
	for ev := range g.Stream(bg, "q", userCtx) {
		if ev.Type == "chunk" {
			break
		}
	}
	if len(f.events("$ld:ai:graph:invocation_success")) != 0 || len(f.events("$ld:ai:graph:node")) != 1 {
		t.Error("abandoned run tracked success or lost the entered node")
	}
	for _, s := range exp.GetSpans().Snapshots() {
		if s.Name() != GraphSpanName {
			continue
		}
		abandoned := false
		for _, a := range s.Attributes() {
			abandoned = abandoned || (a.Key == "launchdarkly.stream.abandoned" && a.Value.AsBool())
		}
		if !abandoned || s.Status().Code != codes.Unset {
			t.Errorf("graph span abandoned=%v status=%v", abandoned, s.Status().Code)
		}
	}
	// A disabled graph surfaces its error on the first receive.
	var gotErr error
	for _, err := range Graph("nope", GraphOptions{Handlers: []*ProviderHandler{rh.handler()}}).Stream(bg, "q", userCtx) {
		gotErr = err
	}
	if !errors.Is(gotErr, ErrGraphDisabled) {
		t.Errorf("err = %v", gotErr)
	}
}

func TestGraphJudge(t *testing.T) {
	withFake(t, twoNode(map[string]any{"gj": enabledConfig("P", "judge", nil)}))
	var reqs []HandlerRequest
	h := createHandler(ProvidesFor{"P", ModeAgent}, func(_ context.Context, req HandlerRequest) (HandlerResult, error) {
		if req.Variables["response_to_evaluate"] != nil {
			reqs = append(reqs, req)
			return HandlerResult{Output: `{"score":0.9,"reasoning":"fine"}`}, nil
		}
		return HandlerResult{Output: "final"}, nil
	}, nil, false)
	resp, err := Graph("g", GraphOptions{Handlers: []*ProviderHandler{h}, GraphJudge: "gj"}).Invoke(bg, "q", userCtx)
	if err != nil || resp.JudgeResults["gj"].Score != 0.9 {
		t.Fatalf("resp=%+v err=%v", resp, err)
	}
	if mh := reqs[0].Variables["message_history"].(string); strings.Contains(mh, "Tool calls") || !strings.HasPrefix(mh, "q\n\nfinal") {
		t.Errorf("graph judge history = %q", mh)
	}
}

// verdictHandler answers a judge call with a verdict and any other call like
// recordingHandler, streaming when the inner handler does.
func verdictHandler(inner *ProviderHandler, verdict string) *ProviderHandler {
	invoke := func(ctx context.Context, req HandlerRequest) (HandlerResult, error) {
		if req.Variables["response_to_evaluate"] != nil {
			return HandlerResult{Output: verdict}, nil
		}
		return inner.Invoke(ctx, req)
	}
	return createHandler(inner.ProvidesFor, invoke, inner.Stream, false)
}

// A node's own judge results are on its node_done; done keeps only the graph judge.
func TestGraphStreamNodeJudgeResults(t *testing.T) {
	leaf := enabledConfig("P", "agent", judgedConfig(map[string]any{"instructions": "leaf work"}))
	withFake(t, twoNode(map[string]any{"leaf-node": leaf, "judge": enabledConfig("P", "judge", nil)}))
	h := verdictHandler((&recordingHandler{stream: true}).handler(), `{"score":0.9,"reasoning":"good"}`)
	events, err := drainGraph(Graph("g", GraphOptions{Handlers: []*ProviderHandler{h}}).Stream(bg, "q", userCtx))
	if err != nil {
		t.Fatal(err)
	}
	var dones []GraphStreamEvent
	for _, e := range events {
		if e.Type == EventNodeDone {
			dones = append(dones, e)
		}
	}
	if len(dones) != 2 || dones[0].NodeKey != "root-node" || dones[1].NodeKey != "leaf-node" {
		t.Fatalf("node_done events = %+v", dones)
	}
	if dones[0].JudgeResults != nil {
		t.Errorf("root node_done judgeResults = %+v, want none", dones[0].JudgeResults)
	}
	if jr := dones[1].JudgeResults["judge"]; jr.Score != 0.9 || jr.Response != "good" {
		t.Errorf("leaf node_done judgeResults = %+v", dones[1].JudgeResults)
	}
	if done := events[len(events)-1]; done.Type != EventDone || done.JudgeResults != nil {
		t.Errorf("done = %+v, want no judgeResults", done)
	}

	// A routing node's judges are on its own node_done too.
	withFake(t, multiEdge())
	h = verdictHandler((&recordingHandler{stream: true, pick: map[string]string{"router": "billing"}}).handler(), `{"score":1,"reasoning":"routed"}`)
	events, err = drainGraph(Graph("route", GraphOptions{Handlers: []*ProviderHandler{h}}).Stream(bg, "q", userCtx))
	if err != nil {
		t.Fatal(err)
	}
	dones = nil
	for _, e := range events {
		if e.Type == EventNodeDone {
			dones = append(dones, e)
		}
	}
	if len(dones) != 2 || dones[0].NodeKey != "router" || dones[1].NodeKey != "billing" {
		t.Fatalf("node_done events = %+v", dones)
	}
	if jr := dones[0].JudgeResults["judge"]; jr.Score != 1 || jr.Response != "routed" {
		t.Errorf("router node_done judgeResults = %+v", dones[0].JudgeResults)
	}
	if dones[1].JudgeResults != nil {
		t.Errorf("billing node_done judgeResults = %+v, want none", dones[1].JudgeResults)
	}
}

func TestGraphFailedResolutionIsNotCached(t *testing.T) {
	f := withFake(t, twoNode(nil))
	leaf := f.variations["leaf-node"]
	delete(f.variations, "leaf-node") // a node that fails to resolve, e.g. a transient fetch error
	rh := &recordingHandler{}
	g := Graph("g", GraphOptions{Handlers: []*ProviderHandler{rh.handler()}})
	if _, err := g.Invoke(bg, "", userCtx); !errors.Is(err, ErrGraphDisabled) {
		t.Fatalf("first call err = %v, want ErrGraphDisabled", err)
	}
	f.variations["leaf-node"] = leaf
	resp, err := g.Invoke(bg, "", userCtx)
	if err != nil || resp.Response != "leaf-node out" {
		t.Errorf("after recovery: resp=%+v err=%v (disabled result was cached)", resp, err)
	}
}

func TestGraphNodeTrackData(t *testing.T) {
	v := 4
	n := &GraphNode{Key: "leaf", Config: AiConfig{Model: Model{Name: "m"}}, Meta: VariationMeta{VariationKey: "var", ModelKey: "mk", ModelVersion: &v}}
	n.Config.Provider.Name = "P"
	td := n.TrackData("g", "run-1")
	want := TrackData{RunID: "run-1", ConfigKey: "leaf", VariationKey: "var", Version: 1, ModelName: "m", ProviderName: "P", GraphKey: "g", ModelKey: "mk", ModelVersion: &v}
	if !reflect.DeepEqual(td, want) {
		t.Errorf("TrackData = %+v, want %+v", td, want)
	}
	n.Meta.Version = 7
	if n.TrackData("g", "r").Version != 7 {
		t.Error("an explicit version must be kept")
	}
}

func graphSpan(t *testing.T, exp *tracetest.InMemoryExporter) tracetest.SpanStub {
	t.Helper()
	for _, s := range exp.GetSpans() {
		if s.Name == GraphSpanName {
			return s
		}
	}
	t.Fatal("no graph span")
	return tracetest.SpanStub{}
}

func spanAbandoned(s tracetest.SpanStub) bool {
	for _, a := range s.Attributes {
		if a.Key == "launchdarkly.stream.abandoned" && a.Value.AsBool() {
			return true
		}
	}
	return false
}

// The graph span ends on every path with the right status, and the run's
// aggregate tokens are tracked on success.
func TestGraphSpanStatusOnEveryPath(t *testing.T) {
	run := func(fn func()) (r any) {
		defer func() { r = recover() }()
		fn()
		return nil
	}
	t.Run("success", func(t *testing.T) {
		f := withFake(t, twoNode(nil))
		exp := withSpans(t)
		if _, err := Graph("g", GraphOptions{Handlers: []*ProviderHandler{(&recordingHandler{}).handler()}}).Invoke(bg, "q", userCtx); err != nil {
			t.Fatal(err)
		}
		if tt := f.events("$ld:ai:graph:total_tokens"); len(tt) != 1 || tt[0].Metric != 4 || tt[0].Data["graphKey"] != "g" {
			t.Errorf("total_tokens = %+v", tt)
		}
		if s := graphSpan(t, exp); s.Status.Code != codes.Ok || s.EndTime.IsZero() || spanAbandoned(s) {
			t.Errorf("graph span status=%v ended=%v", s.Status.Code, !s.EndTime.IsZero())
		}
	})
	t.Run("node error", func(t *testing.T) {
		f := withFake(t, twoNode(nil))
		exp := withSpans(t)
		_, err := Graph("g", GraphOptions{Handlers: []*ProviderHandler{(&recordingHandler{failOn: "leaf-node"}).handler()}}).Invoke(bg, "q", userCtx)
		if err == nil {
			t.Fatal("want error")
		}
		s := graphSpan(t, exp)
		if s.Status.Code != codes.Error || s.EndTime.IsZero() || len(s.Events) == 0 || spanAbandoned(s) {
			t.Errorf("graph span status=%v events=%d", s.Status.Code, len(s.Events))
		}
		if len(f.events("$ld:ai:graph:total_tokens")) != 0 || len(f.events("$ld:ai:graph:duration:total")) != 1 {
			t.Error("a failed run tracked total tokens or lost its duration")
		}
	})
	t.Run("node panic", func(t *testing.T) {
		f := withFake(t, twoNode(nil))
		exp := withSpans(t)
		h := createHandler(ProvidesFor{"P", ModeAgent}, func(context.Context, HandlerRequest) (HandlerResult, error) { panic("node bug") }, nil, false)
		if r := run(func() { _, _ = Graph("g", GraphOptions{Handlers: []*ProviderHandler{h}}).Invoke(bg, "q", userCtx) }); r == nil {
			t.Fatal("the panic was swallowed")
		}
		if s := graphSpan(t, exp); s.Status.Code != codes.Error || s.EndTime.IsZero() || spanAbandoned(s) {
			t.Errorf("graph span status=%v abandoned=%v", s.Status.Code, spanAbandoned(s))
		}
		if len(f.events("$ld:ai:graph:invocation_failure")) != 1 || len(f.events("$ld:ai:generation:error")) != 1 {
			t.Error("a panicking node was not tracked as a failure")
		}
	})
	t.Run("consumer panic", func(t *testing.T) {
		f := withFake(t, twoNode(nil))
		exp := withSpans(t)
		g := Graph("g", GraphOptions{Handlers: []*ProviderHandler{(&recordingHandler{stream: true}).handler()}})
		if r := run(func() {
			for ev := range g.Stream(bg, "q", userCtx) {
				if ev.Type == EventChunk {
					panic("consumer bug")
				}
			}
		}); r != "consumer bug" {
			t.Fatalf("panic value = %v, want the consumer's own", r)
		}
		if s := graphSpan(t, exp); s.Status.Code != codes.Unset || s.EndTime.IsZero() || !spanAbandoned(s) {
			t.Errorf("graph span status=%v abandoned=%v", s.Status.Code, spanAbandoned(s))
		}
		if len(f.events("$ld:ai:graph:invocation_failure")) != 0 || len(f.events("$ld:ai:generation:error")) != 0 {
			t.Error("a consumer panic was tracked as a failure")
		}
	})
}

// The graph span's path names every node a run visits, in order: here a
// model-chosen handoff (root picks b) followed by a fixed single-edge hop.
func TestGraphSpanPathAcrossHops(t *testing.T) {
	withFake(t, map[string]any{
		"g": map[string]any{"root": "root-node", "edges": map[string]any{
			"root-node": []any{map[string]any{"key": "a"}, map[string]any{"key": "b"}},
			"b":         []any{map[string]any{"key": "leaf-node"}},
		}},
		"root-node": nodeCfg("root work"), "a": nodeCfg("a work"), "b": nodeCfg("b work"), "leaf-node": nodeCfg("leaf work"),
	})
	exp := withSpans(t)
	rh := &recordingHandler{pick: map[string]string{"root-node": "b"}}
	if _, err := Graph("g", GraphOptions{Handlers: []*ProviderHandler{rh.handler()}}).Invoke(bg, "q", userCtx); err != nil {
		t.Fatal(err)
	}
	path := "<missing>"
	for _, s := range exp.GetSpans().Snapshots() {
		for _, kv := range s.Attributes() {
			if s.Name() == GraphSpanName && kv.Key == "launchdarkly.graph.path" {
				path = kv.Value.AsString()
			}
		}
	}
	if path != "root-node->b->leaf-node" {
		t.Errorf("path = %q, calls = %d", path, len(rh.calls))
	}
}

// Graph node routing normalizes a handler's declared mode like the config
// routing rule: a "completion" handler serves a messages node, and an agent
// handler for the same provider is not picked instead.
func TestGraphRoutingNormalizesHandlerMode(t *testing.T) {
	leaf := enabledConfig("P", "completion", nil)
	withFake(t, map[string]any{
		"g":         map[string]any{"root": "root-node", "edges": map[string]any{"root-node": []any{map[string]any{"key": "leaf-node"}}}},
		"root-node": nodeCfg("root work"), "leaf-node": leaf,
	})
	var served []string
	mk := func(mode Mode, tag string) *ProviderHandler {
		return createHandler(ProvidesFor{"P", mode}, func(_ context.Context, req HandlerRequest) (HandlerResult, error) {
			served = append(served, tag+":"+req.Variables[VarLD].(TrackData).ConfigKey)
			return HandlerResult{Output: "out"}, nil
		}, nil, false)
	}
	hs := []*ProviderHandler{mk(ModeAgent, "agent"), mk(Mode("completion"), "legacy")}
	if _, err := Graph("g", GraphOptions{Handlers: hs}).Invoke(bg, "q", userCtx); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(served, []string{"agent:root-node", "legacy:leaf-node"}) {
		t.Errorf("served = %v", served)
	}
}
