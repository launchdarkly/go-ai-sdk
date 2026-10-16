package openaiagents

import (
	"context"
	"errors"
	"maps"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/openai/openai-go/v3"
	"go.opentelemetry.io/otel/codes"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

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

// twoNode is triage → billing; triage has `lookup`, billing has `refund`.
func twoNode(extra map[string]map[string]any) *fakeLD {
	v := map[string]map[string]any{
		"g":       graphFlag("triage", map[string][]any{"triage": {map[string]any{"key": "billing", "handoff": map[string]any{"description": "billing questions"}}}}),
		"triage":  node("Route {{tone}}", "triage-model", "lookup"),
		"billing": node("Bill", "billing-model", "refund"),
	}
	for k, n := range extra {
		v[k] = n
	}
	return &fakeLD{variations: v}
}

var user = ldcontext.New("u-1")

func resolve(t *testing.T, ld *fakeLD, key string) *aiserver.GraphDefinition {
	t.Helper()
	def, err := aiserver.ResolveGraph(bg, key, user, aiserver.GraphOptions{})
	if err != nil {
		t.Fatal(err)
	}
	return def
}

func handlersFor(calls *[]string) map[string]aiserver.ToolHandler {
	mk := func(name string) aiserver.ToolFunc {
		return tool(func(args map[string]any) (any, error) { *calls = append(*calls, name); return name + " done", nil })
	}
	return map[string]aiserver.ToolHandler{"lookup": mk("lookup"), "refund": mk("refund")}
}

// Topology, handoffs, instructions and tool forwarding, inspected on
// the built agents.
func TestToOpenAIAgentsTopology(t *testing.T) {
	ld := twoNode(nil)
	installClient(t, ld)
	var calls []string
	g := ToOpenAIAgents(resolve(t, ld, "g"), &NativeGraphOptions{ToolHandlers: handlersFor(&calls)})
	agents, err := g.buildAgents(bg, map[string]any{"tone": "calm"}, "run", false)
	if err != nil {
		t.Fatal(err)
	}
	if len(agents) != 2 {
		t.Fatalf("want one agent per node, got %d", len(agents))
	}
	root, leaf := agents["triage"], agents["billing"]
	if len(root.handoffs) != 1 || root.handoffs[0].target != leaf || root.handoffs[0].tool != "transfer_to_billing" ||
		!strings.Contains(root.handoffs[0].description, "billing questions") {
		t.Errorf("root handoffs = %+v", root.handoffs)
	}
	if len(leaf.handoffs) != 0 {
		t.Errorf("terminal node has handoffs: %+v", leaf.handoffs)
	}
	if root.instructions != "Route calm" || leaf.instructions != "Bill" {
		t.Errorf("instructions = %q / %q", root.instructions, leaf.instructions)
	}
	if !reflect.DeepEqual(slices.Sorted(maps.Keys(root.tools)), []string{"lookup"}) || !reflect.DeepEqual(slices.Sorted(maps.Keys(leaf.tools)), []string{"refund"}) {
		t.Errorf("tools = %v / %v", root.tools, leaf.tools)
	}
}

// Execution: hooks → LD events, span.
func TestToOpenAIAgentsRun(t *testing.T) {
	ld := twoNode(nil)
	installClient(t, ld)
	exp := tracer(t)
	f, c := fakeClient(t,
		response("r1", callOutput("transfer_to_billing", "{}"), 3, 1, 1),
		response("r2", callOutput("refund", `{"id":1}`), 4, 0, 2),
		response("r3", textOutput("refunded"), 5, 0, 3))
	var calls []string
	def := resolve(t, ld, "g")
	res, err := ToOpenAIAgents(def, &NativeGraphOptions{ToolHandlers: handlersFor(&calls), Context: user, Handler: Options{Client: c}}).Invoke(bg, "hello", aiserver.WithVariables(map[string]any{"tone": "calm"}))
	if err != nil {
		t.Fatal(err)
	}
	if res.Response != "refunded" || res.Usage != (aiserver.TokenUsage{Input: 12, Output: 6, Total: 18}) || !slices.Equal(calls, []string{"refund"}) {
		t.Errorf("res = %+v calls %v", res, calls)
	}

	r0, r1, r2 := f.request(0), f.request(1), f.request(2)
	if r0["model"] != "triage-model" || r0["instructions"] != "Route calm" || !reflect.DeepEqual(inputTurns(r0), []turn{{"user", "hello"}}) ||
		!reflect.DeepEqual(toolNames(r0), []string{"lookup", "transfer_to_billing"}) {
		t.Errorf("root request = %v", r0)
	}
	if r1["model"] != "billing-model" || r1["instructions"] != "Bill" || r1["previous_response_id"] != "r1" ||
		!reflect.DeepEqual(inputTurns(r1), []turn{{"function_call_output", `{"assistant":"billing"}`}}) ||
		!reflect.DeepEqual(toolNames(r1), []string{"refund"}) {
		t.Errorf("leaf request = %v", r1)
	}
	if !reflect.DeepEqual(inputTurns(r2), []turn{{"function_call_output", "refund done"}}) {
		t.Errorf("tool result request = %v", r2)
	}

	// Every node run gets per-node events; the handoff is tracked once its
	// target has answered.
	want := []string{"$ld:ai:graph:node", "$ld:ai:duration:total", "$ld:ai:generation:success", "$ld:ai:tokens:total", "$ld:ai:tokens:input", "$ld:ai:tokens:output",
		"$ld:ai:graph:node", "$ld:ai:tool_call", "$ld:ai:duration:total", "$ld:ai:generation:success", "$ld:ai:tokens:total", "$ld:ai:tokens:input", "$ld:ai:tokens:output",
		"$ld:ai:graph:handoff_success", "$ld:ai:graph:duration:total", "$ld:ai:graph:total_tokens", "$ld:ai:graph:invocation_success"}
	if got := ld.aiEvents(); !reflect.DeepEqual(sortTokens(got), sortTokens(want)) {
		t.Errorf("events = %v", got)
	}
	nodes := ld.events("$ld:ai:graph:node")
	runID := nodes[0].data["runId"]
	if nodes[0].data["nodeKey"] != "triage" || nodes[0].data["index"] != 0.0 || nodes[1].data["nodeKey"] != "billing" ||
		nodes[1].data["index"] != 1.0 || nodes[0].value != 1 || nodes[1].data["graphKey"] != "g" || nodes[1].data["runId"] != runID {
		t.Errorf("node events = %+v", nodes)
	}
	if h := ld.events("$ld:ai:graph:handoff_success")[0]; h.data["sourceKey"] != "triage" || h.data["targetKey"] != "billing" || h.data["configKey"] != "g" || h.data["variationKey"] != "gv" || h.data["runId"] != runID {
		t.Errorf("handoff = %+v", h.data)
	}
	if gen := ld.events("$ld:ai:generation:success"); len(gen) != 2 || gen[0].data["configKey"] != "triage" || gen[1].data["configKey"] != "billing" ||
		gen[1].data["graphKey"] != "g" || gen[1].data["variationKey"] != "v-billing-model" || gen[1].data["modelName"] != "billing-model" || gen[1].value != 1 {
		t.Errorf("generation = %+v", gen)
	}
	if tok := ld.events("$ld:ai:tokens:total"); len(tok) != 2 || tok[0].value != 4 || tok[1].value != 14 {
		t.Errorf("node tokens = %+v", tok)
	}
	if nodes[0].data["configKey"] != "g" {
		t.Errorf("graph:node carries node track data: %v", nodes[0].data)
	}
	if tc := ld.events("$ld:ai:tool_call")[0]; tc.data["toolKey"] != "refund" || tc.data["configKey"] != "billing" {
		t.Errorf("tool_call = %+v", tc.data)
	}
	if tt := ld.events("$ld:ai:graph:total_tokens")[0]; tt.value != 18 || tt.data["runId"] != runID {
		t.Errorf("total_tokens = %+v", tt)
	}
	if len(ld.events("$ld:ai:graph:path")) != 0 {
		t.Error("$ld:ai:graph:path tracked")
	}

	spans := spansByName(exp)[aiserver.GraphSpanName]
	if len(spans) != 1 {
		t.Fatalf("graph spans = %v", spansByName(exp))
	}
	a := attrs(spans[0])
	for k := range a {
		if strings.HasPrefix(k, "context.contextKeys.") {
			t.Errorf("ld.ai.graph carries context identity %s; it belongs on handler root spans only", k)
		}
	}
	if a["ld.ai.graph.key"] != "g" || a["launchdarkly.graph.path"] != "triage->billing" || a["gen_ai.usage.total_tokens"] != "18" ||
		spans[0].Status().Code != codes.Ok {
		t.Errorf("graph span attrs = %v status %v", a, spans[0].Status())
	}
	// One openai.agent.run span per node run under the graph span, with the
	// node's association and the context identity; chat / execute_tool
	// children carry neither.
	byName := spansByName(exp)
	nodeSpans := byName[SpanName]
	if len(nodeSpans) != 2 {
		t.Fatalf("node spans = %d", len(nodeSpans))
	}
	for i, key := range []string{"triage", "billing"} {
		ns := nodeSpans[i]
		na := attrs(ns)
		if ns.Parent().SpanID() != spans[0].SpanContext().SpanID() || na["launchdarkly.config.key"] != key || na["context.contextKeys.user"] != "u-1" ||
			na["gen_ai.operation.name"] != "invoke_agent" || ns.Status().Code != codes.Ok || ns.EndTime().IsZero() {
			t.Errorf("%s span = %v", key, na)
		}
	}
	if na := attrs(nodeSpans[1]); na["gen_ai.usage.total_tokens"] != "14" || na["gen_ai.completion.0.content"] != "refunded" {
		t.Errorf("leaf span = %v", na)
	}
	if len(byName["chat billing-model"]) != 2 || len(byName["execute_tool refund"]) != 1 {
		t.Errorf("child spans = %v", byName)
	}
	for name, ss := range byName {
		if name == SpanName || name == aiserver.GraphSpanName {
			continue
		}
		for k := range attrs(ss[0]) {
			if strings.HasPrefix(k, "context.") || k == "launchdarkly.config.key" {
				t.Errorf("child %s carries %s", name, k)
			}
		}
	}
}

// sortTokens orders each run of $ld:ai:tokens:* events, which have no fixed order.
func sortTokens(events []string) []string {
	out := slices.Clone(events)
	for i := 0; i < len(out); {
		j := i
		for j < len(out) && strings.HasPrefix(out[j], "$ld:ai:tokens:") {
			j++
		}
		slices.Sort(out[i:j])
		i = max(j, i+1)
	}
	return out
}

// A handoff whose target fails is a handoff_failure, never a success.
func TestToOpenAIAgentsHandoffFailure(t *testing.T) {
	ld := twoNode(nil)
	installClient(t, ld)
	exp := tracer(t)
	_, c := fakeClient(t, toolResp("r1", "transfer_to_billing", "{}"), "STATUS400")
	if _, err := ToOpenAIAgents(resolve(t, ld, "g"), &NativeGraphOptions{Context: user, Handler: Options{Client: c}}).Invoke(bg, "q"); err == nil {
		t.Fatal("want error")
	}
	if hs := ld.events("$ld:ai:graph:handoff_success"); len(hs) != 0 {
		t.Errorf("handoff_success for a failed target: %+v", hs)
	}
	hf := ld.events("$ld:ai:graph:handoff_failure")
	if len(hf) != 1 || hf[0].data["sourceKey"] != "triage" || hf[0].data["targetKey"] != "billing" || hf[0].data["configKey"] != "g" {
		t.Errorf("handoff_failure = %+v", hf)
	}
	if ge := ld.events("$ld:ai:generation:error"); len(ge) != 1 || ge[0].data["configKey"] != "billing" {
		t.Errorf("generation:error = %+v", ge)
	}
	if ns := spansByName(exp)[SpanName]; len(ns) != 2 || ns[1].Status().Code != codes.Error || ns[1].EndTime().IsZero() {
		t.Errorf("failed node span not ended ERROR")
	}
}

// a node re-entered through a handoff is recorded once.
func TestToOpenAIAgentsNodeOnce(t *testing.T) {
	ld := twoNode(map[string]map[string]any{"g": graphFlag("triage", map[string][]any{
		"triage": {map[string]any{"key": "billing"}}, "billing": {map[string]any{"key": "triage"}},
	})})
	installClient(t, ld)
	exp := tracer(t)
	_, c := fakeClient(t,
		toolResp("r1", "transfer_to_billing", "{}"), toolResp("r2", "transfer_to_triage", "{}"), textResp("back", 1, 1))
	if _, err := ToOpenAIAgents(resolve(t, ld, "g"), &NativeGraphOptions{Context: user, Handler: Options{Client: c}}).Invoke(bg, "q"); err != nil {
		t.Fatal(err)
	}
	var keys []any
	for _, n := range ld.events("$ld:ai:graph:node") {
		keys = append(keys, n.data["nodeKey"])
	}
	if !reflect.DeepEqual(keys, []any{"triage", "billing"}) || len(ld.events("$ld:ai:graph:handoff_success")) != 2 {
		t.Errorf("node keys = %v handoffs %d", keys, len(ld.events("$ld:ai:graph:handoff_success")))
	}
	if p := attrs(spansByName(exp)[aiserver.GraphSpanName][0])["launchdarkly.graph.path"]; p != "triage->billing" {
		t.Errorf("path = %q", p)
	}
}

// error path, no-context path, and definition errors.
func TestToOpenAIAgentsErrors(t *testing.T) {
	t.Run("run error tracks failure, re-throws, ends span", func(t *testing.T) {
		ld := twoNode(nil)
		installClient(t, ld)
		exp := tracer(t)
		_, c := fakeClient(t, "STATUS400")
		_, err := ToOpenAIAgents(resolve(t, ld, "g"), &NativeGraphOptions{Context: user, Handler: Options{Client: c}}).Invoke(bg, "q")
		var apiErr *openai.Error
		if !errors.As(err, &apiErr) {
			t.Fatalf("err = %v", err)
		}
		if got := ld.aiEvents(); !reflect.DeepEqual(got, []string{"$ld:ai:graph:node", "$ld:ai:duration:total", "$ld:ai:generation:error", "$ld:ai:graph:duration:total", "$ld:ai:graph:invocation_failure"}) {
			t.Errorf("events = %v", got)
		}
		s := spansByName(exp)[aiserver.GraphSpanName]
		if len(s) != 1 || s[0].Status().Code != codes.Error {
			t.Errorf("graph span = %v", s)
		}
	})

	t.Run("no context: nothing tracked", func(t *testing.T) {
		ld := twoNode(nil)
		installClient(t, ld)
		_, c := fakeClient(t, toolResp("r1", "transfer_to_billing", "{}"), textResp("ok", 1, 1))
		var calls []string
		if _, err := ToOpenAIAgents(resolve(t, ld, "g"), &NativeGraphOptions{ToolHandlers: handlersFor(&calls), Handler: Options{Client: c}}).Invoke(bg, "q"); err != nil {
			t.Fatal(err)
		}
		if got := ld.aiEvents(); len(got) != 0 {
			t.Errorf("tracked without a context: %v", got)
		}
	})

	t.Run("disabled, rootless and nil definitions", func(t *testing.T) {
		ld := &fakeLD{variations: map[string]map[string]any{}}
		installClient(t, ld)
		_, err := aiserver.ResolveGraph(bg, "off-graph", user, aiserver.GraphOptions{})
		if err == nil || !strings.Contains(err.Error(), "off-graph") {
			t.Fatalf("off-graph err = %v", err)
		}
		_, err = ToOpenAIAgents(&aiserver.GraphDefinition{Key: "off-graph"}, nil).Invoke(bg, "q")
		if !errors.Is(err, aiserver.ErrGraphDisabled) || !strings.Contains(err.Error(), "off-graph") {
			t.Errorf("disabled err = %v", err)
		}
		_, err = ToOpenAIAgents(&aiserver.GraphDefinition{Key: "rootless", Enabled: true}, nil).Invoke(bg, "q")
		if !errors.Is(err, ErrNoRoot) || !strings.Contains(err.Error(), "rootless") {
			t.Errorf("rootless err = %v", err)
		}
		if _, err = ToOpenAIAgents(nil, nil).Invoke(bg, "q"); !errors.Is(err, ErrNilGraph) {
			t.Errorf("nil err = %v", err)
		}
	})
}

// history (root only, multimodal) and NativeTool wiring.
func TestToOpenAIAgentsHistoryAndNativeTool(t *testing.T) {
	ld := twoNode(map[string]map[string]any{"billing": node("Bill", "billing-model", "search")})
	installClient(t, ld)
	f, c := fakeClient(t, toolResp("r1", "transfer_to_billing", "{}"), textResp("ok", 1, 1))
	history := []aiserver.Message{{Role: "user", Blocks: []aiserver.ContentBlock{{Type: "image", Source: &aiserver.ImageSource{Type: "base64", MediaType: "image/png", Data: "AAA"}}}}}
	_, err := ToOpenAIAgents(resolve(t, ld, "g"), &NativeGraphOptions{Handler: Options{Client: c},
		ToolHandlers: map[string]aiserver.ToolHandler{"search": &aiserver.NativeTool{ToolName: "web_search"}}}).Invoke(bg, "what is it?", aiserver.WithHistory(history))
	if err != nil {
		t.Fatal(err)
	}
	want := []turn{{"user", `[{"detail":"auto","image_url":"data:image/png;base64,AAA","type":"input_image"}]`}, {"user", "what is it?"}}
	if got := inputTurns(f.request(0)); !reflect.DeepEqual(got, want) {
		t.Errorf("root input = %v", got)
	}
	if strings.Contains(toJSON(f.request(1)["input"]), "image") {
		t.Error("handoff target received the original history")
	}
	if got := toolNames(f.request(1)); !reflect.DeepEqual(got, []string{"hosted:web_search"}) {
		t.Errorf("leaf tools = %v (NativeTool must be hosted, not a function tool)", got)
	}
}

// OpenAIGraph pre-wires exactly this package's handler,
// forwards the key and options, and ignores caller handlers.
func TestOpenAIGraph(t *testing.T) {
	ld := &fakeLD{variations: map[string]map[string]any{
		"g":    graphFlag("solo", nil),
		"solo": node("Solo", "test-model", "lookup"),
	}}
	installClient(t, ld)
	f, c := fakeClient(t,
		sse(nil, toolResp("r1", "lookup", "{}")),
		sse([]string{"fine"}, textResp("fine", 1, 1)))
	bogusCalled := false
	bogus := &aiserver.ProviderHandler{ProvidesFor: aiserver.ProvidesFor{Provider: "OpenAI", Mode: aiserver.ModeAgent},
		Invoke: func(context.Context, aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
			bogusCalled = true
			return aiserver.HandlerResult{}, nil
		}}
	var calls []string
	res, err := OpenAIGraph("g", aiserver.GraphOptions{Handlers: []*aiserver.ProviderHandler{bogus, bogus}, ToolHandlers: handlersFor(&calls)},
		Options{Client: c}).Invoke(bg, "q", user)
	if err != nil {
		t.Fatal(err)
	}
	if res.Response != "fine" || bogusCalled || f.count() != 2 || !slices.Equal(calls, []string{"lookup"}) || ld.keys[0] != "g" {
		t.Errorf("res=%+v bogus=%v requests=%d calls=%v keys=%v", res, bogusCalled, f.count(), calls, ld.keys)
	}
	if got := ld.usageHelpers(); !reflect.DeepEqual(got, []string{"openai-agents.openaiGraph"}) {
		t.Errorf("usage helpers = %v", got)
	}
}

// toOpenAIAgents reports only itself; running the native graph emits
// no client.graph.* or factory helper.
func TestToOpenAIAgentsSdkUsage(t *testing.T) {
	ld := twoNode(nil)
	installClient(t, ld)
	def := resolve(t, ld, "g")
	ld.resetTracks() // drop client.resolveGraph from the resolve above
	_, c := fakeClient(t, toolResp("r1", "transfer_to_billing", "{}"), textResp("ok", 1, 1))
	if _, err := ToOpenAIAgents(def, &NativeGraphOptions{Context: user, Handler: Options{Client: c}}).Invoke(bg, "q"); err != nil {
		t.Fatal(err)
	}
	if got := ld.usageHelpers(); !reflect.DeepEqual(got, []string{"openai-agents.toOpenAIAgents"}) {
		t.Errorf("usage helpers = %v", got)
	}
}

// A panicking tool in a native run re-panics, and the node run's spans and
// the graph span are ended ERROR (an unended span is never exported).
func TestToOpenAIAgentsToolPanicEndsSpans(t *testing.T) {
	ld := twoNode(nil)
	installClient(t, ld)
	exp := tracer(t)
	_, c := fakeClient(t, response("r1", callOutput("lookup", "{}"), 1, 0, 1))
	tools := map[string]aiserver.ToolHandler{"lookup": tool(func(map[string]any) (any, error) { panic("tool blew up") })}
	func() {
		defer func() {
			if p := recover(); p != "tool blew up" {
				t.Errorf("panic = %v", p)
			}
		}()
		_, _ = ToOpenAIAgents(resolve(t, ld, "g"), &NativeGraphOptions{ToolHandlers: tools, Context: user, Handler: Options{Client: c}}).Invoke(bg, "q")
	}()
	byName := spansByName(exp)
	for _, name := range []string{aiserver.GraphSpanName, SpanName, "execute_tool lookup"} {
		if s := byName[name]; len(s) != 1 || s[0].Status().Code != codes.Error {
			t.Errorf("%s not ended ERROR: %v", name, s)
		}
	}
}
