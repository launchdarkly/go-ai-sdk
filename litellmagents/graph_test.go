package litellmagents

import (
	"context"
	"errors"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/openai/openai-go/v3"
	"go.opentelemetry.io/otel/codes"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/kitbase"
)

func node(instr, provider, model string, params map[string]any, tools ...string) map[string]any {
	ts := map[string]any{}
	for _, t := range tools {
		ts[t] = map[string]any{"name": t, "type": "function", "description": t + " tool", "parameters": map[string]any{"type": "object"}}
	}
	m := map[string]any{"name": model}
	if params != nil {
		m["parameters"] = params
	}
	return map[string]any{
		"model": m, "provider": map[string]any{"name": provider},
		"instructions": instr, "tools": ts,
		"_ldMeta": map[string]any{"enabled": true, "variationKey": "v-" + model, "mode": "agent"},
	}
}

func graphFlag(root string, edges map[string][]any) map[string]any {
	return map[string]any{"root": root, "edges": edges, "_ldMeta": map[string]any{"enabled": true, "variationKey": "gv"}}
}

// twoNode is triage → billing on different upstreams; triage has `lookup`,
// billing has `refund`.
func twoNode(extra map[string]map[string]any) *fakeLD {
	v := map[string]map[string]any{
		"g":       graphFlag("triage", map[string][]any{"triage": {map[string]any{"key": "billing", "handoff": map[string]any{"description": "billing questions"}}}}),
		"triage":  node("Route {{tone}}", "Anthropic", "anthropic/claude-triage", nil, "lookup"),
		"billing": node("Bill", "Gemini", "gemini/billing", nil, "refund"),
	}
	for k, n := range extra {
		v[k] = n
	}
	return &fakeLD{variations: v}
}

var user = ldcontext.New("u-1")

func resolve(t *testing.T, key string) *aiserver.GraphDefinition {
	t.Helper()
	def, err := aiserver.ResolveGraph(bg, key, user, aiserver.GraphOptions{})
	if err != nil {
		t.Fatal(err)
	}
	return def
}

func adapter(t *testing.T, def *aiserver.GraphDefinition, opts *NativeGraphOptions) *NativeGraph {
	t.Helper()
	g, err := ToLiteLLMAgents(def, opts)
	if err != nil || g == nil {
		t.Fatalf("ToLiteLLMAgents = %v, %v", g, err)
	}
	return g
}

func handlersFor(calls *[]string) map[string]aiserver.ToolHandler {
	mk := func(name string) aiserver.ToolFunc {
		return tool(func(map[string]any) (any, error) { *calls = append(*calls, name); return name + " done", nil })
	}
	return map[string]aiserver.ToolHandler{"lookup": mk("lookup"), "refund": mk("refund")}
}

// One proxy client serves every node; each node's model name is sent as
// configured; a handoff makes the target's instructions the system message
// and resends the conversation; events and the graph span follow
// toOpenAIAgents.
func TestToLiteLLMAgentsRun(t *testing.T) {
	ld := twoNode(nil)
	installClient(t, ld)
	exp := tracer(t)
	tripwire(t)
	f, url := proxy(t,
		completion(`{"role":"assistant","content":null,"tool_calls":`+callsJSON([]call{{"transfer_to_billing", "{}"}})+`}`, "tool_calls", 3, 1, 1),
		completion(`{"role":"assistant","content":null,"tool_calls":`+callsJSON([]call{{"refund", `{"id":1}`}})+`}`, "tool_calls", 4, 0, 2),
		textResp("refunded", 5, 3))
	var calls []string
	res, err := adapter(t, resolve(t, "g"), &NativeGraphOptions{ToolHandlers: handlersFor(&calls), Context: user, Handler: Options{BaseURL: url}}).
		Invoke(bg, "hello", aiserver.WithVariables(map[string]any{"tone": "calm"}))
	if err != nil {
		t.Fatal(err)
	}
	if res.Response != "refunded" || res.Usage != (aiserver.TokenUsage{Input: 12, Output: 6, Total: 18}) || !slices.Equal(calls, []string{"refund"}) {
		t.Errorf("res = %+v calls %v", res, calls)
	}
	if f.count() != 3 {
		t.Fatalf("proxy requests = %d, want 3 (one client for every node)", f.count())
	}
	r0, r1 := f.request(0), f.request(1)
	if m := messages(r0); r0["model"] != "anthropic/claude-triage" || len(m) != 2 || m[0] != (msg{role: "system", content: "Route calm"}) || m[1] != (msg{role: "user", content: "hello"}) ||
		!reflect.DeepEqual(toolNames(r0), []string{"lookup", "transfer_to_billing"}) {
		t.Errorf("root request = %v", r0)
	}
	m1 := messages(r1)
	if r1["model"] != "gemini/billing" || len(m1) != 4 || m1[0] != (msg{role: "system", content: "Bill"}) || m1[1] != (msg{role: "user", content: "hello"}) ||
		m1[2].calls != "transfer_to_billing" || m1[3].role != "tool" || m1[3].callID != "call_transfer_to_billing" ||
		!reflect.DeepEqual(toolNames(r1), []string{"refund"}) {
		t.Errorf("handoff target request = %+v (model %v, tools %v)", m1, r1["model"], toolNames(r1))
	}

	nodes := ld.events("$ld:ai:graph:node")
	if len(nodes) != 2 || nodes[0].data["nodeKey"] != "triage" || nodes[0].data["index"] != 0.0 || nodes[1].data["nodeKey"] != "billing" || nodes[1].data["index"] != 1.0 {
		t.Errorf("node events = %+v", nodes)
	}
	if h := ld.events("$ld:ai:graph:handoff_success"); len(h) != 1 || h[0].data["sourceKey"] != "triage" || h[0].data["targetKey"] != "billing" || h[0].data["configKey"] != "g" {
		t.Errorf("handoff = %+v", h)
	}
	for _, ev := range []string{"$ld:ai:graph:duration:total", "$ld:ai:graph:total_tokens", "$ld:ai:graph:invocation_success"} {
		if e := ld.events(ev); len(e) != 1 || e[0].data["configKey"] != "g" {
			t.Errorf("%s = %+v, want one on the graph key", ev, e)
		}
	}
	if tt := ld.events("$ld:ai:graph:total_tokens"); len(tt) == 1 && tt[0].value != 18 {
		t.Errorf("total_tokens = %v", tt[0].value)
	}
	spans := spansByName(exp)[kitbase.GraphSpanName]
	if len(spans) != 1 || attrs(spans[0])["launchdarkly.graph.path"] != "triage->billing" || spans[0].Status().Code != codes.Ok {
		t.Errorf("graph span = %v", spans)
	}
	if byName := spansByName(exp); len(byName["chat gemini/billing"]) != 2 || len(byName["execute_tool refund"]) != 1 {
		t.Errorf("child spans = %v", byName)
	}
	if n := len(spansByName(exp)[SpanName]); n != 0 {
		t.Errorf("%d %s spans, want none in a graph run", n, SpanName)
	}
}

// Each node forwards exactly the allowlist from its own parameters; only the
// root's max_turns caps the run.
func TestToLiteLLMAgentsParameters(t *testing.T) {
	ld := twoNode(map[string]map[string]any{
		"triage":  node("Route", "Anthropic", "anthropic/claude-triage", map[string]any{"temperature": 0.1, "store": true, "max_turns": 2.0, "metadata": map[string]any{"a": 1.0}}),
		"billing": node("Bill", "Gemini", "gemini/billing", map[string]any{"top_p": 0.7, "api_base": "http://evil", "max_turns": 50.0}),
	})
	ld.variations["g"] = graphFlag("triage", map[string][]any{"triage": {map[string]any{"key": "billing"}}, "billing": {map[string]any{"key": "billing"}}})
	installClient(t, ld)
	f, c := fakeClient(t, toolResp(call{"transfer_to_billing", "{}"}), toolResp(call{"transfer_to_billing", "{}"}), textResp("x", 1, 1))
	_, err := adapter(t, resolve(t, "g"), &NativeGraphOptions{Handler: Options{Client: c}}).Invoke(bg, "q")
	if f.count() < 2 {
		t.Fatalf("requests = %d", f.count())
	}
	r0, r1 := f.request(0), f.request(1)
	if r0["temperature"] != 0.1 || r0["store"] != nil || r0["metadata"] != nil || r0["max_turns"] != nil {
		t.Errorf("root request = %v", r0)
	}
	if r1["top_p"] != 0.7 || r1["temperature"] != nil || strings.Contains(f.rec(1).raw, "evil") {
		t.Errorf("billing request = %v", r1)
	}
	// billing hands back to itself until the root's cap of 2 turns stops it.
	if !errors.Is(err, ErrMaxTurns) || f.count() != 2 {
		t.Errorf("err = %v after %d requests, want ErrMaxTurns after 2", err, f.count())
	}
}

// The adapter applies the Bedrock region prefix like the handler.
func TestToLiteLLMAgentsBedrockPrefix(t *testing.T) {
	triage := node("Route", "Bedrock", "bedrock/anthropic.claude-x", nil)
	triage["model"].(map[string]any)["region"] = "us"
	ld := &fakeLD{variations: map[string]map[string]any{"g": graphFlag("triage", nil), "triage": triage}}
	installClient(t, ld)
	f, c := fakeClient(t, textResp("ok", 1, 1))
	if _, err := adapter(t, resolve(t, "g"), &NativeGraphOptions{Handler: Options{Client: c}}).Invoke(bg, "q"); err != nil {
		t.Fatal(err)
	}
	if f.request(0)["model"] != "bedrock/us.anthropic.claude-x" {
		t.Errorf("model = %v", f.request(0)["model"])
	}
}

// Root history (multimodal) precedes input; system history is not
// duplicated; NativeTool is never a node tool.
func TestToLiteLLMAgentsHistoryAndNativeTool(t *testing.T) {
	ld := twoNode(map[string]map[string]any{"billing": node("Bill", "Gemini", "gemini/billing", nil, "search")})
	installClient(t, ld)
	f, c := fakeClient(t, toolResp(call{"transfer_to_billing", "{}"}), textResp("ok", 1, 1))
	history := []aiserver.Message{
		{Role: "system", Text: "stale system"},
		{Role: "user", Blocks: []aiserver.ContentBlock{{Type: "image", Source: &aiserver.ImageSource{Type: "base64", MediaType: "image/png", Data: "AAA"}}}},
	}
	_, err := adapter(t, resolve(t, "g"), &NativeGraphOptions{Handler: Options{Client: c},
		ToolHandlers: map[string]aiserver.ToolHandler{"search": &aiserver.NativeTool{ToolName: "web_search"}}}).Invoke(bg, "what is it?", aiserver.WithHistory(history))
	if err != nil {
		t.Fatal(err)
	}
	m := messages(f.request(0))
	if len(m) != 3 || m[0].role != "system" || strings.Contains(m[0].content, "stale") || m[1].role != "user" ||
		!strings.Contains(m[1].content, "data:image/png;base64,AAA") || m[2] != (msg{role: "user", content: "what is it?"}) {
		t.Errorf("root messages = %+v", m)
	}
	if got := toolNames(f.request(1)); len(got) != 0 || strings.Contains(f.rec(1).raw, "web_search") {
		t.Errorf("billing tools = %v, want the NativeTool skipped", got)
	}
}

// An empty, non-nil history is the same as no history.
func TestToLiteLLMAgentsEmptyHistory(t *testing.T) {
	installClient(t, twoNode(nil))
	requests := func(opts ...aiserver.CallOption) []msg {
		f, c := fakeClient(t, textResp("ok", 1, 1))
		if _, err := adapter(t, resolve(t, "g"), &NativeGraphOptions{Handler: Options{Client: c}}).Invoke(bg, "hello", opts...); err != nil {
			t.Fatal(err)
		}
		return messages(f.request(0))
	}
	none, empty := requests(), requests(aiserver.WithHistory([]aiserver.Message{}))
	if len(none) != 2 || !slices.Equal(none, empty) {
		t.Errorf("no history = %+v, empty history = %+v, want the same system + user pair", none, empty)
	}
}

// A missing proxy, a failing ClientFactory and a run error share one error
// path: the graph span records the error and ends, and invocation_failure
// is tracked on the graph key. Definition errors come from Invoke too.
func TestToLiteLLMAgentsErrors(t *testing.T) {
	setupErrs := []struct {
		name string
		opts Options
		want error
	}{
		{"no proxy", Options{}, ErrNoBaseURL},
		{"client factory", Options{ClientFactory: func(aiserver.AiConfig) (*openai.Client, error) { return nil, errFactory }}, errFactory},
	}
	for _, tc := range setupErrs {
		t.Run(tc.name, func(t *testing.T) {
			tripwire(t)
			ld := twoNode(nil)
			installClient(t, ld)
			exp := tracer(t)
			g := adapter(t, resolve(t, "g"), &NativeGraphOptions{Context: user, Handler: tc.opts})
			if _, err := g.Invoke(bg, "q"); !errors.Is(err, tc.want) {
				t.Fatalf("err = %v, want %v", err, tc.want)
			}
			if f := ld.events("$ld:ai:graph:invocation_failure"); len(f) != 1 || f[0].data["configKey"] != "g" {
				t.Errorf("invocation_failure = %+v", f)
			}
			if s := spansByName(exp)[kitbase.GraphSpanName]; len(s) != 1 || s[0].Status().Code != codes.Error || s[0].EndTime().IsZero() {
				t.Errorf("graph span = %v", s)
			}
		})
	}

	t.Run("run error", func(t *testing.T) {
		ld := twoNode(nil)
		installClient(t, ld)
		exp := tracer(t)
		_, c := fakeClient(t, "STATUS400")
		_, err := adapter(t, resolve(t, "g"), &NativeGraphOptions{Context: user, Handler: Options{Client: c}}).Invoke(bg, "q")
		var apiErr *openai.Error
		if !errors.As(err, &apiErr) {
			t.Fatalf("err = %v", err)
		}
		if len(ld.events("$ld:ai:graph:invocation_failure")) != 1 {
			t.Error("invocation_failure not tracked")
		}
		if s := spansByName(exp)[kitbase.GraphSpanName]; len(s) != 1 || s[0].Status().Code != codes.Error {
			t.Errorf("graph span = %v", s)
		}
	})

	t.Run("no context: nothing tracked", func(t *testing.T) {
		ld := twoNode(nil)
		installClient(t, ld)
		_, c := fakeClient(t, toolResp(call{"transfer_to_billing", "{}"}), textResp("ok", 1, 1))
		if _, err := adapter(t, resolve(t, "g"), &NativeGraphOptions{Handler: Options{Client: c}}).Invoke(bg, "q"); err != nil {
			t.Fatal(err)
		}
		for _, tr := range ld.tracks {
			if strings.HasPrefix(tr.event, "$ld:ai:graph:") {
				t.Errorf("tracked without a context: %s", tr.event)
			}
		}
	})

	t.Run("building agents fails: span ERROR and invocation_failure", func(t *testing.T) {
		ld := &fakeLD{variations: map[string]map[string]any{}}
		installClient(t, ld)
		exp := tracer(t)
		_, c := fakeClient(t)
		_, err := adapter(t, &aiserver.GraphDefinition{Key: "rootless", Enabled: true}, &NativeGraphOptions{Context: user, Handler: Options{Client: c}}).Invoke(bg, "q")
		if !errors.Is(err, ErrNoRoot) {
			t.Fatalf("err = %v, want ErrNoRoot", err)
		}
		if f := ld.events("$ld:ai:graph:invocation_failure"); len(f) != 1 || f[0].data["configKey"] != "rootless" {
			t.Errorf("invocation_failure = %+v", f)
		}
		if s := spansByName(exp)[kitbase.GraphSpanName]; len(s) != 1 || s[0].Status().Code != codes.Error || s[0].EndTime().IsZero() {
			t.Errorf("graph span = %v", s)
		}
	})

	t.Run("disabled, rootless and nil definitions", func(t *testing.T) {
		installClient(t, &fakeLD{variations: map[string]map[string]any{}})
		_, c := fakeClient(t)
		opts := &NativeGraphOptions{Handler: Options{Client: c}}
		if _, err := adapter(t, resolve(t, "off-graph"), opts).Invoke(bg, "q"); !errors.Is(err, aiserver.ErrGraphDisabled) {
			t.Errorf("disabled err = %v", err)
		}
		if _, err := adapter(t, &aiserver.GraphDefinition{Key: "rootless", Enabled: true}, opts).Invoke(bg, "q"); !errors.Is(err, ErrNoRoot) {
			t.Errorf("rootless err = %v", err)
		}
		if _, err := adapter(t, nil, opts).Invoke(bg, "q"); !errors.Is(err, ErrNilGraph) {
			t.Errorf("nil err = %v", err)
		}
	})
}

var errFactory = errors.New("factory failed")

// LiteLLMGraph pre-binds exactly this package's wildcard handler with the
// proxy options, ignores caller handlers, and reports only itself.
func TestLiteLLMGraph(t *testing.T) {
	tripwire(t)
	ld := &fakeLD{variations: map[string]map[string]any{
		"g":    graphFlag("solo", nil),
		"solo": node("Solo", "Mistral", "mistral/solo", nil, "lookup"),
	}}
	installClient(t, ld)
	f, url := proxy(t, sseToolCall("lookup", "{}"), sse([]string{"fine"}, 1, 1))
	bogusCalled := false
	bogus := &aiserver.ProviderHandler{ProvidesFor: aiserver.ProvidesFor{Provider: "Mistral", Mode: aiserver.ModeAgent},
		Invoke: func(context.Context, aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
			bogusCalled = true
			return aiserver.HandlerResult{}, nil
		}}
	var calls []string
	g, err := LiteLLMGraph("g", aiserver.GraphOptions{Handlers: []*aiserver.ProviderHandler{bogus}, ToolHandlers: handlersFor(&calls)}, Options{BaseURL: url, APIKey: "k"})
	if err != nil {
		t.Fatal(err)
	}
	res, err := g.Invoke(bg, "q", user)
	if err != nil {
		t.Fatal(err)
	}
	if res.Response != "fine" || bogusCalled || f.count() != 2 || f.rec(0).auth != "Bearer k" || !slices.Equal(calls, []string{"lookup"}) {
		t.Errorf("res=%+v bogus=%v requests=%d auth=%q calls=%v", res, bogusCalled, f.count(), f.rec(0).auth, calls)
	}
	if got := ld.usageHelpers(); !reflect.DeepEqual(got, []string{"litellm-agents.litellmGraph"}) {
		t.Errorf("usage helpers = %v", got)
	}
}
