package aiserver

import (
	"context"
	"errors"
	"maps"
	"reflect"
	"slices"
	"strings"
	"testing"
)

// Regression tests for bugs fixed in the core. Each was shown to fail with
// its fix reverted.

// A graph judge whose handler fails reports invocation_failure only, never
// success as well.
func TestGraphJudgeFailureTracksOnlyFailure(t *testing.T) {
	f := withFake(t, twoNode(map[string]any{"gj": enabledConfig("P", "judge", nil)}))
	boom := errors.New("judge provider down")
	h := createHandler(ProvidesFor{"P", ModeAgent}, func(_ context.Context, req HandlerRequest) (HandlerResult, error) {
		if req.Variables["response_to_evaluate"] != nil {
			return HandlerResult{}, boom
		}
		return HandlerResult{Output: "final"}, nil
	}, nil, CaptureDefault)
	_, err := Graph("g", GraphOptions{Handlers: []*ProviderHandler{h}, GraphJudge: "gj"}).Invoke(bg, "q", userCtx)
	if !errors.Is(err, boom) {
		t.Fatalf("err = %v, want the judge failure", err)
	}
	if n := len(f.events("$ld:ai:graph:invocation_success")); n != 0 {
		t.Errorf("invocation_success tracked %d times after a failed graph judge", n)
	}
	if n := len(f.events("$ld:ai:graph:invocation_failure")); n != 1 {
		t.Errorf("invocation_failure = %d, want 1", n)
	}
	if n := len(f.events("$ld:ai:graph:duration:total")); n != 1 {
		t.Errorf("duration:total = %d, want 1", n)
	}
}

// With no handler list, a parent handler for another provider is not used
// for a judge: that would send the judge's model to the wrong API.
func TestPickJudgeHandlerNilHandlersSkipsMismatchedParent(t *testing.T) {
	openai := &ProviderHandler{ProvidesFor: ProvidesFor{"OpenAI", ModeMessages}}
	if h, _, ok := pickJudgeHandler(openai, nil, "Anthropic", ModeMessages); ok || h != nil {
		t.Errorf("mismatched parent used: %v %v", h, ok)
	}
	if h, _, ok := pickJudgeHandler(openai, nil, "OpenAI", ModeMessages); !ok || h != openai {
		t.Errorf("matching parent not used: %v %v", h, ok)
	}
}

// A graph judge on another provider routes to that provider's handler, not
// the root node's.
func TestGraphJudgeUsesItsOwnProvidersHandler(t *testing.T) {
	v := twoNode(map[string]any{"gj": enabledConfig("Anthropic", "judge", nil)})
	v["root-node"].(map[string]any)["provider"] = map[string]any{"name": "OpenAI"}
	v["leaf-node"].(map[string]any)["provider"] = map[string]any{"name": "OpenAI"}
	withFake(t, v)
	var openaiJudged, anthropicJudged int
	openai := createHandler(ProvidesFor{"OpenAI", ModeAgent}, func(_ context.Context, req HandlerRequest) (HandlerResult, error) {
		if req.Variables["response_to_evaluate"] != nil {
			openaiJudged++
		}
		return HandlerResult{Output: "final"}, nil
	}, nil, CaptureDefault)
	anthropic := createHandler(ProvidesFor{"Anthropic", ModeMessages}, func(context.Context, HandlerRequest) (HandlerResult, error) {
		anthropicJudged++
		return HandlerResult{Output: `{"score":0.5,"reasoning":"ok"}`}, nil
	}, nil, CaptureDefault)
	resp, err := Graph("g", GraphOptions{Handlers: []*ProviderHandler{openai, anthropic}, GraphJudge: "gj"}).Invoke(bg, "q", userCtx)
	if err != nil {
		t.Fatal(err)
	}
	if anthropicJudged != 1 || openaiJudged != 0 || resp.JudgeResults["gj"].Score != 0.5 {
		t.Errorf("anthropic judged %d, openai judged %d, results %v", anthropicJudged, openaiJudged, resp.JudgeResults)
	}
}

// judgeVisibleKeys is every variable a judge handler receives: the two judge
// variables plus the context and track data every invocation gets.
var judgeVisibleKeys = []string{"__ld", "__ldContextTyped", "ldContext", "message_history", "response_to_evaluate"}

func TestRunJudgeVariablesAreExactlyTheJudgeKeys(t *testing.T) {
	if _, has := reflect.TypeFor[JudgeTask]().FieldByName("Variables"); has {
		t.Error("JudgeTask carries caller variables; a judge must see only its own")
	}
	_ = Shutdown(bg)
	t.Cleanup(func() { _ = Shutdown(bg) })
	setClientForTest(&fakeLD{})
	var got map[string]any
	h := createHandler(ProvidesFor{"P", ModeMessages}, func(_ context.Context, req HandlerRequest) (HandlerResult, error) {
		got = req.Variables
		return HandlerResult{Output: `{"score":1,"reasoning":"r"}`}, nil
	}, nil, CaptureDefault)
	task := JudgeTask{ConfigKey: "judge", JudgeConfig: AiConfig{Instructions: "grade {{message_history}}"},
		ActualOutput: "out", UserInput: "in", UserContext: userCtx, JudgeProvider: "P", JudgeMode: ModeMessages}
	if _, err := RunJudge(bg, task, []*ProviderHandler{h}); err != nil {
		t.Fatal(err)
	}
	if keys := slices.Sorted(maps.Keys(got)); !slices.Equal(keys, judgeVisibleKeys) {
		t.Errorf("judge variables = %v, want %v", keys, judgeVisibleKeys)
	}
	if got["response_to_evaluate"] != "out" || !strings.HasPrefix(got["message_history"].(string), "in\n\nout") {
		t.Errorf("judge variables = %v", got)
	}
}

// The graph span carries config association but no context identity; that
// belongs on the root handler span only.
func TestGraphSpanHasNoContextIdentity(t *testing.T) {
	withFake(t, twoNode(nil))
	exp := withSpans(t)
	rh := &recordingHandler{}
	if _, err := Graph("g", GraphOptions{Handlers: []*ProviderHandler{rh.handler()}}).Invoke(bg, "q", userCtx); err != nil {
		t.Fatal(err)
	}
	found := false
	for _, s := range exp.GetSpans().Snapshots() {
		if s.Name() != GraphSpanName {
			continue
		}
		found = true
		attrs := map[string]string{}
		for _, a := range s.Attributes() {
			attrs[string(a.Key)] = a.Value.Emit()
			if strings.HasPrefix(string(a.Key), "context.contextKeys.") {
				t.Errorf("graph span carries %s", a.Key)
			}
		}
		if attrs["launchdarkly.config.key"] != "g" || attrs["launchdarkly.graph.key"] != "g" || attrs["launchdarkly.run.id"] == "" ||
			attrs["launchdarkly.operation.type"] != "gen_ai" {
			t.Errorf("graph span config association = %v", attrs)
		}
		ff := false
		for _, e := range s.Events() {
			if e.Name != "feature_flag" {
				continue
			}
			ff = true
			for _, a := range e.Attributes {
				if a.Key == "feature_flag.context.id" || a.Key == "feature_flag.contextKeys" {
					t.Errorf("graph feature_flag event carries %s", a.Key)
				}
			}
		}
		if !ff {
			t.Error("graph span has no feature_flag event")
		}
	}
	if !found {
		t.Fatal("no graph span")
	}
}

// The "Tools available" line is byte-stable however Go orders the map.
func TestTrajectoryToolOrderIsDeterministic(t *testing.T) {
	names := []string{"echo", "alpha", "delta", "charlie", "bravo"}
	tools := map[string]any{}
	th := map[string]ToolHandler{}
	for _, n := range names {
		tools[n] = map[string]any{"name": n, "type": "function", "parameters": map[string]any{}}
		th[n] = ToolFunc(func(context.Context, map[string]any) (any, error) { return n + " done", nil })
	}
	withFake(t, map[string]any{
		"cfg":   enabledConfig("P", "", judgedConfig(map[string]any{"tools": tools})),
		"judge": enabledConfig("P", "judge", nil),
	})
	var histories []string
	h := createHandler(ProvidesFor{"P", ModeMessages}, func(ctx context.Context, req HandlerRequest) (HandlerResult, error) {
		if mh, ok := req.Variables["message_history"].(string); ok {
			histories = append(histories, mh)
			return HandlerResult{Output: `{"score":1,"reasoning":"r"}`}, nil
		}
		for _, n := range names { // a fixed call order
			_, _ = req.ToolHandlers[n].(ToolFunc)(ctx, nil)
		}
		return HandlerResult{Output: "answer"}, nil
	}, nil, CaptureDefault)
	for range 20 {
		if _, err := Config(ConfigArgs{Key: "cfg", Handlers: []*ProviderHandler{h}, ToolHandlers: th}).Invoke(bg, "q", userCtx); err != nil {
			t.Fatal(err)
		}
	}
	if len(histories) != 20 {
		t.Fatalf("judge ran %d times", len(histories))
	}
	for i, mh := range histories {
		if mh != histories[0] {
			t.Fatalf("run %d history differs:\n%s\n---\n%s", i, histories[0], mh)
		}
	}
	if !strings.Contains(histories[0], "Tools available: alpha, bravo, charlie, delta, echo") {
		t.Errorf("history = %s", histories[0])
	}
}

// A BYOC client whose Flush panics is still closed, and Shutdown returns.
func TestShutdownClosesClientWhoseFlushPanics(t *testing.T) {
	_ = Shutdown(bg)
	t.Cleanup(func() { _ = Shutdown(bg) })
	f := &fakeLD{flushPanic: true}
	if _, err := InitWithClient(bg, f, InitOptions{OTLPEndpoint: unreachableOTLP}); err != nil {
		t.Fatal(err)
	}
	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Errorf("Shutdown panicked: %v", r)
			}
		}()
		_ = Shutdown(bg)
	}()
	if f.flushes != 1 || f.closed != 1 {
		t.Errorf("flushes=%d closed=%d, want 1 and 1", f.flushes, f.closed)
	}
	if _, err := GetClient(); err == nil {
		t.Error("singleton not cleared")
	}
}
