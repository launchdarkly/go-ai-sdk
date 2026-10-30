package aiserver

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
	"time"

	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

func judgedConfig(extra map[string]any) map[string]any {
	cfg := map[string]any{"judgeConfiguration": map[string]any{"judges": []any{map[string]any{"key": "judge", "samplingRate": 1.0}}}}
	for k, v := range extra {
		cfg[k] = v
	}
	return cfg
}

// spanJudgeHandler answers like a model that calls tool "lookup" when it has
// one, and returns a fixed verdict for judge calls, opening a real HandlerRun.
func spanJudgeHandler(verdict string, capture bool, judgeErr error, judgeReqs *[]HandlerRequest) *ProviderHandler {
	cc := CaptureOff
	if capture {
		cc = CaptureOn
	}
	return createHandler(ProvidesFor{"P", ModeMessages}, func(ctx context.Context, req HandlerRequest) (HandlerResult, error) {
		if req.Variables["response_to_evaluate"] == nil {
			if fn, ok := req.ToolHandlers["lookup"].(ToolFunc); ok {
				if _, err := fn(ctx, map[string]any{"q": "x"}); err != nil {
					return HandlerResult{}, err
				}
			}
			return HandlerResult{Output: "answer"}, nil
		}
		*judgeReqs = append(*judgeReqs, req)
		run := StartHandlerRun(ctx, HandlerSpanOptions{Tracer: TracerPrefix + "/test", Name: "judge.span", System: "p", RequestModel: "m"}, req, Prompt{})
		if judgeErr != nil {
			run.Fail(judgeErr)
			return HandlerResult{}, judgeErr
		}
		run.Finish("m", verdict)
		time.Sleep(15 * time.Millisecond) // work after End must not count in the span
		return HandlerResult{Output: verdict}, nil
	}, nil, cc)
}

func withSpans(t *testing.T) *tracetest.InMemoryExporter {
	exp := tracetest.NewInMemoryExporter()
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(sdktrace.NewTracerProvider(sdktrace.WithSyncer(exp)))
	t.Cleanup(func() { otel.SetTracerProvider(prev) })
	return exp
}

func TestJudgeTrajectoryAndPathsAgree(t *testing.T) {
	tools := map[string]any{"lookup": map[string]any{"name": "lookup", "type": "function", "parameters": map[string]any{}}}
	withFake(t, map[string]any{
		"cfg":   enabledConfig("P", "", judgedConfig(map[string]any{"tools": tools})),
		"judge": enabledConfig("P", "judge", nil),
	})
	withSpans(t)
	var judgeReqs []HandlerRequest
	h := spanJudgeHandler(`{"score":1,"reasoning":"r"}`, false, nil, &judgeReqs)
	th := map[string]ToolHandler{
		"lookup":   ToolFunc(func(context.Context, map[string]any) (any, error) { return "found", nil }),
		"unlisted": ToolFunc(func(context.Context, map[string]any) (any, error) { return "x", nil }),
	}
	args := ConfigArgs{Key: "cfg", Handlers: []*ProviderHandler{h}, ToolHandlers: th}
	if _, err := Config(args).Invoke(bg, "question", userCtx); err != nil {
		t.Fatal(err)
	}
	inline := judgeReqs[0].Variables["message_history"].(string)
	iq, it, io := strings.Index(inline, "question"), strings.Index(inline, "1. lookup"), strings.Index(inline, "answer")
	if iq < 0 || it < iq || io < it || !strings.Contains(inline, `"q":"x"`) || !strings.Contains(inline, "result: found") {
		t.Fatalf("trajectory not between input and output:\n%s", inline)
	}
	if strings.Contains(inline, "unlisted") {
		t.Error("a tool the config never offered is described")
	}
	if len(judgeReqs[0].ToolHandlers) != 0 {
		t.Error("judge received tools")
	}

	args.SkipJudges = true
	resp, err := Config(args).Invoke(bg, "question", userCtx)
	if err != nil || len(resp.JudgeTasks) != 1 {
		t.Fatalf("tasks=%v err=%v", resp.JudgeTasks, err)
	}
	// The task survives a serialization boundary.
	b, err := json.Marshal(resp.JudgeTasks[0])
	if err != nil {
		t.Fatal(err)
	}
	var task JudgeTask
	if err := json.Unmarshal(b, &task); err != nil {
		t.Fatal(err)
	}
	if _, err := RunJudge(bg, task, []*ProviderHandler{h}); err != nil {
		t.Fatal(err)
	}
	if deferred := judgeReqs[1].Variables["message_history"].(string); deferred != inline {
		t.Errorf("inline and deferred histories differ:\n%s\n---\n%s", inline, deferred)
	}
}

func TestNoToolsNoTrajectoryBlock(t *testing.T) {
	withFake(t, map[string]any{"cfg": enabledConfig("P", "", judgedConfig(nil)), "judge": enabledConfig("P", "judge", nil)})
	withSpans(t)
	var reqs []HandlerRequest
	h := spanJudgeHandler(`{"score":1,"reasoning":"r"}`, false, nil, &reqs)
	if _, err := Config(ConfigArgs{Key: "cfg", Handlers: []*ProviderHandler{h}}).Invoke(bg, "q", userCtx); err != nil {
		t.Fatal(err)
	}
	if got := reqs[0].Variables["message_history"]; got != "q\n\nanswer\n\n"+FormattingInstructions {
		t.Errorf("history = %q", got)
	}
}

func TestJudgeEvaluationSpan(t *testing.T) {
	for _, capture := range []bool{false, true} {
		withFake(t, map[string]any{"cfg": enabledConfig("P", "", judgedConfig(nil)), "judge": enabledConfig("P", "judge", nil)})
		exp := withSpans(t)
		var reqs []HandlerRequest
		h := spanJudgeHandler(`{"score":0.75,"reasoning":"because"}`, capture, nil, &reqs)
		resp, err := Config(ConfigArgs{Key: "cfg", Handlers: []*ProviderHandler{h}}).Invoke(bg, "q", userCtx)
		if err != nil || resp.JudgeResults["judge"].Response != "because" {
			t.Fatalf("resp=%+v err=%v", resp, err)
		}
		spans := exp.GetSpans().Snapshots()
		if len(spans) != 1 {
			t.Fatalf("spans = %d", len(spans))
		}
		s := spans[0]
		attrs := map[string]string{}
		for _, a := range s.Attributes() {
			attrs[string(a.Key)] = a.Value.Emit()
		}
		if attrs["gen_ai.evaluation.name"] != "judge" || attrs["gen_ai.evaluation.score.value"] != "0.75" {
			t.Errorf("evaluation attrs = %v", attrs)
		}
		if _, has := attrs["gen_ai.evaluation.score.label"]; has {
			t.Error("score.label invented")
		}
		if _, has := attrs["gen_ai.evaluation.explanation"]; has != capture {
			t.Errorf("explanation present=%v with capture=%v", has, capture)
		}
		found := false
		for _, e := range s.Events() {
			found = found || e.Name == "gen_ai.evaluation.result"
		}
		if !found {
			t.Error("missing gen_ai.evaluation.result event")
		}
		if d := s.EndTime().Sub(s.StartTime()); d >= 15*time.Millisecond {
			t.Errorf("span duration %v includes post-handler work", d)
		}
	}
}

func TestJudgeFailures(t *testing.T) {
	withFake(t, map[string]any{"cfg": enabledConfig("P", "", judgedConfig(nil)), "judge": enabledConfig("P", "judge", nil)})
	exp := withSpans(t)
	var reqs []HandlerRequest
	boom := errors.New("judge provider down")
	h := spanJudgeHandler("", false, boom, &reqs)
	if _, err := Config(ConfigArgs{Key: "cfg", Handlers: []*ProviderHandler{h}}).Invoke(bg, "q", userCtx); !errors.Is(err, boom) {
		t.Errorf("provider failure must reject, got %v", err)
	}
	for _, verdict := range []string{"prose", `{"score":"high","reasoning":"r"}`, `{"score":null}`} {
		exp.Reset()
		h := spanJudgeHandler(verdict, false, nil, &reqs)
		resp, err := Config(ConfigArgs{Key: "cfg", Handlers: []*ProviderHandler{h}}).Invoke(bg, "q", userCtx)
		if err != nil || len(resp.JudgeResults) != 0 {
			t.Errorf("%s: resp=%v err=%v", verdict, resp.JudgeResults, err)
		}
		for _, s := range exp.GetSpans().Snapshots() {
			for _, a := range s.Attributes() {
				if a.Key == "gen_ai.evaluation.score.value" {
					t.Errorf("%s: non-numeric score written", verdict)
				}
			}
		}
	}
}

func TestPickJudgeHandlerCollapse(t *testing.T) {
	agent := &ProviderHandler{ProvidesFor: ProvidesFor{"*", ModeAgent}}
	h, collapse, ok := pickJudgeHandler(agent, []*ProviderHandler{agent}, "Anthropic", ModeMessages)
	if !ok || h != agent || !collapse {
		t.Errorf("wildcard agent fallback: %v %v %v", h, collapse, ok)
	}
	if _, _, ok := pickJudgeHandler(&ProviderHandler{ProvidesFor: ProvidesFor{"OpenAI", ModeMessages}}, []*ProviderHandler{{ProvidesFor: ProvidesFor{"OpenAI", ModeMessages}}}, "Anthropic", ModeMessages); ok {
		t.Error("incompatible provider must be skipped")
	}
	got := collapseMessages(AiConfig{Messages: []Message{{Role: "system", Text: "a"}, {Role: "user", Text: "b"}}})
	if got.Instructions != "a\n\nb" || got.Messages != nil {
		t.Errorf("collapse = %+v", got)
	}
}

// A deferred judge result is tracked under the evaluated config, with exactly
// the payload the inline path sends (AIC-3576).
func TestDeferredJudgeTrackDataMatchesInline(t *testing.T) {
	parent := enabledConfig("P", "", judgedConfig(nil))
	parent["_ldMeta"] = map[string]any{"enabled": true, "variationKey": "v1", "version": 3, "modelKey": "parent-model", "modelVersion": 7}
	judge := enabledConfig("P", "judge", map[string]any{"evaluationMetricKey": "quality"})
	judge["_ldMeta"] = map[string]any{"enabled": true, "variationKey": "j1", "version": 5, "mode": "judge", "modelKey": "judge-model", "modelVersion": 2}
	f := withFake(t, map[string]any{"cfg": parent, "judge": judge})
	h := stubHandler("P", ModeMessages, `{"score":0.8,"reasoning":"good"}`, nil, nil)
	args := ConfigArgs{Key: "cfg", Handlers: []*ProviderHandler{h}}

	if _, err := Config(args).Invoke(bg, "question", userCtx); err != nil {
		t.Fatal(err)
	}
	q := f.events("quality")
	if len(q) != 1 {
		t.Fatalf("metric events = %v", q)
	}
	inline := q[0].Data

	args.SkipJudges = true
	resp, err := Config(args).Invoke(bg, "question", userCtx)
	if err != nil || len(resp.JudgeTasks) != 1 {
		t.Fatalf("tasks=%v err=%v", resp.JudgeTasks, err)
	}
	res, err := RunJudge(bg, resp.JudgeTasks[0], []*ProviderHandler{h})
	if err != nil || res == nil {
		t.Fatalf("res=%v err=%v", res, err)
	}
	want := resp.TrackData
	want.JudgeConfigKey = "judge"
	if !reflect.DeepEqual(res.TrackData, want) {
		t.Errorf("deferred trackData = %+v, want parent's %+v", res.TrackData, want)
	}

	// Same keys and values as the inline payload; only the per-invocation runId differs.
	b, _ := json.Marshal(res.TrackData)
	var deferred map[string]any
	_ = json.Unmarshal(b, &deferred)
	delete(deferred, "runId")
	delete(inline, "runId")
	if !reflect.DeepEqual(deferred, inline) {
		t.Errorf("deferred %v != inline %v", deferred, inline)
	}

	// A graph judge's task carries the graphKey in its parent trackData; it survives.
	task := resp.JudgeTasks[0]
	task.ParentTrackData.GraphKey = "g1"
	if res, _ := RunJudge(bg, task, []*ProviderHandler{h}); res == nil || res.TrackData.GraphKey != "g1" {
		t.Errorf("graphKey lost: %+v", res)
	}
}
