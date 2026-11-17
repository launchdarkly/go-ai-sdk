package aiserver

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/launchdarkly/go-ai-sdk/internal/kitbase"
)

// A TypeSafe judge variation carries its questions in classifiers, with no
// instructions and no messages.
var typesafeClassifiers = []any{
	map[string]any{"key": "migration_intent", "eventKey": "ev-intent", "type": "noul", "instructions": "Does the user want to migrate?"},
	map[string]any{"key": "migration", "eventKey": "ev-migration", "type": "noul", "instructions": "Did they migrate?"},
}

const typesafePayload = `{"kind":"typesafe","results":[` +
	`{"key":"migration_intent","eventKey":"ev-intent","score":0.9,"reason":"yes"},` +
	`{"key":"migration","eventKey":"ev-migration","score":0.2,"reason":"no"}]}`

func typesafeJudgeVariation() map[string]any {
	v := enabledConfig("TypeSafe", "judge", map[string]any{
		"classifiers": typesafeClassifiers, "messages": []any{}, "evaluationMetricKey": "quality",
	})
	delete(v, "instructions")
	return v
}

// typesafeEnv serves a parent config "cfg" judged by "judge", a TypeSafe
// judge whose handler returns output with the given content capture.
// judgeReqs records the judge calls.
func typesafeEnv(t *testing.T, output string, capture ContentCapture, judgeReqs *[]HandlerRequest) (*fakeLD, []*ProviderHandler) {
	f := withFake(t, map[string]any{"cfg": enabledConfig("P", "", judgedConfig(nil)), "judge": typesafeJudgeVariation()})
	parent := stubHandler("P", ModeMessages, "answer", nil, nil)
	jev := createHandler(ProvidesFor{"TypeSafe", ModeMessages}, func(ctx context.Context, req HandlerRequest) (HandlerResult, error) {
		*judgeReqs = append(*judgeReqs, req)
		run := kitStartHandlerRun(ctx, typesafeSpanOptions, req, prompt{})
		run.Finish("jev", output)
		return HandlerResult{Output: output, Usage: map[string]any{"input_tokens": 10, "output_tokens": 4}}, nil
	}, nil, capture)
	return f, []*ProviderHandler{parent, jev}
}

func TestClassifiersOnlyConfigIsValid(t *testing.T) {
	base := func(classifiers any) map[string]any {
		return map[string]any{"model": map[string]any{"name": "jev"}, "provider": map[string]any{"name": "TypeSafe"},
			"messages": []any{}, "classifiers": classifiers}
	}
	cfg, err := kitParseAiConfig(base(typesafeClassifiers))
	if err != nil {
		t.Fatalf("classifiers-only config rejected: %v", err)
	}
	b, _ := json.Marshal(cfg)
	if !strings.Contains(string(b), `"classifiers"`) {
		t.Errorf("parsed config lost its classifiers: %s", b)
	}
	if _, err := kitParseAiConfig(base([]any{})); err == nil {
		t.Error("an empty classifiers list must not make a config valid")
	}
}

func TestTypesafeJudgeExpandsInline(t *testing.T) {
	var judgeReqs []HandlerRequest
	f, handlers := typesafeEnv(t, typesafePayload, CaptureOn, &judgeReqs)
	exp := withSpans(t)
	resp, err := Config(ConfigArgs{Key: "cfg", Handlers: handlers}).Invoke(bg, "q", userCtx)
	if err != nil {
		t.Fatal(err)
	}
	if len(judgeReqs) != 1 {
		t.Fatalf("judge calls = %d, want one call for every label", len(judgeReqs))
	}
	b, _ := json.Marshal(judgeReqs[0].Config)
	if !strings.Contains(string(b), `"classifiers"`) {
		t.Errorf("judge handler config has no classifiers: %s", b)
	}
	if _, has := resp.JudgeResults["judge"]; has || len(resp.JudgeResults) != 2 {
		t.Fatalf("judge results %v, want only the two label keys", resp.JudgeResults)
	}
	for key, want := range map[string]struct {
		score  float64
		reason string
	}{"judge.migration_intent": {0.9, "yes"}, "judge.migration": {0.2, "no"}} {
		r := resp.JudgeResults[key]
		if r.Score != want.score || r.Response != want.reason || r.Usage.Input != 10 || r.Usage.Output != 4 {
			t.Errorf("%s = %+v, want score %v, reason %q and the full call usage", key, r, want.score, want.reason)
		}
	}
	// Each label is tracked on its own eventKey, never on evaluationMetricKey.
	if q := f.events("quality"); len(q) != 0 {
		t.Errorf("evaluationMetricKey tracked: %v", q)
	}
	for ev, want := range map[string]struct {
		score float64
		key   string
	}{"ev-intent": {0.9, "judge.migration_intent"}, "ev-migration": {0.2, "judge.migration"}} {
		got := f.events(ev)
		if len(got) != 1 || got[0].Metric != want.score || got[0].Data["judgeConfigKey"] != want.key {
			t.Errorf("track %s = %+v, want %v with judgeConfigKey %s", ev, got, want.score, want.key)
		}
	}

	// One judge span: attributes from the first label, one event per label.
	judgeSpan := typesafeJudgeSpan(t, exp)
	if judgeSpan.attrs["gen_ai.evaluation.name"] != "judge.migration_intent" || judgeSpan.attrs["gen_ai.evaluation.score.value"] != "0.9" {
		t.Errorf("span attributes %v, want the first label", judgeSpan.attrs)
	}
	if got := judgeSpan.eventNames(); strings.Join(got, ",") != "judge.migration_intent,judge.migration" {
		t.Errorf("evaluation events named %v", got)
	}
}

// The explanation is the label's reason, written only when the judge
// handler captures content.
func TestTypesafeJudgeExplanationFollowsCapture(t *testing.T) {
	for _, capture := range []ContentCapture{CaptureOn, CaptureOff} {
		on := capture == CaptureOn
		var judgeReqs []HandlerRequest
		_, handlers := typesafeEnv(t, typesafePayload, capture, &judgeReqs)
		exp := withSpans(t)
		resp, err := Config(ConfigArgs{Key: "cfg", Handlers: handlers}).Invoke(bg, "q", userCtx)
		if err != nil {
			t.Fatal(err)
		}
		if resp.JudgeResults["judge.migration_intent"].Response != "yes" {
			t.Errorf("capture on=%v: the reason must still reach the caller, got %+v", on, resp.JudgeResults)
		}
		judgeSpan := typesafeJudgeSpan(t, exp)
		got, has := judgeSpan.attrs["gen_ai.evaluation.explanation"]
		if on && got != "yes" {
			t.Errorf("capture on: span explanation = %q (present=%v), want the first label's reason", got, has)
		}
		if !on && has {
			t.Errorf("capture off: span carries explanation %q", got)
		}
		want := map[string]string{"judge.migration_intent": "yes", "judge.migration": "no"}
		if n := len(judgeSpan.eventNames()); n != len(want) {
			t.Fatalf("evaluation events = %d, want one per label", n)
		}
		for _, e := range judgeSpan.events {
			if e.Name != "gen_ai.evaluation.result" {
				continue
			}
			var name, expl string
			hasExpl := false
			for _, a := range e.Attributes {
				switch a.Key {
				case "gen_ai.evaluation.name":
					name = a.Value.AsString()
				case "gen_ai.evaluation.explanation":
					expl, hasExpl = a.Value.AsString(), true
				}
			}
			if on && expl != want[name] {
				t.Errorf("capture on: event %s explanation = %q (present=%v), want %q", name, expl, hasExpl, want[name])
			}
			if !on && hasExpl {
				t.Errorf("capture off: event %s carries explanation %q", name, expl)
			}
		}
	}
}

func typesafeJudgeSpan(t *testing.T, exp *tracetest.InMemoryExporter) *spanView {
	t.Helper()
	for _, s := range exp.GetSpans().Snapshots() {
		if s.Name() == typesafeSpanOptions.Name {
			return viewSpan(s)
		}
	}
	t.Fatal("no judge span")
	return nil
}

func TestTypesafeJudgeTaskRoundTrip(t *testing.T) {
	var judgeReqs []HandlerRequest
	f, handlers := typesafeEnv(t, typesafePayload, CaptureOn, &judgeReqs)
	resp, err := Config(ConfigArgs{Key: "cfg", Handlers: handlers, SkipJudges: true}).Invoke(bg, "q", userCtx)
	if err != nil || len(resp.JudgeTasks) != 1 {
		t.Fatalf("tasks=%v err=%v", resp.JudgeTasks, err)
	}
	b, err := json.Marshal(resp.JudgeTasks[0])
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), `"classifiers"`) {
		t.Fatalf("serialized task lost the classifiers: %s", b)
	}
	var task JudgeTask
	if err := json.Unmarshal(b, &task); err != nil {
		t.Fatal(err)
	}
	out, err := RunJudge(bg, task, handlers)
	if err != nil || out == nil {
		t.Fatalf("RunJudge = %+v, %v", out, err)
	}
	if cb, _ := json.Marshal(judgeReqs[0].Config); !strings.Contains(string(cb), `"classifiers"`) {
		t.Errorf("deferred judge handler config has no classifiers: %s", cb)
	}
	// The top-level fields are the first label's.
	if out.Score != 0.9 || out.Response != "yes" || out.TrackData.JudgeConfigKey != "judge.migration_intent" {
		t.Errorf("RunJudge = %+v, want the first label", out)
	}
	if len(f.events("quality")) != 0 || len(f.events("ev-intent")) != 0 {
		t.Error("RunJudge must not track; the caller tracks each metric")
	}
}

// A payload that says kind typesafe but is malformed is skipped and logged
// inline, and RunJudge returns (nil, nil).
func TestMalformedTypesafePayload(t *testing.T) {
	for name, output := range map[string]string{
		"no results":       `{"kind":"typesafe"}`,
		"empty results":    `{"kind":"typesafe","results":[]}`,
		"missing key":      `{"kind":"typesafe","results":[{"eventKey":"e","score":1}]}`,
		"missing score":    `{"kind":"typesafe","results":[{"key":"k","eventKey":"e"}]}`,
		"string score":     `{"kind":"typesafe","results":[{"key":"k","eventKey":"e","score":"high"}]}`,
		"missing eventKey": `{"kind":"typesafe","results":[{"key":"k","score":1}]}`,
	} {
		t.Run(name, func(t *testing.T) {
			var judgeReqs []HandlerRequest
			f, handlers := typesafeEnv(t, output, CaptureOn, &judgeReqs)
			log := captureSDKLog(t)
			resp, err := Config(ConfigArgs{Key: "cfg", Handlers: handlers}).Invoke(bg, "q", userCtx)
			if err != nil {
				t.Fatal(err)
			}
			if len(resp.JudgeResults) != 0 || len(f.events("quality"))+len(f.events("e")) != 0 {
				t.Errorf("malformed payload produced results %v or tracks", resp.JudgeResults)
			}
			if !strings.Contains(log.buf.String(), "level=ERROR") || !strings.Contains(log.buf.String(), "judge") {
				t.Errorf("not logged:\n%s", log.buf.String())
			}
			task := JudgeTask{ConfigKey: "judge", JudgeProvider: "TypeSafe", JudgeMode: ModeMessages, ActualOutput: "a", UserContext: userCtx}
			if out, err := RunJudge(bg, task, handlers); out != nil || err != nil {
				t.Errorf("RunJudge = %+v, %v, want nil, nil", out, err)
			}
		})
	}
}

var typesafeSpanOptions = kitbase.HandlerSpanOptions{Tracer: kitbase.TracerPrefix + "/test", Name: "typesafe.judge", System: "typesafe", RequestModel: "jev"}

// spanView is a finished span's attributes and its evaluation event names.
type spanView struct {
	attrs  map[string]string
	events []sdktrace.Event
}

func viewSpan(s sdktrace.ReadOnlySpan) *spanView {
	v := &spanView{attrs: map[string]string{}, events: s.Events()}
	for _, a := range s.Attributes() {
		v.attrs[string(a.Key)] = a.Value.Emit()
	}
	return v
}

func (v *spanView) eventNames() []string {
	var names []string
	for _, e := range v.events {
		if e.Name != "gen_ai.evaluation.result" {
			continue
		}
		for _, a := range e.Attributes {
			if a.Key == "gen_ai.evaluation.name" {
				names = append(names, a.Value.AsString())
			}
		}
	}
	return names
}
