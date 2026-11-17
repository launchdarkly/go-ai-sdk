package aiserver

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestClassifiersField(t *testing.T) {
	cfg, err := kitParseAiConfig(map[string]any{"model": map[string]any{"name": "jev"}, "provider": map[string]any{"name": "TypeSafe"},
		"classifiers": typesafeClassifiers})
	if err != nil {
		t.Fatal(err)
	}
	if len(cfg.Classifiers) != 2 || cfg.Classifiers[0]["key"] != "migration_intent" || cfg.Classifiers[1]["eventKey"] != "ev-migration" {
		t.Errorf("Classifiers = %v", cfg.Classifiers)
	}
}

func TestTypesafeJudgeResultEventKeys(t *testing.T) {
	var judgeReqs []HandlerRequest
	_, handlers := typesafeEnv(t, typesafePayload, CaptureOn, &judgeReqs)
	resp, err := Config(ConfigArgs{Key: "cfg", Handlers: handlers}).Invoke(bg, "q", userCtx)
	if err != nil {
		t.Fatal(err)
	}
	if a, b := resp.JudgeResults["judge.migration_intent"].EventKey, resp.JudgeResults["judge.migration"].EventKey; a != "ev-intent" || b != "ev-migration" {
		t.Errorf("event keys %q %q", a, b)
	}
	// A classic judge result has no event key, and the JSON omits it.
	b, _ := json.Marshal(JudgeResult{Score: 1})
	var m map[string]any
	_ = json.Unmarshal(b, &m)
	if _, has := m["eventKey"]; has {
		t.Errorf("classic result JSON %s has eventKey", b)
	}
}

func TestRunJudgeTypesafeMetrics(t *testing.T) {
	var judgeReqs []HandlerRequest
	_, handlers := typesafeEnv(t, typesafePayload, CaptureOn, &judgeReqs)
	resp, err := Config(ConfigArgs{Key: "cfg", Handlers: handlers, SkipJudges: true}).Invoke(bg, "q", userCtx)
	if err != nil || len(resp.JudgeTasks) != 1 {
		t.Fatalf("tasks=%v err=%v", resp.JudgeTasks, err)
	}
	out, err := RunJudge(bg, resp.JudgeTasks[0], handlers)
	if err != nil || out == nil {
		t.Fatalf("RunJudge = %+v, %v", out, err)
	}
	if out.EventKey != "ev-intent" {
		t.Errorf("EventKey %q, want the first label's", out.EventKey)
	}
	usage := TokenUsage{Input: 10, Output: 4, Total: 14}
	wantResults := map[string]JudgeResult{
		"judge.migration_intent": {Score: 0.9, Response: "yes", Usage: usage, EventKey: "ev-intent"},
		"judge.migration":        {Score: 0.2, Response: "no", Usage: usage, EventKey: "ev-migration"},
	}
	if !reflect.DeepEqual(out.Results, wantResults) {
		t.Errorf("Results = %+v\nwant %+v", out.Results, wantResults)
	}
	wantMetrics := []JudgeMetric{
		{EventKey: "ev-intent", Score: 0.9, JudgeConfigKey: "judge.migration_intent"},
		{EventKey: "ev-migration", Score: 0.2, JudgeConfigKey: "judge.migration"},
	}
	if !reflect.DeepEqual(out.Metrics, wantMetrics) {
		t.Errorf("Metrics = %+v, want %+v", out.Metrics, wantMetrics)
	}
}

// A classic deferred judge has no per-label results or metrics.
func TestRunJudgeClassicHasNoMetrics(t *testing.T) {
	withFake(t, map[string]any{"judge": enabledConfig("P", "judge", nil)})
	h := stubHandler("P", ModeMessages, `{"score":0.5,"reasoning":"r"}`, nil, nil)
	task := JudgeTask{ConfigKey: "judge", JudgeProvider: "P", JudgeMode: ModeMessages, ActualOutput: "a", UserContext: userCtx}
	out, err := RunJudge(bg, task, []*ProviderHandler{h})
	if err != nil || out == nil || out.Results != nil || out.Metrics != nil || out.EventKey != "" {
		t.Errorf("RunJudge = %+v, %v", out, err)
	}
}
