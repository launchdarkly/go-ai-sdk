package shadowrequests

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/launchdarkly/go-sdk-common/v3/ldreason"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

const judgeMetric = "quality-score"

var shadowFields = []string{"isShadow", "primaryVariationKey", "shadowRequestId", "ownerKind", "ownerId"}

func judgeVariation() map[string]any {
	return map[string]any{
		"_ldMeta":             map[string]any{"enabled": true, "variationKey": "judge-v1", "version": 1, "mode": "completion"},
		"model":               map[string]any{"name": "judge-model"},
		"provider":            map[string]any{"name": "JudgeCo"},
		"instructions":        "Score {{response_to_evaluate}}",
		"evaluationMetricKey": judgeMetric,
	}
}

func judges(rate float64) map[string]any {
	return map[string]any{"judges": []any{map[string]any{"key": "quality", "samplingRate": rate}}}
}

// installJudged serves a fallthrough shadow whose variation has the quality
// judge at rate. With onServed, A carries the same judge at rate 1.
func installJudged(t *testing.T, rate float64, onServed bool) *shadowLD {
	t.Helper()
	payload := fallthroughPayload()
	variations := payload["shadowConfiguration"].(map[string]any)["variations"].(map[string]any)
	variations["claude-candidate"].(map[string]any)["judgeConfiguration"] = judges(rate)
	if onServed {
		payload["judgeConfiguration"] = judges(1)
	}
	f := &shadowLD{value: payload, reason: ldreason.NewEvalReasonFallthrough(),
		values: map[string]any{"quality": judgeVariation()}}
	installClient(t, f)
	return f
}

type judged struct {
	pair
	judge *stub
}

func newJudged() judged {
	return judged{newPair(), &stub{out: `{"score":0.8,"reasoning":"good"}`}}
}

func (j judged) handlers() []*aiserver.ProviderHandler {
	return append(j.pair.handlers(), j.judge.handler("JudgeCo"))
}

func (f *shadowLD) metrics(event string) []trackedEvent {
	f.mu.Lock()
	defer f.mu.Unlock()
	var out []trackedEvent
	for _, t := range f.tracks {
		if t.event == event {
			out = append(out, t)
		}
	}
	return out
}

func judgedTask(t *testing.T, s *ShadowAIConfig) ShadowTask {
	t.Helper()
	var tasks []ShadowTask
	s.opts.OnTask = func(task ShadowTask) { tasks = append(tasks, task) }
	invokeAndWait(t, s)
	if len(tasks) != 1 {
		t.Fatalf("tasks %d", len(tasks))
	}
	b, err := json.Marshal(tasks[0])
	if err != nil {
		t.Fatal(err)
	}
	var decoded ShadowTask
	if err := json.Unmarshal(b, &decoded); err != nil {
		t.Fatal(err)
	}
	return decoded
}

func must(b []byte, err error) []byte {
	if err != nil {
		panic(err)
	}
	return b
}

func TestShadowJudgeScoresTheShadowResponse(t *testing.T) {
	f := installJudged(t, 1, false)
	j := newJudged()
	resp := invokeAndWait(t, mustShadow(t, j.handlers(), ShadowOptions{}))
	if j.judge.count() != 1 {
		t.Fatalf("judge calls %d", j.judge.count())
	}
	req := j.judge.last()
	if req.UserInput != "shadow" || req.Variables["response_to_evaluate"] != "shadow" || req.Variables["input"] != "hello" {
		t.Fatalf("judge saw input %q, variables %v", req.UserInput, req.Variables)
	}
	scores := f.metrics(judgeMetric)
	if len(scores) != 1 || scores[0].metric != 0.8 {
		t.Fatalf("judge metrics %+v", scores)
	}
	d := scores[0].data
	if d["isShadow"] != true || d["variationKey"] != "claude-candidate" || d["shadowRequestId"] != "shadow-1" ||
		d["primaryVariationKey"] != "gpt-main" || d["judgeConfigKey"] != "quality" || d["runId"] != resp.TrackData.RunID {
		t.Fatalf("shadow judge metric data %v", d)
	}
}

func TestServedJudgeEventsCarryNoShadowField(t *testing.T) {
	f := installJudged(t, 1, true)
	j := newJudged()
	invokeAndWait(t, mustShadow(t, j.handlers(), ShadowOptions{}))
	scores := f.metrics(judgeMetric)
	if len(scores) != 2 {
		t.Fatalf("judge metrics %d, want one for A and one for B", len(scores))
	}
	var served, shadow int
	for _, s := range scores {
		if s.data["variationKey"] == "gpt-main" {
			served++
			for _, k := range shadowFields {
				if _, ok := s.data[k]; ok {
					t.Errorf("A's judge event carries %s", k)
				}
			}
		} else if s.data["isShadow"] == true && s.data["variationKey"] == "claude-candidate" {
			shadow++
		}
	}
	if served != 1 || shadow != 1 {
		t.Fatalf("served %d, shadow %d: %+v", served, shadow, scores)
	}
}

func TestShadowJudgeSamplingRate(t *testing.T) {
	for _, tc := range []struct {
		rate  float64
		calls int
	}{{0, 0}, {1, 1}} {
		f := installJudged(t, tc.rate, false)
		j := newJudged()
		task := judgedTask(t, mustShadow(t, j.handlers(), ShadowOptions{}))
		res := RunShadow(context.Background(), task, j.handlers())
		if res.Error != "" || j.judge.count() != tc.calls || len(f.metrics(judgeMetric)) != tc.calls {
			t.Fatalf("rate %v: error %q, judge calls %d", tc.rate, res.Error, j.judge.count())
		}
		if (res.JudgeResults == nil) != (tc.calls == 0) {
			t.Fatalf("rate %v: JudgeResults %v", tc.rate, res.JudgeResults)
		}
		if tc.calls == 1 && (res.JudgeResults["quality"].Score != 0.8 || res.JudgeResults["quality"].Response != "good") {
			t.Fatalf("JudgeResults %+v", res.JudgeResults)
		}
		_ = aiserver.Shutdown(context.Background())
	}
}

func TestNoShadowJudgeWhenTheShadowFails(t *testing.T) {
	for name, set := range map[string]func(*stub){
		"error":   func(s *stub) { s.err = errors.New("provider down") },
		"timeout": func(s *stub) { s.delay = time.Second },
	} {
		t.Run(name, func(t *testing.T) {
			f := installJudged(t, 1, false)
			j := newJudged()
			set(j.b)
			task := judgedTask(t, mustShadow(t, j.handlers(), ShadowOptions{Timeout: 30 * time.Millisecond}))
			res := RunShadow(context.Background(), task, j.handlers())
			if res.Error == "" || j.judge.count() != 0 || len(f.metrics(judgeMetric)) != 0 || res.JudgeResults != nil {
				t.Fatalf("error %q, judge calls %d, results %v", res.Error, j.judge.count(), res.JudgeResults)
			}
			if name == "timeout" && res.Error != "timeout" {
				t.Fatalf("error %q", res.Error)
			}
		})
	}
}

func TestNoShadowJudgeWithoutAMatchingShadowHandler(t *testing.T) {
	installJudged(t, 1, false)
	j := newJudged()
	task := judgedTask(t, mustShadow(t, j.handlers(), ShadowOptions{}))
	res := RunShadow(context.Background(), task, []*aiserver.ProviderHandler{j.judge.handler("JudgeCo")})
	if res.Error == "" || j.judge.count() != 0 {
		t.Fatalf("error %q, judge calls %d", res.Error, j.judge.count())
	}
}

func TestAShadowJudgeFailureIsNotAShadowFailure(t *testing.T) {
	for name, set := range map[string]func(*stub){
		"raises":      func(s *stub) { s.err = errors.New("judge down") },
		"unparseable": func(s *stub) { s.out = "prose" },
	} {
		t.Run(name, func(t *testing.T) {
			f := installJudged(t, 1, false)
			j := newJudged()
			set(j.judge)
			task := judgedTask(t, mustShadow(t, j.handlers(), ShadowOptions{}))
			res := RunShadow(context.Background(), task, j.handlers())
			if res.Error != "" || res.JudgeResults != nil || j.judge.count() != 1 {
				t.Fatalf("error %q, results %v, judge calls %d", res.Error, res.JudgeResults, j.judge.count())
			}
			var shadowSuccess int
			for _, e := range f.metrics("$ld:ai:generation:success") {
				if e.data["isShadow"] == true {
					shadowSuccess++
				}
			}
			if shadowSuccess != 1 {
				t.Fatalf("shadow successes %d", shadowSuccess)
			}
		})
	}
}

func TestShadowTimeoutCoversOnlyTheShadowHandler(t *testing.T) {
	installJudged(t, 1, false)
	j := newJudged()
	j.judge.delay = 150 * time.Millisecond
	task := judgedTask(t, mustShadow(t, j.handlers(), ShadowOptions{Timeout: 50 * time.Millisecond}))
	res := RunShadow(context.Background(), task, j.handlers())
	if res.Error != "" || res.JudgeResults["quality"].Score != 0.8 {
		t.Fatalf("error %q, results %v", res.Error, res.JudgeResults)
	}
}

func TestSkipJudgesDoesNotSkipShadowJudges(t *testing.T) {
	f := installJudged(t, 1, true)
	j := newJudged()
	s, err := ShadowConfig(aiserver.ConfigArgs{Key: "cfg", Handlers: j.handlers(), SkipJudges: true}, ShadowOptions{})
	if err != nil {
		t.Fatal(err)
	}
	resp := invokeAndWait(t, s)
	if len(resp.JudgeTasks) != 1 {
		t.Fatalf("A's judge tasks %d", len(resp.JudgeTasks))
	}
	scores := f.metrics(judgeMetric)
	if len(scores) != 1 || scores[0].data["isShadow"] != true {
		t.Fatalf("judge metrics %+v", scores)
	}
}

func TestInvokeReturnsBeforeTheShadowJudgeRuns(t *testing.T) {
	installJudged(t, 1, false)
	j := newJudged()
	j.judge.delay = 300 * time.Millisecond
	start := time.Now()
	resp, err := mustShadow(t, j.handlers(), ShadowOptions{}).Invoke(context.Background(), "hello", shadowUser)
	if err != nil || resp.Response != "served" {
		t.Fatalf("response %v, error %v", resp.Response, err)
	}
	if elapsed := time.Since(start); elapsed > 150*time.Millisecond {
		t.Fatalf("Invoke waited %v for the shadow judge", elapsed)
	}
	running.Wait()
	if j.judge.count() != 1 {
		t.Fatalf("judge calls %d", j.judge.count())
	}
}

func TestBackgroundAndDeferredShadowJudgesAgree(t *testing.T) {
	judgeEvents := func(f *shadowLD) []trackedEvent { return withoutRunID(f.metrics(judgeMetric)) }
	f := installJudged(t, 1, false)
	j := newJudged()
	invokeAndWait(t, mustShadow(t, j.handlers(), ShadowOptions{}))
	background := judgeEvents(f)
	_ = aiserver.Shutdown(context.Background())

	f = installJudged(t, 1, false)
	j = newJudged()
	task := judgedTask(t, mustShadow(t, j.handlers(), ShadowOptions{}))
	RunShadow(context.Background(), task, j.handlers())
	deferred := judgeEvents(f)
	if len(background) != 1 || !reflect.DeepEqual(background, deferred) {
		t.Fatalf("background %+v\ndeferred %+v", background, deferred)
	}
}

func TestShadowResultJudgeResultsRoundTripJSON(t *testing.T) {
	in := ShadowResult{TrackData: map[string]any{"isShadow": true},
		JudgeResults: map[string]aiserver.JudgeResult{"quality": {Score: 0.8, Response: "good"}}}
	b, _ := json.Marshal(in)
	var out ShadowResult
	if err := json.Unmarshal(b, &out); err != nil || !reflect.DeepEqual(in, out) {
		t.Fatalf("round trip %+v, %v", out, err)
	}
	b, _ = json.Marshal(ShadowResult{})
	if string(b) != string(must(json.Marshal(struct {
		Usage     aiserver.TokenUsage `json:"usage"`
		TrackData map[string]any      `json:"trackData"`
	}{}))) {
		t.Fatalf("a result with no judge has a judgeResults key: %s", b)
	}
}
