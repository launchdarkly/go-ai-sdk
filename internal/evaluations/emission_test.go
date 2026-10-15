package evaluations

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"regexp"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

func TestClientResolution(t *testing.T) {
	t.Run("init failure fails the run before any request", func(t *testing.T) {
		e := newEnv(t, 1)
		m := e.module()
		e.byoc, m.sdkKey, e.initErr = false, "sdk-key", errors.New("boom")
		runErr(t, m, baseArgs(handler("OpenAI", aiserver.ModeMessages, nil)), ErrNoClient, "boom")
		e.assertNoRequests()
	})
	t.Run("existing client is reused; a module key is warned about", func(t *testing.T) {
		e := newEnv(t, 1)
		m := e.module()
		m.sdkKey = "sdk-key"
		initClient = func(context.Context, aiserver.InitOptions) (aiserver.LDClient, error) {
			t.Fatal("initClient must not be called when a client exists")
			return nil, nil
		}
		mustRun(t, m, baseArgs(handler("OpenAI", aiserver.ModeMessages, nil)))
		if !strings.Contains(e.logs.String(), "SDK key is ignored") {
			t.Fatalf("missing warning: %s", e.logs)
		}
	})
	t.Run("bring-your-own client is reused silently", func(t *testing.T) {
		e := newEnv(t, 1)
		mustRun(t, e.module(), baseArgs(handler("OpenAI", aiserver.ModeMessages, nil)))
		if e.logs.Len() != 0 {
			t.Fatalf("unexpected log: %s", e.logs)
		}
	})
	t.Run("no client initializes one with exactly the key", func(t *testing.T) {
		e := newEnv(t, 1)
		m := e.module()
		e.byoc, m.sdkKey = false, "sdk-key"
		mustRun(t, m, baseArgs(handler("OpenAI", aiserver.ModeMessages, nil)))
		if len(e.initCalls) != 1 || e.initCalls[0] != (aiserver.InitOptions{SDKKey: "sdk-key"}) {
			t.Fatalf("init calls %+v", e.initCalls)
		}
	})
	t.Run("no client and no key", func(t *testing.T) {
		e := newEnv(t, 1)
		m := e.module()
		e.byoc = false
		runErr(t, m, baseArgs(handler("OpenAI", aiserver.ModeMessages, nil)), ErrNoClient, "no initialized LaunchDarkly client")
		e.assertNoRequests()
	})
}

func TestGenerationEventEmission(t *testing.T) {
	e := newEnv(t, 2)
	mustRun(t, e.module(), baseArgs(handler("OpenAI", aiserver.ModeMessages, func(context.Context, aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
		return aiserver.HandlerResult{Output: "ok", Usage: map[string]any{"input_tokens": 3.0, "output_tokens": 4.0}}, nil
	})))
	if e.ld.variations != 0 {
		t.Fatal("emission must not be gated by a flag")
	}
	evs := e.ld.events(GenerationEventName)
	if len(evs) != 2 {
		t.Fatalf("events %d", len(evs))
	}
	rfc := regexp.MustCompile(`^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?Z$`)
	ids := map[any]bool{}
	for _, ev := range evs {
		if ev.Metric != 1 || ev.Ctx.Kind() != "evaluation" || ev.Ctx.Key() != "run-1" ||
			ev.Ctx.GetValue("projectKey").StringValue() != "proj" || ev.Ctx.GetValue("evaluationId").StringValue() != "eval-1" {
			t.Fatalf("metric/context %v %v", ev.Metric, ev.Ctx)
		}
		d := ev.Data
		for k, want := range map[string]any{"projectKey": "proj", "evaluationId": "eval-1", "evaluationRunId": "run-1", "runId": "run-1",
			"datasetId": "ds-id", "evaluationKey": "eval-key", "evaluationVersion": 3.0, "datasetKey": "ds", "status": "COMPLETE", "output": "ok"} {
			if d[k] != want {
				t.Errorf("%s = %v, want %v", k, d[k], want)
			}
		}
		if !sameJSON(d["usage"], map[string]any{"inputTokens": 3.0, "outputTokens": 4.0}) {
			t.Errorf("usage %v", d["usage"])
		}
		for _, k := range []string{"inputTokens", "outputTokens", "input", "expectedOutput", "expected_output", "metadata", "variables",
			"error", "errorMessage", "tools", "trajectory", "toolCalls", "verdict", "contentHash"} {
			if _, has := d[k]; has {
				t.Errorf("field %s must be absent", k)
			}
		}
		for _, k := range []string{"emittedAt", "startedAt", "generatedAt"} {
			if s, _ := d[k].(string); !rfc.MatchString(s) {
				t.Errorf("%s = %v", k, d[k])
			}
		}
		if _, ok := d["latencyMs"].(float64); !ok {
			t.Errorf("latencyMs %v", d["latencyMs"])
		}
		if id, _ := d["eventId"].(string); len(id) != 64 {
			t.Errorf("eventId %v", d["eventId"])
		}
		ids[d["eventId"]] = true
	}
	if len(ids) != 2 {
		t.Fatal("changing rowIndex must change eventId")
	}
}

func sameJSON(a, b any) bool { return toJSON(a) == toJSON(b) }

func TestInlineRunOmitsDatasetIdentity(t *testing.T) {
	e := newEnv(t, 1)
	args := baseArgs(handler("OpenAI", aiserver.ModeMessages, nil))
	args.Dataset = InlineRows{map[string]any{"input": "x"}}
	args.Criteria = []Criterion{mustScorer(t, "s", constScorer(1), nil)}
	mustRun(t, e.module(), args)
	for _, ev := range e.ld.tracks {
		for _, k := range []string{"datasetId", "datasetKey"} {
			if _, has := ev.Data[k]; has {
				t.Fatalf("%s on inline %s event", k, ev.Event)
			}
		}
	}
	gen := e.ld.events(GenerationEventName)[0].Data
	// Python: identity without datasetId, rowIndex 0... checked through the same function.
	want, _ := eventID(map[string]any{"projectKey": "proj", "evaluationId": "eval-1", "evaluationRunId": "run-1", "runId": "run-1", "rowIndex": 0})
	if gen["eventId"] != want {
		t.Fatalf("eventId over the wrong identity set")
	}
}

func TestCanonicalJSONMatchesPython(t *testing.T) {
	cases := []struct {
		identity map[string]any
		bytes    string
		digest   string
	}{
		{map[string]any{"projectKey": "proj", "evaluationId": "eval-1", "evaluationRunId": "run-1", "runId": "run-1", "rowIndex": 1},
			`{"evaluationId":"eval-1","evaluationRunId":"run-1","projectKey":"proj","rowIndex":1,"runId":"run-1"}`,
			"c8c1bc7d6ea3f69748037db8acbb329b7f833f89ae63a2c255764995f35a8034"},
		{map[string]any{"projectKey": "p<>&é東\u2028\u2029😀\n\x1f\"\\", "evaluationId": "e", "evaluationRunId": "r", "runId": "r", "rowIndex": 2, "criterionType": "Accuracy"},
			`{"criterionType":"Accuracy","evaluationId":"e","evaluationRunId":"r","projectKey":"p<>&\u00e9\u6771\u2028\u2029\ud83d\ude00\n\u001f\"\\","rowIndex":2,"runId":"r"}`,
			"e3b0d69cda28c5fc210186258e383ceea1cd18b5f9be3ba2195140a3ef453ff5"},
	}
	for _, tc := range cases {
		b, err := canonicalJSON(tc.identity)
		if err != nil || string(b) != tc.bytes {
			t.Fatalf("canonical\n got %s\nwant %s", b, tc.bytes)
		}
		sum := sha256.Sum256(b)
		if id, _ := eventID(tc.identity); id != tc.digest || hex.EncodeToString(sum[:]) != tc.digest {
			t.Fatalf("digest %s", id)
		}
	}
}

func TestLastFlushPrecedesFirstSummaryPoll(t *testing.T) {
	for _, withCriteria := range []bool{false, true} {
		e := newEnv(t, 3)
		var pollsAtFlush []int
		e.ld.onFlush = func() { pollsAtFlush = append(pollsAtFlush, e.count("/summary")) }
		args := baseArgs(handler("OpenAI", aiserver.ModeMessages, nil))
		if withCriteria {
			args.Criteria = []Criterion{mustScorer(t, "s", constScorer(1), nil)}
		}
		mustRun(t, e.module(), args)
		want := 1
		if withCriteria {
			want = 2 // phase boundary + final
		}
		if len(pollsAtFlush) != want || slices.Max(pollsAtFlush) != 0 {
			t.Fatalf("criteria=%v: flushes saw summary polls %v", withCriteria, pollsAtFlush)
		}
		log := e.logSnapshot()
		if slices.Index(log, runPath) > slices.Index(log, "flush") {
			t.Fatal("flush before run creation")
		}
	}
}

func TestFlushBound(t *testing.T) {
	e := newEnv(t, 5)
	m := e.module()
	m.flushBound = 2
	args := baseArgs(handler("OpenAI", aiserver.ModeMessages, nil))
	args.Criteria = []Criterion{mustScorer(t, "s", constScorer(1), nil)}
	mustRun(t, m, args)
	// 5 generation events: flush at 2, 4, then boundary; 5 criterion
	// events: flush at 2, 4, then final.
	if e.ld.flushes != 6 {
		t.Fatalf("flushes = %d", e.ld.flushes)
	}
	log := e.logSnapshot()
	since := 0
	for _, s := range log {
		switch {
		case strings.HasPrefix(s, "track "):
			if since++; since > 2 {
				t.Fatal("more than bound events between flushes")
			}
		case s == "flush":
			since = 0
		}
	}
}

func TestCriterionEventFailureIsIsolated(t *testing.T) {
	e := newEnv(t, 3)
	e.ld.trackErr = func(event string, d map[string]any) error {
		if event == CriterionEventName && d["rowIndex"] == 0.0 {
			return errors.New("track exploded")
		}
		return nil
	}
	args := baseArgs(handler("OpenAI", aiserver.ModeMessages, nil))
	args.Criteria = []Criterion{mustScorer(t, "s", constScorer(1), nil)}
	mustRun(t, e.module(), args)
	if len(e.ld.events(CriterionEventName)) != 2 || len(e.ld.events(GenerationEventName)) != 3 {
		t.Fatalf("tracked %d criterion, %d generation", len(e.ld.events(CriterionEventName)), len(e.ld.events(GenerationEventName)))
	}
	if e.ld.flushes != 2 || e.count("/summary") != 1 || !strings.Contains(e.logs.String(), "track exploded") {
		t.Fatalf("flushes %d, polls %d, logs %s", e.ld.flushes, e.count("/summary"), e.logs)
	}
}

func TestPanickingTrackIsIsolated(t *testing.T) {
	e := newEnv(t, 2)
	e.ld.trackErr = func(event string, d map[string]any) error {
		if d["rowIndex"] == 0.0 {
			panic("caller code")
		}
		return nil
	}
	mustRun(t, e.module(), baseArgs(handler("OpenAI", aiserver.ModeMessages, nil)))
	if len(e.ld.events(GenerationEventName)) != 1 {
		t.Fatal("the other row's event must still be tracked")
	}
}

func TestEvaluationPhaseFailureStillFlushes(t *testing.T) {
	e := newEnv(t, 2)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	args := baseArgs(handler("OpenAI", aiserver.ModeMessages, nil))
	args.Concurrency = 1
	args.Criteria = []Criterion{mustScorer(t, "s", func(context.Context, DatasetRow, any) (float64, error) {
		cancel()
		return 1, nil
	}, nil)}
	_, err := e.module().Run(ctx, args)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v", err)
	}
	if len(e.ld.events(GenerationEventName)) != 2 || e.ld.flushes < 2 {
		t.Fatalf("generation events %d, flushes %d", len(e.ld.events(GenerationEventName)), e.ld.flushes)
	}
	if log := e.logSnapshot(); log[len(log)-1] != "flush" {
		t.Fatalf("final flush missing: %v", log)
	}
}

func TestTimestampsAreUTCZ(t *testing.T) {
	if s := timestamp(time.Date(2026, 1, 2, 3, 4, 5, 6000, time.FixedZone("x", 3600))); s != "2026-01-02T02:04:05.000006Z" {
		t.Fatal(s)
	}
}

// waitingLD exposes FlushAndWait, as *ld.LDClient does.
type waitingLD struct {
	*fakeLD
	waits []time.Duration
}

func (w *waitingLD) FlushAndWait(d time.Duration) bool {
	w.waits = append(w.waits, d)
	w.Flush()
	return true
}

func TestBlockingFlushIsPreferred(t *testing.T) {
	e := newEnv(t, 1)
	w := &waitingLD{fakeLD: e.ld}
	getClient = func() (aiserver.LDClient, error) { return w, nil }
	mustRun(t, e.module(), baseArgs(handler("OpenAI", aiserver.ModeMessages, nil)))
	if len(w.waits) != 1 || w.waits[0] != flushWait || e.ld.flushes != 1 {
		t.Fatalf("waits %v flushes %d", w.waits, e.ld.flushes)
	}
}

// A typed nil (here a nil map) marshals to null, so it must be dropped like
// an untyped nil: output appears only when non-null.
func TestTypedNilOutputIsOmitted(t *testing.T) {
	e := newEnv(t, 1)
	mustRun(t, e.module(), baseArgs(handler("OpenAI", aiserver.ModeMessages, func(context.Context, aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
		return aiserver.HandlerResult{Output: map[string]any(nil)}, nil
	})))
	d := e.ld.events(GenerationEventName)[0].Data
	if _, has := d["output"]; has || d["status"] != "COMPLETE" {
		t.Fatalf("typed nil output serialized: %v", d)
	}
}
