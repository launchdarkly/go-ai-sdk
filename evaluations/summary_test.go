package evaluations

import (
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

const summaryRoute = "projects/proj/evaluations/eval-1/runs/run-1/summary"

func TestSummaryParsing(t *testing.T) {
	cases := []struct {
		name string
		body any
		want RunSummary
		term bool
	}{
		{"statusCounts, no state", map[string]any{"statusCounts": map[string]any{"total": 3, "passed": 2, "failed": 1}},
			RunSummary{TotalRows: 3, PassedRows: 2, FailedRows: 1}, true},
		{"top-level keys", map[string]any{"total_rows": 2, "error_rows": 2}, RunSummary{TotalRows: 2, ErrorRows: 2}, true},
		{"state cannot make it terminal", map[string]any{"state": "COMPLETED", "total": 2, "passed": 1, "pending": 1},
			RunSummary{TotalRows: 2, PassedRows: 1, PendingRows: 1}, false},
		{"empty object", map[string]any{}, RunSummary{}, false},
		{"empty body", nil, RunSummary{}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s, err := parseSummary(roundTrip(tc.body))
			tc.want.Criteria = []CriterionSummary{}
			if err != nil || !reflect.DeepEqual(s, tc.want) || s.terminal() != tc.term {
				t.Fatalf("summary %+v terminal=%v err=%v", s, s.terminal(), err)
			}
		})
	}
	if _, err := parseSummary([]any{}); err == nil {
		t.Fatal("a non-object summary must fail")
	}
}

func TestSummaryPolling(t *testing.T) {
	t.Run("realistic payload needs one poll", func(t *testing.T) {
		e := newEnv(t, 2)
		mustRun(t, e.module(), baseArgs(handler("OpenAI", aiserver.ModeMessages, nil)))
		if e.count("/summary") != 1 {
			t.Fatal(e.sequence())
		}
	})
	t.Run("pending then terminal polls twice", func(t *testing.T) {
		e := newEnv(t, 2)
		e.route("GET", summaryRoute, resp{body: map[string]any{"statusCounts": map[string]any{"total": 2, "pending": 2}}},
			resp{body: map[string]any{"statusCounts": map[string]any{"total": 2, "passed": 2}}})
		mustRun(t, e.module(), baseArgs(handler("OpenAI", aiserver.ModeMessages, nil)))
		if e.count("/summary") != 2 {
			t.Fatal(e.sequence())
		}
	})
	t.Run("empty body keeps polling", func(t *testing.T) {
		e := newEnv(t, 1)
		e.route("GET", summaryRoute, resp{body: map[string]any{}}, resp{body: map[string]any{"total": 1, "passed": 1}})
		mustRun(t, e.module(), baseArgs(handler("OpenAI", aiserver.ModeMessages, nil)))
		if e.count("/summary") != 2 {
			t.Fatal(e.sequence())
		}
	})
	t.Run("run-status endpoint is not used without a pass rate threshold", func(t *testing.T) {
		e := newEnv(t, 1)
		e.route("GET", summaryRoute, resp{body: map[string]any{"statusCounts": map[string]any{"total": 1, "passed": 1},
			"criterionSummaries": []any{map[string]any{"criterionType": "s", "passed": 1}}}})
		args := baseArgs(handler("OpenAI", aiserver.ModeMessages, nil))
		args.Criteria = []Criterion{mustScorer(t, "s", constScorer(1), nil)}
		mustRun(t, e.module(), args)
		if e.runReads() != 0 {
			t.Fatal(e.sequence())
		}
	})
}

func TestSummaryTimeout(t *testing.T) {
	cases := []struct {
		name     string
		counts   map[string]any
		criteria bool
		want     []string
	}{
		{"nothing arrived", map[string]any{"total": 3, "pending": 3}, false,
			[]string{"total_rows=3", "accounted_rows=0", "pending_rows=3", "Undelivered result events", "No rows were accounted", "event transport"}},
		{"some rows short, with criteria", map[string]any{"total": 3, "passed": 2, "pending": 1}, true,
			[]string{"accounted_rows=2", "Some rows are short", "1 criteria"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			e := newEnv(t, 3)
			e.route("GET", summaryRoute, resp{body: map[string]any{"statusCounts": tc.counts}})
			args := baseArgs(handler("OpenAI", aiserver.ModeMessages, nil))
			args.PollTimeout = time.Nanosecond
			if tc.criteria {
				args.Criteria = []Criterion{mustScorer(t, "s", constScorer(1), nil)}
			}
			runErr(t, e.module(), args, ErrPollTimeout, tc.want...)
		})
	}
}

func TestPollDefaults(t *testing.T) {
	e := newEnv(t, 1)
	defaultPollInterval, defaultPollTimeout = 2*time.Second, 3*time.Minute
	r, err := e.module().prepare(baseArgs(handler("OpenAI", aiserver.ModeMessages, nil)))
	if err != nil {
		t.Fatal(err)
	}
	if r.interval != 2*time.Second || r.timeout != time.Second {
		t.Fatalf("interval %v timeout %v", r.interval, r.timeout)
	}
	args := baseArgs(handler("OpenAI", aiserver.ModeMessages, nil))
	args.PollTimeout = 0
	if r, _ = e.module().prepare(args); r.timeout != 3*time.Minute {
		t.Fatalf("default timeout %v", r.timeout)
	}
}

func TestRunResultPassed(t *testing.T) {
	cases := []struct {
		name   string
		counts map[string]any
		passed bool
	}{
		{"one error row", map[string]any{"total": 2, "passed": 1, "error": 1}, false},
		{"one failed row", map[string]any{"total": 2, "passed": 1, "failed": 1}, false},
		{"all passed", map[string]any{"total": 2, "passed": 2}, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			e := newEnv(t, 2)
			e.route("GET", summaryRoute, resp{body: map[string]any{"statusCounts": tc.counts}})
			res := mustRun(t, e.module(), baseArgs(handler("OpenAI", aiserver.ModeMessages, nil)))
			if res.Passed != tc.passed || res.RunID != "run-1" || res.Summary.TotalRows != 2 {
				t.Fatalf("result %+v", res)
			}
		})
	}
}

func TestCriterionSummaries(t *testing.T) {
	t.Run("entries parsed in order, non-objects skipped", func(t *testing.T) {
		s, err := parseSummary(roundTrip(map[string]any{
			"statusCounts": map[string]any{"total": 10, "passed": 9, "failed": 1},
			"criterionSummaries": []any{
				map[string]any{"criterionType": "judge-a", "criterionPassRateThreshold": 0.8, "passed": 9, "failed": 1,
					"error": 0, "passRate": 0.9, "passRateVerdict": "pass"},
				map[string]any{"criterionType": "scorer-b", "passed": 4},
				"not an object",
			},
		}))
		if err != nil {
			t.Fatal(err)
		}
		want := []CriterionSummary{
			{CriterionType: "judge-a", PassRateThreshold: ptr(0.8), PassedRows: 9, FailedRows: 1, PassRate: ptr(0.9), PassRateVerdict: EvalVerdictPass},
			{CriterionType: "scorer-b", PassedRows: 4},
		}
		if !reflect.DeepEqual(s.Criteria, want) {
			t.Fatalf("got %+v", s.Criteria)
		}
		if b := s.Criteria[1]; b.PassRateThreshold != nil || b.PassRate != nil || b.PassRateVerdict != "" {
			t.Fatalf("second entry must have null threshold, pass rate and verdict: %+v", b)
		}
	})
	for name, body := range map[string]map[string]any{
		"no key":   {"total": 1, "passed": 1},
		"not list": {"total": 1, "passed": 1, "criterionSummaries": map[string]any{}},
	} {
		t.Run(name+" gives an empty list", func(t *testing.T) {
			s, err := parseSummary(roundTrip(body))
			if err != nil || s.Criteria == nil || len(s.Criteria) != 0 || !s.terminal() {
				t.Fatalf("%+v %v", s, err)
			}
		})
	}
	t.Run("unknown verdict is none", func(t *testing.T) {
		s, _ := parseSummary(roundTrip(map[string]any{"criterionSummaries": []any{
			map[string]any{"criterionType": "x", "passRateVerdict": "maybe"}}}))
		if s.Criteria[0].PassRateVerdict != "" {
			t.Fatalf("%+v", s.Criteria[0])
		}
	})
}

const runRoute = "projects/proj/evaluations/eval-1/runs/run-1"

// thresholdSummary is a terminal 10-row summary, 9 passed and 1 failed, with
// one criterion at pass rate threshold 0.8. The row rule alone fails it.
func thresholdSummary() resp {
	return resp{body: map[string]any{
		"statusCounts": map[string]any{"total": 10, "passed": 9, "failed": 1},
		"criterionSummaries": []any{map[string]any{"criterionType": "s", "criterionPassRateThreshold": 0.8,
			"passed": 9, "failed": 1, "passRate": 0.9, "passRateVerdict": "pass"}},
	}}
}

func runRead(state string, verdict any) resp {
	body := map[string]any{"id": "run-1", "evaluationId": "eval-1", "state": state}
	if verdict != nil {
		body["verdict"] = verdict
	}
	return resp{body: body}
}

// verdictEnv routes a thresholded summary and the given run reads, and
// returns args for a run whose criterion sets a pass rate threshold.
func verdictEnv(t *testing.T, summary resp, reads ...resp) (*fakeEnv, RunOptions) {
	t.Helper()
	e := newEnv(t, 10)
	e.route("GET", summaryRoute, summary)
	if len(reads) > 0 {
		e.route("GET", runRoute, reads...)
	}
	args := baseArgs(handler("OpenAI", aiserver.ModeMessages, nil))
	args.Criteria = []Criterion{mustScorer(t, "s", constScorer(1), &ScorerOptions{PassRateThreshold: ptr(0.8)})}
	return e, args
}

func (e *fakeEnv) runReads() int {
	n := 0
	for _, s := range e.sequence() {
		if s == "GET "+runRoute {
			n++
		}
	}
	return n
}

func TestRunResultVerdict(t *testing.T) {
	for _, tc := range []struct {
		verdict EvalVerdict
		passed  bool
	}{{EvalVerdictPass, true}, {EvalVerdictFail, false}} {
		t.Run("passed follows verdict "+string(tc.verdict), func(t *testing.T) {
			e, args := verdictEnv(t, thresholdSummary(), runRead("COMPLETE", string(tc.verdict)))
			res := mustRun(t, e.module(), args)
			if res.Verdict != tc.verdict || res.Passed != tc.passed || e.runReads() != 1 {
				t.Fatalf("result %+v, %v", res, e.sequence())
			}
		})
	}
	t.Run("null verdict falls back to row rule", func(t *testing.T) {
		defer patchVerdictWait(0)()
		e, args := verdictEnv(t, thresholdSummary(), runRead("COMPLETE", nil))
		res := mustRun(t, e.module(), args)
		if res.Verdict != "" || res.Passed {
			t.Fatalf("result %+v", res)
		}
	})
	t.Run("verdict read requests the run with encoded segments", func(t *testing.T) {
		e, args := verdictEnv(t, thresholdSummary(), runRead("COMPLETE", "pass"))
		mustRun(t, e.module(), args)
		seq := e.sequence()
		if seq[len(seq)-1] != "GET "+runRoute || seq[len(seq)-2] != "GET "+summaryRoute {
			t.Fatal(seq)
		}
	})
}

func patchVerdictWait(d time.Duration) func() {
	old := verdictWaitTimeout
	verdictWaitTimeout = d
	return func() { verdictWaitTimeout = old }
}

func TestVerdictDecidedFromSummary(t *testing.T) {
	cases := map[string][]any{
		"empty criterionSummaries": {},
		"every threshold null": {map[string]any{"criterionType": "s", "criterionPassRateThreshold": nil, "passed": 9, "failed": 1},
			map[string]any{"criterionType": "t", "passed": 10}},
	}
	for name, entries := range cases {
		t.Run(name, func(t *testing.T) {
			summary := resp{body: map[string]any{"statusCounts": map[string]any{"total": 10, "passed": 9, "failed": 1},
				"criterionSummaries": entries}}
			e, args := verdictEnv(t, summary)
			res := mustRun(t, e.module(), args)
			if e.runReads() != 0 || res.Verdict != "" || res.Passed {
				t.Fatalf("result %+v, %v", res, e.sequence())
			}
		})
	}
}

func TestVerdictWait(t *testing.T) {
	t.Run("lag", func(t *testing.T) {
		e, args := verdictEnv(t, thresholdSummary(),
			runRead("EVALUATED", nil), runRead("COMPLETE", nil), runRead("COMPLETE", "pass"))
		res := mustRun(t, e.module(), args)
		if e.runReads() != 3 || res.Verdict != EvalVerdictPass || !res.Passed {
			t.Fatalf("result %+v, %v", res, e.sequence())
		}
	})
	t.Run("deadline", func(t *testing.T) {
		defer patchVerdictWait(0)()
		e, args := verdictEnv(t, thresholdSummary(), runRead("COMPLETE", nil))
		res := mustRun(t, e.module(), args)
		logs := e.logs.String()
		if e.runReads() != 1 || res.Verdict != "" || res.Passed ||
			!strings.Contains(logs, "no verdict was recorded in time") || !strings.Contains(logs, "runId=run-1") ||
			!strings.Contains(logs, "state=COMPLETE") || !strings.Contains(logs, "deadline=0s") {
			t.Fatalf("result %+v, %v\n%s", res, e.sequence(), logs)
		}
	})
	for _, state := range []string{"CANCELLED", "PERMANENT_ERROR"} {
		t.Run("unjudged state "+state, func(t *testing.T) {
			e, args := verdictEnv(t, thresholdSummary(), runRead(state, nil))
			res := mustRun(t, e.module(), args)
			if e.runReads() != 1 || res.Verdict != "" || res.Passed || strings.Contains(e.logs.String(), "level=WARN") {
				t.Fatalf("result %+v, %v\n%s", res, e.sequence(), e.logs)
			}
		})
	}
	// Neither state is terminal on the server: TEMPORARY_ERROR can still reach
	// COMPLETE, and STALLED resumes to PENDING. Adding either to
	// unjudgedStates would end the wait after one read.
	for _, state := range []string{"TEMPORARY_ERROR", "STALLED"} {
		t.Run("recoverable state "+state, func(t *testing.T) {
			e, args := verdictEnv(t, thresholdSummary(), runRead(state, nil), runRead("COMPLETE", "pass"))
			res := mustRun(t, e.module(), args)
			if e.runReads() != 2 || res.Verdict != EvalVerdictPass {
				t.Fatalf("result %+v, %v", res, e.sequence())
			}
		})
	}
	for name, v := range map[string]any{"maybe": "maybe", "non-string": 1} {
		t.Run("unrecognised verdict "+name, func(t *testing.T) {
			e, args := verdictEnv(t, thresholdSummary(), runRead("COMPLETE", v), runRead("COMPLETE", "pass"))
			res := mustRun(t, e.module(), args)
			logs := e.logs.String()
			if e.runReads() != 1 || res.Verdict != "" || res.Passed ||
				!strings.Contains(logs, "does not recognise") || strings.Contains(logs, "deadline") {
				t.Fatalf("result %+v, %v\n%s", res, e.sequence(), logs)
			}
			if name == "maybe" && !strings.Contains(logs, `maybe`) {
				t.Fatal(logs)
			}
		})
	}
	t.Run("failed read does not fail the run", func(t *testing.T) {
		e, args := verdictEnv(t, thresholdSummary(), resp{status: 403, body: "forbidden"})
		res := mustRun(t, e.module(), args)
		logs := e.logs.String()
		if e.runReads() != 1 || res == nil || res.Verdict != "" || res.Passed ||
			!strings.Contains(logs, "could not read the evaluation run verdict") || !strings.Contains(logs, "runId=run-1") ||
			!strings.Contains(logs, "403") {
			t.Fatalf("result %+v, %v\n%s", res, e.sequence(), logs)
		}
	})
	t.Run("invalid run response is a failed read", func(t *testing.T) {
		e, args := verdictEnv(t, thresholdSummary(), resp{body: map[string]any{"id": "run-1", "evaluationId": "eval-1", "verdict": "pass"}})
		res := mustRun(t, e.module(), args)
		if res.Verdict != "" || !strings.Contains(e.logs.String(), `missing string field \"state\"`) {
			t.Fatalf("result %+v\n%s", res, e.logs)
		}
	})
}

func TestVerdictWaitDefault(t *testing.T) {
	if verdictWaitTimeout != 30*time.Second {
		t.Fatalf("default verdict wait %v", verdictWaitTimeout)
	}
}
