package evaluations

import (
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
			if err != nil || s != tc.want || s.terminal() != tc.term {
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
	t.Run("run-status endpoint is never used", func(t *testing.T) {
		e := newEnv(t, 1)
		mustRun(t, e.module(), baseArgs(handler("OpenAI", aiserver.ModeMessages, nil)))
		for _, s := range e.sequence() {
			if strings.HasSuffix(s, "/runs/run-1") {
				t.Fatal(s)
			}
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
