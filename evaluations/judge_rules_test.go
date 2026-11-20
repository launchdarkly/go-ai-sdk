package evaluations

import (
	"context"
	"math"
	"strings"
	"testing"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

// The harness owns the {score, reasoning} verdict, so a judge's own
// outputFormat is dropped, with one warning per judge per run.
func TestJudgeOutputFormatDropped(t *testing.T) {
	schema := map[string]any{"type": "object", "properties": map[string]any{"label": map[string]any{"type": "string"}}}
	e, args, rec := judgeEnv(t, 2, verdict(0.7, "ok"), map[string]any{"outputFormat": schema})
	mustRun(t, e.module(), args)
	if rec.n() != 2 {
		t.Fatalf("judge calls %d", rec.n())
	}
	for i, req := range rec.all() {
		if req.Config.OutputFormat != nil {
			t.Errorf("call %d: judge config has OutputFormat %v", i, req.Config.OutputFormat)
		}
		if _, has := req.Config.Raw["outputFormat"]; has {
			t.Errorf("call %d: judge Raw still has outputFormat", i)
		}
	}
	for _, ev := range e.ld.events(criterionEventName) {
		if ev.Data["status"] != "COMPLETE" || ev.Data["score"] != 0.7 {
			t.Errorf("criterion %v", ev.Data)
		}
	}
	n := 0
	for _, line := range strings.Split(e.logs.String(), "\n") {
		if strings.Contains(line, "level=WARN") && strings.Contains(line, "outputFormat") && strings.Contains(line, "acc") {
			n++
		}
	}
	if n != 1 {
		t.Errorf("warnings naming the judge and outputFormat = %d, want 1:\n%s", n, e.logs.String())
	}
}

func TestJudgeEmptyReasoningFallsBackToReason(t *testing.T) {
	for name, output := range map[string]string{
		"empty list":   `{"score":0.6,"reasoning":[],"reason":"fallback"}`,
		"empty object": `{"score":0.6,"reasoning":{},"reason":"fallback"}`,
	} {
		t.Run(name, func(t *testing.T) {
			e, args, _ := judgeEnv(t, 1, func(context.Context, aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
				return aiserver.HandlerResult{Output: output}, nil
			}, nil)
			mustRun(t, e.module(), args)
			if ev := e.ld.events(criterionEventName)[0].Data; ev["status"] != "COMPLETE" || ev["reason"] != "fallback" {
				t.Fatalf("event %v", ev)
			}
		})
	}
}

// invalid_score names the offending value as JSON, never as a Go repr.
func TestInvalidScoreMessageNamesJSON(t *testing.T) {
	g := &genResult{row: renderRow(0, "q", nil, nil, nil), status: statusComplete, output: "o"}
	judgeWith := func(o any) *critResult {
		r := &runState{judges: map[string]*resolvedJudge{"j": {handler: handler("OpenAI", "messages",
			func(context.Context, aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
				return aiserver.HandlerResult{Output: o}, nil
			})}}}
		return r.runJudge(context.Background(), g, mustJudge(t, "j", nil))
	}
	scorer := func(v float64) *critResult {
		return runScorer(context.Background(), g, mustScorer(t, "s", constScorer(v), nil))
	}
	cases := []struct {
		name     string
		got      *critResult
		want     string
		mustNots []string
	}{
		{"judge null", judgeWith(`{"score":null,"reasoning":"r"}`), "got null", []string{"<nil>"}},
		{"judge bool", judgeWith(`{"score":true,"reasoning":"r"}`), "got true", nil},
		{"judge string", judgeWith(`{"score":"high (0.9)","reasoning":"r"}`), `got "high (0.9)"`, nil},
		{"judge out of range", judgeWith(`{"score":1.5,"reasoning":"r"}`), "got 1.5", nil},
		{"judge NaN", judgeWith(map[string]any{"score": math.NaN(), "reasoning": "r"}), "got NaN", nil},
		{"judge +Inf", judgeWith(map[string]any{"score": math.Inf(1), "reasoning": "r"}), "got Infinity", []string{"+Inf"}},
		{"judge -Inf", judgeWith(map[string]any{"score": math.Inf(-1), "reasoning": "r"}), "got -Infinity", []string{"-Inf "}},
		{"scorer NaN", scorer(math.NaN()), "got NaN", nil},
		{"scorer +Inf", scorer(math.Inf(1)), "got Infinity", []string{"+Inf"}},
		{"scorer -Inf", scorer(math.Inf(-1)), "got -Infinity", nil},
		{"scorer out of range", scorer(1.5), "got 1.5", nil},
	}
	for _, tc := range cases {
		if tc.got.cause != causeInvalidScore {
			t.Errorf("%s: cause %q, want %s", tc.name, tc.got.cause, causeInvalidScore)
			continue
		}
		if !strings.HasSuffix(tc.got.errMsg, tc.want) {
			t.Errorf("%s: message %q does not end with %q", tc.name, tc.got.errMsg, tc.want)
		}
		for _, bad := range tc.mustNots {
			if strings.Contains(tc.got.errMsg+" ", bad) {
				t.Errorf("%s: message %q has the Go repr %q", tc.name, tc.got.errMsg, bad)
			}
		}
	}
}
