package judging

import (
	"errors"
	"math"
	"testing"
)

func TestBuildMessageHistory(t *testing.T) {
	if got := BuildMessageHistory("in", "traj", "out"); got != "in\n\ntraj\n\nout\n\n"+FormattingInstructions {
		t.Errorf("got %q", got)
	}
	if got := BuildMessageHistory("", "", "out"); got != "out\n\n"+FormattingInstructions {
		t.Errorf("empty parts not skipped: %q", got)
	}
}

func TestParseVerdict(t *testing.T) {
	score, reason, err := ParseVerdict("```json\n{\"score\":0.5,\"reasoning\":\"ok\"}\n```")
	if err != nil || score != 0.5 || reason != "ok" {
		t.Errorf("fenced: %v %q %v", score, reason, err)
	}
	if _, r, _ := ParseVerdict(map[string]any{"score": 1.0, "reason": "alias"}); r != "alias" {
		t.Errorf("reason alias: %q", r)
	}
	for _, bad := range []any{"prose", "{}", "[1]", 3.0, nil} {
		if _, _, err := ParseVerdict(bad); !errors.Is(err, ErrInvalidVerdict) {
			t.Errorf("%v: err %v", bad, err)
		}
	}
}

func TestNumericScore(t *testing.T) {
	if f, ok := NumericScore(0.8); !ok || f != 0.8 {
		t.Error("finite rejected")
	}
	for _, bad := range []any{nil, "0.9", true, map[string]any{}, math.NaN(), math.Inf(1)} {
		if _, ok := NumericScore(bad); ok {
			t.Errorf("%v accepted", bad)
		}
	}
}

// Reasoning is judged empty after stringifying.
func TestParseVerdictStringifiesReasoning(t *testing.T) {
	for in, want := range map[string]string{
		`{"score":0.5,"reasoning":42}`:                 "42",
		`{"score":0.5,"reasoning":{"why":"ok"}}`:       `{"why":"ok"}`,
		`{"score":0.5,"reasoning":null,"reason":true}`: "true",
		`{"score":0.5,"reasoning":""}`:                 "",
	} {
		if _, r, err := ParseVerdict(in); err != nil || r != want {
			t.Errorf("%s: %q %v", in, r, err)
		}
	}
}
