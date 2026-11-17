package evaluations

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

// typesafeJudge serves a classifiers-only judge variation (no instructions,
// no messages) whose handler returns output.
func typesafeJudge(t *testing.T, output string) (*fakeEnv, RunOptions, *recorder) {
	return judgeEnv(t, 1, func(context.Context, aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
		return aiserver.HandlerResult{Output: output}, nil
	}, map[string]any{"messages": []any{}, "classifiers": []any{
		map[string]any{"key": "migration_intent", "eventKey": "ev-intent", "type": "noul", "instructions": "Migrate?"},
		map[string]any{"key": "migration", "eventKey": "ev-migration", "type": "noul", "instructions": "Migrated?"},
	}})
}

// A criterion event is one score: the first label's, with that label's key
// as the reason.
func TestTypesafeJudgeScoresFirstLabel(t *testing.T) {
	e, args, rec := typesafeJudge(t, `{"kind":"typesafe","results":[`+
		`{"key":"migration_intent","eventKey":"ev-intent","score":0.9,"reason":"yes"},`+
		`{"key":"migration","eventKey":"ev-migration","score":0.2,"reason":"no"}]}`)
	mustRun(t, e.module(), args)
	if rec.n() != 1 {
		t.Fatalf("judge calls %d", rec.n())
	}
	if b, _ := json.Marshal(rec.all()[0].Config); !strings.Contains(string(b), `"classifiers"`) {
		t.Errorf("judge handler config has no classifiers: %s", b)
	}
	evs := e.ld.events(criterionEventName)
	if len(evs) != 1 {
		t.Fatalf("criterion events %d, want one per judge", len(evs))
	}
	if ev := evs[0].Data; ev["status"] != "COMPLETE" || ev["score"] != 0.9 || ev["reason"] != "migration_intent" {
		t.Errorf("criterion event %v", ev)
	}
}

func TestMalformedTypesafeJudgeFailsCriterion(t *testing.T) {
	e, args, _ := typesafeJudge(t, `{"kind":"typesafe","results":[{"key":"k","score":1}]}`)
	mustRun(t, e.module(), args)
	// The TypeSafe parser, not the {score, reasoning} one, names the missing field.
	ev := e.ld.events(criterionEventName)[0].Data
	if msg, _ := ev["errorMessage"].(string); ev["status"] != "ERROR" || !strings.Contains(msg, "eventKey") {
		t.Errorf("criterion event %v, want ERROR naming the missing eventKey", ev)
	}
}
