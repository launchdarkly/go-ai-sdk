package judging

import (
	"errors"
	"math"
	"reflect"
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
		// An empty list or object is empty, so the reason alias is used.
		`{"score":0.5,"reasoning":[],"reason":"fallback"}`: "fallback",
		`{"score":0.5,"reasoning":{},"reason":"fallback"}`: "fallback",
		`{"score":0.5,"reasoning":[]}`:                     "",
		`{"score":0.5,"reasoning":"r","reason":[]}`:        "r",
		`{"score":0.5,"reasoning":["a"]}`:                  `["a"]`,
	} {
		if _, r, err := ParseVerdict(in); err != nil || r != want {
			t.Errorf("%s: %q %v", in, r, err)
		}
	}
}

func TestTypesafeEntries(t *testing.T) {
	entries, err := TypesafeEntries("```json\n" + `{"kind":"typesafe","results":[{"key":"a","eventKey":"ea","score":1,"reason":"yes"},{"key":"b","eventKey":"eb","score":0,"reason":7}]}` + "\n```")
	want := []TypesafeEntry{{Key: "a", EventKey: "ea", Score: 1, Reason: "yes"}, {Key: "b", EventKey: "eb", Score: 0, Reason: ""}}
	if err != nil || !reflect.DeepEqual(entries, want) {
		t.Errorf("entries = %+v, %v", entries, err)
	}
	// Not a TypeSafe payload: nil, nil, so the {score, reasoning} parser runs.
	for _, in := range []any{`{"score":1,"reasoning":"r"}`, `{"kind":"other","results":[]}`, "not json", nil, map[string]any{"score": 1.0}} {
		if e, err := TypesafeEntries(in); e != nil || err != nil {
			t.Errorf("%v: %v %v", in, e, err)
		}
	}
	// A decoded object works too.
	if e, err := TypesafeEntries(map[string]any{"kind": "typesafe", "results": []any{map[string]any{"key": "a", "eventKey": "e", "score": 0.5}}}); err != nil || len(e) != 1 {
		t.Errorf("decoded: %v %v", e, err)
	}
	// A handler's literal map may carry Go integer scores.
	if e, err := TypesafeEntries(map[string]any{"kind": "typesafe", "results": []any{
		map[string]any{"key": "a", "eventKey": "ea", "score": 1}, map[string]any{"key": "b", "eventKey": "eb", "score": uint8(0)},
	}}); err != nil || len(e) != 2 || e[0].Score != 1 || e[1].Score != 0 {
		t.Errorf("integer scores: %v %v", e, err)
	}
	if _, err := TypesafeEntries(map[string]any{"kind": "typesafe", "results": []any{map[string]any{"key": "a", "eventKey": "e", "score": true}}}); err == nil {
		t.Error("a Go bool score was accepted")
	}
	for name, in := range map[string]string{
		"no results":       `{"kind":"typesafe"}`,
		"empty results":    `{"kind":"typesafe","results":[]}`,
		"results not list": `{"kind":"typesafe","results":{}}`,
		"entry not object": `{"kind":"typesafe","results":[1]}`,
		"missing key":      `{"kind":"typesafe","results":[{"eventKey":"e","score":1}]}`,
		"empty key":        `{"kind":"typesafe","results":[{"key":"","eventKey":"e","score":1}]}`,
		"missing score":    `{"kind":"typesafe","results":[{"key":"k","eventKey":"e"}]}`,
		"bool score":       `{"kind":"typesafe","results":[{"key":"k","eventKey":"e","score":true}]}`,
		"missing eventKey": `{"kind":"typesafe","results":[{"key":"k","score":1}]}`,
		"empty eventKey":   `{"kind":"typesafe","results":[{"key":"k","eventKey":"","score":1}]}`,
	} {
		if e, err := TypesafeEntries(in); err == nil || e != nil {
			t.Errorf("%s: %v %v, want an error", name, e, err)
		}
	}
}
