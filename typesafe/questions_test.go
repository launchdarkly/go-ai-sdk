package typesafe

import (
	"errors"
	"reflect"
	"strings"
	"testing"
)

// Every classifiers entry becomes one question, in order, with its criteria
// kept as the decoded JSON value.
func TestExtractQuestions(t *testing.T) {
	got, err := ExtractQuestions(configOf(t, judgeVariation("jev-latest", classifiers)))
	if err != nil {
		t.Fatal(err)
	}
	want := []Question{
		{Key: "tone", EventKey: "$ld:ai:judge:jev:tone", Type: Noul, Instructions: "Is the tone appropriate?",
			Criteria: map[string]any{"true": "Polite", "false": "Harmful"}},
		{Key: "accuracy", EventKey: "$ld:ai:judge:jev:accuracy", Type: Score, Instructions: "How accurate is the response?",
			Criteria: []any{"Great", "Ok", "Bad"}},
		{Key: "route", EventKey: "$ld:ai:judge:jev:route", Type: Choice, Instructions: "What is the tone?",
			Criteria: map[string]any{"calm": "Calm", "angry": "Angry"}},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("questions =\n%#v\nwant\n%#v", got, want)
	}

	t.Run("noul without criteria", func(t *testing.T) {
		got, err := ExtractQuestions(configOf(t, judgeVariation("jev", []any{
			map[string]any{"key": "k", "eventKey": "ek", "instructions": "i?", "type": "noul"},
		})))
		if err != nil || len(got) != 1 || got[0].Criteria != nil {
			t.Errorf("got %#v, %v; want one noul question with no criteria", got, err)
		}
	})
}

func TestExtractQuestionsRejects(t *testing.T) {
	label := func(edit func(map[string]any)) map[string]any {
		m := map[string]any{"key": "k", "eventKey": "ek", "instructions": "i?", "type": "noul"}
		edit(m)
		return m
	}
	other := map[string]any{"key": "k2", "eventKey": "ek2", "instructions": "j?", "type": "noul"}
	cases := map[string]map[string]any{
		"no classifiers":                {"model": map[string]any{"name": "jev"}, "provider": map[string]any{"name": "TypeSafe"}, "instructions": "unused"},
		"empty classifiers":             judgeVariation("jev", []any{}),
		"missing key":                   judgeVariation("jev", []any{label(func(m map[string]any) { delete(m, "key") })}),
		"empty key":                     judgeVariation("jev", []any{label(func(m map[string]any) { m["key"] = "" })}),
		"missing eventKey":              judgeVariation("jev", []any{label(func(m map[string]any) { delete(m, "eventKey") })}),
		"empty eventKey":                judgeVariation("jev", []any{label(func(m map[string]any) { m["eventKey"] = "" })}),
		"missing instructions":          judgeVariation("jev", []any{label(func(m map[string]any) { delete(m, "instructions") })}),
		"empty instructions":            judgeVariation("jev", []any{label(func(m map[string]any) { m["instructions"] = "" })}),
		"instructions not a string":     judgeVariation("jev", []any{label(func(m map[string]any) { m["instructions"] = map[string]any{"q": "i?"} })}),
		"missing type":                  judgeVariation("jev", []any{label(func(m map[string]any) { delete(m, "type") })}),
		"unsupported type":              judgeVariation("jev", []any{label(func(m map[string]any) { m["type"] = "box" })}),
		"duplicate key":                 judgeVariation("jev", []any{label(func(map[string]any) {}), label(func(m map[string]any) { m["eventKey"] = "ek2" })}),
		"duplicate eventKey":            judgeVariation("jev", []any{label(func(map[string]any) {}), label(func(m map[string]any) { m["key"] = "k2" })}),
		"duplicate after a valid entry": judgeVariation("jev", []any{other, label(func(map[string]any) {}), label(func(m map[string]any) { m["eventKey"] = "ek3" })}),
		"choice without criteria":       judgeVariation("jev", []any{label(func(m map[string]any) { m["type"] = "choice" })}),
		"choice with empty criteria": judgeVariation("jev", []any{label(func(m map[string]any) {
			m["type"], m["criteria"] = "choice", map[string]any{}
		})}),
		"choice criteria a list": judgeVariation("jev", []any{label(func(m map[string]any) {
			m["type"], m["criteria"] = "choice", []any{"calm", "angry"}
		})}),
		"score without criteria": judgeVariation("jev", []any{label(func(m map[string]any) { m["type"] = "score" })}),
		"score with one level": judgeVariation("jev", []any{label(func(m map[string]any) {
			m["type"], m["criteria"] = "score", []any{"only"}
		})}),
		"score criteria an object": judgeVariation("jev", []any{label(func(m map[string]any) {
			m["type"], m["criteria"] = "score", map[string]any{"0": "a", "1": "b"}
		})}),
	}
	for name, variation := range cases {
		t.Run(name, func(t *testing.T) {
			got, err := ExtractQuestions(configOf(t, variation))
			if got != nil || !errors.Is(err, ErrInvalidClassifiers) || !strings.Contains(err.Error(), "classifiers") {
				t.Errorf("got %#v, %v; want nil and an ErrInvalidClassifiers error that mentions classifiers", got, err)
			}
		})
	}
}
