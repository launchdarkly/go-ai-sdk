package openaiwire

import (
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/openai/openai-go/v3/shared"
)

func TestModelParametersReasoning(t *testing.T) {
	for _, tc := range []struct {
		name string
		in   any
		want any // nil: absent
	}{
		{"effort and summary kept", map[string]any{"effort": "low", "summary": "auto"}, map[string]any{"effort": "low", "summary": "auto"}},
		{"context, mode and generate_summary dropped",
			map[string]any{"effort": "high", "context": "x", "mode": "x", "generate_summary": "auto", "unknown": 1.0},
			map[string]any{"effort": "high"}},
		{"summary alone kept", map[string]any{"summary": "concise"}, map[string]any{"summary": "concise"}},
		{"neither set", map[string]any{"context": "x", "mode": "x"}, nil},
		{"null sub-keys", map[string]any{"effort": nil, "summary": nil}, nil},
		{"not an object", "low", nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := ModelParameters(map[string]any{"reasoning": tc.in}, []string{"reasoning"})["reasoning"]
			if tc.want == nil && ok {
				t.Errorf("reasoning = %v, want absent", got)
			}
			if tc.want != nil && !reflect.DeepEqual(got, tc.want) {
				t.Errorf("reasoning = %v, want %v", got, tc.want)
			}
		})
	}
}

// reasoning is dropped unless the caller allows it.
func TestModelParametersReasoningNotAllowed(t *testing.T) {
	if got := ModelParameters(map[string]any{"reasoning": map[string]any{"effort": "low"}}, nil); len(got) != 0 {
		t.Errorf("got %v, want nothing", got)
	}
}

// Every ReasoningParam field is forwarded or dropped on purpose, so an
// openai-go bump that adds one fails here until it is decided.
func TestEveryReasoningFieldDecided(t *testing.T) {
	dropped := []string{"context", "generate_summary", "mode"}
	ty := reflect.TypeFor[shared.ReasoningParam]()
	for i := range ty.NumField() {
		k, _, _ := strings.Cut(ty.Field(i).Tag.Get("json"), ",")
		if k != "" && k != "-" && !slices.Contains(reasoningKeys, k) && !slices.Contains(dropped, k) {
			t.Errorf("ReasoningParam.%s (%q) has no forward or drop decision", ty.Field(i).Name, k)
		}
	}
}
