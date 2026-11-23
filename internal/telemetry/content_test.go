package telemetry

import (
	"encoding/json"
	"testing"
)

func TestValueString(t *testing.T) {
	self := map[string]any{}
	self["me"] = self

	cases := []struct {
		name string
		in   any
		want string
	}{
		{"non-ASCII and < kept raw", map[string]any{"city": "Zürich", "q": "a<b"}, `{"city":"Zürich","q":"a<b"}`},
		{"angle brackets and ampersand kept raw", map[string]any{"s": "<a>&"}, `{"s":"<a>&"}`},
		{"string unchanged", "a<b", "a<b"},
		{"nil", nil, ""},
		{"typed nil map", map[string]any(nil), ""},
		{"typed nil slice", []string(nil), ""},
		{"typed nil pointer", (*struct{})(nil), ""},
		{"map cycle falls back to the type name", self, "map[string]interface {}"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := ValueString(tc.in); got != tc.want {
				t.Errorf("ValueString = %q, want %q", got, tc.want)
			}
		})
	}
}

// A cycle behind a value JSON cannot encode must not reach fmt.Sprint, which
// would follow it until the stack overflows.
func TestValueStringCycleBehindUnsupported(t *testing.T) {
	s := make([]any, 2)
	s[0] = func() {}
	s[1] = s
	m := map[string]any{"a": func() {}}
	m["b"] = m
	p := &[]any{nil}
	(*p)[0] = p
	cases := []struct {
		name string
		in   any
		want string
	}{
		{"slice", s, "[]interface {}"},
		{"map", m, "map[string]interface {}"},
		{"pointer", p, "*[]interface {}"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := ValueString(tc.in); got != tc.want {
				t.Errorf("ValueString = %q, want %q", got, tc.want)
			}
		})
	}
}

type nullMarshaler struct{}

func (nullMarshaler) MarshalJSON() ([]byte, error) { return []byte("null"), nil }

func TestValueStringJSONNull(t *testing.T) {
	cases := []struct {
		name string
		in   any
		want string
	}{
		{"raw message null", json.RawMessage("null"), ""},
		{"MarshalJSON null", nullMarshaler{}, ""},
		{"string null stays", "null", "null"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := ValueString(tc.in); got != tc.want {
				t.Errorf("ValueString = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestValueStringLineSeparatorsRaw(t *testing.T) {
	cases := []struct {
		name string
		in   any
		want string
	}{
		{"U+2028 and U+2029 raw", map[string]any{"a": "x\u2028y\u2029z"}, "{\"a\":\"x\u2028y\u2029z\"}"},
		{"literal backslash-u2028 text unchanged", map[string]any{"a": `\u2028`}, `{"a":"\\u2028"}`},
		{"escaped backslash before raw U+2028", map[string]any{"a": "\\\u2028"}, "{\"a\":\"\\\\\u2028\"}"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := ValueString(tc.in); got != tc.want {
				t.Errorf("ValueString = %q, want %q", got, tc.want)
			}
		})
	}
}
