package telemetry

import "testing"

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
