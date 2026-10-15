package aiserver

import (
	"reflect"
	"strings"
	"testing"
)

func TestParseTemplate(t *testing.T) {
	vars := map[string]any{
		"name": "world", "greeting": "Hi", "n": 42.0, "ok": true,
		"user": map[string]any{"name": "Alice"},
		"a":    map[string]any{"b": map[string]any{"c": 42.0}},
	}
	cases := map[string]string{
		"Hello {{name}}":        "Hello world",
		"{{greeting}} {{name}}": "Hi world",
		"Hello {{missing}}":     "Hello {{missing}}",
		"Hi {{user.name}}":      "Hi Alice",
		"{{user.missing}}":      "{{user.missing}}",
		"{{a.b.c}}":             "42",
		"{{n}} {{ok}}":          "42 true",
		"no placeholders":       "no placeholders",
	}
	for in, want := range cases {
		if got := ParseTemplate(in, vars); got != want {
			t.Errorf("ParseTemplate(%q) = %q, want %q", in, got, want)
		}
	}
	if got := ParseTemplate("Hello {{name}}", map[string]any{}); got != "Hello {{name}}" {
		t.Errorf("empty vars substituted: %q", got)
	}
}

func TestParseJSONWithPossibleFences(t *testing.T) {
	want := map[string]any{"a": 1.0}
	for _, in := range []string{
		`{"a":1}`,
		"```json\n{\"a\":1}\n```",
		"```\n{\"a\":1}\n```",
		"```json\n\n{\"a\":1}\n\n```",
		"```json\r\n{\"a\":1}\r\n```",
		"```\r\n{\"a\":1}\r\n```",
		"  ```json\n{\"a\":1}\n```",
	} {
		got, ok := ParseJSONWithPossibleFences(in)
		if !ok || !reflect.DeepEqual(got, want) {
			t.Errorf("%q -> %v %v", in, got, ok)
		}
	}
	if _, ok := ParseJSONWithPossibleFences("not json"); ok {
		t.Error("invalid JSON parsed")
	}
	nested := `{"x":[1,{"y":[true,null]}]}`
	if got, ok := ParseJSONWithPossibleFences(nested); !ok || got.(map[string]any)["x"] == nil {
		t.Errorf("nested: %v", got)
	}
}

func TestParseUsage(t *testing.T) {
	cases := []struct {
		in   map[string]any
		want TokenUsage
	}{
		{map[string]any{"input_tokens": 10.0, "output_tokens": 5.0}, TokenUsage{10, 5, 15, nil}},
		{map[string]any{"inputTokens": 10.0, "outputTokens": 5.0}, TokenUsage{10, 5, 15, nil}},
		{map[string]any{"input": 10.0, "output": 5.0, "total": 999.0}, TokenUsage{10, 5, 15, nil}},
		{map[string]any{"foo": 1.0}, TokenUsage{}},
		{map[string]any{"input_tokens": 1.0, "output_tokens": 1.0, "inputTokens": 7.0, "outputTokens": 7.0}, TokenUsage{1, 1, 2, nil}},
	}
	for _, c := range cases {
		if got := ParseUsage(c.in); !reflect.DeepEqual(got, c.want) {
			t.Errorf("ParseUsage(%v) = %+v, want %+v", c.in, got, c.want)
		}
	}
}

func TestNormalizeMode(t *testing.T) {
	for in, want := range map[string]Mode{"agent": ModeAgent, "completion": ModeMessages, "judge": ModeMessages, "": ModeMessages, "zzz": ModeMessages} {
		if got := NormalizeMode(in); got != want {
			t.Errorf("NormalizeMode(%q) = %q", in, got)
		}
	}
}

func base() map[string]any {
	return map[string]any{"model": map[string]any{"name": "m"}, "provider": map[string]any{"name": "p"}, "instructions": "x"}
}

func with(k string, v any) map[string]any {
	m := base()
	m[k] = v
	return m
}

func TestParseAiConfig(t *testing.T) {
	valid := []map[string]any{
		base(),
		{"model": map[string]any{"name": "m"}, "provider": map[string]any{"name": "p"}, "messages": []any{map[string]any{"role": "user", "content": "hi"}}},
		with("tools", map[string]any{"t": map[string]any{"name": "t", "type": "function", "parameters": map[string]any{}}}),
		with("judgeConfiguration", map[string]any{"judges": []any{map[string]any{"key": "j", "samplingRate": 1.0}}}),
		with("outputFormat", map[string]any{"type": "object", "properties": map[string]any{}}),
		with("skills", []any{}),
		with("skills", []any{map[string]any{"key": "pdf-extraction", "version": 2.0}}),
	}
	for i, v := range valid {
		if _, err := ParseAiConfig(v); err != nil {
			t.Errorf("valid[%d]: %v", i, err)
		}
	}
	invalid := []map[string]any{
		{"model": map[string]any{"name": "m"}, "provider": map[string]any{"name": "p"}},
		{"model": map[string]any{"name": "m"}, "provider": map[string]any{"name": "p"}, "messages": []any{}},
		{"model": map[string]any{"name": "m"}, "provider": map[string]any{"name": "p"}, "messages": []any{map[string]any{"role": "admin", "content": "x"}}},
		{"model": map[string]any{}, "provider": map[string]any{"name": "p"}, "instructions": "x"},
		{"model": map[string]any{"name": "m"}, "instructions": "x"},
		with("tools", map[string]any{"t": map[string]any{"name": "t", "type": "class", "parameters": map[string]any{}}}),
	}
	for i, v := range invalid {
		if _, err := ParseAiConfig(v); err == nil {
			t.Errorf("invalid[%d] accepted", i)
		}
	}
	for i, s := range []any{nil, "x", map[string]any{}, 1.0, []any{"pdf"}, []any{map[string]any{"version": 1.0}},
		[]any{map[string]any{"key": 1.0, "version": 1.0}}, []any{map[string]any{"key": "Bad", "version": 1.0}},
		[]any{map[string]any{"key": "-x", "version": 1.0}}, []any{map[string]any{"key": "a/b", "version": 1.0}},
		[]any{map[string]any{"key": "", "version": 1.0}}, []any{map[string]any{"key": strings.Repeat("a", 257), "version": 1.0}},
		[]any{map[string]any{"key": "k"}}, []any{map[string]any{"key": "k", "version": 0.0}}, []any{map[string]any{"key": "k", "version": 2.5}},
		[]any{map[string]any{"key": "k", "version": "2"}}, []any{map[string]any{"key": "k", "version": true}}} {
		_, err := ParseAiConfig(with("skills", s))
		if err == nil || !strings.Contains(err.Error(), "skills") {
			t.Errorf("skills[%d]=%v: err %v", i, s, err)
		}
	}
	cfg, _ := ParseAiConfig(with("skills", []any{map[string]any{"key": "k", "version": 2.0}}))
	if !reflect.DeepEqual(cfg.Raw["skills"], []any{map[string]any{"key": "k", "version": 2.0}}) {
		t.Errorf("skills did not round-trip: %v", cfg.Raw["skills"])
	}
}
