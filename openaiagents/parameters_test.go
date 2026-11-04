package openaiagents

import (
	"encoding/json"
	"maps"
	"reflect"
	"strings"
	"testing"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/neverforward"
)

// request runs one turn with params through Invoke or Stream and returns the
// request body, without the stream flag that tells the two apart.
func request(t *testing.T, params map[string]any, stream bool) map[string]any {
	t.Helper()
	resp := textResp("ok", 1, 1)
	if stream {
		resp = sse([]string{"ok"}, response("r1", textOutput("ok"), 1, 0, 1))
	}
	h, f, _ := setup(t, resp)
	req := aiserver.HandlerRequest{Config: cfg(func(c *aiserver.AiConfig) { c.Instructions = "sys"; c.Model.Parameters = params }), UserInput: "q"}
	if stream {
		if _, err := collect(t, h, req); err != nil {
			t.Fatal(err)
		}
	} else {
		invoke(t, h, req)
	}
	body := f.request(0)
	delete(body, "stream")
	return body
}

var paths = []struct {
	name   string
	stream bool
}{{"invoke", false}, {"stream", true}}

// None of the never-forwarded keys (store, truncation, context_management and
// prompt_cache_retention among them) reaches the request, and both paths
// send the same request.
func TestNeverForwardedParameters(t *testing.T) {
	params := maps.Clone(neverforward.Params)
	params["temperature"] = 0.3
	var bodies []map[string]any
	for _, p := range paths {
		body := request(t, params, p.stream)
		raw, _ := json.Marshal(body)
		if strings.Contains(string(raw), neverforward.Marker) {
			t.Errorf("%s: a never-forwarded value reached the request: %s", p.name, raw)
		}
		for _, k := range []string{"store", "truncation", "context_management", "prompt_cache_retention"} {
			if _, ok := body[k]; ok {
				t.Errorf("%s: %s forwarded", p.name, k)
			}
		}
		if body["temperature"] != 0.3 {
			t.Errorf("%s: temperature = %v", p.name, body["temperature"])
		}
		bodies = append(bodies, body)
	}
	if !reflect.DeepEqual(bodies[0], bodies[1]) {
		t.Errorf("invoke and stream differ:\n%v\n%v", bodies[0], bodies[1])
	}
}

// reasoning keeps effort and summary, and tool_choice is forwarded only as a
// string; anything else is dropped.
func TestNestedParameters(t *testing.T) {
	for _, tc := range []struct {
		name   string
		params map[string]any
		key    string
		want   any // nil: absent
	}{
		{"reasoning sub-keys trimmed", map[string]any{"reasoning": map[string]any{"effort": "low", "summary": "auto", "x": 1.0}},
			"reasoning", map[string]any{"effort": "low", "summary": "auto"}},
		{"reasoning with neither", map[string]any{"reasoning": map[string]any{"x": 1.0}}, "reasoning", nil},
		{"reasoning not an object", map[string]any{"reasoning": "low"}, "reasoning", nil},
		{"tool_choice string", map[string]any{"tool_choice": "required"}, "tool_choice", "required"},
		{"tool_choice object", map[string]any{"tool_choice": map[string]any{"type": "function", "name": "a"}}, "tool_choice", nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, p := range paths {
				got, ok := request(t, tc.params, p.stream)[tc.key]
				if tc.want == nil && ok {
					t.Errorf("%s: %s = %v, want absent", p.name, tc.key, got)
				}
				if tc.want != nil && !reflect.DeepEqual(got, tc.want) {
					t.Errorf("%s: %s = %v, want %v", p.name, tc.key, got, tc.want)
				}
			}
		})
	}
}
