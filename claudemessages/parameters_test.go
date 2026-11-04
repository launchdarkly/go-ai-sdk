package claudemessages

import (
	"context"
	"encoding/json"
	"maps"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/neverforward"
)

// allowed is the forwarded-parameter table, with a valid value for each key.
var allowed = map[string]any{
	"cache_control":  map[string]any{"type": "ephemeral"},
	"effort":         "low",
	"max_tokens":     77.0,
	"output_config":  map[string]any{"effort": "high"},
	"service_tier":   "auto",
	"stop_sequences": []any{"END"},
	"temperature":    0.5,
	"thinking":       map[string]any{"type": "enabled", "budget_tokens": 2048.0},
	"tool_choice":    map[string]any{"type": "auto"},
	"top_k":          5.0,
	"top_p":          0.9,
}

// request runs one turn with params through Invoke or Stream and returns the
// request body, without the stream flag that tells the two apart.
func request(t *testing.T, params map[string]any, stream bool, edit func(*aiserver.HandlerRequest)) map[string]any {
	t.Helper()
	resp := textResp("ok", 1, 1)
	if stream {
		resp = sse([]string{"ok"}, "", 1, 1)
	}
	h, f, _ := setup(t, resp)
	req := aiserver.HandlerRequest{Config: cfg(func(c *aiserver.AiConfig) { c.Instructions = "sys"; c.Model.Parameters = params }), UserInput: "q"}
	if edit != nil {
		edit(&req)
	}
	if stream {
		for _, err := range h.Stream(bg, req) {
			if err != nil {
				t.Fatal(err)
			}
		}
	} else if _, err := h.Invoke(bg, req); err != nil {
		t.Fatal(err)
	}
	body := f.requests[0]
	delete(body, "stream")
	return body
}

var paths = []struct {
	name   string
	stream bool
}{{"invoke", false}, {"stream", true}}

// Setting one key at a time, the keys that change the request are exactly
// the table, on both paths.
func TestParameterListMatchesTable(t *testing.T) {
	probe := maps.Clone(neverforward.Params)
	maps.Copy(probe, allowed)
	for k, v := range map[string]any{"frequency_penalty": 1.0, "stream": false, "model": "other", "messages": []any{},
		"system": "other", "tools": []any{}, "unknown_key": 1.0} {
		probe[k] = v
	}
	for _, p := range paths {
		t.Run(p.name, func(t *testing.T) {
			base := request(t, nil, p.stream, nil)
			var got []string
			for k, v := range probe {
				if !reflect.DeepEqual(request(t, map[string]any{k: v}, p.stream, nil), base) {
					got = append(got, k)
				}
			}
			slices.Sort(got)
			if want := slices.Sorted(maps.Keys(allowed)); !slices.Equal(got, want) {
				t.Errorf("forwarded keys = %v\nwant %v", got, want)
			}
		})
	}
}

// Each known setting reaches the request in its provider shape, a zero
// temperature included, and invoke and stream send the same parameters.
func TestKnownParametersForwarded(t *testing.T) {
	want := map[string]any{
		"cache_control": allowed["cache_control"], "max_tokens": 77.0, "output_config": map[string]any{"effort": "high"},
		"service_tier": "auto", "stop_sequences": []any{"END"}, "temperature": 0.5, "thinking": allowed["thinking"],
		"tool_choice": allowed["tool_choice"], "top_k": 5.0, "top_p": 0.9,
	}
	var bodies []map[string]any
	for _, p := range paths {
		body := request(t, allowed, p.stream, nil)
		for k, v := range want {
			if !reflect.DeepEqual(body[k], v) {
				t.Errorf("%s: %s = %v, want %v", p.name, k, body[k], v)
			}
		}
		if _, ok := body["effort"]; ok {
			t.Errorf("%s: effort forwarded at the top level", p.name)
		}
		bodies = append(bodies, body)
	}
	if !reflect.DeepEqual(bodies[0], bodies[1]) {
		t.Errorf("invoke and stream differ:\n%v\n%v", bodies[0], bodies[1])
	}
	if body := request(t, map[string]any{"temperature": 0.0}, false, nil); body["temperature"] != 0.0 {
		t.Errorf("a zero temperature was dropped: %v", body["temperature"])
	}
}

// None of the never-forwarded keys reaches the request, renamed or nested,
// and both paths send the same request.
func TestNeverForwardedParameters(t *testing.T) {
	params := maps.Clone(neverforward.Params)
	params["temperature"] = 0.3
	var bodies []map[string]any
	for _, p := range paths {
		body := request(t, params, p.stream, nil)
		raw, _ := json.Marshal(body)
		if strings.Contains(string(raw), neverforward.Marker) {
			t.Errorf("%s: a never-forwarded value reached the request: %s", p.name, raw)
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

// Typed objects keep only the sub-keys their type takes; a malformed one is
// dropped, not passed through.
func TestNestedParameters(t *testing.T) {
	for _, tc := range []struct {
		name   string
		params map[string]any
		key    string
		want   any // nil: absent
	}{
		{"thinking without type", map[string]any{"thinking": map[string]any{"budget_tokens": 1024.0}}, "thinking", nil},
		{"thinking between_tools", map[string]any{"thinking": map[string]any{"type": "between_tools"}}, "thinking", nil},
		{"thinking enabled without budget", map[string]any{"thinking": map[string]any{"type": "enabled"}}, "thinking", nil},
		{"thinking not an object", map[string]any{"thinking": "enabled"}, "thinking", nil},
		{"thinking sub-keys trimmed", map[string]any{"thinking": map[string]any{"type": "adaptive", "display": "summarized", "budget_tokens": 1.0, "x": 1.0}},
			"thinking", map[string]any{"type": "adaptive", "display": "summarized"}},
		{"tool_choice tool without name", map[string]any{"tool_choice": map[string]any{"type": "tool"}}, "tool_choice", nil},
		{"tool_choice string", map[string]any{"tool_choice": "auto"}, "tool_choice", nil},
		{"tool_choice unknown type", map[string]any{"tool_choice": map[string]any{"type": "required"}}, "tool_choice", nil},
		{"tool_choice sub-keys trimmed", map[string]any{"tool_choice": map[string]any{"type": "tool", "name": "a", "disable_parallel_tool_use": true, "x": 1.0}},
			"tool_choice", map[string]any{"type": "tool", "name": "a", "disable_parallel_tool_use": true}},
		{"tool_choice none keeps type only", map[string]any{"tool_choice": map[string]any{"type": "none", "disable_parallel_tool_use": true}},
			"tool_choice", map[string]any{"type": "none"}},
		{"cache_control unknown type", map[string]any{"cache_control": map[string]any{"type": "persistent"}}, "cache_control", nil},
		{"cache_control sub-keys trimmed", map[string]any{"cache_control": map[string]any{"type": "ephemeral", "ttl": "1h", "x": 1.0}},
			"cache_control", map[string]any{"type": "ephemeral", "ttl": "1h"}},
		{"effort not a string", map[string]any{"effort": 5.0}, "output_config", nil},
		{"output_config not an object", map[string]any{"output_config": "low"}, "output_config", nil},
		{"output_config effort not a string", map[string]any{"output_config": map[string]any{"effort": 1.0}}, "output_config", nil},
		{"output_config effort wins", map[string]any{"effort": "low", "output_config": map[string]any{"effort": "high"}},
			"output_config", map[string]any{"effort": "high"}},
		{"top-level effort moves", map[string]any{"effort": "low", "output_config": map[string]any{"effort": 1.0}},
			"output_config", map[string]any{"effort": "low"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, p := range paths {
				got, ok := request(t, tc.params, p.stream, nil)[tc.key]
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

// output_config keeps only effort: a format would be a second
// structured-output schema, and the handler owns structured output.
func TestNoSecondStructuredOutputSchema(t *testing.T) {
	params := map[string]any{"output_config": map[string]any{"effort": "low",
		"format": map[string]any{"type": "json_schema", "schema": map[string]any{"type": "object"}}}}
	for _, p := range paths {
		if got := request(t, params, p.stream, nil)["output_config"]; !reflect.DeepEqual(got, map[string]any{"effort": "low"}) {
			t.Errorf("%s: output_config = %v", p.name, got)
		}
	}
}

// model, messages, system and tools come from the handler, not parameters.
func TestHandlerOwnedParametersWin(t *testing.T) {
	params := map[string]any{"model": "other", "messages": []any{"other"}, "system": "other",
		"tools": []any{map[string]any{"name": "other", "input_schema": map[string]any{"type": "object"}}}}
	offer := func(r *aiserver.HandlerRequest) {
		r.Config.Tools = map[string]aiserver.Tool{"a": {Name: "a", Type: "function", Parameters: map[string]any{"type": "object"}}}
		r.ToolHandlers = map[string]aiserver.ToolHandler{
			"a": aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) { return "ok", nil })}
	}
	for _, p := range paths {
		body := request(t, params, p.stream, offer)
		tools, _ := body["tools"].([]any)
		if body["model"] != "test-model" || system(body) != "sys" || lastText(body) != "q" ||
			len(tools) != 1 || tools[0].(map[string]any)["name"] != "a" {
			t.Errorf("%s: handler-owned fields overridden: %v", p.name, body)
		}
	}
}
