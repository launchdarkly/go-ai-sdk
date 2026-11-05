package openaimessages

import (
	"context"
	"encoding/json"
	"maps"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/openai/openai-go/v3/responses"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/neverforward"
)

// allowed is the forwarded-parameter table, with a valid value for each key.
var allowed = map[string]any{
	"max_output_tokens":     300.0,
	"max_completion_tokens": 200.0,
	"max_tokens":            100.0,
	"max_tool_calls":        3.0,
	"parallel_tool_calls":   false,
	"prompt_cache_key":      "k",
	"reasoning":             map[string]any{"effort": "low", "summary": "auto"},
	"service_tier":          "flex",
	"temperature":           0.5,
	"tool_choice":           "auto",
	"top_logprobs":          2.0,
	"top_p":                 0.9,
}

// request runs one turn with params through Invoke or Stream and returns the
// request body. On Stream it checks the handler kept stream true, then
// removes it so the two paths compare equal.
func request(t *testing.T, params map[string]any, stream bool, edit func(*aiserver.HandlerRequest)) map[string]any {
	t.Helper()
	resp := textResp("ok", 1, 1)
	if stream {
		resp = sse([]string{"ok"}, response("r1", textOutput("ok"), 1, 0, 1))
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
	if stream {
		if body["stream"] != true {
			t.Errorf("stream = %v, want the handler's true", body["stream"])
		}
		delete(body, "stream")
	}
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
	for k, v := range map[string]any{"frequency_penalty": 1.0, "stream": false, "model": "other", "input": "other",
		"tools": []any{}, "text": map[string]any{"verbosity": "low"}, "background": true, "prompt": map[string]any{"id": "p"},
		"access_programs": []any{"x"}, "prompt_cache_options": map[string]any{"x": 1.0}, "unknown_key": 1.0} {
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

// Each known setting reaches the request, max_output_tokens winning the
// rename, and invoke and stream send the same parameters.
func TestKnownParametersForwarded(t *testing.T) {
	want := maps.Clone(allowed)
	delete(want, "max_completion_tokens")
	delete(want, "max_tokens")
	var bodies []map[string]any
	for _, p := range paths {
		body := request(t, allowed, p.stream, nil)
		for k, v := range want {
			if !reflect.DeepEqual(body[k], v) {
				t.Errorf("%s: %s = %v, want %v", p.name, k, body[k], v)
			}
		}
		for _, k := range []string{"max_completion_tokens", "max_tokens"} {
			if _, ok := body[k]; ok {
				t.Errorf("%s: %s forwarded without renaming", p.name, k)
			}
		}
		bodies = append(bodies, body)
	}
	if !reflect.DeepEqual(bodies[0], bodies[1]) {
		t.Errorf("invoke and stream differ:\n%v\n%v", bodies[0], bodies[1])
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

// reasoning keeps effort and summary; a malformed reasoning or tool_choice is
// dropped, not passed through.
func TestNestedParameters(t *testing.T) {
	for _, tc := range []struct {
		name   string
		params map[string]any
		key    string
		want   any // nil: absent
	}{
		{"reasoning sub-keys trimmed", map[string]any{"reasoning": map[string]any{"effort": "low", "generate_summary": "auto"}},
			"reasoning", map[string]any{"effort": "low"}},
		{"reasoning with neither", map[string]any{"reasoning": map[string]any{"x": 1.0}}, "reasoning", nil},
		{"reasoning not an object", map[string]any{"reasoning": "low"}, "reasoning", nil},
		{"tool_choice any string", map[string]any{"tool_choice": "required"}, "tool_choice", "required"},
		{"tool_choice object", map[string]any{"tool_choice": map[string]any{"type": "function", "name": "a"}},
			"tool_choice", map[string]any{"type": "function", "name": "a"}},
		{"tool_choice object without type", map[string]any{"tool_choice": map[string]any{"name": "a"}}, "tool_choice", nil},
		{"tool_choice number", map[string]any{"tool_choice": 1.0}, "tool_choice", nil},
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

// model, input, instructions and tools come from the handler, not parameters.
func TestHandlerOwnedParametersWin(t *testing.T) {
	params := map[string]any{"model": "other", "input": "other", "instructions": "other",
		"tools": []any{map[string]any{"type": "function", "name": "other"}}}
	offer := func(r *aiserver.HandlerRequest) {
		r.Config.Tools = map[string]aiserver.Tool{"a": {Name: "a", Type: "function", Parameters: map[string]any{"type": "object"}}}
		r.ToolHandlers = map[string]aiserver.ToolHandler{
			"a": aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) { return "ok", nil })}
	}
	for _, p := range paths {
		body := request(t, params, p.stream, offer)
		tools, _ := body["tools"].([]any)
		input, _ := body["input"].([]any)
		_, hasInstructions := body["instructions"]
		turns := inputTurns(body)
		if body["model"] != "test-model" || hasInstructions || len(input) != 2 ||
			turns[0] != (turn{"system", "sys"}) || turns[1] != (turn{"user", "q"}) ||
			len(tools) != 1 || tools[0].(map[string]any)["name"] != "a" {
			t.Errorf("%s: handler-owned fields overridden: %v", p.name, body)
		}
	}
}

// Every ResponseNewParams field has a decision, so an openai-go bump that
// adds one fails here until it is forwarded or excluded.
func TestEveryRequestFieldDecided(t *testing.T) {
	decided := map[string]bool{}
	for _, k := range forwardedParameters {
		decided[k] = true
	}
	for _, k := range []string{
		"input", "model", "previous_response_id", "text", "tools", // handler-owned
		"background", "conversation", "prompt", "stream_options", // would break the call
		"instructions",                                                        // prompt content
		"context_management", "prompt_cache_retention", "store", "truncation", // retention, server-side state
		"include", "moderation", "metadata", "safety_identifier", "user", // output, safety, identity
		"access_programs", "prompt_cache_options", // not generation settings
	} {
		decided[k] = true
	}
	ty := reflect.TypeFor[responses.ResponseNewParams]()
	for i := range ty.NumField() {
		k, _, _ := strings.Cut(ty.Field(i).Tag.Get("json"), ",")
		if k != "" && k != "-" && !decided[k] {
			t.Errorf("ResponseNewParams.%s (%q) has no forward or drop decision", ty.Field(i).Name, k)
		}
	}
}
