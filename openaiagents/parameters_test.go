package openaiagents

import (
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

// allowed is forwardedParameters plus the keys the runner maps onto the
// request (the token-limit renames and verbosity), with a valid value for each.
var allowed = map[string]any{
	"max_output_tokens":     300.0,
	"max_completion_tokens": 200.0,
	"max_tokens":            100.0,
	"parallel_tool_calls":   false,
	"reasoning":             map[string]any{"effort": "low", "summary": "auto"},
	"temperature":           0.5,
	"tool_choice":           "required",
	"top_p":                 0.9,
	"verbosity":             "low",
}

// request runs one turn with params through Invoke or Stream and returns the
// request body. On Stream it checks the handler kept stream true, then
// removes it so the two paths compare equal.
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
// forwardedParameters plus the renames and verbosity, on both paths.
func TestParameterListMatchesTable(t *testing.T) {
	probe := maps.Clone(neverforward.Params)
	maps.Copy(probe, allowed)
	for k, v := range map[string]any{"frequency_penalty": 1.0, "stream": false, "model": "other", "input": "other",
		"tools": []any{}, "max_tool_calls": 3.0, "prompt_cache_key": "k", "service_tier": "flex", "top_logprobs": 2.0,
		"access_programs": []any{"x"}, "prompt_cache_options": map[string]any{"x": 1.0}, "max_turns": 2.0, "unknown_key": 1.0} {
		probe[k] = v
	}
	want := append([]string{"max_completion_tokens", "max_tokens", "verbosity"}, forwardedParameters...)
	slices.Sort(want)
	if got := slices.Sorted(maps.Keys(allowed)); !slices.Equal(got, want) {
		t.Fatalf("allowed = %v, want forwardedParameters plus renames: %v", got, want)
	}
	for _, p := range paths {
		t.Run(p.name, func(t *testing.T) {
			base := request(t, nil, p.stream)
			var got []string
			for k, v := range probe {
				if !reflect.DeepEqual(request(t, map[string]any{k: v}, p.stream), base) {
					got = append(got, k)
				}
			}
			slices.Sort(got)
			if !slices.Equal(got, want) {
				t.Errorf("forwarded keys = %v\nwant %v", got, want)
			}
		})
	}
}

// Each known setting reaches the request, max_output_tokens winning the
// rename, and invoke and stream send the same parameters.
func TestKnownParametersForwarded(t *testing.T) {
	want := maps.Clone(allowed)
	for _, k := range []string{"max_completion_tokens", "max_tokens", "verbosity"} {
		delete(want, k)
	}
	var bodies []map[string]any
	for _, p := range paths {
		body := request(t, allowed, p.stream)
		for k, v := range want {
			if !reflect.DeepEqual(body[k], v) {
				t.Errorf("%s: %s = %v, want %v", p.name, k, body[k], v)
			}
		}
		if !reflect.DeepEqual(body["text"], map[string]any{"verbosity": "low"}) {
			t.Errorf("%s: text = %v, want verbosity low", p.name, body["text"])
		}
		for _, k := range []string{"max_completion_tokens", "max_tokens", "verbosity"} {
			if _, ok := body[k]; ok {
				t.Errorf("%s: %s forwarded as is", p.name, k)
			}
		}
		bodies = append(bodies, body)
	}
	if !reflect.DeepEqual(bodies[0], bodies[1]) {
		t.Errorf("invoke and stream differ:\n%v\n%v", bodies[0], bodies[1])
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
		"input", "instructions", "model", "previous_response_id", "text", "tools", // handler-owned
		"background", "conversation", "prompt", "stream_options", // would break the run
		"context_management", "prompt_cache_retention", "store", "truncation", // retention, server-side state
		"include", "moderation", "metadata", "safety_identifier", "user", // output, safety, identity
		"access_programs", "prompt_cache_options", // not generation settings
		"max_tool_calls", "prompt_cache_key", "service_tier", "top_logprobs", // not in the Agents table
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
		{"store false", map[string]any{"store": false}, "store", nil},
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
