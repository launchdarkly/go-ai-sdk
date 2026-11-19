package litellmagents

import (
	"maps"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/neverforward"
)

var paths = []struct {
	name   string
	stream bool
}{{"invoke", false}, {"stream", true}}

// sent runs one turn with params (a tool is offered, so parallel_tool_calls
// applies) and returns the recorded request.
func sent(t *testing.T, params map[string]any, stream bool) recorded {
	t.Helper()
	resp := textResp("ok", 1, 1)
	if stream {
		resp = sse([]string{"ok"}, 1, 1)
	}
	h, f, _ := setup(t, resp)
	req := aiserver.HandlerRequest{Config: cfg(func(c *aiserver.AiConfig) {
		c.Instructions = "sys"
		c.Model.Parameters = params
		c.Tools = map[string]aiserver.Tool{"a": funcTool("")}
	}), UserInput: "q", ToolHandlers: map[string]aiserver.ToolHandler{"a": tool(func(map[string]any) (any, error) { return "", nil })}}
	if stream {
		if _, err := collect(t, h, req); err != nil {
			t.Fatal(err)
		}
	} else {
		invoke(t, h, req)
	}
	if f.count() == 0 {
		t.Fatal("no request sent")
	}
	return f.rec(0)
}

// added is the set of top-level request keys params added to the request.
func added(t *testing.T, params map[string]any, stream bool) map[string]any {
	t.Helper()
	base := sent(t, nil, stream).body
	got := sent(t, params, stream).body
	out := map[string]any{}
	for k, v := range got {
		if _, ok := base[k]; !ok {
			out[k] = v
		}
	}
	return out
}

// The request gains exactly the LiteLLM Agents allowlist: reasoning keeps
// effort only (as reasoning_effort), verbosity is top-level, max_turns caps
// the run, and store, metadata, include_usage, top_logprobs and every
// unlisted key are dropped. Same on both paths.
func TestAllowlistForwarded(t *testing.T) {
	params := map[string]any{
		"frequency_penalty": 0.1, "max_tokens": 100.0, "parallel_tool_calls": false, "presence_penalty": 0.2,
		"temperature": 0.5, "tool_choice": "required", "top_p": 0.9,
		"reasoning": map[string]any{"effort": "low", "summary": "auto", "smuggled": true}, "verbosity": "low", "max_turns": 3.0,
		"store": true, "metadata": map[string]any{"team": "a"}, "include_usage": true, "top_logprobs": 2.0,
		"seed": 7.0, "stop": []any{"x"}, "logprobs": true, "max_completion_tokens": 50.0, "n": 2.0, "service_tier": "flex",
	}
	want := map[string]any{
		"frequency_penalty": 0.1, "max_tokens": 100.0, "parallel_tool_calls": false, "presence_penalty": 0.2,
		"temperature": 0.5, "tool_choice": "required", "top_p": 0.9, "reasoning_effort": "low", "verbosity": "low",
	}
	for _, p := range paths {
		t.Run(p.name, func(t *testing.T) {
			if got := added(t, params, p.stream); !reflect.DeepEqual(got, want) {
				t.Errorf("added keys = %v\nwant %v", got, want)
			}
		})
	}
}

// camelCase spellings are read; snake_case wins; text.verbosity wins over
// verbosity; a non-string tool_choice is dropped.
func TestParameterSpellings(t *testing.T) {
	cases := []struct {
		name   string
		params map[string]any
		want   map[string]any
	}{
		{"camelCase", map[string]any{"topP": 0.5, "maxTokens": 40.0, "parallelToolCalls": false, "frequencyPenalty": 0.3, "presencePenalty": 0.4},
			map[string]any{"top_p": 0.5, "max_tokens": 40.0, "parallel_tool_calls": false, "frequency_penalty": 0.3, "presence_penalty": 0.4}},
		{"snake_case wins", map[string]any{"toolChoice": "none", "tool_choice": "auto", "topP": 0.1, "top_p": 0.2},
			map[string]any{"tool_choice": "auto", "top_p": 0.2}},
		{"text.verbosity wins", map[string]any{"verbosity": "low", "text": map[string]any{"verbosity": "high", "format": "evil"}},
			map[string]any{"verbosity": "high"}},
		{"object tool_choice dropped", map[string]any{"tool_choice": map[string]any{"type": "function", "function": map[string]any{"name": "a"}}, "temperature": 0.2},
			map[string]any{"temperature": 0.2}},
		{"camelCase maxTurns caps the run, not forwarded", map[string]any{"maxTurns": 4.0, "temperature": 0.2},
			map[string]any{"temperature": 0.2}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := added(t, tc.params, false); !reflect.DeepEqual(got, tc.want) {
				t.Errorf("added = %v, want %v", got, tc.want)
			}
		})
	}
}

// reasoning without a usable effort sends nothing.
func TestUnusableReasoningDropped(t *testing.T) {
	for _, r := range []any{"high", map[string]any{"summary": "auto"}, map[string]any{"smuggled": 1.0}, map[string]any{"effort": 3.0}, nil} {
		if got := added(t, map[string]any{"reasoning": r}, false); len(got) != 0 {
			t.Errorf("reasoning %v added %v", r, got)
		}
	}
}

// LiteLLM's own spellings of the never-forwarded categories (endpoints,
// routing, connection, credentials, data location, mock replies, live objects).
var litellmSpellings = map[string]any{
	"api_base": neverforward.Marker, "custom_llm_provider": neverforward.Marker, "api_version": neverforward.Marker,
	"deployment_id": neverforward.Marker, "model_list": []any{neverforward.Marker}, "fallbacks": []any{neverforward.Marker},
	"num_retries": neverforward.Marker, "vertex_credentials": neverforward.Marker, "aws_region_name": neverforward.Marker,
	"mock_response": neverforward.Marker, "shared_session": neverforward.Marker,
}

// No never-forwarded key, in any category or LiteLLM spelling, reaches the
// request on either path, renamed or nested.
func TestNeverForwardedParameters(t *testing.T) {
	params := maps.Clone(neverforward.Params)
	maps.Copy(params, litellmSpellings)
	for _, p := range paths {
		t.Run(p.name, func(t *testing.T) {
			r := sent(t, params, p.stream)
			if strings.Contains(r.raw, neverforward.Marker) {
				t.Errorf("a never-forwarded value reached the request: %s", r.raw)
			}
			for _, k := range slices.Sorted(maps.Keys(params)) {
				if _, ok := r.body[k]; ok && k != "model" && k != "messages" && k != "tools" && k != "stream" {
					t.Errorf("%s forwarded", k)
				}
			}
		})
	}
}
