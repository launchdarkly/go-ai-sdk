package bedrockmessages

import (
	"reflect"
	"testing"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

// paramsRequest sends one turn with params as model.parameters and returns
// the request body the provider received.
func paramsRequest(t *testing.T, params map[string]any) map[string]any {
	t.Helper()
	h, f, _ := setup(t, Options{}, textReply("ok", 1, 1))
	c := cfg(func(c *aiserver.AiConfig) { c.Model.Parameters = params })
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: c, UserInput: "q"}); err != nil {
		t.Fatal(err)
	}
	return f.reqs()[0].body
}

// model.parameters are read in snake_case only: a camelCase spelling of a
// key is dropped, so the request is the same as with no parameters. The
// snake_case key changes the request when this handler forwards it.
func TestModelParametersSnakeCaseOnly(t *testing.T) {
	base := paramsRequest(t, nil)
	for _, tc := range []struct {
		camel, snake string
		value        any
		forwarded    bool
	}{
		{"maxTokens", "max_tokens", 5.0, true},
		{"topP", "top_p", 0.5, true},
		{"topK", "top_k", 5.0, false},
		{"frequencyPenalty", "frequency_penalty", 0.5, false},
		{"presencePenalty", "presence_penalty", 0.5, false},
		{"stopSequences", "stop_sequences", []any{"END"}, true},
	} {
		if got := paramsRequest(t, map[string]any{tc.camel: tc.value}); !reflect.DeepEqual(got, base) {
			t.Errorf("{%s: %v} changed the request: %v", tc.camel, tc.value, got)
		}
		if got := paramsRequest(t, map[string]any{tc.snake: tc.value}); reflect.DeepEqual(got, base) == tc.forwarded {
			t.Errorf("{%s: %v}: forwarded = %v, want %v", tc.snake, tc.value, !tc.forwarded, tc.forwarded)
		}
	}
}
