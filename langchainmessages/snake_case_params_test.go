package langchainmessages

import (
	"testing"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/llms"
)

// model.parameters are read in snake_case only: a camelCase
// spelling of a table key is not in the table, so it forwards nothing.
func TestModelParametersSnakeCaseOnly(t *testing.T) {
	maxTokens := func(o llms.CallOptions) bool { return o.MaxTokens == 5 }
	topP := func(o llms.CallOptions) bool { return o.TopP != nil && *o.TopP == 0.5 }
	for _, tc := range []struct {
		params  map[string]any
		forward bool
		set     func(llms.CallOptions) bool
	}{
		{map[string]any{"maxTokens": 5}, false, maxTokens},
		{map[string]any{"max_tokens": 5}, true, maxTokens},
		{map[string]any{"topP": 0.5}, false, topP},
		{map[string]any{"top_p": 0.5}, true, topP},
	} {
		h, f, _ := setup(t, textTurn("ok", 1, 1))
		c := cfg(func(c *aiserver.AiConfig) { c.Model.Parameters = tc.params })
		if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: c, UserInput: "q"}); err != nil {
			t.Fatal(err)
		}
		if o := f.calls[0].opts; tc.set(o) != tc.forward {
			t.Errorf("%v: forwarded = %v, want %v (call options %+v)", tc.params, !tc.forward, tc.forward, o)
		}
	}
}
