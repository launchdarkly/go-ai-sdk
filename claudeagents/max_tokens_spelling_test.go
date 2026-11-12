package claudeagents

import (
	"testing"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

// Claude Agents reads both max_tokens and maxTokens; max_tokens wins when
// both are set. (The messages handlers read snake_case only.)
func TestMaxTokensBothSpellings(t *testing.T) {
	for _, tc := range []struct {
		params map[string]any
		want   float64
	}{
		{map[string]any{"maxTokens": 5.0}, 5},
		{map[string]any{"max_tokens": 6.0}, 6},
		{map[string]any{"max_tokens": 6.0, "maxTokens": 5.0}, 6},
	} {
		e := setup(t, nil, textResp("ok", 1, 1))
		c := cfg(func(c *aiserver.AiConfig) { c.Model.Parameters = tc.params })
		if _, err := e.handler().Invoke(bg, aiserver.HandlerRequest{Config: c, UserInput: "q"}); err != nil {
			t.Fatal(err)
		}
		if got := e.api.requests[0]["max_tokens"]; got != tc.want {
			t.Errorf("%v: max_tokens = %v, want %v", tc.params, got, tc.want)
		}
	}
}
