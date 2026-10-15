package claudeagents

import (
	"reflect"
	"testing"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

// The JavaScript FORWARDED_QUERY_OPTION_KEYS, mapped onto the Messages API.
func TestQueryOptionForwarding(t *testing.T) {
	cases := []struct {
		name   string
		params map[string]any
		want   map[string]any // expected body fields; nil value = must be absent
	}{
		{"thinking snake budget", map[string]any{"thinking": map[string]any{"type": "enabled", "budget_tokens": float64(1024), "extra": 1}},
			map[string]any{"thinking": map[string]any{"type": "enabled", "budget_tokens": float64(1024)}}},
		{"thinking camel budget and display", map[string]any{"thinking": map[string]any{"type": "enabled", "budgetTokens": float64(2048), "display": "summarized"}},
			map[string]any{"thinking": map[string]any{"type": "enabled", "budget_tokens": float64(2048), "display": "summarized"}}},
		{"thinking without type dropped", map[string]any{"thinking": map[string]any{"budget_tokens": float64(1)}},
			map[string]any{"thinking": nil}},
		{"maxThinkingTokens", map[string]any{"maxThinkingTokens": float64(4096)},
			map[string]any{"thinking": map[string]any{"type": "enabled", "budget_tokens": float64(4096)}}},
		{"thinking wins over max_thinking_tokens", map[string]any{"max_thinking_tokens": float64(9), "thinking": map[string]any{"type": "disabled"}},
			map[string]any{"thinking": map[string]any{"type": "disabled"}}},
		{"effort", map[string]any{"effort": "low"},
			map[string]any{"output_config": map[string]any{"effort": "low"}}},
		{"outputFormat to structured outputs", map[string]any{"effort": "high", "output_format": map[string]any{"type": "json_schema", "schema": map[string]any{"type": "object"}, "extra": 1}},
			map[string]any{"output_config": map[string]any{"effort": "high", "format": map[string]any{"type": "json_schema", "schema": map[string]any{"type": "object"}}}}},
		{"unlisted keys dropped", map[string]any{"maxBudgetUsd": float64(1), "cwd": "/", "temperature": float64(0.5), "outputFormat": map[string]any{"type": "json_schema"}},
			map[string]any{"cwd": nil, "temperature": nil, "maxBudgetUsd": nil, "max_budget_usd": nil, "output_config": nil, "outputFormat": nil}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			e := setup(t, nil, textResp("ok", 1, 1))
			c := cfg(func(c *aiserver.AiConfig) { c.Model.Parameters = tc.params })
			if _, err := e.handler().Invoke(bg, aiserver.HandlerRequest{Config: c, UserInput: "q"}); err != nil {
				t.Fatal(err)
			}
			body := e.api.reqs()[0]
			for k, want := range tc.want {
				got, ok := body[k]
				if want == nil {
					if ok {
						t.Errorf("%s sent: %v", k, got)
					}
				} else if !reflect.DeepEqual(got, want) {
					t.Errorf("%s = %v, want %v", k, got, want)
				}
			}
		})
	}
}

func TestBetasHeader(t *testing.T) {
	e := setup(t, nil, textResp("ok", 1, 1))
	c := cfg(func(c *aiserver.AiConfig) {
		c.Model.Parameters = map[string]any{"betas": []any{"context-1m-2025-08-07", "", 3, "x-beta"}}
	})
	if _, err := e.handler().Invoke(bg, aiserver.HandlerRequest{Config: c, UserInput: "q"}); err != nil {
		t.Fatal(err)
	}
	if got := e.api.headers[0].Get("anthropic-beta"); got != "context-1m-2025-08-07,x-beta" {
		t.Errorf("anthropic-beta = %q", got)
	}
}

// fallbackModel: an overloaded primary retries once on the fallback, which the
// rest of the run keeps; client errors (4xx) do not fall back.
func TestFallbackModel(t *testing.T) {
	e := setup(t, nil, "STATUS529", textResp("ok", 1, 1))
	c := cfg(func(c *aiserver.AiConfig) { c.Model.Parameters = map[string]any{"fallback_model": "backup"} })
	res, err := e.handler().Invoke(bg, aiserver.HandlerRequest{Config: c, UserInput: "q"})
	if err != nil || res.Output != "ok" {
		t.Fatalf("got %q %v", res.Output, err)
	}
	r := e.api.reqs()
	if len(r) != 2 || r[0]["model"] != c.Model.Name || r[1]["model"] != "backup" {
		t.Errorf("models = %v", r)
	}
	if n := len(e.spans("chat " + c.Model.Name)); n != 1 {
		t.Errorf("failed primary chat spans = %d", n)
	}
	if n := len(e.spans("chat backup")); n != 1 {
		t.Errorf("fallback chat spans = %d", n)
	}

	e = setup(t, nil, "STATUS400", textResp("ok", 1, 1))
	if _, err := e.handler().Invoke(bg, aiserver.HandlerRequest{Config: c, UserInput: "q"}); err == nil || len(e.api.reqs()) != 1 {
		t.Errorf("4xx fell back: %v, %d requests", err, len(e.api.reqs()))
	}
}
