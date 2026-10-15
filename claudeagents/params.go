package claudeagents

import (
	"errors"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// queryOptions is config.model.parameters narrowed to the JavaScript handler's
// allowlist (claude-agents FORWARDED_QUERY_OPTION_KEYS) and mapped from the
// Agent SDK's query() options onto the Messages API calls this handler makes.
// maxTurns is read by maxStepsFor; maxBudgetUsd has no Messages API field and
// needs per-model pricing, so it is left out.
type queryOptions struct {
	body          map[string]any         // thinking, output_config.effort / .format
	reqOpts       []option.RequestOption // betas → anthropic-beta
	fallbackModel string
}

// param reads key in camelCase or snake_case; snake_case wins, as in the
// JavaScript camelizeModelParameters.
func param(params map[string]any, camel string) (any, bool) {
	var snake strings.Builder
	for _, r := range camel {
		if r >= 'A' && r <= 'Z' {
			snake.WriteByte('_')
			r += 'a' - 'A'
		}
		snake.WriteRune(r)
	}
	if v, ok := params[snake.String()]; ok && v != nil {
		return v, true
	}
	v, ok := params[camel]
	return v, ok && v != nil
}

func buildQueryOptions(params map[string]any) queryOptions {
	q := queryOptions{body: map[string]any{}}
	if t := thinkingParam(params); t != nil {
		q.body["thinking"] = t
	}
	oc := map[string]any{}
	if e, ok := param(params, "effort"); ok {
		oc["effort"] = e
	}
	if f := outputFormatParam(params); f != nil {
		oc["format"] = f
	}
	if len(oc) > 0 {
		q.body["output_config"] = oc
	}
	if b, ok := param(params, "betas"); ok {
		if list, ok := b.([]any); ok {
			var names []string
			for _, v := range list {
				if s, ok := v.(string); ok && s != "" {
					names = append(names, s)
				}
			}
			if len(names) > 0 {
				q.reqOpts = append(q.reqOpts, option.WithHeader("anthropic-beta", strings.Join(names, ",")))
			}
		}
	}
	if m, ok := param(params, "fallbackModel"); ok {
		q.fallbackModel, _ = m.(string)
	}
	return q
}

// thinkingParam builds the Messages API thinking block from thinking (the
// LaunchDarkly UI's shape, either budget spelling) or, failing that, the
// deprecated maxThinkingTokens. Like toThinkingOption in JavaScript, only
// type, the budget and display are kept, and a value without a string type is
// dropped.
func thinkingParam(params map[string]any) map[string]any {
	if v, ok := param(params, "thinking"); ok {
		src, ok := v.(map[string]any)
		if !ok {
			return nil
		}
		typ, ok := src["type"].(string)
		if !ok {
			return nil
		}
		out := map[string]any{"type": typ}
		budget, ok := src["budget_tokens"].(float64)
		if !ok {
			budget, ok = src["budgetTokens"].(float64)
		}
		if ok {
			out["budget_tokens"] = budget
		}
		if d, ok := src["display"].(string); ok {
			out["display"] = d
		}
		return out
	}
	if v, ok := param(params, "maxThinkingTokens"); ok {
		if n, ok := v.(float64); ok && n > 0 {
			return map[string]any{"type": "enabled", "budget_tokens": n}
		}
	}
	return nil
}

// outputFormatParam maps the Agent SDK's outputFormat ({type: "json_schema",
// schema}) onto Messages API structured outputs, which take the same shape.
// Like toOutputFormatOption in JavaScript, anything else is dropped and the
// schema is passed through as written.
func outputFormatParam(params map[string]any) map[string]any {
	v, _ := param(params, "outputFormat")
	src, _ := v.(map[string]any)
	schema, ok := src["schema"].(map[string]any)
	if !ok || src["type"] != "json_schema" {
		return nil
	}
	return map[string]any{"type": "json_schema", "schema": schema}
}

// shouldFallBack reports a failure the Agent SDK's fallbackModel covers: the
// primary model is overloaded (529) or the provider failed (5xx).
func shouldFallBack(err error) bool {
	var apiErr *anthropic.Error
	return errors.As(err, &apiErr) && apiErr.StatusCode >= 500
}
