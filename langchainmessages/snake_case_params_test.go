package langchainmessages

import (
	"reflect"
	"testing"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/llms"
)

// model.parameters are read in snake_case only: a camelCase spelling of a
// table key is not in the table, so it forwards nothing.
func TestModelParametersSnakeCaseOnly(t *testing.T) {
	for _, tc := range []struct {
		camel, snake string
		value        any
		set          func(llms.CallOptions) bool
	}{
		{"maxTokens", "max_tokens", 5, func(o llms.CallOptions) bool { return o.MaxTokens == 5 }},
		{"topP", "top_p", 0.5, func(o llms.CallOptions) bool { return o.TopP != nil && *o.TopP == 0.5 }},
		{"topK", "top_k", 5, func(o llms.CallOptions) bool { return o.TopK != nil && *o.TopK == 5 }},
		{"frequencyPenalty", "frequency_penalty", 0.5, func(o llms.CallOptions) bool { return o.FrequencyPenalty != nil && *o.FrequencyPenalty == 0.5 }},
		{"presencePenalty", "presence_penalty", 0.5, func(o llms.CallOptions) bool { return o.PresencePenalty != nil && *o.PresencePenalty == 0.5 }},
		{"stopSequences", "stop_sequences", []any{"END"}, func(o llms.CallOptions) bool { return reflect.DeepEqual(o.StopWords, []string{"END"}) }},
	} {
		for key, forward := range map[string]bool{tc.camel: false, tc.snake: true} {
			h, f, _ := setup(t, textTurn("ok", 1, 1))
			c := cfg(func(c *aiserver.AiConfig) { c.Model.Parameters = map[string]any{key: tc.value} })
			if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: c, UserInput: "q"}); err != nil {
				t.Fatal(err)
			}
			if o := f.calls[0].opts; tc.set(o) != forward {
				t.Errorf("{%s: %v}: forwarded = %v, want %v (call options %+v)", key, tc.value, !forward, forward, o)
			}
		}
	}
}

// stop is another name for stop_sequences; when both are set, stop_sequences wins.
func TestModelParametersStopSequencesWinsOverStop(t *testing.T) {
	h, f, _ := setup(t, textTurn("ok", 1, 1))
	c := cfg(func(c *aiserver.AiConfig) {
		c.Model.Parameters = map[string]any{"stop": []any{"A"}, "stop_sequences": []any{"B"}}
	})
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: c, UserInput: "q"}); err != nil {
		t.Fatal(err)
	}
	if got := f.calls[0].opts.StopWords; !reflect.DeepEqual(got, []string{"B"}) {
		t.Errorf("stop words = %v, want [B] (stop_sequences wins)", got)
	}
}
