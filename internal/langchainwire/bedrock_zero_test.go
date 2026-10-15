package langchainwire

import (
	"encoding/json"
	"testing"
)

func TestWithZeroTemperature(t *testing.T) {
	cases := []struct {
		name, in string
		want     any // temperature in the output, or nil for absent
	}{
		{"anthropic body without temperature gets 0", `{"anthropic_version":"bedrock-2023-05-31","max_tokens":10}`, 0.0},
		{"an explicit value is kept", `{"anthropic_version":"x","temperature":0.7}`, 0.7},
		{"other model families are untouched", `{"textGenerationConfig":{}}`, nil},
		{"invalid JSON is untouched", `not json`, nil},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var m map[string]any
			out := withZeroTemperature([]byte(c.in))
			if err := json.Unmarshal(out, &m); err != nil {
				if string(out) != c.in {
					t.Fatalf("unparseable body changed: %s", out)
				}
				return
			}
			if got := m["temperature"]; got != c.want {
				t.Errorf("temperature = %v, want %v", got, c.want)
			}
		})
	}
}
