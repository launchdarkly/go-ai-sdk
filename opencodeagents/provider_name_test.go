package opencodeagents

import (
	"testing"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

// providerCases: the configured provider, the gen_ai.provider.name it becomes
// and the gen_ai.system it keeps. Bedrock is written aws.bedrock in the
// provider name in any case; gen_ai.system stays the lower-cased provider.
var providerCases = []struct{ provider, want, system string }{
	{"Bedrock", "aws.bedrock", "bedrock"},
	{"bedrock", "aws.bedrock", "bedrock"},
	{"Anthropic", "anthropic", "anthropic"},
	{"", "opencode", "opencode"},
}

func TestProviderName(t *testing.T) {
	for _, tc := range providerCases {
		t.Run("provider="+tc.provider, func(t *testing.T) {
			e := setup(t, nil)
			f := newFake(t, fakeSpec{Events: reply("ok", 1, 1)})
			c := cfg(func(c *aiserver.AiConfig) { c.Provider.Name = tc.provider })
			if _, err := invoke(t, f.handler(t), aiserver.HandlerRequest{Config: c, UserInput: "q"}); err != nil {
				t.Fatal(err)
			}
			n := 0
			for _, s := range e.exp.GetSpans().Snapshots() {
				a := attrs(s)
				p, ok := a["gen_ai.provider.name"]
				if !ok {
					continue
				}
				n++
				if p != tc.want {
					t.Errorf("%s gen_ai.provider.name = %q, want %q", s.Name(), p, tc.want)
				}
				if a["gen_ai.system"] != tc.system {
					t.Errorf("%s gen_ai.system = %q, want %q", s.Name(), a["gen_ai.system"], tc.system)
				}
			}
			if n < 2 {
				t.Errorf("%d spans write gen_ai.provider.name, want at least 2 (root and chat)", n)
			}
		})
	}
}
