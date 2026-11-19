package litellmmessages

import "testing"

// A configured provider becomes gen_ai.provider.name: Bedrock is written
// aws.bedrock in any case, others lower-case, none is litellm. gen_ai.system
// stays litellm on every span that writes the name.
func TestProviderName(t *testing.T) {
	for _, tc := range []struct{ provider, want string }{
		{"Bedrock", "aws.bedrock"},
		{"bedrock", "aws.bedrock"},
		{"Anthropic", "anthropic"},
		{"", "litellm"},
	} {
		t.Run("provider="+tc.provider, func(t *testing.T) {
			h, _, exp := setup(t, text("ok", 1, 1))
			if _, err := h.Invoke(bg, req(cfg("anthropic/claude-sonnet-4", tc.provider, nil), "q")); err != nil {
				t.Fatal(err)
			}
			n := 0
			for name, spans := range spansByName(exp) {
				for _, s := range spans {
					a := attrs(s)
					p, ok := a["gen_ai.provider.name"]
					if !ok {
						continue
					}
					n++
					if p != tc.want {
						t.Errorf("%s gen_ai.provider.name = %q, want %q", name, p, tc.want)
					}
					if a["gen_ai.system"] != "litellm" {
						t.Errorf("%s gen_ai.system = %q, want litellm", name, a["gen_ai.system"])
					}
				}
			}
			if n < 2 { // root and chat
				t.Errorf("%d spans write gen_ai.provider.name, want at least 2", n)
			}
		})
	}
}
