package litellmmessages

import "testing"

// An Authorization entry in OPENAI_CUSTOM_HEADERS must not drop the proxy key
// or leak its value; with no key the placeholder openai-go requires is sent.
func TestProxyKeySurvivesCustomAuthorizationHeader(t *testing.T) {
	cases := []struct {
		name     string
		apiKey   string
		envKey   string
		wantAuth string
	}{
		{"key from options", "proxy-key", "", "Bearer proxy-key"},
		{"key from LITELLM_API_KEY", "", "env-key", "Bearer env-key"},
		{"no key", "", "", "Bearer not-needed"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("OPENAI_CUSTOM_HEADERS", "Authorization: Bearer leak\nX-Leak: yes")
			t.Setenv("LITELLM_BASE_URL", "")
			t.Setenv("LITELLM_API_KEY", tc.envKey)
			p := newProxy(t, text("ok", 1, 1))
			h, err := NewHandler(Options{BaseURL: p.url, APIKey: tc.apiKey})
			if err != nil {
				t.Fatal(err)
			}
			if _, err := h.Invoke(bg, req(cfg("anthropic/claude-sonnet-4", "Anthropic", nil), "q")); err != nil {
				t.Fatal(err)
			}
			if p.count() != 1 {
				t.Fatalf("proxy requests = %d, want 1", p.count())
			}
			r := p.request(0)
			if got := r.header.Get("Authorization"); got != tc.wantAuth {
				t.Errorf("Authorization = %q, want %q", got, tc.wantAuth)
			}
			if v := r.header.Get("X-Leak"); v != "" {
				t.Errorf("X-Leak = %q reached the proxy", v)
			}
		})
	}
}
