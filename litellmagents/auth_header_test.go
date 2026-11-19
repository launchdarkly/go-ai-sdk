package litellmagents

import (
	"testing"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

// An Authorization entry in OPENAI_CUSTOM_HEADERS must not drop the proxy key
// or leak its value; no configured key sends no Authorization header at all.
func TestProxyKeySurvivesCustomAuthorizationHeader(t *testing.T) {
	cases := []struct {
		name     string
		opts     func(url string) Options
		envKey   string
		wantAuth string // "" means no Authorization header
	}{
		{"key from options", func(u string) Options { return Options{BaseURL: u, APIKey: "proxy-key"} }, "", "Bearer proxy-key"},
		{"key from LITELLM_API_KEY", func(u string) Options { return Options{BaseURL: u} }, "env-key", "Bearer env-key"},
		{"no key", func(u string) Options { return Options{BaseURL: u} }, "", ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tripwire(t)
			t.Setenv("OPENAI_CUSTOM_HEADERS", "Authorization: Bearer leak\nX-Leak: yes")
			t.Setenv("LITELLM_BASE_URL", "")
			t.Setenv("LITELLM_API_KEY", tc.envKey)
			f, url := proxy(t, textResp("ok", 1, 1))
			h := handler(t, tc.opts(url))
			invoke(t, h, aiserver.HandlerRequest{Config: cfg(nil), UserInput: "q"})
			if f.count() != 1 {
				t.Fatalf("proxy requests = %d, want 1", f.count())
			}
			r := f.rec(0)
			if r.auth != tc.wantAuth {
				t.Errorf("Authorization = %q, want %q", r.auth, tc.wantAuth)
			}
			if v := r.header.Get("X-Leak"); v != "" {
				t.Errorf("X-Leak = %q reached the proxy", v)
			}
		})
	}
}
