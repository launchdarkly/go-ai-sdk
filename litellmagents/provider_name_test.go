package litellmagents

import (
	"testing"

	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

// providerCases: the configured provider and the gen_ai.provider.name it
// becomes. Bedrock is written aws.bedrock in any case; others lower-case.
var providerCases = []struct{ provider, want string }{
	{"Bedrock", "aws.bedrock"},
	{"bedrock", "aws.bedrock"},
	{"Anthropic", "anthropic"},
	{"", "litellm"},
}

// checkProviderName requires at least min spans to write gen_ai.provider.name,
// every one equal to want, and gen_ai.system to stay litellm on each.
func checkProviderName(t *testing.T, exp *tracetest.InMemoryExporter, want string, min int) {
	t.Helper()
	n := 0
	for name, spans := range spansByName(exp) {
		for _, s := range spans {
			a := attrs(s)
			p, ok := a["gen_ai.provider.name"]
			if !ok {
				continue
			}
			n++
			if p != want {
				t.Errorf("%s gen_ai.provider.name = %q, want %q", name, p, want)
			}
			if a["gen_ai.system"] != "litellm" {
				t.Errorf("%s gen_ai.system = %q, want litellm", name, a["gen_ai.system"])
			}
		}
	}
	if n < min {
		t.Errorf("%d spans write gen_ai.provider.name, want at least %d", n, min)
	}
}

func TestProviderNameOnHandlerSpans(t *testing.T) {
	for _, tc := range providerCases {
		t.Run("provider="+tc.provider, func(t *testing.T) {
			h, _, exp := setup(t, textResp("ok", 1, 1))
			invoke(t, h, aiserver.HandlerRequest{Config: cfg(func(c *aiserver.AiConfig) { c.Provider.Name = tc.provider })})
			checkProviderName(t, exp, tc.want, 2) // root and chat
		})
	}
}

func TestProviderNameOnGraphSpans(t *testing.T) {
	for _, tc := range providerCases {
		t.Run("provider="+tc.provider, func(t *testing.T) {
			ld := &fakeLD{variations: map[string]map[string]any{
				"g":    graphFlag("solo", nil),
				"solo": node("Solo", tc.provider, "anthropic/claude-solo", nil),
			}}
			installClient(t, ld)
			exp := tracer(t)
			tripwire(t)
			_, url := proxy(t, textResp("fine", 1, 1))
			if _, err := adapter(t, resolve(t, "g"), &NativeGraphOptions{Context: user, Handler: Options{BaseURL: url}}).Invoke(bg, "q"); err != nil {
				t.Fatal(err)
			}
			checkProviderName(t, exp, tc.want, 1) // the chat span; node spans do not write it
		})
	}
}
