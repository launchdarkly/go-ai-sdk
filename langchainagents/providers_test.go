package langchainagents

import (
	"errors"
	"slices"
	"testing"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

func TestOptionsProvidersScopeTheWildcard(t *testing.T) {
	h, err := NewHandler(Options{})
	if err != nil || h.Providers != nil {
		t.Fatalf("unscoped: Providers = %v, err = %v; want nil, nil", h.Providers, err)
	}
	h, err = NewHandler(Options{Providers: []string{"Bedrock", "Bedrock"}})
	if err != nil {
		t.Fatal(err)
	}
	if h.ProvidesFor != (aiserver.ProvidesFor{Provider: "*", Mode: aiserver.ModeAgent}) || !slices.Equal(h.Providers, []string{"Bedrock"}) {
		t.Errorf("handler = %+v / %v", h.ProvidesFor, h.Providers)
	}
	for _, bad := range [][]string{{}, {""}, {"Bedrock "}} {
		if _, err := NewHandler(Options{Providers: bad}); !errors.Is(err, aiserver.ErrInvalidProviders) {
			t.Errorf("NewHandler Providers %q: err = %v, want ErrInvalidProviders", bad, err)
		}
		if g, err := LangChainGraph("g", aiserver.GraphOptions{}, Options{Providers: bad}); !errors.Is(err, aiserver.ErrInvalidProviders) || g != nil {
			t.Errorf("LangChainGraph Providers %q: %v, %v; want nil, ErrInvalidProviders", bad, g, err)
		}
	}
	if _, err := LangChainAgents(bg, "cfg", "q", ldcontext.New("u"), &InvokeOptions{Handler: Options{Providers: []string{""}}}); !errors.Is(err, aiserver.ErrInvalidProviders) {
		t.Errorf("LangChainAgents invalid list: err = %v", err)
	}
}
