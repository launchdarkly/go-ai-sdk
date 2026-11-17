package langchainmessages

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
	names := []string{"Bedrock", "Anthropic", "Bedrock"}
	h, err = NewHandler(Options{Providers: names})
	if err != nil {
		t.Fatal(err)
	}
	if h.ProvidesFor != (aiserver.ProvidesFor{Provider: "*", Mode: aiserver.ModeMessages}) || !slices.Equal(h.Providers, []string{"Bedrock", "Anthropic"}) {
		t.Fatalf("handler = %+v / %v", h.ProvidesFor, h.Providers)
	}
	names[0] = "Gemini"
	if h.Providers[0] != "Bedrock" {
		t.Error("the handler must own its list")
	}
	for _, bad := range [][]string{{}, {""}, {" Bedrock"}} {
		if _, err := NewHandler(Options{Providers: bad}); !errors.Is(err, aiserver.ErrInvalidProviders) {
			t.Errorf("Providers %q: err = %v, want ErrInvalidProviders", bad, err)
		}
	}
}

// The wrapper builds its handler from InvokeOptions.Handler, so the list
// reaches routing: an unlisted provider finds no handler and no model runs.
func TestLangChainMessagesForwardsProviders(t *testing.T) {
	ld := &fakeLD{variation: map[string]any{
		"model": map[string]any{"name": "test-model"}, "provider": map[string]any{"name": "Anthropic"},
		"instructions": "Hi", "_ldMeta": map[string]any{"enabled": true, "variationKey": "v"},
	}}
	if _, err := aiserver.InitWithClient(bg, ld, aiserver.InitOptions{OTLPEndpoint: "http://127.0.0.1:1/v1/traces"}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = aiserver.Shutdown(bg) })
	f := &fakeModel{turns: []scripted{textTurn("one", 1, 1)}}
	_, err := LangChainMessages(bg, "cfg", "hello", ldcontext.New("u"), &InvokeOptions{Handler: Options{Model: f, Providers: []string{"Bedrock"}}})
	if !errors.Is(err, aiserver.ErrNoHandler) || len(f.calls) != 0 {
		t.Errorf("err = %v, model calls = %d; want ErrNoHandler and none", err, len(f.calls))
	}
	if _, err := LangChainMessages(bg, "cfg", "hello", ldcontext.New("u"), &InvokeOptions{Handler: Options{Model: f, Providers: []string{""}}}); !errors.Is(err, aiserver.ErrInvalidProviders) {
		t.Errorf("invalid list: err = %v", err)
	}
}
