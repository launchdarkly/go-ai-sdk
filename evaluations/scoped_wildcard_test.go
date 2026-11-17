package evaluations

import (
	"testing"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

func scopedHandler(mode aiserver.Mode, providers ...string) *aiserver.ProviderHandler {
	h := handler("*", mode, nil)
	h.Providers = providers
	return h
}

// Pool identity is the Registry's: on a "*" handler it includes the provider
// set, compared as a set.
func TestPoolScopedWildcardIdentity(t *testing.T) {
	bedrock, anthropic := scopedHandler(aiserver.ModeMessages, "Bedrock"), scopedHandler(aiserver.ModeMessages, "Anthropic")
	if _, err := validatePool(nil, []*aiserver.ProviderHandler{bedrock, anthropic}); err != nil {
		t.Errorf("wildcards scoped to different providers are distinct: %v", err)
	}
	if _, err := validatePool(nil, []*aiserver.ProviderHandler{bedrock, handler("*", aiserver.ModeMessages, nil)}); err != nil {
		t.Errorf("a scoped and an unscoped wildcard are distinct: %v", err)
	}
	for name, pool := range map[string][]*aiserver.ProviderHandler{
		"same set, another order": {scopedHandler(aiserver.ModeMessages, "Bedrock", "Anthropic"), scopedHandler(aiserver.ModeMessages, "Anthropic", "Bedrock")},
		"a repeated name":         {scopedHandler(aiserver.ModeMessages, "Bedrock"), scopedHandler("completion", "Bedrock", "Bedrock")},
		"two unscoped":            {handler("*", aiserver.ModeMessages, nil), handler("*", aiserver.ModeMessages, nil)},
	} {
		if _, err := validatePool(nil, pool); err == nil {
			t.Errorf("%s: want the duplicate error", name)
		}
	}
}

func TestPoolScopedWildcardSelection(t *testing.T) {
	bedrock, two, any := scopedHandler(aiserver.ModeMessages, "Bedrock"), scopedHandler(aiserver.ModeMessages, "Bedrock", "Anthropic"), handler("*", aiserver.ModeMessages, nil)
	pool := []*aiserver.ProviderHandler{any, two, bedrock}
	if h := selectHandler(pool, "Bedrock", aiserver.ModeMessages); h != bedrock {
		t.Errorf("Bedrock: got %v, want the one-name list", h)
	}
	if h := selectHandler(pool, "Anthropic", aiserver.ModeMessages); h != two {
		t.Errorf("Anthropic: got %v, want the two-name list", h)
	}
	if h := selectHandler(pool, "OpenAI", aiserver.ModeMessages); h != any {
		t.Errorf("OpenAI: got %v, want the unscoped wildcard", h)
	}
	if h := selectHandler([]*aiserver.ProviderHandler{bedrock}, "OpenAI", aiserver.ModeMessages); h != nil {
		t.Errorf("unlisted provider selected %v", h)
	}
}

// A generation whose provider is outside every scoped list, with no exact
// handler and no unscoped wildcard, fails before any I/O naming the provider.
func TestRunRejectsGenerationOutsideScopedLists(t *testing.T) {
	e := newEnv(t, 1)
	m := e.module()
	runErr(t, m, baseArgs(scopedHandler(aiserver.ModeMessages, "Bedrock")), ErrInvalidArgument, `"OpenAI"`)
	e.assertNoRequests()
}
