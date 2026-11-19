package aiserver

import (
	"fmt"
	"slices"
	"strings"

	"github.com/launchdarkly/go-ai-sdk/internal/kitbase"
)

// WithProviders returns a copy of h scoped to providers: each name once, in
// first-seen order, in a list the copy owns. h is not changed. An empty
// list, a blank name or a name with surrounding whitespace returns an error
// wrapping ErrInvalidProviders.
func (h *ProviderHandler) WithProviders(providers ...string) (*ProviderHandler, error) {
	if h == nil {
		return nil, errorf("WithProviders on a nil handler")
	}
	list, err := normalizeProviders(providers)
	if err != nil {
		return nil, err
	}
	c := *h
	c.Providers = list
	return &c, nil
}

// normalizeProviders validates a provider list and drops repeats, keeping
// first-seen order. Names are not trimmed.
func normalizeProviders(providers []string) ([]string, error) {
	if len(providers) == 0 {
		return nil, ErrInvalidProviders
	}
	out := make([]string, 0, len(providers))
	for _, name := range providers {
		if name == "" || strings.TrimSpace(name) != name {
			return nil, fmt.Errorf("%w: got %q", ErrInvalidProviders, name)
		}
		if !slices.Contains(out, name) {
			out = append(out, name)
		}
	}
	return out, nil
}

// providerSet is the sorted distinct names of a list; nil stays nil.
func providerSet(providers []string) []string {
	if providers == nil {
		return nil
	}
	s := slices.Clone(providers)
	slices.Sort(s)
	return slices.Compact(s)
}

// isWildcard reports a "*" handler.
func isWildcard(h *ProviderHandler) bool { return h.ProvidesFor.Provider == kitbase.WildcardProvider }

// sameIdentity is the Registry and evaluations-pool identity: ProvidesFor
// with the mode normalized, plus the provider set on a wildcard. An absent
// list is its own identity; on an exact handler the list is ignored.
func sameIdentity(a, b *ProviderHandler) bool {
	if a.ProvidesFor.Provider != b.ProvidesFor.Provider || handlerMode(a) != handlerMode(b) {
		return false
	}
	if !isWildcard(a) {
		return true
	}
	if (a.Providers == nil) != (b.Providers == nil) {
		return false
	}
	return slices.Equal(providerSet(a.Providers), providerSet(b.Providers))
}

// wildcardCovers reports that a "*" handler accepts provider: an unscoped
// wildcard accepts any name, a scoped one only the names in its list.
func wildcardCovers(h *ProviderHandler, provider string) bool {
	if !isWildcard(h) {
		return false
	}
	return h.Providers == nil || slices.Contains(h.Providers, provider)
}

// covers reports that h may serve provider in some mode: it names the
// provider exactly, or it is a wildcard that accepts it.
func covers(h *ProviderHandler, provider string) bool {
	return h.ProvidesFor.Provider == provider || wildcardCovers(h, provider)
}

// bestWildcard is the "*" handler of mode that accepts provider: the
// shortest scoped list that names it (distinct names; ties keep the earlier
// handler), else the first unscoped wildcard. modeOK decides mode matching.
func bestWildcard(handlers []*ProviderHandler, provider string, modeOK func(*ProviderHandler) bool) *ProviderHandler {
	var scoped, unscoped *ProviderHandler
	best := 0
	for _, h := range handlers {
		if !modeOK(h) || !wildcardCovers(h, provider) {
			continue
		}
		if h.Providers == nil {
			if unscoped == nil {
				unscoped = h
			}
			continue
		}
		if n := len(providerSet(h.Providers)); scoped == nil || n < best {
			scoped, best = h, n
		}
	}
	if scoped != nil {
		return scoped
	}
	return unscoped
}

// selectFor is the shared selection order: the exact provider in a mode
// modeOK accepts, then the best wildcard.
func selectFor(handlers []*ProviderHandler, provider string, modeOK func(*ProviderHandler) bool) *ProviderHandler {
	for _, h := range handlers {
		if h.ProvidesFor.Provider == provider && modeOK(h) {
			return h
		}
	}
	return bestWildcard(handlers, provider, modeOK)
}

func modeIs(mode Mode) func(*ProviderHandler) bool {
	return func(h *ProviderHandler) bool { return handlerMode(h) == mode }
}
