package aiserver

import (
	"maps"
	"slices"
	"strings"
	"sync"
)

// Registry holds handlers and tools shared by Config and Graph.
type Registry struct {
	mu       sync.Mutex
	handlers []*ProviderHandler
	tools    map[string]ToolHandler
}

type RegistryEntries struct {
	Handlers []*ProviderHandler
	Tools    map[string]ToolHandler
}

func NewRegistry(initial RegistryEntries) *Registry {
	r := &Registry{tools: map[string]ToolHandler{}}
	r.Register(initial)
	return r
}

// Register appends handlers and tools. A handler with the identity of one
// already registered (ProvidesFor, plus the provider set on a "*" handler)
// replaces it in its slot, and a duplicate tool key replaces that tool; each
// logs a warning.
func (r *Registry) Register(e RegistryEntries) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, h := range e.Handlers {
		if i := r.index(h); i >= 0 {
			args := []any{"provider", h.ProvidesFor.Provider, "mode", h.ProvidesFor.Mode}
			if isWildcard(h) && h.Providers != nil {
				args = append(args, "providers", strings.Join(providerSet(h.Providers), ","))
			}
			sdkLogger().Warn("multiple handlers registered; the last one will be used", args...)
			// In place: position breaks ties between wildcards.
			r.handlers[i] = h
			continue
		}
		r.handlers = append(r.handlers, h)
	}
	for k, fn := range e.Tools {
		if _, dup := r.tools[k]; dup {
			sdkLogger().Warn("multiple tools registered; the last one will be used", "tool", k)
		}
		r.tools[k] = fn
	}
}

// index is the slot of the handler with h's identity, or -1. A handler with
// no ProvidesFor has no identity.
func (r *Registry) index(h *ProviderHandler) int {
	if h.ProvidesFor.isZero() {
		return -1
	}
	return slices.IndexFunc(r.handlers, func(e *ProviderHandler) bool { return !e.ProvidesFor.isZero() && sameIdentity(e, h) })
}

func (r *Registry) Handlers() []*ProviderHandler {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]*ProviderHandler(nil), r.handlers...)
}

func (r *Registry) Tools() map[string]ToolHandler {
	r.mu.Lock()
	defer r.mu.Unlock()
	return maps.Clone(r.tools)
}

// Compose returns a new registry where b overrides a on conflict.
func Compose(a, b *Registry) *Registry {
	r := NewRegistry(RegistryEntries{Handlers: a.Handlers(), Tools: a.Tools()})
	r.Register(RegistryEntries{Handlers: b.Handlers(), Tools: b.Tools()})
	return r
}

// resolveHandlers puts local handlers ahead of registry handlers.
func resolveHandlers(r *Registry, local []*ProviderHandler) []*ProviderHandler {
	if r == nil {
		return local
	}
	reg := r.Handlers()
	if len(local) == 0 {
		if len(reg) == 0 {
			return nil
		}
		return reg
	}
	return append(append([]*ProviderHandler(nil), local...), reg...)
}

// resolveTools overlays local tools on registry tools.
func resolveTools(r *Registry, local map[string]ToolHandler) map[string]ToolHandler {
	if r == nil {
		return local
	}
	reg := r.Tools()
	if local == nil {
		if len(reg) == 0 {
			return nil
		}
		return reg
	}
	maps.Copy(reg, local)
	return reg
}
