package aiserver

import (
	"maps"
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

// Register appends handlers and tools. A duplicate providesFor or tool key
// logs a warning and the last one wins.
func (r *Registry) Register(e RegistryEntries) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, h := range e.Handlers {
		if !h.ProvidesFor.IsZero() {
			kept := r.handlers[:0:0]
			for _, existing := range r.handlers {
				if existing.ProvidesFor.Provider == h.ProvidesFor.Provider && handlerMode(existing) == handlerMode(h) {
					sdkLogger().Warn("multiple handlers registered; the last one will be used", "provider", h.ProvidesFor.Provider, "mode", h.ProvidesFor.Mode)
					continue
				}
				kept = append(kept, existing)
			}
			r.handlers = kept
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
