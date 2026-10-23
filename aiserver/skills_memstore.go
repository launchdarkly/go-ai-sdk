package aiserver

import (
	"fmt"
	"maps"
	"slices"
	"strconv"
	"sync"
)

// objectSet holds raw skill objects keyed by (key, version). An object with no
// usable version is held under its key alone ("loose") so verification can
// withhold it with a reason instead of it reading as absent. Shared by
// InMemorySkillStore and FDv2SkillStore so both answer pins identically.
// Not safe for concurrent use; callers lock.
type objectSet struct {
	versions map[string]map[int]any
	loose    map[string]any
}

func newObjectSet() *objectSet {
	return &objectSet{versions: map[string]map[int]any{}, loose: map[string]any{}}
}

func (s *objectSet) put(key string, raw any) {
	obj, _ := raw.(map[string]any)
	if v, ok := versionOf(obj["version"]); ok {
		if s.versions[key] == nil {
			s.versions[key] = map[int]any{}
		}
		s.versions[key][v] = raw
		return
	}
	s.loose[key] = raw
}

// get answers a lookup: the newest version for 0, else exactly that version.
// A pin falls through to the loose entry only when the key holds no
// well-formed version.
func (s *objectSet) get(key string, version int) any {
	held := s.versions[key]
	if len(held) == 0 {
		return s.loose[key]
	}
	if version != 0 {
		return held[version]
	}
	newest := 0
	for v := range held {
		newest = max(newest, v)
	}
	return held[newest]
}

// snapshot is one entry per (key, version), under opaque keys.
func (s *objectSet) snapshot() map[string]any {
	out := map[string]any{}
	for key, held := range s.versions {
		for v, raw := range held {
			out[key+":"+strconv.Itoa(v)] = raw
		}
	}
	for key, raw := range s.loose {
		out[key] = raw
	}
	return out
}

func (s *objectSet) clone() *objectSet {
	c := newObjectSet()
	for key, held := range s.versions {
		c.versions[key] = map[int]any{}
		for v, raw := range held {
			c.versions[key][v] = raw
		}
	}
	for key, raw := range s.loose {
		c.loose[key] = raw
	}
	return c
}

func (s *objectSet) size() int {
	n := len(s.loose)
	for _, held := range s.versions {
		n += len(held)
	}
	return n
}

// listenerSet is a registration list shared by both shipped stores.
type listenerSet struct {
	mu        sync.Mutex
	listeners []*Listener
}

func (ls *listenerSet) add(storeName, kind string, l *Listener) error {
	if kind != skillObjectKind {
		return fmt.Errorf("%w: %s notifies only %q changes, so a listener on %q would never fire; register it on %q",
			ErrUnsupportedKind, storeName, skillObjectKind, kind, skillObjectKind)
	}
	if l == nil {
		return invalidArg("%s.AddListener needs a non-nil *Listener", storeName)
	}
	ls.mu.Lock()
	defer ls.mu.Unlock()
	ls.listeners = append(ls.listeners, l)
	return nil
}

func (ls *listenerSet) remove(kind string, l *Listener) {
	if kind != skillObjectKind {
		return
	}
	ls.mu.Lock()
	defer ls.mu.Unlock()
	if i := slices.Index(ls.listeners, l); i >= 0 {
		ls.listeners = slices.Delete(ls.listeners, i, i+1)
	}
}

func (ls *listenerSet) notify(changes []any) {
	ls.mu.Lock()
	listeners := slices.Clone(ls.listeners)
	ls.mu.Unlock()
	for _, raw := range changes {
		for _, l := range listeners {
			l.Notify(raw)
		}
	}
}

// InMemorySkillStore is a SkillStore for local development, tests and bring
// your own content. It holds raw objects verbatim and validates nothing; the
// accessors verify everything they return. It can hold several versions of
// one key. Safe for concurrent use.
type InMemorySkillStore struct {
	mu        sync.RWMutex
	objects   *objectSet
	listeners listenerSet
}

// NewInMemorySkillStore builds a store from raw objects. Each is filed under
// its own "key" field, or under its map key when it has no string key.
func NewInMemorySkillStore(objects map[string]any) *InMemorySkillStore {
	s := &InMemorySkillStore{objects: newObjectSet()}
	for _, mapKey := range slices.Sorted(maps.Keys(objects)) {
		raw := objects[mapKey]
		obj, _ := raw.(map[string]any)
		key, ok := obj["key"].(string)
		if !ok {
			key = mapKey
		}
		s.objects.put(key, raw)
	}
	return s
}

// Put adds or replaces a raw object, keyed by its own key and version, then
// notifies every listener with the object verbatim. A second version of a key
// is kept beside the first.
func (s *InMemorySkillStore) Put(raw map[string]any) error {
	key, ok := raw["key"].(string)
	if !ok {
		return invalidArg("a raw skill object must carry a string \"key\"")
	}
	s.mu.Lock()
	s.objects.put(key, raw)
	s.mu.Unlock()
	s.listeners.notify([]any{raw})
	return nil
}

// GetObject implements SkillStore. A kind other than "skill" returns nil.
func (s *InMemorySkillStore) GetObject(kind, key string, version int) (any, error) {
	if kind != skillObjectKind {
		return nil, nil
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.objects.get(key, version), nil
}

// AllObjects implements SkillStore: one entry per (key, version), under
// opaque map keys.
func (s *InMemorySkillStore) AllObjects(kind string) (map[string]any, error) {
	if kind != skillObjectKind {
		return map[string]any{}, nil
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.objects.snapshot(), nil
}

// AddListener registers l for each Put. Any kind other than "skill" returns
// an error wrapping ErrUnsupportedKind, since such a listener would never fire.
func (s *InMemorySkillStore) AddListener(kind string, l *Listener) error {
	return s.listeners.add("InMemorySkillStore", kind, l)
}

// RemoveListener removes one registration of l; a no-op if none.
func (s *InMemorySkillStore) RemoveListener(kind string, l *Listener) {
	s.listeners.remove(kind, l)
}
