package aiserver

import (
	"context"
	"fmt"
	"log/slog"
	"reflect"
	"sync"

	"github.com/launchdarkly/go-ai-sdk/internal/bridge"
)

// SkillStore is the seam the accessors and WriteSkills read skill content
// through. Implement it to supply your own store.
//
// Raw objects are wire-shaped maps with camelCase fields:
//
//	{"key": "pdf-extraction", "version": 2, "content": "---\n...",
//	 "contentHash": "9f3a...", "name": "PDF Extraction", "description": "..."}
//
// Everything a store returns is untrusted: it is verified before it becomes a
// Skill, so a store does no validation of its own.
//
//   - GetObject returns one object, or nil when the key is not held. version 0
//     means the newest held; any other version means exactly that version. An
//     error means the store could not answer (an outage, not a deletion).
//   - AllObjects returns one entry per (key, version). The map keys are opaque;
//     identity is read from each object's own key and version fields.
//
// Optional halves of the seam are probed with type assertions:
// ListenerStore, ListenerRemover and InitializationProber.
type SkillStore interface {
	GetObject(kind, key string, version int) (any, error)
	AllObjects(kind string) (map[string]any, error)
}

// ListenerStore is the optional half of the seam that pushes changes.
// WatchSkills requires it. AddListener returns an error wrapping
// ErrUnsupportedKind for a kind the store never notifies.
type ListenerStore interface {
	AddListener(kind string, l *Listener) error
}

// ListenerRemover unregisters one registration of l. Removing a listener that
// is not registered is a no-op.
type ListenerRemover interface {
	RemoveListener(kind string, l *Listener)
}

// InitializationProber is the optional readiness probe. A store without it is
// treated as initialized; one returning an error is treated as not
// initialized, so WriteSkills does not prune against a store that cannot say
// whether it has heard from LaunchDarkly yet.
type InitializationProber interface {
	IsInitialized() (bool, error)
}

// Listener is a change callback registered with a ListenerStore. It is a
// pointer so that RemoveListener can find the registration: Go func values are
// not comparable.
type Listener struct {
	fn func(raw any)
}

// NewListener wraps fn. fn receives each changed raw object verbatim and
// unverified; a revocation is a {"key", "version"} tombstone with no content.
func NewListener(fn func(raw any)) *Listener { return &Listener{fn: fn} }

// Notify calls the listener, recovering a panic so one bad listener cannot
// stop delivery.
func (l *Listener) Notify(raw any) {
	if l == nil || l.fn == nil {
		return
	}
	defer func() {
		if r := recover(); r != nil {
			currentLogger().Error("a skill store change listener panicked; delivery continues", "panic", r)
		}
	}()
	l.fn(raw)
}

// telemetryEmitter is the private, structurally typed skills telemetry seam. The default is a no-op.
type telemetryEmitter interface {
	Record(signal string, properties map[string]any)
}

type noopEmitter struct{}

func (noopEmitter) Record(string, map[string]any) {}

var (
	skillStateMu sync.Mutex
	store        SkillStore
	emitter      telemetryEmitter = noopEmitter{}
	logger       *slog.Logger
)

func init() {
	// Shutdown clears skills state through this bridge hook.
	bridge.OnShutdown(func(context.Context) { clearSkillState() })
}

// SetSkillStore configures the store the accessors and WriteSkills read
// from. It applies on every call, including after the client exists. A nil
// store (or a typed nil pointer) is ignored and never clears a configured
// store; Shutdown does that.
//
// Replacing a store does not close the previous one, and a running watcher
// stays subscribed to the store it started with.
func SetSkillStore(s SkillStore) {
	if isNil(s) {
		return
	}
	skillStateMu.Lock()
	defer skillStateMu.Unlock()
	store = s
}

func isNil(v any) bool {
	if v == nil {
		return true
	}
	rv := reflect.ValueOf(v)
	switch rv.Kind() {
	case reflect.Pointer, reflect.Map, reflect.Slice, reflect.Func, reflect.Interface, reflect.Chan:
		return rv.IsNil()
	}
	return false
}

// clearSkillState drops the configured store and the telemetry emitter. Called by
// Shutdown through the bridge hook, and by tests.
func clearSkillState() {
	skillStateMu.Lock()
	defer skillStateMu.Unlock()
	store = nil
	emitter = noopEmitter{}
}

// setSkillEmitterForTesting installs a recording emitter. nil restores the no-op.
func setSkillEmitterForTesting(e telemetryEmitter) {
	skillStateMu.Lock()
	defer skillStateMu.Unlock()
	if e == nil {
		e = noopEmitter{}
	}
	emitter = e
}

// setSkillLogger replaces the skills logger. nil restores the SDK logger
// (SetLogger).
func setSkillLogger(l *slog.Logger) {
	skillStateMu.Lock()
	defer skillStateMu.Unlock()
	logger = l
}

func currentLogger() *slog.Logger {
	skillStateMu.Lock()
	l := logger
	skillStateMu.Unlock()
	if l == nil {
		return sdkLogger()
	}
	return l
}

func currentStore() SkillStore {
	skillStateMu.Lock()
	defer skillStateMu.Unlock()
	return store
}

func requireStore() (SkillStore, error) {
	s := currentStore()
	if s == nil {
		return nil, ErrNoSkillStore
	}
	return s, nil
}

// storeIsInitialized is the readiness probe's three answers: absent
// means ready, present means the store's own answer, failing means not ready.
func storeIsInitialized(s SkillStore) (ready bool) {
	p, ok := s.(InitializationProber)
	if !ok {
		return true
	}
	defer func() {
		if r := recover(); r != nil {
			currentLogger().Warn("the skill store's IsInitialized panicked; treating the store as not yet initialized", "panic", r)
			ready = false
		}
	}()
	ok, err := p.IsInitialized()
	if err != nil {
		currentLogger().Warn("the skill store's IsInitialized failed; treating the store as not yet initialized", "error", err)
		return false
	}
	return ok
}

// getObject calls the store, converting a panic into an error: a store that
// panics has failed to answer, which is an outage.
func getObject(s SkillStore, key string, version int) (raw any, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("the skill store panicked: %v", r)
		}
	}()
	return s.GetObject(skillObjectKind, key, version)
}

func allObjects(s SkillStore) (objects map[string]any, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("the skill store panicked: %v", r)
		}
	}()
	return s.AllObjects(skillObjectKind)
}
