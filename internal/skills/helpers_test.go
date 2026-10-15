package skills

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"log/slog"
	"maps"
	"strings"
	"sync"
	"testing"
)

// signal is one recorded telemetry emission.
type signal struct {
	name  string
	props map[string]any
}

// recorder is a recording telemetry emitter.
type recorder struct {
	mu      sync.Mutex
	signals []signal
}

func (r *recorder) Record(name string, props map[string]any) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.signals = append(r.signals, signal{name, maps.Clone(props)})
}

func (r *recorder) all() []signal {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]signal(nil), r.signals...)
}

func (r *recorder) named(name string) []signal {
	var out []signal
	for _, s := range r.all() {
		if s.name == name {
			out = append(out, s)
		}
	}
	return out
}

// logEntry is one captured log record.
type logEntry struct {
	level slog.Level
	msg   string
	attrs map[string]any
}

type logCapture struct {
	mu      sync.Mutex
	entries []logEntry
}

func (c *logCapture) Enabled(context.Context, slog.Level) bool { return true }
func (c *logCapture) Handle(_ context.Context, r slog.Record) error {
	e := logEntry{level: r.Level, msg: r.Message, attrs: map[string]any{}}
	r.Attrs(func(a slog.Attr) bool {
		e.attrs[a.Key] = a.Value.Any()
		return true
	})
	c.mu.Lock()
	defer c.mu.Unlock()
	c.entries = append(c.entries, e)
	return nil
}
func (c *logCapture) WithAttrs([]slog.Attr) slog.Handler { return c }
func (c *logCapture) WithGroup(string) slog.Handler      { return c }

func (c *logCapture) all() []logEntry {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]logEntry(nil), c.entries...)
}

// records returns the integrity log records' structured mappings.
func (c *logCapture) records() []map[string]any {
	var out []map[string]any
	for _, e := range c.all() {
		if r, ok := e.attrs["ld_skills"].(map[string]any); ok {
			out = append(out, r)
		}
	}
	return out
}

func (c *logCapture) contains(level slog.Level, substr string) bool {
	for _, e := range c.all() {
		if e.level == level && strings.Contains(e.msg, substr) {
			return true
		}
	}
	return false
}

// setup resets module state and installs a recorder and a log capture.
func setup(t *testing.T) (*recorder, *logCapture) {
	t.Helper()
	ClearState()
	rec := &recorder{}
	logs := &logCapture{}
	SetEmitterForTesting(rec)
	SetLogger(slog.New(logs))
	t.Cleanup(func() {
		ClearState()
		SetLogger(nil)
	})
	return rec, logs
}

func sha(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:])
}

// rawSkill is a valid wire object.
func rawSkill(key string, version int, content string) map[string]any {
	return map[string]any{"key": key, "version": version, "content": content, "contentHash": sha(content), "name": "N " + key, "description": "D"}
}

// skillOf is a verified Skill built directly (BYOC), valid hash.
func skillOf(key string, version int, content string) Skill {
	return NewSkill(SkillFields{Key: key, Version: version, Content: []byte(content), ContentHash: sha(content)})
}

// funcStore is a hand-built store for the cases no shipped store produces.
type funcStore struct {
	get   func(kind, key string, version int) (any, error)
	all   func(kind string) (map[string]any, error)
	calls int
	mu    sync.Mutex
}

func (f *funcStore) GetObject(kind, key string, version int) (any, error) {
	f.mu.Lock()
	f.calls++
	f.mu.Unlock()
	if f.get == nil {
		return nil, nil
	}
	return f.get(kind, key, version)
}

func (f *funcStore) AllObjects(kind string) (map[string]any, error) {
	if f.all == nil {
		return map[string]any{}, nil
	}
	return f.all(kind)
}

var errStoreDown = errors.New("store is down")

// probeStore wraps a store with a readiness probe answer.
type probeStore struct {
	SkillStore
	ready bool
	err   error
	panic bool
}

func (p *probeStore) IsInitialized() (bool, error) {
	if p.panic {
		panic("probe exploded")
	}
	return p.ready, p.err
}
