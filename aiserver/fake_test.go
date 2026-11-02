package aiserver

import (
	"context"
	"encoding/json"
	"sync"
	"testing"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/launchdarkly/go-sdk-common/v3/ldvalue"
)

type trackCall struct {
	Event  string
	Metric float64
	Data   map[string]any
	Ctx    ldcontext.Context
}

// fakeLD is the hand-written in-memory LD client double (a "mock" in the JS and Python tests).
type fakeLD struct {
	mu         sync.Mutex
	variations map[string]any
	tracks     []trackCall
	closed     int
	trackPanic bool
	calls      map[string]int
	evalErr    error // returned with the default for a missing key
	evalCtxs   map[string][]ldcontext.Context
	flushes    int
	flushPanic bool
}

func (f *fakeLD) JSONVariation(key string, c ldcontext.Context, def ldvalue.Value) (ldvalue.Value, error) {
	f.mu.Lock()
	if f.calls == nil {
		f.calls = map[string]int{}
		f.evalCtxs = map[string][]ldcontext.Context{}
	}
	f.calls[key]++
	f.evalCtxs[key] = append(f.evalCtxs[key], c)
	f.mu.Unlock()
	v, ok := f.variations[key]
	if !ok {
		return def, f.evalErr
	}
	b, _ := json.Marshal(v)
	return ldvalue.Parse(b), nil
}

func (f *fakeLD) TrackMetric(event string, c ldcontext.Context, m float64, data ldvalue.Value) error {
	if f.trackPanic {
		panic("track exploded")
	}
	var d map[string]any
	_ = json.Unmarshal([]byte(data.JSONString()), &d)
	f.mu.Lock()
	defer f.mu.Unlock()
	f.tracks = append(f.tracks, trackCall{event, m, d, c})
	return nil
}
func (f *fakeLD) Flush() {
	f.flushes++
	if f.flushPanic {
		panic("flush exploded")
	}
}
func (f *fakeLD) Close() error { f.closed++; return nil }

func (f *fakeLD) events(name string) []trackCall {
	f.mu.Lock()
	defer f.mu.Unlock()
	var out []trackCall
	for _, t := range f.tracks {
		if t.Event == name {
			out = append(out, t)
		}
	}
	return out
}

// withFake installs a fake client and resets singleton state after the test.
func withFake(t *testing.T, variations map[string]any) *fakeLD {
	t.Helper()
	f := &fakeLD{variations: variations}
	setClientForTest(f)
	t.Cleanup(func() { _ = Shutdown(context.Background()) })
	return f
}

func enabledConfig(provider, mode string, extra map[string]any) map[string]any {
	cfg := map[string]any{
		"model":        map[string]any{"name": "test-model"},
		"provider":     map[string]any{"name": provider},
		"instructions": "be helpful",
		"_ldMeta":      map[string]any{"enabled": true, "variationKey": "v1", "version": 3, "mode": mode},
	}
	for k, v := range extra {
		cfg[k] = v
	}
	return cfg
}

func stubHandler(provider string, mode Mode, out string, usage map[string]any, seen *HandlerRequest) *ProviderHandler {
	return createHandler(ProvidesFor{provider, mode}, func(_ context.Context, req HandlerRequest) (HandlerResult, error) {
		if seen != nil {
			*seen = req
		}
		return HandlerResult{Output: out, Usage: usage}, nil
	}, nil, false)
}

var userCtx = ldcontext.NewBuilder("user-123").SetString("email", "user@example.com").Build()
