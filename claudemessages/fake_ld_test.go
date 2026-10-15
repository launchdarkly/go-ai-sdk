package claudemessages

import (
	"encoding/json"
	"sync"
	"testing"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/launchdarkly/go-sdk-common/v3/ldvalue"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

type trackCall struct {
	Event  string
	Metric float64
	Data   map[string]any
}

// fakeLD serves variations by key and records track calls.
type fakeLD struct {
	mu         sync.Mutex
	variations map[string]any
	tracks     []trackCall
}

func (f *fakeLD) JSONVariation(key string, _ ldcontext.Context, def ldvalue.Value) (ldvalue.Value, error) {
	v, ok := f.variations[key]
	if !ok {
		return def, nil
	}
	b, _ := json.Marshal(v)
	return ldvalue.Parse(b), nil
}

func (f *fakeLD) TrackMetric(event string, _ ldcontext.Context, m float64, data ldvalue.Value) error {
	var d map[string]any
	_ = json.Unmarshal([]byte(data.JSONString()), &d)
	f.mu.Lock()
	defer f.mu.Unlock()
	f.tracks = append(f.tracks, trackCall{event, m, d})
	return nil
}
func (f *fakeLD) Flush()       {}
func (f *fakeLD) Close() error { return nil }

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

func (f *fakeLD) helpers() []string {
	var out []string
	for _, t := range f.events("$ld:ai:sdk:usage") {
		out = append(out, t.Data["helper"].(string))
	}
	return out
}

// withLD installs a fresh fake LD client serving variations, then puts exp
// back as the span sink (InitWithClient installs its own provider).
func withLD(t *testing.T, exp *tracetest.InMemoryExporter, variations map[string]any) *fakeLD {
	t.Helper()
	_ = aiserver.Shutdown(bg)
	ld := &fakeLD{variations: variations}
	if _, err := aiserver.InitWithClient(bg, ld, aiserver.InitOptions{OTLPEndpoint: "http://127.0.0.1:1/v1/traces"}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = aiserver.Shutdown(bg) })
	otel.SetTracerProvider(sdktrace.NewTracerProvider(sdktrace.WithSyncer(exp)))
	return ld
}

// variation is an enabled messages-mode Anthropic flag variation.
func variation(instructions string) map[string]any {
	return map[string]any{
		"model": map[string]any{"name": "test-model"}, "provider": map[string]any{"name": "Anthropic"},
		"instructions": instructions, "_ldMeta": map[string]any{"enabled": true, "variationKey": "v"},
	}
}
