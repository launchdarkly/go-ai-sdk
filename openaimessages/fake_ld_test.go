package openaimessages

import (
	"encoding/json"
	"sync"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/launchdarkly/go-sdk-common/v3/ldvalue"
)

type trackCall struct {
	event string
	ctx   ldcontext.Context
	value float64
	data  map[string]any
}

// fakeLD serves one variation for every key and records track calls.
type fakeLD struct {
	variation map[string]any
	mu        sync.Mutex
	tracks    []trackCall
}

func (f *fakeLD) JSONVariation(string, ldcontext.Context, ldvalue.Value) (ldvalue.Value, error) {
	b, _ := json.Marshal(f.variation)
	return ldvalue.Parse(b), nil
}

func (f *fakeLD) TrackMetric(event string, c ldcontext.Context, v float64, data ldvalue.Value) error {
	var d map[string]any
	_ = json.Unmarshal([]byte(data.JSONString()), &d)
	f.mu.Lock()
	defer f.mu.Unlock()
	f.tracks = append(f.tracks, trackCall{event, c, v, d})
	return nil
}

func (f *fakeLD) Flush()       {}
func (f *fakeLD) Close() error { return nil }

// usageHelpers lists the helper of every `$ld:ai:sdk:usage` event, in order.
func (f *fakeLD) usageHelpers() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	var out []string
	for _, t := range f.tracks {
		if t.event == "$ld:ai:sdk:usage" {
			out = append(out, t.data["helper"].(string))
		}
	}
	return out
}
