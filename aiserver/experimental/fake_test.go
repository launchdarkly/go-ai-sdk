package experimental_test

import (
	"encoding/json"
	"sync"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/launchdarkly/go-sdk-common/v3/ldvalue"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

type track struct {
	event string
	data  map[string]any
}

type fakeLD struct {
	mu     sync.Mutex
	tracks []track
}

func (f *fakeLD) JSONVariation(string, ldcontext.Context, ldvalue.Value) (ldvalue.Value, error) {
	return ldvalue.Null(), nil
}
func (f *fakeLD) TrackMetric(event string, _ ldcontext.Context, _ float64, data ldvalue.Value) error {
	var d map[string]any
	_ = json.Unmarshal([]byte(data.JSONString()), &d)
	f.mu.Lock()
	defer f.mu.Unlock()
	f.tracks = append(f.tracks, track{event, d})
	return nil
}
func (f *fakeLD) Flush()       {}
func (f *fakeLD) Close() error { return nil }

func (f *fakeLD) usage() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	var out []string
	for _, tr := range f.tracks {
		if tr.event == "$ld:ai:sdk:usage" {
			out = append(out, tr.data["helper"].(string))
			if tr.data["helperPackageName"] != aiserver.PackageName || tr.data["aiSdkName"] != aiserver.PackageName || tr.data["aiSdkLanguage"] != "go" {
				panic("usage event identity")
			}
		}
	}
	return out
}
