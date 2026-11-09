package sdkapi_test

import (
	"context"
	"encoding/json"
	"slices"
	"sync"
	"testing"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/launchdarkly/go-sdk-common/v3/ldvalue"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/sdkapi"
)

type fakeLD struct {
	mu      sync.Mutex
	helpers []string
}

func (f *fakeLD) JSONVariation(string, ldcontext.Context, ldvalue.Value) (ldvalue.Value, error) {
	return ldvalue.Null(), nil
}
func (f *fakeLD) TrackMetric(event string, _ ldcontext.Context, _ float64, data ldvalue.Value) error {
	if event != "$ld:ai:sdk:usage" {
		return nil
	}
	var d map[string]any
	_ = json.Unmarshal([]byte(data.JSONString()), &d)
	f.mu.Lock()
	defer f.mu.Unlock()
	f.helpers = append(f.helpers, d["helper"].(string))
	return nil
}
func (f *fakeLD) Flush()       {}
func (f *fakeLD) Close() error { return nil }

func (f *fakeLD) usage() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.helpers)
}

func install(t *testing.T) *fakeLD {
	t.Helper()
	ctx := context.Background()
	_ = aiserver.Shutdown(ctx)
	f := &fakeLD{}
	if _, err := aiserver.InitWithClient(ctx, f, aiserver.InitOptions{}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = aiserver.Shutdown(ctx) })
	return f
}

// Every call below fails to resolve (the fake serves no config); usage is
// reported when the call is made, so the failure doesn't matter.
func run(cfg *aiserver.AIConfig, g *aiserver.AgentGraph) {
	ctx, user := context.Background(), ldcontext.New("u")
	_, _ = cfg.Invoke(ctx, "", user)
	for range cfg.Stream(ctx, "", user) {
	}
	_, _ = g.Invoke(ctx, "", user)
	for range g.Stream(ctx, "", user) {
	}
}

func TestSDKCallsReportNoUsage(t *testing.T) {
	f := install(t)
	run(sdkapi.Config(aiserver.ConfigArgs{Key: "cfg"}), sdkapi.Graph("graph", aiserver.GraphOptions{}))
	if got := f.usage(); len(got) != 0 {
		t.Fatalf("sdkapi calls reported usage: %v", got)
	}
}

func TestDirectCallsReportUsage(t *testing.T) {
	f := install(t)
	run(aiserver.Config(aiserver.ConfigArgs{Key: "cfg"}), aiserver.Graph("graph", aiserver.GraphOptions{}))
	want := []string{"client.config.invoke", "client.config.stream", "client.graph.invoke", "client.graph.stream"}
	if got := f.usage(); !slices.Equal(got, want) {
		t.Fatalf("usage %v, want %v", got, want)
	}
}
