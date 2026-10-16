package aiserver

import (
	"bytes"
	"context"
	"errors"
	"io"
	"log/slog"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	ld "github.com/launchdarkly/go-server-sdk/v7"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"

	"github.com/launchdarkly/go-ai-sdk/internal/bridge"
)

// withLDFactory replaces the LD client constructor for one test.
func withLDFactory(t *testing.T, fn func(key string) (LDClient, error)) {
	t.Helper()
	_ = Shutdown(bg)
	orig := newLDClient
	newLDClient = func(key string, _ ld.Config, _ time.Duration) (LDClient, error) { return fn(key) }
	t.Cleanup(func() { newLDClient = orig; _ = Shutdown(bg) })
}

const unreachableOTLP = "http://127.0.0.1:1/v1/traces"

func TestInitFailureNotCachedAndClosesClient(t *testing.T) {
	t.Setenv("LD_SDK_KEY", "")
	failed := &fakeLD{}
	good := &fakeLD{}
	var keys []string
	withLDFactory(t, func(key string) (LDClient, error) {
		keys = append(keys, key)
		if len(keys) == 1 {
			return failed, errors.New("initialization timed out")
		}
		return good, nil
	})
	if _, err := InitClient(bg, InitOptions{}); err == nil {
		t.Fatal("missing key must fail")
	}
	if _, err := InitClient(bg, InitOptions{SDKKey: "k1", OTLPEndpoint: unreachableOTLP}); err == nil {
		t.Fatal("timeout must fail")
	}
	if failed.closed != 1 {
		t.Errorf("client built by a failed init was not closed (closed=%d)", failed.closed)
	}
	if _, err := GetClient(); err == nil {
		t.Error("failed init left a client behind")
	}
	c, err := InitClient(bg, InitOptions{SDKKey: "k2", OTLPEndpoint: unreachableOTLP})
	if err != nil || c != good || len(keys) != 2 || keys[1] != "k2" {
		t.Errorf("retry: c=%v err=%v keys=%v", c, err, keys)
	}
}

func TestInitConcurrentCallersShareOneClient(t *testing.T) {
	built := 0
	var mu sync.Mutex
	withLDFactory(t, func(string) (LDClient, error) {
		mu.Lock()
		defer mu.Unlock()
		built++
		return &fakeLD{}, nil
	})
	var wg sync.WaitGroup
	for range 10 {
		wg.Go(func() { _, _ = InitClient(bg, InitOptions{SDKKey: "k", OTLPEndpoint: unreachableOTLP}) })
	}
	wg.Wait()
	if built != 1 {
		t.Errorf("built %d clients", built)
	}
}

func TestBYOCAndSdkInfoLifecycle(t *testing.T) {
	_ = Shutdown(bg)
	t.Cleanup(func() { _ = Shutdown(bg) })
	restorePackagesAfter(t)
	prev := otel.GetTracerProvider()
	t.Cleanup(func() { otel.SetTracerProvider(prev) })

	byoc := &fakeLD{}
	c, err := InitWithClient(bg, byoc, InitOptions{OTLPEndpoint: unreachableOTLP})
	if err != nil || c != byoc {
		t.Fatalf("byoc: %v %v", c, err)
	}
	if got, _ := GetClient(); got != byoc {
		t.Error("GetClient did not return the BYOC client")
	}
	if _, ok := otel.GetTracerProvider().(*sdktrace.TracerProvider); !ok {
		t.Error("BYOC did not set up telemetry")
	}
	if len(byoc.events("$ld:ai:sdk:info")) != 1 {
		t.Errorf("sdk:info = %v", byoc.events("$ld:ai:sdk:info"))
	}
	// Late registration reports on the next init, including the idempotent path.
	RegisterAiSdkPackage("late-pkg", "1.2.3")
	RegisterAiSdkPackage("late-pkg", "1.2.3")
	_, _ = InitClient(bg, InitOptions{})
	info := byoc.events("$ld:ai:sdk:info")
	if len(info) != 2 || info[1].Data["aiSdkName"] != "late-pkg" {
		t.Errorf("late registration: %v", info)
	}
	// Replacing a BYOC client without Shutdown does not reset reporting.
	other := &fakeLD{}
	_, _ = InitWithClient(bg, other, InitOptions{})
	if len(other.events("$ld:ai:sdk:info")) != 0 {
		t.Error("replacing BYOC re-reported packages")
	}
	// Shutdown re-arms reporting for the next client.
	_ = Shutdown(bg)
	again := &fakeLD{}
	_, _ = InitWithClient(bg, again, InitOptions{})
	if len(again.events("$ld:ai:sdk:info")) != 2 {
		t.Errorf("after shutdown: %v", again.events("$ld:ai:sdk:info"))
	}
	// The test-only setter emits nothing.
	_ = Shutdown(bg)
	quiet := &fakeLD{}
	setClientForTest(quiet)
	if len(quiet.tracks) != 0 {
		t.Error("test setter emitted events")
	}
}

func TestUsageReporting(t *testing.T) {
	f := withFake(t, map[string]any{"cfg": enabledConfig("P", "", nil)})
	h := stubHandler("P", ModeMessages, "ok", map[string]any{"input": 1.0, "output": 1.0}, nil)
	for range 2 {
		_, _ = Config(ConfigArgs{Key: "cfg", Handlers: []*ProviderHandler{h}}).Invoke(bg, "", userCtx)
	}
	_ = Config(ConfigArgs{Key: "cfg", Handlers: []*ProviderHandler{h}}).Stream(bg, "", userCtx) // reports without iterating
	_ = CreateHandler(ProvidesFor{"X", ModeMessages}, nil, nil, false)
	got := map[string]int{}
	for _, e := range f.events("$ld:ai:sdk:usage") {
		got[e.Data["helper"].(string)]++
		if e.Data["aiSdkName"] != PackageName || e.Data["helperPackageName"] != PackageName || e.Data["aiSdkLanguage"] != "go" || e.Metric != 1 {
			t.Errorf("payload = %v", e.Data)
		}
	}
	want := map[string]int{"client.config.invoke": 1, "client.config.stream": 1, "client.createHandler": 1}
	if len(got) != len(want) || got["client.config.invoke"] != 1 || got["client.config.stream"] != 1 || got["client.createHandler"] != 1 {
		t.Errorf("usage helpers = %v", got)
	}
	// Existing events gain no usage fields.
	for _, e := range f.events("$ld:ai:generation:success") {
		for _, k := range []string{"helper", "aiSdkName", "helperPackageName"} {
			if _, has := e.Data[k]; has {
				t.Errorf("generation event carries %s", k)
			}
		}
	}
	// ConfigInternal (used by wrappers) does not report.
	_ = Shutdown(bg)
	g := &fakeLD{variations: map[string]any{"cfg": enabledConfig("P", "", nil)}}
	setClientForTest(g)
	_, _ = ConfigInternal(ConfigArgs{Key: "cfg", Handlers: []*ProviderHandler{h}}).Invoke(bg, "", userCtx)
	if len(g.events("$ld:ai:sdk:usage")) != 0 {
		t.Error("internal config reported usage")
	}
}

func TestUsageThrowingTrackNotRetried(t *testing.T) {
	_ = Shutdown(bg)
	t.Cleanup(func() { _ = Shutdown(bg) })
	p := &fakeLD{trackPanic: true}
	setClientForTest(p)
	reportUsage("client.inspectConfig", "", "")
	p.trackPanic = false
	reportUsage("client.inspectConfig", "", "")
	if len(p.events("$ld:ai:sdk:usage")) != 0 {
		t.Error("a helper whose report failed was retried")
	}
}

func TestInspectConfigRecoversOnceKeyAvailable(t *testing.T) {
	t.Setenv("LD_SDK_KEY", "")
	f := &fakeLD{variations: map[string]any{"cfg": enabledConfig("P", "", nil)}}
	withLDFactory(t, func(string) (LDClient, error) { return f, nil })
	if r, err := InspectConfig(bg, "cfg", userCtx); err == nil || r.Enabled {
		t.Fatal("expected fallback without a key")
	}
	t.Setenv("LD_SDK_KEY", "now-set")
	if r, err := InspectConfig(bg, "cfg", userCtx); err != nil || !r.Enabled {
		t.Error("inspectConfig did not recover after the key appeared")
	}
	for _, e := range f.tracks {
		if e.Event != "$ld:ai:sdk:info" && e.Event != "$ld:ai:sdk:usage" {
			t.Errorf("inspectConfig emitted %s", e.Event)
		}
	}
}

func TestShutdownHooksAndIdempotence(t *testing.T) {
	_ = Shutdown(bg)
	var buf bytes.Buffer
	var bufMu sync.Mutex
	SetLogger(slog.New(slog.NewTextHandler(lockedWriter{&bufMu, &buf}, nil)))
	t.Cleanup(func() { SetLogger(nil) })
	// The bridge has no unregister, so these hooks go inert after this test
	// rather than run (and panic) in every later Shutdown.
	var active atomic.Bool
	active.Store(true)
	t.Cleanup(func() { active.Store(false) })
	var ran atomic.Int32
	bridge.OnShutdown(func(context.Context) {
		if active.Load() {
			ran.Add(1)
		}
	})
	bridge.OnShutdown(func(context.Context) {
		if active.Load() {
			panic("experimental cleanup broke")
		}
	})
	// Hooks run even when no client exists.
	if err := Shutdown(bg); err != nil || ran.Load() != 1 {
		t.Errorf("no-client shutdown: err=%v ran=%d", err, ran.Load())
	}
	f := &fakeLD{}
	setClientForTest(f)
	if err := Shutdown(bg); err != nil {
		t.Errorf("a panicking experimental hook failed Shutdown: %v", err)
	}
	if ran.Load() != 2 || f.closed != 1 {
		t.Errorf("ran=%d closed=%d, want 2 and 1", ran.Load(), f.closed)
	}
	bufMu.Lock()
	logged := buf.String()
	bufMu.Unlock()
	if !strings.Contains(logged, "experimental shutdown hook failed") {
		t.Errorf("hook failure not logged: %q", logged)
	}
	if err := Shutdown(bg); err != nil || f.closed != 1 {
		t.Errorf("second Shutdown must be a no-op: err=%v closed=%d", err, f.closed)
	}
}

type lockedWriter struct {
	mu *sync.Mutex
	w  io.Writer
}

func (l lockedWriter) Write(p []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.w.Write(p)
}

func TestConversationID(t *testing.T) {
	exp := tracetest.NewInMemoryExporter()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(ConversationIDSpanProcessor{}), sdktrace.WithSyncer(exp))
	sdkTracer, other := tp.Tracer(TracerPrefix+"/x"), tp.Tracer("db.postgres")

	ctx := WithConversationID(bg, "conv-1")
	_, s := sdkTracer.Start(ctx, "sdk")
	s.End()
	_, o := other.Start(ctx, "pg")
	o.End()
	_, pre := sdkTracer.Start(ctx, "preset", trace.WithAttributes(attribute.String("gen_ai.conversation.id", "own")))
	pre.End()
	_, none := sdkTracer.Start(WithConversationID(bg, ""), "unbound")
	none.End()
	a, b := WithConversationID(bg, "A"), WithConversationID(bg, "B")
	_, sa := sdkTracer.Start(b, "nested-b")
	_, sb := sdkTracer.Start(a, "nested-a")
	sa.End()
	sb.End()

	got := map[string]string{}
	for _, sp := range exp.GetSpans().Snapshots() {
		for _, kv := range sp.Attributes() {
			if kv.Key == "gen_ai.conversation.id" {
				got[sp.Name()] = kv.Value.AsString()
			}
		}
	}
	want := map[string]string{"sdk": "conv-1", "preset": "own", "nested-a": "A", "nested-b": "B"}
	for k, v := range want {
		if got[k] != v {
			t.Errorf("%s = %q, want %q", k, got[k], v)
		}
	}
	if _, has := got["pg"]; has {
		t.Error("non-SDK tracer was stamped")
	}
	if _, has := got["unbound"]; has {
		t.Error("an id was invented")
	}
}

func TestContextIdentityEmitPath(t *testing.T) {
	exp := tracetest.NewInMemoryExporter()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exp))
	c := ldcontext.NewBuilder("a%b:c").Kind("org").Name("Acme").SetString("email", "x@y.z").Build()
	_, span := tp.Tracer("t").Start(bg, "root")
	SetLDSpanAttributes(span, map[string]any{VarLD: TrackData{ConfigKey: "k"}, varLDContextTyped: c})
	span.End()
	s := exp.GetSpans().Snapshots()[0]
	ev := map[string]string{}
	for _, e := range s.Events() {
		if e.Name == "feature_flag" {
			for _, a := range e.Attributes {
				ev[string(a.Key)] = a.Value.AsString()
			}
		}
	}
	if ev["feature_flag.context.id"] != "org:a%25b%3Ac" || ev["feature_flag.contextKeys"] != `{"org":"a%b:c"}` {
		t.Errorf("event = %v", ev)
	}
	for _, a := range s.Attributes() {
		if a.Value.AsString() == "Acme" || a.Value.AsString() == "x@y.z" {
			t.Errorf("context attribute leaked: %s", a.Key)
		}
	}
	// No context → no identity, no panic.
	exp.Reset()
	_, span = tp.Tracer("t").Start(bg, "root")
	SetLDSpanAttributes(span, map[string]any{VarLD: TrackData{ConfigKey: "k"}})
	span.End()
	for _, a := range exp.GetSpans().Snapshots()[0].Attributes() {
		if a.Key == "context.contextKeys.user" {
			t.Error("identity written without a context")
		}
	}
}

func TestBYOCTelemetryUsesEnvSDKKey(t *testing.T) {
	_ = Shutdown(bg)
	t.Cleanup(func() { _ = Shutdown(bg) })
	prev := otel.GetTracerProvider()
	t.Cleanup(func() { otel.SetTracerProvider(prev) })
	t.Setenv("LD_SDK_KEY", "env-sdk-key")

	if _, err := InitWithClient(bg, &fakeLD{}, InitOptions{OTLPEndpoint: unreachableOTLP}); err != nil {
		t.Fatal(err)
	}
	exp := tracetest.NewInMemoryExporter()
	tracerProvider.RegisterSpanProcessor(sdktrace.NewSimpleSpanProcessor(exp))
	_, span := otel.Tracer("t").Start(bg, "probe")
	span.End()
	var got string
	for _, kv := range exp.GetSpans().Snapshots()[0].Resource().Attributes() {
		if kv.Key == "highlight.project_id" {
			got = kv.Value.AsString()
		}
	}
	if got != "env-sdk-key" {
		t.Errorf("highlight.project_id = %q, want the LD_SDK_KEY value", got)
	}
}

func TestSetLoggerIsSafeWhileLogging(t *testing.T) {
	t.Cleanup(func() { SetLogger(nil) })
	r := NewRegistry(RegistryEntries{})
	var wg sync.WaitGroup
	for i := range 8 {
		wg.Go(func() {
			if i%2 == 0 {
				SetLogger(slog.New(slog.NewTextHandler(io.Discard, nil)))
				return
			}
			// A duplicate tool logs a warning through the SDK logger.
			r.Register(RegistryEntries{Tools: map[string]ToolHandler{"t": ToolFunc(nil)}})
			r.Register(RegistryEntries{Tools: map[string]ToolHandler{"t": ToolFunc(nil)}})
		})
	}
	wg.Wait()
	SetLogger(nil)
	if sdkLogger() != slog.Default() {
		t.Error("SetLogger(nil) did not restore slog.Default()")
	}
}

func TestSlowInitDoesNotBlockGetClientOrTracking(t *testing.T) {
	release := make(chan struct{})
	var releaseOnce sync.Once
	unblock := func() { releaseOnce.Do(func() { close(release) }) }
	started := make(chan struct{})
	f := &fakeLD{}
	withLDFactory(t, func(string) (LDClient, error) {
		close(started)
		<-release // LaunchDarkly is still connecting
		return f, nil
	})
	done := make(chan error, 1)
	go func() {
		_, err := InitClient(bg, InitOptions{SDKKey: "k", OTLPEndpoint: unreachableOTLP})
		done <- err
	}()
	<-started
	t.Cleanup(unblock) // registered after withLDFactory, so it runs before its Shutdown

	returned := make(chan struct{})
	go func() {
		_, _ = GetClient()
		Track("$ld:ai:generation:success", userCtx, TrackData{}, 1)
		reportUsage("client.inspectConfig", "", "")
		close(returned)
	}()
	select {
	case <-returned:
	case <-time.After(2 * time.Second):
		t.Fatal("GetClient / Track / reportUsage waited for an in-flight InitClient")
	}

	unblock()
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if c, err := GetClient(); err != nil || c != f {
		t.Errorf("after init: %v %v", c, err)
	}
}

func TestHeldUsageIsDedupedAndDroppedOnShutdown(t *testing.T) {
	_ = Shutdown(bg)
	t.Cleanup(func() { _ = Shutdown(bg) })
	for range 100 { // a client that never initializes: calls keep coming
		reportUsage("client.inspectConfig", "", "")
	}
	reportMu.Lock()
	held := len(pendingUsage)
	reportMu.Unlock()
	if held != 1 {
		t.Fatalf("held usage records = %d, want 1", held)
	}
	_ = Shutdown(bg) // drops the held record
	f := &fakeLD{}
	if _, err := InitWithClient(bg, f, InitOptions{OTLPEndpoint: unreachableOTLP}); err != nil {
		t.Fatal(err)
	}
	if u := f.events("$ld:ai:sdk:usage"); len(u) != 0 {
		t.Errorf("a record held before Shutdown was sent to the next client: %v", u)
	}
}

// The feature_flag event names its provider and, when LD_ENVIRONMENT_ID is
// set, the environment (feature_flag.set.id), which also rides on every
// event payload as environmentId. A span without `__ld` still gets the
// dashboard's operation-type filter.
func TestFeatureFlagEventAndEnvironmentID(t *testing.T) {
	exp := withSpans(t)
	ffEvent := func() map[string]string {
		out := map[string]string{}
		for _, e := range exp.GetSpans()[0].Events {
			if e.Name == "feature_flag" {
				for _, a := range e.Attributes {
					out[string(a.Key)] = a.Value.AsString()
				}
			}
		}
		return out
	}
	_, span := otel.Tracer("t").Start(bg, "bare")
	SetLDSpanAttributes(span, nil)
	span.End()
	s := exp.GetSpans()[0]
	if len(s.Attributes) != 1 || s.Attributes[0] != attribute.String("launchdarkly.operation.type", "gen_ai") || len(s.Events) != 0 {
		t.Errorf("span without __ld: attrs=%v events=%v", s.Attributes, s.Events)
	}

	exp.Reset()
	_, span = otel.Tracer("t").Start(bg, "no-env")
	SetLDSpanAttributes(span, map[string]any{VarLD: TrackData{ConfigKey: "k"}})
	span.End()
	if ev := ffEvent(); ev["feature_flag.provider.name"] != "LaunchDarkly" || ev["feature_flag.key"] != "k" {
		t.Errorf("feature_flag event = %v", ev)
	} else if _, has := ev["feature_flag.set.id"]; has {
		t.Error("set.id written without an environment id")
	}

	t.Setenv("LD_ENVIRONMENT_ID", "env-1")
	f := withFake(t, map[string]any{"cfg": enabledConfig("P", "", nil)})
	var seen HandlerRequest
	if _, err := Config(ConfigArgs{Key: "cfg", Handlers: []*ProviderHandler{stubHandler("P", ModeMessages, "ok", nil, &seen)}}).Invoke(bg, "q", userCtx); err != nil {
		t.Fatal(err)
	}
	if ev := f.events("$ld:ai:generation:success"); len(ev) != 1 || ev[0].Data["environmentId"] != "env-1" {
		t.Errorf("generation payload = %+v", ev)
	}
	exp.Reset()
	_, span = otel.Tracer("t").Start(bg, "env")
	SetLDSpanAttributes(span, seen.Variables)
	span.End()
	if ev := ffEvent(); ev["feature_flag.set.id"] != "env-1" {
		t.Errorf("feature_flag event = %v", ev)
	}
}
