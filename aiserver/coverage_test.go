package aiserver

import (
	"compress/gzip"
	"context"
	"errors"
	"fmt"
	"io"
	"iter"
	"log/slog"
	"maps"
	"net/http"
	"net/http/httptest"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

// Tests for behaviour not covered elsewhere.

// countingHandler is a slog spy: it counts every record at any level.
type countingHandler struct {
	mu sync.Mutex
	n  int
}

func (h *countingHandler) Enabled(context.Context, slog.Level) bool { return true }
func (h *countingHandler) Handle(context.Context, slog.Record) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.n++
	return nil
}
func (h *countingHandler) WithAttrs([]slog.Attr) slog.Handler { return h }
func (h *countingHandler) WithGroup(string) slog.Handler      { return h }
func (h *countingHandler) count() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.n
}

func TestParseJSONInvalidLogsNothing(t *testing.T) {
	spy := &countingHandler{}
	prev := slog.Default()
	SetLogger(slog.New(spy))
	slog.SetDefault(slog.New(spy))
	t.Cleanup(func() { SetLogger(nil); slog.SetDefault(prev) })
	if v, ok := ParseJSONWithPossibleFences("not json"); ok || v != nil {
		t.Errorf("got %v %v", v, ok)
	}
	if n := spy.count(); n != 0 {
		t.Errorf("invalid JSON logged %d records", n)
	}
}

func TestParseAiConfigOptionalFieldsAndSkillEdges(t *testing.T) {
	cfg := base()
	cfg["tools"] = map[string]any{"t": map[string]any{"name": "t", "type": "function", "parameters": map[string]any{}}}
	cfg["judgeConfiguration"] = map[string]any{"judges": []any{map[string]any{"key": "j", "samplingRate": 1.0}}}
	cfg["evaluationMetricKey"] = "accuracy"
	got, err := ParseAiConfig(cfg)
	if err != nil || got.EvaluationMetricKey != "accuracy" || got.Tools["t"].Name != "t" || got.JudgeConfiguration.Judges[0].Key != "j" {
		t.Errorf("optional fields: %+v %v", got, err)
	}
	for _, s := range []any{
		[]any{map[string]any{"key": ".x", "version": 1.0}},
		[]any{map[string]any{"key": "k", "version": -1.0}},
	} {
		if _, err := ParseAiConfig(with("skills", s)); err == nil {
			t.Errorf("skills %v accepted", s)
		}
	}
}

func handlerSet(hs []*ProviderHandler) map[*ProviderHandler]bool {
	out := map[*ProviderHandler]bool{}
	for _, h := range hs {
		out[h] = true
	}
	return out
}

func TestRegistryConstructorAndCompose(t *testing.T) {
	h := &ProviderHandler{ProvidesFor: ProvidesFor{"P", ModeMessages}}
	fn := ToolFunc(func(context.Context, map[string]any) (any, error) { return "t", nil })
	r := NewRegistry(RegistryEntries{Handlers: []*ProviderHandler{h}, Tools: map[string]ToolHandler{"t": fn}})
	if hs := r.Handlers(); len(hs) != 1 || hs[0] != h {
		t.Errorf("handlers = %v", hs)
	}
	if tools := r.Tools(); len(tools) != 1 || tools["t"] == nil {
		t.Errorf("tools = %v", tools)
	}

	tool := func(s string) ToolFunc {
		return func(context.Context, map[string]any) (any, error) { return s, nil }
	}
	aShared := &ProviderHandler{ProvidesFor: ProvidesFor{"S", ModeMessages}}
	bShared := &ProviderHandler{ProvidesFor: ProvidesFor{"S", ModeMessages}}
	aOnly := &ProviderHandler{ProvidesFor: ProvidesFor{"A", ModeMessages}}
	bOnly := &ProviderHandler{ProvidesFor: ProvidesFor{"B", ModeAgent}}
	a := NewRegistry(RegistryEntries{Handlers: []*ProviderHandler{aShared, aOnly}, Tools: map[string]ToolHandler{"shared": tool("a"), "a-only": tool("a")}})
	b := NewRegistry(RegistryEntries{Handlers: []*ProviderHandler{bShared, bOnly}, Tools: map[string]ToolHandler{"shared": tool("b"), "b-only": tool("b")}})
	c := Compose(a, b)
	if c == a || c == b {
		t.Fatal("Compose returned an input")
	}
	got := handlerSet(c.Handlers())
	if len(got) != 3 || !got[bShared] || !got[aOnly] || !got[bOnly] || got[aShared] {
		t.Errorf("composed handlers: aShared=%v bShared=%v aOnly=%v bOnly=%v n=%d", got[aShared], got[bShared], got[aOnly], got[bOnly], len(got))
	}
	tools := c.Tools()
	if keys := slices.Sorted(maps.Keys(tools)); !slices.Equal(keys, []string{"a-only", "b-only", "shared"}) {
		t.Errorf("composed tools = %v", keys)
	}
	if v, _ := tools["shared"].(ToolFunc)(bg, nil); v != "b" {
		t.Errorf("tool conflict: a won (%v)", v)
	}
	if len(a.Handlers()) != 2 || len(a.Tools()) != 2 || len(b.Handlers()) != 2 || len(b.Tools()) != 2 {
		t.Error("Compose mutated an input")
	}
	if v, _ := a.Tools()["shared"].(ToolFunc)(bg, nil); v != "a" {
		t.Error("Compose rewrote a's tool")
	}
}

func TestResolveHandlersAndTools(t *testing.T) {
	local := []*ProviderHandler{{ProvidesFor: ProvidesFor{"L", ModeMessages}}}
	if got := resolveHandlers(nil, local); len(got) != 1 || got[0] != local[0] {
		t.Errorf("no registry: %v", got)
	}
	lt := map[string]ToolHandler{"x": ToolFunc(func(context.Context, map[string]any) (any, error) { return "local", nil })}
	if got := resolveTools(nil, lt); reflect.ValueOf(got).UnsafePointer() != reflect.ValueOf(lt).UnsafePointer() {
		t.Error("no registry: local tools not returned unchanged")
	}
	reg := NewRegistry(RegistryEntries{Tools: map[string]ToolHandler{
		"x": ToolFunc(func(context.Context, map[string]any) (any, error) { return "registry", nil }),
		"y": ToolFunc(func(context.Context, map[string]any) (any, error) { return "registry", nil }),
	}})
	if got := resolveTools(reg, nil); len(got) != 2 || got["x"] == nil || got["y"] == nil {
		t.Errorf("registry only: %v", got)
	}
	got := resolveTools(reg, lt)
	if v, _ := got["x"].(ToolFunc)(bg, nil); v != "local" || got["y"] == nil || len(got) != 2 {
		t.Errorf("overlay: x=%v tools=%v", v, got)
	}
	if got := resolveTools(NewRegistry(RegistryEntries{}), nil); got != nil {
		t.Errorf("empty registry, no local tools: %v, want nil", got)
	}
}

type toolToken struct{ n int }

func TestWrappedToolsPreserveResults(t *testing.T) {
	if got := WrapToolHandlers(nil, userCtx, TrackData{}); got == nil || len(got) != 0 {
		t.Errorf("WrapToolHandlers(nil) = %#v, want a non-nil empty map", got)
	}
	f := withFake(t, nil)
	token := &toolToken{1}
	broke := errors.New("tool broke")
	tools := map[string]ToolHandler{
		"ok":   ToolFunc(func(context.Context, map[string]any) (any, error) { return token, nil }),
		"fail": ToolFunc(func(context.Context, map[string]any) (any, error) { return nil, broke }),
	}
	wrapped := WrapToolHandlers(tools, userCtx, TrackData{})
	if v, err := wrapped["ok"].(ToolFunc)(bg, nil); v != token || err != nil {
		t.Errorf("tracking wrapper changed the result: %v %v", v, err)
	}
	f.tracks = nil
	// Through the recorder and the tracking wrapper together, as a handler sees it.
	var gotV any
	var gotErr error
	h := createHandler(ProvidesFor{"P", ModeMessages}, func(ctx context.Context, req HandlerRequest) (HandlerResult, error) {
		gotV, _ = req.ToolHandlers["ok"].(ToolFunc)(ctx, map[string]any{"a": 1.0})
		_, gotErr = req.ToolHandlers["fail"].(ToolFunc)(ctx, nil)
		return HandlerResult{Output: "done"}, nil
	}, nil, false)
	cfg := AiConfig{Tools: map[string]Tool{"ok": {Name: "ok", Type: "function"}, "fail": {Name: "fail", Type: "function"}}}
	res, err := executeAndTrack(bg, execArgs{configKey: "cfg", config: cfg, userCtx: userCtx, handler: h, tools: tools})
	if err != nil {
		t.Fatal(err)
	}
	if gotV != token || gotErr != broke {
		t.Errorf("handler saw %v / %v, want the tool's own value and error", gotV, gotErr)
	}
	if !strings.Contains(res.Trajectory, "error: tool broke") || !strings.Contains(res.Trajectory, "result: {}") {
		t.Errorf("trajectory = %s", res.Trajectory)
	}
	if n := len(f.events("$ld:ai:tool_call")); n != 2 {
		t.Errorf("tool_call events = %d", n)
	}
}

// restorePackagesAfter undoes RegisterAiSdkPackage calls made by a test:
// registration is process-global, and a leaked package changes every later
// test's sdk:info count.
func restorePackagesAfter(t *testing.T) {
	reportMu.Lock()
	prev := slices.Clone(knownPkgs)
	reportMu.Unlock()
	t.Cleanup(func() {
		reportMu.Lock()
		knownPkgs = prev
		reportMu.Unlock()
	})
}

func TestSdkInfoPayloadAndRegistry(t *testing.T) {
	_ = Shutdown(bg)
	t.Cleanup(func() { _ = Shutdown(bg) })
	prev := otel.GetTracerProvider()
	t.Cleanup(func() { otel.SetTracerProvider(prev) })
	restorePackagesAfter(t)
	RegisterAiSdkPackage("two-version-pkg", "1.0.0")
	RegisterAiSdkPackage("two-version-pkg", "2.0.0")
	f := &fakeLD{}
	if _, err := InitWithClient(bg, f, InitOptions{OTLPEndpoint: unreachableOTLP}); err != nil {
		t.Fatal(err)
	}
	var versions []string
	sawCore := false
	for _, e := range f.events("$ld:ai:sdk:info") {
		if keys := slices.Sorted(maps.Keys(e.Data)); !slices.Equal(keys, []string{"aiSdkLanguage", "aiSdkName", "aiSdkVersion"}) {
			t.Errorf("sdk:info keys = %v", keys)
		}
		if e.Data["aiSdkLanguage"] != "go" || e.Metric != 1 {
			t.Errorf("sdk:info = %+v", e)
		}
		if !e.Ctx.Equal(internalContext) || e.Ctx.Kind() != "ld_ai" || e.Ctx.Key() != "ld-internal-tracking" || !e.Ctx.Anonymous() {
			t.Errorf("sdk:info context = %v", e.Ctx)
		}
		switch e.Data["aiSdkName"] {
		case "two-version-pkg":
			versions = append(versions, e.Data["aiSdkVersion"].(string))
		case PackageName:
			sawCore = e.Data["aiSdkVersion"] == PackageVersion
		}
	}
	if slices.Sort(versions); !slices.Equal(versions, []string{"1.0.0", "2.0.0"}) || !sawCore {
		t.Errorf("two versions of one package: %v, core reported: %v", versions, sawCore)
	}

	// A panicking track fails nothing, and the package counts as reported.
	_ = Shutdown(bg)
	p := &fakeLD{trackPanic: true}
	if _, err := InitWithClient(bg, p, InitOptions{}); err != nil {
		t.Fatalf("a panicking track failed init: %v", err)
	}
	p.trackPanic = false
	if _, err := InitClient(bg, InitOptions{}); err != nil {
		t.Fatal(err)
	}
	if n := len(p.events("$ld:ai:sdk:info")); n != 0 {
		t.Errorf("sdk:info retried %d times after a panicking track", n)
	}
}

func TestInitClientSDKKeyOptionBeatsEnv(t *testing.T) {
	t.Setenv("LD_SDK_KEY", "env-key")
	var got []string
	withLDFactory(t, func(key string) (LDClient, error) { got = append(got, key); return &fakeLD{}, nil })
	if _, err := InitClient(bg, InitOptions{SDKKey: "option-key", OTLPEndpoint: unreachableOTLP}); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, []string{"option-key"}) {
		t.Errorf("client built with %v", got)
	}
}

// Propagators, conversation-id processor, gzip export,
// and Shutdown flushing the client and shutting down the tracer provider.
func TestInitClientTelemetryAndShutdown(t *testing.T) {
	var mu sync.Mutex
	var encodings, bodies, paths []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body := []byte{}
		if r.Header.Get("Content-Encoding") == "gzip" {
			if zr, err := gzip.NewReader(r.Body); err == nil {
				body, _ = io.ReadAll(zr)
			}
		}
		mu.Lock()
		paths = append(paths, r.URL.Path)
		encodings = append(encodings, r.Header.Get("Content-Encoding"))
		bodies = append(bodies, string(body))
		mu.Unlock()
		w.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(srv.Close)
	prevTP, prevProp := otel.GetTracerProvider(), otel.GetTextMapPropagator()
	t.Cleanup(func() { otel.SetTracerProvider(prevTP); otel.SetTextMapPropagator(prevProp) })
	f := &fakeLD{}
	withLDFactory(t, func(string) (LDClient, error) { return f, nil })
	if _, err := InitClient(bg, InitOptions{SDKKey: "k", OTLPEndpoint: srv.URL + "/", Environment: "test-env"}); err != nil {
		t.Fatal(err)
	}
	fields := otel.GetTextMapPropagator().Fields()
	for _, want := range []string{"traceparent", "tracestate", "baggage"} {
		if !slices.Contains(fields, want) {
			t.Errorf("propagator fields %v lack %s", fields, want)
		}
	}
	tp := tracerProvider
	_, span := otel.Tracer(TracerPrefix+"/probe").Start(WithConversationID(bg, "conv-init"), "probe-span")
	span.End()

	if err := Shutdown(bg); err != nil {
		t.Fatalf("shutdown: %v", err)
	}
	if f.flushes != 1 || f.closed != 1 {
		t.Errorf("flushes=%d closed=%d", f.flushes, f.closed)
	}
	if _, s := tp.Tracer("after").Start(bg, "after"); s.IsRecording() {
		t.Error("tracer provider was not shut down")
	}
	mu.Lock()
	defer mu.Unlock()
	if len(encodings) == 0 {
		t.Fatal("no OTLP export reached the endpoint")
	}
	for i, enc := range encodings {
		if enc != "gzip" {
			t.Errorf("export %d Content-Encoding = %q", i, enc)
		}
	}
	for _, p := range paths {
		if p != "/v1/traces" {
			t.Errorf("exported to %q, want /v1/traces under the endpoint", p)
		}
	}
	all := strings.Join(bodies, "")
	for _, want := range []string{"launchdarkly.project_id", "service.name", "go-sdk", "deployment.environment", "test-env"} {
		if !strings.Contains(all, want) {
			t.Errorf("exported resource lacks %q", want)
		}
	}
	if !strings.Contains(all, "probe-span") || !strings.Contains(all, "gen_ai.conversation.id") || !strings.Contains(all, "conv-init") {
		t.Error("exported span lacks the conversation id: the processor did not run before the exporter")
	}
}

func TestBYOCNeverBuildsAClient(t *testing.T) {
	built := 0
	withLDFactory(t, func(string) (LDClient, error) { built++; return &fakeLD{}, nil })
	prev := otel.GetTracerProvider()
	t.Cleanup(func() { otel.SetTracerProvider(prev) })
	if _, err := InitWithClient(bg, &fakeLD{}, InitOptions{SDKKey: "k", OTLPEndpoint: unreachableOTLP}); err != nil {
		t.Fatal(err)
	}
	if _, err := InitClient(bg, InitOptions{SDKKey: "k"}); err != nil {
		t.Fatal(err)
	}
	if built != 0 {
		t.Errorf("BYOC built %d LaunchDarkly clients", built)
	}
}

func TestConfigInvokeForwardsContextAndRequest(t *testing.T) {
	f := withFake(t, map[string]any{"cfg": enabledConfig("P", "", nil)})
	orgCtx := ldcontext.NewBuilder("org-1").Kind("org").SetString("plan", "pro").Build()
	var seen HandlerRequest
	h := stubHandler("P", ModeMessages, "ok", nil, &seen)
	tools := map[string]ToolHandler{"lookup": ToolFunc(func(context.Context, map[string]any) (any, error) { return "x", nil })}
	if _, err := Config(ConfigArgs{Key: "cfg", Handlers: []*ProviderHandler{h}, ToolHandlers: tools}).
		Invoke(bg, "the input", orgCtx, WithVariables(map[string]any{"v": "1"})); err != nil {
		t.Fatal(err)
	}
	if ctxs := f.evalCtxs["cfg"]; len(ctxs) != 1 || !ctxs[0].Equal(orgCtx) {
		t.Errorf("JSONVariation contexts = %v", ctxs)
	}
	if seen.UserInput != "the input" || seen.Config.Model.Name != "test-model" || seen.Config.Instructions != "be helpful" || seen.Config.Provider.Name != "P" {
		t.Errorf("request = %+v", seen)
	}
	if _, ok := seen.ToolHandlers["lookup"].(ToolFunc); !ok || seen.Variables["v"] != "1" || seen.Variables[VarLDContext].(map[string]any)["key"] != "org-1" {
		t.Errorf("tools=%v variables=%v", seen.ToolHandlers, seen.Variables)
	}
}

func TestConfigModeMismatchAndMissingProvider(t *testing.T) {
	noProvider := enabledConfig("", "", nil)
	withFake(t, map[string]any{"agent": enabledConfig("OpenAI", "agent", nil), "noprov": noProvider})
	h := stubHandler("OpenAI", ModeMessages, "ok", nil, nil)
	if _, err := Config(ConfigArgs{Key: "agent", Handlers: []*ProviderHandler{h}}).Invoke(bg, "q", userCtx); !errors.Is(err, ErrNoHandler) {
		t.Errorf("mode mismatch err = %v, want ErrNoHandler", err)
	}
	if _, err := Config(ConfigArgs{Key: "noprov", Handlers: []*ProviderHandler{stubHandler("*", ModeMessages, "ok", nil, nil)}}).Invoke(bg, "q", userCtx); err == nil {
		t.Error(`provider.name "" must error`)
	}
}

func tokenEvents(f *fakeLD) int {
	return len(f.events("$ld:ai:tokens:total")) + len(f.events("$ld:ai:tokens:input")) + len(f.events("$ld:ai:tokens:output"))
}

func streamHandler(provider string, out string, usage map[string]any, err error, called *int) *ProviderHandler {
	return createHandler(ProvidesFor{provider, ModeMessages}, func(context.Context, HandlerRequest) (HandlerResult, error) {
		return HandlerResult{Output: `{"score":1,"reasoning":"judged"}`}, nil
	}, func(_ context.Context, req HandlerRequest) iter.Seq2[HandlerStreamEvent, error] {
		return func(yield func(HandlerStreamEvent, error) bool) {
			if called != nil {
				*called++
			}
			if !yield(HandlerStreamEvent{Type: EventChunk, Text: out}, nil) {
				return
			}
			if err != nil {
				yield(HandlerStreamEvent{}, err)
				return
			}
			yield(HandlerStreamEvent{Type: EventDone, Usage: usage}, nil)
		}
	}, false)
}

func drain(seq iter.Seq2[StreamEvent, error]) (events []StreamEvent, err error) {
	for ev, e := range seq {
		if e != nil {
			return events, e
		}
		events = append(events, ev)
	}
	return events, nil
}

func TestZeroUsageEmitsNoTokenEvents(t *testing.T) {
	f := withFake(t, map[string]any{"cfg": enabledConfig("P", "", nil)})
	zero := map[string]any{"input": 0.0, "output": 0.0}
	if _, err := Config(ConfigArgs{Key: "cfg", Handlers: []*ProviderHandler{stubHandler("P", ModeMessages, "ok", zero, nil)}}).Invoke(bg, "q", userCtx); err != nil {
		t.Fatal(err)
	}
	if _, err := drain(Config(ConfigArgs{Key: "cfg", Handlers: []*ProviderHandler{streamHandler("P", "ok", zero, nil, nil)}}).Stream(bg, "q", userCtx)); err != nil {
		t.Fatal(err)
	}
	if n := tokenEvents(f); n != 0 || len(f.events("$ld:ai:generation:success")) != 2 {
		t.Errorf("token events = %d", n)
	}
	// Non-zero stream usage does emit.
	if _, err := drain(Config(ConfigArgs{Key: "cfg", Handlers: []*ProviderHandler{streamHandler("P", "ok", map[string]any{"input": 2.0, "output": 3.0}, nil, nil)}}).Stream(bg, "q", userCtx)); err != nil {
		t.Fatal(err)
	}
	if tot := f.events("$ld:ai:tokens:total"); len(tot) != 1 || tot[0].Metric != 5 || len(f.events("$ld:ai:tokens:input")) != 1 || len(f.events("$ld:ai:tokens:output")) != 1 {
		t.Errorf("stream token events = %d, total = %v", tokenEvents(f), f.events("$ld:ai:tokens:total"))
	}
}

func TestSkipJudgesWithoutJudgeConfiguration(t *testing.T) {
	f := withFake(t, map[string]any{"cfg": enabledConfig("P", "", nil)})
	h := stubHandler("P", ModeMessages, "ok", nil, nil)
	resp, err := Config(ConfigArgs{Key: "cfg", Handlers: []*ProviderHandler{h}, SkipJudges: true}).Invoke(bg, "q", userCtx)
	if err != nil || resp.JudgeTasks == nil || len(resp.JudgeTasks) != 0 || resp.JudgeResults != nil {
		t.Errorf("tasks=%#v results=%v err=%v", resp.JudgeTasks, resp.JudgeResults, err)
	}
	tasks, err := BuildJudgeTasks(bg, BuildJudgeTasksArgs{Config: AiConfig{JudgeConfiguration: &JudgeConfiguration{Judges: []JudgeRef{}}}, UserContext: userCtx, Handler: h})
	if err != nil || tasks == nil || len(tasks) != 0 {
		t.Errorf("empty judges: %#v %v", tasks, err)
	}
	if !maps.Equal(f.calls, map[string]int{"cfg": 1}) {
		t.Errorf("variations evaluated = %v, want only cfg once", f.calls)
	}
}

func TestStructuredObjectOutputReturnedAsIs(t *testing.T) {
	withFake(t, map[string]any{"cfg": enabledConfig("P", "", map[string]any{"outputFormat": map[string]any{"type": "object"}})})
	out := map[string]any{"answer": 42.0}
	h := createHandler(ProvidesFor{"P", ModeMessages}, func(context.Context, HandlerRequest) (HandlerResult, error) {
		return HandlerResult{Output: out}, nil
	}, nil, false)
	resp, err := Config(ConfigArgs{Key: "cfg", Handlers: []*ProviderHandler{h}}).Invoke(bg, "q", userCtx)
	if err != nil {
		t.Fatal(err)
	}
	m, ok := resp.Response.(map[string]any)
	if !ok || reflect.ValueOf(m).UnsafePointer() != reflect.ValueOf(out).UnsafePointer() {
		t.Errorf("response = %#v, want the handler's own map", resp.Response)
	}
}

// The inline judge sees exactly the judge keys, with the
// trajectory inside message_history and no separate variable for it.
func TestInlineJudgeVariablesAreExact(t *testing.T) {
	tools := map[string]any{"lookup": map[string]any{"name": "lookup", "type": "function", "parameters": map[string]any{}}}
	withFake(t, map[string]any{
		"cfg":   enabledConfig("P", "", judgedConfig(map[string]any{"tools": tools})),
		"judge": enabledConfig("P", "judge", nil),
	})
	var judgeVars map[string]any
	h := createHandler(ProvidesFor{"P", ModeMessages}, func(ctx context.Context, req HandlerRequest) (HandlerResult, error) {
		if _, isJudge := req.Variables["message_history"]; isJudge {
			judgeVars = req.Variables
			return HandlerResult{Output: `{"score":1,"reasoning":"r"}`}, nil
		}
		_, _ = req.ToolHandlers["lookup"].(ToolFunc)(ctx, nil)
		return HandlerResult{Output: "answer"}, nil
	}, nil, false)
	th := map[string]ToolHandler{"lookup": ToolFunc(func(context.Context, map[string]any) (any, error) { return "found", nil })}
	if _, err := Config(ConfigArgs{Key: "cfg", Handlers: []*ProviderHandler{h}, ToolHandlers: th}).Invoke(bg, "q", userCtx, WithVariables(map[string]any{"caller": "x"})); err != nil {
		t.Fatal(err)
	}
	if keys := slices.Sorted(maps.Keys(judgeVars)); !slices.Equal(keys, judgeVisibleKeys) {
		t.Errorf("judge variables = %v, want %v", keys, judgeVisibleKeys)
	}
	if !strings.Contains(judgeVars["message_history"].(string), "1. lookup") {
		t.Error("trajectory missing from message_history")
	}
}

func TestGraphJudgeTrajectoryScope(t *testing.T) {
	tool := func(n string) map[string]any {
		return map[string]any{n: map[string]any{"name": n, "type": "function", "parameters": map[string]any{}}}
	}
	root := nodeCfg("root work")
	root["tools"] = tool("alpha")
	leaf := enabledConfig("P", "agent", judgedConfig(map[string]any{"instructions": "leaf work", "tools": tool("beta")}))
	withFake(t, twoNode(map[string]any{"root-node": root, "leaf-node": leaf,
		"judge": enabledConfig("P", "judge", nil), "gj": enabledConfig("P", "judge", nil)}))
	histories := map[string]string{}
	h := createHandler(ProvidesFor{"P", ModeAgent}, func(ctx context.Context, req HandlerRequest) (HandlerResult, error) {
		if mh, isJudge := req.Variables["message_history"].(string); isJudge {
			histories[req.Variables[VarLD].(TrackData).ConfigKey] = mh
			return HandlerResult{Output: `{"score":1,"reasoning":"r"}`}, nil
		}
		for name := range req.Config.Tools {
			if fn, ok := req.ToolHandlers[name].(ToolFunc); ok {
				_, _ = fn(ctx, nil)
			}
		}
		return HandlerResult{Output: "out"}, nil
	}, nil, false)
	th := map[string]ToolHandler{}
	for _, n := range []string{"alpha", "beta"} {
		th[n] = ToolFunc(func(context.Context, map[string]any) (any, error) { return n + " result", nil })
	}
	if _, err := Graph("g", GraphOptions{Handlers: []*ProviderHandler{h}, ToolHandlers: th, GraphJudge: "gj"}).Invoke(bg, "q", userCtx); err != nil {
		t.Fatal(err)
	}
	node, graph := histories["judge"], histories["gj"]
	if !strings.Contains(node, "1. beta") || strings.Contains(node, "alpha") {
		t.Errorf("per-node judge history = %q", node)
	}
	if graph == "" || strings.Contains(graph, "Tool") || strings.Contains(graph, "alpha") || strings.Contains(graph, "beta") {
		t.Errorf("graph judge history = %q", graph)
	}
}

func TestBuildJudgeTasksCollapseAndExclusion(t *testing.T) {
	withFake(t, map[string]any{"j-p": enabledConfig("P", "judge", nil), "j-other": enabledConfig("Other", "judge", nil)})
	cfg := AiConfig{JudgeConfiguration: &JudgeConfiguration{Judges: []JudgeRef{{Key: "j-p", SamplingRate: 1}, {Key: "j-other", SamplingRate: 1}}}}
	wildAgent := &ProviderHandler{ProvidesFor: ProvidesFor{"*", ModeAgent}}
	tasks, err := BuildJudgeTasks(bg, BuildJudgeTasksArgs{Config: cfg, UserContext: userCtx, Handler: wildAgent, Handlers: []*ProviderHandler{wildAgent}})
	if err != nil || len(tasks) != 2 || !tasks[0].CollapseMessages || !tasks[1].CollapseMessages {
		t.Errorf("wildcard agent: %s %v", taskSummary(tasks), err)
	}
	p := &ProviderHandler{ProvidesFor: ProvidesFor{"P", ModeMessages}}
	tasks, err = BuildJudgeTasks(bg, BuildJudgeTasksArgs{Config: cfg, UserContext: userCtx, Handler: p, Handlers: []*ProviderHandler{p}})
	if err != nil || len(tasks) != 1 || tasks[0].ConfigKey != "j-p" || tasks[0].CollapseMessages {
		t.Errorf("provider exclusion: %s %v", taskSummary(tasks), err)
	}
}

func taskSummary(tasks []JudgeTask) string {
	var parts []string
	for _, tk := range tasks {
		parts = append(parts, fmt.Sprintf("%s(collapse=%v)", tk.ConfigKey, tk.CollapseMessages))
	}
	return "[" + strings.Join(parts, " ") + "]"
}

func TestConfigStreamTrackingJudgesAndRouting(t *testing.T) {
	f := withFake(t, map[string]any{
		"cfg":     enabledConfig("Q", "", nil),
		"judged":  enabledConfig("P", "", judgedConfig(nil)),
		"judge":   enabledConfig("P", "judge", nil),
		"nomatch": enabledConfig("Z", "", nil),
	})
	var pCalls, qCalls int
	p, q := streamHandler("P", "p-out", nil, nil, &pCalls), streamHandler("Q", "q-out", nil, nil, &qCalls)
	events, err := drain(Config(ConfigArgs{Key: "cfg", Handlers: []*ProviderHandler{p, q}}).Stream(bg, "q", userCtx))
	if err != nil || pCalls != 0 || qCalls != 1 || events[len(events)-1].Response != "q-out" {
		t.Errorf("routing: p=%d q=%d events=%+v err=%v", pCalls, qCalls, events, err)
	}
	if len(f.events("$ld:ai:duration:total")) != 1 {
		t.Error("stream did not track duration")
	}
	if done := events[len(events)-1]; done.Type != EventDone || done.JudgeResults != nil {
		t.Errorf("no judges: done = %+v", done)
	}

	// Handler stream error: generation:error + duration, and the error surfaces.
	boom := errors.New("stream broke")
	f.tracks = nil
	_, err = drain(Config(ConfigArgs{Key: "cfg", Handlers: []*ProviderHandler{streamHandler("Q", "partial", nil, boom, nil)}}).Stream(bg, "q", userCtx))
	if !errors.Is(err, boom) || len(f.events("$ld:ai:generation:error")) != 1 || len(f.events("$ld:ai:duration:total")) != 1 || len(f.events("$ld:ai:generation:success")) != 0 {
		t.Errorf("stream error: err=%v tracks=%v", err, f.tracks)
	}

	// Judge results ride on done.
	var seenVars map[string]any
	judged := createHandler(ProvidesFor{"P", ModeMessages}, func(context.Context, HandlerRequest) (HandlerResult, error) {
		return HandlerResult{Output: `{"score":0.4,"reasoning":"meh"}`}, nil
	}, func(_ context.Context, req HandlerRequest) iter.Seq2[HandlerStreamEvent, error] {
		seenVars = req.Variables
		return func(yield func(HandlerStreamEvent, error) bool) {
			yield(HandlerStreamEvent{Type: EventChunk, Text: "answer"}, nil)
		}
	}, false)
	events, err = drain(Config(ConfigArgs{Key: "judged", Handlers: []*ProviderHandler{judged}}).Stream(bg, "q", userCtx))
	if done := events[len(events)-1]; err != nil || done.JudgeResults["judge"].Score != 0.4 || done.JudgeResults["judge"].Response != "meh" {
		t.Errorf("judged done = %+v err=%v", done, err)
	}
	if lc, _ := seenVars[VarLDContext].(map[string]any); lc["key"] != "user-123" || lc["email"] != "user@example.com" {
		t.Errorf("stream handler ldContext = %v", seenVars[VarLDContext])
	}

	if _, err := drain(Config(ConfigArgs{Key: "nomatch", Handlers: []*ProviderHandler{p, q}}).Stream(bg, "q", userCtx)); !errors.Is(err, ErrNoHandler) {
		t.Errorf("no match err = %v", err)
	}
}

func drainGraph(seq iter.Seq2[GraphStreamEvent, error]) (events []GraphStreamEvent, err error) {
	for ev, e := range seq {
		if e != nil {
			return events, e
		}
		events = append(events, ev)
	}
	return events, nil
}

func TestGraphStreamParity(t *testing.T) {
	f := withFake(t, twoNode(map[string]any{"gj": enabledConfig("P", "judge", nil)}))
	if _, err := drainGraph(Graph("g", GraphOptions{}).Stream(bg, "q", userCtx)); err == nil {
		t.Error("no handlers must error")
	}

	// A node stream failure: invocation_failure, the entered root node, and the error.
	_, err := drainGraph(Graph("g", GraphOptions{Handlers: []*ProviderHandler{(&recordingHandler{stream: true, failOn: "root-node"}).handler()}}).Stream(bg, "q", userCtx))
	if err == nil || !strings.Contains(err.Error(), "node failed") {
		t.Errorf("err = %v", err)
	}
	if n := f.events("$ld:ai:graph:node"); len(n) != 1 || n[0].Data["nodeKey"] != "root-node" || n[0].Data["index"] != 0.0 {
		t.Errorf("node events = %+v", n)
	}
	if len(f.events("$ld:ai:graph:invocation_failure")) != 1 || len(f.events("$ld:ai:graph:invocation_success")) != 0 {
		t.Error("failure tracking wrong")
	}

	// Fallback to Invoke for a handler without Stream, with a graph judge.
	f.tracks = nil
	h := createHandler(ProvidesFor{"P", ModeAgent}, func(_ context.Context, req HandlerRequest) (HandlerResult, error) {
		if req.Variables["response_to_evaluate"] != nil {
			return HandlerResult{Output: `{"score":0.8,"reasoning":"good"}`}, nil
		}
		return HandlerResult{Output: req.Variables[VarLD].(TrackData).ConfigKey + " out", Usage: map[string]any{"input": 1.0, "output": 1.0}}, nil
	}, nil, false)
	events, err := drainGraph(Graph("g", GraphOptions{Handlers: []*ProviderHandler{h}, GraphJudge: "gj"}).Stream(bg, "q", userCtx))
	if err != nil {
		t.Fatal(err)
	}
	var shape []string
	for _, e := range events {
		shape = append(shape, string(e.Type)+":"+e.NodeKey+e.Text)
	}
	want := []string{"node_start:root-node", "chunk:root-noderoot-node out", "node_done:root-node", "handoff:",
		"node_start:leaf-node", "chunk:leaf-nodeleaf-node out", "node_done:leaf-node", "done:"}
	if !slices.Equal(shape, want) {
		t.Errorf("fallback events = %v", shape)
	}
	if done := events[len(events)-1]; done.JudgeResults["gj"].Score != 0.8 || done.Response != "leaf-node out" {
		t.Errorf("done = %+v", done)
	}
	if gen := f.events("$ld:ai:generation:success"); len(gen) != 3 || gen[0].Data["graphKey"] != "g" || gen[1].Data["graphKey"] != "g" {
		t.Errorf("per-node generation events = %+v", gen)
	}
	if len(f.events("$ld:ai:graph:handoff_success")) != 1 || len(f.events("$ld:ai:graph:duration:total")) != 1 || len(f.events("$ld:ai:graph:invocation_success")) != 1 {
		t.Errorf("stream graph tracking = %v", f.tracks)
	}
}

// Graph().Invoke handler spans and the graph judge's span all
// descend from ld.ai.graph in its trace.
func TestGraphInvokeSpanParenting(t *testing.T) {
	withFake(t, twoNode(map[string]any{"gj": enabledConfig("P", "judge", nil)}))
	exp := withSpans(t)
	rh := &recordingHandler{}
	if _, err := Graph("g", GraphOptions{Handlers: []*ProviderHandler{rh.handler()}, GraphJudge: "gj"}).Invoke(bg, "q", userCtx); err != nil {
		t.Fatal(err)
	}
	spans := exp.GetSpans().Snapshots()
	var graph sdktrace.ReadOnlySpan
	for _, s := range spans {
		if s.Name() == GraphSpanName {
			graph = s
		}
	}
	if graph == nil {
		t.Fatal("no graph span")
	}
	names := []string{}
	for _, s := range spans {
		if s.Name() == GraphSpanName {
			continue
		}
		names = append(names, s.Name())
		if s.Parent().SpanID() != graph.SpanContext().SpanID() || s.SpanContext().TraceID() != graph.SpanContext().TraceID() {
			t.Errorf("%s not parented under the graph span", s.Name())
		}
	}
	slices.Sort(names)
	if !slices.Equal(names, []string{"handler gj", "handler leaf-node", "handler root-node"}) {
		t.Errorf("spans = %v", names)
	}
}

func TestGraphSpanCarriesConversationID(t *testing.T) {
	withFake(t, twoNode(nil))
	exp := tracetest.NewInMemoryExporter()
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(ConversationIDSpanProcessor{}), sdktrace.WithSyncer(exp)))
	t.Cleanup(func() { otel.SetTracerProvider(prev) })
	rh := &recordingHandler{}
	if _, err := Graph("g", GraphOptions{Handlers: []*ProviderHandler{rh.handler()}}).Invoke(WithConversationID(bg, "conv-g"), "q", userCtx); err != nil {
		t.Fatal(err)
	}
	found := false
	for _, s := range exp.GetSpans().Snapshots() {
		if s.Name() != GraphSpanName {
			continue
		}
		for _, a := range s.Attributes() {
			found = found || (a.Key == "gen_ai.conversation.id" && a.Value.AsString() == "conv-g")
		}
		if s.Status().Code != codes.Ok {
			t.Error("graph span not OK")
		}
	}
	if !found {
		t.Error("ld.ai.graph lacks gen_ai.conversation.id")
	}
}

func TestContextKeysAttributeForEscapedKey(t *testing.T) {
	exp := tracetest.NewInMemoryExporter()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exp))
	c := ldcontext.NewBuilder("a%b:c").Kind("org").Build()
	_, span := tp.Tracer("t").Start(bg, "root")
	SetLDSpanAttributes(span, map[string]any{VarLD: TrackData{ConfigKey: "k"}, varLDContextTyped: c})
	span.End()
	got := ""
	for _, a := range exp.GetSpans().Snapshots()[0].Attributes() {
		if a.Key == "context.contextKeys.org" {
			got = a.Value.AsString()
		}
	}
	if got != "a%b:c" {
		t.Errorf("context.contextKeys.org = %q, want the raw key", got)
	}
}

func TestHelperUsagePerPublicFunction(t *testing.T) {
	_ = Shutdown(bg)
	t.Cleanup(func() { _ = Shutdown(bg) })
	h := (&recordingHandler{}).handler()
	rows := []struct {
		helper string
		call   func()
	}{
		{"client.graph.invoke", func() { _, _ = Graph("g", GraphOptions{Handlers: []*ProviderHandler{h}}).Invoke(bg, "q", userCtx) }},
		{"client.graph.stream", func() { _ = Graph("g", GraphOptions{Handlers: []*ProviderHandler{h}}).Stream(bg, "q", userCtx) }},
		{"client.resolveGraph", func() { _ = ResolveGraph(bg, "g", userCtx, GraphOptions{}) }},
		{"client.inspectConfig", func() { _ = InspectConfig(bg, "root-node", userCtx) }},
		{"client.runJudge", func() { _, _ = RunJudge(bg, JudgeTask{}, nil) }},
		{"client.buildJudgeTasks", func() { _, _ = BuildJudgeTasks(bg, BuildJudgeTasksArgs{}) }},
	}
	for _, r := range rows {
		_ = Shutdown(bg)
		f := &fakeLD{variations: twoNode(nil)}
		setClientForTest(f)
		r.call()
		usage := f.events("$ld:ai:sdk:usage")
		if len(usage) != 1 {
			t.Errorf("%s: %d usage events %v", r.helper, len(usage), usage)
			continue
		}
		u := usage[0]
		if u.Data["helper"] != r.helper || u.Data["aiSdkName"] != PackageName || u.Data["aiSdkVersion"] != PackageVersion ||
			u.Data["aiSdkLanguage"] != "go" || u.Data["helperPackageName"] != PackageName || u.Data["helperPackageVersion"] != u.Data["aiSdkVersion"] || u.Metric != 1 {
			t.Errorf("%s: payload = %+v", r.helper, u)
		}
		if !u.Ctx.Equal(internalContext) || !u.Ctx.Anonymous() || u.Ctx.Kind() != "ld_ai" {
			t.Errorf("%s: context = %v", r.helper, u.Ctx)
		}
	}

	// Shutdown re-arms reporting.
	_ = Shutdown(bg)
	first := &fakeLD{variations: twoNode(nil)}
	setClientForTest(first)
	_ = InspectConfig(bg, "root-node", userCtx)
	_ = InspectConfig(bg, "root-node", userCtx)
	_ = Shutdown(bg)
	second := &fakeLD{variations: twoNode(nil)}
	setClientForTest(second)
	_ = InspectConfig(bg, "root-node", userCtx)
	if len(first.events("$ld:ai:sdk:usage")) != 1 || len(second.events("$ld:ai:sdk:usage")) != 1 {
		t.Errorf("usage across shutdown: first=%d second=%d", len(first.events("$ld:ai:sdk:usage")), len(second.events("$ld:ai:sdk:usage")))
	}
}

// A native tool is not recorded or listed, in either judge path, but
// its call is still tracked.
func TestNativeToolOutOfTrajectory(t *testing.T) {
	tools := map[string]any{
		"lookup":     map[string]any{"name": "lookup", "type": "function", "parameters": map[string]any{}},
		"web_search": map[string]any{"name": "web_search", "type": "function", "parameters": map[string]any{}},
	}
	f := withFake(t, map[string]any{
		"cfg":   enabledConfig("P", "", judgedConfig(map[string]any{"tools": tools})),
		"judge": enabledConfig("P", "judge", nil),
	})
	var history string
	h := createHandler(ProvidesFor{"P", ModeMessages}, func(ctx context.Context, req HandlerRequest) (HandlerResult, error) {
		if mh, isJudge := req.Variables["message_history"].(string); isJudge {
			history = mh
			return HandlerResult{Output: `{"score":1,"reasoning":"r"}`}, nil
		}
		_, _ = req.ToolHandlers["lookup"].(ToolFunc)(ctx, nil)
		req.ToolHandlers["web_search"].(*NativeTool).Track()
		return HandlerResult{Output: "answer"}, nil
	}, nil, false)
	th := map[string]ToolHandler{
		"lookup":     ToolFunc(func(context.Context, map[string]any) (any, error) { return "found", nil }),
		"web_search": &NativeTool{ToolName: "web_search"},
	}
	args := ConfigArgs{Key: "cfg", Handlers: []*ProviderHandler{h}, ToolHandlers: th}
	if _, err := Config(args).Invoke(bg, "q", userCtx); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(history, "Tools available: lookup\n") || strings.Contains(history, "web_search") {
		t.Errorf("inline history = %s", history)
	}
	var keys []string
	for _, e := range f.events("$ld:ai:tool_call") {
		keys = append(keys, e.Data["toolKey"].(string))
	}
	if slices.Sort(keys); !slices.Equal(keys, []string{"lookup", "web_search"}) {
		t.Errorf("tool_call keys = %v", keys)
	}
	args.SkipJudges = true
	resp, err := Config(args).Invoke(bg, "q", userCtx)
	if err != nil || len(resp.JudgeTasks) != 1 {
		t.Fatalf("tasks=%v err=%v", resp.JudgeTasks, err)
	}
	if tr := resp.JudgeTasks[0].Trajectory; !strings.HasPrefix(tr, "Tools available: lookup\n") || strings.Contains(tr, "web_search") {
		t.Errorf("deferred trajectory = %s", tr)
	}
}

func TestFailedInvocationKeepsItsTrajectory(t *testing.T) {
	withFake(t, nil)
	boom := errors.New("model crashed")
	h := createHandler(ProvidesFor{"P", ModeMessages}, func(ctx context.Context, req HandlerRequest) (HandlerResult, error) {
		_, _ = req.ToolHandlers["lookup"].(ToolFunc)(ctx, map[string]any{"q": "x"})
		return HandlerResult{}, boom
	}, nil, false)
	cfg := AiConfig{Tools: map[string]Tool{"lookup": {Name: "lookup", Type: "function"}}}
	tools := map[string]ToolHandler{"lookup": ToolFunc(func(context.Context, map[string]any) (any, error) { return "found", nil })}
	res, err := executeAndTrack(bg, execArgs{configKey: "cfg", config: cfg, userCtx: userCtx, handler: h, tools: tools})
	if !errors.Is(err, boom) {
		t.Fatalf("err = %v", err)
	}
	if !strings.Contains(res.Trajectory, "1. lookup") || !strings.Contains(res.Trajectory, `arguments: {"q":"x"}`) || !strings.Contains(res.Trajectory, "result: found") {
		t.Errorf("failed unit trajectory = %q", res.Trajectory)
	}
}
