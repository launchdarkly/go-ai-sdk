package aiserver

import (
	"bytes"
	"context"
	"errors"
	"iter"
	"log/slog"
	"reflect"
	"strings"
	"testing"
	"time"

	ld "github.com/launchdarkly/go-server-sdk/v7"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
)

var bg = context.Background()

func TestRegistry(t *testing.T) {
	var buf bytes.Buffer
	SetLogger(slog.New(slog.NewTextHandler(&buf, nil)))
	t.Cleanup(func() { SetLogger(slog.Default()) })

	a1 := stubHandler("A", ModeMessages, "1", nil, nil)
	a2 := stubHandler("A", ModeMessages, "2", nil, nil)
	anon := &ProviderHandler{}
	r := NewRegistry(RegistryEntries{})
	r.Register(RegistryEntries{Handlers: []*ProviderHandler{a1, anon, anon}})
	r.Register(RegistryEntries{Handlers: []*ProviderHandler{a2}, Tools: map[string]ToolHandler{"t": ToolFunc(nil)}})
	r.Register(RegistryEntries{Tools: map[string]ToolHandler{"t": &NativeTool{ToolName: "X"}}})
	hs := r.Handlers()
	if len(hs) != 3 || hs[2] != a2 {
		t.Fatalf("handlers = %v", hs)
	}
	if _, native := r.Tools()["t"].(*NativeTool); !native {
		t.Error("duplicate tool did not replace")
	}
	if strings.Count(buf.String(), "level=WARN") != 2 {
		t.Errorf("want 2 warnings, got: %s", buf.String())
	}

	b := NewRegistry(RegistryEntries{Handlers: []*ProviderHandler{a1}})
	c := Compose(r, b)
	if len(r.Handlers()) != 3 || c == r {
		t.Error("compose mutated input")
	}
	if got := c.Handlers(); got[len(got)-1] != a1 {
		t.Error("b did not override a")
	}

	if resolveHandlers(NewRegistry(RegistryEntries{}), nil) != nil || resolveTools(NewRegistry(RegistryEntries{}), nil) != nil {
		t.Error("empty registry should resolve to nil")
	}
	if got := resolveHandlers(b, []*ProviderHandler{a2}); got[0] != a2 || got[1] != a1 {
		t.Error("local handlers must precede registry handlers")
	}
}

func TestConfigInvokeRoutingAndTracking(t *testing.T) {
	f := withFake(t, map[string]any{"cfg": enabledConfig("Anthropic", "completion", nil)})
	var seen HandlerRequest
	anth := stubHandler("Anthropic", ModeMessages, "hi", map[string]any{"input_tokens": 3.0, "output_tokens": 2.0}, &seen)
	other := stubHandler("OpenAI", ModeMessages, "wrong", nil, nil)
	toolCalled := false
	tools := map[string]ToolHandler{"web-search": ToolFunc(func(context.Context, map[string]any) (any, error) { toolCalled = true; return "r", nil })}

	resp, err := Config(ConfigArgs{Key: "cfg", Handlers: []*ProviderHandler{other, anth}, ToolHandlers: tools}).Invoke(bg, "q", userCtx, WithVariables(map[string]any{"someVariable": 5.0, "ldContext": "custom"}))
	if err != nil {
		t.Fatal(err)
	}
	if resp.Response != "hi" || resp.Usage.Total != 5 || resp.JudgeTasks != nil {
		t.Errorf("resp = %+v", resp)
	}
	lc := seen.Variables["ldContext"].(map[string]any)
	if lc["key"] != "user-123" || lc["email"] != "user@example.com" || lc["kind"] != "user" || seen.Variables["someVariable"] != 5.0 {
		t.Errorf("variables = %v", seen.Variables)
	}
	if _, err := seen.ToolHandlers["web-search"].(ToolFunc)(bg, nil); err != nil || !toolCalled {
		t.Error("wrapped tool not callable")
	}
	tc := f.events("$ld:ai:tool_call")
	if len(tc) != 1 || tc[0].Data["toolKey"] != "web-search" {
		t.Errorf("tool_call events = %v", tc)
	}
	for _, e := range []string{"$ld:ai:duration:total", "$ld:ai:generation:success", "$ld:ai:tokens:total", "$ld:ai:tokens:input", "$ld:ai:tokens:output"} {
		if len(f.events(e)) != 1 {
			t.Errorf("missing %s", e)
		}
	}
	if td := f.events("$ld:ai:generation:success")[0].Data; td["configKey"] != "cfg" || td["variationKey"] != "v1" || td["version"] != 3.0 {
		t.Errorf("trackData = %v", td)
	}
}

func TestConfigRoutingErrors(t *testing.T) {
	withFake(t, map[string]any{
		"oa":    enabledConfig("OpenAI", "", nil),
		"agent": enabledConfig("OpenAI", "agent", nil),
		"off":   map[string]any{"_ldMeta": map[string]any{"enabled": false}},
	})
	wild := stubHandler("*", ModeMessages, "wild", nil, nil)
	explicit := stubHandler("OpenAI", ModeMessages, "explicit", nil, nil)
	anth := stubHandler("Anthropic", ModeMessages, "", nil, nil)

	if r, _ := Config(ConfigArgs{Key: "oa", Handlers: []*ProviderHandler{wild}}).Invoke(bg, "", userCtx); r.Response != "wild" {
		t.Error("wildcard not selected")
	}
	if r, _ := Config(ConfigArgs{Key: "oa", Handlers: []*ProviderHandler{wild, explicit}}).Invoke(bg, "", userCtx); r.Response != "explicit" {
		t.Error("explicit should beat wildcard")
	}
	for name, args := range map[string]ConfigArgs{
		"wildcard wrong mode": {Key: "agent", Handlers: []*ProviderHandler{wild}},
		"no match":            {Key: "oa", Handlers: []*ProviderHandler{anth}},
		"disabled":            {Key: "off", Handlers: []*ProviderHandler{explicit}},
	} {
		if _, err := Config(args).Invoke(bg, "", userCtx); err == nil {
			t.Errorf("%s: expected error", name)
		}
	}
	_, err := Config(ConfigArgs{Key: "oa", Handlers: []*ProviderHandler{anth}}).Invoke(bg, "", userCtx)
	if err == nil || !strings.Contains(err.Error(), "OpenAI") {
		t.Errorf("error should name provider: %v", err)
	}
	reg := NewRegistry(RegistryEntries{Handlers: []*ProviderHandler{explicit}})
	if r, _ := Config(ConfigArgs{Key: "oa", Registry: reg}).Invoke(bg, "", userCtx); r.Response != "explicit" {
		t.Error("registry handler not used")
	}
}

func TestConfigHandlerErrorTracked(t *testing.T) {
	f := withFake(t, map[string]any{"cfg": enabledConfig("P", "", nil)})
	boom := errors.New("boom")
	h := CreateHandlerInternal(ProvidesFor{"P", ModeMessages}, func(context.Context, HandlerRequest) (HandlerResult, error) {
		return HandlerResult{}, boom
	}, nil, false)
	if _, err := Config(ConfigArgs{Key: "cfg", Handlers: []*ProviderHandler{h}}).Invoke(bg, "", userCtx); !errors.Is(err, boom) {
		t.Fatalf("err = %v", err)
	}
	if len(f.events("$ld:ai:generation:error")) != 1 || len(f.events("$ld:ai:duration:total")) != 1 || len(f.events("$ld:ai:tokens:total")) != 0 {
		t.Error("error-path tracking wrong")
	}
}

func TestOutputFormatParsing(t *testing.T) {
	schema := map[string]any{"outputFormat": map[string]any{"type": "object"}}
	withFake(t, map[string]any{"fmt": enabledConfig("P", "", schema), "plain": enabledConfig("P", "", nil)})
	run := func(key, out string) any {
		r, err := Config(ConfigArgs{Key: key, Handlers: []*ProviderHandler{stubHandler("P", ModeMessages, out, nil, nil)}}).Invoke(bg, "", userCtx)
		if err != nil {
			t.Fatal(err)
		}
		return r.Response
	}
	if got := run("fmt", "```json\n{\"a\":1}\n```"); !reflect.DeepEqual(got, map[string]any{"a": 1.0}) {
		t.Errorf("fenced: %v", got)
	}
	if got := run("fmt", "not json"); got != "not json" {
		t.Errorf("best effort: %v", got)
	}
	if got := run("plain", `{"a":1}`); got != `{"a":1}` {
		t.Errorf("no outputFormat should not parse: %v", got)
	}
}

func TestStream(t *testing.T) {
	f := withFake(t, map[string]any{"cfg": enabledConfig("P", "", nil)})
	h := CreateHandlerInternal(ProvidesFor{"P", ModeMessages}, nil, func(_ context.Context, req HandlerRequest) iter.Seq2[HandlerStreamEvent, error] {
		return func(yield func(HandlerStreamEvent, error) bool) {
			if req.Variables["ldContext"] == nil {
				yield(HandlerStreamEvent{}, errors.New("no ldContext"))
				return
			}
			for _, c := range []string{"Hello", " world"} {
				if !yield(HandlerStreamEvent{Type: "chunk", Text: c}, nil) {
					return
				}
			}
			yield(HandlerStreamEvent{Type: "done", Usage: map[string]any{"input": 1.0, "output": 2.0}}, nil)
		}
	}, false)
	var events []StreamEvent
	for ev, err := range Config(ConfigArgs{Key: "cfg", Handlers: []*ProviderHandler{h}}).Stream(bg, "q", userCtx) {
		if err != nil {
			t.Fatal(err)
		}
		events = append(events, ev)
	}
	if len(events) != 3 || events[0].Text != "Hello" || events[2].Type != "done" || events[2].Response != "Hello world" || events[2].Usage.Total != 3 || events[2].JudgeResults != nil {
		t.Fatalf("events = %+v", events)
	}
	if len(f.events("$ld:ai:generation:success")) != 1 || len(f.events("$ld:ai:tokens:total")) != 1 {
		t.Error("stream tracking missing")
	}

	// Fallback: no Stream func → one chunk with the full output.
	events = nil
	for ev := range Config(ConfigArgs{Key: "cfg", Handlers: []*ProviderHandler{stubHandler("P", ModeMessages, "full", nil, nil)}}).Stream(bg, "", userCtx) {
		events = append(events, ev)
	}
	if len(events) != 2 || events[0].Text != "full" || events[1].Response != "full" {
		t.Errorf("fallback events = %+v", events)
	}
}

func TestJudges(t *testing.T) {
	judged := map[string]any{"judgeConfiguration": map[string]any{"judges": []any{
		map[string]any{"key": "judge", "samplingRate": 1.0},
		map[string]any{"key": "never", "samplingRate": 0.0},
	}}}
	f := withFake(t, map[string]any{
		"cfg":   enabledConfig("P", "", judged),
		"judge": enabledConfig("P", "judge", map[string]any{"evaluationMetricKey": "quality"}),
	})
	var judgeReq HandlerRequest
	h := CreateHandlerInternal(ProvidesFor{"P", ModeMessages}, func(_ context.Context, req HandlerRequest) (HandlerResult, error) {
		if req.Variables["response_to_evaluate"] != nil {
			judgeReq = req
			return HandlerResult{Output: `{"score":0.8,"reasoning":"good"}`}, nil
		}
		return HandlerResult{Output: "answer"}, nil
	}, nil, false)
	tools := map[string]ToolHandler{"t": ToolFunc(func(context.Context, map[string]any) (any, error) { return nil, nil })}

	resp, err := Config(ConfigArgs{Key: "cfg", Handlers: []*ProviderHandler{h}, ToolHandlers: tools}).Invoke(bg, "question", userCtx)
	if err != nil {
		t.Fatal(err)
	}
	if r := resp.JudgeResults["judge"]; r.Score != 0.8 || r.Response != "good" || len(resp.JudgeResults) != 1 {
		t.Errorf("judgeResults = %+v", resp.JudgeResults)
	}
	if len(judgeReq.ToolHandlers) != 0 {
		t.Error("judge must not receive caller tools")
	}
	if mh := judgeReq.Variables["message_history"].(string); !strings.HasPrefix(mh, "question\n\nanswer\n\n") || !strings.HasSuffix(mh, FormattingInstructions) {
		t.Errorf("message_history = %q", mh)
	}
	if q := f.events("quality"); len(q) != 1 || q[0].Metric != 0.8 || q[0].Data["judgeConfigKey"] != "judge" {
		t.Errorf("metric events = %v", q)
	}

	resp, _ = Config(ConfigArgs{Key: "cfg", Handlers: []*ProviderHandler{h}, SkipJudges: true}).Invoke(bg, "question", userCtx)
	if resp.JudgeResults != nil || len(resp.JudgeTasks) != 1 || resp.JudgeTasks[0].UserInput != "question" {
		t.Errorf("skipJudges resp = %+v", resp)
	}
}

func TestInspectConfig(t *testing.T) {
	withFake(t, map[string]any{
		"on":  enabledConfig("P", "", nil),
		"off": map[string]any{"_ldMeta": map[string]any{"enabled": false}},
		"bad": map[string]any{"_ldMeta": map[string]any{"enabled": true}},
	})
	if r, err := InspectConfig(bg, "on", userCtx); err != nil || !r.Enabled || r.Config == nil || r.Meta.VariationKey != "v1" {
		t.Errorf("on = %+v err = %v", r, err)
	}
	if r, err := InspectConfig(bg, "off", userCtx); err != nil || r.Enabled || r.Config != nil || r.Meta == nil {
		t.Errorf("off = %+v err = %v", r, err)
	}
	if r, err := InspectConfig(bg, "bad", userCtx); err == nil || r.Config != nil {
		t.Errorf("bad = %+v err = %v", r, err)
	}
	if r, err := InspectConfig(bg, "missing", userCtx); !errors.Is(err, ErrConfigNotFound) || r.Meta != nil {
		t.Errorf("missing = %+v err = %v", r, err)
	}
}

func TestLifecycleAndReporting(t *testing.T) {
	_ = Shutdown(bg)
	t.Setenv("LD_SDK_KEY", "")
	if _, err := GetClient(); err == nil {
		t.Fatal("GetClient before init should error")
	}
	if _, err := InitClient(bg, InitOptions{}); err == nil || !strings.Contains(err.Error(), "LD_SDK_KEY") {
		t.Fatalf("missing key err = %v", err)
	}

	built := 0
	f := &fakeLD{}
	orig := newLDClient
	newLDClient = func(key string, _ ld.Config, _ time.Duration) (LDClient, error) { built++; return f, nil }
	t.Cleanup(func() { newLDClient = orig })

	reportUsage("client.buildJudgeTasks", "", "") // held until a client exists
	c, err := InitClient(bg, InitOptions{SDKKey: "sdk-1", OTLPEndpoint: "http://127.0.0.1:1/v1/traces"})
	if err != nil || c != f {
		t.Fatalf("init = %v %v", c, err)
	}
	_, _ = InitClient(bg, InitOptions{})
	if built != 1 {
		t.Errorf("init not idempotent: built %d", built)
	}
	if info := f.events("$ld:ai:sdk:info"); len(info) != 1 || info[0].Data["aiSdkLanguage"] != "go" {
		t.Errorf("sdk:info = %v", info)
	}
	reportUsage("client.buildJudgeTasks", "", "")
	if u := f.events("$ld:ai:sdk:usage"); len(u) != 1 || u[0].Data["helper"] != "client.buildJudgeTasks" {
		t.Errorf("usage = %v", u)
	}
	if err := Shutdown(bg); err != nil || f.closed != 1 {
		t.Errorf("shutdown: %v closed=%d", err, f.closed)
	}
	if err := Shutdown(bg); err != nil {
		t.Error("second shutdown should be a no-op")
	}

	// A panicking track must not fail anything and is not retried.
	p := &fakeLD{trackPanic: true}
	if _, err := InitWithClient(bg, p, InitOptions{}); err != nil {
		t.Fatal(err)
	}
	_ = Shutdown(bg)
}

func TestContextIdentity(t *testing.T) {
	multi := func(cs ...ldcontext.Context) ldcontext.Context { return ldcontext.NewMulti(cs...) }
	kind := func(k, key string) ldcontext.Context { return ldcontext.NewWithKind(ldcontext.Kind(k), key) }
	cases := []struct {
		c        ldcontext.Context
		id, keys string
	}{
		{ldcontext.New("u-1"), "u-1", `{"user":"u-1"}`},
		{ldcontext.New("a%b:c"), "a%b:c", `{"user":"a%b:c"}`},
		{kind("org", "a%b:c"), "org:a%25b%3Ac", `{"org":"a%b:c"}`},
		{multi(ldcontext.New("u-1"), kind("org", "o-1")), "org:o-1:user:u-1", `{"org":"o-1","user":"u-1"}`},
		{ldcontext.New("José"), "José", `{"user":"José"}`},
		{multi(kind("2", "b"), kind("10", "a")), "10:a:2:b", `{"10":"a","2":"b"}`},
	}
	for _, c := range cases {
		id, per, ok := ContextIdentity(c.c)
		if !ok || id != c.id || compactSortedJSON(per) != c.keys {
			t.Errorf("%v: id=%q keys=%s ok=%v", c.c, id, compactSortedJSON(per), ok)
		}
	}
	if _, _, ok := ContextIdentity(kind("not a kind!", "x")); ok {
		t.Error("empty kind should be unusable")
	}
}

func TestSentinelErrors(t *testing.T) {
	_ = Shutdown(bg)
	if _, err := GetClient(); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("GetClient err = %v", err)
	}
	f := withFake(t, map[string]any{
		"oa":  enabledConfig("OpenAI", "", nil),
		"off": map[string]any{"_ldMeta": map[string]any{"enabled": false}},
	})
	anth := stubHandler("Anthropic", ModeMessages, "", nil, nil)
	cases := []struct {
		key  string
		want error
	}{
		{"missing", ErrConfigNotFound},
		{"off", ErrConfigDisabled},
		{"oa", ErrNoHandler},
	}
	for _, c := range cases {
		_, err := Config(ConfigArgs{Key: c.key, Handlers: []*ProviderHandler{anth}}).Invoke(bg, "", userCtx)
		if !errors.Is(err, c.want) {
			t.Errorf("%s: err = %v, want %v", c.key, err, c.want)
		}
	}
	// The SDK's evaluation error is kept, so the message names the real cause.
	f.evalErr = errors.New("feature flag evaluation called before LaunchDarkly client initialization completed")
	_, err := Config(ConfigArgs{Key: "missing", Handlers: []*ProviderHandler{anth}}).Invoke(bg, "", userCtx)
	if !errors.Is(err, ErrConfigNotFound) || !strings.Contains(err.Error(), "before LaunchDarkly client initialization") {
		t.Errorf("evaluation error lost: %v", err)
	}
	if _, err := Graph("missing-graph", GraphOptions{Handlers: []*ProviderHandler{anth}}).Invoke(bg, "", userCtx); err == nil || !strings.Contains(err.Error(), "before LaunchDarkly client initialization") {
		t.Errorf("graph evaluation error lost: %v", err)
	}
}

func TestCallOptionsAndText(t *testing.T) {
	schema := map[string]any{"outputFormat": map[string]any{"type": "object"}}
	withFake(t, map[string]any{"cfg": enabledConfig("P", "", nil), "fmt": enabledConfig("P", "", schema)})
	var seen HandlerRequest
	h := stubHandler("P", ModeMessages, `{"a":1}`, nil, &seen)

	// No options at all: no variables beyond the injected ones, no history.
	resp, err := Config(ConfigArgs{Key: "cfg", Handlers: []*ProviderHandler{h}}).Invoke(bg, "q", userCtx)
	if err != nil || seen.History != nil || seen.Variables["x"] != nil || resp.Text() != `{"a":1}` {
		t.Fatalf("no options: resp=%+v history=%v err=%v", resp, seen.History, err)
	}
	hist := []Message{{Role: "user", Text: "earlier"}}
	_, _ = Config(ConfigArgs{Key: "cfg", Handlers: []*ProviderHandler{h}}).
		Invoke(bg, "q", userCtx, WithVariables(map[string]any{"x": "1"}), WithHistory(hist))
	if seen.Variables["x"] != "1" || len(seen.History) != 1 {
		t.Errorf("options not applied: vars=%v history=%v", seen.Variables, seen.History)
	}
	// A parsed structured output renders back to JSON text.
	resp, _ = Config(ConfigArgs{Key: "fmt", Handlers: []*ProviderHandler{h}}).Invoke(bg, "q", userCtx)
	if _, isMap := resp.Response.(map[string]any); !isMap || resp.Text() != `{"a":1}` {
		t.Errorf("structured: Response=%#v Text=%q", resp.Response, resp.Text())
	}
}

// A handler declared with the legacy "completion" mode serves messages
// configs: the handler's mode is normalized like the config's.
func TestSelectHandlerNormalizesHandlerMode(t *testing.T) {
	legacy := &ProviderHandler{ProvidesFor: ProvidesFor{Provider: "OpenAI", Mode: Mode("completion")}}
	agent := &ProviderHandler{ProvidesFor: ProvidesFor{Provider: "OpenAI", Mode: ModeAgent}}
	if h, err := SelectHandler([]*ProviderHandler{agent, legacy}, "OpenAI", ModeMessages); err != nil || h != legacy {
		t.Errorf("messages: got %v %v", h, err)
	}
	if h, err := SelectHandler([]*ProviderHandler{legacy, agent}, "OpenAI", ModeAgent); err != nil || h != agent {
		t.Errorf("agent: got %v %v", h, err)
	}
}

// A panicking handler is Go's other error path: generation error and duration
// are tracked, success is not, and the panic still reaches the caller. A
// panic from the consumer's own loop body is not a handler failure.
func TestHandlerPanicTracksGenerationError(t *testing.T) {
	boom := func(context.Context, HandlerRequest) (HandlerResult, error) { panic("handler bug") }
	streamBoom := func(context.Context, HandlerRequest) iter.Seq2[HandlerStreamEvent, error] {
		return func(yield func(HandlerStreamEvent, error) bool) {
			if yield(HandlerStreamEvent{Type: EventChunk, Text: "a"}, nil) {
				panic("stream bug")
			}
		}
	}
	chunky := func(context.Context, HandlerRequest) iter.Seq2[HandlerStreamEvent, error] {
		return func(yield func(HandlerStreamEvent, error) bool) {
			if yield(HandlerStreamEvent{Type: EventChunk, Text: "a"}, nil) {
				yield(HandlerStreamEvent{Type: EventDone}, nil)
			}
		}
	}
	run := func(fn func()) (r any) {
		defer func() { r = recover() }()
		fn()
		return nil
	}
	for name, c := range map[string]struct {
		h        *ProviderHandler
		consumer func(*AIConfig)
		wantErr  int
	}{
		"invoke": {CreateHandlerInternal(ProvidesFor{"P", ModeMessages}, boom, nil, false),
			func(a *AIConfig) { _, _ = a.Invoke(bg, "q", userCtx) }, 1},
		"stream": {CreateHandlerInternal(ProvidesFor{"P", ModeMessages}, boom, streamBoom, false),
			func(a *AIConfig) {
				for range a.Stream(bg, "q", userCtx) {
				}
			}, 1},
		"stream fallback": {CreateHandlerInternal(ProvidesFor{"P", ModeMessages}, boom, nil, false),
			func(a *AIConfig) {
				for range a.Stream(bg, "q", userCtx) {
				}
			}, 1},
		"consumer": {CreateHandlerInternal(ProvidesFor{"P", ModeMessages}, boom, chunky, false),
			func(a *AIConfig) {
				for range a.Stream(bg, "q", userCtx) {
					panic("consumer bug")
				}
			}, 0},
	} {
		t.Run(name, func(t *testing.T) {
			f := withFake(t, map[string]any{"cfg": enabledConfig("P", "", nil)})
			if r := run(func() { c.consumer(Config(ConfigArgs{Key: "cfg", Handlers: []*ProviderHandler{c.h}})) }); r == nil {
				t.Fatal("the panic was swallowed")
			}
			if n := len(f.events("$ld:ai:generation:error")); n != c.wantErr {
				t.Errorf("generation:error tracked %d times, want %d", n, c.wantErr)
			}
			if n := len(f.events("$ld:ai:duration:total")); n != c.wantErr {
				t.Errorf("duration tracked %d times, want %d", n, c.wantErr)
			}
			if len(f.events("$ld:ai:generation:success")) != 0 {
				t.Error("a panicking run tracked success")
			}
		})
	}
}

// Tracking needs a context: an undefined one sends nothing.
func TestTrackSkipsUndefinedContext(t *testing.T) {
	fake := &fakeLD{}
	if _, err := InitWithClient(bg, fake, InitOptions{}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = Shutdown(bg) })
	Track("$ld:ai:generation:success", ldcontext.Context{}, TrackData{}, 1)
	if got := fake.events("$ld:ai:generation:success"); len(got) != 0 {
		t.Errorf("tracked with an undefined context: %v", got)
	}
	Track("$ld:ai:generation:success", ldcontext.New("u"), TrackData{}, 1)
	if len(fake.events("$ld:ai:generation:success")) != 1 {
		t.Error("a defined context was not tracked")
	}
}

// A judge whose config cannot be resolved (missing, or toggled off) is
// skipped: the generation already succeeded and is returned, inline and as
// deferred tasks, and the other judges still run.
func TestUnresolvableJudgeIsSkipped(t *testing.T) {
	judged := map[string]any{"judgeConfiguration": map[string]any{"judges": []any{
		map[string]any{"key": "missing", "samplingRate": 1.0},
		map[string]any{"key": "off", "samplingRate": 1.0},
		map[string]any{"key": "judge", "samplingRate": 1.0},
	}}}
	off := enabledConfig("P", "judge", nil)
	off["_ldMeta"] = map[string]any{"enabled": false}
	withFake(t, map[string]any{
		"cfg":   enabledConfig("P", "", judged),
		"off":   off,
		"judge": enabledConfig("P", "judge", nil),
	})
	h := CreateHandlerInternal(ProvidesFor{"P", ModeMessages}, func(_ context.Context, req HandlerRequest) (HandlerResult, error) {
		if req.Variables["response_to_evaluate"] != nil {
			return HandlerResult{Output: `{"score":0.5,"reasoning":"ok"}`}, nil
		}
		return HandlerResult{Output: "answer"}, nil
	}, nil, false)
	resp, err := Config(ConfigArgs{Key: "cfg", Handlers: []*ProviderHandler{h}}).Invoke(bg, "q", userCtx)
	if err != nil || resp.Response != "answer" || len(resp.JudgeResults) != 1 || resp.JudgeResults["judge"].Score != 0.5 {
		t.Fatalf("inline: resp=%+v err=%v", resp, err)
	}
	resp, err = Config(ConfigArgs{Key: "cfg", Handlers: []*ProviderHandler{h}, SkipJudges: true}).Invoke(bg, "q", userCtx)
	if err != nil || len(resp.JudgeTasks) != 1 || resp.JudgeTasks[0].ConfigKey != "judge" {
		t.Fatalf("deferred: tasks=%+v err=%v", resp.JudgeTasks, err)
	}
}
