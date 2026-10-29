package experimental

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"log/slog"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/launchdarkly/go-sdk-common/v3/ldreason"
	"github.com/launchdarkly/go-sdk-common/v3/ldvalue"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

type trackedEvent struct {
	event  string
	metric float64
	data   map[string]any
}

// shadowLD is an LD client double with JSONVariationDetail.
type shadowLD struct {
	mu             sync.Mutex
	value          any
	reason         ldreason.EvaluationReason
	variationCalls int
	detailCalls    int
	tracks         []trackedEvent
}

func (f *shadowLD) JSONVariation(_ string, _ ldcontext.Context, def ldvalue.Value) (ldvalue.Value, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.variationCalls++
	if f.value == nil {
		return def, nil
	}
	return ldvalue.FromJSONMarshal(f.value), nil
}

func (f *shadowLD) JSONVariationDetail(_ string, _ ldcontext.Context, def ldvalue.Value) (ldvalue.Value, ldreason.EvaluationDetail, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.detailCalls++
	v := def
	if f.value != nil {
		v = ldvalue.FromJSONMarshal(f.value)
	}
	return v, ldreason.NewEvaluationDetail(v, 0, f.reason), nil
}

func (f *shadowLD) TrackMetric(event string, _ ldcontext.Context, m float64, data ldvalue.Value) error {
	var d map[string]any
	_ = json.Unmarshal([]byte(data.JSONString()), &d)
	f.mu.Lock()
	defer f.mu.Unlock()
	f.tracks = append(f.tracks, trackedEvent{event, m, d})
	return nil
}
func (f *shadowLD) Flush()       {}
func (f *shadowLD) Close() error { return nil }

func (f *shadowLD) events(shadow bool) []trackedEvent {
	f.mu.Lock()
	defer f.mu.Unlock()
	var out []trackedEvent
	for _, t := range f.tracks {
		if !strings.HasPrefix(t.event, "$ld:ai:") || strings.HasPrefix(t.event, "$ld:ai:sdk:") {
			continue
		}
		if (t.data["isShadow"] == true) == shadow {
			out = append(out, t)
		}
	}
	return out
}

func (f *shadowLD) usage() []string {
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

// plainLD has no JSONVariationDetail.
type plainLD struct{ f *shadowLD }

func (p plainLD) JSONVariation(k string, c ldcontext.Context, d ldvalue.Value) (ldvalue.Value, error) {
	return p.f.JSONVariation(k, c, d)
}
func (p plainLD) TrackMetric(e string, c ldcontext.Context, m float64, d ldvalue.Value) error {
	return p.f.TrackMetric(e, c, m, d)
}
func (p plainLD) Flush()       {}
func (p plainLD) Close() error { return nil }

var shadowUser = ldcontext.New("user-123")

func future() float64 { return float64(time.Now().Add(time.Hour).UnixMilli()) }

func entry(id string) map[string]any {
	return map[string]any{"id": id, "variationKey": "claude-candidate", "samplingRate": 1.0,
		"endsAt": future(), "ownerKind": "member", "ownerId": "m-1"}
}

func candidate() map[string]any {
	return map[string]any{
		"_ldMeta":            map[string]any{"enabled": true, "variationKey": "claude-candidate", "version": 7, "mode": "completion"},
		"model":              map[string]any{"name": "claude"},
		"provider":           map[string]any{"name": "Anthropic"},
		"instructions":       "candidate",
		"tools":              map[string]any{"search": map[string]any{"name": "search", "type": "function", "parameters": map[string]any{}}},
		"judgeConfiguration": map[string]any{"judges": []any{map[string]any{"key": "judge", "samplingRate": 1.0}}},
	}
}

func served(shadows map[string]any) map[string]any {
	return map[string]any{
		"_ldMeta":      map[string]any{"enabled": true, "variationKey": "gpt-main", "version": 4, "mode": "completion"},
		"model":        map[string]any{"name": "gpt-5"},
		"provider":     map[string]any{"name": "OpenAI"},
		"instructions": "be helpful",
		"shadowConfiguration": map[string]any{
			"shadows":    shadows,
			"variations": map[string]any{"claude-candidate": candidate()},
		},
	}
}

func fallthroughPayload() map[string]any {
	return served(map[string]any{"fallthrough": []any{entry("shadow-1")}})
}

// stub is a recording handler.
type stub struct {
	mu      sync.Mutex
	calls   []aiserver.HandlerRequest
	streams int
	out     string
	err     error
	delay   time.Duration
	panics  bool
	span    bool
}

func (s *stub) handler(provider string) *aiserver.ProviderHandler {
	return &aiserver.ProviderHandler{
		ProvidesFor: aiserver.ProvidesFor{Provider: provider, Mode: aiserver.ModeMessages},
		Invoke: func(ctx context.Context, req aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
			s.mu.Lock()
			s.calls = append(s.calls, req)
			s.mu.Unlock()
			if s.span {
				run := aiserver.StartHandlerRun(ctx, aiserver.HandlerSpanOptions{Tracer: aiserver.TracerPrefix + "/test", Name: "test.run", System: "test"}, req, aiserver.ComposePrompt(req, aiserver.ModeMessages))
				defer run.Finish("", s.out)
			}
			if s.panics {
				panic("shadow exploded")
			}
			if s.delay > 0 {
				select {
				case <-time.After(s.delay):
				case <-ctx.Done():
					time.Sleep(10 * time.Millisecond)
				}
			}
			return aiserver.HandlerResult{Output: s.out, Usage: map[string]any{"input": 3, "output": 4}}, s.err
		},
		Stream: func(context.Context, aiserver.HandlerRequest) iter.Seq2[aiserver.HandlerStreamEvent, error] {
			s.mu.Lock()
			s.streams++
			s.mu.Unlock()
			return func(yield func(aiserver.HandlerStreamEvent, error) bool) {
				if s.err != nil {
					yield(aiserver.HandlerStreamEvent{}, s.err)
					return
				}
				if yield(aiserver.HandlerStreamEvent{Type: aiserver.EventChunk, Text: s.out}, nil) {
					yield(aiserver.HandlerStreamEvent{Type: aiserver.EventDone, Usage: map[string]any{"input": 3, "output": 4}}, nil)
				}
			}
		},
	}
}

func (s *stub) count() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.calls)
}

func (s *stub) last() aiserver.HandlerRequest {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.calls[len(s.calls)-1]
}

func install(t *testing.T, value any, reason ldreason.EvaluationReason) *shadowLD {
	t.Helper()
	f := &shadowLD{value: value, reason: reason}
	installClient(t, f)
	return f
}

func installClient(t *testing.T, c aiserver.LDClient) {
	t.Helper()
	ctx := context.Background()
	if _, err := aiserver.InitWithClient(ctx, c, aiserver.InitOptions{}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		running.Wait()
		_ = aiserver.Shutdown(ctx)
	})
}

type pair struct{ a, b *stub }

func newPair() pair { return pair{&stub{out: "served"}, &stub{out: "shadow"}} }

func (p pair) handlers() []*aiserver.ProviderHandler {
	return []*aiserver.ProviderHandler{p.a.handler("OpenAI"), p.b.handler("Anthropic")}
}

func mustShadow(t *testing.T, handlers []*aiserver.ProviderHandler, opts ShadowOptions) *ShadowAIConfig {
	t.Helper()
	s, err := ShadowConfig(aiserver.ConfigArgs{Key: "cfg", Handlers: handlers}, opts)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func invokeAndWait(t *testing.T, s *ShadowAIConfig, opts ...aiserver.CallOption) aiserver.ProviderResponse {
	t.Helper()
	resp, err := s.Invoke(context.Background(), "hello", shadowUser, opts...)
	if err != nil {
		t.Fatal(err)
	}
	running.Wait()
	return resp
}

func withoutRunID(events []trackedEvent) []trackedEvent {
	out := make([]trackedEvent, len(events))
	for i, e := range events {
		d := map[string]any{}
		for k, v := range e.data {
			if k != "runId" {
				d[k] = v
			}
		}
		m := e.metric
		if e.event == "$ld:ai:duration:total" {
			m = 0
		}
		out[i] = trackedEvent{e.event, m, d}
	}
	return out
}

func TestConfigNeverCallsDetailOrRunsShadows(t *testing.T) {
	f := install(t, fallthroughPayload(), ldreason.NewEvalReasonFallthrough())
	p := newPair()
	cfg := aiserver.Config(aiserver.ConfigArgs{Key: "cfg", Handlers: p.handlers()})
	if _, err := cfg.Invoke(context.Background(), "hello", shadowUser); err != nil {
		t.Fatal(err)
	}
	for _, err := range cfg.Stream(context.Background(), "hello", shadowUser) {
		if err != nil {
			t.Fatal(err)
		}
	}
	running.Wait()
	if f.detailCalls != 0 || f.variationCalls != 2 {
		t.Fatalf("detail %d, variation %d", f.detailCalls, f.variationCalls)
	}
	if p.a.count() != 1 || p.a.streams != 1 || p.b.count() != 0 || len(f.events(true)) != 0 {
		t.Fatalf("served %d/%d, shadow %d, shadow events %d", p.a.count(), p.a.streams, p.b.count(), len(f.events(true)))
	}
}

func TestShadowOptionsAreConfigArgsPlusTwo(t *testing.T) {
	var names []string
	for f := range reflect.TypeFor[ShadowOptions]().Fields() {
		names = append(names, f.Name)
	}
	if !slices.Equal(names, []string{"Timeout", "OnTask"}) {
		t.Fatalf("ShadowOptions fields: %v", names)
	}
	if reflect.TypeOf(ShadowConfig).In(0) != reflect.TypeFor[aiserver.ConfigArgs]() {
		t.Fatal("ShadowConfig must take Config's own arguments")
	}
}

func TestPrimaryMatchesPlainConfig(t *testing.T) {
	f := install(t, fallthroughPayload(), ldreason.NewEvalReasonFallthrough())
	p := newPair()
	plain, err := aiserver.Config(aiserver.ConfigArgs{Key: "cfg", Handlers: p.handlers()}).Invoke(context.Background(), "hello", shadowUser)
	if err != nil {
		t.Fatal(err)
	}
	plainEvents := f.events(false)
	shadowed := invokeAndWait(t, mustShadow(t, p.handlers(), ShadowOptions{}))
	shadowedEvents := f.events(false)[len(plainEvents):]
	if plain.Response != shadowed.Response || plain.Usage != shadowed.Usage {
		t.Fatalf("plain %+v, shadowed %+v", plain, shadowed)
	}
	plain.TrackData.RunID, shadowed.TrackData.RunID = "", ""
	if plain.TrackData != shadowed.TrackData {
		t.Fatalf("track data: %+v vs %+v", plain.TrackData, shadowed.TrackData)
	}
	if !reflect.DeepEqual(withoutRunID(plainEvents), withoutRunID(shadowedEvents)) {
		t.Fatalf("events differ:\n%v\n%v", plainEvents, shadowedEvents)
	}
	if p.b.count() != 1 {
		t.Fatalf("shadow ran %d times", p.b.count())
	}
}

func TestShadowConfigurationNeverReachesAHandler(t *testing.T) {
	install(t, fallthroughPayload(), ldreason.NewEvalReasonFallthrough())
	p := newPair()
	invokeAndWait(t, mustShadow(t, p.handlers(), ShadowOptions{}))
	for _, req := range []aiserver.HandlerRequest{p.a.last(), p.b.last()} {
		if _, ok := req.Config.Raw["shadowConfiguration"]; ok {
			t.Fatal("shadowConfiguration reached a handler")
		}
	}
}

func TestPrimaryFailureStartsNoShadows(t *testing.T) {
	install(t, fallthroughPayload(), ldreason.NewEvalReasonFallthrough())
	p := newPair()
	p.a.err = errors.New("served failed")
	s := mustShadow(t, p.handlers(), ShadowOptions{})
	if _, err := s.Invoke(context.Background(), "hello", shadowUser); err == nil {
		t.Fatal("want the served error")
	}
	for _, err := range s.Stream(context.Background(), "hello", shadowUser) {
		_ = err
	}
	running.Wait()
	if p.b.count() != 0 {
		t.Fatal("a shadow ran after the served call failed")
	}
}

func TestDisabledPrimaryFailsLikeConfig(t *testing.T) {
	value := fallthroughPayload()
	value["_ldMeta"].(map[string]any)["enabled"] = false
	install(t, value, ldreason.NewEvalReasonFallthrough())
	p := newPair()
	_, err := mustShadow(t, p.handlers(), ShadowOptions{}).Invoke(context.Background(), "hello", shadowUser)
	if !errors.Is(err, aiserver.ErrConfigDisabled) || p.b.count() != 0 {
		t.Fatalf("err %v, shadow calls %d", err, p.b.count())
	}
}

func TestMalformedConfigurationIsIgnored(t *testing.T) {
	cases := map[string]any{
		"not an object":  "nope",
		"shadows list":   map[string]any{"shadows": []any{}, "variations": map[string]any{}},
		"slot not list":  map[string]any{"shadows": map[string]any{"fallthrough": "x"}, "variations": map[string]any{"claude-candidate": candidate()}},
		"entry no id":    map[string]any{"shadows": map[string]any{"fallthrough": []any{map[string]any{"variationKey": "claude-candidate", "samplingRate": 1.0, "endsAt": future()}}}, "variations": map[string]any{"claude-candidate": candidate()}},
		"string rate":    map[string]any{"shadows": map[string]any{"fallthrough": []any{map[string]any{"id": "s", "variationKey": "claude-candidate", "samplingRate": "1", "endsAt": future()}}}, "variations": map[string]any{"claude-candidate": candidate()}},
		"no endsAt":      map[string]any{"shadows": map[string]any{"fallthrough": []any{map[string]any{"id": "s", "variationKey": "claude-candidate", "samplingRate": 1.0}}}, "variations": map[string]any{"claude-candidate": candidate()}},
		"entry not obj":  map[string]any{"shadows": map[string]any{"fallthrough": []any{"x"}}, "variations": map[string]any{"claude-candidate": candidate()}},
		"bad variation":  map[string]any{"shadows": map[string]any{"fallthrough": []any{entry("s")}}, "variations": map[string]any{"claude-candidate": map[string]any{"_ldMeta": map[string]any{"enabled": true}}}},
		"variations nil": map[string]any{"shadows": map[string]any{"fallthrough": []any{entry("s")}}},
	}
	for name, configuration := range cases {
		t.Run(name, func(t *testing.T) {
			value := fallthroughPayload()
			value["shadowConfiguration"] = configuration
			install(t, value, ldreason.NewEvalReasonFallthrough())
			p := newPair()
			resp := invokeAndWait(t, mustShadow(t, p.handlers(), ShadowOptions{}))
			if resp.Response != "served" || p.b.count() != 0 {
				t.Fatalf("response %v, shadow calls %d", resp.Response, p.b.count())
			}
		})
	}
}

func TestEvaluatesOnceWithJSONVariationDetail(t *testing.T) {
	f := install(t, fallthroughPayload(), ldreason.NewEvalReasonFallthrough())
	invokeAndWait(t, mustShadow(t, newPair().handlers(), ShadowOptions{}))
	if f.detailCalls != 1 || f.variationCalls != 0 {
		t.Fatalf("detail %d, variation %d", f.detailCalls, f.variationCalls)
	}
}

func TestClientWithoutDetailRunsNoShadows(t *testing.T) {
	f := &shadowLD{value: fallthroughPayload(), reason: ldreason.NewEvalReasonFallthrough()}
	installClient(t, plainLD{f})
	p := newPair()
	resp := invokeAndWait(t, mustShadow(t, p.handlers(), ShadowOptions{}))
	if resp.Response != "served" || p.b.count() != 0 || f.variationCalls != 1 {
		t.Fatalf("response %v, shadow %d, variation calls %d", resp.Response, p.b.count(), f.variationCalls)
	}
}

func TestRuleMatchUsesOnlyThatRule(t *testing.T) {
	value := served(map[string]any{"r1": []any{entry("rule-shadow")}, "fallthrough": []any{entry("ft-shadow")}})
	f := install(t, value, ldreason.NewEvalReasonRuleMatch(0, "r1"))
	p := newPair()
	invokeAndWait(t, mustShadow(t, p.handlers(), ShadowOptions{}))
	ev := f.events(true)
	if p.b.count() != 1 || len(ev) == 0 || ev[0].data["shadowRequestId"] != "rule-shadow" {
		t.Fatalf("shadow calls %d, events %v", p.b.count(), ev)
	}
}

func TestFallthroughUsesTheFallthroughSlot(t *testing.T) {
	value := served(map[string]any{"r1": []any{entry("rule-shadow")}, "fallthrough": []any{entry("ft-shadow")}})
	f := install(t, value, ldreason.NewEvalReasonFallthrough())
	p := newPair()
	invokeAndWait(t, mustShadow(t, p.handlers(), ShadowOptions{}))
	ev := f.events(true)
	if p.b.count() != 1 || ev[0].data["shadowRequestId"] != "ft-shadow" {
		t.Fatalf("shadow calls %d, events %v", p.b.count(), ev)
	}
}

func TestOtherReasonsRunNothing(t *testing.T) {
	value := served(map[string]any{"r1": []any{entry("a")}, "fallthrough": []any{entry("b")}})
	for name, reason := range map[string]ldreason.EvaluationReason{
		"target":       ldreason.NewEvalReasonTargetMatch(),
		"off":          ldreason.NewEvalReasonOff(),
		"prerequisite": ldreason.NewEvalReasonPrerequisiteFailed("p"),
		"error":        ldreason.NewEvalReasonError(ldreason.EvalErrorException),
		"none":         {},
		"rule no id":   ldreason.NewEvalReasonRuleMatch(0, ""),
	} {
		t.Run(name, func(t *testing.T) {
			install(t, value, reason)
			p := newPair()
			invokeAndWait(t, mustShadow(t, p.handlers(), ShadowOptions{}))
			if p.b.count() != 0 {
				t.Fatal("a shadow ran")
			}
		})
	}
}

func TestEntryPastItsEndIsSkipped(t *testing.T) {
	past := entry("past")
	past["endsAt"] = float64(time.Now().Add(-time.Minute).UnixMilli())
	f := install(t, served(map[string]any{"fallthrough": []any{past, entry("live")}}), ldreason.NewEvalReasonFallthrough())
	p := newPair()
	invokeAndWait(t, mustShadow(t, p.handlers(), ShadowOptions{}))
	if p.b.count() != 1 || f.events(true)[0].data["shadowRequestId"] != "live" {
		t.Fatalf("shadow calls %d", p.b.count())
	}
}

func TestMissingOrDisabledVariationIsSkipped(t *testing.T) {
	missing := entry("missing")
	missing["variationKey"] = "nope"
	disabled := fallthroughPayload()
	disabled["shadowConfiguration"].(map[string]any)["variations"].(map[string]any)["claude-candidate"].(map[string]any)["_ldMeta"].(map[string]any)["enabled"] = false
	for name, value := range map[string]any{
		"missing":  served(map[string]any{"fallthrough": []any{missing}}),
		"disabled": disabled,
	} {
		t.Run(name, func(t *testing.T) {
			install(t, value, ldreason.NewEvalReasonFallthrough())
			p := newPair()
			invokeAndWait(t, mustShadow(t, p.handlers(), ShadowOptions{}))
			if p.b.count() != 0 {
				t.Fatal("a shadow ran")
			}
		})
	}
}

func TestNoMatchingHandlerIsSkippedAndLogged(t *testing.T) {
	install(t, fallthroughPayload(), ldreason.NewEvalReasonFallthrough())
	var logs bytes.Buffer
	aiserver.SetLogger(slog.New(slog.NewTextHandler(&logs, &slog.HandlerOptions{Level: slog.LevelDebug})))
	t.Cleanup(func() { aiserver.SetLogger(nil) })
	p := newPair()
	resp := invokeAndWait(t, mustShadow(t, []*aiserver.ProviderHandler{p.a.handler("OpenAI")}, ShadowOptions{}))
	if resp.Response != "served" || !strings.Contains(logs.String(), "no handler") {
		t.Fatalf("response %v, logs %q", resp.Response, logs.String())
	}
}

func TestSamplingIsSticky(t *testing.T) {
	for i := range 100 {
		c := ldcontext.New(fmt.Sprint("u", i))
		first := inSample("s1", c, 0.5)
		for range 5 {
			if inSample("s1", c, 0.5) != first {
				t.Fatal("sampling is not sticky")
			}
		}
	}
}

func TestSamplingHonorsTheRate(t *testing.T) {
	in := 0
	for i := range 10000 {
		if inSample("shadow-1", ldcontext.New(fmt.Sprint("user-", i)), 0.2) {
			in++
		}
	}
	if in < 1800 || in > 2200 {
		t.Fatalf("%d of 10000 sampled at 0.2", in)
	}
}

func TestSamplingEdges(t *testing.T) {
	for i := range 200 {
		c := ldcontext.New(fmt.Sprint("u", i))
		if !inSample("s", c, 1) || inSample("s", c, 0) {
			t.Fatal("rate 1 must run every context and rate 0 none")
		}
	}
}

func TestSamplingBucketMatchesTheOtherSDKs(t *testing.T) {
	user := ldcontext.New("u1")
	if !inSample("s1", user, 0.7312) || inSample("s1", user, 0.7311) {
		t.Fatal("s1/u1 bucket must be 0.7311362")
	}
	org := ldcontext.NewWithKind("org", "acme")
	if !inSample("shadow-1", org, 0.399) || inSample("shadow-1", org, 0.3989) {
		t.Fatal("shadow-1/org:acme bucket must be 0.3989622")
	}
}

func TestContextWithoutKeyIsNotSampled(t *testing.T) {
	if inSample("s1", ldcontext.Context{}, 1) || inSample("s1", ldcontext.New(""), 1) {
		t.Fatal("a context with no usable key must not be sampled")
	}
}

func TestStreamRunsTheShadowAsAPlainInvokeAfterDone(t *testing.T) {
	f := install(t, fallthroughPayload(), ldreason.NewEvalReasonFallthrough())
	p := newPair()
	var types []aiserver.EventType
	for ev, err := range mustShadow(t, p.handlers(), ShadowOptions{}).Stream(context.Background(), "hello", shadowUser) {
		if err != nil {
			t.Fatal(err)
		}
		types = append(types, ev.Type)
	}
	running.Wait()
	if !slices.Equal(types, []aiserver.EventType{aiserver.EventChunk, aiserver.EventDone}) {
		t.Fatalf("events %v", types)
	}
	if p.a.streams != 1 || p.b.count() != 1 || p.b.streams != 0 {
		t.Fatalf("served streams %d, shadow invokes %d, shadow streams %d", p.a.streams, p.b.count(), p.b.streams)
	}
	if served := f.events(false); served[0].data["runId"] != f.events(true)[0].data["runId"] {
		t.Fatal("the shadow must carry the served run id")
	}
}

func TestStreamMatchesPlainConfigStream(t *testing.T) {
	f := install(t, fallthroughPayload(), ldreason.NewEvalReasonFallthrough())
	p := newPair()
	collect := func(seq iter.Seq2[aiserver.StreamEvent, error]) []aiserver.StreamEvent {
		var out []aiserver.StreamEvent
		for ev, err := range seq {
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, ev)
		}
		return out
	}
	plain := collect(aiserver.Config(aiserver.ConfigArgs{Key: "cfg", Handlers: p.handlers()}).Stream(context.Background(), "hello", shadowUser))
	plainEvents := f.events(false)
	shadowed := collect(mustShadow(t, p.handlers(), ShadowOptions{}).Stream(context.Background(), "hello", shadowUser))
	running.Wait()
	if !reflect.DeepEqual(plain, shadowed) {
		t.Fatalf("stream differs: %v vs %v", plain, shadowed)
	}
	if !reflect.DeepEqual(withoutRunID(plainEvents), withoutRunID(f.events(false)[len(plainEvents):])) {
		t.Fatal("served stream events differ from Config's")
	}
}

func TestStreamDeferredHandsOverTheTask(t *testing.T) {
	install(t, fallthroughPayload(), ldreason.NewEvalReasonFallthrough())
	p := newPair()
	var tasks []ShadowTask
	s := mustShadow(t, p.handlers(), ShadowOptions{OnTask: func(task ShadowTask) { tasks = append(tasks, task) }})
	for _, err := range s.Stream(context.Background(), "hello", shadowUser) {
		if err != nil {
			t.Fatal(err)
		}
	}
	if len(tasks) != 1 || p.b.count() != 0 {
		t.Fatalf("tasks %d, shadow calls %d", len(tasks), p.b.count())
	}
}

func TestShadowGetsTheSameInputOnTheStrippedConfig(t *testing.T) {
	install(t, fallthroughPayload(), ldreason.NewEvalReasonFallthrough())
	p := newPair()
	history := []aiserver.Message{{Role: "user", Text: "earlier"}, {Role: "assistant", Text: "reply"}}
	invokeAndWait(t, mustShadow(t, p.handlers(), ShadowOptions{}),
		aiserver.WithVariables(map[string]any{"topic": "flags"}), aiserver.WithHistory(history))
	a, b := p.a.last(), p.b.last()
	if b.UserInput != "hello" || !reflect.DeepEqual(b.History, history) || b.Variables["topic"] != "flags" {
		t.Fatalf("shadow request %+v", b)
	}
	if b.Config.Model.Name != "claude" || len(b.Config.Tools) != 0 || b.Config.JudgeConfiguration != nil || len(b.ToolHandlers) != 0 {
		t.Fatalf("shadow config %+v, tools %v", b.Config, b.ToolHandlers)
	}
	for _, k := range []string{"tools", "judgeConfiguration", "shadowConfiguration"} {
		if _, ok := b.Config.Raw[k]; ok {
			t.Errorf("%s reached the shadow", k)
		}
	}
	if len(history) != 2 || a.UserInput != "hello" {
		t.Fatal("the shadow changed the caller's history")
	}
}

func TestBackgroundShadowDoesNotDelayTheCaller(t *testing.T) {
	install(t, fallthroughPayload(), ldreason.NewEvalReasonFallthrough())
	p := newPair()
	p.b.delay = 300 * time.Millisecond
	start := time.Now()
	resp, err := mustShadow(t, p.handlers(), ShadowOptions{}).Invoke(context.Background(), "hello", shadowUser)
	if err != nil || resp.Response != "served" {
		t.Fatal(err)
	}
	if elapsed := time.Since(start); elapsed > 200*time.Millisecond {
		t.Fatalf("the caller waited %v", elapsed)
	}
}

func TestFailingOrTimedOutShadowNeverReachesTheCaller(t *testing.T) {
	for name, b := range map[string]*stub{
		"error":   {err: errors.New("shadow failed")},
		"panic":   {panics: true},
		"timeout": {delay: time.Second},
	} {
		t.Run(name, func(t *testing.T) {
			f := install(t, fallthroughPayload(), ldreason.NewEvalReasonFallthrough())
			p := pair{&stub{out: "served"}, b}
			resp := invokeAndWait(t, mustShadow(t, p.handlers(), ShadowOptions{Timeout: 50 * time.Millisecond}))
			if resp.Response != "served" || len(f.events(false)) != 5 {
				t.Fatalf("response %v, served events %v", resp.Response, f.events(false))
			}
			if errs := eventsNamed(f.events(true), "$ld:ai:generation:error"); len(errs) != 1 {
				t.Fatalf("shadow error events %d", len(errs))
			}
		})
	}
}

func eventsNamed(events []trackedEvent, name string) []trackedEvent {
	var out []trackedEvent
	for _, e := range events {
		if e.event == name {
			out = append(out, e)
		}
	}
	return out
}

func TestTimeoutMustBePositive(t *testing.T) {
	if _, err := ShadowConfig(aiserver.ConfigArgs{Key: "cfg"}, ShadowOptions{Timeout: -time.Second}); err == nil {
		t.Fatal("a negative timeout must fail")
	}
	s, err := ShadowConfig(aiserver.ConfigArgs{Key: "cfg"}, ShadowOptions{})
	if err != nil || s.opts.Timeout != 30*time.Second {
		t.Fatalf("default timeout %v, %v", s.opts.Timeout, err)
	}
}

func deferredTask(t *testing.T, p pair, opts ShadowOptions) (*shadowLD, ShadowTask) {
	t.Helper()
	f := install(t, fallthroughPayload(), ldreason.NewEvalReasonFallthrough())
	var tasks []ShadowTask
	opts.OnTask = func(task ShadowTask) { tasks = append(tasks, task) }
	invokeAndWait(t, mustShadow(t, p.handlers(), opts))
	if len(tasks) != 1 {
		t.Fatalf("tasks %d", len(tasks))
	}
	return f, tasks[0]
}

func TestTimeoutIsTrackedOnce(t *testing.T) {
	p := newPair()
	p.b.delay = time.Second
	f, task := deferredTask(t, p, ShadowOptions{Timeout: 30 * time.Millisecond})
	res := RunShadow(context.Background(), task, p.handlers())
	time.Sleep(50 * time.Millisecond)
	ev := f.events(true)
	if res.Error != "timeout" || len(eventsNamed(ev, "$ld:ai:duration:total")) != 1 ||
		len(eventsNamed(ev, "$ld:ai:generation:error")) != 1 || len(ev) != 2 {
		t.Fatalf("result %+v, events %v", res, ev)
	}
}

func TestTimeoutIsReportedWhenTheHandlerStopsAtTheDeadline(t *testing.T) {
	stopsAtDeadline := &aiserver.ProviderHandler{Invoke: func(ctx context.Context, _ aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
		<-ctx.Done()
		return aiserver.HandlerResult{}, ctx.Err()
	}}
	h := withDeadline(stopsAtDeadline, time.Microsecond)
	for i := 0; i < 2000; i++ {
		if _, err := h.Invoke(context.Background(), aiserver.HandlerRequest{}); !errors.Is(err, errShadowTimeout) {
			t.Fatalf("run %d: got %v, want the shadow timeout", i, err)
		}
	}
}

func TestHandlerTimeoutErrorIsAnOrdinaryFailure(t *testing.T) {
	p := newPair()
	p.b.err = context.DeadlineExceeded
	f, task := deferredTask(t, p, ShadowOptions{})
	res := RunShadow(context.Background(), task, p.handlers())
	if res.Error != context.DeadlineExceeded.Error() || len(eventsNamed(f.events(true), "$ld:ai:generation:error")) != 1 {
		t.Fatalf("result %+v", res)
	}
}

func TestReportsOnlyShadowConfigUsage(t *testing.T) {
	f := install(t, fallthroughPayload(), ldreason.NewEvalReasonFallthrough())
	invokeAndWait(t, mustShadow(t, newPair().handlers(), ShadowOptions{}))
	if got := f.usage(); !slices.Equal(got, []string{"experimental.shadowConfig"}) {
		t.Fatalf("usage %v", got)
	}
}

func TestRunShadowReportsUsage(t *testing.T) {
	p := newPair()
	f, task := deferredTask(t, p, ShadowOptions{})
	RunShadow(context.Background(), task, p.handlers())
	if got := f.usage(); !slices.Equal(got, []string{"experimental.shadowConfig", "experimental.runShadow"}) {
		t.Fatalf("usage %v", got)
	}
}

func TestOnTaskReceivesEachSampledShadowAndNothingRuns(t *testing.T) {
	value := served(map[string]any{"fallthrough": []any{entry("s-1"), entry("s-2")}})
	f := install(t, value, ldreason.NewEvalReasonFallthrough())
	p := newPair()
	var ids []string
	invokeAndWait(t, mustShadow(t, p.handlers(), ShadowOptions{OnTask: func(task ShadowTask) { ids = append(ids, task.ShadowRequestID) }}))
	if !slices.Equal(ids, []string{"s-1", "s-2"}) || p.b.count() != 0 || len(f.events(true)) != 0 {
		t.Fatalf("ids %v, shadow calls %d", ids, p.b.count())
	}
}

func TestOnTaskPanicIsLoggedAndSwallowed(t *testing.T) {
	install(t, fallthroughPayload(), ldreason.NewEvalReasonFallthrough())
	var logs bytes.Buffer
	aiserver.SetLogger(slog.New(slog.NewTextHandler(&logs, nil)))
	t.Cleanup(func() { aiserver.SetLogger(nil) })
	resp := invokeAndWait(t, mustShadow(t, newPair().handlers(), ShadowOptions{OnTask: func(ShadowTask) { panic("queue full") }}))
	if resp.Response != "served" || !strings.Contains(logs.String(), "OnTask callback failed") {
		t.Fatalf("response %v, logs %q", resp.Response, logs.String())
	}
}

func TestTaskCopiesVariablesAndHistory(t *testing.T) {
	install(t, fallthroughPayload(), ldreason.NewEvalReasonFallthrough())
	var task ShadowTask
	vars := map[string]any{"nested": map[string]any{"k": "v"}}
	history := []aiserver.Message{{Role: "user", Blocks: []aiserver.ContentBlock{{Type: "text", Text: "hi"}}}}
	invokeAndWait(t, mustShadow(t, newPair().handlers(), ShadowOptions{OnTask: func(t ShadowTask) { task = t }}),
		aiserver.WithVariables(vars), aiserver.WithHistory(history))
	vars["nested"].(map[string]any)["k"] = "changed"
	history[0].Blocks[0].Text = "changed"
	if task.Variables["nested"].(map[string]any)["k"] != "v" || task.History[0].Blocks[0].Text != "hi" {
		t.Fatal("the task shares the caller's variables or history")
	}
	if task.UserContext.Key() != shadowUser.Key() {
		t.Fatal("the task lost the context")
	}
}

func TestServedResponseSurvivesAFailureBuildingShadows(t *testing.T) {
	install(t, fallthroughPayload(), ldreason.NewEvalReasonFallthrough())
	var logs bytes.Buffer
	aiserver.SetLogger(slog.New(slog.NewTextHandler(&logs, nil)))
	t.Cleanup(func() { aiserver.SetLogger(nil) })
	p := newPair()
	handlers := []*aiserver.ProviderHandler{p.a.handler("OpenAI"), nil}
	resp := invokeAndWait(t, mustShadow(t, handlers, ShadowOptions{}))
	if resp.Response != "served" || !strings.Contains(logs.String(), "could not be prepared") {
		t.Fatalf("response %v, logs %q", resp.Response, logs.String())
	}
}

func TestTaskSurvivesJSONAndMatchesBackgroundMode(t *testing.T) {
	p := newPair()
	f, task := deferredTask(t, p, ShadowOptions{})
	b, err := json.Marshal(task)
	if err != nil {
		t.Fatal(err)
	}
	var decoded ShadowTask
	if err := json.Unmarshal(b, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.ShadowRequestID != task.ShadowRequestID || decoded.UserContext.Key() != task.UserContext.Key() ||
		decoded.Timeout != task.Timeout || !reflect.DeepEqual(decoded.TrackData, task.TrackData) {
		t.Fatalf("decoded %+v", decoded)
	}
	res := RunShadow(context.Background(), decoded, p.handlers())
	deferred := f.events(true)

	f2 := install(t, fallthroughPayload(), ldreason.NewEvalReasonFallthrough())
	invokeAndWait(t, mustShadow(t, p.handlers(), ShadowOptions{}))
	if res.Error != "" || !reflect.DeepEqual(withoutRunID(deferred), withoutRunID(f2.events(true))) {
		t.Fatalf("result %+v\ndeferred %v\nbackground %v", res, deferred, f2.events(true))
	}
	if res.TrackData["variationKey"] != "claude-candidate" || res.TrackData["isShadow"] != true || res.Usage.Total != 7 {
		t.Fatalf("result %+v", res)
	}
}

func TestRunShadowNeverFails(t *testing.T) {
	p := newPair()
	_, task := deferredTask(t, p, ShadowOptions{})
	if res := RunShadow(context.Background(), task, nil); res.Error == "" {
		t.Fatal("no handler must be an error result")
	}
	p.b.panics = true
	if res := RunShadow(context.Background(), task, p.handlers()); res.Error == "" || res.TrackData["shadowRequestId"] != "shadow-1" {
		t.Fatalf("panicking handler: %+v", res)
	}
}

func TestShadowEventsDescribeTheShadowVariation(t *testing.T) {
	f := install(t, fallthroughPayload(), ldreason.NewEvalReasonFallthrough())
	resp := invokeAndWait(t, mustShadow(t, newPair().handlers(), ShadowOptions{}))
	ev := f.events(true)
	if len(ev) == 0 {
		t.Fatal("no shadow events")
	}
	want := map[string]any{
		"configKey": "cfg", "variationKey": "claude-candidate", "version": 7.0, "modelName": "claude",
		"providerName": "Anthropic", "runId": resp.TrackData.RunID, "isShadow": true,
		"primaryVariationKey": "gpt-main", "shadowRequestId": "shadow-1", "ownerKind": "member", "ownerId": "m-1",
	}
	for _, e := range ev {
		for k, v := range want {
			if e.data[k] != v {
				t.Errorf("%s %s = %v, want %v", e.event, k, e.data[k], v)
			}
		}
	}
}

func TestNothingIsAddedToServedEvents(t *testing.T) {
	f := install(t, fallthroughPayload(), ldreason.NewEvalReasonFallthrough())
	resp := invokeAndWait(t, mustShadow(t, newPair().handlers(), ShadowOptions{}))
	b, _ := json.Marshal(resp.TrackData)
	for _, e := range f.events(false) {
		for _, k := range []string{"isShadow", "primaryVariationKey", "shadowRequestId", "ownerKind", "ownerId"} {
			if _, ok := e.data[k]; ok || strings.Contains(string(b), k) {
				t.Errorf("%s carries %s", e.event, k)
			}
		}
	}
}

func recordSpans(t *testing.T) *tracetest.SpanRecorder {
	t.Helper()
	rec := tracetest.NewSpanRecorder()
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(rec)))
	t.Cleanup(func() { otel.SetTracerProvider(prev) })
	return rec
}

func rootAttrs(rec *tracetest.SpanRecorder) []map[attribute.Key]attribute.Value {
	var out []map[attribute.Key]attribute.Value
	for _, s := range rec.Ended() {
		m := map[attribute.Key]attribute.Value{}
		for _, a := range s.Attributes() {
			m[a.Key] = a.Value
		}
		out = append(out, m)
	}
	return out
}

func TestShadowSpanAttributesReachTheRootSpan(t *testing.T) {
	install(t, fallthroughPayload(), ldreason.NewEvalReasonFallthrough())
	rec := recordSpans(t)
	p := newPair()
	p.a.span, p.b.span = true, true
	invokeAndWait(t, mustShadow(t, p.handlers(), ShadowOptions{}))
	spans := rootAttrs(rec)
	if len(spans) != 2 {
		t.Fatalf("spans %d", len(spans))
	}
	servedSpan, shadowSpan := spans[0], spans[1]
	if _, ok := servedSpan["launchdarkly.is_shadow"]; ok {
		t.Fatal("the served span is marked as a shadow")
	}
	if !shadowSpan["launchdarkly.is_shadow"].AsBool() ||
		shadowSpan["launchdarkly.shadow.request.id"].AsString() != "shadow-1" ||
		shadowSpan["launchdarkly.shadow.primary.variation.key"].AsString() != "gpt-main" ||
		shadowSpan["launchdarkly.variation.key"].AsString() != "claude-candidate" {
		t.Fatalf("shadow span %v", shadowSpan)
	}
}

func TestTemplateVariablesCannotSetSpanAttributes(t *testing.T) {
	install(t, served(map[string]any{}), ldreason.NewEvalReasonFallthrough())
	rec := recordSpans(t)
	p := newPair()
	p.a.span = true
	forged := map[string]any{
		"__ldSpanAttributes": []attribute.KeyValue{attribute.Bool("launchdarkly.is_shadow", true)},
	}
	invokeAndWait(t, mustShadow(t, p.handlers(), ShadowOptions{}), aiserver.WithVariables(forged))
	if _, ok := rootAttrs(rec)[0]["launchdarkly.is_shadow"]; ok {
		t.Fatal("a template variable set a span attribute")
	}
}

func TestShutdownCancelsRunningShadows(t *testing.T) {
	f := install(t, fallthroughPayload(), ldreason.NewEvalReasonFallthrough())
	started := make(chan struct{})
	cancelled := make(chan struct{})
	blocking := &aiserver.ProviderHandler{
		ProvidesFor: aiserver.ProvidesFor{Provider: "Anthropic", Mode: aiserver.ModeMessages},
		Invoke: func(ctx context.Context, _ aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
			close(started)
			<-ctx.Done()
			close(cancelled)
			return aiserver.HandlerResult{}, ctx.Err()
		},
	}
	p := newPair()
	s := mustShadow(t, []*aiserver.ProviderHandler{p.a.handler("OpenAI"), blocking}, ShadowOptions{Timeout: time.Minute})
	if _, err := s.Invoke(context.Background(), "hello", shadowUser); err != nil {
		t.Fatal(err)
	}
	<-started
	if err := aiserver.Shutdown(context.Background()); err != nil {
		t.Fatal(err)
	}
	select {
	case <-cancelled:
	case <-time.After(5 * time.Second):
		t.Fatal("Shutdown did not cancel the running shadow")
	}
	running.Wait()
	if ev := f.events(true); len(ev) != 0 {
		t.Fatalf("a cancelled shadow recorded %v", ev)
	}
}
