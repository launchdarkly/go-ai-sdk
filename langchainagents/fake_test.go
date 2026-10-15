package langchainagents

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"testing"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/launchdarkly/go-sdk-common/v3/ldvalue"
	"github.com/tmc/langchaingo/llms"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

var bg = context.Background()

// scripted is one canned model turn: chunks are streamed (when the caller
// streams) before resp is returned; err fails the call.
type scripted struct {
	resp   *llms.ContentResponse
	chunks []string
	err    error
}

type call struct {
	msgs []llms.MessageContent
	opts llms.CallOptions
}

// fakeModel is an llms.Model that replays scripted turns and records calls.
// The last turn repeats when repeat is set.
type fakeModel struct {
	mu     sync.Mutex
	turns  []scripted
	repeat bool
	calls  []call
}

func (f *fakeModel) GenerateContent(ctx context.Context, msgs []llms.MessageContent, options ...llms.CallOption) (*llms.ContentResponse, error) {
	var o llms.CallOptions
	for _, opt := range options {
		opt(&o)
	}
	f.mu.Lock()
	f.calls = append(f.calls, call{msgs: append([]llms.MessageContent(nil), msgs...), opts: o})
	i := len(f.calls) - 1
	if f.repeat && i >= len(f.turns) {
		i = len(f.turns) - 1
	}
	f.mu.Unlock()
	if i >= len(f.turns) {
		return nil, errors.New("fake: no more scripted turns")
	}
	t := f.turns[i]
	if t.err != nil {
		return nil, t.err
	}
	if o.StreamingFunc != nil {
		for _, c := range t.chunks {
			if err := o.StreamingFunc(ctx, []byte(c)); err != nil {
				return nil, err
			}
		}
	}
	return t.resp, nil
}

func (f *fakeModel) Call(context.Context, string, ...llms.CallOption) (string, error) {
	return "", errors.New("fake: Call is not used")
}

func (f *fakeModel) call(i int) call {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.calls[i]
}

func (f *fakeModel) count() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.calls)
}

func usageInfo(in, out int) map[string]any {
	return map[string]any{"InputTokens": in, "OutputTokens": out, "CacheReadInputTokens": 0, "CacheCreationInputTokens": 0}
}

func textTurn(text string, in, out int) scripted {
	return scripted{resp: &llms.ContentResponse{Choices: []*llms.ContentChoice{{
		Content: text, StopReason: "end_turn", GenerationInfo: usageInfo(in, out),
	}}}}
}

func streamTurn(chunks []string, in, out int, calls ...llms.ToolCall) scripted {
	stop := "end_turn"
	if len(calls) > 0 {
		stop = "tool_use"
	}
	return scripted{chunks: chunks, resp: &llms.ContentResponse{Choices: []*llms.ContentChoice{{
		Content: strings.Join(chunks, ""), StopReason: stop, ToolCalls: calls, GenerationInfo: usageInfo(in, out),
	}}}}
}

func toolCall(name string, args map[string]any) llms.ToolCall {
	b, _ := json.Marshal(args)
	return llms.ToolCall{ID: "call_" + name, Type: "function", FunctionCall: &llms.FunctionCall{Name: name, Arguments: string(b)}}
}

func toolTurn(name string, args map[string]any) scripted {
	return scripted{resp: &llms.ContentResponse{Choices: []*llms.ContentChoice{{
		StopReason: "tool_use", ToolCalls: []llms.ToolCall{toolCall(name, args)}, GenerationInfo: usageInfo(1, 1),
	}}}}
}

func withTracer(t *testing.T) *tracetest.InMemoryExporter {
	t.Helper()
	exp := tracetest.NewInMemoryExporter()
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(sdktrace.NewTracerProvider(sdktrace.WithSyncer(exp)))
	t.Cleanup(func() { otel.SetTracerProvider(prev) })
	return exp
}

// setup returns a handler over a fake model plus the span exporter.
func setup(t *testing.T, turns ...scripted) (*aiserver.ProviderHandler, *fakeModel, *tracetest.InMemoryExporter) {
	t.Helper()
	exp := withTracer(t)
	f := &fakeModel{turns: turns}
	return newHandler(Options{Model: f}), f, exp
}

type trackCall struct {
	event string
	ctx   ldcontext.Context
	value float64
	data  map[string]any
}

// fakeLD serves per-key variations (fallback for unknown keys) and records
// variation keys and track calls.
type fakeLD struct {
	variations map[string]map[string]any
	fallback   map[string]any
	mu         sync.Mutex
	keys       []string
	tracks     []trackCall
}

func (f *fakeLD) JSONVariation(key string, _ ldcontext.Context, def ldvalue.Value) (ldvalue.Value, error) {
	f.mu.Lock()
	f.keys = append(f.keys, key)
	f.mu.Unlock()
	v, ok := f.variations[key]
	if !ok {
		v = f.fallback
	}
	if v == nil {
		return def, nil
	}
	b, _ := json.Marshal(v)
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

func (f *fakeLD) requestedKeys() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.keys...)
}

// events returns the track calls for event, in order.
func (f *fakeLD) events(event string) []trackCall {
	f.mu.Lock()
	defer f.mu.Unlock()
	var out []trackCall
	for _, t := range f.tracks {
		if t.event == event {
			out = append(out, t)
		}
	}
	return out
}

func (f *fakeLD) resetTracks() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.tracks = nil
}

// usageHelpers lists the helper of every `$ld:ai:sdk:usage` event, in order.
func (f *fakeLD) usageHelpers() []string {
	var out []string
	for _, t := range f.events("$ld:ai:sdk:usage") {
		out = append(out, t.data["helper"].(string))
	}
	return out
}

// aiEvents lists every `$ld:ai:` event except sdk info/usage, in order.
func (f *fakeLD) aiEvents() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	var out []string
	for _, t := range f.tracks {
		if strings.HasPrefix(t.event, "$ld:ai:") && !strings.HasPrefix(t.event, "$ld:ai:sdk:") {
			out = append(out, t.event)
		}
	}
	return out
}

// installClient makes fake the LD client and routes spans to the returned
// exporter.
func installClient(t *testing.T, fake *fakeLD) *tracetest.InMemoryExporter {
	t.Helper()
	_ = aiserver.Shutdown(bg)
	if _, err := aiserver.InitWithClient(bg, fake, aiserver.InitOptions{OTLPEndpoint: "http://127.0.0.1:1/v1/traces"}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = aiserver.Shutdown(bg) })
	return withTracer(t) // InitWithClient installed its own provider
}
