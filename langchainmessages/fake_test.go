package langchainmessages

import (
	"context"
	"encoding/json"
	"errors"
	"sync"
	"testing"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/launchdarkly/go-sdk-common/v3/ldvalue"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/llms"
)

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
type fakeModel struct {
	mu    sync.Mutex
	turns []scripted
	calls []call
}

func (f *fakeModel) GenerateContent(ctx context.Context, msgs []llms.MessageContent, options ...llms.CallOption) (*llms.ContentResponse, error) {
	var o llms.CallOptions
	for _, opt := range options {
		opt(&o)
	}
	f.mu.Lock()
	f.calls = append(f.calls, call{msgs: append([]llms.MessageContent(nil), msgs...), opts: o})
	i := len(f.calls) - 1
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

func usageInfo(in, out int) map[string]any {
	return map[string]any{"InputTokens": in, "OutputTokens": out, "CacheReadInputTokens": 0, "CacheCreationInputTokens": 0}
}

func textTurn(text string, in, out int) scripted {
	return scripted{resp: &llms.ContentResponse{Choices: []*llms.ContentChoice{{
		Content: text, StopReason: "end_turn", GenerationInfo: usageInfo(in, out),
	}}}}
}

func streamTurn(chunks []string, in, out int, calls ...llms.ToolCall) scripted {
	text := ""
	for _, c := range chunks {
		text += c
	}
	stop := "end_turn"
	if len(calls) > 0 {
		stop = "tool_use"
	}
	return scripted{chunks: chunks, resp: &llms.ContentResponse{Choices: []*llms.ContentChoice{{
		Content: text, StopReason: stop, ToolCalls: calls, GenerationInfo: usageInfo(in, out),
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

type trackEvent struct {
	key  string
	data ldvalue.Value
}

// fakeLD serves one variation for every key and records track calls.
type fakeLD struct {
	variation map[string]any
	mu        sync.Mutex
	events    []trackEvent
	keys      []string // every flag key evaluated
}

func (f *fakeLD) JSONVariation(key string, _ ldcontext.Context, _ ldvalue.Value) (ldvalue.Value, error) {
	f.mu.Lock()
	f.keys = append(f.keys, key)
	f.mu.Unlock()
	b, _ := json.Marshal(f.variation)
	return ldvalue.Parse(b), nil
}

func (f *fakeLD) TrackMetric(key string, _ ldcontext.Context, _ float64, data ldvalue.Value) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.events = append(f.events, trackEvent{key, data})
	return nil
}

func (f *fakeLD) Flush()       {}
func (f *fakeLD) Close() error { return nil }

func (f *fakeLD) requestedKeys() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.keys...)
}

func (f *fakeLD) usage() []ldvalue.Value {
	f.mu.Lock()
	defer f.mu.Unlock()
	var out []ldvalue.Value
	for _, e := range f.events {
		if e.key == "$ld:ai:sdk:usage" {
			out = append(out, e.data)
		}
	}
	return out
}
