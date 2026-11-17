package litellmmessages

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/launchdarkly/go-sdk-common/v3/ldvalue"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

var bg = context.Background()

// recorded is one request the fake proxy received.
type recorded struct {
	path   string
	header http.Header
	body   map[string]any
}

// fakeProxy is a LiteLLM proxy double: it replays canned answers in order
// and records every request. An answer starting "data:" is an SSE stream;
// "STATUS500" is a retryable server error; "HOLD" streams one text chunk and
// keeps the connection open until the client goes away (reported on closed).
type fakeProxy struct {
	mu       sync.Mutex
	answers  []string
	requests []recorded
	closed   chan struct{}
	url      string
}

func newProxy(t *testing.T, answers ...string) *fakeProxy {
	t.Helper()
	p := &fakeProxy{answers: answers, closed: make(chan struct{}, 1)}
	srv := httptest.NewServer(p)
	t.Cleanup(srv.Close)
	p.url = srv.URL
	return p
}

func (p *fakeProxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	raw, _ := io.ReadAll(r.Body)
	var body map[string]any
	_ = json.Unmarshal(raw, &body)
	p.mu.Lock()
	p.requests = append(p.requests, recorded{path: r.URL.Path, header: r.Header.Clone(), body: body})
	i := len(p.requests) - 1
	p.mu.Unlock()
	answer := `STATUS500`
	if i < len(p.answers) {
		answer = p.answers[i]
	}
	switch {
	case answer == "STATUS500":
		w.Header().Set("Retry-After-Ms", "1")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = io.WriteString(w, `{"error":{"message":"upstream down","type":"server_error"}}`)
	case answer == "HOLD":
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w, "data: %s\n\n", chunk(`{"role":"assistant","content":"first"}`, ""))
		w.(http.Flusher).Flush()
		select {
		case <-r.Context().Done():
			p.closed <- struct{}{}
		case <-time.After(5 * time.Second):
		}
	case strings.HasPrefix(answer, "data:"):
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, answer)
	default:
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, answer)
	}
}

func (p *fakeProxy) request(i int) recorded {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.requests[i]
}

func (p *fakeProxy) count() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.requests)
}

const servedModel = "served-model-2026" // what the proxy reports

type call struct{ id, name, args string }

// completion is a blocking Chat Completions answer.
func completion(text string, calls []call, in, out int) string {
	finish := "stop"
	msg := map[string]any{"role": "assistant", "content": text}
	if len(calls) > 0 {
		finish = "tool_calls"
		var tcs []any
		for _, c := range calls {
			tcs = append(tcs, map[string]any{"id": c.id, "type": "function", "function": map[string]any{"name": c.name, "arguments": c.args}})
		}
		msg["tool_calls"] = tcs
	}
	b, _ := json.Marshal(map[string]any{
		"id": "c", "object": "chat.completion", "created": 0, "model": servedModel,
		"choices": []any{map[string]any{"index": 0, "finish_reason": finish, "message": msg}},
		"usage":   map[string]any{"prompt_tokens": in, "completion_tokens": out, "total_tokens": in + out},
	})
	return string(b)
}

func text(s string, in, out int) string { return completion(s, nil, in, out) }

func chunk(delta, finish string) string {
	f := "null"
	if finish != "" {
		f = `"` + finish + `"`
	}
	return `{"id":"s","object":"chat.completion.chunk","created":0,"model":"` + servedModel + `","choices":[{"index":0,"delta":` + delta + `,"finish_reason":` + f + `}]}`
}

// stream is an SSE answer: the chunks, a usage chunk, then [DONE].
func stream(in, out int, chunks ...string) string {
	var b strings.Builder
	for _, c := range chunks {
		fmt.Fprintf(&b, "data: %s\n\n", c)
	}
	fmt.Fprintf(&b, `data: {"id":"s","object":"chat.completion.chunk","created":0,"model":"%s","choices":[],"usage":{"prompt_tokens":%d,"completion_tokens":%d,"total_tokens":%d}}`+"\n\n", servedModel, in, out, in+out)
	b.WriteString("data: [DONE]\n\n")
	return b.String()
}

func retries(n int) *int { return &n }

// setup is a handler on a fresh proxy (no retries) with spans exported to exp.
func setup(t *testing.T, answers ...string) (*aiserver.ProviderHandler, *fakeProxy, *tracetest.InMemoryExporter) {
	t.Helper()
	p := newProxy(t, answers...)
	exp := withTracer(t)
	h, err := NewHandler(Options{BaseURL: p.url, APIKey: "proxy-key", MaxRetries: retries(0)})
	if err != nil {
		t.Fatalf("NewHandler: %v", err)
	}
	return h, p, exp
}

func withTracer(t *testing.T) *tracetest.InMemoryExporter {
	t.Helper()
	exp := tracetest.NewInMemoryExporter()
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(sdktrace.NewTracerProvider(sdktrace.WithSyncer(exp)))
	t.Cleanup(func() { otel.SetTracerProvider(prev) })
	return exp
}

// cfg is an evaluated config for model under provider.
func cfg(model, provider string, edit func(*aiserver.AiConfig)) aiserver.AiConfig {
	c := aiserver.AiConfig{Model: aiserver.Model{Name: model}}
	c.Provider.Name = provider
	if edit != nil {
		edit(&c)
	}
	return c
}

func req(c aiserver.AiConfig, input string) aiserver.HandlerRequest {
	return aiserver.HandlerRequest{Config: c, UserInput: input}
}

// drain runs a stream to the end and returns its events and first error.
func drain(h *aiserver.ProviderHandler, r aiserver.HandlerRequest) ([]aiserver.HandlerStreamEvent, error) {
	var evs []aiserver.HandlerStreamEvent
	for ev, err := range h.Stream(bg, r) {
		if err != nil {
			return evs, err
		}
		evs = append(evs, ev)
	}
	return evs, nil
}

func tool(fn func(map[string]any) (any, error)) aiserver.ToolFunc {
	return func(_ context.Context, args map[string]any) (any, error) { return fn(args) }
}

func funcTool(desc string, params map[string]any) aiserver.Tool {
	return aiserver.Tool{Type: "function", Description: desc, Parameters: params}
}

func spansByName(exp *tracetest.InMemoryExporter) map[string][]sdktrace.ReadOnlySpan {
	out := map[string][]sdktrace.ReadOnlySpan{}
	for _, s := range exp.GetSpans().Snapshots() {
		out[s.Name()] = append(out[s.Name()], s)
	}
	return out
}

func attrs(s sdktrace.ReadOnlySpan) map[string]string {
	out := map[string]string{}
	for _, a := range s.Attributes() {
		out[string(a.Key)] = a.Value.Emit()
	}
	return out
}

// fakeLD serves one variation for every key and records track calls.
type fakeLD struct {
	variation map[string]any
	mu        sync.Mutex
	helpers   []string
}

func (f *fakeLD) JSONVariation(string, ldcontext.Context, ldvalue.Value) (ldvalue.Value, error) {
	b, _ := json.Marshal(f.variation)
	return ldvalue.Parse(b), nil
}

func (f *fakeLD) TrackMetric(event string, _ ldcontext.Context, _ float64, data ldvalue.Value) error {
	if event != "$ld:ai:sdk:usage" {
		return nil
	}
	var d map[string]any
	_ = json.Unmarshal([]byte(data.JSONString()), &d)
	f.mu.Lock()
	defer f.mu.Unlock()
	f.helpers = append(f.helpers, fmt.Sprint(d["helper"]))
	return nil
}

func (f *fakeLD) Flush()       {}
func (f *fakeLD) Close() error { return nil }

func installClient(t *testing.T, fake *fakeLD) {
	t.Helper()
	_ = aiserver.Shutdown(bg)
	if _, err := aiserver.InitWithClient(bg, fake, aiserver.InitOptions{OTLPEndpoint: "http://127.0.0.1:1/v1/traces"}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = aiserver.Shutdown(bg) })
}
