package openaiagents

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

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/launchdarkly/go-sdk-common/v3/ldvalue"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

var bg = context.Background()

// fakeOpenAI replays canned responses and records request bodies.
type fakeOpenAI struct {
	mu        sync.Mutex
	responses []string // JSON bodies; SSE streams start with "event:"; "STATUS400" is an API error
	requests  []map[string]any
}

func (f *fakeOpenAI) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	b, _ := io.ReadAll(r.Body)
	var body map[string]any
	_ = json.Unmarshal(b, &body)
	f.mu.Lock()
	f.requests = append(f.requests, body)
	i := len(f.requests) - 1
	f.mu.Unlock()
	if i >= len(f.responses) {
		http.Error(w, `{"error":{"message":"no more responses","type":"server_error"}}`, 500)
		return
	}
	resp := f.responses[i]
	switch {
	case resp == "STATUS400":
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(400)
		resp = `{"error":{"message":"bad request","type":"invalid_request_error"}}`
	case strings.HasPrefix(resp, "event:"):
		w.Header().Set("Content-Type", "text/event-stream")
	default:
		w.Header().Set("Content-Type", "application/json")
	}
	_, _ = io.WriteString(w, resp)
}

func (f *fakeOpenAI) request(i int) map[string]any {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.requests[i]
}

func (f *fakeOpenAI) count() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.requests)
}

const servedModel = "gpt-test-2026-01-01" // what the API reports; requests use "test-model"

func textOutput(text string) string {
	return fmt.Sprintf(`[{"type":"message","id":"msg","role":"assistant","status":"completed","content":[{"type":"output_text","text":%q,"annotations":[]}]}]`, text)
}

func callItem(name, args string) string {
	return fmt.Sprintf(`{"type":"function_call","id":"fc_%s","call_id":"call_%s","name":%q,"arguments":%q,"status":"completed"}`, name, name, name, args)
}

func callOutput(name, args string) string { return "[" + callItem(name, args) + "]" }

func response(id, output string, in, cached, out int) string {
	return fmt.Sprintf(`{"id":%q,"object":"response","created_at":0,"status":"completed","model":%q,"output":%s,"usage":{"input_tokens":%d,"input_tokens_details":{"cached_tokens":%d},"output_tokens":%d,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":%d}}`,
		id, servedModel, output, in, cached, out, in+out)
}

func textResp(text string, in, out int) string {
	return response("resp_text", textOutput(text), in, 0, out)
}

func toolResp(id, name, args string) string { return response(id, callOutput(name, args), 1, 0, 1) }

// sse streams deltas and ends with response.completed carrying final.
func sse(deltas []string, final string) string {
	var b strings.Builder
	ev := func(name, data string) { fmt.Fprintf(&b, "event: %s\ndata: %s\n\n", name, data) }
	ev("response.created", `{"type":"response.created","sequence_number":0,"response":{"id":"r","status":"in_progress","output":[]}}`)
	for i, d := range deltas {
		ev("response.output_text.delta", fmt.Sprintf(`{"type":"response.output_text.delta","sequence_number":%d,"item_id":"msg","output_index":0,"content_index":0,"delta":%q}`, i+1, d))
	}
	ev("response.completed", fmt.Sprintf(`{"type":"response.completed","sequence_number":99,"response":%s}`, final))
	return b.String()
}

func fakeClient(t *testing.T, responses ...string) (*fakeOpenAI, *openai.Client) {
	t.Helper()
	f := &fakeOpenAI{responses: responses}
	srv := httptest.NewServer(f)
	t.Cleanup(srv.Close)
	c := openai.NewClient(option.WithBaseURL(srv.URL), option.WithAPIKey("test"), option.WithMaxRetries(0))
	return f, &c
}

// tracer routes spans to a fresh in-memory exporter for the test.
func tracer(t *testing.T) *tracetest.InMemoryExporter {
	t.Helper()
	exp := tracetest.NewInMemoryExporter()
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(sdktrace.NewTracerProvider(sdktrace.WithSyncer(exp)))
	t.Cleanup(func() { otel.SetTracerProvider(prev) })
	return exp
}

func setup(t *testing.T, responses ...string) (*aiserver.ProviderHandler, *fakeOpenAI, *tracetest.InMemoryExporter) {
	t.Helper()
	f, c := fakeClient(t, responses...)
	return newHandler(Options{Client: c}), f, tracer(t)
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

// resetTracks forgets every recorded track call.
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

// installClient makes fake the LD client with a fresh reported-usage set and
// routes spans to the returned exporter.
func installClient(t *testing.T, fake *fakeLD) *tracetest.InMemoryExporter {
	t.Helper()
	_ = aiserver.Shutdown(bg)
	if _, err := aiserver.InitWithClient(bg, fake, aiserver.InitOptions{OTLPEndpoint: "http://127.0.0.1:1/v1/traces"}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = aiserver.Shutdown(bg) })
	return tracer(t) // InitWithClient installed its own provider
}
