package litellmagents

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

// recorded is one request the fake proxy received.
type recorded struct {
	path, auth string
	header     http.Header
	raw        string
	body       map[string]any
}

// fakeProxy is a LiteLLM proxy speaking Chat Completions. It replays canned
// responses and records requests. A response starting with "data:" is an SSE
// stream; "HANG:" + SSE writes the stream, then holds the connection until
// the client goes away and records that; "STATUS400" is an API error.
type fakeProxy struct {
	mu        sync.Mutex
	responses []string
	requests  []recorded
	closed    chan struct{} // closed when a HANG response sees the client leave
}

func (f *fakeProxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	b, _ := io.ReadAll(r.Body)
	var body map[string]any
	_ = json.Unmarshal(b, &body)
	f.mu.Lock()
	f.requests = append(f.requests, recorded{r.URL.Path, r.Header.Get("Authorization"), r.Header.Clone(), string(b), body})
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
		_, _ = io.WriteString(w, `{"error":{"message":"bad request","type":"invalid_request_error"}}`)
	case strings.HasPrefix(resp, "HANG:"):
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, strings.TrimPrefix(resp, "HANG:"))
		w.(http.Flusher).Flush()
		<-r.Context().Done()
		close(f.closed)
	case strings.HasPrefix(resp, "data:"):
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, resp)
	default:
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, resp)
	}
}

func (f *fakeProxy) request(i int) map[string]any { return f.rec(i).body }

// rec returns request i, or the zero value when fewer were sent, so a
// failing assertion reports instead of panicking.
func (f *fakeProxy) rec(i int) recorded {
	f.mu.Lock()
	defer f.mu.Unlock()
	if i >= len(f.requests) {
		return recorded{}
	}
	return f.requests[i]
}

func (f *fakeProxy) count() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.requests)
}

// proxy starts a fake proxy and returns it with its base URL.
func proxy(t *testing.T, responses ...string) (*fakeProxy, string) {
	t.Helper()
	f := &fakeProxy{responses: responses, closed: make(chan struct{})}
	srv := httptest.NewServer(f)
	t.Cleanup(srv.Close)
	return f, srv.URL
}

// fakeClient returns a fake proxy and an openai-go client pointed at it.
func fakeClient(t *testing.T, responses ...string) (*fakeProxy, *openai.Client) {
	t.Helper()
	f, url := proxy(t, responses...)
	c := openai.NewClient(option.WithBaseURL(url), option.WithAPIKey("test"), option.WithMaxRetries(0))
	return f, &c
}

// tripwire fails the test if anything reaches it, and points the OPENAI_*
// environment variables at it, so a default client would be caught.
func tripwire(t *testing.T) {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Errorf("request reached the default OpenAI endpoint: %s %s", r.Method, r.URL.Path)
		http.Error(w, "{}", 500)
	}))
	t.Cleanup(srv.Close)
	t.Setenv("OPENAI_BASE_URL", srv.URL)
	t.Setenv("OPENAI_API_KEY", "sk-from-openai-env")
	t.Setenv("LITELLM_BASE_URL", "")
	t.Setenv("LITELLM_API_KEY", "")
}

const servedModel = "served-2026-01-01" // what the proxy reports; requests use the config's name

func usageJSON(in, cached, out int) string {
	return fmt.Sprintf(`{"prompt_tokens":%d,"completion_tokens":%d,"total_tokens":%d,"prompt_tokens_details":{"cached_tokens":%d}}`, in, out, in+out, cached)
}

// completion is a blocking Chat Completions response.
func completion(message, finish string, in, cached, out int) string {
	return fmt.Sprintf(`{"id":"chatcmpl","object":"chat.completion","created":0,"model":%q,"choices":[{"index":0,"message":%s,"finish_reason":%q}],"usage":%s}`,
		servedModel, message, finish, usageJSON(in, cached, out))
}

func textResp(text string, in, out int) string {
	return completion(fmt.Sprintf(`{"role":"assistant","content":%q}`, text), "stop", in, 0, out)
}

// call is one tool call in a response.
type call struct{ name, args string }

func callsJSON(calls []call) string {
	var parts []string
	for _, c := range calls {
		parts = append(parts, fmt.Sprintf(`{"id":"call_%s","type":"function","function":{"name":%q,"arguments":%q}}`, c.name, c.name, c.args))
	}
	return "[" + strings.Join(parts, ",") + "]"
}

func toolResp(calls ...call) string {
	return completion(`{"role":"assistant","content":null,"tool_calls":`+callsJSON(calls)+`}`, "tool_calls", 1, 0, 1)
}

func chunk(delta string, finish string) string {
	f := "null"
	if finish != "" {
		f = fmt.Sprintf("%q", finish)
	}
	return fmt.Sprintf("data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":%q,\"choices\":[{\"index\":0,\"delta\":%s,\"finish_reason\":%s}]}\n\n", servedModel, delta, f)
}

// sse streams text deltas, then a usage chunk and [DONE].
func sse(deltas []string, in, out int) string {
	var b strings.Builder
	for _, d := range deltas {
		b.WriteString(chunk(fmt.Sprintf(`{"content":%q}`, d), ""))
	}
	b.WriteString(chunk(`{}`, "stop"))
	fmt.Fprintf(&b, "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":%q,\"choices\":[],\"usage\":%s}\n\n", servedModel, usageJSON(in, 0, out))
	b.WriteString("data: [DONE]\n\n")
	return b.String()
}

// sseToolCall streams one tool call whose arguments arrive in fragments.
func sseToolCall(name string, fragments ...string) string {
	var b strings.Builder
	b.WriteString(chunk(fmt.Sprintf(`{"tool_calls":[{"index":0,"id":"call_%s","type":"function","function":{"name":%q,"arguments":""}}]}`, name, name), ""))
	for _, f := range fragments {
		b.WriteString(chunk(fmt.Sprintf(`{"tool_calls":[{"index":0,"function":{"arguments":%q}}]}`, f), ""))
	}
	b.WriteString(chunk(`{}`, "tool_calls"))
	fmt.Fprintf(&b, "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":%q,\"choices\":[],\"usage\":%s}\n\n", servedModel, usageJSON(1, 0, 1))
	b.WriteString("data: [DONE]\n\n")
	return b.String()
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

// handler builds a handler with opts, failing the test on an error.
func handler(t *testing.T, opts Options) *aiserver.ProviderHandler {
	t.Helper()
	h, err := NewHandler(opts)
	if err != nil || h == nil {
		t.Fatalf("NewHandler = %v, %v", h, err)
	}
	return h
}

func setup(t *testing.T, responses ...string) (*aiserver.ProviderHandler, *fakeProxy, *tracetest.InMemoryExporter) {
	t.Helper()
	f, c := fakeClient(t, responses...)
	return handler(t, Options{Client: c}), f, tracer(t)
}

type trackCall struct {
	event string
	ctx   ldcontext.Context
	value float64
	data  map[string]any
}

// fakeLD serves per-key variations (fallback for unknown keys) and records
// track calls.
type fakeLD struct {
	variations map[string]map[string]any
	fallback   map[string]any
	mu         sync.Mutex
	tracks     []trackCall
}

func (f *fakeLD) JSONVariation(key string, _ ldcontext.Context, def ldvalue.Value) (ldvalue.Value, error) {
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

func (f *fakeLD) usageHelpers() []string {
	var out []string
	for _, t := range f.events("$ld:ai:sdk:usage") {
		out = append(out, t.data["helper"].(string))
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
	return tracer(t)
}
