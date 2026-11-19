package typesafe

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"sync"
	"testing"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/launchdarkly/go-sdk-common/v3/ldvalue"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

var bg = context.Background()

// realNetwork records every request the guard transport refused.
var realNetwork struct {
	mu   sync.Mutex
	urls []string
}

// guardTransport lets loopback requests (httptest servers) through and
// refuses everything else, so no test can reach the real TypeSafe API.
type guardTransport struct{ next http.RoundTripper }

func (g guardTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	if ip := net.ParseIP(r.URL.Hostname()); ip == nil || !ip.IsLoopback() {
		realNetwork.mu.Lock()
		realNetwork.urls = append(realNetwork.urls, r.URL.String())
		realNetwork.mu.Unlock()
		return nil, errors.New("test guard: real network request refused")
	}
	return g.next.RoundTrip(r)
}

// TestMain points the client at an unreachable loopback port unless a test
// sets its own fake, guards the default transport, and fails the run if any
// request tried to leave the machine.
func TestMain(m *testing.M) {
	os.Setenv("TYPESAFE_BASE_URL", "http://127.0.0.1:1")
	os.Setenv("TYPESAFE_API_KEY", "test-key")
	os.Unsetenv("TYPESAFE_DEFAULT_MODEL")
	http.DefaultTransport = guardTransport{next: http.DefaultTransport}
	code := m.Run()
	if len(realNetwork.urls) > 0 {
		fmt.Fprintf(os.Stderr, "requests tried to reach the real network: %v\n", realNetwork.urls)
		code = 1
	}
	os.Exit(code)
}

// jevRequest is one request the fake System One server received.
type jevRequest struct {
	method, path string
	header       http.Header
	body         map[string]any
}

// reply is one canned fake response.
type reply struct {
	status  int // 0 means 200
	body    string
	headers map[string]string
}

// fakeJev is a System One server. It replays replies in order and repeats
// the last one when it runs out.
type fakeJev struct {
	mu       sync.Mutex
	replies  []reply
	requests []jevRequest
	srv      *httptest.Server
}

func (f *fakeJev) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	b, _ := io.ReadAll(r.Body)
	var body map[string]any
	_ = json.Unmarshal(b, &body)
	f.mu.Lock()
	f.requests = append(f.requests, jevRequest{r.Method, r.URL.Path, r.Header.Clone(), body})
	rp := f.replies[min(len(f.requests), len(f.replies))-1]
	f.mu.Unlock()
	for k, v := range rp.headers {
		w.Header().Set(k, v)
	}
	w.Header().Set("Content-Type", "application/json")
	if rp.status != 0 {
		w.WriteHeader(rp.status)
	}
	_, _ = io.WriteString(w, rp.body)
}

func (f *fakeJev) reqs() []jevRequest {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]jevRequest(nil), f.requests...)
}

// newJev starts a fake System One server and points TYPESAFE_BASE_URL at it.
func newJev(t *testing.T, replies ...reply) *fakeJev {
	t.Helper()
	f := &fakeJev{replies: replies}
	f.srv = httptest.NewServer(f)
	t.Cleanup(f.srv.Close)
	t.Setenv("TYPESAFE_BASE_URL", f.srv.URL)
	t.Setenv("TYPESAFE_API_KEY", "test-key")
	return f
}

// ok is a 200 System One response.
func ok(answers string, in, out int) reply {
	return reply{body: fmt.Sprintf(`{"model":"jev-2026-01-01","answers":%s,"usage":{"input_tokens":%d,"output_tokens":%d}}`, answers, in, out)}
}

// The same three labels as the JavaScript handler tests, one of each type.
var classifiers = []any{
	map[string]any{"key": "tone", "eventKey": "$ld:ai:judge:jev:tone", "instructions": "Is the tone appropriate?",
		"type": "noul", "criteria": map[string]any{"true": "Polite", "false": "Harmful"}},
	map[string]any{"key": "accuracy", "eventKey": "$ld:ai:judge:jev:accuracy", "instructions": "How accurate is the response?",
		"type": "score", "criteria": []any{"Great", "Ok", "Bad"}},
	map[string]any{"key": "route", "eventKey": "$ld:ai:judge:jev:route", "instructions": "What is the tone?",
		"type": "choice", "criteria": map[string]any{"calm": "Calm", "angry": "Angry"}},
}

const threeAnswers = `{"tone":{"type":"noul","noul":0.25},` +
	`"accuracy":{"type":"score","score":1,"confidence":0.7,"legend":{"0":"Great","1":"Ok","2":"Bad"}},` +
	`"route":{"type":"choice","choice":"calm","confidence":0.6,"probabilities":{"calm":0.8,"angry":0.2}}}`

// judgeVariation is a TypeSafe judge variation as LaunchDarkly delivers it:
// classifiers, an empty messages list, and no instructions.
func judgeVariation(model string, labels any) map[string]any {
	return map[string]any{
		"model": map[string]any{"name": model}, "provider": map[string]any{"name": "TypeSafe"},
		"messages": []any{}, "classifiers": labels,
		"_ldMeta": map[string]any{"enabled": true, "variationKey": "jev-v", "version": 1, "mode": "judge"},
	}
}

// configOf decodes a variation into the AiConfig a handler receives.
func configOf(t *testing.T, variation map[string]any) aiserver.AiConfig {
	t.Helper()
	b, err := json.Marshal(variation)
	if err != nil {
		t.Fatal(err)
	}
	var cfg aiserver.AiConfig
	if err := json.Unmarshal(b, &cfg); err != nil {
		t.Fatal(err)
	}
	return cfg
}

func must[T any](v T, err error) T {
	if err != nil {
		panic(err)
	}
	return v
}

// output decodes the handler's JSON output text.
func output(t *testing.T, res aiserver.HandlerResult) map[string]any {
	t.Helper()
	s, isText := res.Output.(string)
	if !isText {
		t.Fatalf("output = %#v, want the JSON text", res.Output)
	}
	var m map[string]any
	if err := json.Unmarshal([]byte(s), &m); err != nil {
		t.Fatalf("output is not JSON: %q", s)
	}
	return m
}

type trackCall struct {
	event string
	ctx   ldcontext.Context
	value float64
	data  map[string]any
}

// fakeLD serves variations by key and records track calls.
type fakeLD struct {
	variations map[string]map[string]any
	mu         sync.Mutex
	tracks     []trackCall
}

func (f *fakeLD) JSONVariation(key string, _ ldcontext.Context, def ldvalue.Value) (ldvalue.Value, error) {
	v, has := f.variations[key]
	if !has {
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

func (f *fakeLD) events(name string) []trackCall {
	f.mu.Lock()
	defer f.mu.Unlock()
	var out []trackCall
	for _, tr := range f.tracks {
		if tr.event == name {
			out = append(out, tr)
		}
	}
	return out
}

// installClient makes fake the LD client with fresh process state and routes
// spans to exp.
func installClient(t *testing.T, fake *fakeLD, exp *tracetest.InMemoryExporter) {
	t.Helper()
	_ = aiserver.Shutdown(bg)
	if _, err := aiserver.InitWithClient(bg, fake, aiserver.InitOptions{OTLPEndpoint: "http://127.0.0.1:1/v1/traces"}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = aiserver.Shutdown(bg) })
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(sdktrace.NewTracerProvider(sdktrace.WithSyncer(exp)))
	t.Cleanup(func() { otel.SetTracerProvider(prev) })
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
	for _, kv := range s.Attributes() {
		out[string(kv.Key)] = kv.Value.Emit()
	}
	return out
}

func eventNames(s sdktrace.ReadOnlySpan) []string {
	var out []string
	for _, e := range s.Events() {
		out = append(out, e.Name)
	}
	return out
}

// spanText is every attribute and event attribute value on s, for content
// checks that must not depend on which attribute carries the content.
func spanText(s sdktrace.ReadOnlySpan) []string {
	var out []string
	add := func(kvs []attribute.KeyValue) {
		for _, kv := range kvs {
			out = append(out, kv.Value.Emit())
		}
	}
	add(s.Attributes())
	for _, e := range s.Events() {
		add(e.Attributes)
	}
	return out
}

func jsonText(v any) (string, error) {
	b, err := json.Marshal(v)
	return string(b), err
}

// eventAttr is the string value of key on the first event named name.
func eventAttr(s sdktrace.ReadOnlySpan, name, key string) (string, bool) {
	for _, e := range s.Events() {
		if e.Name != name {
			continue
		}
		for _, a := range e.Attributes {
			if string(a.Key) == key {
				return a.Value.AsString(), true
			}
		}
	}
	return "", false
}
