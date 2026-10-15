package claudeagents

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

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/launchdarkly/go-sdk-common/v3/ldvalue"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

var bg = context.Background()

var user = ldcontext.New("user-123")

// fakeAnthropic replays canned responses in order and records request bodies.
type fakeAnthropic struct {
	mu        sync.Mutex
	responses []string // JSON bodies, SSE streams ("event:" prefix), "STATUS400" or "STATUS529"
	requests  []map[string]any
	headers   []http.Header
}

func (f *fakeAnthropic) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	b, _ := io.ReadAll(r.Body)
	var body map[string]any
	_ = json.Unmarshal(b, &body)
	f.mu.Lock()
	f.requests = append(f.requests, body)
	f.headers = append(f.headers, r.Header.Clone())
	i := len(f.requests) - 1
	f.mu.Unlock()
	if i >= len(f.responses) {
		http.Error(w, `{"type":"error","error":{"type":"api_error","message":"no more"}}`, 500)
		return
	}
	resp := f.responses[i]
	switch {
	case resp == "STATUS400":
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(400)
		resp = `{"type":"error","error":{"type":"invalid_request_error","message":"bad"}}`
	case resp == "STATUS529":
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(529)
		resp = `{"type":"error","error":{"type":"overloaded_error","message":"overloaded"}}`
	case strings.HasPrefix(resp, "event:"):
		w.Header().Set("Content-Type", "text/event-stream")
	default:
		w.Header().Set("Content-Type", "application/json")
	}
	_, _ = io.WriteString(w, resp)
}

func (f *fakeAnthropic) reqs() []map[string]any {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]map[string]any(nil), f.requests...)
}

func msgJSON(stop string, content string, in, out int) string {
	return fmt.Sprintf(`{"id":"m","type":"message","role":"assistant","model":"test-model","stop_reason":%q,
"content":[%s],"usage":{"input_tokens":%d,"output_tokens":%d,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}`, stop, content, in, out)
}

func textResp(text string, in, out int) string {
	return msgJSON("end_turn", fmt.Sprintf(`{"type":"text","text":%q}`, text), in, out)
}

func toolResp(name, input string) string {
	return msgJSON("tool_use", fmt.Sprintf(`{"type":"tool_use","id":"tu_%s","name":%q,"input":%s}`, name, name, input), 1, 1)
}

// sse streams text deltas; tool, when set, appends a tool_use block and stops for tools.
func sse(deltas []string, tool string, in, out int) string {
	var b strings.Builder
	ev := func(name, data string) { fmt.Fprintf(&b, "event: %s\ndata: %s\n\n", name, data) }
	ev("message_start", fmt.Sprintf(`{"type":"message_start","message":{"id":"m","type":"message","role":"assistant","model":"test-model","content":[],"stop_reason":null,"usage":{"input_tokens":%d,"output_tokens":0}}}`, in))
	ev("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`)
	for _, d := range deltas {
		ev("content_block_delta", fmt.Sprintf(`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":%q}}`, d))
	}
	ev("content_block_stop", `{"type":"content_block_stop","index":0}`)
	stop := "end_turn"
	if tool != "" {
		stop = "tool_use"
		ev("content_block_start", fmt.Sprintf(`{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"tu_1","name":%q,"input":{}}}`, tool))
		ev("content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"city\":\"Paris\"}"}}`)
		ev("content_block_stop", `{"type":"content_block_stop","index":1}`)
	}
	ev("message_delta", fmt.Sprintf(`{"type":"message_delta","delta":{"stop_reason":%q},"usage":{"output_tokens":%d}}`, stop, out))
	ev("message_stop", `{"type":"message_stop"}`)
	return b.String()
}

type trackCall struct {
	Event  string
	Metric float64
	Data   map[string]any
}

// fakeLD serves variations by key and records track calls.
type fakeLD struct {
	mu         sync.Mutex
	variations map[string]any
	tracks     []trackCall
}

func (f *fakeLD) JSONVariation(key string, _ ldcontext.Context, def ldvalue.Value) (ldvalue.Value, error) {
	v, ok := f.variations[key]
	if !ok {
		return def, nil
	}
	b, _ := json.Marshal(v)
	return ldvalue.Parse(b), nil
}

func (f *fakeLD) TrackMetric(event string, _ ldcontext.Context, m float64, data ldvalue.Value) error {
	var d map[string]any
	_ = json.Unmarshal([]byte(data.JSONString()), &d)
	f.mu.Lock()
	defer f.mu.Unlock()
	f.tracks = append(f.tracks, trackCall{event, m, d})
	return nil
}
func (f *fakeLD) Flush()       {}
func (f *fakeLD) Close() error { return nil }

func (f *fakeLD) events(name string) []trackCall {
	f.mu.Lock()
	defer f.mu.Unlock()
	var out []trackCall
	for _, t := range f.tracks {
		if t.Event == name {
			out = append(out, t)
		}
	}
	return out
}

func (f *fakeLD) helpers() []string {
	var out []string
	for _, t := range f.events("$ld:ai:sdk:usage") {
		out = append(out, t.Data["helper"].(string))
	}
	return out
}

type env struct {
	ld   *fakeLD
	api  *fakeAnthropic
	exp  *tracetest.InMemoryExporter
	opts Options
}

// setup installs a fresh fake LD client, an httptest Anthropic server and an
// in-memory span exporter (with the conversation-id processor, as InitClient
// registers it).
func setup(t *testing.T, variations map[string]any, responses ...string) *env {
	t.Helper()
	_ = aiserver.Shutdown(bg)
	ld := &fakeLD{variations: variations}
	if _, err := aiserver.InitWithClient(bg, ld, aiserver.InitOptions{OTLPEndpoint: "http://127.0.0.1:1/v1/traces"}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = aiserver.Shutdown(bg) })
	exp := tracetest.NewInMemoryExporter()
	otel.SetTracerProvider(sdktrace.NewTracerProvider( // replaces the one InitWithClient installed
		sdktrace.WithSpanProcessor(aiserver.ConversationIDSpanProcessor{}), sdktrace.WithSyncer(exp)))
	api := &fakeAnthropic{responses: responses}
	srv := httptest.NewServer(api)
	t.Cleanup(srv.Close)
	c := anthropic.NewClient(option.WithBaseURL(srv.URL), option.WithAPIKey("test"), option.WithMaxRetries(0))
	return &env{ld: ld, api: api, exp: exp, opts: Options{Client: &c}}
}

func (e *env) handler() *aiserver.ProviderHandler { return newHandler(e.opts) }

func (e *env) spans(name string) []sdktrace.ReadOnlySpan {
	var out []sdktrace.ReadOnlySpan
	for _, s := range e.exp.GetSpans().Snapshots() {
		if s.Name() == name {
			out = append(out, s)
		}
	}
	return out
}

func attrs(s sdktrace.ReadOnlySpan) map[string]string {
	m := map[string]string{}
	for _, a := range s.Attributes() {
		m[string(a.Key)] = a.Value.Emit()
	}
	return m
}

func cfg(extra func(*aiserver.AiConfig)) aiserver.AiConfig {
	c := aiserver.AiConfig{Model: aiserver.Model{Name: "test-model"}}
	c.Provider.Name = "Anthropic"
	if extra != nil {
		extra(&c)
	}
	return c
}

// agentVariation is an enabled agent-mode Anthropic flag variation.
func agentVariation(instructions string, extra map[string]any) map[string]any {
	v := map[string]any{
		"model": map[string]any{"name": "test-model"}, "provider": map[string]any{"name": "Anthropic"},
		"instructions": instructions,
		"_ldMeta":      map[string]any{"enabled": true, "variationKey": "v1", "version": 2, "mode": "agent"},
	}
	for k, x := range extra {
		v[k] = x
	}
	return v
}

func system(req map[string]any) string {
	s, _ := req["system"].([]any)
	if len(s) == 0 {
		return ""
	}
	return s[0].(map[string]any)["text"].(string)
}

func messages(req map[string]any) []map[string]any {
	var out []map[string]any
	for _, m := range req["messages"].([]any) {
		out = append(out, m.(map[string]any))
	}
	return out
}

func roles(req map[string]any) []string {
	var out []string
	for _, m := range messages(req) {
		out = append(out, m["role"].(string))
	}
	return out
}

func firstText(m map[string]any) string {
	for _, b := range m["content"].([]any) {
		if blk := b.(map[string]any); blk["type"] == "text" {
			return blk["text"].(string)
		}
	}
	return ""
}

func toolNames(req map[string]any) []string {
	var out []string
	tools, _ := req["tools"].([]any)
	for _, t := range tools {
		out = append(out, t.(map[string]any)["name"].(string))
	}
	return out
}
