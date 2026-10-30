package evaluations

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/launchdarkly/go-sdk-common/v3/ldvalue"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/sdkapi"
)

// recReq is one request the fake transport saw.
type recReq struct {
	Method string
	URL    string // full URL as sent
	Path   string // escaped path after /api/v2/, no query
	Query  string
	Header http.Header
	Body   any // decoded JSON, nil when absent
	Raw    []byte
}

// resp is one canned answer. err simulates a transport failure.
type resp struct {
	status int
	body   any // string is sent verbatim; anything else is JSON-encoded
	header http.Header
	err    error
}

// fakeEnv is the recording management API transport plus the stub SDK
// client, sharing one ordered log so cross-boundary ordering can be asserted.
type fakeEnv struct {
	t      *testing.T
	mu     sync.Mutex
	log    []string
	reqs   []recReq
	routes map[string][]resp // "METHOD path" → queue; the last answer repeats
	ld     *fakeLD
	// judges maps a judge key to the variation flag delivery serves; a
	// value of type error makes resolution fail.
	judges      map[string]any
	judgeCalls  []judgeCall
	initCalls   []aiserver.InitOptions
	logs        *bytes.Buffer
	byoc        bool // getClient returns ld
	initErr     error
	project     string
	rows        int
	criteriaOff bool
}

type judgeCall struct {
	Key string
	Ctx ldcontext.Context
}

type trackCall struct {
	Event  string
	Ctx    ldcontext.Context
	Metric float64
	Data   map[string]any
}

type fakeLD struct {
	env        *fakeEnv
	tracks     []trackCall
	flushes    int
	onFlush    func()
	trackErr   func(event string, data map[string]any) error
	variations int
}

func (f *fakeLD) JSONVariation(string, ldcontext.Context, ldvalue.Value) (ldvalue.Value, error) {
	f.env.mu.Lock()
	f.variations++
	f.env.mu.Unlock()
	panic("variation must never gate emission")
}

func (f *fakeLD) TrackMetric(event string, c ldcontext.Context, m float64, data ldvalue.Value) error {
	var d map[string]any
	_ = json.Unmarshal([]byte(data.JSONString()), &d)
	if f.trackErr != nil {
		if err := f.trackErr(event, d); err != nil {
			return err
		}
	}
	f.env.mu.Lock()
	defer f.env.mu.Unlock()
	f.tracks = append(f.tracks, trackCall{event, c, m, d})
	f.env.log = append(f.env.log, "track "+event)
	return nil
}

func (f *fakeLD) Flush() {
	f.env.mu.Lock()
	f.flushes++
	f.env.log = append(f.env.log, "flush")
	cb := f.onFlush
	f.env.mu.Unlock()
	if cb != nil {
		cb()
	}
}

func (f *fakeLD) Close() error { return nil }

func (f *fakeLD) events(name string) []trackCall {
	f.env.mu.Lock()
	defer f.env.mu.Unlock()
	var out []trackCall
	for _, t := range f.tracks {
		if t.Event == name {
			out = append(out, t)
		}
	}
	return out
}

// newEnv installs the seams and a happy-path backend: project "proj",
// hosted dataset "ds" with rows rows, evaluation "eval-1" v3, run "run-1"
// whose summary is terminal with every row passed.
func newEnv(t *testing.T, rows int) *fakeEnv {
	t.Helper()
	e := &fakeEnv{t: t, routes: map[string][]resp{}, judges: map[string]any{}, logs: &bytes.Buffer{}, byoc: true, project: "proj", rows: rows}
	e.ld = &fakeLD{env: e}
	oldGet, oldInit, oldJudge := getClient, initClient, resolveJudgeConfig
	oldInterval, oldTimeout := defaultPollInterval, defaultPollTimeout
	t.Cleanup(func() {
		getClient, initClient, resolveJudgeConfig = oldGet, oldInit, oldJudge
		defaultPollInterval, defaultPollTimeout = oldInterval, oldTimeout
	})
	getClient = func() (aiserver.LDClient, error) {
		if e.byoc {
			return e.ld, nil
		}
		return nil, errors.New("client not initialized")
	}
	initClient = func(_ context.Context, o aiserver.InitOptions) (aiserver.LDClient, error) {
		e.mu.Lock()
		e.initCalls = append(e.initCalls, o)
		e.mu.Unlock()
		if e.initErr != nil {
			return nil, e.initErr
		}
		return e.ld, nil
	}
	resolveJudgeConfig = func(_ context.Context, key string, c ldcontext.Context) (map[string]any, error) {
		e.mu.Lock()
		e.judgeCalls = append(e.judgeCalls, judgeCall{key, c})
		e.log = append(e.log, "judge "+key)
		v, ok := e.judges[key]
		e.mu.Unlock()
		switch t := v.(type) {
		case error:
			return nil, t
		case map[string]any:
			return roundTrip(t).(map[string]any), nil // flag delivery serves decoded JSON
		}
		if !ok {
			return nil, fmt.Errorf("flag %q not found", key)
		}
		return nil, nil
	}
	defaultPollInterval = time.Millisecond
	e.route("GET", "projects/proj/datasets/ds", resp{body: map[string]any{"id": "ds-id", "key": "ds"}})
	items := make([]any, rows)
	for i := range items {
		items[i] = map[string]any{"rowIndex": i, "input": fmt.Sprintf("q%d", i), "expectedOutput": fmt.Sprintf("a%d", i)}
	}
	e.route("GET", "projects/proj/datasets/ds/rows", resp{body: map[string]any{"items": items, "totalCount": rows}})
	e.route("POST", "projects/proj/evaluations", resp{status: 201, body: map[string]any{"id": "eval-1", "version": 3}})
	e.route("POST", "projects/proj/evaluations/eval-1/runs", resp{status: 201, body: map[string]any{"id": "run-1", "evaluationId": "eval-1", "state": "PENDING"}})
	e.route("POST", "projects/proj/evaluations/eval-1/runs/run-1/dataset-rows", resp{body: map[string]any{}})
	e.route("GET", "projects/proj/evaluations/eval-1/runs/run-1/summary", resp{body: map[string]any{"statusCounts": map[string]any{"total": rows, "passed": rows}}})
	return e
}

// route replaces the answers for "METHOD path".
func (e *fakeEnv) route(method, path string, answers ...resp) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.routes[method+" "+path] = answers
}

func (e *fakeEnv) RoundTrip(req *http.Request) (*http.Response, error) {
	var raw []byte
	if req.Body != nil {
		raw, _ = io.ReadAll(req.Body)
	}
	var body any
	if len(raw) > 0 {
		_ = json.Unmarshal(raw, &body)
	}
	path := strings.TrimPrefix(req.URL.EscapedPath(), "/api/v2/")
	e.mu.Lock()
	e.reqs = append(e.reqs, recReq{req.Method, req.URL.String(), path, req.URL.RawQuery, req.Header.Clone(), body, raw})
	e.log = append(e.log, req.Method+" "+path)
	key := req.Method + " " + path
	queue := e.routes[key]
	var r resp
	if len(queue) == 0 {
		r = resp{status: 404, body: "no route for " + key}
	} else {
		r = queue[0]
		if len(queue) > 1 {
			e.routes[key] = queue[1:]
		}
	}
	e.mu.Unlock()
	if r.err != nil {
		return nil, r.err
	}
	var b []byte
	switch v := r.body.(type) {
	case nil:
	case string:
		b = []byte(v)
	default:
		b, _ = json.Marshal(v)
	}
	status := r.status
	if status == 0 {
		status = 200
	}
	h := r.header
	if h == nil {
		h = http.Header{}
	}
	return &http.Response{StatusCode: status, Header: h, Body: io.NopCloser(bytes.NewReader(b)), Request: req}, nil
}

// module builds a Module over the fake transport with instant sleeps.
func (e *fakeEnv) module() *Module {
	e.t.Helper()
	e.t.Setenv("LD_API_TOKEN", "api-token")
	m, err := Init(Options{HTTPClient: &http.Client{Transport: e}, Logger: slog.New(slog.NewTextHandler(e.logs, nil))})
	if err != nil {
		e.t.Fatalf("Init: %v", err)
	}
	m.api.sleep = func(ctx context.Context, d time.Duration) error { return ctx.Err() }
	return m
}

func (e *fakeEnv) requests() []recReq {
	e.mu.Lock()
	defer e.mu.Unlock()
	return slices.Clone(e.reqs)
}

// sequence is "METHOD path[?query]" per request.
func (e *fakeEnv) sequence() []string {
	var out []string
	for _, r := range e.requests() {
		s := r.Method + " " + r.Path
		if r.Query != "" {
			s += "?" + r.Query
		}
		out = append(out, s)
	}
	return out
}

func (e *fakeEnv) logSnapshot() []string {
	e.mu.Lock()
	defer e.mu.Unlock()
	return slices.Clone(e.log)
}

func (e *fakeEnv) body(method, path string) any {
	for _, r := range e.requests() {
		if r.Method == method && r.Path == path {
			return r.Body
		}
	}
	e.t.Fatalf("no %s %s request in %v", method, path, e.sequence())
	return nil
}

func (e *fakeEnv) count(substr string) int {
	n := 0
	for _, s := range e.sequence() {
		if strings.Contains(s, substr) {
			n++
		}
	}
	return n
}

func (e *fakeEnv) assertNoRequests() {
	e.t.Helper()
	if seq := e.sequence(); len(seq) != 0 {
		e.t.Fatalf("expected zero requests, got %v", seq)
	}
}

// handler is a stub ProviderHandler.
func handler(provider string, mode aiserver.Mode, fn func(ctx context.Context, req aiserver.HandlerRequest) (aiserver.HandlerResult, error)) *aiserver.ProviderHandler {
	if fn == nil {
		fn = func(_ context.Context, req aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
			return aiserver.HandlerResult{Output: "out:" + req.UserInput}, nil
		}
	}
	return sdkapi.CreateHandler(aiserver.ProvidesFor{Provider: provider, Mode: mode}, fn, nil, aiserver.CaptureDefault)
}

// recorder captures every call a stub handler gets.
type recorder struct {
	mu    sync.Mutex
	calls []aiserver.HandlerRequest
}

func (r *recorder) wrap(fn func(ctx context.Context, req aiserver.HandlerRequest) (aiserver.HandlerResult, error)) func(ctx context.Context, req aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
	return func(ctx context.Context, req aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
		r.mu.Lock()
		r.calls = append(r.calls, req)
		r.mu.Unlock()
		if fn == nil {
			return aiserver.HandlerResult{Output: "out:" + req.UserInput}, nil
		}
		return fn(ctx, req)
	}
}

func (r *recorder) n() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.calls)
}

func (r *recorder) all() []aiserver.HandlerRequest {
	r.mu.Lock()
	defer r.mu.Unlock()
	return slices.Clone(r.calls)
}

func gen() *GenerationConfig {
	return &GenerationConfig{Provider: "OpenAI", Model: "gpt-4o", Mode: ModeCompletion}
}

// baseArgs is a valid hosted, generation-source run over one OpenAI
// messages handler.
func baseArgs(h *aiserver.ProviderHandler) RunArgs {
	return RunArgs{ProjectKey: "proj", Key: "eval-key", Dataset: DatasetKey("ds"), Handler: h, Generation: gen(), PollTimeout: time.Second}
}

// judgeVariation is what flag delivery serves for a judge.
func judgeVariation(provider, mode string, extra map[string]any) map[string]any {
	v := map[string]any{
		"model":    map[string]any{"name": "judge-model"},
		"provider": map[string]any{"name": provider},
		"messages": []any{map[string]any{"role": "system", "content": "Grade {{message_history}}"}},
		"_ldMeta":  map[string]any{"enabled": true, "variationKey": "jv", "version": 7, "mode": mode},
	}
	for k, val := range extra {
		v[k] = val
	}
	return v
}

func verdict(score any, reasoning string) func(context.Context, aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
	return func(context.Context, aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
		b, _ := json.Marshal(map[string]any{"score": score, "reasoning": reasoning})
		return aiserver.HandlerResult{Output: string(b)}, nil
	}
}

func mustJudge(t *testing.T, key string, opts *JudgeOptions) *Judge {
	t.Helper()
	j, err := NewJudge(key, opts)
	if err != nil {
		t.Fatal(err)
	}
	return j
}

func mustScorer(t *testing.T, name string, fn ScorerFunc, opts *ScorerOptions) *Scorer {
	t.Helper()
	s, err := NewScorer(name, fn, opts)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func constScorer(v float64) ScorerFunc {
	return func(context.Context, DatasetRow, any) (float64, error) { return v, nil }
}

func ptr[T any](v T) *T { return &v }

func mustRun(t *testing.T, m *Module, args RunArgs) *EvalRunResult {
	t.Helper()
	res, err := m.Run(context.Background(), args)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	return res
}

func runErr(t *testing.T, m *Module, args RunArgs, sentinel error, contains ...string) error {
	t.Helper()
	_, err := m.Run(context.Background(), args)
	if err == nil {
		t.Fatalf("Run: expected an error")
	}
	if sentinel != nil && !errors.Is(err, sentinel) {
		t.Fatalf("Run error %q does not wrap %v", err, sentinel)
	}
	for _, s := range contains {
		if !strings.Contains(err.Error(), s) {
			t.Fatalf("Run error %q does not contain %q", err, s)
		}
	}
	return err
}

func toJSON(v any) string {
	b, _ := json.Marshal(v)
	return string(b)
}

// roundTrip normalizes a Go value to its decoded-JSON form for comparison.
func roundTrip(v any) any {
	var out any
	_ = json.Unmarshal([]byte(toJSON(v)), &out)
	return out
}
