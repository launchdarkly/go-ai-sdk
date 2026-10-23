//go:build unix

package aiserver

import (
	"context"
	"errors"
	"fmt"
	"go/parser"
	"go/token"
	"io"
	"log/slog"
	"net/http"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

// fakeResp is one scripted answer of the in-process requester double.
type fakeResp struct {
	status  int
	header  http.Header
	body    string
	hold    bool  // after body, block until the request is cancelled
	readErr error // after body, fail the read (a stream dying mid-read)
	err     error // transport error instead of a response
	gate    chan struct{}
	then    string // sent once gate is closed, on the same connection
}

// fakeTransport is an http.RoundTripper: no socket is ever opened. An
// exhausted script holds the request open (never a standing 304).
type fakeTransport struct {
	mu     sync.Mutex
	reqs   []*http.Request
	script []fakeResp
}

func (f *fakeTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	f.mu.Lock()
	f.reqs = append(f.reqs, req)
	r := fakeResp{hold: true}
	if len(f.script) > 0 {
		r, f.script = f.script[0], f.script[1:]
	}
	f.mu.Unlock()
	if r.err != nil {
		return nil, r.err
	}
	if r.status == 0 {
		r.status = 200
	}
	if r.header == nil {
		r.header = http.Header{}
	}
	return &http.Response{StatusCode: r.status, Header: r.header, Request: req,
		Body: &ctxBody{ctx: req.Context(), data: []byte(r.body), hold: r.hold, err: r.readErr, gate: r.gate, then: []byte(r.then)}}, nil
}

func (f *fakeTransport) push(r ...fakeResp) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.script = append(f.script, r...)
}

func (f *fakeTransport) requests() []*http.Request {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]*http.Request(nil), f.reqs...)
}

type ctxBody struct {
	ctx  context.Context
	data []byte
	hold bool
	err  error
	gate chan struct{}
	then []byte
}

func (b *ctxBody) Read(p []byte) (int, error) {
	if len(b.data) == 0 && b.gate != nil {
		select {
		case <-b.gate:
			b.data, b.then, b.gate = b.then, nil, nil
		case <-b.ctx.Done():
			return 0, b.ctx.Err()
		}
	}
	if len(b.data) > 0 {
		n := copy(p, b.data)
		b.data = b.data[n:]
		return n, nil
	}
	if b.err != nil {
		return 0, b.err
	}
	if b.hold {
		<-b.ctx.Done()
		return 0, b.ctx.Err()
	}
	return 0, io.EOF
}

func (b *ctxBody) Close() error { return nil }

// sse renders events as a stream body; an empty data string means no data line.
func sse(events ...[2]string) string {
	var sb strings.Builder
	for _, e := range events {
		sb.WriteString("event: " + e[0] + "\n")
		if e[1] != "" {
			sb.WriteString("data: " + e[1] + "\n")
		}
		sb.WriteString("\n")
	}
	return sb.String()
}

func ev(name string, data []byte) [2]string { return [2]string{name, string(data)} }

func commitEvents(id, selector string, puts ...[]byte) [][2]string {
	out := [][2]string{ev("server-intent", intent("xfer-full", id))}
	for _, p := range puts {
		out = append(out, ev("put-object", p))
	}
	return append(out, ev("payload-transferred", transferred(selector)))
}

func pollBody(events ...[2]string) string {
	var parts []string
	for _, e := range events {
		parts = append(parts, fmt.Sprintf(`{"event":%q,"data":%s}`, e[0], e[1]))
	}
	return `{"events":[` + strings.Join(parts, ",") + `]}`
}

func newTestStore(t *testing.T, mode FDv2Mode, ft *fakeTransport, mutate func(*FDv2SkillStoreOptions)) *FDv2SkillStore {
	t.Helper()
	opts := &FDv2SkillStoreOptions{Mode: mode, InitialBackoff: time.Millisecond, MaxBackoff: 5 * time.Millisecond,
		PollInterval: time.Hour, ReadTimeout: time.Minute}
	if mutate != nil {
		mutate(opts)
	}
	opts.transport = ft
	s, err := NewFDv2SkillStore("sdk-test-key", opts)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(s.Close)
	return s
}

func eventually(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(2 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

func start(t *testing.T, s *FDv2SkillStore) {
	t.Helper()
	if _, err := s.Start(); err != nil {
		t.Fatal(err)
	}
}

func TestFDv2KindsOnEveryRequest(t *testing.T) {
	_, _ = setup(t)
	ft := &fakeTransport{}
	for i := range 3 {
		ft.push(fakeResp{body: sse(commitEvents("p", fmt.Sprintf("(p:p:%d)", i+1), put("a:1", "a"))...)})
	}
	s := newTestStore(t, FDv2ModeStream, ft, nil)
	start(t, s)
	eventually(t, "four requests", func() bool { return len(ft.requests()) >= 4 })
	for i, req := range ft.requests() {
		q := req.URL.Query()
		if !slices.Equal(q["kinds"], []string{"agent-skill"}) {
			t.Fatalf("request %d kinds=%v", i, q["kinds"])
		}
		for k := range q {
			if k != "kinds" && k != "basis" {
				t.Fatalf("request %d carries %q", i, k)
			}
		}
		if (q.Get("basis") != "") != (i > 0) {
			t.Fatalf("request %d basis=%q", i, q.Get("basis"))
		}
		if req.Header.Get("Authorization") != "sdk-test-key" || req.URL.Host != "stream.launchdarkly.com" || req.URL.Path != "/sdk/stream" {
			t.Fatalf("request %d: %v %v", i, req.URL, req.Header)
		}
	}
	ft2 := &fakeTransport{}
	p := newTestStore(t, FDv2ModePoll, ft2, nil)
	start(t, p)
	eventually(t, "a poll", func() bool { return len(ft2.requests()) == 1 })
	if r := ft2.requests()[0]; r.URL.Query().Get("kinds") != "agent-skill" || r.URL.Host != "sdk.launchdarkly.com" || r.URL.Path != "/sdk/poll" {
		t.Fatalf("poll request %v", r.URL)
	}
}

func TestFDv2Endpoints(t *testing.T) {
	_, _ = setup(t)
	s, _ := NewFDv2SkillStore("sdk-x", nil)
	if s.baseURI != "https://sdk.launchdarkly.com" || s.streamURI != "https://stream.launchdarkly.com" {
		t.Fatalf("defaults %s %s", s.baseURI, s.streamURI)
	}
	s, _ = NewFDv2SkillStore("sdk-x", &FDv2SkillStoreOptions{BaseURI: "https://relay.example/"})
	if s.baseURI != "https://relay.example" || s.streamURI != "https://relay.example" {
		t.Fatalf("base alone must move both: %s %s", s.baseURI, s.streamURI)
	}
	s, _ = NewFDv2SkillStore("sdk-x", &FDv2SkillStoreOptions{BaseURI: "https://a.example", StreamURI: "https://b.example"})
	if s.baseURI != "https://a.example" || s.streamURI != "https://b.example" {
		t.Fatal("separate overrides")
	}
	for _, bad := range []FDv2SkillStoreOptions{{BaseURI: "http://sdk.example"}, {StreamURI: "http://localhost"}, {BaseURI: "https://"}, {BaseURI: "ftp://x"}} {
		if _, err := NewFDv2SkillStore("sdk-x", &bad); !errors.Is(err, ErrInvalidArgument) {
			t.Fatalf("%+v accepted", bad)
		}
	}
}

func TestFDv2Credentials(t *testing.T) {
	_, logs := setup(t)
	for _, key := range []string{"mob-123", "0123456789abcdef01234567"} {
		_, err := NewFDv2SkillStore(key, nil)
		if !errors.Is(err, ErrInvalidArgument) || !strings.Contains(err.Error(), "server-side") {
			t.Fatalf("%s: %v", key, err)
		}
	}
	if _, err := NewFDv2SkillStore("  ", nil); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("empty: %v", err)
	}
	if _, err := NewFDv2SkillStore("relay-key", nil); err != nil {
		t.Fatalf("unprefixed: %v", err)
	}
	if !logs.contains(slog.LevelWarn, "does not look like") {
		t.Fatal("unprefixed key must warn")
	}
}

func TestFDv2OptionValidation(t *testing.T) {
	_, _ = setup(t)
	bad := []FDv2SkillStoreOptions{
		{Mode: "push"}, {PollInterval: -1}, {ReadTimeout: -1}, {InitialBackoff: -1}, {MaxBackoff: -1},
		{InitialBackoff: 2 * time.Second, MaxBackoff: time.Second},
	}
	for _, o := range bad {
		if _, err := NewFDv2SkillStore("sdk-x", &o); !errors.Is(err, ErrInvalidArgument) {
			t.Fatalf("%+v accepted", o)
		}
	}
	if _, err := NewFDv2SkillStore("sdk-x", &FDv2SkillStoreOptions{InitialBackoff: time.Second, MaxBackoff: time.Second}); err != nil {
		t.Fatalf("initial == max: %v", err)
	}
	stream, _ := NewFDv2SkillStore("sdk-x", nil)
	poll, _ := NewFDv2SkillStore("sdk-x", &FDv2SkillStoreOptions{Mode: FDv2ModePoll})
	if stream.mode != FDv2ModeStream || stream.readTimeout != 300*time.Second || poll.readTimeout != 10*time.Second || poll.pollInterval != 30*time.Second {
		t.Fatalf("defaults stream=%v/%v poll=%v/%v", stream.mode, stream.readTimeout, poll.readTimeout, poll.pollInterval)
	}
	if stream.initialBackoff != time.Second || stream.maxBackoff != 30*time.Second {
		t.Fatal("backoff defaults")
	}
}

func TestFDv2WaitAndInitialized(t *testing.T) {
	_, _ = setup(t)
	ft := &fakeTransport{}
	s := newTestStore(t, FDv2ModeStream, ft, nil)
	if ok, _ := s.IsInitialized(); ok {
		t.Fatal("initialized before any payload")
	}
	if s.WaitForSkills(20 * time.Millisecond) {
		t.Fatal("wait true with nothing delivered")
	}
	ft.push(fakeResp{body: sse(commitEvents("p", "(p:p:1)", put("a:1", "a"))...), hold: true})
	start(t, s)
	if !s.WaitForSkills(5 * time.Second) {
		t.Fatal("wait false after a commit")
	}
	if ok, _ := s.IsInitialized(); !ok {
		t.Fatal("not initialized after a commit")
	}
	if got, err := s.GetObject("skill", "a", 0); got == nil || err != nil {
		t.Fatalf("held %v %v", got, err)
	}
	s.Close()
	if ok, _ := s.IsInitialized(); !ok {
		t.Fatal("IsInitialized must stay true after Close")
	}
	if got, _ := s.GetObject("skill", "a", 0); got == nil {
		t.Fatal("a closed store must keep serving what it received")
	}
}

func TestFDv2WaitReturnsEarlyWhenDeliveryEnds(t *testing.T) {
	_, _ = setup(t)
	s := newTestStore(t, FDv2ModeStream, &fakeTransport{}, nil)
	go func() { time.Sleep(20 * time.Millisecond); s.Close() }()
	began := time.Now()
	if s.WaitForSkills(10*time.Second) || time.Since(began) > 2*time.Second {
		t.Fatal("wait did not return false early on Close")
	}
}

func Test422IsFatalAndRecoverableByStart(t *testing.T) {
	rec, _ := setup(t)
	ft := &fakeTransport{}
	ft.push(fakeResp{status: 503}, fakeResp{status: 503}, fakeResp{status: 422})
	s := newTestStore(t, FDv2ModeStream, ft, nil)
	start(t, s)
	began := time.Now()
	if s.WaitForSkills(10 * time.Second) {
		t.Fatal("wait true after a 422")
	}
	if time.Since(began) > 3*time.Second {
		t.Fatal("wait did not return early on a fatal 422")
	}
	d := s.Diagnostics()
	msg := s.Failed()
	if msg == "" || d.LastError == "" || d.ConnectionFailures != 2 {
		t.Fatalf("failed=%q diag=%+v", msg, d)
	}
	if !strings.Contains(msg, "not enabled for this LaunchDarkly account") || !strings.Contains(msg, "view-scoped") {
		t.Fatalf("the 422 message must name both causes: %q", msg)
	}
	if strings.Contains(strings.ToLower(msg), "no skills") || strings.Contains(msg, "created later") {
		t.Fatalf("the 422 message must not blame skill existence: %q", msg)
	}
	if n := len(ft.requests()); n != 3 {
		t.Fatalf("requests %d: a fatal status must not be retried", n)
	}
	if ok, _ := s.IsInitialized(); ok {
		t.Fatal("initialized after a 422")
	}

	if len(rec.all()) != 0 {
		t.Fatalf("the transport recorded telemetry: %v", rec.all())
	}
	// Composed with WriteAllSkills: nothing on disk changes.
	root := t.TempDir()
	mustWriteAny(t, root)
	SetSkillStore(s)
	report, _ := WriteAllSkills(root, nil)
	if report.OK() || slices.Contains(actionKinds(report), "a=removed") {
		t.Fatalf("a 422 store pruned: %v", actionKinds(report))
	}

	// Recovery: the gate opens, Start runs delivery afresh.
	ft.push(fakeResp{body: sse(commitEvents("p", "(p:p:1)", put("a:1", "a"))...), hold: true})
	start(t, s)
	if s.Failed() != "" || s.Diagnostics().ConnectionFailures != 0 {
		t.Fatalf("Start did not reset: failed=%q diag=%+v", s.Failed(), s.Diagnostics())
	}
	if !s.WaitForSkills(5 * time.Second) {
		t.Fatal("restarted store did not deliver")
	}
	start(t, s) // a no-op while running
	time.Sleep(20 * time.Millisecond)
	if n := len(ft.requests()); n != 4 {
		t.Fatalf("Start while running opened another connection: %d requests", n)
	}
	if n := len(rec.named(signalMaterialized)); n != 1 || len(rec.all()) != 1 {
		t.Fatalf("the transport recorded telemetry: %v", rec.all())
	}
}

func mustWriteAny(t *testing.T, root string) {
	t.Helper()
	mustWrite(t, []Skill{skillOf("a", 1, "a")}, root, nil)
}

func TestFDv2WaitAfterRestartWaitsForTheNewRun(t *testing.T) {
	_, _ = setup(t)
	ft := &fakeTransport{}
	ft.push(fakeResp{status: 401})
	s := newTestStore(t, FDv2ModeStream, ft, nil)
	start(t, s)
	eventually(t, "give up", func() bool { return s.Failed() != "" })
	start(t, s) // the script is exhausted: the new run holds
	began := time.Now()
	if s.WaitForSkills(100*time.Millisecond) || time.Since(began) < 80*time.Millisecond {
		t.Fatal("a wait after a restart must wait for the new run")
	}
}

func TestFDv2FatalStatuses(t *testing.T) {
	for _, status := range []int{401, 403, 404, 405, 406, 414, 422, 501} {
		_, _ = setup(t)
		ft := &fakeTransport{}
		ft.push(fakeResp{status: status})
		s := newTestStore(t, FDv2ModeStream, ft, nil)
		start(t, s)
		eventually(t, fmt.Sprintf("give up on %d", status), func() bool { return s.Failed() != "" })
		time.Sleep(10 * time.Millisecond)
		if n := len(ft.requests()); n != 1 {
			t.Fatalf("%d retried: %d requests", status, n)
		}
		if status == 401 && !strings.Contains(s.Failed(), "SDK key") {
			t.Fatalf("401 message %q", s.Failed())
		}
	}
	_, _ = setup(t)
	ft := &fakeTransport{}
	ft.push(fakeResp{status: 503}, fakeResp{status: 503})
	s := newTestStore(t, FDv2ModeStream, ft, nil)
	start(t, s)
	eventually(t, "retries", func() bool { return len(ft.requests()) >= 3 })
	if s.Failed() != "" {
		t.Fatal("503 must be retried, not fatal")
	}
}

func TestFDv2ClassificationIsExhaustive(t *testing.T) {
	for status := 100; status < 600; status++ {
		if status >= 200 && status < 300 || status == 304 {
			continue
		}
		var fatal *fatalError
		var rec *recoverableError
		err := classifyStatus(status, http.Header{})
		if errors.As(err, &fatal) == errors.As(err, &rec) {
			t.Fatalf("%d is not exactly one class", status)
		}
		if status == 400 && !rec.stale {
			t.Fatal("400 must be the stale-state class")
		}
	}
}

func Test400RetriesOnceStatelessThenGivesUp(t *testing.T) {
	_, _ = setup(t)
	ft := &fakeTransport{}
	ft.push(fakeResp{body: sse(commitEvents("p", "(p:p:1)", put("a:1", "a"))...)}, fakeResp{status: 400}, fakeResp{status: 400})
	s := newTestStore(t, FDv2ModeStream, ft, nil)
	start(t, s)
	eventually(t, "give up", func() bool { return s.Failed() != "" })
	reqs := ft.requests()
	if len(reqs) != 3 || reqs[1].URL.Query().Get("basis") == "" || reqs[2].URL.Query().Get("basis") != "" || reqs[2].Header.Get("If-None-Match") != "" {
		t.Fatalf("400 retry shape: %d requests", len(reqs))
	}
	_, _ = setup(t)
	ft = &fakeTransport{}
	ft.push(fakeResp{status: 400})
	s = newTestStore(t, FDv2ModeStream, ft, nil)
	start(t, s)
	eventually(t, "give up", func() bool { return s.Failed() != "" })
	if len(ft.requests()) != 1 {
		t.Fatal("a 400 on a stateless request must not be retried")
	}
}

func TestFDv2ResponseTooLargeIsFatal(t *testing.T) {
	_, _ = setup(t)
	old := maxResponseBytes
	maxResponseBytes = 200
	t.Cleanup(func() { maxResponseBytes = old })
	ft := &fakeTransport{}
	ft.push(fakeResp{body: pollBody(ev("server-intent", intent("xfer-full", "p"))) + strings.Repeat(" ", 300)})
	s := newTestStore(t, FDv2ModePoll, ft, nil)
	start(t, s)
	eventually(t, "give up", func() bool { return s.Failed() != "" })
	if len(ft.requests()) != 1 || s.Diagnostics().ConnectionFailures != 0 || s.Diagnostics().LastError == "" {
		t.Fatalf("requests=%d diag=%+v", len(ft.requests()), s.Diagnostics())
	}
	// On a stream, after a commit: last known good still served.
	_, _ = setup(t)
	ft = &fakeTransport{}
	ft.push(fakeResp{body: sse(commitEvents("p", "(p:p:1)", put("a:1", "a"))...)})
	ft.push(fakeResp{body: sse(ev("put-object", put("b:1", strings.Repeat("x", 300))))})
	s = newTestStore(t, FDv2ModeStream, ft, nil)
	start(t, s)
	eventually(t, "give up", func() bool { return s.Failed() != "" })
	if got, _ := s.GetObject("skill", "a", 0); got == nil {
		t.Fatal("last known good lost")
	}
}

func TestFDv2Absences(t *testing.T) {
	var fields []string
	rt := reflect.TypeOf(StoreDiagnostics{})
	for i := range rt.NumField() {
		fields = append(fields, rt.Field(i).Name)
	}
	want := []string{"PayloadsTransferred", "SkillObjectsReceived", "ObjectsIgnored", "ObjectsRevoked", "PayloadsIgnored", "HashlessObjects", "ConnectionFailures", "LastError"}
	if !slices.Equal(fields, want) {
		t.Fatalf("StoreDiagnostics fields %v", fields)
	}
	if _, has := reflect.TypeOf(FDv2SkillStoreOptions{}).FieldByName("MaxConsecutiveFailures"); has {
		t.Fatal("MaxConsecutiveFailures must not exist")
	}
	fset := token.NewFileSet()
	pkgs, err := parser.ParseDir(fset, ".", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range pkgs["aiserver"].Files {
		for name := range f.Scope.Objects {
			if strings.Contains(name, "NoSkillPayload") || strings.Contains(name, "PayloadUnavailable") {
				t.Fatalf("%s must not exist", name)
			}
		}
	}
}

func TestBackoff(t *testing.T) {
	for attempt := 1; attempt <= 200; attempt++ {
		for range 20 {
			d := backoffDelay(attempt, time.Second, 30*time.Second)
			if d > 30*time.Second || d <= 0 {
				t.Fatalf("attempt %d: %v", attempt, d)
			}
		}
	}
	if d := backoffDelay(10000, time.Second, 30*time.Second); d <= 0 || d > 30*time.Second {
		t.Fatalf("attempt 10000: %v", d)
	}
	s, _ := NewFDv2SkillStore("sdk-x", &FDv2SkillStoreOptions{InitialBackoff: 2 * time.Second, MaxBackoff: 10 * time.Second})
	for header, want := range map[string]time.Duration{"1000": 10 * time.Second, "0": 2 * time.Second, "5": 5 * time.Second} {
		re := classifyStatus(503, http.Header{"Retry-After": {header}}).(*recoverableError)
		if got := s.retryDelay(re, 1); got != want {
			t.Fatalf("Retry-After %s: %v, want %v", header, got, want)
		}
	}
	re := classifyStatus(503, http.Header{"Retry-After": {"Wed, 21 Oct 2015 07:28:00 GMT"}}).(*recoverableError)
	if re.hasRetry {
		t.Fatal("the HTTP-date form must fall back to backoff")
	}
}

func TestConnectionFailuresCountOnlyConsecutive(t *testing.T) {
	_, _ = setup(t)
	ft := &fakeTransport{}
	ft.push(fakeResp{status: 503},
		fakeResp{body: pollBody(commitEvents("p", "(p:p:1)", put("a:1", "a"))...)},
		fakeResp{status: 503})
	s := newTestStore(t, FDv2ModePoll, ft, func(o *FDv2SkillStoreOptions) { o.PollInterval = time.Millisecond; o.ReadTimeout = time.Hour })
	start(t, s)
	eventually(t, "the 4th poll to park", func() bool { return len(ft.requests()) >= 4 })
	if n := s.Diagnostics().ConnectionFailures; n != 1 {
		t.Fatalf("fail, succeed, fail: %d failures", n)
	}
	s.mu.Lock()
	attempt := s.backoffAttempt
	s.mu.Unlock()
	if attempt != 1 {
		t.Fatalf("poll mode: a completed poll must reset the delay, attempt=%d", attempt)
	}

	// An announced transfer that drops every time is not health.
	_, _ = setup(t)
	ft = &fakeTransport{}
	for range 3 {
		ft.push(fakeResp{body: sse(ev("server-intent", intent("xfer-full", "p")))})
	}
	s = newTestStore(t, FDv2ModeStream, ft, nil)
	start(t, s)
	eventually(t, "three drops", func() bool { return s.Diagnostics().ConnectionFailures == 3 })
}

func TestGoodbyeRecycleAndCatastrophe(t *testing.T) {
	_, recycleLogs := setup(t)
	ft := &fakeTransport{}
	for range 3 {
		ft.push(fakeResp{body: sse(ev("server-intent", intent("none", "p")), ev("goodbye", []byte(`{"reason":"recycle"}`)))})
	}
	s := newTestStore(t, FDv2ModeStream, ft, nil)
	start(t, s)
	eventually(t, "4 requests", func() bool { return len(ft.requests()) >= 4 })
	if d := s.Diagnostics(); d.ConnectionFailures != 0 || d.LastError != "" {
		t.Fatalf("a recycle counted as a failure: %+v", d)
	}
	// A routine recycle is not news: nothing above debug.
	var sawDebug bool
	for _, e := range recycleLogs.all() {
		if e.level > slog.LevelDebug {
			t.Fatalf("a recycle logged at %v: %q", e.level, e.msg)
		}
		sawDebug = true
	}
	if !sawDebug {
		t.Fatal("no debug line was captured, so the level check was not exercised")
	}

	_, _ = setup(t)
	ft = &fakeTransport{}
	ft.push(fakeResp{body: sse(ev("goodbye", []byte(`{"reason":"early"}`)))})
	s = newTestStore(t, FDv2ModeStream, ft, nil)
	start(t, s)
	eventually(t, "a counted goodbye", func() bool { return s.Diagnostics().ConnectionFailures == 1 })

	_, logs := setup(t)
	ft = &fakeTransport{}
	events := commitEvents("p", "(p:p:1)", put("a:1", "a"))
	events = append(events, ev("goodbye", []byte(`{"reason":"meltdown","catastrophe":true,"silent":true}`)))
	ft.push(fakeResp{body: sse(events...)})
	s = newTestStore(t, FDv2ModeStream, ft, nil)
	start(t, s)
	eventually(t, "reconnect", func() bool { return len(ft.requests()) >= 2 })
	d := s.Diagnostics()
	if s.Failed() != "" || d.ConnectionFailures != 1 || !strings.Contains(d.LastError, "meltdown") {
		t.Fatalf("catastrophe: failed=%q %+v", s.Failed(), d)
	}
	if !logs.contains(slog.LevelError, "catastrophic") {
		t.Fatal("a catastrophic goodbye must log at ERROR even when silent")
	}
	if ft.requests()[1].URL.Query().Get("basis") != "(p:p:1)" {
		t.Fatal("reconnect did not carry the committed basis")
	}
}

func TestBackoffResetsOnlyAfterALongConnection(t *testing.T) {
	_, _ = setup(t)
	drop := fakeResp{body: sse(ev("server-intent", intent("none", "p")))}
	ft := &fakeTransport{}
	ft.push(drop, drop, drop)
	s := newTestStore(t, FDv2ModeStream, ft, nil)
	start(t, s)
	eventually(t, "4 requests", func() bool { return len(ft.requests()) >= 4 })
	s.mu.Lock()
	attempt := s.backoffAttempt
	s.mu.Unlock()
	if attempt != 3 || s.Diagnostics().ConnectionFailures > 1 {
		t.Fatalf("answer-and-drop must back off: attempt=%d failures=%d", attempt, s.Diagnostics().ConnectionFailures)
	}
	old := backoffResetInterval
	backoffResetInterval = 0
	t.Cleanup(func() { backoffResetInterval = old })
	_, _ = setup(t)
	ft = &fakeTransport{}
	ft.push(drop, drop, drop)
	s = newTestStore(t, FDv2ModeStream, ft, nil)
	start(t, s)
	eventually(t, "4 requests", func() bool { return len(ft.requests()) >= 4 })
	s.mu.Lock()
	attempt = s.backoffAttempt
	s.mu.Unlock()
	if attempt != 1 {
		t.Fatalf("a connection held past the threshold must retry at the first step: attempt=%d", attempt)
	}
}

func TestRecoverableFailuresRetryIndefinitely(t *testing.T) {
	_, _ = setup(t)
	ft := &fakeTransport{}
	ft.push(fakeResp{body: sse(commitEvents("p", "(p:p:1)", put("a:1", "a"))...)})
	for range 15 {
		ft.push(fakeResp{status: 503})
	}
	s := newTestStore(t, FDv2ModeStream, ft, nil)
	start(t, s)
	eventually(t, "16 failures", func() bool { return s.Diagnostics().ConnectionFailures >= 16 })
	if s.Failed() != "" {
		t.Fatal("gave up on recoverable failures")
	}
	if got, err := s.GetObject("skill", "a", 0); got == nil || err != nil {
		t.Fatal("a transport failure emptied the store")
	}
}

func TestStreamDyingMidReadIsRetried(t *testing.T) {
	_, _ = setup(t)
	ft := &fakeTransport{}
	ft.push(fakeResp{body: sse(ev("server-intent", intent("xfer-full", "p"))), readErr: errors.New("connection reset")})
	ft.push(fakeResp{body: sse(commitEvents("p", "(p:p:1)", put("a:1", "a"))...), hold: true})
	s := newTestStore(t, FDv2ModeStream, ft, nil)
	start(t, s)
	if !s.WaitForSkills(5 * time.Second) {
		t.Fatal("delivery did not resume")
	}
}

func TestCloseInterruptsAParkedRead(t *testing.T) {
	_, _ = setup(t)
	ft := &fakeTransport{}
	s := newTestStore(t, FDv2ModeStream, ft, nil)
	start(t, s)
	eventually(t, "a parked read", func() bool { return len(ft.requests()) == 1 })
	time.Sleep(20 * time.Millisecond)
	if len(ft.requests()) != 1 || s.Diagnostics().ConnectionFailures != 0 {
		t.Fatal("an idle stream did not stay alive")
	}
	began := time.Now()
	s.Close()
	if time.Since(began) > time.Second {
		t.Fatal("Close did not interrupt the read promptly")
	}
	s.Close() // idempotent
	if _, err := s.Start(); !errors.Is(err, ErrStoreClosed) {
		t.Fatalf("Start after Close: %v", err)
	}
}

func TestStreamReadTimeoutReconnects(t *testing.T) {
	_, _ = setup(t)
	ft := &fakeTransport{}
	s := newTestStore(t, FDv2ModeStream, ft, func(o *FDv2SkillStoreOptions) { o.ReadTimeout = 30 * time.Millisecond })
	start(t, s)
	eventually(t, "a reconnect after the read timeout", func() bool { return len(ft.requests()) >= 2 })
}

func TestStartChainsAndIsIdempotent(t *testing.T) {
	_, _ = setup(t)
	ft := &fakeTransport{}
	s := newTestStore(t, FDv2ModeStream, ft, nil)
	got, err := s.Start()
	if err != nil || got != s {
		t.Fatal("Start must return the store")
	}
	_, _ = s.Start()
	time.Sleep(20 * time.Millisecond)
	if n := len(ft.requests()); n != 1 {
		t.Fatalf("a second Start began a second delivery loop: %d requests", n)
	}
}

func TestKnownEventWithBadDataFailsTheConnection(t *testing.T) {
	for _, name := range []string{"server-intent", "put-object", "delete-object", "payload-transferred", "goodbye", "error"} {
		for _, body := range []string{"event: " + name + "\n\n", "event: " + name + "\ndata:\n\n", "event: " + name + "\ndata: {trunc\n\n"} {
			_, logs := setup(t)
			ft := &fakeTransport{}
			ft.push(fakeResp{body: sse(commitEvents("p", "(p:p:1)", put("a:1", "a"))...) + body})
			s := newTestStore(t, FDv2ModeStream, ft, nil)
			start(t, s)
			eventually(t, "a failed connection", func() bool { return s.Diagnostics().ConnectionFailures == 1 })
			if !logs.contains(slog.LevelWarn, name) {
				t.Fatalf("%s %q: no warning naming the event", name, body)
			}
			if got, _ := s.GetObject("skill", "a", 0); got == nil {
				t.Fatal("committed content lost")
			}
		}
	}
}

func TestKnownPollEventWithoutDataFailsThePoll(t *testing.T) {
	for name := range knownEvents {
		_, _ = setup(t)
		ft := &fakeTransport{}
		ft.push(fakeResp{body: pollBody(commitEvents("p", "(p:p:1)", put("a:1", "a"))...)},
			fakeResp{body: `{"events":[{"event":"` + name + `"}]}`})
		s := newTestStore(t, FDv2ModePoll, ft, func(o *FDv2SkillStoreOptions) { o.PollInterval = time.Millisecond })
		start(t, s)
		eventually(t, "a failed poll", func() bool { return s.Diagnostics().ConnectionFailures == 1 })
		if d := s.Diagnostics(); !strings.Contains(d.LastError, name) {
			t.Fatalf("%s: last error %q does not name the event", name, d.LastError)
		}
		if got, _ := s.GetObject("skill", "a", 0); got == nil {
			t.Fatalf("%s: committed content lost", name)
		}
	}
}

func TestTruncatedEventIsRetransmittedNotSkipped(t *testing.T) {
	_, _ = setup(t)
	ft := &fakeTransport{}
	ft.push(fakeResp{body: sse(commitEvents("p", "(p:p:1)", put("a:1", "a"), put("revoked:1", "r"))...)})
	ft.push(fakeResp{body: sse(
		ev("server-intent", intent("xfer-changes", "p")),
		ev("put-object", put("c:1", "c")),
		[2]string{"delete-object", `{"kind":"skill","key":"revoked:1"`},
		ev("payload-transferred", transferred("(p:p:2)")),
	)})
	ft.push(fakeResp{body: sse(
		ev("server-intent", intent("xfer-changes", "p")),
		ev("put-object", put("c:1", "c")),
		ev("delete-object", del("revoked:1")),
		ev("payload-transferred", transferred("(p:p:2)")),
	), hold: true})
	s := newTestStore(t, FDv2ModeStream, ft, nil)
	start(t, s)
	eventually(t, "the retransmission", func() bool { got, _ := s.GetObject("skill", "c", 0); return got != nil })
	reqs := ft.requests()
	if reqs[2].URL.Query().Get("basis") != "(p:p:1)" {
		t.Fatalf("the reconnect after a truncated event must carry the old basis: %q", reqs[2].URL.Query().Get("basis"))
	}
	if got, _ := s.GetObject("skill", "revoked", 0); got != nil {
		t.Fatal("the revocation was lost")
	}

	// xfer-full with an intact put and a truncated put: nothing commits.
	_, _ = setup(t)
	ft = &fakeTransport{}
	ft.push(fakeResp{body: sse(commitEvents("p", "(p:p:1)", put("a:1", "a"))...)})
	ft.push(fakeResp{body: sse(
		ev("server-intent", intent("xfer-full", "p")),
		ev("put-object", put("b:1", "b")),
		[2]string{"put-object", `{"kind":"skill","key":"a:1","object":{"content":"a`},
		ev("payload-transferred", transferred("(p:p:2)")),
	)})
	s = newTestStore(t, FDv2ModeStream, ft, nil)
	start(t, s)
	eventually(t, "third request", func() bool { return len(ft.requests()) >= 3 })
	if got, _ := s.GetObject("skill", "a", 0); got == nil {
		t.Fatal("a was revoked by a broken full transfer")
	}
	if got, _ := s.GetObject("skill", "b", 0); got != nil {
		t.Fatal("part of a broken transfer committed")
	}
	if ft.requests()[2].URL.Query().Get("basis") != "(p:p:1)" {
		t.Fatal("the basis advanced past a broken transfer")
	}
}

func TestConditionalPollingCannotManufactureAPayload(t *testing.T) {
	_, _ = setup(t)
	// A body under an unrecognised intent earns no etag.
	ft := &fakeTransport{}
	ft.push(fakeResp{header: http.Header{"Etag": {`"e1"`}}, body: pollBody(
		ev("server-intent", intent("xfer-sideways", "p")), ev("put-object", put("a:1", "a")), ev("payload-transferred", transferred("(p:p:1)")))},
		fakeResp{status: 304})
	s := newTestStore(t, FDv2ModePoll, ft, func(o *FDv2SkillStoreOptions) { o.PollInterval = time.Millisecond })
	start(t, s)
	eventually(t, "two polls", func() bool { return len(ft.requests()) >= 2 })
	if ft.requests()[1].Header.Get("If-None-Match") != "" {
		t.Fatal("an etag was adopted from a body that applied nothing")
	}
	if ok, _ := s.IsInitialized(); ok {
		t.Fatal("initialized by an unrecognised intent")
	}
	if all, _ := s.AllObjects("skill"); len(all) != 0 {
		t.Fatal("objects held from an unrecognised intent")
	}

	// A 304 to a request with no etag: not a payload, not a failure.
	_, _ = setup(t)
	ft = &fakeTransport{}
	ft.push(fakeResp{status: 304})
	s = newTestStore(t, FDv2ModePoll, ft, nil)
	start(t, s)
	eventually(t, "one poll", func() bool { return len(ft.requests()) >= 1 })
	if ft.requests()[0].Header.Get("If-None-Match") != "" {
		t.Fatal("the first request carried an etag")
	}
	if s.WaitForSkills(30*time.Millisecond) || s.Diagnostics().ConnectionFailures != 0 {
		t.Fatal("a 304 established a payload or counted as a failure")
	}

	// An earned etag (a none intent) is confirmed by a 304.
	_, _ = setup(t)
	ft = &fakeTransport{}
	ft.push(fakeResp{body: pollBody(commitEvents("p", "(p:p:1)", put("a:1", "a"))...)},
		fakeResp{header: http.Header{"Etag": {`"e2"`}}, body: pollBody(ev("server-intent", intent("none", "p")))},
		fakeResp{status: 304})
	s = newTestStore(t, FDv2ModePoll, ft, func(o *FDv2SkillStoreOptions) { o.PollInterval = time.Millisecond })
	start(t, s)
	eventually(t, "four polls", func() bool { return len(ft.requests()) >= 4 })
	if ft.requests()[2].Header.Get("If-None-Match") != `"e2"` {
		t.Fatalf("the earned etag was not offered: %q", ft.requests()[2].Header.Get("If-None-Match"))
	}
	if ok, _ := s.IsInitialized(); !ok {
		t.Fatal("not initialized")
	}
	if got, _ := s.GetObject("skill", "a", 0); got == nil {
		t.Fatal("content lost on a 304")
	}
}

func TestFDv2ListenersFireAtCommitPerObject(t *testing.T) {
	_, _ = setup(t)
	ft := &fakeTransport{}
	ft.push(fakeResp{body: sse(commitEvents("p", "(p:p:1)", put("a:1", "a"), put("b:1", "b"), put("c:1", "c"))...), hold: true})
	s := newTestStore(t, FDv2ModeStream, ft, nil)
	var mu sync.Mutex
	var seen []string
	l := NewListener(func(raw any) {
		mu.Lock()
		defer mu.Unlock()
		seen = append(seen, raw.(map[string]any)["key"].(string))
	})
	if err := s.AddListener("flag", l); !errors.Is(err, ErrUnsupportedKind) {
		t.Fatalf("listener on another kind: %v", err)
	}
	_ = s.AddListener("skill", l)
	start(t, s)
	_ = s.WaitForSkills(5 * time.Second)
	eventually(t, "three notifications", func() bool { mu.Lock(); defer mu.Unlock(); return len(seen) == 3 })
	s.RemoveListener("skill", l)
	s.RemoveListener("skill", l) // no-op
}

// the batching is in when listeners fire, not how many times. A
// payload streaming in notifies nobody until payload-transferred commits it,
// and a delete-object reaches listeners as a content-free tombstone.
func TestFDv2ListenersFireOnlyAtTheCommit(t *testing.T) {
	_, _ = setup(t)
	gate, revoke := make(chan struct{}), make(chan struct{})
	ft := &fakeTransport{}
	ft.push(
		fakeResp{
			body: sse(ev("server-intent", intent("xfer-full", "p")),
				ev("put-object", put("a:1", "a")), ev("put-object", put("b:1", "b")), ev("put-object", put("c:1", "c"))),
			gate: gate,
			then: sse(ev("payload-transferred", transferred("(p:p:1)"))),
		},
		fakeResp{gate: revoke, hold: true, then: sse(ev("server-intent", intent("xfer-changes", "p")),
			ev("delete-object", del("a:1")), ev("payload-transferred", transferred("(p:p:2)")))},
	)
	s := newTestStore(t, FDv2ModeStream, ft, nil)
	var mu sync.Mutex
	var seen []map[string]any
	_ = s.AddListener("skill", NewListener(func(raw any) {
		mu.Lock()
		defer mu.Unlock()
		seen = append(seen, raw.(map[string]any))
	}))
	count := func() int { mu.Lock(); defer mu.Unlock(); return len(seen) }
	start(t, s)
	eventually(t, "the three puts to be read", func() bool { return s.Diagnostics().SkillObjectsReceived == 3 })
	time.Sleep(20 * time.Millisecond) // a listener firing per put would have fired by now
	if n := count(); n != 0 {
		t.Fatalf("%d notifications mid-transfer; listeners fire only at the commit", n)
	}
	close(gate)
	eventually(t, "the commit", func() bool { return count() >= 3 })
	if n := count(); n != 3 {
		t.Fatalf("notifications after the commit = %d, want one per changed object", n)
	}
	close(revoke)
	eventually(t, "the revocation", func() bool { return count() == 4 })
	mu.Lock()
	tomb := seen[3]
	mu.Unlock()
	if got := sortedKeys(tomb); !slices.Equal(got, []string{"key", "version"}) || tomb["key"] != "a" || tomb["version"] != 1 {
		t.Fatalf("a revocation must notify a {key, version} tombstone with no content: %v", tomb)
	}
}

// giving up ends the delivery run, not the store. A payload held
// before a 422 is served while the store is given up and across the Start
// that recovers it, and the new run's backoff starts from its first step.
func Test422KeepsServingTheHeldPayloadAndStartResetsBackoff(t *testing.T) {
	_, _ = setup(t)
	ft := &fakeTransport{}
	// The commit's stream then ends, two 503s grow the backoff, and a 422 ends the run.
	ft.push(fakeResp{body: sse(commitEvents("p", "(p:p:1)", put("a:1", "held"))...)},
		fakeResp{status: 503}, fakeResp{status: 503}, fakeResp{status: 422})
	s := newTestStore(t, FDv2ModeStream, ft, nil)
	SetSkillStore(s)
	start(t, s)
	eventually(t, "give up", func() bool { return s.Failed() != "" })
	served := func(when string) {
		t.Helper()
		got, err := GetSkill("a", 0)
		if err != nil || got == nil || string(got.Content()) != "held" {
			t.Fatalf("%s: the held payload was not served: %v %v", when, got, err)
		}
	}
	served("given up")
	s.mu.Lock()
	grown := s.backoffAttempt
	s.mu.Unlock()
	if grown < 2 {
		t.Fatalf("backoff attempt before the restart = %d; the reset would not be observable", grown)
	}

	start(t, s) // the script is exhausted: the new run's request holds
	s.mu.Lock()
	attempt := s.backoffAttempt
	s.mu.Unlock()
	if attempt != 0 || s.Failed() != "" || s.Diagnostics().ConnectionFailures != 0 {
		t.Fatalf("after Start: backoff attempt %d, failed %q, diag %+v", attempt, s.Failed(), s.Diagnostics())
	}
	served("restarted, before the new run delivers")
	if ok, _ := s.IsInitialized(); !ok {
		t.Fatal("a restart must not uninitialize a store that holds a payload")
	}
}

func TestFDv2BeforeFirstPayloadWriteAllDeletesNothing(t *testing.T) {
	_, _ = setup(t)
	root := t.TempDir()
	mustWrite(t, []Skill{skillOf("old", 1, "o")}, root, nil)
	ft := &fakeTransport{}
	s := newTestStore(t, FDv2ModeStream, ft, nil)
	SetSkillStore(s)
	r, _ := WriteAllSkills(root, nil)
	if r.OK() || !strings.Contains(r.Errors()[0].ErrorMessage(), "WaitForSkills") || slices.Contains(actionKinds(r), "old=removed") {
		t.Fatalf("before the first payload: %v", actionKinds(r))
	}
	ft.push(fakeResp{body: sse(commitEvents("p", "(p:p:1)")...), hold: true}) // the environment's last skill revoked
	start(t, s)
	if !s.WaitForSkills(5 * time.Second) {
		t.Fatal("no commit")
	}
	r, _ = WriteAllSkills(root, nil)
	if !slices.Equal(actionKinds(r), []string{"old=removed"}) {
		t.Fatalf("after the commit: %v", actionKinds(r))
	}
}

// the transport adds no dependency: its files import only the
// standard library.
func TestFDv2ImportsOnlyTheStandardLibrary(t *testing.T) {
	fset := token.NewFileSet()
	for _, file := range []string{"skills_fdv2_reader.go", "skills_fdv2_store.go"} {
		f, err := parser.ParseFile(fset, file, nil, parser.ImportsOnly)
		if err != nil {
			t.Fatal(err)
		}
		for _, imp := range f.Imports {
			path, _ := strconv.Unquote(imp.Path.Value)
			if strings.Contains(strings.SplitN(path, "/", 2)[0], ".") {
				t.Fatalf("%s imports %s", file, path)
			}
		}
	}
}
