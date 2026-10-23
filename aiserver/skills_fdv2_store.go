package aiserver

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Default endpoints: LaunchDarkly polls and streams from different hosts.
const (
	fdv2DefaultBaseURI   = "https://sdk.launchdarkly.com"
	fdv2DefaultStreamURI = "https://stream.launchdarkly.com"
	fdv2PollPath         = "/sdk/poll"
	fdv2StreamPath       = "/sdk/stream"
)

const (
	defaultPollInterval      = 30 * time.Second
	defaultPollReadTimeout   = 10 * time.Second
	defaultStreamReadTimeout = 300 * time.Second
	defaultInitialBackoff    = time.Second
	defaultMaxBackoff        = 30 * time.Second
	fdv2CloseTimeout         = 5 * time.Second
)

// maxResponseBytes is the most the transport holds from one poll body or one
// streamed event; crossing it is fatal. A variable so tests can lower it.
var maxResponseBytes = 64 * 1024 * 1024

// backoffResetInterval is how long a stream must stay open before its end
// resets the backoff delay, as in the base server-side SDKs. Internal; tests
// override it.
var backoffResetInterval = 60 * time.Second

// FDv2Mode selects how FDv2SkillStore receives delivery.
type FDv2Mode string

// The two delivery modes. The zero value "" means FDv2ModeStream.
const (
	// FDv2ModeStream holds a long-lived connection: revocations arrive in seconds.
	FDv2ModeStream FDv2Mode = "stream"
	// FDv2ModePoll polls every PollInterval: revocations are one interval late.
	FDv2ModePoll FDv2Mode = "poll"
)

// FDv2SkillStoreOptions configure NewFDv2SkillStore. Zero values select the
// defaults; negative durations are an ErrInvalidArgument.
type FDv2SkillStoreOptions struct {
	// BaseURI is the origin for GET /sdk/poll (default
	// https://sdk.launchdarkly.com). Given without StreamURI it serves both
	// endpoints, for a relay or private instance. Must be https.
	BaseURI string
	// StreamURI is the origin for GET /sdk/stream (default
	// https://stream.launchdarkly.com). Must be https.
	StreamURI string
	// Mode is FDv2ModeStream (default) or FDv2ModePoll.
	Mode FDv2Mode
	// PollInterval is the time between polls (default 30s).
	PollInterval time.Duration
	// ReadTimeout is the only network timeout. In poll mode it bounds a whole
	// request (default 10s); in stream mode, each wait for more bytes
	// (default 300s).
	ReadTimeout time.Duration
	// InitialBackoff is the first retry delay (default 1s).
	InitialBackoff time.Duration
	// MaxBackoff caps every retry delay, Retry-After included (default 30s).
	// InitialBackoff may not exceed it.
	MaxBackoff time.Duration

	// transport is the requester seam: tests inject an in-process
	// http.RoundTripper so no socket is opened.
	transport http.RoundTripper
}

// FDv2SkillStore is the SkillStore fed by LaunchDarkly's FDv2 delivery
// channel, authenticated with the environment's server-side SDK key. It holds
// raw wire objects; the accessors verify them. Delivery runs on a background
// goroutine started by Start; GetObject only reads what has already arrived,
// so gate boot on WaitForSkills.
//
// A transport failure never empties the store: the last known good content is
// served throughout, and Failed and Diagnostics report degradation. Close is
// final.
type FDv2SkillStore struct {
	mode           FDv2Mode
	pollInterval   time.Duration
	readTimeout    time.Duration
	initialBackoff time.Duration
	maxBackoff     time.Duration
	baseURI        string
	streamURI      string
	sdkKey         string
	client         *http.Client

	listeners listenerSet

	mu             sync.Mutex
	objects        *objectSet
	reader         *fdv2Reader
	basis          string
	etag           string
	etagBasis      string
	carriedState   bool
	closed         bool
	running        bool
	cancel         context.CancelFunc
	done           chan struct{}
	firstPayload   bool
	released       chan struct{}
	releasedClosed bool
	failedReason   string
	failures       int
	backoffAttempt int
	connectedAt    time.Time
	answered       bool
}

var clientSideIDRE = regexp.MustCompile(`^[0-9a-f]{20,}$`)

// NewFDv2SkillStore validates the credential and options and builds a store.
// Call Start to begin delivery. A mobile key or client-side environment ID is
// refused: skill content is customer-confidential and server-side only.
func NewFDv2SkillStore(sdkKey string, opts *FDv2SkillStoreOptions) (*FDv2SkillStore, error) {
	if opts == nil {
		opts = &FDv2SkillStoreOptions{}
	}
	key := strings.TrimSpace(sdkKey)
	switch {
	case key == "":
		return nil, invalidArg("FDv2SkillStore requires a LaunchDarkly server-side SDK key (sdk-...); none was given")
	case strings.HasPrefix(key, "mob-"):
		return nil, invalidArg("FDv2SkillStore was given a mobile key (mob-...). Agent Skills are server-side only: " +
			"skill content is customer-confidential and is never delivered to a mobile or client-side process. " +
			"Use the environment's server-side SDK key (sdk-...)")
	case clientSideIDRE.MatchString(key):
		return nil, invalidArg("FDv2SkillStore was given what looks like a client-side environment ID. Agent Skills " +
			"are server-side only: skill content is customer-confidential. Use the environment's server-side SDK key (sdk-...)")
	case !strings.HasPrefix(key, "sdk-"):
		currentLogger().Warn("the credential given to FDv2SkillStore does not look like a LaunchDarkly server-side SDK key (sdk-...)")
	}
	base, stream := opts.BaseURI, opts.StreamURI
	if stream == "" {
		stream = base
		if stream == "" {
			stream = fdv2DefaultStreamURI
		}
	}
	if base == "" {
		base = fdv2DefaultBaseURI
	}
	for _, u := range []struct{ name, value string }{{"BaseURI", base}, {"StreamURI", stream}} {
		parsed, err := url.Parse(u.value)
		if err != nil || parsed.Scheme != "https" || parsed.Host == "" {
			return nil, invalidArg("FDv2SkillStore refuses %s %q: an https:// URI with a host is required, so the SDK key is never sent in cleartext", u.name, u.value)
		}
	}
	mode := opts.Mode
	if mode == "" {
		mode = FDv2ModeStream
	}
	if mode != FDv2ModeStream && mode != FDv2ModePoll {
		return nil, invalidArg("Mode must be %q or %q, got %q", FDv2ModeStream, FDv2ModePoll, mode)
	}
	durations := []struct {
		name  string
		value *time.Duration
		def   time.Duration
	}{
		{"PollInterval", &opts.PollInterval, defaultPollInterval},
		{"ReadTimeout", &opts.ReadTimeout, map[FDv2Mode]time.Duration{FDv2ModeStream: defaultStreamReadTimeout, FDv2ModePoll: defaultPollReadTimeout}[mode]},
		{"InitialBackoff", &opts.InitialBackoff, defaultInitialBackoff},
		{"MaxBackoff", &opts.MaxBackoff, defaultMaxBackoff},
	}
	resolved := make([]time.Duration, len(durations))
	for i, d := range durations {
		switch {
		case *d.value < 0:
			return nil, invalidArg("%s must be positive, got %v", d.name, *d.value)
		case *d.value == 0:
			resolved[i] = d.def
		default:
			resolved[i] = *d.value
		}
	}
	if resolved[2] > resolved[3] {
		return nil, invalidArg("InitialBackoff (%v) must not exceed MaxBackoff (%v)", resolved[2], resolved[3])
	}
	transport := opts.transport
	if transport == nil {
		transport = http.DefaultTransport
	}
	objects := newObjectSet()
	s := &FDv2SkillStore{
		mode: mode, pollInterval: resolved[0], readTimeout: resolved[1], initialBackoff: resolved[2], maxBackoff: resolved[3],
		baseURI: strings.TrimRight(base, "/"), streamURI: strings.TrimRight(stream, "/"), sdkKey: key,
		// Redirects are never followed: that would forward the SDK key to
		// whatever host Location names.
		client:   &http.Client{Transport: transport, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }},
		objects:  objects,
		reader:   newFDv2Reader(objects),
		released: make(chan struct{}),
	}
	return s, nil
}

// Start begins delivery on a background goroutine and returns the store, so
// it chains. It is a no-op while delivery runs. A store that gave up on a
// fatal status (Failed is set) is started afresh, with Failed cleared and the
// failure count and backoff reset. After Close it returns ErrStoreClosed.
func (s *FDv2SkillStore) Start() (*FDv2SkillStore, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil, fmt.Errorf("%w: Close is final, so delivery cannot resume; construct a new FDv2SkillStore. "+
			"Held content is still readable from the closed store", ErrStoreClosed)
	}
	if s.running && s.failedReason == "" {
		return s, nil
	}
	s.failedReason = ""
	s.failures = 0
	s.reader.diag.ConnectionFailures = 0
	s.backoffAttempt = 0
	if !s.firstPayload && s.releasedClosed {
		s.released = make(chan struct{})
		s.releasedClosed = false
	}
	ctx, cancel := context.WithCancel(context.Background())
	s.cancel = cancel
	s.running = true
	s.done = make(chan struct{})
	go s.run(ctx, s.done)
	return s, nil
}

// Close stops delivery and waits briefly for the goroutine to exit. It is
// idempotent and final; held content keeps being served.
func (s *FDv2SkillStore) Close() {
	s.mu.Lock()
	s.closed = true
	cancel, done := s.cancel, s.done
	s.releaseLocked()
	s.mu.Unlock()
	if cancel != nil {
		cancel()
	}
	if done != nil {
		select {
		case <-done:
		case <-time.After(fdv2CloseTimeout):
		}
	}
}

// WaitForSkills blocks until the first payload commits or timeout elapses,
// and reports whether one has. It returns false early when delivery has ended
// (Close, or a fatal status) with nothing committed. True does not mean any
// skill verified; see Diagnostics.
func (s *FDv2SkillStore) WaitForSkills(timeout time.Duration) bool {
	s.mu.Lock()
	ch := s.released
	s.mu.Unlock()
	t := time.NewTimer(max(timeout, 0))
	defer t.Stop()
	select {
	case <-ch:
	case <-t.C:
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.firstPayload
}

// IsInitialized is WaitForSkills without the wait: true once a payload has
// committed, and still true after Close. Nothing but a commit makes it true.
func (s *FDv2SkillStore) IsInitialized() (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.firstPayload, nil
}

// Failed is why delivery stopped for good, or "" while it runs.
func (s *FDv2SkillStore) Failed() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.failedReason
}

// Diagnostics returns a snapshot of the transport counters.
func (s *FDv2SkillStore) Diagnostics() StoreDiagnostics {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.reader.diag
}

// GetObject implements SkillStore. It never fails: it reads what arrived.
func (s *FDv2SkillStore) GetObject(kind, key string, version int) (any, error) {
	if kind != skillObjectKind {
		return nil, nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.objects.get(key, version), nil
}

// AllObjects implements SkillStore. Map keys are opaque.
func (s *FDv2SkillStore) AllObjects(kind string) (map[string]any, error) {
	if kind != skillObjectKind {
		return map[string]any{}, nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.objects.snapshot(), nil
}

// AddListener registers l to be called once per changed object when a
// payload commits, never mid-transfer, on the delivery goroutine: keep it
// cheap. A revocation passes a {"key", "version"} tombstone with no content.
func (s *FDv2SkillStore) AddListener(kind string, l *Listener) error {
	return s.listeners.add("FDv2SkillStore", kind, l)
}

// RemoveListener removes one registration of l; a no-op if none.
func (s *FDv2SkillStore) RemoveListener(kind string, l *Listener) {
	s.listeners.remove(kind, l)
}

// releaseLocked lets every WaitForSkills caller go. Call with mu held.
func (s *FDv2SkillStore) releaseLocked() {
	if !s.releasedClosed {
		close(s.released)
		s.releasedClosed = true
	}
}

// Transport failures are one of exactly two classes.
type fatalError struct{ msg string }

func (e *fatalError) Error() string { return e.msg }

type recoverableError struct {
	msg        string
	retryAfter time.Duration // valid only when hasRetry
	hasRetry   bool
	recycled   bool
	stale      bool // HTTP 400: drop client state and retry once
}

func (e *recoverableError) Error() string { return e.msg }

func recoverable(format string, args ...any) error {
	return &recoverableError{msg: fmt.Sprintf(format, args...)}
}

const requestAdvice = "The request carries only the SDK key, kinds=agent-skill and, after the first payload, a basis " +
	"selector; check the base URI and that the endpoint speaks FDv2."

// classifyStatus answers every non-success status with exactly one class.
func classifyStatus(status int, header http.Header) error {
	switch status {
	case http.StatusUnauthorized:
		return &fatalError{"LaunchDarkly rejected the SDK key (HTTP 401); skill delivery cannot start. Check that the key is the environment's server-side SDK key"}
	case http.StatusForbidden:
		return &fatalError{"LaunchDarkly returned HTTP 403; the SDK key is not permitted to receive Agent Skills. Check the key, and contact LaunchDarkly support to enable FDv2 delivery for the account"}
	case http.StatusNotFound:
		return &fatalError{"LaunchDarkly returned HTTP 404 for the FDv2 endpoint; check the base URI, and that this instance serves /sdk/poll and /sdk/stream"}
	case http.StatusUnprocessableEntity:
		return &fatalError{"LaunchDarkly will not assign an Agent Skills payload to this connection (HTTP 422). The two causes " +
			"are: Agent Skills delivery is not enabled for this LaunchDarkly account, or the SDK key is view-scoped. Check " +
			"both; once fixed, call Start() on this store to resume delivery"}
	case http.StatusMethodNotAllowed, http.StatusNotAcceptable, http.StatusRequestURITooLong, http.StatusNotImplemented:
		return &fatalError{fmt.Sprintf("LaunchDarkly returned HTTP %d, which retrying will not fix. %s", status, requestAdvice)}
	case http.StatusBadRequest:
		return &recoverableError{msg: "LaunchDarkly returned HTTP 400. " + requestAdvice, stale: true}
	}
	d, ok := retryAfter(header)
	return &recoverableError{msg: fmt.Sprintf("LaunchDarkly returned HTTP %d", status), retryAfter: d, hasRetry: ok}
}

// retryAfter reads Retry-After in seconds. The HTTP-date form, a negative or
// a non-finite value is ignored, and the backoff schedule applies.
func retryAfter(h http.Header) (time.Duration, bool) {
	f, err := strconv.ParseFloat(strings.TrimSpace(h.Get("Retry-After")), 64)
	if err != nil || math.IsNaN(f) || math.IsInf(f, 0) || f < 0 {
		return 0, false
	}
	return time.Duration(min(f, float64(math.MaxInt64/2)/float64(time.Second)) * float64(time.Second)), true
}

// backoffDelay is exponential, capped, with subtractive jitter so the cap is
// a real ceiling. The exponent is clamped so any attempt number stays finite.
func backoffDelay(attempt int, base, ceiling time.Duration) time.Duration {
	exp := min(max(0, attempt-1), 62)
	d := min(ceiling.Seconds(), base.Seconds()*math.Pow(2, float64(exp)))
	return time.Duration(d * (1 - 0.5*rand.Float64()) * float64(time.Second))
}

// retryDelay is the backoff schedule, or a server-requested Retry-After
// floored at InitialBackoff and capped at MaxBackoff. Retry-After is not
// jittered: jitter would reconnect sooner than the server asked.
func (s *FDv2SkillStore) retryDelay(re *recoverableError, attempt int) time.Duration {
	if re.hasRetry {
		return min(max(re.retryAfter, s.initialBackoff), s.maxBackoff)
	}
	return backoffDelay(attempt, s.initialBackoff, s.maxBackoff)
}

func (s *FDv2SkillStore) url(origin, path, basis string) string {
	q := url.Values{"kinds": {fdv2PayloadKind}}
	if basis != "" {
		q.Set("basis", basis)
	}
	return origin + path + "?" + q.Encode()
}

func (s *FDv2SkillStore) run(ctx context.Context, done chan struct{}) {
	defer close(done)
	for ctx.Err() == nil {
		s.mu.Lock()
		s.answered = false
		s.mu.Unlock()
		var err error
		if s.mode == FDv2ModeStream {
			err = s.streamOnce(ctx)
		} else {
			err = s.pollOnce(ctx)
		}
		if ctx.Err() != nil {
			return // Close interrupted the request: not a failure
		}
		if err == nil {
			s.recordSuccess()
			s.mu.Lock()
			s.backoffAttempt = 0 // a completed poll resets the delay
			s.mu.Unlock()
			if !sleep(ctx, s.pollInterval) {
				return
			}
			continue
		}
		var fatal *fatalError
		if errors.As(err, &fatal) {
			s.giveUp(fatal.msg)
			return
		}
		var re *recoverableError
		if !errors.As(err, &re) {
			re = &recoverableError{msg: err.Error()}
		}
		s.mu.Lock()
		if re.stale {
			if !s.carriedState {
				s.mu.Unlock()
				s.giveUp(re.msg)
				return
			}
			s.basis, s.etag, s.etagBasis = "", "", ""
		}
		s.reader.abandon()
		answered := s.answered
		if !(re.recycled && answered) {
			s.failures++
			s.reader.diag.ConnectionFailures = s.failures
			s.reader.diag.LastError = re.msg
		}
		if !s.connectedAt.IsZero() && time.Since(s.connectedAt) >= backoffResetInterval {
			s.backoffAttempt = 0
		}
		s.connectedAt = time.Time{}
		s.backoffAttempt++
		attempt := s.backoffAttempt
		s.mu.Unlock()
		delay := s.retryDelay(re, attempt)
		if answered {
			currentLogger().Debug("the FDv2 stream ended after a complete answer; reconnecting", "reason", re.msg, "delay", delay)
		} else {
			currentLogger().Warn("skill delivery failed; retrying", "reason", re.msg, "delay", delay)
		}
		if !sleep(ctx, delay) {
			return
		}
	}
}

func sleep(ctx context.Context, d time.Duration) bool {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-t.C:
		return true
	}
}

func (s *FDv2SkillStore) recordSuccess() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.failures = 0
	s.answered = true
	s.reader.diag.ConnectionFailures = 0
}

func (s *FDv2SkillStore) giveUp(reason string) {
	s.mu.Lock()
	s.failedReason = reason
	s.reader.diag.LastError = reason
	s.running = false
	s.releaseLocked()
	s.mu.Unlock()
	currentLogger().Error("skill delivery has stopped and will not retry; the store keeps serving the last content it "+
		"received. Call Start() on this store once the cause is fixed", "reason", reason)
}

// apply feeds one event to the reader, publishes a commit, and returns the
// disconnect the event calls for. completed reports a commit or a none
// intent: the only exchanges an etag may be adopted on.
func (s *FDv2SkillStore) apply(name string, data []byte) (completed bool, err error) {
	s.mu.Lock()
	out := s.reader.handle(name, data)
	if out.committed && out.basis != "" {
		s.basis = out.basis
	}
	s.mu.Unlock()
	if out.committed || out.upToDate {
		s.recordSuccess()
	}
	if out.committed {
		s.mu.Lock()
		s.firstPayload = true
		s.releaseLocked()
		s.mu.Unlock()
		s.listeners.notify(out.changes)
	}
	if out.disconnect != "" {
		return false, &recoverableError{msg: out.disconnect, recycled: out.recycled}
	}
	return out.committed || out.upToDate, nil
}

func (s *FDv2SkillStore) request(ctx context.Context, origin, path, basis string, header http.Header) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, s.url(origin, path, basis), nil)
	if err != nil {
		return nil, &fatalError{fmt.Sprintf("the FDv2 request could not be built: %v", err)}
	}
	req.Header = header
	req.Header.Set("Authorization", s.sdkKey)
	resp, err := s.client.Do(req)
	if err != nil {
		return nil, recoverable("the FDv2 request failed: %v", err)
	}
	return resp, nil
}

func (s *FDv2SkillStore) pollOnce(ctx context.Context) error {
	s.mu.Lock()
	basis := s.basis
	etag := ""
	if s.etagBasis == basis {
		etag = s.etag
	}
	s.carriedState = basis != "" || etag != ""
	s.mu.Unlock()
	header := http.Header{"Accept": {"application/json"}}
	if etag != "" {
		header.Set("If-None-Match", etag)
	}
	reqCtx, cancel := context.WithTimeout(ctx, s.readTimeout)
	defer cancel()
	resp, err := s.request(reqCtx, s.baseURI, fdv2PollPath, basis, header)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusNotModified {
		// A 304 confirms the payload held; it cannot establish one.
		return nil
	}
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		return classifyStatus(resp.StatusCode, resp.Header)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, int64(maxResponseBytes)+1))
	if err != nil {
		return recoverable("reading the polling response failed: %v", err)
	}
	if len(body) > maxResponseBytes {
		return &fatalError{fmt.Sprintf("the polling response exceeded the %d-byte transport bound; nothing from it was applied", maxResponseBytes)}
	}
	events, err := decodePollBody(body)
	if err != nil {
		return err
	}
	completed := false
	for _, e := range events {
		c, err := s.apply(e.name, e.data)
		if err != nil {
			return err
		}
		completed = completed || c
	}
	// An etag describes the body it came with, so it is earned only when that
	// body is what the store now holds.
	if completed {
		newEtag := resp.Header.Get("ETag")
		s.mu.Lock()
		if newEtag != "" {
			s.etag = newEtag
		}
		// Sent again only with the basis it was issued for.
		s.etagBasis = basis
		s.mu.Unlock()
	}
	return nil
}

type wireEvent struct {
	name string
	data []byte
}

func decodePollBody(body []byte) ([]wireEvent, error) {
	var parsed struct {
		Events []map[string]json.RawMessage `json:"events"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil || parsed.Events == nil {
		return nil, recoverable("the polling response was not a JSON object with an events array")
	}
	var out []wireEvent
	for _, e := range parsed.Events {
		name := jsonString(e["event"])
		if name == "" {
			continue
		}
		data := e["data"]
		if knownEvents[name] && len(bytes.TrimSpace(data)) == 0 {
			return nil, recoverable("an FDv2 %q event in the polling response carried no data", name)
		}
		out = append(out, wireEvent{name, data})
	}
	return out, nil
}

func (s *FDv2SkillStore) streamOnce(ctx context.Context) error {
	s.mu.Lock()
	basis := s.basis
	s.carriedState = basis != ""
	s.mu.Unlock()
	connCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	// ReadTimeout bounds the wait for headers, then each wait for bytes.
	idle := time.AfterFunc(s.readTimeout, cancel)
	defer idle.Stop()
	resp, err := s.request(connCtx, s.streamURI, fdv2StreamPath, basis,
		http.Header{"Accept": {"text/event-stream"}, "Cache-Control": {"no-cache"}})
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		return classifyStatus(resp.StatusCode, resp.Header)
	}
	s.mu.Lock()
	s.connectedAt = time.Now()
	s.mu.Unlock()
	reader := newSSEReader(&idleReader{r: resp.Body, timer: idle, timeout: s.readTimeout})
	for {
		name, data, err := reader.next()
		if err != nil {
			if errors.Is(err, io.EOF) {
				return recoverable("the FDv2 stream closed unexpectedly")
			}
			return err
		}
		if _, err := s.apply(name, data); err != nil {
			return err
		}
	}
}

// idleReader re-arms the read deadline on every read.
type idleReader struct {
	r       io.Reader
	timer   *time.Timer
	timeout time.Duration
}

func (ir *idleReader) Read(p []byte) (int, error) {
	n, err := ir.r.Read(p)
	if n > 0 {
		ir.timer.Reset(ir.timeout)
	}
	return n, err
}

// sseReader decodes an SSE body into (event, data) pairs: event and data
// fields, multi-line data joined with newlines, a blank line dispatching, ":"
// comments skipped. A known event whose data is missing, empty or not JSON
// fails the connection as recoverable, never skipped: the payload-transferred
// after it would otherwise commit without it.
type sseReader struct {
	r *bufio.Reader
}

func newSSEReader(r io.Reader) *sseReader { return &sseReader{r: bufio.NewReaderSize(r, 64*1024)} }

func (s *sseReader) line() (string, error) {
	var buf []byte
	for {
		chunk, err := s.r.ReadSlice('\n')
		buf = append(buf, chunk...)
		if len(buf) > maxResponseBytes {
			return "", &fatalError{fmt.Sprintf("an FDv2 stream line exceeded the %d-byte transport bound; nothing from the in-flight payload was applied", maxResponseBytes)}
		}
		if errors.Is(err, bufio.ErrBufferFull) {
			continue
		}
		if err != nil {
			if errors.Is(err, io.EOF) && len(buf) == 0 {
				return "", io.EOF
			}
			if errors.Is(err, io.EOF) {
				return "", recoverable("the FDv2 stream ended mid-line")
			}
			return "", recoverable("reading the FDv2 stream failed: %v", err)
		}
		return strings.TrimRight(string(buf), "\r\n"), nil
	}
}

func (s *sseReader) next() (string, []byte, error) {
	var (
		name    string
		data    []string
		hasData bool
		size    int
	)
	for {
		line, err := s.line()
		if err != nil {
			return "", nil, err
		}
		if line == "" {
			if name == "" {
				data, hasData, size = nil, false, 0
				continue
			}
			payload := []byte(strings.Join(data, "\n"))
			if knownEvents[name] && (!hasData || !json.Valid(payload)) {
				msg := fmt.Sprintf("an FDv2 %q event's data was not JSON; the connection was dropped and nothing from the in-flight payload was applied", name)
				currentLogger().Warn(msg + "; reconnecting")
				return "", nil, recoverable("%s", msg)
			}
			return name, payload, nil
		}
		if strings.HasPrefix(line, ":") {
			continue
		}
		size += len(line)
		if size > maxResponseBytes {
			return "", nil, &fatalError{fmt.Sprintf("an FDv2 stream event exceeded the %d-byte transport bound; nothing from the in-flight payload was applied", maxResponseBytes)}
		}
		field, value, _ := strings.Cut(line, ":")
		value = strings.TrimPrefix(value, " ")
		switch field {
		case "event":
			name = value
		case "data":
			data = append(data, value)
			hasData = true
		}
	}
}
