package evaluations

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"regexp"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// apiFixture is an apiClient over the fake transport that records delays.
func apiFixture(t *testing.T) (*fakeEnv, *apiClient, *[]time.Duration) {
	e := newEnv(t, 1)
	c := newAPIClient("raw-token", &http.Client{Transport: e}, 0, 0)
	var delays []time.Duration
	c.sleep = func(_ context.Context, d time.Duration) error { delays = append(delays, d); return nil }
	c.random = func() float64 { return 0.25 }
	return e, c, &delays
}

func TestAPIURLAndQuery(t *testing.T) {
	e, c, _ := apiFixture(t)
	e.route("GET", "projects/p/things", resp{body: map[string]any{}})
	if _, err := c.get(context.Background(), "projects/p/things", param{"mode", "all"}, param{"skip", nil}, param{"q", "a b&c"}); err != nil {
		t.Fatal(err)
	}
	got := e.requests()[0]
	if got.URL != "https://app.launchdarkly.com/api/v2/projects/p/things?mode=all&q=a+b%26c" {
		t.Fatalf("url = %s", got.URL)
	}
	if strings.Contains(got.Query, "skip") {
		t.Fatal("nil parameter must be omitted")
	}
}

func TestAPIHeadersAndBody(t *testing.T) {
	e, c, _ := apiFixture(t)
	e.route("GET", "g", resp{body: map[string]any{}})
	e.route("POST", "p", resp{body: map[string]any{}})
	if _, err := c.get(context.Background(), "g"); err != nil {
		t.Fatal(err)
	}
	if _, err := c.post(context.Background(), "p", map[string]any{"a": 1}, false); err != nil {
		t.Fatal(err)
	}
	reqs := e.requests()
	version := regexp.MustCompile(`^(\d{8}|beta)$`)
	for _, r := range reqs {
		if r.Header.Get("Authorization") != "raw-token" {
			t.Errorf("%s Authorization = %q, want the raw token", r.Method, r.Header.Get("Authorization"))
		}
		if r.Header.Get("Accept") != "application/json" {
			t.Errorf("%s Accept = %q", r.Method, r.Header.Get("Accept"))
		}
		if !version.MatchString(r.Header.Get("LD-API-Version")) {
			t.Errorf("%s LD-API-Version = %q", r.Method, r.Header.Get("LD-API-Version"))
		}
		if !strings.Contains(r.Header.Get("User-Agent"), "evaluations-go") {
			t.Errorf("%s User-Agent = %q", r.Method, r.Header.Get("User-Agent"))
		}
	}
	if _, ok := reqs[0].Header["Content-Type"]; ok {
		t.Error("GET must not send Content-Type")
	}
	if reqs[1].Header.Get("Content-Type") != "application/json" || string(reqs[1].Raw) != `{"a":1}` {
		t.Errorf("POST body %s / content type %q", reqs[1].Raw, reqs[1].Header.Get("Content-Type"))
	}
}

func TestAPIResponseHandling(t *testing.T) {
	e, c, _ := apiFixture(t)
	e.route("GET", "ok", resp{body: map[string]any{"x": 1.0}})
	e.route("GET", "empty", resp{status: 204})
	e.route("GET", "bad", resp{body: "{not json"})
	e.route("GET", "missing", resp{status: 404, body: "nope"})
	ctx := context.Background()
	if v, err := c.get(ctx, "ok"); err != nil || v.(map[string]any)["x"] != 1.0 {
		t.Fatalf("ok: %v %v", v, err)
	}
	if v, err := c.get(ctx, "empty"); err != nil || v != nil {
		t.Fatalf("empty: %v %v", v, err)
	}
	if _, err := c.get(ctx, "bad"); !errors.Is(err, ErrInvalidResponse) {
		t.Fatalf("bad JSON: %v", err)
	}
	_, err := c.get(ctx, "missing")
	var apiErr *APIError
	if !errors.As(err, &apiErr) || !errors.Is(err, ErrAPI) {
		t.Fatalf("404: %v", err)
	}
	if apiErr.Status != 404 || apiErr.Method != "GET" || apiErr.Path != "missing" || apiErr.Body != "nope" {
		t.Fatalf("fields: %+v", apiErr)
	}
	for _, s := range []string{"404", "GET", "missing", "nope"} {
		if !strings.Contains(err.Error(), s) {
			t.Errorf("message %q lacks %q", err, s)
		}
	}
}

func TestAPIRetryPolicy(t *testing.T) {
	timeout := &url.Error{Op: "Get", URL: "x", Err: context.DeadlineExceeded}
	cases := []struct {
		name       string
		method     string
		idempotent bool
		answers    []resp
		wantCalls  int
		wantErr    error
	}{
		{"429 POST is retried", "POST", false, []resp{{status: 429}, {status: 201, body: map[string]any{"id": "x"}}}, 2, nil},
		{"503 GET is retried", "GET", false, []resp{{status: 503}, {status: 200, body: map[string]any{}}}, 2, nil},
		{"503 POST fails at once", "POST", false, []resp{{status: 503}, {status: 200}}, 1, ErrAPI},
		{"transport failure on POST is not replayed", "POST", false, []resp{{err: timeout}, {status: 200}}, 1, ErrTransport},
		{"transport failure on GET is replayed", "GET", false, []resp{{err: timeout}, {status: 200, body: map[string]any{}}}, 2, nil},
		{"idempotent POST 503 then 200", "POST", true, []resp{{status: 503}, {status: 200, body: map[string]any{}}}, 2, nil},
		{"idempotent POST timeout then success", "POST", true, []resp{{err: timeout}, {status: 200, body: map[string]any{}}}, 2, nil},
		{"403 is never retried", "GET", false, []resp{{status: 403}, {status: 200}}, 1, ErrAPI},
		{"retries are bounded", "GET", false, []resp{{status: 503}}, defaultMaxAttempts, ErrAPI},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			e, c, _ := apiFixture(t)
			e.route(tc.method, "x", tc.answers...)
			var body any
			if tc.method == "POST" {
				body = map[string]any{"b": 1}
			}
			v, err := c.request(context.Background(), tc.method, "x", nil, body, tc.idempotent)
			if got := len(e.requests()); got != tc.wantCalls {
				t.Fatalf("requests = %d, want %d", got, tc.wantCalls)
			}
			if tc.wantErr == nil && (err != nil || v == nil) {
				t.Fatalf("want success body, got %v %v", v, err)
			}
			if tc.wantErr != nil && !errors.Is(err, tc.wantErr) {
				t.Fatalf("err = %v, want %v", err, tc.wantErr)
			}
		})
	}
}

func TestAPIMaxAttemptsConfigurable(t *testing.T) {
	e, c, _ := apiFixture(t)
	c.maxAttempts = 2
	e.route("GET", "x", resp{status: 500})
	_, _ = c.get(context.Background(), "x")
	if n := len(e.requests()); n != 2 {
		t.Fatalf("requests = %d, want 2", n)
	}
}

func TestAPIRetryAfterAndBackoff(t *testing.T) {
	e, c, delays := apiFixture(t)
	now := time.Date(2026, 10, 13, 12, 0, 0, 0, time.UTC)
	c.now = func() time.Time { return now }
	date := now.Add(7 * time.Second).Format(http.TimeFormat)
	e.route("GET", "x",
		resp{status: 429, header: http.Header{"Retry-After": {"3"}}},
		resp{status: 503, header: http.Header{"Retry-After": {date}}},
		resp{status: 503, header: http.Header{"Retry-After": {"soon"}}},
		resp{status: 200, body: map[string]any{}})
	if _, err := c.get(context.Background(), "x"); err != nil {
		t.Fatal(err)
	}
	// attempt 2 has no usable header: 0.5s*2^2 = 2s, plus jitter 0.25*min(1s, 2s).
	want := []time.Duration{3 * time.Second, 7 * time.Second, 2*time.Second + 250*time.Millisecond}
	if len(*delays) != len(want) {
		t.Fatalf("delays = %v", *delays)
	}
	for i := range want {
		if (*delays)[i] != want[i] {
			t.Fatalf("delays = %v, want %v", *delays, want)
		}
	}
	if d := c.retryDelay(10, nil); d != maxBackoff+250*time.Millisecond {
		t.Fatalf("capped delay = %v", d)
	}
}

func TestAPIContextCancelStopsRetries(t *testing.T) {
	e, c, _ := apiFixture(t)
	e.route("GET", "x", resp{status: 503})
	ctx, cancel := context.WithCancel(context.Background())
	c.sleep = func(ctx context.Context, d time.Duration) error { cancel(); return ctx.Err() }
	_, err := c.get(ctx, "x")
	if !errors.Is(err, context.Canceled) || len(e.requests()) != 1 {
		t.Fatalf("err %v after %d requests", err, len(e.requests()))
	}
}

func TestSegmentEncoding(t *testing.T) {
	if got := segment("a/b c?#%é:@"); got != "a%2Fb%20c%3F%23%25%C3%A9%3A%40" {
		t.Fatalf("segment = %s", got)
	}
}

// TestAPIRealHTTPTimeout drives net/http end to end against a loopback
// httptest server: the per-attempt timeout fires and the GET is retried.
func TestAPIRealHTTPTimeout(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 {
			select {
			case <-r.Context().Done():
			case <-time.After(2 * time.Second):
			}
			return
		}
		if r.URL.EscapedPath() != "/api/v2/projects/p%2Fq/x" || r.Header.Get("Authorization") != "tok" {
			w.WriteHeader(400)
			return
		}
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer srv.Close()
	target, _ := url.Parse(srv.URL)
	rt := roundTripFunc(func(r *http.Request) (*http.Response, error) {
		if r.URL.Host != "app.launchdarkly.com" {
			t.Errorf("request went to %s", r.URL.Host)
		}
		r = r.Clone(r.Context())
		r.URL.Scheme, r.URL.Host = target.Scheme, target.Host
		return http.DefaultTransport.RoundTrip(r)
	})
	c := newAPIClient("tok", &http.Client{Transport: rt}, 50*time.Millisecond, 3)
	c.sleep = func(context.Context, time.Duration) error { return nil }
	v, err := c.get(context.Background(), "projects/"+segment("p/q")+"/x")
	if err != nil || v.(map[string]any)["ok"] != true || calls.Load() != 2 {
		t.Fatalf("v=%v err=%v calls=%d", v, err, calls.Load())
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }
