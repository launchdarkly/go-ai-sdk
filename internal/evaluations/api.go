package evaluations

import (
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
	"strconv"
	"strings"
	"time"
)

const (
	// APIBaseURI is the fixed LaunchDarkly management API host. It is
	// deliberately not configurable and is unrelated to LD_BASE_URI.
	APIBaseURI = "https://app.launchdarkly.com"
	// UIBaseURI is the fixed host used only to build run links.
	UIBaseURI = "https://app.launchdarkly.com"

	// apiVersion is the LD-API-Version date sent on every request:
	// YYYYMMDD or "beta"; the value matches the Python SDK.
	apiVersion = "20240415"
	userAgent  = "launchdarkly-ai-evaluations-go"

	defaultRequestTimeout = 30 * time.Second
	// defaultMaxAttempts is Python's max_retries=3 plus the first attempt.
	defaultMaxAttempts = 4
	maxBackoff         = 30 * time.Second
)

// APIError is a non-2xx management API response. errors.Is(err, ErrAPI)
// reports true for it.
type APIError struct {
	Status int
	Method string
	Path   string
	Body   string
}

// Error includes the method, path, status and body.
func (e *APIError) Error() string {
	return fmt.Sprintf("LaunchDarkly API %s %s failed with %d: %s", e.Method, e.Path, e.Status, e.Body)
}

// Unwrap makes errors.Is(err, ErrAPI) true.
func (e *APIError) Unwrap() error { return ErrAPI }

// param is one query parameter; a nil value is omitted. A slice keeps
// the caller's order, which url.Values would sort.
type param struct {
	key   string
	value any
}

type apiClient struct {
	token       string
	http        *http.Client
	timeout     time.Duration
	maxAttempts int
	// sleep and random are the injectable delay and jitter sources.
	sleep  func(ctx context.Context, d time.Duration) error
	random func() float64
	now    func() time.Time
}

func newAPIClient(token string, hc *http.Client, timeout time.Duration, maxAttempts int) *apiClient {
	if hc == nil {
		hc = http.DefaultClient
	}
	if timeout <= 0 {
		timeout = defaultRequestTimeout
	}
	if maxAttempts < 1 {
		maxAttempts = defaultMaxAttempts
	}
	return &apiClient{token: token, http: hc, timeout: timeout, maxAttempts: maxAttempts,
		sleep: sleepCtx, random: rand.Float64, now: time.Now}
}

func sleepCtx(ctx context.Context, d time.Duration) error {
	if d <= 0 {
		return ctx.Err()
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

// segment percent-encodes one path segment, escaping everything except
// unreserved characters (Python's quote(value, safe="")).
func segment(s string) string { return strings.ReplaceAll(url.QueryEscape(s), "+", "%20") }

func (c *apiClient) urlFor(path string, params []param) string {
	u := APIBaseURI + "/api/v2/" + strings.TrimPrefix(path, "/")
	var q []string
	for _, p := range params {
		if p.value == nil {
			continue
		}
		q = append(q, url.QueryEscape(p.key)+"="+url.QueryEscape(fmt.Sprint(p.value)))
	}
	if len(q) > 0 {
		u += "?" + strings.Join(q, "&")
	}
	return u
}

func (c *apiClient) get(ctx context.Context, path string, params ...param) (any, error) {
	return c.request(ctx, http.MethodGet, path, params, nil, false)
}

func (c *apiClient) post(ctx context.Context, path string, body any, idempotent bool) (any, error) {
	return c.request(ctx, http.MethodPost, path, nil, body, idempotent)
}

// request sends one call, retrying only where a replay cannot duplicate
// work: 429 always, 5xx and transport failures only for GET/HEAD and POSTs
// the caller marks idempotent.
func (c *apiClient) request(ctx context.Context, method, path string, params []param, body any, idempotent bool) (any, error) {
	var payload []byte
	if body != nil {
		var err error
		if payload, err = json.Marshal(body); err != nil {
			return nil, fmt.Errorf("evaluations: encode %s %s body: %w", method, path, err)
		}
	}
	retrySafe := idempotent || method == http.MethodGet || method == http.MethodHead
	target := c.urlFor(path, params)
	for attempt := 0; ; attempt++ {
		last := attempt+1 >= c.maxAttempts
		status, respBody, header, err := c.send(ctx, method, target, payload)
		if err != nil {
			if ctx.Err() != nil {
				return nil, fmt.Errorf("evaluations: %s %s: %w", method, path, ctx.Err())
			}
			if !retrySafe || last {
				return nil, fmt.Errorf("evaluations: %w: LaunchDarkly API %s %s failed after %d attempt(s): %w",
					ErrTransport, method, path, attempt+1, err)
			}
			if err := c.sleep(ctx, c.retryDelay(attempt, nil)); err != nil {
				return nil, fmt.Errorf("evaluations: %s %s: %w", method, path, err)
			}
			continue
		}
		retryable := status == http.StatusTooManyRequests || (status >= 500 && retrySafe)
		if retryable && !last {
			if err := c.sleep(ctx, c.retryDelay(attempt, header)); err != nil {
				return nil, fmt.Errorf("evaluations: %s %s: %w", method, path, err)
			}
			continue
		}
		if status < 200 || status >= 300 {
			return nil, &APIError{Status: status, Method: method, Path: path, Body: string(respBody)}
		}
		if len(respBody) == 0 {
			return nil, nil
		}
		var out any
		if err := json.Unmarshal(respBody, &out); err != nil {
			return nil, fmt.Errorf("evaluations: %w: LaunchDarkly API %s %s returned invalid JSON: %w",
				ErrInvalidResponse, method, path, err)
		}
		return out, nil
	}
}

func (c *apiClient) send(ctx context.Context, method, target string, payload []byte) (int, []byte, http.Header, error) {
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()
	var body io.Reader
	if payload != nil {
		body = bytes.NewReader(payload)
	}
	req, err := http.NewRequestWithContext(ctx, method, target, body)
	if err != nil {
		return 0, nil, nil, err
	}
	req.Header.Set("Authorization", c.token)
	req.Header.Set("Accept", "application/json")
	req.Header.Set("LD-API-Version", apiVersion)
	req.Header.Set("User-Agent", userAgent)
	if payload != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return 0, nil, nil, err
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, nil, nil, err
	}
	return resp.StatusCode, b, resp.Header, nil
}

// retryDelay honours Retry-After (seconds or an HTTP date), otherwise uses
// capped exponential backoff plus jitter. Constants are Python's.
func (c *apiClient) retryDelay(attempt int, header http.Header) time.Duration {
	if h := strings.TrimSpace(header.Get("Retry-After")); h != "" {
		if secs, err := strconv.ParseFloat(h, 64); err == nil && !math.IsNaN(secs) && !math.IsInf(secs, 0) {
			return max(0, time.Duration(secs*float64(time.Second)))
		}
		if when, err := http.ParseTime(h); err == nil {
			return max(0, when.Sub(c.now()))
		}
	}
	exp := min(maxBackoff, time.Duration(float64(500*time.Millisecond)*math.Pow(2, float64(attempt))))
	jitter := time.Duration(c.random() * float64(min(time.Second, exp)))
	return exp + jitter
}

// notFound wraps a 404 as ErrNotFound with a message naming the resource;
// any other error passes through.
func notFound(err error, format string, args ...any) error {
	var apiErr *APIError
	if errors.As(err, &apiErr) && apiErr.Status == http.StatusNotFound {
		return fmt.Errorf("evaluations: %w: "+format+": %w", append(append([]any{ErrNotFound}, args...), err)...)
	}
	return err
}

func asObject(v any, what string) (map[string]any, error) {
	m, ok := v.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("evaluations: %w: LaunchDarkly returned an invalid %s response", ErrInvalidResponse, what)
	}
	return m, nil
}

func requireString(m map[string]any, key, what string) (string, error) {
	s, ok := m[key].(string)
	if !ok || s == "" {
		return "", fmt.Errorf("evaluations: %w: LaunchDarkly %s response is missing string field %q", ErrInvalidResponse, what, key)
	}
	return s, nil
}

// asInt reports a JSON number that is an integer.
func asInt(v any) (int, bool) {
	f, ok := v.(float64)
	if !ok || f != math.Trunc(f) || math.IsInf(f, 0) {
		return 0, false
	}
	return int(f), true
}
