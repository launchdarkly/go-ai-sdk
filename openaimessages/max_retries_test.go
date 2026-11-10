package openaimessages

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/openai/openai-go/v3"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

// retries returns a pointer to n, for Options.MaxRetries.
func retries(n int) *int { return &n }

// failingOpenAI answers every request with a retryable 500 and a 1 ms
// retry hint, and counts requests: one call makes 1 + its retry count.
type failingOpenAI struct{ n atomic.Int32 }

func (f *failingOpenAI) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	_, _ = io.Copy(io.Discard, r.Body)
	f.n.Add(1)
	w.Header().Set("Retry-After-Ms", "1")
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusInternalServerError)
	_, _ = io.WriteString(w, `{"error":{"message":"down","type":"server_error"}}`)
}

// failingEnv points the client NewHandler builds at a failing server.
func failingEnv(t *testing.T) *failingOpenAI {
	t.Helper()
	f := &failingOpenAI{}
	srv := httptest.NewServer(f)
	t.Cleanup(srv.Close)
	t.Setenv("OPENAI_BASE_URL", srv.URL)
	t.Setenv("OPENAI_API_KEY", "test")
	return f
}

func mustHandler(t *testing.T, o Options) *aiserver.ProviderHandler {
	t.Helper()
	h, err := NewHandler(o)
	if err != nil {
		t.Fatal(err)
	}
	return h
}

// callOnce runs one turn through Invoke or Stream. Every request fails, so the
// turn's error is expected and ignored.
func callOnce(h *aiserver.ProviderHandler, stream bool, params map[string]any) {
	req := aiserver.HandlerRequest{Config: cfg(func(c *aiserver.AiConfig) { c.Model.Parameters = params }), UserInput: "q"}
	if stream {
		for _, err := range h.Stream(bg, req) {
			if err != nil {
				return
			}
		}
		return
	}
	_, _ = h.Invoke(bg, req)
}

func wantRequests(t *testing.T, f *failingOpenAI, want int32) {
	t.Helper()
	if got := f.n.Load(); got != want {
		t.Errorf("requests = %d, want %d", got, want)
	}
}

func TestMaxRetriesReachesTheClient(t *testing.T) {
	f := failingEnv(t)
	callOnce(mustHandler(t, Options{MaxRetries: retries(7)}), false, nil)
	wantRequests(t, f, 8)
}

func TestMaxRetriesNegativeIsRefusedAtConstruction(t *testing.T) {
	_, err := NewHandler(Options{MaxRetries: retries(-1)})
	if err == nil || !strings.Contains(err.Error(), "MaxRetries") {
		t.Fatalf("NewHandler(MaxRetries: -1) error = %v, want one naming MaxRetries", err)
	}
}

// Unset, the option is not passed, so openai-go's own default applies:
// 2 retries, plus the first attempt.
func TestMaxRetriesUnsetKeepsTheVendorDefault(t *testing.T) {
	f := failingEnv(t)
	callOnce(mustHandler(t, Options{}), false, nil)
	wantRequests(t, f, 3)
}

func TestMaxRetriesZeroIsHonoured(t *testing.T) {
	f := failingEnv(t)
	callOnce(mustHandler(t, Options{MaxRetries: retries(0)}), false, nil)
	wantRequests(t, f, 1)
}

func TestMaxRetriesModelParametersCannotSetIt(t *testing.T) {
	f := failingEnv(t)
	callOnce(mustHandler(t, Options{MaxRetries: retries(3)}), false, map[string]any{"max_retries": 50})
	wantRequests(t, f, 4)
}

func TestMaxRetriesInvokeAndStreamAgree(t *testing.T) {
	for _, stream := range []bool{false, true} {
		f := failingEnv(t)
		callOnce(mustHandler(t, Options{MaxRetries: retries(2)}), stream, nil)
		if got := f.n.Load(); got != 3 {
			t.Errorf("stream=%v: requests = %d, want 3", stream, got)
		}
	}
}

func TestMaxRetriesConvenienceExportForwardsIt(t *testing.T) {
	f := failingEnv(t)
	installClient(t, &fakeLD{variation: variation}, tracetest.NewInMemoryExporter())
	_, _ = OpenAIMessages(bg, "cfg", "q", ldcontext.New("u"), &InvokeOptions{Handler: Options{MaxRetries: retries(4)}})
	wantRequests(t, f, 5)
}

// The count applies only to a client NewHandler builds, so passing a client
// as well is refused rather than silently ignoring the count.
func TestMaxRetriesWithClientIsRefused(t *testing.T) {
	c := openai.NewClient()
	_, err := NewHandler(Options{Client: &c, MaxRetries: retries(3)})
	if err == nil || !strings.Contains(err.Error(), "MaxRetries") {
		t.Fatalf("NewHandler(Client + MaxRetries) error = %v, want one naming MaxRetries", err)
	}
}

// The convenience function returns the handler's construction error instead
// of dropping it, and makes no request.
func TestMaxRetriesConvenienceExportReturnsTheRefusal(t *testing.T) {
	f := failingEnv(t)
	installClient(t, &fakeLD{variation: variation}, tracetest.NewInMemoryExporter())
	_, err := OpenAIMessages(bg, "cfg", "q", ldcontext.New("u"), &InvokeOptions{Handler: Options{MaxRetries: retries(-1)}})
	if err == nil || !strings.Contains(err.Error(), "MaxRetries") {
		t.Fatalf("OpenAIMessages(MaxRetries: -1) error = %v, want one naming MaxRetries", err)
	}
	wantRequests(t, f, 0)
}

// must panics on err, for test fixtures built from valid options.
func must[T any](v T, err error) T {
	if err != nil {
		panic(err)
	}
	return v
}

// newHandler checks the options itself: a negative count reaching it is an
// error, never a vendor SDK panic.
func TestMaxRetriesNegativeIsRefusedByNewHandlerItself(t *testing.T) {
	h, err := newHandler(Options{MaxRetries: retries(-1)})
	if h != nil || err == nil || !strings.Contains(err.Error(), "MaxRetries") {
		t.Fatalf("newHandler(MaxRetries: -1) = %v, %v; want nil and an error naming MaxRetries", h, err)
	}
}
