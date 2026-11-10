package langchainmessages

import (
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/launchdarkly/go-sdk-common/v3/ldvalue"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/llms"
)

// retries returns a pointer to n, for Options.MaxRetries.
func retries(n int) *int { return &n }

// failingProvider answers every request with a retryable 500 and a 1 ms retry
// hint, and counts requests: one call makes 1 + its retry count. One server
// stands in for OpenAI, Anthropic and Bedrock Runtime.
type failingProvider struct{ n atomic.Int32 }

func (f *failingProvider) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	_, _ = io.Copy(io.Discard, r.Body)
	f.n.Add(1)
	w.Header().Set("Retry-After-Ms", "1")
	w.Header().Set("X-Amzn-ErrorType", "InternalServerException")
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusInternalServerError)
	_, _ = io.WriteString(w, `{"message":"down","error":{"message":"down","type":"server_error"}}`)
}

// failingEnv points every model the handler builds at a failing server.
func failingEnv(t *testing.T) *failingProvider {
	t.Helper()
	f := &failingProvider{}
	srv := httptest.NewServer(f)
	t.Cleanup(srv.Close)
	for k, v := range map[string]string{
		"OPENAI_BASE_URL": srv.URL, "OPENAI_API_KEY": "test",
		"ANTHROPIC_BASE_URL": srv.URL, "ANTHROPIC_API_KEY": "test",
		"AWS_ENDPOINT_URL_BEDROCK_RUNTIME": srv.URL, "AWS_REGION": "us-east-1",
		"AWS_ACCESS_KEY_ID": "test", "AWS_SECRET_ACCESS_KEY": "test",
		"AWS_CONFIG_FILE":             filepath.Join(t.TempDir(), "none"),
		"AWS_SHARED_CREDENTIALS_FILE": filepath.Join(t.TempDir(), "none"),
		"AWS_EC2_METADATA_DISABLED":   "true",
	} {
		t.Setenv(k, v)
	}
	// A bearer token needs HTTPS; the fake server is plain HTTP, so sign with SigV4.
	t.Setenv("AWS_BEARER_TOKEN_BEDROCK", "")
	os.Unsetenv("AWS_BEARER_TOKEN_BEDROCK")
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

func providerCfg(provider string, params map[string]any) aiserver.AiConfig {
	return cfg(func(c *aiserver.AiConfig) { c.Provider.Name = provider; c.Model.Parameters = params })
}

// callOnce runs one turn through Invoke or Stream. Every request fails, so the
// turn's error is expected and ignored.
func callOnce(h *aiserver.ProviderHandler, c aiserver.AiConfig, stream bool) {
	req := aiserver.HandlerRequest{Config: c, UserInput: "q"}
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

func wantRequests(t *testing.T, f *failingProvider, want int32) {
	t.Helper()
	if got := f.n.Load(); got != want {
		t.Errorf("requests = %d, want %d", got, want)
	}
}

// routes are the built-in models the handler builds from config.provider.name.
// Bedrock is slower: the AWS retryer's backoff ignores the fake's retry hint,
// so its counts stay small. Do not "fix" the fake for it.
var routes = []string{"OpenAI", "Anthropic", "Bedrock"}

func TestMaxRetriesReachesTheModel(t *testing.T) {
	for _, route := range routes {
		t.Run(route, func(t *testing.T) {
			n := 7
			if route == "Bedrock" {
				n = 1
			}
			f := failingEnv(t)
			callOnce(mustHandler(t, Options{MaxRetries: retries(n)}), providerCfg(route, nil), false)
			wantRequests(t, f, int32(n)+1)
		})
	}
}

// Unset, the option is not passed, so each library's own default applies:
// openai-go and anthropic-sdk-go retry twice (3 requests); the AWS SDK makes
// 3 attempts in all.
func TestMaxRetriesUnsetKeepsTheVendorDefault(t *testing.T) {
	for _, route := range routes {
		t.Run(route, func(t *testing.T) {
			f := failingEnv(t)
			callOnce(mustHandler(t, Options{}), providerCfg(route, nil), false)
			wantRequests(t, f, 3)
		})
	}
}

// 0 means exactly one attempt on every route. On Bedrock it must not reach the
// AWS SDK as a max-attempts value of 0, which the SDK reads as its default.
func TestMaxRetriesZeroIsHonoured(t *testing.T) {
	for _, route := range routes {
		t.Run(route, func(t *testing.T) {
			f := failingEnv(t)
			callOnce(mustHandler(t, Options{MaxRetries: retries(0)}), providerCfg(route, nil), false)
			wantRequests(t, f, 1)
		})
	}
}

func TestMaxRetriesNegativeIsRefusedAtConstruction(t *testing.T) {
	_, err := NewHandler(Options{MaxRetries: retries(-1)})
	if err == nil || !strings.Contains(err.Error(), "MaxRetries") {
		t.Fatalf("NewHandler(MaxRetries: -1) error = %v, want one naming MaxRetries", err)
	}
}

func TestMaxRetriesModelParametersCannotSetIt(t *testing.T) {
	f := failingEnv(t)
	callOnce(mustHandler(t, Options{MaxRetries: retries(3)}), providerCfg("OpenAI", map[string]any{"max_retries": 50}), false)
	wantRequests(t, f, 4)
}

func TestMaxRetriesInvokeAndStreamAgree(t *testing.T) {
	for _, stream := range []bool{false, true} {
		f := failingEnv(t)
		callOnce(mustHandler(t, Options{MaxRetries: retries(2)}), providerCfg("OpenAI", nil), stream)
		if got := f.n.Load(); got != 3 {
			t.Errorf("stream=%v: requests = %d, want 3", stream, got)
		}
	}
}

// retryLD serves one OpenAI messages variation for every key.
type retryLD struct{}

func (retryLD) JSONVariation(string, ldcontext.Context, ldvalue.Value) (ldvalue.Value, error) {
	return ldvalue.Parse([]byte(`{"model":{"name":"test-model"},"provider":{"name":"OpenAI"},` +
		`"messages":[{"role":"user","content":"{{user_input}}"}],"_ldMeta":{"enabled":true,"variationKey":"v"}}`)), nil
}
func (retryLD) TrackMetric(string, ldcontext.Context, float64, ldvalue.Value) error { return nil }
func (retryLD) Flush()                                                              {}
func (retryLD) Close() error                                                        { return nil }

func installRetryLD(t *testing.T) {
	t.Helper()
	_ = aiserver.Shutdown(bg)
	if _, err := aiserver.InitWithClient(bg, retryLD{}, aiserver.InitOptions{OTLPEndpoint: "http://127.0.0.1:1/v1/traces"}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = aiserver.Shutdown(bg) })
}

func TestMaxRetriesConvenienceExportForwardsIt(t *testing.T) {
	f := failingEnv(t)
	installRetryLD(t)
	_, _ = LangChainMessages(bg, "cfg", "q", ldcontext.New("u"), &InvokeOptions{Handler: Options{MaxRetries: retries(4)}})
	wantRequests(t, f, 5)
}

// The count applies only to a model the handler builds, so passing a model or
// a model factory as well is refused rather than silently ignoring the count.
func TestMaxRetriesWithModelOrFactoryIsRefused(t *testing.T) {
	for name, o := range map[string]Options{
		"Model":        {Model: &fakeModel{}},
		"ModelFactory": {ModelFactory: func(aiserver.AiConfig) (llms.Model, error) { return &fakeModel{}, nil }},
	} {
		t.Run(name, func(t *testing.T) {
			o.MaxRetries = retries(3)
			_, err := NewHandler(o)
			if err == nil || !strings.Contains(err.Error(), "MaxRetries") {
				t.Fatalf("NewHandler(%s + MaxRetries) error = %v, want one naming MaxRetries", name, err)
			}
		})
	}
}

// The convenience function returns the handler's construction error instead
// of dropping it, and makes no request.
func TestMaxRetriesConvenienceExportReturnsTheRefusal(t *testing.T) {
	f := failingEnv(t)
	installRetryLD(t)
	_, err := LangChainMessages(bg, "cfg", "q", ldcontext.New("u"), &InvokeOptions{Handler: Options{MaxRetries: retries(-1)}})
	if err == nil || !strings.Contains(err.Error(), "MaxRetries") {
		t.Fatalf("LangChainMessages(MaxRetries: -1) error = %v, want one naming MaxRetries", err)
	}
	wantRequests(t, f, 0)
}
