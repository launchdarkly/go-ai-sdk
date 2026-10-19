package langchainmessages

import (
	"encoding/json"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

// TestBedrockExplicitZeroTemperatureReachesTheModel drives the real handler
// with default Bedrock routing against a local fake Bedrock Converse
// endpoint: an explicit temperature of 0 must be sent, not dropped.
func TestBedrockExplicitZeroTemperatureReachesTheModel(t *testing.T) {
	var mu sync.Mutex
	var bodies []map[string]any
	var paths []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		var m map[string]any
		_ = json.Unmarshal(b, &m)
		mu.Lock()
		bodies = append(bodies, m)
		paths = append(paths, r.URL.Path)
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"output":{"message":{"role":"assistant","content":[{"text":"ok"}]}},"stopReason":"end_turn","usage":{"inputTokens":3,"outputTokens":1,"totalTokens":4},"metrics":{"latencyMs":1}}`)
	}))
	t.Cleanup(srv.Close)
	for k, v := range map[string]string{
		"AWS_ENDPOINT_URL_BEDROCK_RUNTIME": srv.URL, "AWS_REGION": "us-east-1", "AWS_MAX_ATTEMPTS": "1",
		"AWS_ACCESS_KEY_ID": "test", "AWS_SECRET_ACCESS_KEY": "test", "AWS_SESSION_TOKEN": "",
		"AWS_BEARER_TOKEN_BEDROCK": "", "AWS_PROFILE": "", "AWS_CONFIG_FILE": "/nonexistent", "AWS_SHARED_CREDENTIALS_FILE": "/nonexistent",
	} {
		t.Setenv(k, v)
	}
	h := newHandler(Options{})
	for _, temp := range []float64{0, 0.4} {
		cfg := aiserver.AiConfig{Instructions: "judge", Model: aiserver.Model{Name: "anthropic.claude-sonnet-4-5-20250929-v1:0",
			Parameters: map[string]any{"temperature": temp}}}
		cfg.Provider.Name = "Bedrock"
		res, err := h.Invoke(bg, aiserver.HandlerRequest{Config: cfg, UserInput: "score this"})
		if err != nil {
			t.Fatalf("temperature %v: %v", temp, err)
		}
		if res.Output != "ok" || res.Usage["input_tokens"] != float64(3) || res.Usage["output_tokens"] != float64(1) {
			t.Errorf("temperature %v: result %#v usage %v", temp, res.Output, res.Usage)
		}
	}
	if len(bodies) != 2 {
		t.Fatalf("requests = %d", len(bodies))
	}
	for i, p := range paths {
		if !strings.HasSuffix(p, "/converse") {
			t.Errorf("request %d path = %q, want the Converse API", i, p)
		}
	}
	inference := func(i int) map[string]any { m, _ := bodies[i]["inferenceConfig"].(map[string]any); return m }
	if got, ok := inference(0)["temperature"]; !ok || got != 0.0 {
		t.Errorf("explicit temperature 0 was not sent: %v (present=%v)", got, ok)
	}
	if got, _ := inference(1)["temperature"].(float64); math.Abs(got-0.4) > 1e-6 {
		t.Errorf("temperature 0.4 = %v", inference(1)["temperature"])
	}
	if got := inference(0)["maxTokens"]; got != float64(2048) {
		t.Errorf("default maxTokens = %v, want 2048", got)
	}
}
