package langchainmessages

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

// TestBedrockExplicitZeroTemperatureReachesTheModel drives the real handler
// with default Bedrock routing against a local fake Bedrock endpoint.
func TestBedrockExplicitZeroTemperatureReachesTheModel(t *testing.T) {
	var mu sync.Mutex
	var bodies []map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		var m map[string]any
		_ = json.Unmarshal(b, &m)
		mu.Lock()
		bodies = append(bodies, m)
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"m","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":3,"output_tokens":1}}`)
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
		if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: cfg, UserInput: "score this"}); err != nil {
			t.Fatalf("temperature %v: %v", temp, err)
		}
	}
	if len(bodies) != 2 {
		t.Fatalf("requests = %d", len(bodies))
	}
	if got, ok := bodies[0]["temperature"]; !ok || got != 0.0 {
		t.Errorf("explicit temperature 0 was not sent: %v (present=%v)", got, ok)
	}
	if got := bodies[1]["temperature"]; got != 0.4 {
		t.Errorf("temperature 0.4 = %v", got)
	}
}
