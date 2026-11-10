package langchainmessages

import (
	"encoding/json"
	"io"
	"maps"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/neverforward"
)

const (
	anthropicJSON = `{"id":"m","type":"message","role":"assistant","model":"m","stop_reason":"end_turn",
"content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":1,"output_tokens":1}}`
	anthropicSSE = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"m\",\"content\":[],\"stop_reason\":null,\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n" +
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\n" +
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n" +
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
	openaiJSON = `{"id":"c","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,
"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`
	openaiSSE = "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"ok\"},\"finish_reason\":null}]}\n\n" +
		"data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}\n\n" +
		"data: [DONE]\n\n"
)

// wireRequest runs one turn through the built-in model for provider against a
// fake API, and returns the request body without the fields that only say
// whether it streams.
func wireRequest(t *testing.T, provider string, params map[string]any, stream bool) map[string]any {
	t.Helper()
	var mu sync.Mutex
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		_ = json.Unmarshal(b, &body)
		streaming, _ := body["stream"].(bool)
		mu.Unlock()
		resp := map[[2]bool]string{{true, false}: anthropicJSON, {true, true}: anthropicSSE, {false, false}: openaiJSON, {false, true}: openaiSSE}[[2]bool{provider == "Anthropic", streaming}]
		if streaming {
			w.Header().Set("Content-Type", "text/event-stream")
		} else {
			w.Header().Set("Content-Type", "application/json")
		}
		_, _ = io.WriteString(w, resp)
	}))
	t.Cleanup(srv.Close)
	t.Setenv("ANTHROPIC_BASE_URL", srv.URL)
	t.Setenv("ANTHROPIC_API_KEY", "test")
	t.Setenv("OPENAI_BASE_URL", srv.URL)
	t.Setenv("OPENAI_API_KEY", "test")
	withTracer(t)
	h := must(newHandler(Options{}))
	req := aiserver.HandlerRequest{Config: cfg(func(c *aiserver.AiConfig) {
		c.Provider.Name = provider
		c.Model = aiserver.Model{Name: "m", Parameters: params}
		c.Instructions = "sys"
	}), UserInput: "q"}
	if stream {
		for _, err := range h.Stream(bg, req) {
			if err != nil {
				t.Fatal(err)
			}
		}
	} else if _, err := h.Invoke(bg, req); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	defer mu.Unlock()
	delete(body, "stream")
	delete(body, "stream_options")
	return body
}

var paths = []struct {
	name   string
	stream bool
}{{"invoke", false}, {"stream", true}}

// None of the never-forwarded keys reaches the provider request, a known
// setting does, and invoke and stream send the same request.
func TestNeverForwardedParameters(t *testing.T) {
	params := maps.Clone(neverforward.Params)
	params["temperature"] = 0.3
	for _, provider := range []string{"Anthropic", "OpenAI"} {
		t.Run(provider, func(t *testing.T) {
			var bodies []map[string]any
			for _, p := range paths {
				body := wireRequest(t, provider, params, p.stream)
				raw, _ := json.Marshal(body)
				if strings.Contains(string(raw), neverforward.Marker) {
					t.Errorf("%s: a never-forwarded value reached the request: %s", p.name, raw)
				}
				if body["temperature"] != 0.3 {
					t.Errorf("%s: temperature = %v", p.name, body["temperature"])
				}
				bodies = append(bodies, body)
			}
			if !reflect.DeepEqual(bodies[0], bodies[1]) {
				t.Errorf("invoke and stream differ:\n%v\n%v", bodies[0], bodies[1])
			}
		})
	}
}

// An output_config.format never reaches Anthropic: it would be a second
// structured-output schema, and the handler owns structured output.
func TestNoSecondStructuredOutputSchema(t *testing.T) {
	params := map[string]any{"output_config": map[string]any{"effort": "low",
		"format": map[string]any{"type": "json_schema", "schema": map[string]any{"type": "object"}}}}
	for _, p := range paths {
		body := wireRequest(t, "Anthropic", params, p.stream)
		if oc, ok := body["output_config"].(map[string]any); ok && oc["format"] != nil {
			t.Errorf("%s: output_config = %v", p.name, oc)
		}
	}
}
