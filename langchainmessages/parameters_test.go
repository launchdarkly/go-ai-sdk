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
	geminiJSON = `{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1}}`
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
		if provider == "Gemini" {
			// Gemini streams by path, not by a body field.
			streaming = strings.HasSuffix(r.URL.Path, ":streamGenerateContent")
			resp = geminiJSON
			if streaming {
				resp = "data: " + geminiJSON + "\n\n"
			}
		}
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
	t.Setenv("GOOGLE_GEMINI_BASE_URL", srv.URL)
	t.Setenv("GOOGLE_API_KEY", "test")
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
	for _, provider := range []string{"Anthropic", "OpenAI", "Gemini"} {
		t.Run(provider, func(t *testing.T) {
			var bodies []map[string]any
			for _, p := range paths {
				body := wireRequest(t, provider, params, p.stream)
				raw, _ := json.Marshal(body)
				if strings.Contains(string(raw), neverforward.Marker) {
					t.Errorf("%s: a never-forwarded value reached the request: %s", p.name, raw)
				}
				temperature := body["temperature"]
				if provider == "Gemini" {
					temperature, _ = body["generationConfig"].(map[string]any)["temperature"]
				}
				if temperature != 0.3 {
					t.Errorf("%s: temperature = %v", p.name, temperature)
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

// The Gemini parameter table (TESTING.md §1.12): max_output_tokens wins over
// max_tokens, stop_sequences over stop, thinking_level is sent upper case and
// wins over thinking_budget, and a value outside a key's allowed set is
// dropped. Invoke and stream send the same generationConfig.
func TestGeminiParameterTable(t *testing.T) {
	for name, tc := range map[string]struct {
		params map[string]any
		want   map[string]any
	}{
		"max_output_tokens wins": {map[string]any{"max_tokens": 10.0, "max_output_tokens": 20.0}, map[string]any{"maxOutputTokens": 20.0}},
		"max_tokens renamed":     {map[string]any{"max_tokens": 10.0}, map[string]any{"maxOutputTokens": 10.0}},
		"stop_sequences wins":    {map[string]any{"stop": []any{"a"}, "stop_sequences": []any{"b"}}, map[string]any{"stopSequences": []any{"b"}}},
		"thinking_level":         {map[string]any{"thinking_level": "low"}, map[string]any{"thinkingConfig": map[string]any{"thinkingLevel": "LOW"}}},
		"thinking_level wins": {map[string]any{"thinking_level": "High", "thinking_budget": 512.0},
			map[string]any{"thinkingConfig": map[string]any{"thinkingLevel": "HIGH"}}},
		"invalid level falls back to the budget": {map[string]any{"thinking_level": "max", "thinking_budget": 512.0},
			map[string]any{"thinkingConfig": map[string]any{"thinkingBudget": 512.0}}},
		"thinking_budget":            {map[string]any{"thinking_budget": -1.0}, map[string]any{"thinkingConfig": map[string]any{"thinkingBudget": -1.0}}},
		"fractional budget dropped":  {map[string]any{"thinking_budget": 1.5}, map[string]any{}},
		"string budget dropped":      {map[string]any{"thinking_budget": "512"}, map[string]any{}},
		"nested thinking_config":     {map[string]any{"thinking_config": map[string]any{"thinking_budget": 5.0}, "include_thoughts": true}, map[string]any{}},
		"media_resolution":           {map[string]any{"media_resolution": "media_resolution_high"}, map[string]any{"mediaResolution": "MEDIA_RESOLUTION_HIGH"}},
		"invalid media_resolution":   {map[string]any{"media_resolution": "ultra"}, map[string]any{}},
		"camelCase keys are dropped": {map[string]any{"maxOutputTokens": 5.0, "thinkingLevel": "low", "mediaResolution": "MEDIA_RESOLUTION_LOW"}, map[string]any{}},
	} {
		t.Run(name, func(t *testing.T) {
			var configs []map[string]any
			for _, p := range paths {
				gc, _ := wireRequest(t, "Gemini", tc.params, p.stream)["generationConfig"].(map[string]any)
				if gc == nil {
					gc = map[string]any{}
				}
				if !reflect.DeepEqual(gc, tc.want) {
					t.Errorf("%s: generationConfig = %v, want %v", p.name, gc, tc.want)
				}
				configs = append(configs, gc)
			}
			if !reflect.DeepEqual(configs[0], configs[1]) {
				t.Errorf("invoke and stream differ: %v, %v", configs[0], configs[1])
			}
		})
	}
}

// The Gemini-only keys never reach another route.
func TestGeminiOnlyParametersStayOnGemini(t *testing.T) {
	params := map[string]any{"max_output_tokens": 77.0, "thinking_level": "low", "thinking_budget": 5.0, "media_resolution": "MEDIA_RESOLUTION_LOW"}
	for _, provider := range []string{"Anthropic", "OpenAI"} {
		raw, _ := json.Marshal(wireRequest(t, provider, params, false))
		for _, leak := range []string{"77", "LOW", "low", "thinking", "media"} {
			if strings.Contains(string(raw), leak) {
				t.Errorf("%s: a Gemini-only key reached the request (%q): %s", provider, leak, raw)
			}
		}
	}
}
