package llms

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"strings"
	"testing"
)

var bg = context.Background()

// fakeServer records each request body and path, and answers with the
// next reply (the last one repeats).
type fakeServer struct {
	bodies  []map[string]any
	paths   []string
	headers []http.Header
}

func newFake(t *testing.T, contentType string, replies ...string) (*fakeServer, string) {
	t.Helper()
	f := &fakeServer{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		var m map[string]any
		_ = json.Unmarshal(b, &m)
		f.bodies = append(f.bodies, m)
		f.paths = append(f.paths, r.URL.Path)
		f.headers = append(f.headers, r.Header.Clone())
		w.Header().Set("Content-Type", contentType)
		i := min(len(f.bodies)-1, len(replies)-1)
		_, _ = io.WriteString(w, replies[i])
	}))
	t.Cleanup(srv.Close)
	return f, srv.URL
}

// unsetenv removes key for the test and restores it afterwards.
func unsetenv(t *testing.T, key string) {
	t.Helper()
	t.Setenv(key, "") // registers the restore
	if err := os.Unsetenv(key); err != nil {
		t.Fatal(err)
	}
}

const openAIReply = `{"id":"c1","object":"chat.completion","created":1,"model":"gpt-4o","choices":[{"index":0,
"message":{"role":"assistant","content":"hi","tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"q\":\"x\"}"}}]},
"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":7,"completion_tokens":3,"total_tokens":10,"prompt_tokens_details":{"cached_tokens":2}}}`

func newTestOpenAI(t *testing.T, url string, schema map[string]any) *OpenAI {
	t.Helper()
	t.Setenv("OPENAI_BASE_URL", "")
	t.Setenv("OPENAI_ORG_ID", "")
	m, err := NewOpenAI(OpenAIOptions{APIKey: "k", BaseURL: url, Model: "gpt-4o", JSONSchema: schema})
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func TestOpenAIRequest(t *testing.T) {
	f, url := newFake(t, "application/json", openAIReply)
	m := newTestOpenAI(t, url, nil)
	msgs := []MessageContent{
		TextParts(ChatMessageTypeSystem, "be brief"),
		{Role: ChatMessageTypeHuman, Parts: []ContentPart{TextContent{Text: "look"}, BinaryContent{MIMEType: "image/png", Data: []byte{1, 2}}}},
		{Role: ChatMessageTypeAI, Parts: []ContentPart{ToolCall{ID: "call_1", Type: "function", FunctionCall: &FunctionCall{Name: "lookup", Arguments: `{"q":"x"}`}}}},
		{Role: ChatMessageTypeTool, Parts: []ContentPart{ToolCallResponse{ToolCallID: "call_1", Name: "lookup", Content: "found"}}},
	}
	_, err := m.GenerateContent(bg, msgs, WithTemperature(0), WithMaxTokens(50), WithSeed(0), WithStopWords([]string{"END"}),
		WithTools([]Tool{{Type: "function", Function: &FunctionDefinition{Name: "lookup", Description: "d", Parameters: map[string]any{"type": "object"}}}}))
	if err != nil {
		t.Fatal(err)
	}
	if f.paths[0] != "/chat/completions" {
		t.Errorf("path = %q", f.paths[0])
	}
	b := f.bodies[0]
	// Explicit zeros are sent; max tokens is max_completion_tokens.
	if b["temperature"] != 0.0 || b["seed"] != 0.0 || b["max_completion_tokens"] != 50.0 || b["max_tokens"] != nil {
		t.Errorf("sampling fields = temperature %v seed %v max_completion_tokens %v max_tokens %v", b["temperature"], b["seed"], b["max_completion_tokens"], b["max_tokens"])
	}
	if _, ok := b["top_p"]; ok {
		t.Error("unset top_p was sent")
	}
	if !reflect.DeepEqual(b["stop"], []any{"END"}) {
		t.Errorf("stop = %v", b["stop"])
	}
	got := b["messages"].([]any)
	want := []any{
		map[string]any{"role": "system", "content": "be brief"},
		map[string]any{"role": "user", "content": []any{
			map[string]any{"type": "text", "text": "look"},
			map[string]any{"type": "image_url", "image_url": map[string]any{"url": "data:image/png;base64,AQI="}},
		}},
		map[string]any{"role": "assistant", "content": "", "tool_calls": []any{
			map[string]any{"id": "call_1", "type": "function", "function": map[string]any{"name": "lookup", "arguments": `{"q":"x"}`}},
		}},
		map[string]any{"role": "tool", "tool_call_id": "call_1", "content": "found"},
	}
	if !reflect.DeepEqual(got, want) {
		gb, _ := json.Marshal(got)
		t.Errorf("messages =\n%s", gb)
	}
	tool := b["tools"].([]any)[0].(map[string]any)["function"].(map[string]any)
	// An object schema without properties gets an empty properties map.
	if !reflect.DeepEqual(tool["parameters"], map[string]any{"type": "object", "properties": map[string]any{}}) || tool["description"] != "d" {
		t.Errorf("tool = %v", tool)
	}
	if f.headers[0].Get("Authorization") != "Bearer k" {
		t.Errorf("auth header = %q", f.headers[0].Get("Authorization"))
	}
}

func TestOpenAIResponse(t *testing.T) {
	_, url := newFake(t, "application/json", openAIReply)
	resp, err := newTestOpenAI(t, url, nil).GenerateContent(bg, []MessageContent{TextParts(ChatMessageTypeHuman, "q")})
	if err != nil {
		t.Fatal(err)
	}
	c := resp.Choices[0]
	if c.Content != "hi" || c.StopReason != "tool_calls" || len(c.ToolCalls) != 1 || c.ToolCalls[0].FunctionCall.Arguments != `{"q":"x"}` {
		t.Errorf("choice = %+v", c)
	}
	if c.GenerationInfo["PromptTokens"] != 7 || c.GenerationInfo["CompletionTokens"] != 3 || c.GenerationInfo["PromptCachedTokens"] != 2 {
		t.Errorf("usage = %v", c.GenerationInfo)
	}
	if am := resp.AssistantMessage(); len(am.Parts) != 2 {
		t.Errorf("assistant message parts = %v", am.Parts)
	}
}

func TestOpenAIResponseFormats(t *testing.T) {
	schema := map[string]any{"type": "object", "properties": map[string]any{"a": map[string]any{"anyOf": []any{map[string]any{"type": "string"}}}}}
	f, url := newFake(t, "application/json", openAIReply)
	if _, err := newTestOpenAI(t, url, schema).GenerateContent(bg, []MessageContent{TextParts(ChatMessageTypeHuman, "q")}, WithJSONMode()); err != nil {
		t.Fatal(err)
	}
	rf := f.bodies[0]["response_format"].(map[string]any)
	js := rf["json_schema"].(map[string]any)
	// The schema wins over JSON mode and is sent verbatim (anyOf survives).
	if rf["type"] != "json_schema" || js["name"] != "output" || js["strict"] != false || !reflect.DeepEqual(js["schema"], schema) {
		t.Errorf("response_format = %v", rf)
	}
	f2, url2 := newFake(t, "application/json", openAIReply)
	if _, err := newTestOpenAI(t, url2, nil).GenerateContent(bg, []MessageContent{TextParts(ChatMessageTypeHuman, "q")}, WithJSONMode()); err != nil {
		t.Fatal(err)
	}
	if rf := f2.bodies[0]["response_format"].(map[string]any); rf["type"] != "json_object" {
		t.Errorf("json mode response_format = %v", rf)
	}
}

func TestOpenAIEmptyChoices(t *testing.T) {
	_, url := newFake(t, "application/json", `{"id":"c1","object":"chat.completion","created":1,"model":"gpt-4o","choices":[]}`)
	if _, err := newTestOpenAI(t, url, nil).GenerateContent(bg, []MessageContent{TextParts(ChatMessageTypeHuman, "q")}); !errors.Is(err, ErrEmptyResponse) {
		t.Fatalf("err = %v, want ErrEmptyResponse", err)
	}
}

func sse(events ...string) string {
	var b strings.Builder
	for _, e := range events {
		b.WriteString("data: " + e + "\n\n")
	}
	b.WriteString("data: [DONE]\n\n")
	return b.String()
}

func TestOpenAIStreaming(t *testing.T) {
	body := sse(
		`{"id":"c","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"}}]}`,
		`{"id":"c","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"lo"}}]}`,
		`{"id":"c","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"q\":"}}]}}]}`,
		`{"id":"c","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"x\"}"}}]}}]}`,
		`{"id":"c","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		`{"id":"c","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[],"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}}`,
	)
	f, url := newFake(t, "text/event-stream", body)
	var chunks []string
	resp, err := newTestOpenAI(t, url, nil).GenerateContent(bg, []MessageContent{TextParts(ChatMessageTypeHuman, "q")},
		WithStreamingFunc(func(_ context.Context, c []byte) error { chunks = append(chunks, string(c)); return nil }))
	if err != nil {
		t.Fatal(err)
	}
	if f.bodies[0]["stream"] != true || !reflect.DeepEqual(f.bodies[0]["stream_options"], map[string]any{"include_usage": true}) {
		t.Errorf("stream fields = %v %v", f.bodies[0]["stream"], f.bodies[0]["stream_options"])
	}
	// Only text reaches the callback; tool calls are merged by index.
	if !reflect.DeepEqual(chunks, []string{"Hel", "lo"}) {
		t.Errorf("chunks = %q", chunks)
	}
	c := resp.Choices[0]
	if c.Content != "Hello" || c.StopReason != "tool_calls" || len(c.ToolCalls) != 1 || c.ToolCalls[0].FunctionCall.Arguments != `{"q":"x"}` {
		t.Errorf("choice = %+v tool calls %+v", c, c.ToolCalls)
	}
	if c.GenerationInfo["PromptTokens"] != 4 || c.GenerationInfo["CompletionTokens"] != 2 {
		t.Errorf("usage = %v", c.GenerationInfo)
	}
}

func TestOpenAIStreamingFuncErrorStops(t *testing.T) {
	_, url := newFake(t, "text/event-stream", sse(
		`{"id":"c","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"a"}}]}`,
		`{"id":"c","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"b"}}]}`))
	stop := errors.New("stop")
	calls := 0
	_, err := newTestOpenAI(t, url, nil).GenerateContent(bg, []MessageContent{TextParts(ChatMessageTypeHuman, "q")},
		WithStreamingFunc(func(context.Context, []byte) error { calls++; return stop }))
	if !errors.Is(err, stop) || calls != 1 {
		t.Fatalf("err = %v after %d calls, want the callback's error after 1", err, calls)
	}
}

func TestOpenAIEnvironment(t *testing.T) {
	f, url := newFake(t, "application/json", openAIReply)
	t.Setenv("OPENAI_API_KEY", "envkey")
	t.Setenv("OPENAI_BASE_URL", "")
	t.Setenv("OPENAI_API_BASE", url+"/")
	t.Setenv("OPENAI_ORGANIZATION", "org-1")
	t.Setenv("OPENAI_ORG_ID", "")
	t.Setenv("OPENAI_MODEL", "gpt-env")
	// OPENAI_API_BASE applies only when OPENAI_BASE_URL is unset (not just empty).
	unsetenv(t, "OPENAI_BASE_URL")
	mm, err := NewOpenAI(OpenAIOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := mm.GenerateContent(bg, []MessageContent{TextParts(ChatMessageTypeHuman, "q")}); err != nil {
		t.Fatal(err)
	}
	if f.bodies[0]["model"] != "gpt-env" || f.headers[0].Get("Authorization") != "Bearer envkey" || f.headers[0].Get("OpenAI-Organization") != "org-1" {
		t.Errorf("model %v auth %q org %q", f.bodies[0]["model"], f.headers[0].Get("Authorization"), f.headers[0].Get("OpenAI-Organization"))
	}
	t.Setenv("OPENAI_API_KEY", "")
	if _, err := NewOpenAI(OpenAIOptions{}); !errors.Is(err, ErrMissingOpenAIKey) {
		t.Errorf("missing key err = %v", err)
	}
}
