package llms

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
)

var geminiSignature = []byte("sig-1")

func geminiReply(parts string, finish string) string {
	return `{"candidates":[{"content":{"role":"model","parts":[` + parts + `]},"finishReason":"` + finish + `"}],
"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"thoughtsTokenCount":3,"cachedContentTokenCount":2,"totalTokenCount":18}}`
}

func newTestGemini(t *testing.T, url string, schema map[string]any) *Gemini {
	t.Helper()
	m, err := NewGemini(GeminiOptions{APIKey: "k", BaseURL: url, JSONSchema: schema})
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func TestGeminiRequest(t *testing.T) {
	f, url := newFake(t, "application/json", geminiReply(`{"text":"done"}`, "STOP"))
	msgs := []MessageContent{
		textParts(ChatMessageTypeSystem, "be ", "brief"),
		{Role: ChatMessageTypeHuman, Parts: []ContentPart{TextContent{Text: "look"}, BinaryContent{MIMEType: "image/png", Data: []byte{1, 2}},
			ImageURLContent{URL: "data:image/jpeg;base64,AAA="}}},
		{Role: ChatMessageTypeAI, Parts: []ContentPart{
			ToolCall{ID: "c1", FunctionCall: &FunctionCall{Name: "lookup", Arguments: `{"q":"x"}`}, Signature: geminiSignature},
			ToolCall{ID: geminiLocalIDPrefix + "1", FunctionCall: &FunctionCall{Name: "other", Arguments: ""}},
		}},
		{Role: ChatMessageTypeTool, Parts: []ContentPart{ToolCallResponse{ToolCallID: "c1", Name: "lookup", Content: `{"found":true}`}}},
		{Role: ChatMessageTypeTool, Parts: []ContentPart{ToolCallResponse{ToolCallID: geminiLocalIDPrefix + "1", Name: "other", Content: "plain"}}},
	}
	_, err := newTestGemini(t, url, nil).GenerateContent(bg, msgs, WithModel("gemini-3.1-pro-preview"), WithTemperature(0), WithTopK(5),
		WithMaxTokens(100), WithStopWords([]string{"END"}),
		WithTools([]Tool{{Type: "function", Function: &FunctionDefinition{Name: "lookup", Description: "d",
			Parameters: map[string]any{"type": "object", "properties": map[string]any{"q": map[string]any{"type": "string"}}}}}}))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(f.paths[0], "/models/gemini-3.1-pro-preview:generateContent") || f.headers[0].Get("X-Goog-Api-Key") != "k" {
		t.Errorf("path %q key %q", f.paths[0], f.headers[0].Get("X-Goog-Api-Key"))
	}
	b := f.bodies[0]
	if sys, _ := json.Marshal(b["systemInstruction"]); !strings.Contains(string(sys), `"text":"be brief"`) {
		t.Errorf("systemInstruction = %s", sys)
	}
	gc, _ := b["generationConfig"].(map[string]any)
	// An explicit zero temperature is sent; unset top_p is not.
	if gc["temperature"] != 0.0 || gc["topK"] != 5.0 || gc["maxOutputTokens"] != 100.0 || !reflect.DeepEqual(gc["stopSequences"], []any{"END"}) {
		t.Errorf("generationConfig = %v", gc)
	}
	if _, ok := gc["topP"]; ok {
		t.Error("unset topP was sent")
	}
	decl := b["tools"].([]any)[0].(map[string]any)["functionDeclarations"].([]any)[0].(map[string]any)
	if decl["name"] != "lookup" || decl["parametersJsonSchema"] == nil {
		t.Errorf("function declaration = %v", decl)
	}
	contents := b["contents"].([]any)
	if len(contents) != 3 {
		t.Fatalf("contents = %d, want 3 (both tool results share one user turn)", len(contents))
	}
	user := contents[0].(map[string]any)["parts"].([]any)
	if len(user) != 3 || user[1].(map[string]any)["inlineData"].(map[string]any)["data"] != "AQI=" ||
		user[2].(map[string]any)["inlineData"].(map[string]any)["mimeType"] != "image/jpeg" {
		ub, _ := json.Marshal(user)
		t.Errorf("user parts = %s", ub)
	}
	model := contents[1].(map[string]any)
	calls := model["parts"].([]any)
	first := calls[0].(map[string]any)
	if model["role"] != "model" || first["thoughtSignature"] != base64.StdEncoding.EncodeToString(geminiSignature) ||
		first["functionCall"].(map[string]any)["id"] != "c1" || !reflect.DeepEqual(first["functionCall"].(map[string]any)["args"], map[string]any{"q": "x"}) {
		t.Errorf("first model part = %v (the thought signature must be sent back)", first)
	}
	second := calls[1].(map[string]any)["functionCall"].(map[string]any)
	if _, ok := second["id"]; ok {
		t.Errorf("a made-up tool call id was sent: %v", second)
	}
	results := contents[2].(map[string]any)["parts"].([]any)
	r0 := results[0].(map[string]any)["functionResponse"].(map[string]any)
	r1 := results[1].(map[string]any)["functionResponse"].(map[string]any)
	if r0["name"] != "lookup" || !reflect.DeepEqual(r0["response"], map[string]any{"found": true}) ||
		!reflect.DeepEqual(r1["response"], map[string]any{"output": "plain"}) {
		t.Errorf("function responses = %v, %v", r0, r1)
	}
}

func TestGeminiResponse(t *testing.T) {
	sig := base64.StdEncoding.EncodeToString(geminiSignature)
	_, url := newFake(t, "application/json", geminiReply(
		`{"text":"thinking","thought":true},{"text":"Let me look."},{"functionCall":{"name":"lookup","args":{"z":1,"a":"x"}},"thoughtSignature":"`+sig+`"}`, "STOP"))
	resp, err := newTestGemini(t, url, nil).GenerateContent(bg, []MessageContent{textParts(ChatMessageTypeHuman, "hi")})
	if err != nil {
		t.Fatal(err)
	}
	c := resp.Choices[0]
	if c.Content != "Let me look." || c.StopReason != "STOP" || len(c.ToolCalls) != 1 {
		t.Fatalf("choice = %+v", c)
	}
	tc := c.ToolCalls[0]
	if tc.ID != geminiLocalIDPrefix+"0" || tc.FunctionCall.Arguments != `{"a":"x","z":1}` || !reflect.DeepEqual(tc.Signature, geminiSignature) {
		t.Errorf("tool call = %+v", tc)
	}
	if !reflect.DeepEqual(c.Parts, []ContentPart{TextContent{Text: "Let me look."}, tc}) {
		t.Errorf("parts = %+v (thought parts are dropped)", c.Parts)
	}
	want := map[string]any{"PromptTokens": 10, "CompletionTokens": 8, "PromptCachedTokens": 2}
	if !reflect.DeepEqual(c.GenerationInfo, want) {
		t.Errorf("usage = %v, want %v", c.GenerationInfo, want)
	}
}

// A blocked or truncated candidate has no parts, but its finish reason still
// reaches the chat span; a plain empty STOP is still an empty response.
func TestGeminiResponseWithoutParts(t *testing.T) {
	for _, finish := range []string{"SAFETY", "MAX_TOKENS"} {
		_, url := newFake(t, "application/json", geminiReply(``, finish))
		resp, err := newTestGemini(t, url, nil).GenerateContent(bg, []MessageContent{textParts(ChatMessageTypeHuman, "hi")})
		if err != nil {
			t.Fatalf("%s: %v", finish, err)
		}
		if c := resp.Choices[0]; c.StopReason != finish || c.Content != "" || len(c.Parts) != 0 {
			t.Errorf("%s: choice = %+v", finish, c)
		}
	}
	_, url := newFake(t, "application/json", geminiReply(``, "STOP"))
	if _, err := newTestGemini(t, url, nil).GenerateContent(bg, []MessageContent{textParts(ChatMessageTypeHuman, "hi")}); !errors.Is(err, ErrEmptyResponse) {
		t.Fatalf("err = %v, want ErrEmptyResponse", err)
	}
}

func TestGeminiJSONSchema(t *testing.T) {
	schema := map[string]any{"type": "object", "properties": map[string]any{"a": map[string]any{"type": "string"}}}
	f, url := newFake(t, "application/json", geminiReply(`{"text":"{}"}`, "STOP"))
	if _, err := newTestGemini(t, url, schema).GenerateContent(bg, []MessageContent{textParts(ChatMessageTypeHuman, "hi")}); err != nil {
		t.Fatal(err)
	}
	gc := f.bodies[0]["generationConfig"].(map[string]any)
	if gc["responseMimeType"] != "application/json" || !reflect.DeepEqual(gc["responseJsonSchema"], schema) {
		t.Errorf("generationConfig = %v", gc)
	}
}

func TestGeminiStreaming(t *testing.T) {
	body := "data: " + `{"candidates":[{"content":{"role":"model","parts":[{"text":"Hel"}]}}]}` + "\n\n" +
		"data: " + `{"candidates":[{"content":{"role":"model","parts":[{"text":"lo"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":2}}` + "\n\n"
	f, url := newFake(t, "text/event-stream", body)
	var chunks []string
	resp, err := newTestGemini(t, url, nil).GenerateContent(bg, []MessageContent{textParts(ChatMessageTypeHuman, "hi")},
		WithStreamingFunc(func(_ context.Context, c []byte) error { chunks = append(chunks, string(c)); return nil }))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(f.paths[0], ":streamGenerateContent") {
		t.Errorf("path = %q", f.paths[0])
	}
	if !reflect.DeepEqual(chunks, []string{"Hel", "lo"}) || resp.Choices[0].Content != "Hello" || resp.Choices[0].StopReason != "STOP" {
		t.Errorf("chunks %q response %+v", chunks, resp.Choices[0])
	}
	if !reflect.DeepEqual(resp.Choices[0].Parts, []ContentPart{TextContent{Text: "Hello"}}) {
		t.Errorf("parts = %+v (streamed text is joined)", resp.Choices[0].Parts)
	}
	if resp.Choices[0].GenerationInfo["PromptTokens"] != 4 || resp.Choices[0].GenerationInfo["CompletionTokens"] != 2 {
		t.Errorf("usage = %v", resp.Choices[0].GenerationInfo)
	}
}

func TestGeminiStreamingStops(t *testing.T) {
	body := "data: " + `{"candidates":[{"content":{"role":"model","parts":[{"text":"Hel"}]}}]}` + "\n\n"
	_, url := newFake(t, "text/event-stream", body)
	stop := errors.New("stop")
	_, err := newTestGemini(t, url, nil).GenerateContent(bg, []MessageContent{textParts(ChatMessageTypeHuman, "hi")},
		WithStreamingFunc(func(context.Context, []byte) error { return stop }))
	if !errors.Is(err, stop) {
		t.Errorf("err = %v, want the StreamingFunc error", err)
	}
}

func TestGeminiAPIKey(t *testing.T) {
	unsetenv(t, "GOOGLE_API_KEY")
	unsetenv(t, "GEMINI_API_KEY")
	if _, err := NewGemini(GeminiOptions{}); !errors.Is(err, ErrMissingGeminiKey) {
		t.Errorf("err = %v, want ErrMissingGeminiKey", err)
	}
	t.Setenv("GEMINI_API_KEY", "g")
	f, url := newFake(t, "application/json", geminiReply(`{"text":"hi"}`, "STOP"))
	m, err := NewGemini(GeminiOptions{BaseURL: url})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := m.GenerateContent(bg, []MessageContent{textParts(ChatMessageTypeHuman, "hi")}); err != nil {
		t.Fatal(err)
	}
	if f.headers[0].Get("X-Goog-Api-Key") != "g" {
		t.Errorf("key = %q, want the GEMINI_API_KEY value", f.headers[0].Get("X-Goog-Api-Key"))
	}
	t.Setenv("GOOGLE_API_KEY", "google")
	m, err = NewGemini(GeminiOptions{BaseURL: url})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := m.GenerateContent(bg, []MessageContent{textParts(ChatMessageTypeHuman, "hi")}); err != nil {
		t.Fatal(err)
	}
	if f.headers[1].Get("X-Goog-Api-Key") != "google" {
		t.Errorf("key = %q, want GOOGLE_API_KEY to win", f.headers[1].Get("X-Goog-Api-Key"))
	}
}

func TestGeminiRejectsImageURL(t *testing.T) {
	_, url := newFake(t, "application/json", geminiReply(`{"text":"hi"}`, "STOP"))
	_, err := newTestGemini(t, url, nil).GenerateContent(bg, []MessageContent{{Role: ChatMessageTypeHuman,
		Parts: []ContentPart{ImageURLContent{URL: "https://example.com/a.png"}}}})
	if err == nil || !strings.Contains(err.Error(), "data: URL") {
		t.Errorf("err = %v", err)
	}
}

func TestGeminiStreamingKeepsToolCallSignature(t *testing.T) {
	sig := base64.StdEncoding.EncodeToString(geminiSignature)
	body := "data: " + `{"candidates":[{"content":{"role":"model","parts":[{"text":"Let me look."}]}}]}` + "\n\n" +
		"data: " + `{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"id":"c1","name":"lookup","args":{"q":"x"}},"thoughtSignature":"` + sig + `"}]},"finishReason":"STOP"}]}` + "\n\n"
	_, url := newFake(t, "text/event-stream", body)
	resp, err := newTestGemini(t, url, nil).GenerateContent(bg, []MessageContent{textParts(ChatMessageTypeHuman, "hi")},
		WithStreamingFunc(func(context.Context, []byte) error { return nil }))
	if err != nil {
		t.Fatal(err)
	}
	c := resp.Choices[0]
	if len(c.ToolCalls) != 1 || c.ToolCalls[0].ID != "c1" || !reflect.DeepEqual(c.ToolCalls[0].Signature, geminiSignature) || c.StopReason != "STOP" {
		t.Errorf("streamed choice = %+v (the next turn needs the signature)", c)
	}
	if len(c.Parts) != 2 {
		t.Errorf("parts = %+v, want the text then the tool call", c.Parts)
	}
}
