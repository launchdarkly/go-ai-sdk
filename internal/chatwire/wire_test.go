package chatwire

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/kitbase"
)

// asJSON is v as the wire sees it, decoded back into plain values.
func asJSON(t *testing.T, v any) any {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	var out any
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatal(err)
	}
	return out
}

func decode(t *testing.T, s string) any {
	t.Helper()
	var out any
	if err := json.Unmarshal([]byte(s), &out); err != nil {
		t.Fatal(err)
	}
	return out
}

func assertJSON(t *testing.T, name string, got any, want string) {
	t.Helper()
	if g, w := asJSON(t, got), decode(t, want); !reflect.DeepEqual(g, w) {
		gb, _ := json.Marshal(g)
		t.Errorf("%s:\n got %s\nwant %s", name, gb, want)
	}
}

func TestMessages(t *testing.T) {
	turns := []aiserver.Message{
		{Role: "system", Text: "be brief"},
		{Role: "user", Text: "hi"},
		{Role: "assistant", Blocks: []aiserver.ContentBlock{{Type: "text", Text: "a"}, {Type: "text", Text: "b"}}},
		{Role: "user", Blocks: []aiserver.ContentBlock{
			{Type: "text", Text: "look"},
			{Type: "image", Source: &aiserver.ImageSource{Type: "url", URL: "https://x/cat.png"}},
			{Type: "image", Source: &aiserver.ImageSource{Type: "base64", MediaType: "image/png", Data: "QUJD"}},
		}},
	}
	assertJSON(t, "messages", Messages("You are helpful.", turns), `[
		{"role":"system","content":"You are helpful."},
		{"role":"system","content":"be brief"},
		{"role":"user","content":"hi"},
		{"role":"assistant","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]},
		{"role":"user","content":[
			{"type":"text","text":"look"},
			{"type":"image_url","image_url":{"url":"https://x/cat.png"}},
			{"type":"image_url","image_url":{"url":"data:image/png;base64,QUJD"}}
		]}
	]`)
	assertJSON(t, "no system", Messages("", []aiserver.Message{{Role: "user", Text: "q"}}), `[{"role":"user","content":"q"}]`)
}

func TestFunctionTools(t *testing.T) {
	tools := map[string]aiserver.Tool{
		"zeta":  {Description: "last", Parameters: map[string]any{"type": "object", "properties": map[string]any{"q": map[string]any{"type": "string"}}}},
		"alpha": {Parameters: map[string]any{}},
	}
	assertJSON(t, "tools", FunctionTools(tools), `[
		{"type":"function","function":{"name":"alpha","description":"","parameters":{}}},
		{"type":"function","function":{"name":"zeta","description":"last",
			"parameters":{"type":"object","properties":{"q":{"type":"string"}}}}}
	]`)
	if FunctionTools(nil) != nil {
		t.Error("no tools must map to nil, so the request omits tools")
	}
}

func TestModelParameters(t *testing.T) {
	in := map[string]any{"temperature": 0.2, "max_tokens": 50, "top_p": nil, "base_url": "http://evil", "api_key": "k"}
	got := ModelParameters(in, []string{"temperature", "max_tokens", "top_p", "seed"})
	if want := map[string]any{"temperature": 0.2, "max_tokens": 50}; !reflect.DeepEqual(got, want) {
		t.Errorf("ModelParameters = %v, want %v (no renaming, nil and unlisted dropped)", got, want)
	}
	if got := ModelParameters(nil, []string{"temperature"}); len(got) != 0 {
		t.Errorf("nil parameters = %v", got)
	}
}

func TestResponseFormat(t *testing.T) {
	schema := map[string]any{"type": "object", "properties": map[string]any{"a": map[string]any{"type": "number"}}}
	assertJSON(t, "response_format", ResponseFormat(schema),
		`{"type":"json_schema","json_schema":{"name":"output","strict":false,"schema":{"type":"object","properties":{"a":{"type":"number"}}}}}`)
}

func completion(t *testing.T, body string) *openai.ChatCompletion {
	t.Helper()
	var c openai.ChatCompletion
	if err := json.Unmarshal([]byte(body), &c); err != nil {
		t.Fatal(err)
	}
	return &c
}

func TestTurnOf(t *testing.T) {
	resp := completion(t, `{"id":"c1","object":"chat.completion","model":"anthropic/claude-sonnet-4","choices":[
		{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":"checking",
			"tool_calls":[
				{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"q\":\"a\"}"}},
				{"id":"call_2","type":"function","function":{"name":"weather","arguments":"{}"}}]}},
		{"index":1,"finish_reason":"stop","message":{"role":"assistant","content":"ignored"}}],
		"usage":{"prompt_tokens":12,"completion_tokens":5,"total_tokens":17,"prompt_tokens_details":{"cached_tokens":4}}}`)
	got, err := TurnOf(resp)
	if err != nil {
		t.Fatal(err)
	}
	want := Turn{
		Text:      "checking",
		ToolCalls: []ToolCall{{ID: "call_1", Name: "lookup", Arguments: `{"q":"a"}`}, {ID: "call_2", Name: "weather", Arguments: "{}"}},
		Usage:     kitbase.SpanUsage{Input: 12, Output: 5, CacheRead: 4},
		Model:     "anthropic/claude-sonnet-4",
		Finish:    "tool_calls",
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("TurnOf =\n %+v\nwant\n %+v", got, want)
	}

	if _, err := TurnOf(completion(t, `{"id":"c2","choices":[]}`)); !errors.Is(err, ErrNoChoices) {
		t.Errorf("no choices: err = %v, want ErrNoChoices", err)
	}
	if _, err := TurnOf(nil); !errors.Is(err, ErrNoChoices) {
		t.Errorf("nil response: err = %v, want ErrNoChoices", err)
	}
	noUsage, err := TurnOf(completion(t, `{"id":"c3","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"x"}}]}`))
	if err != nil || !noUsage.Usage.Unreported {
		t.Errorf("a response without usage must be Unreported: %+v %v", noUsage.Usage, err)
	}
}

func TestToolTurnMessages(t *testing.T) {
	turn := Turn{Text: "checking", ToolCalls: []ToolCall{{ID: "call_1", Name: "lookup", Arguments: `{"q":"a"}`}, {ID: "call_2", Name: "weather", Arguments: "{}"}}}
	assertJSON(t, "assistant", AssistantMessage(turn), `{"role":"assistant","content":"checking","tool_calls":[
		{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"q\":\"a\"}"}},
		{"id":"call_2","type":"function","function":{"name":"weather","arguments":"{}"}}]}`)
	assertJSON(t, "assistant without text", AssistantMessage(Turn{Text: "", ToolCalls: turn.ToolCalls[:1]}), `{"role":"assistant","content":null,"tool_calls":[
		{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"q\":\"a\"}"}}]}`)
	assertJSON(t, "tool", ToolMessage("call_1", "42"), `{"role":"tool","tool_call_id":"call_1","content":"42"}`)
}

// fakeServer serves one Chat Completions stream per request from chunks,
// recording each request body. After the chunks it ends the stream, or, with
// hold, keeps the connection open until the client goes away and reports
// that on closed.
type fakeServer struct {
	chunks []string
	hold   bool
	status int
	bodies []map[string]any
	closed chan struct{}
}

func (f *fakeServer) start(t *testing.T) *openai.Client {
	t.Helper()
	f.closed = make(chan struct{}, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("path = %s, want /chat/completions", r.URL.Path)
		}
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		f.bodies = append(f.bodies, body)
		if f.status != 0 {
			w.WriteHeader(f.status)
			_, _ = io.WriteString(w, `{"error":{"message":"upstream down","type":"server_error"}}`)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		for _, c := range f.chunks {
			fmt.Fprintf(w, "data: %s\n\n", c)
			w.(http.Flusher).Flush()
		}
		if f.hold {
			select {
			case <-r.Context().Done():
				f.closed <- struct{}{}
			case <-time.After(5 * time.Second):
			}
			return
		}
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	t.Cleanup(srv.Close)
	c := openai.NewClient(option.WithBaseURL(srv.URL), option.WithAPIKey("k"), option.WithMaxRetries(0))
	return &c
}

func chunk(delta string, finish string) string {
	f := "null"
	if finish != "" {
		f = `"` + finish + `"`
	}
	return `{"id":"s1","object":"chat.completion.chunk","model":"gw/model-x","choices":[{"index":0,"delta":` + delta + `,"finish_reason":` + f + `}]}`
}

const usageChunk = `{"id":"s1","object":"chat.completion.chunk","model":"gw/model-x","choices":[],"usage":{"prompt_tokens":9,"completion_tokens":3,"total_tokens":12}}`

func streamParams() openai.ChatCompletionNewParams {
	return openai.ChatCompletionNewParams{Model: "anthropic/claude-sonnet-4", Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hi")}}
}

func TestStreamTurnText(t *testing.T) {
	f := &fakeServer{chunks: []string{
		chunk(`{"role":"assistant","content":"Hel"}`, ""),
		chunk(`{"content":"lo"}`, ""),
		chunk(`{}`, "stop"),
		usageChunk,
	}}
	client := f.start(t)
	var deltas []string
	turn, stopped, err := StreamTurn(context.Background(), client, streamParams(), func(s string) bool { deltas = append(deltas, s); return true })
	if err != nil || stopped {
		t.Fatalf("StreamTurn: stopped=%v err=%v", stopped, err)
	}
	if !reflect.DeepEqual(deltas, []string{"Hel", "lo"}) {
		t.Errorf("deltas = %q", deltas)
	}
	want := Turn{Text: "Hello", Usage: kitbase.SpanUsage{Input: 9, Output: 3}, Model: "gw/model-x", Finish: "stop"}
	if !reflect.DeepEqual(turn, want) {
		t.Errorf("turn = %+v, want %+v", turn, want)
	}
	if len(f.bodies) != 1 {
		t.Fatalf("requests = %d", len(f.bodies))
	}
	b := f.bodies[0]
	if b["stream"] != true || !reflect.DeepEqual(b["stream_options"], map[string]any{"include_usage": true}) {
		t.Errorf("request must stream and ask for usage: stream=%v stream_options=%v", b["stream"], b["stream_options"])
	}
	if b["model"] != "anthropic/claude-sonnet-4" {
		t.Errorf("model = %v, want it unchanged", b["model"])
	}
}

// Two parallel calls arrive interleaved, with the id and name only in each
// call's first fragment and the arguments split across chunks.
func TestStreamTurnJoinsToolCallFragmentsByIndex(t *testing.T) {
	f := &fakeServer{chunks: []string{
		chunk(`{"role":"assistant","tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"lookup","arguments":""}}]}`, ""),
		chunk(`{"tool_calls":[{"index":1,"id":"call_b","type":"function","function":{"name":"weather","arguments":"{\"ci"}}]}`, ""),
		chunk(`{"tool_calls":[{"index":0,"function":{"arguments":"{\"q\":"}}]}`, ""),
		chunk(`{"tool_calls":[{"index":1,"function":{"arguments":"ty\":\"Oslo\"}"}}]}`, ""),
		chunk(`{"tool_calls":[{"index":0,"function":{"arguments":"\"a\"}"}}]}`, ""),
		chunk(`{}`, "tool_calls"),
		usageChunk,
	}}
	client := f.start(t)
	turn, _, err := StreamTurn(context.Background(), client, streamParams(), func(string) bool { return true })
	if err != nil {
		t.Fatal(err)
	}
	want := []ToolCall{{ID: "call_a", Name: "lookup", Arguments: `{"q":"a"}`}, {ID: "call_b", Name: "weather", Arguments: `{"city":"Oslo"}`}}
	if !reflect.DeepEqual(turn.ToolCalls, want) || turn.Finish != "tool_calls" {
		t.Errorf("tool calls = %+v finish = %q, want %+v", turn.ToolCalls, turn.Finish, want)
	}
}

func TestStreamTurnAbandonmentClosesTheBody(t *testing.T) {
	f := &fakeServer{chunks: []string{chunk(`{"role":"assistant","content":"first"}`, "")}, hold: true}
	client := f.start(t)
	_, stopped, err := StreamTurn(context.Background(), client, streamParams(), func(string) bool { return false })
	if err != nil || !stopped {
		t.Fatalf("stopped=%v err=%v, want stopped and no error", stopped, err)
	}
	select {
	case <-f.closed:
	case <-time.After(3 * time.Second):
		t.Fatal("the stream body was not closed after the consumer stopped")
	}
}

func TestStreamTurnErrors(t *testing.T) {
	f := &fakeServer{status: http.StatusInternalServerError}
	client := f.start(t)
	if _, _, err := StreamTurn(context.Background(), client, streamParams(), func(string) bool { return true }); err == nil || !strings.Contains(err.Error(), "500") {
		t.Errorf("HTTP 500: err = %v", err)
	}
	empty := &fakeServer{chunks: []string{usageChunk}}
	client = empty.start(t)
	if _, _, err := StreamTurn(context.Background(), client, streamParams(), func(string) bool { return true }); !errors.Is(err, ErrNoChoices) {
		t.Errorf("a stream with no choice: err = %v, want ErrNoChoices", err)
	}
}
