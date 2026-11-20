package opencodeagents

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

func startBridge(t *testing.T, tools map[string]aiserver.ToolFunc) (*bridge, string) {
	t.Helper()
	b := newBridge()
	for k, fn := range tools {
		b.register(k, k+" tool", map[string]any{"type": "object", "properties": map[string]any{"x": map[string]any{"type": "string"}}}, fn)
	}
	url, err := b.start(bg)
	if err != nil {
		t.Fatalf("start: %v", err)
	}
	t.Cleanup(b.stop)
	if !strings.HasPrefix(url, "http://127.0.0.1:") {
		t.Fatalf("url = %q", url)
	}
	if len(b.token) < 32 {
		t.Fatalf("token %q is too short", b.token)
	}
	return b, url
}

func rpcCall(t *testing.T, url, token string, msg map[string]any) (int, map[string]any) {
	t.Helper()
	body, _ := json.Marshal(msg)
	req, _ := http.NewRequest("POST", url, bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	var out map[string]any
	_ = json.Unmarshal(raw, &out)
	return resp.StatusCode, out
}

func TestBridgeRejectsAMissingOrWrongToken(t *testing.T) {
	b, url := startBridge(t, nil)
	for _, tok := range []string{"", "wrong", b.token + "x"} {
		if code, _ := rpcCall(t, url, tok, map[string]any{"jsonrpc": "2.0", "id": 1, "method": "initialize"}); code != 401 {
			t.Errorf("POST with token %q: %d", tok, code)
		}
		req, _ := http.NewRequest("GET", url, nil)
		if tok != "" {
			req.Header.Set("Authorization", "Bearer "+tok)
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if resp.StatusCode != 401 {
			t.Errorf("GET with token %q: %d", tok, resp.StatusCode)
		}
	}
}

func TestBridgeProtocol(t *testing.T) {
	var seen map[string]any
	b, url := startBridge(t, map[string]aiserver.ToolFunc{
		"echo": func(_ context.Context, args map[string]any) (any, error) {
			seen = args
			return map[string]any{"got": args["x"]}, nil
		},
		"fail": func(context.Context, map[string]any) (any, error) { return nil, errors.New("it broke") },
	})
	code, init := rpcCall(t, url, b.token, map[string]any{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": map[string]any{}})
	res, _ := init["result"].(map[string]any)
	if code != 200 || init["id"] != float64(1) || res["protocolVersion"] == nil || res["capabilities"].(map[string]any)["tools"] == nil {
		t.Errorf("initialize = %d %v", code, init)
	}
	if code, body := rpcCall(t, url, b.token, map[string]any{"jsonrpc": "2.0", "method": "notifications/initialized"}); code != 202 || body != nil {
		t.Errorf("notification = %d %v", code, body)
	}
	_, list := rpcCall(t, url, b.token, map[string]any{"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
	tools, _ := list["result"].(map[string]any)["tools"].([]any)
	names := map[string]map[string]any{}
	for _, x := range tools {
		m := x.(map[string]any)
		names[m["name"].(string)] = m
	}
	if len(names) != 2 || names["echo"]["description"] != "echo tool" || names["echo"]["inputSchema"].(map[string]any)["type"] != "object" {
		t.Errorf("tools/list = %v", list)
	}
	_, call := rpcCall(t, url, b.token, map[string]any{"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": map[string]any{"name": "echo", "arguments": map[string]any{"x": "hi"}}})
	if r, _ := call["result"].(map[string]any); r["isError"] != false || jsonString(r["content"]) != `[{"text":"{\"got\":\"hi\"}","type":"text"}]` || seen["x"] != "hi" {
		t.Errorf("tools/call echo = %v (seen %v)", call, seen)
	}
	_, failed := rpcCall(t, url, b.token, map[string]any{"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": map[string]any{"name": "fail", "arguments": map[string]any{}}})
	if r, _ := failed["result"].(map[string]any); r["isError"] != true || !strings.Contains(jsonString(r["content"]), "it broke") {
		t.Errorf("tools/call fail = %v", failed)
	}
	_, unknown := rpcCall(t, url, b.token, map[string]any{"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": map[string]any{"name": "nope"}})
	if e, _ := unknown["error"].(map[string]any); e["code"] != float64(-32602) {
		t.Errorf("unknown tool = %v", unknown)
	}
	_, method := rpcCall(t, url, b.token, map[string]any{"jsonrpc": "2.0", "id": 6, "method": "resources/list"})
	if e, _ := method["error"].(map[string]any); e["code"] != float64(-32601) {
		t.Errorf("unknown method = %v", method)
	}
}

// GET is a live event stream, not 405: ": connected" at once, keepalives
// after that, ended by stop.
func TestBridgeGetIsALiveEventStream(t *testing.T) {
	short(t, &keepaliveInterval, 50*time.Millisecond)
	b, url := startBridge(t, nil)
	req, _ := http.NewRequest("GET", url, nil)
	req.Header.Set("Authorization", "Bearer "+b.token)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 || !strings.HasPrefix(resp.Header.Get("Content-Type"), "text/event-stream") {
		t.Fatalf("GET = %d %s", resp.StatusCode, resp.Header.Get("Content-Type"))
	}
	r := bufio.NewReader(resp.Body)
	if line, _ := r.ReadString('\n'); line != ": connected\n" {
		t.Errorf("first line = %q", line)
	}
	_, _ = r.ReadString('\n') // blank line ending the comment
	if line, _ := r.ReadString('\n'); line != ": keepalive\n" {
		t.Errorf("second comment = %q", line)
	}
	done := make(chan struct{})
	go func() { _, _ = io.Copy(io.Discard, r); close(done) }()
	b.stop()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Error("stop did not end the stream")
	}
}

func TestBridgeToolName(t *testing.T) {
	for _, tc := range [][3]string{
		{"ld_0123456789abcdef", "get_weather", "ld_0123456789abcdef_get_weather"},
		{"ld_ab", "get weather.v2", "ld_ab_get_weather_v2"},
		{"ld_ab", "a-b/c", "ld_ab_a-b_c"},
	} {
		if got := bridgeToolName(tc[0], tc[1]); got != tc[2] {
			t.Errorf("bridgeToolName(%q, %q) = %q, want %q", tc[0], tc[1], got, tc[2])
		}
	}
}

func TestBridgesDoNotShareTokensOrPorts(t *testing.T) {
	a, ua := startBridge(t, nil)
	b, ub := startBridge(t, nil)
	if a.token == b.token || ua == ub {
		t.Error("two bridges share a token or a URL")
	}
	if code, _ := rpcCall(t, ua, b.token, map[string]any{"jsonrpc": "2.0", "id": 1, "method": "initialize"}); code != 401 {
		t.Errorf("one bridge accepted another's token: %d", code)
	}
}

func TestBridgeStopBeforeStart(t *testing.T) {
	newBridge().stop() // must not block or fail
}
