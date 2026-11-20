//go:build opencode_live

package opencodeagents

// The live contract: one bridged-tool turn against a real `opencode` binary
// of the pinned version, with a local stub OpenAI-compatible provider, so it
// needs no credentials. Run it with
//
//	go test -tags opencode_live -run TestLiveBridgedToolTurn ./opencodeagents/
//
// with `opencode` on PATH (or OPENCODE_BIN set). Bump pinnedOpenCode and the
// CI job's install together, on purpose.

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

const pinnedOpenCode = "1.19.5"

// stubProvider is an OpenAI-compatible chat completions server. Given a
// tool whose name ends in _get_weather and no tool result yet, it calls that
// tool; otherwise it answers with text that quotes the last tool result.
type stubProvider struct {
	mu       sync.Mutex
	requests []map[string]any
}

func (p *stubProvider) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if !strings.HasSuffix(r.URL.Path, "/chat/completions") {
		http.NotFound(w, r)
		return
	}
	b, _ := io.ReadAll(r.Body)
	var req map[string]any
	_ = json.Unmarshal(b, &req)
	p.mu.Lock()
	p.requests = append(p.requests, req)
	p.mu.Unlock()

	toolName, lastTool := "", ""
	tools, _ := req["tools"].([]any)
	for _, t := range tools {
		fn, _ := t.(map[string]any)["function"].(map[string]any)
		if n, _ := fn["name"].(string); strings.HasSuffix(n, "_get_weather") {
			toolName = n
		}
	}
	msgs, _ := req["messages"].([]any)
	for _, m := range msgs {
		mm, _ := m.(map[string]any)
		if mm["role"] == "tool" {
			lastTool = fmt.Sprint(mm["content"])
		}
	}
	var delta map[string]any
	finish := "stop"
	switch {
	case toolName != "" && lastTool == "":
		delta = map[string]any{"role": "assistant", "tool_calls": []any{map[string]any{"index": 0, "id": "call_1", "type": "function",
			"function": map[string]any{"name": toolName, "arguments": `{"city":"Paris"}`}}}}
		finish = "tool_calls"
	case lastTool != "":
		delta = map[string]any{"role": "assistant", "content": "Forecast: " + lastTool}
	default:
		delta = map[string]any{"role": "assistant", "content": "Weather chat"}
	}
	usage := map[string]any{"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
	if stream, _ := req["stream"].(bool); !stream {
		msg := map[string]any{"role": "assistant", "content": delta["content"]}
		if tc, ok := delta["tool_calls"]; ok {
			msg["tool_calls"] = tc
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"id": "c1", "object": "chat.completion", "created": 1, "model": "stub-model",
			"choices": []any{map[string]any{"index": 0, "message": msg, "finish_reason": finish}}, "usage": usage})
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	chunk := func(v map[string]any) {
		b, _ := json.Marshal(v)
		fmt.Fprintf(w, "data: %s\n\n", b)
		w.(http.Flusher).Flush()
	}
	chunk(map[string]any{"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "stub-model",
		"choices": []any{map[string]any{"index": 0, "delta": delta, "finish_reason": nil}}})
	chunk(map[string]any{"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "stub-model",
		"choices": []any{map[string]any{"index": 0, "delta": map[string]any{}, "finish_reason": finish}}, "usage": usage})
	fmt.Fprint(w, "data: [DONE]\n\n")
}

func TestLiveBridgedToolTurn(t *testing.T) {
	bin := os.Getenv("OPENCODE_BIN")
	if bin == "" {
		var err error
		if bin, err = exec.LookPath("opencode"); err != nil {
			t.Fatal("opencode is not on PATH; install opencode-ai@" + pinnedOpenCode)
		}
	}
	out, err := exec.Command(bin, "--version").Output()
	if v := strings.TrimSpace(string(out)); err != nil || v != pinnedOpenCode {
		t.Fatalf("opencode --version = %q (%v), want %s", v, err, pinnedOpenCode)
	}

	stub := &stubProvider{}
	api := httptest.NewServer(stub)
	defer api.Close()

	root := t.TempDir()
	work, home := filepath.Join(root, "work"), filepath.Join(root, "home")
	for _, d := range []string{work, home} {
		if err := os.Mkdir(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	config := filepath.Join(root, "opencode.json")
	cfgJSON, _ := json.Marshal(map[string]any{
		"$schema": "https://opencode.ai/config.json", "autoupdate": false, "share": "disabled",
		"provider": map[string]any{"stub": map[string]any{
			"npm": "@ai-sdk/openai-compatible", "name": "Stub",
			"options": map[string]any{"baseURL": api.URL + "/v1", "apiKey": "not-a-key"},
			"models":  map[string]any{"stub-model": map[string]any{"name": "Stub model", "tool_call": true}},
		}},
		"model": "stub/stub-model", "small_model": "stub/stub-model",
	})
	if err := os.WriteFile(config, cfgJSON, 0o600); err != nil {
		t.Fatal(err)
	}

	tools := map[string]any{"get_weather": map[string]any{"name": "get_weather", "type": "function", "description": "Weather for a city",
		"parameters": map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}, "required": []any{"city"}}}}
	e := setup(t, map[string]any{"cfg": agentVariation("Use the weather tool.", map[string]any{"tools": tools})})
	short(t, &startupTimeout, 120*time.Second)
	short(t, &drainFallback, 5*time.Second)

	var calls []map[string]any
	var mu sync.Mutex
	weather := aiserver.ToolFunc(func(_ context.Context, args map[string]any) (any, error) {
		mu.Lock()
		calls = append(calls, args)
		mu.Unlock()
		return "sunny in " + fmt.Sprint(args["city"]), nil
	})
	ctx, cancel := context.WithTimeout(bg, 4*time.Minute)
	defer cancel()
	res, err := OpenCodeAgents(ctx, "cfg", "What is the weather in Paris?", user, &InvokeOptions{
		Handler: Options{WorkDir: work, ConfigPath: config, Command: []string{bin},
			Env: []string{"PATH=" + os.Getenv("PATH"), "HOME=" + home,
				"XDG_CONFIG_HOME=" + filepath.Join(home, ".config"), "XDG_DATA_HOME=" + filepath.Join(home, ".local", "share"),
				"XDG_CACHE_HOME=" + filepath.Join(home, ".cache"), "XDG_STATE_HOME=" + filepath.Join(home, ".local", "state")},
			Model: func(*aiserver.AiConfig) (string, string, error) { return "stub", "stub-model", nil }},
		ToolHandlers: map[string]aiserver.ToolHandler{"get_weather": weather},
	})
	if err != nil {
		t.Fatalf("live turn: %v", err)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(calls) != 1 || calls[0]["city"] != "Paris" {
		t.Errorf("bridged tool calls = %v", calls)
	}
	if got := fmt.Sprint(res.Response); !strings.Contains(got, "sunny in Paris") {
		t.Errorf("response = %q", got)
	}
	if tc := e.ld.events("$ld:ai:tool_call"); len(tc) != 1 || tc[0].Data["toolKey"] != "get_weather" {
		t.Errorf("tool_call events = %+v", tc)
	}
	if ts := e.spans("execute_tool"); len(ts) != 1 || attrs(ts[0])["gen_ai.tool.name"] != "get_weather" {
		t.Errorf("execute_tool spans = %d", len(ts))
	}
	t.Logf("opencode %s: %d provider requests, response %q", pinnedOpenCode, len(stub.requests), res.Response)
}
