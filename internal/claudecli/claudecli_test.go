package claudecli

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"
)

// collect runs to the end and returns every message and the last error.
func collect(ctx context.Context, opts Options, prompt string) ([]Message, error) {
	var out []Message
	var last error
	for m, err := range Run(ctx, opts, prompt) {
		if err != nil {
			last = err
			continue
		}
		out = append(out, m)
	}
	return out, last
}

func resultOnly() fakeScript { return fakeScript{Steps: []fakeStep{{Send: msg(resultLine)}}} }

func mustRun(t *testing.T, opts Options) []Message {
	t.Helper()
	msgs, err := collect(context.Background(), opts, "hello")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	return msgs
}

func jsonEqual(t *testing.T, got, want []byte) bool {
	t.Helper()
	var g, w any
	if err := json.Unmarshal(got, &g); err != nil {
		t.Fatalf("got %s: %v", got, err)
	}
	if err := json.Unmarshal(want, &w); err != nil {
		t.Fatalf("want %s: %v", want, err)
	}
	return reflect.DeepEqual(g, w)
}

// The argv follows claude-agent-sdk's order. --tools and --setting-sources
// are always present, so an empty value isolates the run instead of falling
// back to the CLI's defaults (every built-in, every settings source).
func TestArgs(t *testing.T) {
	const mcpConfig = `{"mcpServers":{"tool-mcp":{"type":"sdk","name":"tool-mcp"}}}`
	tests := []struct {
		name string
		opts Options
		want []string
	}{
		{"defaults", Options{}, []string{
			"--output-format", "stream-json", "--verbose", "--system-prompt", "",
			"--tools", "", "--setting-sources=", "--input-format", "stream-json",
		}},
		{"explicit empty lists", Options{Tools: []string{}, SettingSources: []string{}}, []string{
			"--output-format", "stream-json", "--verbose", "--system-prompt", "",
			"--tools", "", "--setting-sources=", "--input-format", "stream-json",
		}},
		{"every option", Options{
			SystemPrompt: "be brief", Tools: []string{"Read", "Bash"},
			AllowedTools: []string{"mcp__tool-mcp__lookup"}, Skills: []string{"launchdarkly:a", "launchdarkly:b"},
			MaxTurns: 3, Model: "us.anthropic.claude-x", MCPServers: []string{"tool-mcp"}, PartialMessages: true,
			SettingSources: []string{"user", "project", "local"}, PluginDirs: []string{"/p1", "/p2"},
		}, []string{
			"--output-format", "stream-json", "--verbose", "--system-prompt", "be brief",
			"--tools", "Read,Bash",
			"--allowedTools", "mcp__tool-mcp__lookup,Skill(launchdarkly:a),Skill(launchdarkly:b)",
			"--max-turns", "3", "--model", "us.anthropic.claude-x",
			"--mcp-config", mcpConfig, "--include-partial-messages",
			"--setting-sources=user,project,local", "--plugin-dir", "/p1", "--plugin-dir", "/p2",
			"--input-format", "stream-json",
		}},
		{"skills alone allow their Skill rules", Options{Skills: []string{"launchdarkly:a"}, Tools: []string{"Skill"}}, []string{
			"--output-format", "stream-json", "--verbose", "--system-prompt", "",
			"--tools", "Skill", "--allowedTools", "Skill(launchdarkly:a)",
			"--setting-sources=", "--input-format", "stream-json",
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			opts, read := newFake(t, resultOnly())
			base := opts
			opts = tt.opts
			opts.Path, opts.Env = base.Path, base.Env
			mustRun(t, opts)
			got := read().Args
			// The MCP config is JSON: compare it as JSON, then the rest exactly.
			for i := range got {
				if i > 0 && got[i-1] == "--mcp-config" {
					if !jsonEqual(t, []byte(got[i]), []byte(mcpConfig)) {
						t.Errorf("--mcp-config = %s, want %s", got[i], mcpConfig)
					}
					got[i] = mcpConfig
				}
			}
			if !slices.Equal(got, tt.want) {
				t.Errorf("argv\n got %q\nwant %q", got, tt.want)
			}
		})
	}
}

func envValues(env []string, key string) []string {
	var vals []string
	for _, kv := range env {
		if k, v, ok := strings.Cut(kv, "="); ok && k == key {
			vals = append(vals, v)
		}
	}
	return vals
}

// The CLI inherits the parent environment minus CLAUDECODE, with
// CLAUDE_CODE_ENTRYPOINT=sdk-go, then Options.Env on top.
func TestEnvironment(t *testing.T) {
	t.Setenv("CLAUDECODE", "1")
	t.Setenv("LD_CANARY", "parent")
	t.Setenv("CLAUDE_CODE_ENTRYPOINT", "cli")

	t.Run("inherited", func(t *testing.T) {
		opts, read := newFake(t, resultOnly())
		mustRun(t, opts)
		env := read().Env
		for key, want := range map[string][]string{
			"CLAUDECODE": nil, "LD_CANARY": {"parent"}, "CLAUDE_CODE_ENTRYPOINT": {"sdk-go"},
		} {
			if got := envValues(env, key); !slices.Equal(got, want) {
				t.Errorf("%s = %q, want %q", key, got, want)
			}
		}
	})
	t.Run("overrides", func(t *testing.T) {
		opts, read := newFake(t, resultOnly())
		opts.Env = append(opts.Env, "CLAUDE_CODE_ENTRYPOINT=custom", "LD_CANARY=override")
		mustRun(t, opts)
		env := read().Env
		for key, want := range map[string][]string{"LD_CANARY": {"override"}, "CLAUDE_CODE_ENTRYPOINT": {"custom"}} {
			if got := envValues(env, key); !slices.Equal(got, want) {
				t.Errorf("%s = %q, want %q", key, got, want)
			}
		}
	})
}

func TestWorkingDir(t *testing.T) {
	t.Run("set", func(t *testing.T) {
		dir, err := filepath.EvalSymlinks(t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		opts, read := newFake(t, resultOnly())
		opts.Dir = dir
		mustRun(t, opts)
		if got := read().Dir; got != dir {
			t.Errorf("cwd = %q, want %q", got, dir)
		}
	})
	t.Run("empty inherits", func(t *testing.T) {
		wd, err := os.Getwd()
		if err != nil {
			t.Fatal(err)
		}
		opts, read := newFake(t, resultOnly())
		mustRun(t, opts)
		if got := read().Dir; got != wd {
			t.Errorf("cwd = %q, want %q", got, wd)
		}
	})
}

type initRequest struct {
	Type      string `json:"type"`
	RequestID string `json:"request_id"`
	Request   map[string]json.RawMessage
}

func TestInitialize(t *testing.T) {
	t.Run("bare", func(t *testing.T) {
		opts, read := newFake(t, resultOnly())
		mustRun(t, opts)
		r := read()
		if len(r.Recv) < 2 {
			t.Fatalf("fake received %d lines, want initialize and the prompt", len(r.Recv))
		}
		var init initRequest
		if err := json.Unmarshal(r.Recv[0].Line, &init); err != nil {
			t.Fatal(err)
		}
		if init.Type != "control_request" || init.RequestID == "" {
			t.Errorf("first line = %s, want a control_request with an id", r.Recv[0].Line)
		}
		if string(init.Request["subtype"]) != `"initialize"` || string(init.Request["hooks"]) != "null" {
			t.Errorf("request = %s, want subtype initialize and hooks null", r.Recv[0].Line)
		}
		for _, k := range []string{"agents", "skills"} {
			if _, ok := init.Request[k]; ok {
				t.Errorf("request has %q, want it omitted", k)
			}
		}
	})
	t.Run("hooks agents skills", func(t *testing.T) {
		opts, read := newFake(t, resultOnly())
		opts.OnHook = func(context.Context, HookInput) error { return nil }
		opts.Agents = map[string]any{"child": map[string]any{"description": "d", "prompt": "p"}}
		opts.Skills = []string{"launchdarkly:a", "launchdarkly:b"}
		mustRun(t, opts)
		r := read()
		var init initRequest
		if err := json.Unmarshal(r.Recv[0].Line, &init); err != nil {
			t.Fatal(err)
		}
		var hooks map[string][]struct {
			Matcher *string  `json:"matcher"`
			IDs     []string `json:"hookCallbackIds"`
		}
		if err := json.Unmarshal(init.Request["hooks"], &hooks); err != nil {
			t.Fatal(err)
		}
		seen := map[string]bool{}
		for _, ev := range []string{HookPreToolUse, HookPostToolUse, HookPostToolUseFailure} {
			ms := hooks[ev]
			if len(ms) != 1 || ms[0].Matcher != nil || len(ms[0].IDs) != 1 || seen[ms[0].IDs[0]] {
				t.Errorf("hooks[%s] = %+v, want one null-matcher entry with one distinct id", ev, ms)
				continue
			}
			seen[ms[0].IDs[0]] = true
		}
		if len(hooks) != 3 {
			t.Errorf("hooks has %d events, want 3", len(hooks))
		}
		if !jsonEqual(t, init.Request["agents"], []byte(`{"child":{"description":"d","prompt":"p"}}`)) {
			t.Errorf("agents = %s", init.Request["agents"])
		}
		if !jsonEqual(t, init.Request["skills"], []byte(`["launchdarkly:a","launchdarkly:b"]`)) {
			t.Errorf("skills = %s", init.Request["skills"])
		}
	})
	t.Run("prompt waits for the response", func(t *testing.T) {
		s := resultOnly()
		s.InitDelayMs = 200
		opts, read := newFake(t, s)
		mustRun(t, opts)
		r := read()
		if len(r.Recv) < 2 || !r.Recv[1].AfterInit {
			t.Errorf("the prompt reached the CLI before the initialize response")
		}
	})
	t.Run("error fails the run", func(t *testing.T) {
		opts, read := newFake(t, fakeScript{InitError: "bad init"})
		_, err := collect(context.Background(), opts, "hello")
		if err == nil || !strings.Contains(err.Error(), "bad init") {
			t.Errorf("err = %v, want the initialize error", err)
		}
		if n := len(read().Recv); n != 1 {
			t.Errorf("fake received %d lines, want only initialize", n)
		}
	})
}

func TestPrompt(t *testing.T) {
	opts, read := newFake(t, resultOnly())
	mustRun(t, opts)
	r := read()
	if len(r.Recv) < 2 {
		t.Fatalf("fake received %d lines", len(r.Recv))
	}
	want := `{"type":"user","session_id":"","message":{"role":"user","content":"hello"},"parent_tool_use_id":null}`
	if !jsonEqual(t, r.Recv[1].Line, []byte(want)) {
		t.Errorf("prompt = %s, want %s", r.Recv[1].Line, want)
	}
}

func TestMessages(t *testing.T) {
	steps := []fakeStep{
		{Send: msg(`{"type":"system","subtype":"init","session_id":"s1","tools":["Read"]}`)},
		{Send: msg(`{"type":"assistant","message":{"model":"claude-x","content":[{"type":"thinking","thinking":"hmm","signature":"sig"},{"type":"text","text":"reading"},{"type":"tool_use","id":"tu1","name":"Read","input":{"file_path":"a.txt"}}],"usage":{"input_tokens":10,"output_tokens":5,"cache_read_input_tokens":2,"cache_creation_input_tokens":1}},"parent_tool_use_id":"p1","session_id":"s1"}`)},
		{Send: msg(`{"type":"rate_limit_event","session_id":"s1"}`)},
		{Send: msg(`{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"tu1","content":"file body","is_error":true}]},"parent_tool_use_id":null,"session_id":"s1"}`)},
		{Send: msg(`{"type":"stream_event","event":{"type":"content_block_delta"},"session_id":"s1"}`)},
		{Send: msg(resultLine)},
		{Send: msg(assistantLine)}, // after the result: never yielded
	}
	opts, read := newFake(t, fakeScript{Steps: steps})
	msgs := mustRun(t, opts)

	var types []string
	for _, m := range msgs {
		types = append(types, m.Type)
	}
	if want := []string{"system", "assistant", "user", "stream_event", "result"}; !slices.Equal(types, want) {
		t.Fatalf("types = %q, want %q (unknown types skipped, nothing after the result)", types, want)
	}
	if m := msgs[0]; m.Subtype != "init" || !bytes.Contains(m.Raw, []byte(`"tools":["Read"]`)) {
		t.Errorf("system = %+v", m)
	}
	a := msgs[1].Assistant
	if a == nil {
		t.Fatal("assistant message not parsed")
	}
	if a.Model != "claude-x" || a.ParentToolUseID != "p1" ||
		a.Usage != (Usage{InputTokens: 10, OutputTokens: 5, CacheReadInputTokens: 2, CacheCreationInputTokens: 1}) {
		t.Errorf("assistant = %+v", a)
	}
	if len(a.Content) != 3 || a.Content[0].Thinking != "hmm" || a.Content[1].Text != "reading" ||
		a.Content[2].Type != "tool_use" || a.Content[2].ID != "tu1" || a.Content[2].Name != "Read" ||
		!jsonEqual(t, a.Content[2].Input, []byte(`{"file_path":"a.txt"}`)) {
		t.Errorf("assistant content = %+v", a.Content)
	}
	u := msgs[2].User
	if u == nil || len(u.Content) != 1 || u.Content[0].ToolUseID != "tu1" || !u.Content[0].IsError ||
		!jsonEqual(t, u.Content[0].Content, []byte(`"file body"`)) {
		t.Errorf("user = %+v", u)
	}
	if !bytes.Contains(msgs[3].Raw, []byte("content_block_delta")) {
		t.Errorf("stream_event raw = %s", msgs[3].Raw)
	}
	res := msgs[4].Result
	if res == nil || *res != (Result{Subtype: "success", Result: "done", NumTurns: 1, StopReason: "end_turn", Usage: Usage{InputTokens: 3, OutputTokens: 4}}) {
		t.Errorf("result = %+v", res)
	}
	if r := read(); !r.EOF {
		t.Error("stdin was not closed after the result")
	}
}

// A user message whose content is a plain string parses with no blocks.
func TestUserStringContent(t *testing.T) {
	opts, _ := newFake(t, fakeScript{Steps: []fakeStep{
		{Send: msg(`{"type":"user","message":{"role":"user","content":"plain"},"parent_tool_use_id":null}`)},
		{Send: msg(resultLine)},
	}})
	msgs := mustRun(t, opts)
	if len(msgs) != 2 || msgs[0].User == nil || len(msgs[0].User.Content) != 0 {
		t.Errorf("msgs = %+v", msgs)
	}
}

// controlResponses maps request id to the response body the SDK sent.
func controlResponses(t *testing.T, r fakeRecord) map[string]map[string]json.RawMessage {
	t.Helper()
	out := map[string]map[string]json.RawMessage{}
	for _, e := range r.Recv {
		var m struct {
			Type     string                     `json:"type"`
			Response map[string]json.RawMessage `json:"response"`
		}
		if json.Unmarshal(e.Line, &m) != nil || m.Type != "control_response" {
			continue
		}
		var id string
		_ = json.Unmarshal(m.Response["request_id"], &id)
		out[id] = m.Response
	}
	return out
}

func subtypeOf(resp map[string]json.RawMessage) string {
	var s string
	_ = json.Unmarshal(resp["subtype"], &s)
	return s
}

func TestHookCallback(t *testing.T) {
	steps := []fakeStep{
		{Hook: HookPreToolUse, HookInput: msg(`{"hook_event_name":"PreToolUse","tool_name":"Read","tool_use_id":"tu1","tool_input":{"file_path":"a"}}`)},
		{Hook: HookPostToolUse, HookInput: msg(`{"hook_event_name":"PostToolUse","tool_name":"Read","tool_use_id":"tu1","tool_input":{"file_path":"a"},"tool_response":"body"}`)},
		{Hook: HookPostToolUseFailure, HookInput: msg(`{"hook_event_name":"PostToolUseFailure","tool_name":"Bash","tool_use_id":"tu2","error":"boom"}`)},
		{Request: msg(`{"subtype":"hook_callback","callback_id":"nope","input":{}}`)},
		{Send: msg(resultLine)},
	}
	opts, read := newFake(t, fakeScript{Steps: steps})
	var mu sync.Mutex
	var got []HookInput
	opts.OnHook = func(_ context.Context, in HookInput) error {
		mu.Lock()
		defer mu.Unlock()
		got = append(got, in)
		if in.Event == HookPostToolUseFailure {
			return errors.New("hook failed")
		}
		return nil
	}
	mustRun(t, opts)

	if len(got) != 3 {
		t.Fatalf("OnHook called %d times, want 3", len(got))
	}
	if g := got[0]; g.Event != HookPreToolUse || g.ToolName != "Read" || g.ToolUseID != "tu1" || !jsonEqual(t, g.ToolInput, []byte(`{"file_path":"a"}`)) {
		t.Errorf("pre = %+v", g)
	}
	if g := got[1]; g.Event != HookPostToolUse || !jsonEqual(t, g.ToolResponse, []byte(`"body"`)) {
		t.Errorf("post = %+v", g)
	}
	if g := got[2]; g.Event != HookPostToolUseFailure || g.ToolName != "Bash" || g.Error != "boom" {
		t.Errorf("failure = %+v", g)
	}
	resp := controlResponses(t, read())
	for id, want := range map[string]string{"cli_1": "success", "cli_2": "success", "cli_3": "error", "cli_4": "error"} {
		if got := subtypeOf(resp[id]); got != want {
			t.Errorf("response to %s = %s, want %s", id, got, want)
		}
	}
	if !jsonEqual(t, resp["cli_1"]["response"], []byte(`{}`)) {
		t.Errorf("hook response body = %s, want {}", resp["cli_1"]["response"])
	}
	if !bytes.Contains(resp["cli_3"]["error"], []byte("hook failed")) {
		t.Errorf("hook error = %s, want the callback's error", resp["cli_3"]["error"])
	}
}

func TestMCPMessage(t *testing.T) {
	steps := []fakeStep{
		{Request: msg(`{"subtype":"mcp_message","server_name":"tool-mcp","message":{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"lookup","arguments":{"q":"x"}}}}`)},
		{Request: msg(`{"subtype":"mcp_message","server_name":"tool-mcp","message":{"jsonrpc":"2.0","method":"notifications/initialized"}}`)},
		{Request: msg(`{"subtype":"mcp_message","server_name":"tool-mcp","message":{"jsonrpc":"2.0","id":2,"method":"fail"}}`)},
		{Send: msg(resultLine)},
	}
	t.Run("answered", func(t *testing.T) {
		opts, read := newFake(t, fakeScript{Steps: steps})
		var servers []string
		var mu sync.Mutex
		opts.OnMCP = func(_ context.Context, server string, m json.RawMessage) (json.RawMessage, error) {
			mu.Lock()
			servers = append(servers, server)
			mu.Unlock()
			switch {
			case bytes.Contains(m, []byte("tools/call")) && bytes.Contains(m, []byte(`"q":"x"`)):
				return json.RawMessage(`{"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"found"}]}}`), nil
			case bytes.Contains(m, []byte("notifications/")):
				return nil, nil
			}
			return nil, errors.New("mcp failed")
		}
		mustRun(t, opts)
		if !slices.Equal(servers, []string{"tool-mcp", "tool-mcp", "tool-mcp"}) {
			t.Errorf("servers = %q", servers)
		}
		resp := controlResponses(t, read())
		if !jsonEqual(t, resp["cli_1"]["response"], []byte(`{"mcp_response":{"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"found"}]}}}`)) {
			t.Errorf("tools/call response = %s", resp["cli_1"]["response"])
		}
		if !jsonEqual(t, resp["cli_2"]["response"], []byte(`{"mcp_response":{"jsonrpc":"2.0","result":{}}}`)) {
			t.Errorf("notification response = %s", resp["cli_2"]["response"])
		}
		if subtypeOf(resp["cli_3"]) != "error" || !bytes.Contains(resp["cli_3"]["error"], []byte("mcp failed")) {
			t.Errorf("failed response = %v", resp["cli_3"])
		}
	})
	t.Run("no OnMCP", func(t *testing.T) {
		opts, read := newFake(t, fakeScript{Steps: steps})
		mustRun(t, opts)
		resp := controlResponses(t, read())
		for _, id := range []string{"cli_1", "cli_2", "cli_3"} {
			if subtypeOf(resp[id]) != "error" {
				t.Errorf("response to %s = %v, want error", id, resp[id])
			}
		}
	})
}

// No permission-prompt tool is configured, so can_use_tool is refused, as
// is any request subtype the SDK does not handle.
func TestUnhandledControlRequests(t *testing.T) {
	opts, read := newFake(t, fakeScript{Steps: []fakeStep{
		{Request: msg(`{"subtype":"can_use_tool","tool_name":"Bash","input":{"command":"ls"}}`)},
		{Request: msg(`{"subtype":"something_new"}`)},
		{Send: msg(resultLine)},
	}})
	opts.OnHook = func(context.Context, HookInput) error { return nil }
	mustRun(t, opts)
	resp := controlResponses(t, read())
	for _, id := range []string{"cli_1", "cli_2"} {
		if subtypeOf(resp[id]) != "error" {
			t.Errorf("response to %s = %v, want error", id, resp[id])
		}
	}
}

func TestVersion(t *testing.T) {
	tests := []struct {
		version string
		wantErr bool
		isOld   bool
	}{
		{"2.0.0 (Claude Code)", false, false},
		{"10.0.0 (Claude Code)", false, false}, // numeric, not lexical
		{"1.9.9 (Claude Code)", true, true},
		{"not a version", true, false},
	}
	for _, tt := range tests {
		t.Run(tt.version, func(t *testing.T) {
			s := resultOnly()
			s.Version = tt.version
			opts, read := newFake(t, s)
			_, err := collect(context.Background(), opts, "hello")
			r := read()
			if r.Versions != 1 {
				t.Errorf("-v ran %d times, want 1", r.Versions)
			}
			if !tt.wantErr {
				if err != nil || r.Starts != 1 {
					t.Errorf("err = %v, starts = %d; want a run", err, r.Starts)
				}
				return
			}
			if err == nil || r.Starts != 0 {
				t.Fatalf("err = %v, starts = %d; want an error and no run", err, r.Starts)
			}
			if errors.Is(err, ErrVersion) != tt.isOld {
				t.Errorf("errors.Is(err, ErrVersion) = %v, want %v (err %v)", !tt.isOld, tt.isOld, err)
			}
			if tt.isOld && (!strings.Contains(err.Error(), "1.9.9") || !strings.Contains(err.Error(), MinVersion)) {
				t.Errorf("err = %v, want both versions named", err)
			}
		})
	}
}

func TestCLINotFound(t *testing.T) {
	_, err := collect(context.Background(), Options{Path: filepath.Join(t.TempDir(), "no-claude")}, "hello")
	if err == nil {
		t.Fatal("want an error for a missing CLI")
	}
}

func interruptSent(t *testing.T, r fakeRecord) bool {
	t.Helper()
	for _, e := range r.Recv {
		var m struct {
			Type    string            `json:"type"`
			Request map[string]string `json:"request"`
		}
		if json.Unmarshal(e.Line, &m) == nil && m.Type == "control_request" && m.Request["subtype"] == "interrupt" {
			return true
		}
	}
	return false
}

// A consumer that stops early abandons the run: interrupt, close stdin,
// then kill the group once the grace period passes.
func TestStopEarly(t *testing.T) {
	opts, read := newFake(t, fakeScript{Steps: []fakeStep{{Send: msg(assistantLine)}, {Hang: true}}})
	start := time.Now()
	for m, err := range Run(context.Background(), opts, "hello") {
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
		if m.Type == "assistant" {
			break
		}
	}
	if d := time.Since(start); d > killGrace+5*time.Second {
		t.Errorf("stopping took %v", d)
	}
	r := read()
	if !interruptSent(t, r) {
		t.Error("no interrupt sent")
	}
	if !r.EOF {
		t.Error("stdin not closed")
	}
	waitGone(t, r.PID)
}

func TestCancel(t *testing.T) {
	opts, read := newFake(t, fakeScript{Steps: []fakeStep{{Send: msg(assistantLine)}, {Hang: true}}})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var last error
	for m, err := range Run(ctx, opts, "hello") {
		if err != nil {
			last = err
			continue
		}
		if m.Type == "assistant" {
			cancel()
		}
	}
	if !errors.Is(last, context.Canceled) {
		t.Errorf("last error = %v, want context.Canceled", last)
	}
	r := read()
	if !interruptSent(t, r) {
		t.Error("no interrupt sent")
	}
	waitGone(t, r.PID)
}

func TestEarlyExitIncludesStderr(t *testing.T) {
	opts, _ := newFake(t, fakeScript{Steps: []fakeStep{
		{StderrBytes: 1 << 20, Stderr: "TAIL-MARKER"},
		{Exit: exitCode(2)},
	}})
	_, err := collect(context.Background(), opts, "hello")
	if err == nil || !strings.Contains(err.Error(), "TAIL-MARKER") {
		t.Fatalf("err = %v, want the stderr tail", err)
	}
	if n := len(err.Error()); n > 16<<10 {
		t.Errorf("error is %d bytes, want the stderr capped", n)
	}
}

func TestExitWithoutResult(t *testing.T) {
	opts, _ := newFake(t, fakeScript{Steps: []fakeStep{{Send: msg(assistantLine)}, {Exit: exitCode(0)}}})
	msgs, err := collect(context.Background(), opts, "hello")
	if err == nil {
		t.Errorf("want an error for a CLI that exits before its result, got %d messages", len(msgs))
	}
}

func TestMalformedOutput(t *testing.T) {
	opts, _ := newFake(t, fakeScript{Steps: []fakeStep{{SendRaw: "not json"}, {Send: msg(resultLine)}}})
	_, err := collect(context.Background(), opts, "hello")
	if err == nil {
		t.Error("want an error for a line that is not JSON")
	}
}
