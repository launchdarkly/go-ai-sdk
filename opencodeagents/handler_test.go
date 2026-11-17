package opencodeagents

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/neverforward"
)

// The factory checks what it can without touching the filesystem.
func TestNewHandlerValidatesOptions(t *testing.T) {
	f := newFake(t, fakeSpec{})
	cases := []struct {
		name string
		edit func(*Options)
	}{
		{"empty WorkDir", func(o *Options) { o.WorkDir = " " }},
		{"empty ConfigPath", func(o *Options) { o.ConfigPath = "" }},
		{"nil Env", func(o *Options) { o.Env = nil }},
		{"empty command part", func(o *Options) { o.Command = []string{os.Args[0], ""} }},
		{"URL command", func(o *Options) { o.Command = []string{"http://127.0.0.1:4096"} }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			o := f.options()
			tc.edit(&o)
			if h, err := NewHandler(o); err == nil || h != nil {
				t.Errorf("NewHandler = %v, %v; want an error", h, err)
			}
		})
	}
	o := f.options()
	o.Env = []string{} // empty, not nil: a valid empty environment
	o.CaptureContent = aiserver.CaptureOff
	h, err := NewHandler(o)
	if err != nil {
		t.Fatal(err)
	}
	if h.ProvidesFor != (aiserver.ProvidesFor{Provider: "*", Mode: aiserver.ModeAgent}) || h.CaptureContent != aiserver.CaptureOff || h.Stream == nil {
		t.Errorf("handler = %+v", h)
	}
	if len(f.states(t)) != 0 {
		t.Error("NewHandler started a process")
	}
}

// A missing working directory or config file fails the call, not the factory,
// and starts no process.
func TestMissingPathsFailTheCall(t *testing.T) {
	setup(t, nil)
	for _, name := range []string{"WorkDir", "ConfigPath"} {
		t.Run(name, func(t *testing.T) {
			f := newFake(t, fakeSpec{Events: reply("hi", 1, 1)})
			o := f.options()
			if name == "WorkDir" {
				o.WorkDir = filepath.Join(f.dir, "missing")
			} else {
				o.ConfigPath = filepath.Join(f.dir, "missing.json")
			}
			h, err := NewHandler(o)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := invoke(t, h, aiserver.HandlerRequest{Config: cfg(nil), UserInput: "q"}); err == nil {
				t.Error("want a call error")
			}
			if len(f.states(t)) != 0 {
				t.Error("a process was started")
			}
		})
	}
}

// One call: start the child on a free loopback port, open /event, create the
// session, post the message, read until this session is idle, stop the child.
func TestInvokeRunsOneSession(t *testing.T) {
	setup(t, nil)
	f := newFake(t, fakeSpec{Events: []map[string]any{assistant("m1"), delta("m1", "p1", "Hello"), delta("m1", "p1", " world"), completed("m1", 3, 5), idle("ses")}})
	o := f.options()
	o.Agent = "build"
	h, err := NewHandler(o)
	if err != nil {
		t.Fatal(err)
	}
	res, err := invoke(t, h, aiserver.HandlerRequest{Config: cfg(nil), UserInput: "hi there"})
	if err != nil {
		t.Fatal(err)
	}
	if res.Output != "Hello world" {
		t.Errorf("output = %#v", res.Output)
	}
	if u := res.Usage; u["input_tokens"] != 3 && u["input_tokens"] != float64(3) || u["output_tokens"] != 5 && u["output_tokens"] != float64(5) {
		t.Errorf("usage = %v", u)
	}
	s := f.state(t)
	if len(s.Argv) != 6 || !reflect.DeepEqual(s.Argv[1:3], []string{"serve", "--port"}) || !reflect.DeepEqual(s.Argv[4:], []string{"--hostname", "127.0.0.1"}) {
		t.Errorf("argv = %q", s.Argv)
	}
	if real, _ := filepath.EvalSymlinks(f.dir); s.Cwd != f.dir && s.Cwd != real {
		t.Errorf("cwd = %q, want %q", s.Cwd, f.dir)
	}
	if v, _ := s.env("OPENCODE_CONFIG"); v != f.cfg || !filepath.IsAbs(v) {
		t.Errorf("OPENCODE_CONFIG = %q", v)
	}
	if v, _ := s.env("OPENCODE_SERVER_USERNAME"); v != "opencode" {
		t.Errorf("username = %q", v)
	}
	if v, _ := s.env("OPENCODE_SERVER_PASSWORD"); len(v) < 32 {
		t.Errorf("password %q is too short", v)
	}
	if s.AuthFailures != 0 {
		t.Errorf("%d requests without basic auth", s.AuthFailures)
	}
	order := strings.Join(s.Requests, ",")
	ev, sess, msg := strings.Index(order, "GET /event"), strings.Index(order, "POST /session,"), strings.Index(order, "POST /session/ses/message")
	if ev < 0 || sess < 0 || msg < 0 || !(ev < sess && sess < msg) {
		t.Errorf("request order = %s", order)
	}
	if len(s.Messages) != 1 {
		t.Fatalf("messages = %v", s.Messages)
	}
	m := s.Messages[0]
	if !reflect.DeepEqual(m["parts"], []any{map[string]any{"type": "text", "text": "hi there"}}) ||
		!reflect.DeepEqual(m["model"], map[string]any{"providerID": "openai", "modelID": "test-model"}) ||
		m["system"] != "be brief" || m["agent"] != "build" {
		t.Errorf("message body = %v", m)
	}
	if len(s.Aborts) != 0 {
		t.Errorf("a finished session was aborted: %v", s.Aborts)
	}
	assertGone(t, s)
}

// Streaming yields text chunks, then exactly one done with the full output.
func TestStream(t *testing.T) {
	setup(t, nil)
	f := newFake(t, fakeSpec{Events: []map[string]any{assistant("m1"), delta("m1", "p1", "a"), delta("m1", "p1", "b"), completed("m1", 1, 2), idle("ses")}})
	var chunks []string
	var dones []string
	for ev, err := range f.handler(t).Stream(bg, aiserver.HandlerRequest{Config: cfg(nil), UserInput: "q"}) {
		if err != nil {
			t.Fatal(err)
		}
		switch ev.Type {
		case aiserver.EventChunk:
			chunks = append(chunks, ev.Text)
		case aiserver.EventDone:
			if ev.Output != nil {
				dones = append(dones, *ev.Output)
			} else {
				dones = append(dones, "<nil>")
			}
		}
	}
	if strings.Join(chunks, "") != "ab" || !reflect.DeepEqual(dones, []string{"ab"}) {
		t.Errorf("chunks = %q, done = %q", chunks, dones)
	}
	assertGone(t, f.state(t))
}

// Env replaces the parent environment; the three OPENCODE_* values replace
// any the caller passed.
func TestChildEnvironmentIsReplaced(t *testing.T) {
	setup(t, nil)
	t.Setenv("LD_OPENCODE_PARENT_CANARY", "leaked")
	f := newFake(t, fakeSpec{Events: reply("ok", 1, 1)})
	o := f.options()
	o.Env = append(o.Env, "FOO=bar", "OPENCODE_SERVER_PASSWORD=caller", "OPENCODE_SERVER_USERNAME=caller", "OPENCODE_CONFIG=/elsewhere.json")
	h, err := NewHandler(o)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := invoke(t, h, aiserver.HandlerRequest{Config: cfg(nil), UserInput: "q"}); err != nil {
		t.Fatal(err)
	}
	s := f.state(t)
	if _, ok := s.env("LD_OPENCODE_PARENT_CANARY"); ok {
		t.Error("the parent environment reached the child")
	}
	if v, _ := s.env("FOO"); v != "bar" {
		t.Errorf("FOO = %q", v)
	}
	for k, bad := range map[string]string{"OPENCODE_SERVER_PASSWORD": "caller", "OPENCODE_SERVER_USERNAME": "caller", "OPENCODE_CONFIG": "/elsewhere.json"} {
		if v, _ := s.env(k); v == bad {
			t.Errorf("%s kept the caller's value", k)
		}
	}
	for _, k := range []string{"OPENCODE_SERVER_PASSWORD", "OPENCODE_SERVER_USERNAME", "OPENCODE_CONFIG"} {
		n := 0
		for _, kv := range s.Env {
			if strings.HasPrefix(kv, k+"=") {
				n++
			}
		}
		if n != 1 {
			t.Errorf("%s appears %d times", k, n)
		}
	}
}

// Every call gets a fresh secret.
func TestEachCallGetsItsOwnProcessAndPassword(t *testing.T) {
	setup(t, nil)
	f := newFake(t, fakeSpec{Events: reply("ok", 1, 1)})
	h := f.handler(t)
	for range 2 {
		if _, err := invoke(t, h, aiserver.HandlerRequest{Config: cfg(nil), UserInput: "q"}); err != nil {
			t.Fatal(err)
		}
	}
	all := f.states(t)
	if len(all) != 2 || all[0].PID == all[1].PID {
		t.Fatalf("processes = %d", len(all))
	}
	p0, _ := all[0].env("OPENCODE_SERVER_PASSWORD")
	p1, _ := all[1].env("OPENCODE_SERVER_PASSWORD")
	if p0 == p1 {
		t.Error("two calls shared a password")
	}
	for _, s := range all {
		assertGone(t, s)
	}
}

func TestDefaultModel(t *testing.T) {
	cases := []struct {
		provider, name, region string
		wantProvider, wantID   string
	}{
		{"OpenAI", "gpt-5", "", "openai", "gpt-5"},
		{"Anthropic", "claude-sonnet-4-5", "", "anthropic", "claude-sonnet-4-5"},
		{"Gemini", "gemini-2.5-pro", "", "google", "gemini-2.5-pro"},
		{"Bedrock", "anthropic.claude-sonnet-4-5", "", "amazon-bedrock", "anthropic.claude-sonnet-4-5"},
		{"bedrock", "anthropic.claude-sonnet-4-5", "us", "amazon-bedrock", "us.anthropic.claude-sonnet-4-5"},
		{"BEDROCK", "us.anthropic.claude-sonnet-4-5", "us", "amazon-bedrock", "us.anthropic.claude-sonnet-4-5"},
		{"OpenAI", "openrouter/qwen/qwen3", "", "openrouter", "qwen/qwen3"},
		{"OpenAI", "bedrock/anthropic.claude-haiku", "eu", "amazon-bedrock", "eu.anthropic.claude-haiku"},
		{"Anthropic", "amazon-bedrock/anthropic.claude-haiku", "us", "amazon-bedrock", "us.anthropic.claude-haiku"},
		{"OpenAI", "gpt-5", "us", "openai", "gpt-5"}, // only amazon-bedrock takes the region
	}
	for _, tc := range cases {
		c := aiserver.AiConfig{Model: aiserver.Model{Name: tc.name, Region: tc.region}}
		c.Provider.Name = tc.provider
		p, m, err := defaultModel(&c)
		if err != nil || p != tc.wantProvider || m != tc.wantID {
			t.Errorf("%s %s (%q) = %q %q %v, want %q %q", tc.provider, tc.name, tc.region, p, m, err, tc.wantProvider, tc.wantID)
		}
	}
}

// Options.Model overrides the mapping; its error fails the call before a
// process starts. The sent id is what telemetry reports.
func TestModelResolver(t *testing.T) {
	e := setup(t, nil)
	f := newFake(t, fakeSpec{Events: reply("ok", 1, 1)})
	o := f.options()
	o.Model = func(c *aiserver.AiConfig) (string, string, error) { return "custom", "m-" + c.Model.Name, nil }
	h, _ := NewHandler(o)
	if _, err := invoke(t, h, aiserver.HandlerRequest{Config: cfg(nil), UserInput: "q"}); err != nil {
		t.Fatal(err)
	}
	if got := f.state(t).Messages[0]["model"]; !reflect.DeepEqual(got, map[string]any{"providerID": "custom", "modelID": "m-test-model"}) {
		t.Errorf("model = %v", got)
	}
	if root := e.spans(SpanName); len(root) != 1 || attrs(root[0])["gen_ai.request.model"] != "m-test-model" {
		t.Errorf("root spans = %d", len(root))
	}

	boom := errors.New("no model")
	f2 := newFake(t, fakeSpec{Events: reply("ok", 1, 1)})
	o2 := f2.options()
	o2.Model = func(*aiserver.AiConfig) (string, string, error) { return "", "", boom }
	h2, _ := NewHandler(o2)
	if _, err := invoke(t, h2, aiserver.HandlerRequest{Config: cfg(nil), UserInput: "q"}); !errors.Is(err, boom) {
		t.Errorf("err = %v", err)
	}
	if len(f2.states(t)) != 0 {
		t.Error("a process was started")
	}
}

// Events from other sessions are dropped; an idle for another session, or
// with no session id, does not end the run.
func TestOtherSessionsAreIgnored(t *testing.T) {
	setup(t, nil)
	f := newFake(t, fakeSpec{Events: []map[string]any{
		otherSession(assistant("x1")), otherSession(delta("x1", "px", "NOPE")), otherSession(idle("ses")), idle(""),
		assistant("m1"), delta("m1", "p1", "mine"), completed("m1", 1, 1), idle("ses"),
	}})
	res, err := invoke(t, f.handler(t), aiserver.HandlerRequest{Config: cfg(nil), UserInput: "q"})
	if err != nil || res.Output != "mine" {
		t.Errorf("output = %#v, err = %v", res.Output, err)
	}
}

// With no idle, the turn ends after a quiet drain window once the message
// POST has returned, keeping text that arrived after it.
func TestQuietBusEndsTheTurn(t *testing.T) {
	setup(t, nil)
	f := newFake(t, fakeSpec{Events: []map[string]any{assistant("m1"), delta("m1", "p1", "late"), completed("m1", 1, 1)}})
	start := time.Now()
	res, err := invoke(t, f.handler(t), aiserver.HandlerRequest{Config: cfg(nil), UserInput: "q"})
	if err != nil || res.Output != "late" {
		t.Errorf("output = %#v, err = %v", res.Output, err)
	}
	if d := time.Since(start); d > 5*time.Second {
		t.Errorf("took %s", d)
	}
}

// A failed message POST fails the call at once, even with /event still open,
// and aborts the session.
func TestFailedMessagePostFailsAtOnce(t *testing.T) {
	setup(t, nil)
	short(t, &drainFallback, time.Minute)
	f := newFake(t, fakeSpec{Status: 500})
	start := time.Now()
	_, err := invoke(t, f.handler(t), aiserver.HandlerRequest{Config: cfg(nil), UserInput: "q"})
	var se *SessionError
	if err == nil || errors.As(err, &se) || !strings.Contains(err.Error(), "500") {
		t.Errorf("err = %v", err)
	}
	if d := time.Since(start); d > 5*time.Second {
		t.Errorf("waited %s for the bus", d)
	}
	s := f.state(t)
	if len(s.Aborts) != 1 {
		t.Errorf("aborts = %v", s.Aborts)
	}
	assertGone(t, s)
}

// session.error fails the run with the error name and never the provider body.
func TestSessionErrorCarriesTheNameOnly(t *testing.T) {
	e := setup(t, nil)
	f := newFake(t, fakeSpec{Events: []map[string]any{assistant("m1"), sessionError("ProviderAuthError", "secret-provider-body")}})
	_, err := invoke(t, f.handler(t), aiserver.HandlerRequest{Config: cfg(nil), UserInput: "q"})
	var se *SessionError
	if !errors.As(err, &se) || se.Name != "ProviderAuthError" {
		t.Fatalf("err = %v", err)
	}
	if strings.Contains(err.Error(), "secret-provider-body") || !strings.Contains(err.Error(), "ProviderAuthError") {
		t.Errorf("message = %q", err.Error())
	}
	root := e.spans(SpanName)
	if len(root) != 1 || root[0].Status().Code.String() != "Error" {
		t.Errorf("root = %v", root)
	}
	for _, s := range e.exp.GetSpans().Snapshots() {
		for _, ev := range s.Events() {
			for _, a := range ev.Attributes {
				if strings.Contains(a.Value.Emit(), "secret-provider-body") {
					t.Errorf("span %s carries the provider body", s.Name())
				}
			}
		}
	}
	assertGone(t, f.state(t))
}

// Stopping a stream early aborts the session and stops the child.
func TestStreamStoppedEarly(t *testing.T) {
	e := setup(t, nil)
	f := newFake(t, fakeSpec{HoldMessage: true, Events: []map[string]any{assistant("m1"), delta("m1", "p1", "first")}})
	for ev, err := range f.handler(t).Stream(bg, aiserver.HandlerRequest{Config: cfg(nil), UserInput: "q"}) {
		if err != nil {
			t.Fatal(err)
		}
		if ev.Type == aiserver.EventChunk {
			break
		}
	}
	s := f.state(t)
	if len(s.Aborts) != 1 {
		t.Errorf("aborts = %v", s.Aborts)
	}
	assertGone(t, s)
	if root := e.spans(SpanName); len(root) != 1 || attrs(root[0])["launchdarkly.stream.abandoned"] != "true" {
		t.Errorf("root = %v", root)
	}
}

// Cancelling the context ends the call with the context's error and stops
// the child.
func TestCancelStopsTheCall(t *testing.T) {
	setup(t, nil)
	f := newFake(t, fakeSpec{HoldMessage: true, Events: []map[string]any{assistant("m1"), delta("m1", "p1", "first")}})
	ctx, cancel := context.WithTimeout(bg, 1500*time.Millisecond)
	defer cancel()
	_, err := f.handler(t).Invoke(ctx, aiserver.HandlerRequest{Config: cfg(nil), UserInput: "q"})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("err = %v", err)
	}
	s := f.state(t)
	if len(s.Aborts) != 1 {
		t.Errorf("aborts = %v", s.Aborts)
	}
	assertGone(t, s)
}

// A child that exits during startup fails the call with its stderr tail.
func TestStartupFailureIncludesStderr(t *testing.T) {
	setup(t, nil)
	f := newFake(t, fakeSpec{ExitStderr: strings.Repeat("x", 4096) + "the-last-words"})
	_, err := invoke(t, f.handler(t), aiserver.HandlerRequest{Config: cfg(nil), UserInput: "q"})
	if err == nil || !strings.Contains(err.Error(), "the-last-words") || strings.Count(err.Error(), "x") > 2100 {
		t.Errorf("err = %.200v", err)
	}
}

// Health needs 200 and healthy:true; otherwise startup times out and the
// child is stopped.
func TestHealthNeedsAHealthyBody(t *testing.T) {
	setup(t, nil)
	short(t, &startupTimeout, 500*time.Millisecond)
	f := newFake(t, fakeSpec{Unhealthy: true})
	if _, err := invoke(t, f.handler(t), aiserver.HandlerRequest{Config: cfg(nil), UserInput: "q"}); err == nil {
		t.Fatal("want a startup error")
	}
	s := f.state(t)
	if len(s.Sessions) != 0 {
		t.Error("a session was opened on an unhealthy server")
	}
	assertGone(t, s)
}

// The bus reducer: what reaches the output, per the Python reducer's rules.
func TestReducer(t *testing.T) {
	cases := []struct {
		name   string
		events []map[string]any
		want   string
	}{
		{"a snapshot does not repeat deltas", []map[string]any{assistant("m1"), delta("m1", "p1", "Hel"), delta("m1", "p1", "lo"), snapshot("m1", "p1", "Hello"), completed("m1", 1, 1), idle("ses")}, "Hello"},
		{"a snapshot that does not extend the text adds nothing", []map[string]any{assistant("m1"), delta("m1", "p1", "abc"), snapshot("m1", "p1", "wxyz"), completed("m1", 1, 1), idle("ses")}, "abc"},
		{"later deltas append to a snapshot", []map[string]any{assistant("m1"), snapshot("m1", "p1", "Hi"), delta("m1", "p1", " there"), completed("m1", 1, 1), idle("ses")}, "Hi there"},
		{"user text is not the reply", []map[string]any{userMessage("u1"), delta("u1", "pu", "question"), assistant("m1"), delta("m1", "p1", "answer"), completed("m1", 1, 1), idle("ses")}, "answer"},
		{"a delta without a part id is dropped", []map[string]any{assistant("m1"), delta("m1", "", "dropped"), delta("m1", "p1", "kept"), completed("m1", 1, 1), idle("ses")}, "kept"},
		{"text before its role waits for it", []map[string]any{delta("m1", "p1", "early"), assistant("m1"), completed("m1", 1, 1), idle("ses")}, "early"},
		{"unfinished text is kept at the end", []map[string]any{assistant("m1"), delta("m1", "p1", "partial"), idle("ses")}, "partial"},
		{"two messages", []map[string]any{assistant("m1"), delta("m1", "p1", "one "), completed("m1", 1, 1), assistant("m2"), delta("m2", "p2", "two"), completed("m2", 1, 1), idle("ses")}, "one two"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			setup(t, nil)
			f := newFake(t, fakeSpec{Events: tc.events})
			res, err := invoke(t, f.handler(t), aiserver.HandlerRequest{Config: cfg(nil), UserInput: "q"})
			if err != nil || res.Output != tc.want {
				t.Errorf("output = %#v, want %q (err %v)", res.Output, tc.want, err)
			}
		})
	}
}

// Usage is each completed assistant message's tokens, counted once.
func TestUsageIsSummedOncePerMessage(t *testing.T) {
	setup(t, nil)
	f := newFake(t, fakeSpec{Events: []map[string]any{
		assistant("m1"), delta("m1", "p1", "a"), completed("m1", 2, 3), completed("m1", 2, 3),
		assistant("m2"), delta("m2", "p2", "b"), completed("m2", 10, 20), idle("ses"),
	}})
	res, err := invoke(t, f.handler(t), aiserver.HandlerRequest{Config: cfg(nil), UserInput: "q"})
	if err != nil {
		t.Fatal(err)
	}
	got := map[string]float64{}
	for k, v := range res.Usage {
		switch n := v.(type) {
		case int:
			got[k] = float64(n)
		case float64:
			got[k] = n
		}
	}
	if got["input_tokens"] != 12 || got["output_tokens"] != 23 {
		t.Errorf("usage = %v", res.Usage)
	}
}

// Instructions (with variables) are the system text. Config messages, history
// and the user input become the prompt; a lone user turn is sent bare.
func TestPromptComposition(t *testing.T) {
	setup(t, nil)
	f := newFake(t, fakeSpec{Events: reply("ok", 1, 1), Calls: [][]map[string]any{reply("ok", 1, 1), reply("ok", 1, 1)}})
	h := f.handler(t)
	c := cfg(func(c *aiserver.AiConfig) {
		c.Instructions = "Help {{name}}."
		c.Messages = []aiserver.Message{{Role: "system", Text: "dropped"}, {Role: "user", Text: "Hi, I am {{name}}"}}
	})
	req := aiserver.HandlerRequest{Config: c, UserInput: "q2", Variables: map[string]any{"name": "Ada"},
		History: []aiserver.Message{{Role: "assistant", Text: "Hello {{name}}"}}}
	if _, err := invoke(t, h, req); err != nil {
		t.Fatal(err)
	}
	if _, err := invoke(t, h, aiserver.HandlerRequest{Config: cfg(nil), UserInput: "just this"}); err != nil {
		t.Fatal(err)
	}
	all := f.states(t)
	if len(all) != 2 {
		t.Fatalf("processes = %d", len(all))
	}
	m := all[0].Messages[0]
	if m["system"] != "Help Ada." {
		t.Errorf("system = %v", m["system"])
	}
	want := "User: Hi, I am Ada\n\nAssistant: Hello {{name}}\n\nUser: q2"
	if got := m["parts"].([]any)[0].(map[string]any)["text"]; got != want {
		t.Errorf("prompt = %q, want %q", got, want)
	}
	if got := all[1].Messages[0]["parts"].([]any)[0].(map[string]any)["text"]; got != "just this" {
		t.Errorf("bare prompt = %q", got)
	}
}

// No model.parameters key reaches the message body.
func TestNeverForwardedParameters(t *testing.T) {
	setup(t, nil)
	f := newFake(t, fakeSpec{Events: reply("ok", 1, 1)})
	params := map[string]any{"temperature": 0.2, "max_tokens": 10, "tool_choice": "auto", "variant": "high", "format": map[string]any{"type": "json"}}
	for k, v := range neverforward.Params {
		params[k] = v
	}
	c := cfg(func(c *aiserver.AiConfig) { c.Model.Parameters = params })
	if _, err := invoke(t, f.handler(t), aiserver.HandlerRequest{Config: c, UserInput: "q"}); err != nil {
		t.Fatal(err)
	}
	s := f.state(t)
	m := s.Messages[0]
	for k := range m {
		if k != "parts" && k != "model" && k != "system" {
			t.Errorf("message body has %q", k)
		}
	}
	for _, body := range append(s.Sessions, s.Messages...) {
		if b := strings.ToLower(jsonString(body)); strings.Contains(b, neverforward.Marker) {
			t.Errorf("a never-forwarded value was sent: %s", b)
		}
	}
}

// A judge routed here runs in its own process and session, with no tools.
func TestJudgeOpensItsOwnSession(t *testing.T) {
	judge := agentVariation("Judge it.", nil)
	judge["_ldMeta"] = map[string]any{"enabled": true, "variationKey": "j1", "version": 1, "mode": "judge"}
	judge["evaluationMetricKey"] = "quality"
	parent := agentVariation("Answer.", map[string]any{
		"tools":              map[string]any{"lookup": map[string]any{"name": "lookup", "type": "function", "parameters": map[string]any{}}},
		"judgeConfiguration": map[string]any{"judges": []any{map[string]any{"key": "judge", "samplingRate": 1.0}}},
	})
	e := setup(t, map[string]any{"cfg": parent, "judge": judge})
	f := newFake(t, fakeSpec{Calls: [][]map[string]any{
		reply("the answer", 1, 1),
		reply(`{"score": 0.75, "reasoning": "because"}`, 1, 1),
	}})
	res, err := OpenCodeAgents(bg, "cfg", "q", user, &InvokeOptions{Handler: f.options(),
		ToolHandlers: map[string]aiserver.ToolHandler{"lookup": OpenCodeGrep}})
	if err != nil || res.Response != "the answer" {
		t.Fatalf("res = %+v, err = %v", res, err)
	}
	if jr, ok := res.JudgeResults["judge"]; !ok || jr.Score != 0.75 || jr.Response != "because" {
		t.Errorf("judge results = %+v", res.JudgeResults)
	}
	all := f.states(t)
	if len(all) != 2 || all[0].PID == all[1].PID {
		t.Fatalf("processes = %d", len(all))
	}
	if got := permission(t, all[1]); !reflect.DeepEqual(got, []string{"deny *"}) {
		t.Errorf("judge permission = %v", got)
	}
	for _, s := range all {
		assertGone(t, s)
	}
	if n := len(e.spans(SpanName)); n != 2 {
		t.Errorf("invoke_agent spans = %d, want one per call", n)
	}
}

func jsonString(v any) string {
	b, _ := json.Marshal(v)
	return string(b)
}
