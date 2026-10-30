package claudeagents

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"go.opentelemetry.io/otel/codes"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

var ldVars = map[string]any{"__ld": aiserver.TrackData{ConfigKey: "test-config", VariationKey: "variation-a", RunID: "run-123", Version: 1, ModelName: "test-model", ProviderName: "TestProvider"}}

func TestFactoryMetadata(t *testing.T) {
	a, b := newHandler(Options{}), newHandler(Options{})
	if a.ProvidesFor != (aiserver.ProvidesFor{Provider: "Anthropic", Mode: aiserver.ModeAgent}) || a.CaptureContent != aiserver.CaptureDefault || a == b || a.Invoke == nil || a.Stream == nil {
		t.Errorf("factory metadata wrong: %+v", a)
	}
	if newHandler(Options{CaptureContent: aiserver.CaptureOn}).CaptureContent != aiserver.CaptureOn {
		t.Error("captureContent not passed through")
	}
}

// Prompt construction in agent mode: instructions, messages, empty input, precedence.
func TestPromptPaths(t *testing.T) {
	e := setup(t, nil, textResp("a", 1, 1), textResp("b", 1, 1), textResp("c", 1, 1), textResp("d", 1, 1), textResp("e", 1, 1))
	h := e.handler()
	invoke := func(c aiserver.AiConfig, input string, vars map[string]any) {
		t.Helper()
		if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: c, UserInput: input, Variables: vars}); err != nil {
			t.Fatal(err)
		}
	}
	// Instructions path: instructions → system, placeholders resolved or left as-is.
	invoke(cfg(func(c *aiserver.AiConfig) { c.Instructions = "Hi {{name}} {{missing}}" }), "q", map[string]any{"name": "Ann"})
	// Messages path: last config turn is user → no extra user turn.
	invoke(cfg(func(c *aiserver.AiConfig) {
		c.Messages = []aiserver.Message{{Role: "system", Text: "sys {{x}}"}, {Role: "system", Text: "more"}, {Role: "user", Text: "u {{x}}"}}
	}), "q", map[string]any{"x": "X"})
	// Messages path: last config turn is assistant → userInput appended.
	invoke(cfg(func(c *aiserver.AiConfig) {
		c.Messages = []aiserver.Message{{Role: "user", Text: "u"}, {Role: "assistant", Text: "a"}}
	}), "q", nil)
	// Empty input on the instructions path still appends "".
	invoke(cfg(func(c *aiserver.AiConfig) { c.Instructions = "inst" }), "", nil)
	// Precedence: agent mode prefers instructions over system messages.
	invoke(cfg(func(c *aiserver.AiConfig) {
		c.Instructions = "inst"
		c.Messages = []aiserver.Message{{Role: "system", Text: "msg-sys"}, {Role: "user", Text: "u"}, {Role: "assistant", Text: "a"}}
	}), "q", nil)

	r := e.api.reqs()
	if system(r[0]) != "Hi Ann {{missing}}" || !reflect.DeepEqual(roles(r[0]), []string{"user"}) || firstText(messages(r[0])[0]) != "q" {
		t.Errorf("instructions path: system=%q roles=%v", system(r[0]), roles(r[0]))
	}
	if system(r[1]) != "sys X\nmore" || !reflect.DeepEqual(roles(r[1]), []string{"user"}) || firstText(messages(r[1])[0]) != "u X" {
		t.Errorf("messages path user-last: system=%q roles=%v", system(r[1]), roles(r[1]))
	}
	if !reflect.DeepEqual(roles(r[2]), []string{"user", "assistant", "user"}) || firstText(messages(r[2])[2]) != "q" {
		t.Errorf("messages path assistant-last: roles=%v", roles(r[2]))
	}
	if !reflect.DeepEqual(roles(r[3]), []string{"user"}) || firstText(messages(r[3])[0]) != "" {
		t.Errorf("empty input: %v", messages(r[3]))
	}
	if system(r[4]) != "inst" || strings.Contains(system(r[4]), "msg-sys") {
		t.Errorf("precedence: system=%q", system(r[4]))
	}
}

// claude-agents best-effort outputFormat.
func TestOutputFormat(t *testing.T) {
	e := setup(t, nil, textResp(`{"name":"x"}`, 1, 1), textResp("plain", 1, 1), sse([]string{"s"}, "", 1, 1))
	h := e.handler()
	schema := map[string]any{"type": "object", "properties": map[string]any{"name": map[string]any{"type": "string"}}}
	withSchema := cfg(func(c *aiserver.AiConfig) { c.Instructions = "inst"; c.OutputFormat = schema })
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: withSchema, UserInput: "q"}); err != nil {
		t.Fatal(err)
	}
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: cfg(func(c *aiserver.AiConfig) { c.Instructions = "inst" }), UserInput: "q"}); err != nil {
		t.Fatal(err)
	}
	for range h.Stream(bg, aiserver.HandlerRequest{Config: withSchema, UserInput: "q"}) {
	}
	r := e.api.reqs()
	if s := system(r[0]); !strings.HasPrefix(s, "inst\n\n") || !strings.Contains(s, `"properties":{"name":{"type":"string"}}`) {
		t.Errorf("schema instruction missing: %q", s)
	}
	if system(r[1]) != "inst" || system(r[2]) != "inst" {
		t.Errorf("schema instruction leaked: %q / stream %q", system(r[1]), system(r[2]))
	}
}

func TestToolsAndLoop(t *testing.T) {
	e := setup(t, nil, toolResp("a", `{"city":"Paris"}`), toolResp("b", `{}`), textResp("final", 2, 3))
	params := map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}, "required": []any{"city"}, "additionalProperties": false}
	c := cfg(func(c *aiserver.AiConfig) {
		c.Instructions = "x"
		c.Tools = map[string]aiserver.Tool{
			"a": {Name: "a", Type: "function", Description: "tool a", Parameters: params},
			"b": {Name: "b", Type: "function", Parameters: map[string]any{"type": "object"}},
			"c": {Name: "c", Type: "function", Parameters: map[string]any{"type": "object"}},
		}
	})
	var calls []string
	var gotArgs map[string]any
	tools := map[string]aiserver.ToolHandler{
		"a": aiserver.ToolFunc(func(_ context.Context, args map[string]any) (any, error) {
			calls, gotArgs = append(calls, "a"), args
			return "sunny", nil
		}),
		"b": aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) {
			calls = append(calls, "b")
			return map[string]any{"ok": true}, nil
		}),
	}
	res, err := e.handler().Invoke(bg, aiserver.HandlerRequest{Config: c, UserInput: "q", ToolHandlers: tools, Variables: ldVars})
	if err != nil {
		t.Fatal(err)
	}
	r := e.api.reqs()
	if !reflect.DeepEqual(toolNames(r[0]), []string{"a", "b"}) {
		t.Fatalf("tools sent = %v", toolNames(r[0]))
	}
	a := r[0]["tools"].([]any)[0].(map[string]any)
	if !reflect.DeepEqual(a["input_schema"], params) || a["description"] != "tool a" {
		t.Errorf("tool a modified: %v", a)
	}
	if !reflect.DeepEqual(calls, []string{"a", "b"}) || gotArgs["city"] != "Paris" || res.Output != "final" {
		t.Errorf("calls=%v args=%v out=%v", calls, gotArgs, res.Output)
	}
	if u := aiserver.ParseUsage(res.Usage); u.Input != 4 || u.Output != 5 {
		t.Errorf("usage = %+v", u)
	}
	// tool results are fed back on the follow-up turn
	last := messages(r[2])
	if res := last[len(last)-1]["content"].([]any)[0].(map[string]any); res["type"] != "tool_result" || res["tool_use_id"] != "tu_b" {
		t.Errorf("tool result not sent back: %v", res)
	}
	if len(e.spans("execute_tool a")) != 1 || len(e.spans("chat test-model")) != 3 {
		t.Error("span tree wrong")
	}
}

func TestNoToolsAndExcluded(t *testing.T) {
	e := setup(t, nil, textResp("x", 1, 1), textResp("y", 1, 1))
	called := false
	fn := aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) { called = true; return nil, nil })
	if _, err := e.handler().Invoke(bg, aiserver.HandlerRequest{Config: cfg(nil), UserInput: "q", ToolHandlers: map[string]aiserver.ToolHandler{"a": fn}}); err != nil {
		t.Fatal(err)
	}
	only := cfg(func(c *aiserver.AiConfig) {
		c.Tools = map[string]aiserver.Tool{"ghost": {Name: "ghost", Parameters: map[string]any{}}}
	})
	if _, err := e.handler().Invoke(bg, aiserver.HandlerRequest{Config: only, UserInput: "q"}); err != nil {
		t.Fatal(err)
	}
	r := e.api.reqs()
	if len(r) != 2 || r[0]["tools"] != nil || r[1]["tools"] != nil || called {
		t.Errorf("tools sent without handlers: %v / %v", r[0]["tools"], r[1]["tools"])
	}
}

// A tool that throws; buildToolMCP tool-not-found names the tool.
func TestToolErrors(t *testing.T) {
	e := setup(t, nil, toolResp("a", `{}`), toolResp("ghost", `{}`))
	boom := errors.New("boom")
	c := cfg(func(c *aiserver.AiConfig) {
		c.Tools = map[string]aiserver.Tool{"a": {Name: "a", Parameters: map[string]any{}}, "ghost": {Name: "ghost", Parameters: map[string]any{}}}
	})
	tools := map[string]aiserver.ToolHandler{"a": aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) { return nil, boom })}
	if _, err := e.handler().Invoke(bg, aiserver.HandlerRequest{Config: c, ToolHandlers: tools}); !errors.Is(err, boom) {
		t.Errorf("tool error not propagated: %v", err)
	}
	_, err := e.handler().Invoke(bg, aiserver.HandlerRequest{Config: c, ToolHandlers: tools})
	if !errors.Is(err, ErrToolNotFound) || !strings.Contains(err.Error(), "ghost") {
		t.Errorf("tool-not-found error = %v", err)
	}
	if s := e.spans(SpanName); len(s) != 2 || s[0].Status().Code != codes.Error || s[1].Status().Code != codes.Error {
		t.Error("root spans not failed")
	}
}

// Agent SDK maxTurns / error_max_turns, and pause_turn continuation.
func TestStepBoundAndPauseTurn(t *testing.T) {
	e := setup(t, nil, toolResp("a", `{}`),
		msgJSON("pause_turn", `{"type":"text","text":"part 1 "}`, 1, 1), textResp("part 2", 1, 1))
	c := cfg(func(c *aiserver.AiConfig) {
		c.Tools = map[string]aiserver.Tool{"a": {Name: "a", Parameters: map[string]any{}}}
		c.Model.Parameters = map[string]any{"max_turns": float64(1)}
	})
	tools := map[string]aiserver.ToolHandler{"a": aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) { return "ok", nil })}
	if _, err := e.handler().Invoke(bg, aiserver.HandlerRequest{Config: c, ToolHandlers: tools}); !errors.Is(err, ErrMaxSteps) {
		t.Errorf("want ErrMaxSteps, got %v", err)
	}
	res, err := e.handler().Invoke(bg, aiserver.HandlerRequest{Config: cfg(nil), UserInput: "q"})
	if err != nil || res.Output != "part 1 part 2" {
		t.Errorf("pause_turn: %v %v", res.Output, err)
	}
	if r := e.api.reqs(); !reflect.DeepEqual(roles(r[2]), []string{"user", "assistant"}) {
		t.Errorf("pause_turn resend roles = %v", roles(r[2]))
	}
}

func TestTelemetry(t *testing.T) {
	e := setup(t, nil, toolResp("a", `{}`), textResp("done", 3, 4), "STATUS400")
	c := cfg(func(c *aiserver.AiConfig) {
		c.Instructions = "be nice"
		c.Tools = map[string]aiserver.Tool{"a": {Name: "a", Parameters: map[string]any{}}}
	})
	tools := map[string]aiserver.ToolHandler{"a": aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) { return "ok", nil })}
	if _, err := e.handler().Invoke(bg, aiserver.HandlerRequest{Config: c, UserInput: "hello", ToolHandlers: tools, Variables: ldVars}); err != nil {
		t.Fatal(err)
	}
	root := e.spans(SpanName)[0]
	got := attrs(root)
	for k, want := range map[string]string{
		"gen_ai.system": "anthropic", "gen_ai.operation.name": "invoke_agent", "gen_ai.request.model": "test-model",
		"gen_ai.response.model": "test-model", "gen_ai.usage.input_tokens": "4", "gen_ai.usage.output_tokens": "5",
		"gen_ai.usage.total_tokens": "9", "gen_ai.usage.prompt_tokens": "4", "gen_ai.usage.completion_tokens": "5",
		"gen_ai.prompt.0.role": "system", "gen_ai.prompt.0.content": "be nice", "gen_ai.prompt.1.content": "hello",
		"gen_ai.completion.0.role": "assistant", "gen_ai.completion.0.content": "done",
		"launchdarkly.operation.type": "gen_ai", "launchdarkly.config.key": "test-config",
		"launchdarkly.variation.key": "variation-a", "launchdarkly.run.id": "run-123",
	} {
		if got[k] != want {
			t.Errorf("root %s = %q, want %q", k, got[k], want)
		}
	}
	if _, has := got["launchdarkly.graph.key"]; has || root.Status().Code != codes.Ok || root.EndTime().IsZero() {
		t.Error("graph key set, status not OK, or not ended")
	}
	events := map[string]string{}
	for _, ev := range root.Events() {
		for _, a := range ev.Attributes {
			events[ev.Name] += a.Value.Emit()
		}
	}
	if !strings.Contains(events["gen_ai.content.prompt"], "hello") || events["gen_ai.content.completion"] != "done" {
		t.Errorf("content events = %v", events)
	}
	for _, s := range e.exp.GetSpans().Snapshots() {
		if s.Name() == SpanName {
			continue
		}
		if s.Parent().SpanID() != root.SpanContext().SpanID() {
			t.Errorf("%s is not a child of the root", s.Name())
		}
		for k := range attrs(s) {
			if strings.HasPrefix(k, "launchdarkly.") || strings.HasPrefix(k, "context.") {
				t.Errorf("child %s carries %s", s.Name(), k)
			}
		}
	}

	// provider error recorded, span ended ERROR, error returned.
	e.exp.Reset()
	_, err := e.handler().Invoke(bg, aiserver.HandlerRequest{Config: c, UserInput: "q"})
	var apiErr *anthropic.Error
	if !errors.As(err, &apiErr) {
		t.Fatalf("want provider error, got %v", err)
	}
	failed := e.spans(SpanName)[0]
	if failed.Status().Code != codes.Error || failed.EndTime().IsZero() || len(failed.Events()) == 0 || failed.Events()[len(failed.Events())-1].Name != "exception" {
		t.Errorf("error span: status=%v events=%v", failed.Status(), failed.Events())
	}
}

func TestStream(t *testing.T) {
	e := setup(t, nil, sse([]string{"Hel", "lo "}, "a", 2, 1), sse([]string{"wor", "ld"}, "", 3, 2), "STATUS400")
	var gotArgs map[string]any
	c := cfg(func(c *aiserver.AiConfig) {
		c.Tools = map[string]aiserver.Tool{"a": {Name: "a", Parameters: map[string]any{}}}
	})
	tools := map[string]aiserver.ToolHandler{"a": aiserver.ToolFunc(func(_ context.Context, args map[string]any) (any, error) { gotArgs = args; return "ok", nil })}
	var chunks []string
	var done *aiserver.HandlerStreamEvent
	for ev, err := range e.handler().Stream(bg, aiserver.HandlerRequest{Config: c, UserInput: "q", ToolHandlers: tools, Variables: ldVars}) {
		if err != nil {
			t.Fatal(err)
		}
		if done != nil {
			t.Fatal("event after done")
		}
		switch ev.Type {
		case "chunk":
			chunks = append(chunks, ev.Text)
		case "done":
			done = &ev
		}
	}
	if !reflect.DeepEqual(chunks, []string{"Hel", "lo ", "wor", "ld"}) || done == nil || *done.Output != "Hello world" || gotArgs["city"] != "Paris" {
		t.Fatalf("chunks=%v done=%+v args=%v", chunks, done, gotArgs)
	}
	if u := aiserver.ParseUsage(done.Usage); u.Input != 5 || u.Output != 3 {
		t.Errorf("usage = %+v", u)
	}
	root := attrs(e.spans(SpanName)[0])
	if root["gen_ai.usage.total_tokens"] != "8" || root["launchdarkly.config.key"] != "test-config" || root["gen_ai.completion.0.content"] != "Hello world" {
		t.Errorf("stream root attrs = %v", root)
	}

	e.exp.Reset()
	var gotErr error
	for _, err := range e.handler().Stream(bg, aiserver.HandlerRequest{Config: cfg(nil), UserInput: "q"}) {
		gotErr = err
	}
	if gotErr == nil || e.spans(SpanName)[0].Status().Code != codes.Error {
		t.Errorf("stream error not propagated: %v", gotErr)
	}
}

func TestStreamAbandoned(t *testing.T) {
	e := setup(t, nil, sse([]string{"a", "b", "c"}, "", 1, 1))
	for range e.handler().Stream(bg, aiserver.HandlerRequest{Config: cfg(nil), UserInput: "q"}) {
		break
	}
	root := e.spans(SpanName)
	if len(root) != 1 || root[0].Status().Code != codes.Unset || attrs(root[0])["launchdarkly.stream.abandoned"] != "true" {
		t.Errorf("abandoned root = %v", root)
	}
}

// agent mode.
func TestHistory(t *testing.T) {
	e := setup(t, nil, textResp("1", 1, 1), textResp("2", 1, 1), textResp("3", 1, 1), textResp("4", 1, 1), textResp("5", 1, 1))
	h := e.handler()
	c := cfg(func(c *aiserver.AiConfig) { c.Instructions = "inst" })
	img := aiserver.ContentBlock{Type: "image", Source: &aiserver.ImageSource{Type: "base64", MediaType: "image/png", Data: "AAAA"}}
	hist := []aiserver.Message{{Role: "system", Text: "drop me"}, {Role: "user", Text: "earlier"}, {Role: "assistant", Text: "reply"}}
	run := func(input string, history []aiserver.Message) {
		t.Helper()
		if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: c, UserInput: input, History: history}); err != nil {
			t.Fatal(err)
		}
	}
	run("now", hist)
	run("now", []aiserver.Message{})
	run("now", nil)
	run("", []aiserver.Message{{Role: "user", Blocks: []aiserver.ContentBlock{img, {Type: "text", Text: "what is this"}}}})
	run("describe", []aiserver.Message{{Role: "user", Blocks: []aiserver.ContentBlock{img}}})
	r := e.api.reqs()
	if system(r[0]) != "inst" || strings.Contains(system(r[0]), "Conversation History") || !reflect.DeepEqual(roles(r[0]), []string{"user", "assistant", "user"}) || firstText(messages(r[0])[0]) != "earlier" || firstText(messages(r[0])[2]) != "now" {
		t.Errorf("history: system=%q roles=%v", system(r[0]), roles(r[0]))
	}
	if !reflect.DeepEqual(r[1]["messages"], r[2]["messages"]) {
		t.Error("empty history differs from no history")
	}
	m := messages(r[3])
	blk := m[0]["content"].([]any)[0].(map[string]any)
	if len(m) != 1 || blk["type"] != "image" || blk["source"].(map[string]any)["data"] != "AAAA" {
		t.Errorf("image history: %v", m)
	}
	if m := messages(r[4]); len(m) != 2 || m[0]["content"].([]any)[0].(map[string]any)["type"] != "image" || firstText(m[1]) != "describe" {
		t.Errorf("image-only + input: %v", m)
	}
}

// A panicking tool re-panics, and every span the run opened is ended ERROR
// (an unended span is never exported).
func TestToolPanicEndsSpans(t *testing.T) {
	e := setup(t, nil, toolResp("a", `{}`))
	c := cfg(func(c *aiserver.AiConfig) {
		c.Tools = map[string]aiserver.Tool{"a": {Name: "a", Parameters: map[string]any{}}}
	})
	tools := map[string]aiserver.ToolHandler{"a": aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) { panic("tool blew up") })}
	func() {
		defer func() {
			if p := recover(); p != "tool blew up" {
				t.Errorf("panic = %v", p)
			}
		}()
		_, _ = e.handler().Invoke(bg, aiserver.HandlerRequest{Config: c, ToolHandlers: tools})
	}()
	for _, name := range []string{SpanName, "execute_tool a"} {
		if s := e.spans(name); len(s) != 1 || s[0].Status().Code != codes.Error {
			t.Errorf("%s not ended ERROR: %v", name, s)
		}
	}
}
