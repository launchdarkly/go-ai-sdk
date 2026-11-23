package claudeagents

import (
	"context"
	"errors"
	"path/filepath"
	"strings"
	"testing"

	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

// One run calls function tool a, which succeeds, then b, which fails and ends
// the run: b's span is labelled tool_error, a's is not.
func TestToolErrorType(t *testing.T) {
	e := setup(t, nil, toolResp("a", `{}`), toolResp("b", `{}`))
	e.opts.CaptureContent = aiserver.CaptureOn
	boom := errors.New("boom")
	c := cfg(func(c *aiserver.AiConfig) {
		c.Instructions = "x"
		c.Tools = map[string]aiserver.Tool{"a": {Name: "a", Type: "function", Parameters: map[string]any{"type": "object"}},
			"b": {Name: "b", Type: "function", Parameters: map[string]any{"type": "object"}}}
	})
	tools := map[string]aiserver.ToolHandler{
		"a": aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) { return "r", nil }),
		"b": aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) { return nil, boom }),
	}
	if _, err := e.handler().Invoke(bg, aiserver.HandlerRequest{Config: c, UserInput: "q", ToolHandlers: tools}); !errors.Is(err, boom) {
		t.Fatalf("err = %v, want the tool error", err)
	}
	checkToolErrorSpans(t, e.exp.GetSpans().Snapshots(), "execute_tool a", "execute_tool b")
}

// A failed host built-in is answered to the model as an is_error result and
// the run continues: the failed Read's span is labelled tool_error and writes
// no result, while the error text still reaches the model.
func TestHostToolErrorType(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "notes.txt"), "remember the milk\n")
	e := setup(t, nil, toolResp("Read", `{"file_path":"notes.txt"}`), toolResp("Read", `{"file_path":"../etc/passwd"}`), textResp("done", 1, 1))
	e.opts.WorkDir, e.opts.CaptureContent = dir, aiserver.CaptureOn
	c := cfg(func(c *aiserver.AiConfig) { c.Tools = map[string]aiserver.Tool{"reader": {Name: "reader"}} })
	res, err := e.handler().Invoke(bg, aiserver.HandlerRequest{Config: c, UserInput: "q",
		ToolHandlers: map[string]aiserver.ToolHandler{"reader": ClaudeRead}})
	if err != nil || res.Output != "done" {
		t.Fatalf("res = %+v err = %v", res, err)
	}
	if out, isErr := toolResult(t, e.api.reqs()[2]); !isErr || !strings.Contains(out, "not allowed") {
		t.Errorf("model got %q isErr=%v, want the error text", out, isErr)
	}
	checkToolErrorSpans(t, e.exp.GetSpans().Snapshots(), "execute_tool Read", "execute_tool Read")
}

// checkToolErrorSpans checks the first span named okName and the last named
// badName: the failed call is ERROR with its exception, error.type tool_error
// and no gen_ai.tool.call.result even though capture is on; the call that
// succeeded is OK with its result and no error.type.
func checkToolErrorSpans(t *testing.T, spans []sdktrace.ReadOnlySpan, okName, badName string) {
	t.Helper()
	okI, badI := -1, -1
	for i, s := range spans {
		if s.Name() == okName && okI < 0 {
			okI = i
		}
		if s.Name() == badName {
			badI = i
		}
	}
	if okI < 0 || badI < 0 || okI == badI {
		t.Fatalf("want spans %q and %q, got %d spans", okName, badName, len(spans))
	}
	ok, bad := spans[okI], spans[badI]
	get := func(s sdktrace.ReadOnlySpan, key string) (string, bool) {
		for _, a := range s.Attributes() {
			if string(a.Key) == key {
				return a.Value.Emit(), true
			}
		}
		return "", false
	}
	if typ, has := get(ok, "error.type"); ok.Status().Code != codes.Ok || has {
		t.Errorf("%s: status=%v error.type=%q, want OK and none", okName, ok.Status().Code, typ)
	}
	if _, has := get(ok, "gen_ai.tool.call.result"); !has {
		t.Errorf("%s: no gen_ai.tool.call.result; content capture is off", okName)
	}
	if typ, _ := get(bad, "error.type"); bad.Status().Code != codes.Error || typ != "tool_error" {
		t.Errorf("%s: status=%v error.type=%q, want ERROR and tool_error", badName, bad.Status().Code, typ)
	}
	if res, has := get(bad, "gen_ai.tool.call.result"); has {
		t.Errorf("%s: failed call wrote gen_ai.tool.call.result=%q", badName, res)
	}
	exception := false
	for _, e := range bad.Events() {
		exception = exception || e.Name == "exception"
	}
	if !exception {
		t.Errorf("%s: no exception event", badName)
	}
}
