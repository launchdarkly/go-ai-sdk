package claudemessages

import (
	"context"
	"errors"
	"testing"

	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

// One run calls tool a, which succeeds, then tool b, which fails and ends the
// run: b's span is labelled tool_error, a's is not. setup turns capture on.
func TestToolErrorType(t *testing.T) {
	h, _, exp := setup(t, toolResp("a", `{}`), toolResp("b", `{}`))
	boom := errors.New("boom")
	c := cfg(func(c *aiserver.AiConfig) {
		c.Instructions = "x"
		c.Tools = map[string]aiserver.Tool{"a": {Name: "a", Type: "function", Parameters: map[string]any{}},
			"b": {Name: "b", Type: "function", Parameters: map[string]any{}}}
	})
	tools := map[string]aiserver.ToolHandler{
		"a": aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) { return "r", nil }),
		"b": aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) { return nil, boom }),
	}
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: c, UserInput: "q", ToolHandlers: tools}); !errors.Is(err, boom) {
		t.Fatalf("err = %v, want the tool error", err)
	}
	checkToolErrorSpans(t, exp.GetSpans().Snapshots(), "execute_tool a", "execute_tool b")
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

// Cancelling the context while a tool blocks ends the tool span ERROR with no
// error.type: a cancelled run is not a tool failure.
func TestToolCancelNoErrorType(t *testing.T) {
	h, _, exp := setup(t, toolResp("a", `{}`))
	ctx, cancel := context.WithCancel(bg)
	defer cancel()
	started := make(chan struct{})
	c := cfg(func(c *aiserver.AiConfig) {
		c.Instructions = "x"
		c.Tools = map[string]aiserver.Tool{"a": {Name: "a", Type: "function", Parameters: map[string]any{}}}
	})
	tools := map[string]aiserver.ToolHandler{
		"a": aiserver.ToolFunc(func(ctx context.Context, _ map[string]any) (any, error) {
			close(started)
			<-ctx.Done()
			return nil, ctx.Err()
		}),
	}
	go func() { <-started; cancel() }()
	if _, err := h.Invoke(ctx, aiserver.HandlerRequest{Config: c, UserInput: "q", ToolHandlers: tools}); !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	for _, s := range exp.GetSpans().Snapshots() {
		if s.Name() != "execute_tool a" {
			continue
		}
		for _, a := range s.Attributes() {
			if string(a.Key) == "error.type" {
				t.Errorf("cancelled tool span has error.type=%q", a.Value.Emit())
			}
		}
		if s.Status().Code != codes.Error {
			t.Errorf("status=%v, want ERROR", s.Status().Code)
		}
		return
	}
	t.Fatal("no execute_tool a span")
}
