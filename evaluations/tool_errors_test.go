package evaluations

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"strings"
	"testing"

	"go.opentelemetry.io/otel/codes"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/trajectory"
)

type ValidationError struct{ field string }

func (e ValidationError) Error() string { return "invalid " + e.field }

type Generic[T any] struct{}

func (Generic[T]) Error() string { return "generic" }

func TestToolErrorResult(t *testing.T) {
	for _, tc := range []struct {
		err  error
		want string
	}{
		{errors.New("tool failed"), "Error: Error: tool failed"},
		{fmt.Errorf("lookup %d failed", 3), "Error: Error: lookup 3 failed"},
		{ValidationError{"q"}, "Error: ValidationError: invalid q"},
		{&ValidationError{"q"}, "Error: ValidationError: invalid q"},
		{Generic[int]{}, "Error: Generic[int]: generic"},
		// The class is the returned (outer) error's; the message keeps the chain.
		{fmt.Errorf("ctx: %w", os.ErrNotExist), "Error: Error: ctx: file does not exist"},
		{&fs.PathError{Op: "open", Path: "/tmp/x", Err: os.ErrNotExist}, "Error: PathError: open /tmp/x: file does not exist"},
		{fmt.Errorf("wrapped: %w", ValidationError{"q"}), "Error: Error: wrapped: invalid q"},
		{errors.Join(errors.New("a"), errors.New("b")), "Error: Error: a\nb"},
		{context.Canceled, "Error: Error: context canceled"},
		{context.DeadlineExceeded, "Error: Error: context deadline exceeded"},
	} {
		if got := toolErrorResult(tc.err); got != tc.want {
			t.Errorf("%T: got %q, want %q", tc.err, got, tc.want)
		}
	}
}

// Spans keep the raw Go type while the model is told "Error". Changing either
// side to match the other must fail here.
func TestToolErrorClassDiffersFromSpanErrorType(t *testing.T) {
	err := errors.New("x")
	if got := errorType(err); got != "errorString" {
		t.Errorf("errorType = %q, want errorString", got)
	}
	if got := toolErrorResult(err); got != "Error: Error: x" {
		t.Errorf("toolErrorResult = %q, want Error: Error: x", got)
	}
}

// A run whose tool fails: the handler gets the error text and finishes, the
// row is COMPLETE, the recorder keeps the call as failed, and a cancelled
// tool is answered too.
func TestFailedToolAnswersTheModel(t *testing.T) {
	for _, tc := range []struct {
		name string
		tool func(ctx context.Context) error
		want string
	}{
		{"plain error", func(context.Context) error { return errors.New("tool failed") }, "Error: Error: tool failed"},
		{"exported type", func(context.Context) error { return &ValidationError{"q"} }, "Error: ValidationError: invalid q"},
		{"wrapped", func(context.Context) error { return fmt.Errorf("lookup: %w", os.ErrNotExist) }, "Error: Error: lookup: file does not exist"},
		{"cancelled", func(context.Context) error {
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			return ctx.Err()
		}, "Error: Error: context canceled"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			e := newEnv(t, 1)
			var got any
			var gotErr error
			h := handler("OpenAI", aiserver.ModeMessages, func(ctx context.Context, req aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
				got, gotErr = req.ToolHandlers["lookup"].(aiserver.ToolFunc)(ctx, map[string]any{"q": "x"})
				return aiserver.HandlerResult{Output: "recovered"}, nil
			})
			args := baseArgs(h)
			args.Tools = []EvalTool{{Key: "lookup", Schema: map[string]any{}, Implementation: aiserver.ToolFunc(func(ctx context.Context, _ map[string]any) (any, error) {
				return nil, tc.tool(ctx)
			})}}
			mustRun(t, e.module(), args)
			if got != tc.want || gotErr != nil {
				t.Fatalf("handler got (%v, %v), want %q", got, gotErr, tc.want)
			}
			if ev := e.ld.events(generationEventName); len(ev) != 1 || ev[0].Data["status"] != "COMPLETE" {
				t.Fatalf("generation events %v", ev)
			}
		})
	}
}

// The answer wraps the recorder, so the recorder still sees the failure.
func TestRowToolsRecordTheFailedCall(t *testing.T) {
	r := &runState{tools: []runTool{{key: "lookup", impl: aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) {
		return nil, errors.New("tool failed")
	})}}}
	rec := trajectory.NewRecorder()
	got, err := r.rowTools(rec)["lookup"].(aiserver.ToolFunc)(context.Background(), map[string]any{"q": "x"})
	if got != "Error: Error: tool failed" || err != nil {
		t.Fatalf("got (%v, %v)", got, err)
	}
	invs := rec.Invocations()
	if len(invs) != 1 || !invs[0].Failed || !invs[0].Done || invs[0].Result != "" || invs[0].Error != "tool failed" {
		t.Fatalf("recorder must keep one failed call with no result: %+v", invs)
	}
}

// A failing handoff-prefixed tool is answered too, as in the JavaScript and
// Python SDKs, though the recorder does not record it.
func TestRowToolsAnswerAFailingHandoffTool(t *testing.T) {
	r := &runState{tools: []runTool{{key: trajectory.HandoffPrefix + "x", impl: aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) {
		return nil, errors.New("handoff failed")
	})}}}
	rec := trajectory.NewRecorder()
	got, err := r.rowTools(rec)[trajectory.HandoffPrefix+"x"].(aiserver.ToolFunc)(context.Background(), nil)
	if got != "Error: Error: handoff failed" || err != nil {
		t.Fatalf("got (%v, %v)", got, err)
	}
	if invs := rec.Invocations(); len(invs) != 0 {
		t.Fatalf("a handoff tool must not be recorded: %+v", invs)
	}
}

// A tool name the config did not offer never reaches a tool wrapper: the
// handler's own error fails the row.
func TestUnofferedToolStaysAHandlerError(t *testing.T) {
	e := newEnv(t, 1)
	h := handler("OpenAI", aiserver.ModeMessages, func(ctx context.Context, req aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
		if _, ok := req.ToolHandlers["missing"]; !ok {
			return aiserver.HandlerResult{}, fmt.Errorf("model called tool %q, which was not offered", "missing")
		}
		return aiserver.HandlerResult{Output: "unexpected"}, nil
	})
	mustRun(t, e.module(), baseArgs(h))
	ev := e.ld.events(generationEventName)
	if len(ev) != 1 || ev[0].Data["status"] != "ERROR" || !strings.Contains(fmt.Sprint(ev[0].Data["errorMessage"]), `"missing"`) {
		t.Fatalf("generation events %v", ev)
	}
}

// A plain function whose tool fails and then returns, with one judge: the
// row is COMPLETE, execute_tool is ERROR with tool_error, the row root and
// chat are OK, and the judge sees the failed call.
func TestEvaluationSpansWhenAToolFails(t *testing.T) {
	judgeCalls := 0
	e, args := spanRun(t, plainGeneration, &judgeCalls)
	args.Tools[0].Implementation = aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) {
		return nil, errors.New("lookup down")
	})
	var judgeVars map[string]any
	invoke := args.Handler.Invoke
	args.Handler.Invoke = func(ctx context.Context, req aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
		if _, isJudge := req.Variables["message_history"]; isJudge {
			judgeVars = req.Variables
		}
		return invoke(ctx, req)
	}
	exp, _ := evalSpans(t, e)
	mustRun(t, e.module(), args)
	spans := spanSet(exp.GetSpans().Snapshots())
	root, chat, tool := spans.one(t, rowSpanName), spans.one(t, "chat gpt-4o"), spans.one(t, "execute_tool lookup")
	for _, sp := range spans {
		if sp.Name() != tool.Name() && sp.Status().Code != codes.Ok {
			t.Errorf("%s status %v", sp.Name(), sp.Status())
		}
	}
	if tool.Status().Code != codes.Error || spanAttrs(tool)["error.type"].AsString() != "tool_error" {
		t.Errorf("execute_tool must be ERROR with error.type tool_error: %v %v", tool.Status(), spanAttrs(tool))
	}
	if !childOf(tool, root) && !childOf(tool, chat) {
		t.Error("execute_tool is outside the row")
	}
	if ev := e.ld.events(generationEventName); len(ev) != 1 || ev[0].Data["status"] != "COMPLETE" {
		t.Fatalf("generation events %v", ev)
	}
	if judgeCalls != 1 || !strings.Contains(toJSON(judgeVars), "error: lookup down") {
		t.Fatalf("judge calls %d, judge must see the failed call: %s", judgeCalls, toJSON(judgeVars))
	}
}
