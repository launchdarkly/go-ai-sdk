package aiserver

import (
	"context"
	"errors"
	"iter"
	"testing"

	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

// freshOptions gives a test the built-in defaults and an unlatched Configure.
func freshOptions(t *testing.T) {
	t.Helper()
	resetOptions()
	t.Cleanup(resetOptions)
}

// captureProbe is a handler that reports what CaptureContentEnabled says
// inside its own call.
func captureProbe(c ContentCapture, got *bool) *ProviderHandler {
	return createHandler(ProvidesFor{"P", ModeMessages}, func(ctx context.Context, _ HandlerRequest) (HandlerResult, error) {
		*got = CaptureContentEnabled(ctx)
		return HandlerResult{Output: "ok"}, nil
	}, nil, c)
}

func TestCapturePrecedence(t *testing.T) {
	cases := []struct {
		name      string
		configure ContentCapture
		handler   ContentCapture
		want      bool
	}{
		{"built-in default", CaptureDefault, CaptureDefault, true},
		{"configured off", CaptureOff, CaptureDefault, false},
		{"configured on", CaptureOn, CaptureDefault, true},
		{"explicit off beats configured on", CaptureOn, CaptureOff, false},
		{"explicit on beats configured off", CaptureOff, CaptureOn, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			freshOptions(t)
			var got bool
			h := captureProbe(tc.handler, &got) // built before Configure: read at run time
			if err := Configure(SDKOptions{CaptureContent: tc.configure}); err != nil {
				t.Fatal(err)
			}
			if _, err := h.Invoke(bg, HandlerRequest{}); err != nil {
				t.Fatal(err)
			}
			if got != tc.want {
				t.Errorf("capture = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestCaptureOutsideAnyHandlerIsTheProcessDefault(t *testing.T) {
	freshOptions(t)
	if !CaptureContentEnabled(bg) {
		t.Fatal("built-in default must be on")
	}
	_ = Configure(SDKOptions{CaptureContent: CaptureOff})
	if CaptureContentEnabled(bg) {
		t.Fatal("configured off must apply outside a handler")
	}
}

// A nested call with no setting of its own follows the process default, not
// the enclosing handler's explicit value.
func TestNestedDefaultHandlerDoesNotInheritExplicitValue(t *testing.T) {
	freshOptions(t)
	var inner bool
	innerH := captureProbe(CaptureDefault, &inner)
	outer := createHandler(ProvidesFor{"P", ModeMessages}, func(ctx context.Context, req HandlerRequest) (HandlerResult, error) {
		return innerH.Invoke(ctx, req)
	}, nil, CaptureOff)
	if _, err := outer.Invoke(bg, HandlerRequest{}); err != nil {
		t.Fatal(err)
	}
	if !inner {
		t.Fatal("inner default handler must follow the process default (on)")
	}
}

func TestConfigureLock(t *testing.T) {
	freshOptions(t)
	if err := Configure(SDKOptions{CaptureContent: CaptureOn}); err != nil {
		t.Fatal(err)
	}
	if err := Configure(SDKOptions{CaptureContent: CaptureOff}); err != nil {
		t.Fatalf("a second Configure before any invocation must replace the first: %v", err)
	}
	var got bool
	if _, err := captureProbe(CaptureDefault, &got).Invoke(bg, HandlerRequest{}); err != nil {
		t.Fatal(err)
	}
	if got {
		t.Fatal("the replacing Configure must win")
	}
	if err := Configure(SDKOptions{CaptureContent: CaptureOff}); err != nil {
		t.Fatalf("identical options after an invocation must be accepted: %v", err)
	}
	if err := Configure(SDKOptions{CaptureContent: CaptureOn}); !errors.Is(err, ErrConfigureLocked) {
		t.Fatalf("different options after an invocation: %v, want ErrConfigureLocked", err)
	}
	if err := Shutdown(bg); err != nil {
		t.Fatal(err)
	}
	if !CaptureContentEnabled(bg) {
		t.Fatal("Shutdown must restore the built-in default")
	}
	if err := Configure(SDKOptions{CaptureContent: CaptureOn}); err != nil {
		t.Fatalf("Shutdown must clear the lock: %v", err)
	}
}

func TestStreamStartLatchesConfigure(t *testing.T) {
	freshOptions(t)
	h := createHandler(ProvidesFor{"P", ModeMessages}, nil, func(context.Context, HandlerRequest) iter.Seq2[HandlerStreamEvent, error] {
		return func(func(HandlerStreamEvent, error) bool) {}
	}, CaptureDefault)
	_ = h.Stream(bg, HandlerRequest{}) // never iterated
	if err := Configure(SDKOptions{CaptureContent: CaptureOff}); !errors.Is(err, ErrConfigureLocked) {
		t.Fatalf("starting a stream must latch Configure: %v", err)
	}
}

// A handler struct built without CreateHandler is still bound through the
// core call path.
func TestCoreCallPathBindsLiteralHandler(t *testing.T) {
	freshOptions(t)
	withFake(t, map[string]any{"cfg": enabledConfig("P", "", nil)})
	var got = true
	h := &ProviderHandler{ProvidesFor: ProvidesFor{"P", ModeMessages}, CaptureContent: CaptureOff,
		Invoke: func(ctx context.Context, _ HandlerRequest) (HandlerResult, error) {
			got = CaptureContentEnabled(ctx)
			return HandlerResult{Output: "ok"}, nil
		}}
	if _, err := Config(ConfigArgs{Key: "cfg", Handlers: []*ProviderHandler{h}}).Invoke(bg, "q", userCtx); err != nil {
		t.Fatal(err)
	}
	if got {
		t.Fatal("explicit CaptureOff on a literal handler must apply through Config")
	}
}

func TestInitOptionsDoNotSetTheCaptureDefault(t *testing.T) {
	freshOptions(t)
	_ = Shutdown(bg)
	t.Cleanup(func() { _ = Shutdown(bg) })
	if _, err := InitWithClient(bg, &fakeLD{}, InitOptions{OTLPEndpoint: unreachableOTLP}); err != nil {
		t.Fatal(err)
	}
	if !CaptureContentEnabled(bg) {
		t.Fatal("init must leave the built-in default on")
	}
}

// --- content gate on handler spans and shared helpers ---------------

func contentSpans(t *testing.T) *tracetest.InMemoryExporter {
	t.Helper()
	exp := tracetest.NewInMemoryExporter()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exp))
	withTracerProvider(t, tp)
	return exp
}

// writeAllContent writes a handler span through the kit and a span through
// the shared helpers, under ctx.
func writeAllContent(ctx context.Context) {
	run := StartHandlerRun(ctx, HandlerSpanOptions{Tracer: TracerPrefix + "/test", Name: "handler.span", System: "p", RequestModel: "m"},
		HandlerRequest{}, Prompt{System: "sys", Turns: []Message{{Role: "user", Text: "hi"}}})
	tool := run.StartTool("lookup", "call_1")
	run.ToolArguments(tool, map[string]any{"q": "x"})
	run.ToolResult(tool, "found")
	run.EndTool(tool, nil)
	run.Finish("m", "answer")

	_, span := tracerFor(TracerPrefix+"/test").Start(ctx, "chat m")
	SetChatContent(ctx, span, ChatContent{
		SystemInstructions: "sys",
		InputMessages:      []ChatMessage{{Role: "user", Parts: []MessagePart{TextPart("hi")}}},
		ToolDefinitions:    []Tool{{Name: "lookup", Description: "find", Parameters: map[string]any{"type": "object"}}},
		OutputMessages: []ChatMessage{{Role: "assistant", Parts: []MessagePart{
			ToolCallPart("call_1", "lookup", map[string]any{"q": "x"}), TextPart("answer")}}},
	})
	SetToolCallArguments(ctx, span, map[string]any{"q": "x"})
	SetToolCallResult(ctx, span, "found")
	span.End()
}

var contentKeys = []string{
	"gen_ai.prompt.0.content", "gen_ai.completion.0.content", "gen_ai.tool.call.arguments", "gen_ai.tool.call.result",
	"gen_ai.system_instructions", "gen_ai.input.messages", "gen_ai.tool.definitions", "gen_ai.output.messages",
}

func presentContent(exp *tracetest.InMemoryExporter) map[string]bool {
	got := map[string]bool{}
	for _, s := range exp.GetSpans().Snapshots() {
		for _, a := range s.Attributes() {
			got[string(a.Key)] = true
		}
	}
	return got
}

func TestContentGateOnHandlerAndHelperSpans(t *testing.T) {
	cases := []struct {
		name      string
		configure ContentCapture
		bound     ContentCapture
		want      bool
	}{
		{"built-in default", CaptureDefault, CaptureDefault, true},
		{"explicit off", CaptureDefault, CaptureOff, false},
		{"configured off", CaptureOff, CaptureDefault, false},
		{"explicit on over configured off", CaptureOff, CaptureOn, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			freshOptions(t)
			exp := contentSpans(t)
			_ = Configure(SDKOptions{CaptureContent: tc.configure})
			writeAllContent(startInvocation(bg, tc.bound))
			got := presentContent(exp)
			for _, k := range contentKeys {
				if got[k] != tc.want {
					t.Errorf("%s present=%v, want %v", k, got[k], tc.want)
				}
			}
			// Identity is written either way.
			for _, k := range []string{"gen_ai.request.model", "gen_ai.tool.name", "gen_ai.tool.call.id"} {
				if !got[k] {
					t.Errorf("%s must be written whatever the gate", k)
				}
			}
		})
	}
}

func TestChatContentShape(t *testing.T) {
	freshOptions(t)
	exp := contentSpans(t)
	writeAllContent(bg)
	var attrs map[string]string
	for _, s := range exp.GetSpans().Snapshots() {
		if s.Name() == "chat m" {
			attrs = map[string]string{}
			for _, a := range s.Attributes() {
				attrs[string(a.Key)] = a.Value.Emit()
			}
		}
	}
	want := map[string]string{
		"gen_ai.system_instructions": `[{"type":"text","content":"sys"}]`,
		"gen_ai.input.messages":      `[{"role":"user","parts":[{"type":"text","content":"hi"}]}]`,
		"gen_ai.tool.definitions":    `[{"description":"find","name":"lookup","parameters":{"type":"object"},"type":"function"}]`,
		"gen_ai.output.messages":     `[{"role":"assistant","parts":[{"type":"tool_call","id":"call_1","name":"lookup","arguments":{"q":"x"}},{"type":"text","content":"answer"}]}]`,
		"gen_ai.tool.call.arguments": `{"q":"x"}`,
		"gen_ai.tool.call.result":    `found`,
	}
	for k, v := range want {
		if attrs[k] != v {
			t.Errorf("%s = %s\nwant %s", k, attrs[k], v)
		}
	}
	// The OpenLLMetry carrier: the system instructions are message 0, and a
	// tool call flattens to {name, arguments} JSON.
	for k, v := range map[string]string{
		"gen_ai.prompt.0.role": "system", "gen_ai.prompt.0.content": "sys",
		"gen_ai.prompt.1.role": "user", "gen_ai.prompt.1.content": "hi",
		"gen_ai.completion.0.role": "assistant", "gen_ai.completion.0.content": `{"name":"lookup","arguments":{"q":"x"}}` + "\nanswer",
	} {
		if attrs[k] != v {
			t.Errorf("%s = %q, want %q", k, attrs[k], v)
		}
	}
}

func TestToolDefinitionsOmitEmptyMembers(t *testing.T) {
	freshOptions(t)
	exp := contentSpans(t)
	_, span := tracerFor(TracerPrefix+"/test").Start(bg, "chat m")
	SetChatContent(bg, span, ChatContent{ToolDefinitions: []Tool{{Name: "bare", Type: "custom"}}})
	span.End()
	for _, a := range exp.GetSpans().Snapshots()[0].Attributes() {
		if a.Key == "gen_ai.tool.definitions" && a.Value.AsString() != `[{"name":"bare","type":"function"}]` {
			t.Errorf("tool definitions %s", a.Value.AsString())
		}
	}
	if evs := exp.GetSpans().Snapshots()[0].Events(); len(evs) != 1 || evs[0].Name != "gen_ai.content.prompt" {
		t.Errorf("the prompt event is written whenever capture is on; no completion event without output: %v", evs)
	}
}

func TestJudgeExplanationFollowsConfiguredDefault(t *testing.T) {
	freshOptions(t)
	withFake(t, map[string]any{
		"cfg":   enabledConfig("P", "", judgedConfig(nil)),
		"judge": enabledConfig("P", "judge", nil),
	})
	exp := withSpans(t)
	_ = Configure(SDKOptions{CaptureContent: CaptureOff})
	h := createHandler(ProvidesFor{"P", ModeMessages}, func(ctx context.Context, req HandlerRequest) (HandlerResult, error) {
		if req.Variables["response_to_evaluate"] == nil {
			return HandlerResult{Output: "answer"}, nil
		}
		run := StartHandlerRun(ctx, HandlerSpanOptions{Tracer: TracerPrefix + "/test", Name: "judge.span", System: "p", RequestModel: "m"}, req, Prompt{})
		run.Finish("m", `{"score":0.5,"reasoning":"because"}`)
		return HandlerResult{Output: `{"score":0.5,"reasoning":"because"}`}, nil
	}, nil, CaptureDefault)
	if _, err := Config(ConfigArgs{Key: "cfg", Handlers: []*ProviderHandler{h}}).Invoke(bg, "q", userCtx); err != nil {
		t.Fatal(err)
	}
	scored := false
	for _, s := range exp.GetSpans().Snapshots() {
		for _, a := range s.Attributes() {
			switch a.Key {
			case "gen_ai.evaluation.explanation":
				t.Fatal("explanation must follow the configured default (off)")
			case "gen_ai.evaluation.score.value":
				scored = true
			}
		}
	}
	if !scored {
		t.Fatal("score must be written whatever the gate")
	}
}
