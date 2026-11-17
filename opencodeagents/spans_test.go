package opencodeagents

import (
	"strings"
	"testing"

	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

// One invoke_agent root per call; one chat span per assistant message and one
// execute_tool span per tool call, all direct children of the root, on both
// the blocking and the streaming path.
func TestSpanTree(t *testing.T) {
	events := []map[string]any{
		assistant("m1"), delta("m1", "p1", "a"),
		toolPart("m1", "c1", "bash", "running", map[string]any{"command": "ls"}, ""),
		toolPart("m1", "c1", "bash", "completed", map[string]any{"command": "ls"}, "out"),
		completed("m1", 2, 3), completed("m1", 2, 3), completedNoTokens("m1"),
		assistant("m2"), delta("m2", "p2", "b"), completed("m2", 10, 20), idle("ses"),
	}
	for _, streaming := range []bool{false, true} {
		name := "invoke"
		if streaming {
			name = "stream"
		}
		t.Run(name, func(t *testing.T) {
			e := setup(t, nil)
			f := newFake(t, fakeSpec{Events: events})
			req := aiserver.HandlerRequest{Config: cfg(withTools("shell")), UserInput: "q", ToolHandlers: map[string]aiserver.ToolHandler{"shell": OpenCodeBash}}
			if streaming {
				for _, err := range f.handler(t).Stream(bg, req) {
					if err != nil {
						t.Fatal(err)
					}
				}
			} else if _, err := invoke(t, f.handler(t), req); err != nil {
				t.Fatal(err)
			}
			root := e.spans(SpanName)
			if len(root) != 1 || root[0].Status().Code == codes.Error {
				t.Fatalf("root spans = %v", root)
			}
			ra := attrs(root[0])
			if ra["gen_ai.operation.name"] != "invoke_agent" || ra["gen_ai.usage.input_tokens"] != "12" || ra["gen_ai.usage.output_tokens"] != "23" || ra["gen_ai.request.model"] != "test-model" {
				t.Errorf("root attrs = %v", ra)
			}
			chats := e.spans("chat")
			if len(chats) != 2 {
				t.Fatalf("chat spans = %d, want one per assistant message", len(chats))
			}
			usage := map[string]bool{}
			for _, c := range chats {
				usage[attrs(c)["gen_ai.usage.input_tokens"]+"/"+attrs(c)["gen_ai.usage.output_tokens"]] = true
			}
			if !usage["2/3"] || !usage["10/20"] {
				t.Errorf("chat usage = %v", usage)
			}
			tools := e.spans("execute_tool")
			if len(tools) != 1 || attrs(tools[0])["gen_ai.tool.name"] != "shell" {
				t.Fatalf("execute_tool spans = %v", tools)
			}
			for _, s := range append(chats, tools...) {
				if s.Parent().SpanID() != root[0].SpanContext().SpanID() {
					t.Errorf("%s is not a child of the root", s.Name())
				}
			}
		})
	}
}

// A failed tool part fails its execute_tool span; the run still succeeds.
func TestFailedToolFailsItsSpan(t *testing.T) {
	e := setup(t, nil)
	f := newFake(t, fakeSpec{Events: []map[string]any{
		assistant("m1"), delta("m1", "p1", "a"),
		toolPart("m1", "c1", "bash", "running", map[string]any{"command": "x"}, ""),
		toolPart("m1", "c1", "bash", "error", map[string]any{"command": "x"}, "provider-secret"),
		completed("m1", 1, 1), idle("ses"),
	}})
	if _, err := invoke(t, f.handler(t), aiserver.HandlerRequest{Config: cfg(withTools("shell")), UserInput: "q",
		ToolHandlers: map[string]aiserver.ToolHandler{"shell": OpenCodeBash}}); err != nil {
		t.Fatal(err)
	}
	tools := e.spans("execute_tool")
	if len(tools) != 1 || tools[0].Status().Code != codes.Error {
		t.Fatalf("tool spans = %v", tools)
	}
	if strings.Contains(spanText(tools[0]), "provider-secret") {
		t.Error("the tool span carries the provider's error text")
	}
	if root := e.spans(SpanName); len(root) != 1 || root[0].Status().Code == codes.Error {
		t.Errorf("root = %v", root)
	}
}

// CaptureOff keeps prompt, output and tool content off every span.
func TestCaptureOff(t *testing.T) {
	e := setup(t, nil)
	f := newFake(t, fakeSpec{Events: []map[string]any{
		assistant("m1"), delta("m1", "p1", "secret-output"),
		toolPart("m1", "c1", "bash", "running", map[string]any{"command": "secret-args"}, ""),
		toolPart("m1", "c1", "bash", "completed", map[string]any{"command": "secret-args"}, "secret-result"),
		completed("m1", 1, 1), idle("ses"),
	}})
	o := f.options()
	o.CaptureContent = aiserver.CaptureOff
	h, err := NewHandler(o)
	if err != nil {
		t.Fatal(err)
	}
	c := cfg(withTools("shell"))
	c.Instructions = "secret-system"
	if _, err := invoke(t, h, aiserver.HandlerRequest{Config: c, UserInput: "secret-input", ToolHandlers: map[string]aiserver.ToolHandler{"shell": OpenCodeBash}}); err != nil {
		t.Fatal(err)
	}
	for _, s := range e.exp.GetSpans().Snapshots() {
		if strings.Contains(spanText(s), "secret-") {
			t.Errorf("span %s carries content: %s", s.Name(), spanText(s))
		}
	}
	// And capture on writes it.
	e2 := setup(t, nil)
	f2 := newFake(t, fakeSpec{Events: reply("visible-output", 1, 1)})
	if _, err := invoke(t, f2.handler(t), aiserver.HandlerRequest{Config: cfg(nil), UserInput: "visible-input"}); err != nil {
		t.Fatal(err)
	}
	root := e2.spans(SpanName)
	if len(root) != 1 || !strings.Contains(spanText(root[0]), "visible-output") || !strings.Contains(spanText(root[0]), "visible-input") {
		t.Errorf("capture on: root = %v", root)
	}
}

// Under Bedrock with a region, telemetry reports the id that was sent.
func TestTelemetryReportsTheSentModelID(t *testing.T) {
	e := setup(t, nil)
	f := newFake(t, fakeSpec{Events: reply("ok", 1, 1)})
	c := cfg(func(c *aiserver.AiConfig) {
		c.Provider.Name = "Bedrock"
		c.Model = aiserver.Model{Name: "anthropic.claude-haiku", Region: "us"}
	})
	if _, err := invoke(t, f.handler(t), aiserver.HandlerRequest{Config: c, UserInput: "q"}); err != nil {
		t.Fatal(err)
	}
	if m := f.state(t).Messages[0]["model"]; jsonString(m) != `{"modelID":"us.anthropic.claude-haiku","providerID":"amazon-bedrock"}` {
		t.Errorf("model = %v", m)
	}
	root := e.spans(SpanName)
	if len(root) != 1 || attrs(root[0])["gen_ai.request.model"] != "us.anthropic.claude-haiku" {
		t.Errorf("root = %v", root)
	}
	if chats := e.spans("chat"); len(chats) != 1 || chats[0].Name() != "chat us.anthropic.claude-haiku" {
		t.Errorf("chat spans = %v", chats)
	}
}

func spanText(s sdktrace.ReadOnlySpan) string {
	var b strings.Builder
	for _, a := range s.Attributes() {
		b.WriteString(a.Value.Emit())
		b.WriteByte(' ')
	}
	for _, ev := range s.Events() {
		for _, a := range ev.Attributes {
			b.WriteString(a.Value.Emit())
			b.WriteByte(' ')
		}
	}
	b.WriteString(s.Status().Description)
	return b.String()
}
