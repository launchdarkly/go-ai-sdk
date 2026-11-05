package claudeagents

import (
	"context"
	"testing"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

// AdapterOptions.Handler.CaptureContent gates content on the native run's
// node, chat and tool spans.
func TestToClaudeAgentsCaptureContent(t *testing.T) {
	for _, tc := range []struct {
		name    string
		capture aiserver.ContentCapture
		want    bool
	}{{"on", aiserver.CaptureOn, true}, {"off", aiserver.CaptureOff, false}} {
		t.Run(tc.name, func(t *testing.T) {
			flags := graphFlags(nil)
			flags["leaf"] = agentVariation("leaf inst", map[string]any{"tools": map[string]any{
				"lookup": map[string]any{"name": "lookup", "type": "function", "parameters": map[string]any{"type": "object"}}}})
			e := setup(t, flags, toolResp("mcp__subagents__leaf", `{"input":"find it"}`), toolResp("lookup", `{"id":"7"}`),
				textResp("leaf found 7", 1, 1), textResp("done", 1, 1))
			tools := map[string]aiserver.ToolHandler{"lookup": aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) {
				return "record 7", nil
			})}
			opts := e.opts
			opts.CaptureContent = tc.capture
			if _, err := ToClaudeAgents(resolve(t), &AdapterOptions{Handler: opts, ToolHandlers: tools}).Invoke(bg, "q"); err != nil {
				t.Fatal(err)
			}
			got := map[string]bool{}
			for _, s := range e.exp.GetSpans().Snapshots() {
				for k := range attrs(s) {
					got[k] = true
				}
				for _, ev := range s.Events() {
					got[ev.Name] = true
				}
			}
			for _, k := range []string{"gen_ai.prompt.0.content", "gen_ai.completion.0.content", "gen_ai.tool.call.arguments",
				"gen_ai.tool.call.result", "gen_ai.content.prompt", "gen_ai.content.completion"} {
				if got[k] != tc.want {
					t.Errorf("%s present=%v, want %v", k, got[k], tc.want)
				}
			}
			if !got["gen_ai.tool.name"] {
				t.Error("gen_ai.tool.name must be written whatever the gate")
			}
		})
	}
}
