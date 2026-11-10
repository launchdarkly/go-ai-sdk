package openaiagents

import (
	"testing"

	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

var graphContentKeys = []string{"gen_ai.prompt.0.content", "gen_ai.completion.0.content", "gen_ai.tool.call.arguments", "gen_ai.tool.call.result"}

// contentOnSpans reports which content attributes and events appear on any
// exported span.
func contentOnSpans(exp *tracetest.InMemoryExporter) map[string]bool {
	got := map[string]bool{}
	for _, s := range exp.GetSpans().Snapshots() {
		for _, a := range s.Attributes() {
			got[string(a.Key)] = true
		}
		for _, ev := range s.Events() {
			got[ev.Name] = true
		}
	}
	return got
}

// NativeGraphOptions.Handler.CaptureContent gates content on every node,
// chat and tool span of the native run.
func TestToOpenAIAgentsCaptureContent(t *testing.T) {
	for _, tc := range []struct {
		name    string
		capture aiserver.ContentCapture
		want    bool
	}{{"on", aiserver.CaptureOn, true}, {"off", aiserver.CaptureOff, false}} {
		t.Run(tc.name, func(t *testing.T) {
			ld := twoNode(nil)
			installClient(t, ld)
			exp := tracer(t)
			_, c := fakeClient(t,
				response("r1", callOutput("transfer_to_billing", "{}"), 3, 1, 1),
				response("r2", callOutput("refund", `{"id":1}`), 4, 0, 2),
				response("r3", textOutput("refunded"), 5, 0, 3))
			var calls []string
			opts := &NativeGraphOptions{ToolHandlers: handlersFor(&calls), Context: user, Handler: Options{Client: c, CaptureContent: tc.capture}}
			if _, err := must(ToOpenAIAgents(resolve(t, ld, "g"), opts)).Invoke(bg, "hello"); err != nil {
				t.Fatal(err)
			}
			got := contentOnSpans(exp)
			for _, k := range append(graphContentKeys, "gen_ai.content.prompt", "gen_ai.content.completion") {
				if got[k] != tc.want {
					t.Errorf("%s present=%v, want %v", k, got[k], tc.want)
				}
			}
			for _, k := range []string{"gen_ai.request.model", "gen_ai.tool.name"} {
				if !got[k] {
					t.Errorf("%s must be written whatever the gate", k)
				}
			}
		})
	}
}
