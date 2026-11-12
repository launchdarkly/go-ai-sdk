package aiserver

import (
	"context"
	"testing"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"

	"github.com/launchdarkly/go-ai-sdk/internal/kitbase"
)

const (
	targetRunID     = "launchdarkly.judge.target.run.id"
	targetConfigKey = "launchdarkly.judge.target.config.key"
	targetVariation = "launchdarkly.judge.target.variation.key"
)

// withStampedSpans installs the SDK span processor ahead of an in-memory
// exporter, as InitClient does; without it nothing is stamped.
func withStampedSpans(t *testing.T) *tracetest.InMemoryExporter {
	exp := tracetest.NewInMemoryExporter()
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(sdktrace.NewTracerProvider(
		sdktrace.WithSpanProcessor(ConversationIDSpanProcessor{}), sdktrace.WithSyncer(exp)))
	t.Cleanup(func() { otel.SetTracerProvider(prev) })
	return exp
}

// targetJudgeHandler opens an SDK root on every call, a chat span under it,
// and for judge calls also a span on a non-SDK tracer and one that already
// carries its own target run id.
func targetJudgeHandler(capture ContentCapture) *ProviderHandler {
	return createHandler(ProvidesFor{"P", ModeMessages}, func(ctx context.Context, req HandlerRequest) (HandlerResult, error) {
		judge := req.Variables["response_to_evaluate"] != nil
		name := "graded"
		if judge {
			name = "judge"
		}
		run := kitStartHandlerRun(ctx, kitbase.HandlerSpanOptions{Tracer: kitbase.TracerPrefix + "/test", Name: name, System: "p", RequestModel: "m"}, req, prompt{})
		run.EndChat(run.StartChat(""), "m", kitbase.SpanUsage{}, "")
		if !judge {
			run.Finish("m", "answer")
			return HandlerResult{Output: "answer"}, nil
		}
		_, other := otel.Tracer("example.com/other").Start(ctx, "other")
		other.End()
		_, own := otel.Tracer(kitbase.TracerPrefix+"/test").Start(ctx, "own", trace.WithAttributes(attribute.String(targetRunID, "own")))
		own.End()
		run.Finish("m", `{"score":0.5,"reasoning":"ok"}`)
		return HandlerResult{Output: `{"score":0.5,"reasoning":"ok"}`}, nil
	}, nil, capture)
}

func targetAttrs(s sdktrace.ReadOnlySpan) map[string]string {
	out := map[string]string{}
	for _, a := range s.Attributes() {
		out[string(a.Key)] = a.Value.Emit()
	}
	return out
}

// checkJudgeTarget asserts the three attributes on every SDK span of the
// judge run and on none of the graded run's spans or the non-SDK span.
func checkJudgeTarget(t *testing.T, path string, exp *tracetest.InMemoryExporter, want TrackData) {
	t.Helper()
	judgeSpans := 0
	for _, s := range exp.GetSpans().Snapshots() {
		a := targetAttrs(s)
		inJudge := s.Name() == "judge" || s.Name() == "own" || s.Name() == "other" ||
			(s.Parent().IsValid() && parentName(exp, s) == "judge")
		switch {
		case s.Name() == "other":
			if a[targetRunID] != "" || a[targetConfigKey] != "" || a[targetVariation] != "" {
				t.Errorf("%s: non-SDK span stamped: %v", path, a)
			}
		case s.Name() == "own":
			if a[targetRunID] != "own" {
				t.Errorf("%s: write-if-absent: run id = %q, want own", path, a[targetRunID])
			}
		case inJudge:
			judgeSpans++
			if a[targetRunID] != want.RunID || a[targetConfigKey] != want.ConfigKey {
				t.Errorf("%s: %s target = %v, want run %q config %q", path, s.Name(), a, want.RunID, want.ConfigKey)
			}
			if got, has := a[targetVariation]; want.VariationKey == "" && has {
				t.Errorf("%s: %s empty variation key written as %q", path, s.Name(), got)
			} else if want.VariationKey != "" && got != want.VariationKey {
				t.Errorf("%s: %s variation = %q, want %q", path, s.Name(), got, want.VariationKey)
			}
		default:
			if a[targetRunID] != "" || a[targetConfigKey] != "" || a[targetVariation] != "" {
				t.Errorf("%s: %s outside the judge run is stamped: %v", path, s.Name(), a)
			}
		}
	}
	if judgeSpans != 2 {
		t.Errorf("%s: judge root + chat spans = %d, want 2", path, judgeSpans)
	}
}

func parentName(exp *tracetest.InMemoryExporter, s sdktrace.ReadOnlySpan) string {
	for _, p := range exp.GetSpans().Snapshots() {
		if p.SpanContext().SpanID() == s.Parent().SpanID() {
			return p.Name()
		}
	}
	return ""
}

// Judge spans name the run they graded, inline and deferred, with capture
// off or on, and leave an empty variation key off.
func TestJudgeTargetAttributes(t *testing.T) {
	for _, variation := range []string{"opus", ""} {
		for _, capture := range []ContentCapture{CaptureOff, CaptureOn} {
			parent := enabledConfig("P", "", judgedConfig(nil))
			parent["_ldMeta"] = map[string]any{"enabled": true, "variationKey": variation, "version": 3}
			withFake(t, map[string]any{"cfg": parent, "judge": enabledConfig("P", "judge", nil)})
			h := targetJudgeHandler(capture)
			args := ConfigArgs{Key: "cfg", Handlers: []*ProviderHandler{h}}

			exp := withStampedSpans(t)
			resp, err := Config(args).Invoke(bg, "q", userCtx)
			if err != nil || resp.JudgeResults["judge"].Score != 0.5 {
				t.Fatalf("inline: resp=%+v err=%v", resp, err)
			}
			if resp.TrackData.RunID == "" || resp.TrackData.VariationKey != variation {
				t.Fatalf("graded track data = %+v", resp.TrackData)
			}
			checkJudgeTarget(t, "inline/"+variation, exp, resp.TrackData)

			args.SkipJudges = true
			resp, err = Config(args).Invoke(bg, "q", userCtx)
			if err != nil || len(resp.JudgeTasks) != 1 {
				t.Fatalf("tasks=%v err=%v", resp.JudgeTasks, err)
			}
			exp.Reset()
			if res, err := RunJudge(bg, resp.JudgeTasks[0], []*ProviderHandler{h}); err != nil || res == nil {
				t.Fatalf("deferred: res=%v err=%v", res, err)
			}
			checkJudgeTarget(t, "deferred/"+variation, exp, resp.TrackData)
		}
	}
}

// In a graph, a per-node judge names the node run it graded and the graph
// judge names the graph run carried by the graph span.
func TestGraphJudgeTargetAttributes(t *testing.T) {
	leaf := enabledConfig("P", "agent", judgedConfig(map[string]any{"instructions": "leaf work"}))
	withFake(t, twoNode(map[string]any{"leaf-node": leaf,
		"judge": enabledConfig("P", "judge", nil), "gj": enabledConfig("P", "judge", nil)}))
	exp := withStampedSpans(t)
	nodeRuns := map[string]TrackData{}
	h := createHandler(ProvidesFor{"P", ModeAgent}, func(ctx context.Context, req HandlerRequest) (HandlerResult, error) {
		td, _ := req.Variables[kitbase.VarLD].(TrackData)
		if req.Variables["response_to_evaluate"] == nil {
			nodeRuns[td.ConfigKey] = td
			return HandlerResult{Output: "out"}, nil
		}
		run := kitStartHandlerRun(ctx, kitbase.HandlerSpanOptions{Tracer: kitbase.TracerPrefix + "/test", Name: "judge " + td.ConfigKey, System: "p", RequestModel: "m"}, req, prompt{})
		run.Finish("m", `{"score":1,"reasoning":"r"}`)
		return HandlerResult{Output: `{"score":1,"reasoning":"r"}`}, nil
	}, nil, CaptureDefault)
	if _, err := Graph("g", GraphOptions{Handlers: []*ProviderHandler{h}, GraphJudge: "gj"}).Invoke(bg, "q", userCtx); err != nil {
		t.Fatal(err)
	}
	byName := map[string]map[string]string{}
	for _, s := range exp.GetSpans().Snapshots() {
		byName[s.Name()] = targetAttrs(s)
	}
	graphRun := byName[kitbase.GraphSpanName]["launchdarkly.run.id"]
	if graphRun == "" || nodeRuns["leaf-node"].RunID == "" {
		t.Fatalf("graph run %q, node runs %v, spans %v", graphRun, nodeRuns, byName)
	}
	for judge, want := range map[string][2]string{
		"judge": {nodeRuns["leaf-node"].RunID, "leaf-node"},
		"gj":    {graphRun, "g"},
	} {
		a := byName["judge "+judge]
		if a[targetRunID] != want[0] || a[targetConfigKey] != want[1] {
			t.Errorf("%s target = run %q config %q, want %q %q", judge, a[targetRunID], a[targetConfigKey], want[0], want[1])
		}
	}
	if a := byName[kitbase.GraphSpanName]; a[targetRunID] != "" {
		t.Errorf("graph span stamped with a judge target: %v", a)
	}
}
