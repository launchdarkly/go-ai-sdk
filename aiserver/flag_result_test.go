package aiserver

import (
	"context"
	"encoding/json"
	"reflect"
	"testing"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/launchdarkly/go-sdk-common/v3/ldvalue"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/launchdarkly/go-ai-sdk/internal/kitbase"
)

const (
	resultVariant = "feature_flag.result.variant"
	resultValue   = "feature_flag.result.value"
)

// flagJSON is the variation as the tracing hook with value inclusion writes it.
func flagJSON(v any) string {
	b, _ := json.Marshal(v)
	return ldvalue.Parse(b).JSONString()
}

// flagEvents maps each span name to its feature_flag event attributes.
func flagEvents(exp *tracetest.InMemoryExporter) map[string]map[string]string {
	out := map[string]map[string]string{}
	for _, s := range exp.GetSpans() {
		for _, e := range s.Events {
			if e.Name != "feature_flag" {
				continue
			}
			attrs := map[string]string{}
			for _, a := range e.Attributes {
				attrs[string(a.Key)] = a.Value.AsString()
			}
			out[s.Name] = attrs
		}
	}
	return out
}

// checkResult asserts the variant and value on one feature_flag event; an
// empty want means the attribute must be absent.
func checkResult(t *testing.T, path string, ev map[string]string, variant, value string) {
	t.Helper()
	if ev == nil {
		t.Errorf("%s: no feature_flag event", path)
		return
	}
	for name, want := range map[string]string{resultVariant: variant, resultValue: value} {
		got, has := ev[name]
		switch {
		case want == "" && has:
			t.Errorf("%s: %s = %q, want absent", path, name, got)
		case want != "" && !sameJSON(got, want):
			t.Errorf("%s: %s = %q, want %q", path, name, got, want)
		}
	}
}

// sameJSON compares JSON texts; an object's key order is not fixed.
func sameJSON(a, b string) bool {
	var x, y any
	if json.Unmarshal([]byte(a), &x) != nil || json.Unmarshal([]byte(b), &y) != nil {
		return a == b
	}
	return reflect.DeepEqual(x, y)
}

// rootHandler opens an SDK root named after the config it runs.
func rootHandler(mode Mode, capture ContentCapture) *ProviderHandler {
	return createHandler(ProvidesFor{"P", mode}, func(ctx context.Context, req HandlerRequest) (HandlerResult, error) {
		td, _ := req.Variables[kitbase.VarLD].(TrackData)
		out := "answer"
		if req.Variables["response_to_evaluate"] != nil {
			out = `{"score":0.5,"reasoning":"ok"}`
		}
		run := kitStartHandlerRun(ctx, kitbase.HandlerSpanOptions{Tracer: kitbase.TracerPrefix + "/test", Name: "root " + td.ConfigKey, System: "p", RequestModel: "m"}, req, prompt{})
		run.Finish("m", out)
		return HandlerResult{Output: out}, nil
	}, nil, capture)
}

// The feature_flag event carries the served variation key as the variant and
// the evaluated variation as the value when the handler captures content;
// otherwise the value is the JSON-quoted variation key, and with no key
// neither is written. Covers invoke, stream, inline and deferred judges, and
// a caller variable that tries to set the value.
func TestFeatureFlagResult(t *testing.T) {
	for _, variation := range []string{"opus", ""} {
		for _, capture := range []ContentCapture{CaptureOn, CaptureOff} {
			parent := enabledConfig("P", "", judgedConfig(nil))
			parent["_ldMeta"] = map[string]any{"enabled": true, "variationKey": variation, "version": 3}
			judge := enabledConfig("P", "judge", nil)
			withFake(t, map[string]any{"cfg": parent, "judge": judge})
			h := rootHandler(ModeMessages, capture)
			args := ConfigArgs{Key: "cfg", Handlers: []*ProviderHandler{h}}

			path := variation + "/capture-on"
			wantParent, wantJudge := flagJSON(parent), flagJSON(judge)
			if capture == CaptureOff {
				path = variation + "/capture-off"
				wantParent, wantJudge = "", `"v1"`
				if variation != "" {
					wantParent = `"` + variation + `"`
				}
			}

			exp := withSpans(t)
			forged := WithVariables(map[string]any{"__ldValue": "forged"})
			if _, err := Config(args).Invoke(bg, "q", userCtx, forged); err != nil {
				t.Fatal(err)
			}
			evs := flagEvents(exp)
			checkResult(t, path+"/invoke", evs["root cfg"], variation, wantParent)
			checkResult(t, path+"/inline judge", evs["root judge"], "v1", wantJudge)

			exp.Reset()
			for _, err := range Config(args).Stream(bg, "q", userCtx, forged) {
				if err != nil {
					t.Fatal(err)
				}
			}
			evs = flagEvents(exp)
			checkResult(t, path+"/stream", evs["root cfg"], variation, wantParent)
			checkResult(t, path+"/stream judge", evs["root judge"], "v1", wantJudge)

			args.SkipJudges = true
			resp, err := Config(args).Invoke(bg, "q", userCtx)
			if err != nil || len(resp.JudgeTasks) != 1 {
				t.Fatalf("tasks=%v err=%v", resp.JudgeTasks, err)
			}
			// A deferred task crosses a process boundary as JSON.
			b, err := json.Marshal(resp.JudgeTasks[0])
			if err != nil {
				t.Fatal(err)
			}
			var task JudgeTask
			if err := json.Unmarshal(b, &task); err != nil {
				t.Fatal(err)
			}
			exp.Reset()
			if res, err := RunJudge(bg, task, []*ProviderHandler{h}); err != nil || res == nil {
				t.Fatalf("deferred: res=%v err=%v", res, err)
			}
			checkResult(t, path+"/deferred judge", flagEvents(exp)["root judge"], "v1", wantJudge)
		}
	}
}

// Graph: each node's root carries its own variation; the graph span, which
// has no handler, follows the process-wide capture default.
func TestGraphFeatureFlagResult(t *testing.T) {
	for _, capture := range []ContentCapture{CaptureDefault, CaptureOff} {
		t.Run(map[ContentCapture]string{CaptureDefault: "on", CaptureOff: "off"}[capture], func(t *testing.T) {
			vars := twoNode(nil)
			graph := vars["g"].(map[string]any)
			graph["_ldMeta"] = map[string]any{"variationKey": "gv"}
			withFake(t, vars)
			if err := Configure(SDKOptions{CaptureContent: capture}); err != nil {
				t.Fatal(err)
			}
			exp := withSpans(t)
			h := rootHandler(ModeAgent, CaptureDefault)
			if _, err := Graph("g", GraphOptions{Handlers: []*ProviderHandler{h}}).Invoke(bg, "q", userCtx); err != nil {
				t.Fatal(err)
			}
			evs := flagEvents(exp)
			if capture == CaptureOff {
				checkResult(t, "off/graph", evs[kitbase.GraphSpanName], "gv", `"gv"`)
				checkResult(t, "off/root-node", evs["root root-node"], "v1", `"v1"`)
				return
			}
			checkResult(t, "on/graph", evs[kitbase.GraphSpanName], "gv", flagJSON(graph))
			checkResult(t, "on/root-node", evs["root root-node"], "v1", flagJSON(vars["root-node"]))
			checkResult(t, "on/leaf-node", evs["root leaf-node"], "v1", flagJSON(vars["leaf-node"]))
		})
	}
}

// Spans written without handler variables (native graph runners) have no
// evaluated variation: the variation key stands in, and an empty key
// writes neither attribute.
func TestFeatureFlagResultWithoutVariation(t *testing.T) {
	exp := withSpans(t)
	for _, key := range []string{"a<b", ""} {
		_, span := otel.Tracer("t").Start(bg, "s"+key)
		kitSetLDSpanAttributesFor(span, TrackData{ConfigKey: "k", VariationKey: key}, ldcontext.Context{})
		span.End()
	}
	evs := flagEvents(exp)
	checkResult(t, "keyed", evs["sa<b"], "a<b", `"a<b"`) // not HTML-escaped, as the tracing hook writes it
	checkResult(t, "empty", evs["s"], "", "")
}
