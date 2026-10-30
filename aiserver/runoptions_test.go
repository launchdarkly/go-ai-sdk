package aiserver

import (
	"context"
	"testing"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/launchdarkly/go-sdk-common/v3/ldvalue"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/launchdarkly/go-ai-sdk/internal/bridge"
)

func TestZeroRunOptionsAreAPlainConfigRun(t *testing.T) {
	f := withFake(t, map[string]any{"cfg": enabledConfig("openai", "completion", nil)})
	h := stubHandler("openai", ModeMessages, "ok", nil, nil)
	run := bridge.ConfigWithOptions(ConfigArgs{Key: "cfg", Handlers: []*ProviderHandler{h}}, bridge.RunOptions{}).(*AIConfig)
	resp, err := run.Invoke(context.Background(), "hi", userCtx)
	if err != nil || resp.Response != "ok" || f.calls["cfg"] != 1 {
		t.Fatalf("resp %+v, err %v, variation calls %v", resp, err, f.calls)
	}
	if resp.TrackData.RunID == "" || resp.TrackData.extra != nil {
		t.Fatalf("track data %+v", resp.TrackData)
	}
}

func TestRunOptionsReplaceTheVariationAndExtendTracking(t *testing.T) {
	f := withFake(t, map[string]any{})
	rec := tracetest.NewSpanRecorder()
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(rec)))
	t.Cleanup(func() { otel.SetTracerProvider(prev) })

	var seen HandlerRequest
	h := createHandler(ProvidesFor{"openai", ModeMessages}, func(ctx context.Context, req HandlerRequest) (HandlerResult, error) {
		seen = req
		StartHandlerRun(ctx, HandlerSpanOptions{Tracer: TracerPrefix + "/test", Name: "test"}, req, Prompt{}).Finish("", "ok")
		return HandlerResult{Output: "ok", Usage: map[string]any{"input": 1, "output": 2}}, nil
	}, nil, false)
	value := ldvalue.FromJSONMarshal(enabledConfig("openai", "completion", nil))
	variation := func(bridge.VariationClient, string, ldcontext.Context, ldvalue.Value) (ldvalue.Value, error) {
		return value, nil
	}
	run := bridge.ConfigWithOptions(ConfigArgs{Key: "cfg", Handlers: []*ProviderHandler{h}}, bridge.RunOptions{
		Variation:      variation,
		RunID:          "run-1",
		TrackData:      map[string]any{"extra": "x"},
		SpanAttributes: []attribute.KeyValue{attribute.Bool("custom.attr", true)},
	}).(*AIConfig)
	resp, err := run.Invoke(context.Background(), "hi", userCtx)
	if err != nil || f.calls["cfg"] != 0 {
		t.Fatalf("err %v, JSONVariation calls %v", err, f.calls)
	}
	if resp.TrackData.RunID != "run-1" || seen.Variables[VarLD].(TrackData).RunID != "run-1" {
		t.Fatalf("run id %q", resp.TrackData.RunID)
	}
	for _, name := range []string{"$ld:ai:duration:total", "$ld:ai:generation:success", "$ld:ai:tokens:total"} {
		ev := f.events(name)
		if len(ev) != 1 || ev[0].Data["extra"] != "x" || ev[0].Data["runId"] != "run-1" || ev[0].Data["variationKey"] != "v1" {
			t.Fatalf("%s: %v", name, ev)
		}
	}
	spans := rec.Ended()
	if len(spans) != 1 {
		t.Fatalf("spans %d", len(spans))
	}
	found := false
	for _, a := range spans[0].Attributes() {
		found = found || (a.Key == "custom.attr" && a.Value.AsBool())
	}
	if !found {
		t.Fatal("span attribute missing from the root span")
	}
}
