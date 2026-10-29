package aiserver

import (
	"context"
	"testing"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"
	"go.opentelemetry.io/otel/trace/noop"
)

func withTracerProvider(t *testing.T, tp trace.TracerProvider) {
	t.Helper()
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(tp)
	t.Cleanup(func() { otel.SetTracerProvider(prev) })
}

func tracerFor(name string) trace.Tracer { return otel.Tracer(name) }

// processorSpans installs a provider with the SDK span processor, as
// InitClient does, in front of an in-memory exporter.
func processorSpans(t *testing.T) *tracetest.InMemoryExporter {
	t.Helper()
	exp := tracetest.NewInMemoryExporter()
	withTracerProvider(t, sdktrace.NewTracerProvider(
		sdktrace.WithSpanProcessor(ConversationIDSpanProcessor{}),
		sdktrace.WithSyncer(exp)))
	return exp
}

func spanAttrMap(t *testing.T, exp *tracetest.InMemoryExporter, name string) map[string]attribute.Value {
	t.Helper()
	for _, s := range exp.GetSpans().Snapshots() {
		if s.Name() == name {
			m := map[string]attribute.Value{}
			for _, a := range s.Attributes() {
				m[string(a.Key)] = a.Value
			}
			return m
		}
	}
	t.Fatalf("no span %q", name)
	return nil
}

func TestWithSpanAttributesStampsSDKSpansWriteIfAbsent(t *testing.T) {
	exp := processorSpans(t)
	ctx := WithSpanAttributes(bg,
		attribute.String("example.label", "outer"), attribute.Int("example.count", 3),
		attribute.Bool("example.flag", true), attribute.Float64("example.ratio", 0.5)) // dropped: not string/int
	ctx = WithSpanAttributes(ctx, attribute.String("example.label", "inner"), attribute.String("example.extra", "x"))

	_, sdk := tracerFor(TracerPrefix+"/test").Start(ctx, "sdk")
	sdk.End()
	_, preset := tracerFor(TracerPrefix+"/test").Start(ctx, "preset", trace.WithAttributes(attribute.String("example.label", "own")))
	preset.End()
	_, foreign := tracerFor("example.com/other").Start(ctx, "foreign")
	foreign.End()

	a := spanAttrMap(t, exp, "sdk")
	if a["example.label"].AsString() != "inner" || a["example.count"].AsInt64() != 3 || a["example.extra"].AsString() != "x" {
		t.Errorf("sdk span attrs %v", a)
	}
	for _, k := range []string{"example.flag", "example.ratio"} {
		if _, has := a[k]; has {
			t.Errorf("%s: only string and integer values are bound", k)
		}
	}
	if got := spanAttrMap(t, exp, "preset")["example.label"].AsString(); got != "own" {
		t.Errorf("an attribute the span already has must be kept, got %q", got)
	}
	if len(spanAttrMap(t, exp, "foreign")) != 0 {
		t.Error("a non-SDK span must not be stamped")
	}
}

func TestWithSpanAttributesAndConversationIDCompose(t *testing.T) {
	exp := processorSpans(t)
	ctx := WithConversationID(bg, "outer-conv")
	ctx = WithSpanAttributes(WithConversationID(ctx, "inner-conv"), attribute.String("example.label", "v"))
	_, s := tracerFor(TracerPrefix+"/test").Start(ctx, "sdk")
	s.End()
	a := spanAttrMap(t, exp, "sdk")
	if a["gen_ai.conversation.id"].AsString() != "inner-conv" || a["example.label"].AsString() != "v" {
		t.Errorf("attrs %v", a)
	}
}

func TestNoBindingLeavesSpansUnchanged(t *testing.T) {
	exp := processorSpans(t)
	_, s := tracerFor(TracerPrefix+"/test").Start(bg, "sdk")
	s.End()
	if a := spanAttrMap(t, exp, "sdk"); len(a) != 0 {
		t.Errorf("no binding must stamp nothing, got %v", a)
	}
}

func TestWithSpanCounterCountsSDKSpans(t *testing.T) {
	processorSpans(t)
	ctx, outer := WithSpanCounter(bg)
	start := func(ctx context.Context, tracer string) {
		_, s := tracerFor(tracer).Start(ctx, "s")
		s.End()
	}
	start(ctx, TracerPrefix+"/test")
	start(ctx, "example.com/other") // not counted
	inner, nested := WithSpanCounter(ctx)
	start(inner, TracerPrefix+"/test")
	start(inner, TracerPrefix+"/test")
	start(bg, TracerPrefix+"/test") // outside both
	if outer.Count() != 3 || nested.Count() != 2 {
		t.Fatalf("outer %d nested %d, want 3 and 2", outer.Count(), nested.Count())
	}
	if _, fresh := WithSpanCounter(bg); fresh.Count() != 0 {
		t.Fatal("a new counter starts at zero")
	}
}

type flushRecorder struct {
	trace.TracerProvider
	flushed int
}

func (f *flushRecorder) ForceFlush(context.Context) error { f.flushed++; return nil }

func TestFlushTelemetry(t *testing.T) {
	rec := &flushRecorder{TracerProvider: noop.NewTracerProvider()}
	withTracerProvider(t, rec)
	if err := FlushTelemetry(bg); err != nil || rec.flushed != 1 {
		t.Fatalf("flush err %v count %d", err, rec.flushed)
	}
	withTracerProvider(t, noop.NewTracerProvider())
	if err := FlushTelemetry(bg); err != nil {
		t.Fatalf("a provider without ForceFlush must be a no-op: %v", err)
	}
}
