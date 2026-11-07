package langchainmessages

import (
	"testing"

	sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

// assertSpanTree checks parentage on spans from a real exporter: exactly one
// root (no parent), and every other span in its trace and a direct child of
// it (chat and execute_tool spans are siblings).
func assertSpanTree(t *testing.T, path string, spans []sdktrace.ReadOnlySpan) {
	t.Helper()
	var roots []sdktrace.ReadOnlySpan
	for _, s := range spans {
		if !s.Parent().IsValid() {
			roots = append(roots, s)
		}
	}
	if len(roots) != 1 || len(spans) < 2 {
		t.Fatalf("%s: %d roots among %d spans", path, len(roots), len(spans))
	}
	root := roots[0].SpanContext()
	for _, s := range spans {
		if s.SpanContext().SpanID() == root.SpanID() {
			continue
		}
		if s.SpanContext().TraceID() != root.TraceID() || s.Parent().SpanID() != root.SpanID() {
			t.Errorf("%s: %s is not a direct child of the root", path, s.Name())
		}
	}
}
