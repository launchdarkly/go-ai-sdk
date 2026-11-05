package telemetry

import (
	"context"
	"sync/atomic"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
)

type spanAttributesKey struct{}

// WithSpanAttributes binds attrs on ctx: every SDK span started while the
// binding is current gets each one, write-if-absent (applied by
// aiserver.ConversationIDSpanProcessor). Only string and integer values are
// bound; others are dropped. A nested binding adds to the enclosing one, and
// its value wins for a key both set.
func WithSpanAttributes(ctx context.Context, attrs ...attribute.KeyValue) context.Context {
	outer := BoundAttributes(ctx)
	merged := make([]attribute.KeyValue, 0, len(outer)+len(attrs))
	seen := map[attribute.Key]bool{}
	for _, a := range attrs {
		if t := a.Value.Type(); (t == attribute.STRING || t == attribute.INT64) && !seen[a.Key] {
			seen[a.Key] = true
			merged = append(merged, a)
		}
	}
	for _, a := range outer {
		if !seen[a.Key] {
			merged = append(merged, a)
		}
	}
	if len(merged) == 0 {
		return ctx
	}
	return context.WithValue(ctx, spanAttributesKey{}, merged)
}

// BoundAttributes is the WithSpanAttributes binding on ctx, if any.
func BoundAttributes(ctx context.Context) []attribute.KeyValue {
	a, _ := ctx.Value(spanAttributesKey{}).([]attribute.KeyValue)
	return a
}

// SpanCounter counts the SDK spans started while it is bound on a context.
type SpanCounter struct {
	n      atomic.Int64
	parent *SpanCounter
}

// Count is the number of SDK spans started so far under the counter,
// including those under a nested counter.
func (c *SpanCounter) Count() int { return int(c.n.Load()) }

type spanCounterKey struct{}

// WithSpanCounter binds a new counter on ctx. An enclosing counter keeps
// counting spans started under the new one.
func WithSpanCounter(ctx context.Context) (context.Context, *SpanCounter) {
	parent, _ := ctx.Value(spanCounterKey{}).(*SpanCounter)
	c := &SpanCounter{parent: parent}
	return context.WithValue(ctx, spanCounterKey{}, c), c
}

// CountSpan adds one SDK span to every counter bound on ctx.
func CountSpan(ctx context.Context) {
	for c, _ := ctx.Value(spanCounterKey{}).(*SpanCounter); c != nil; c = c.parent {
		c.n.Add(1)
	}
}

// Flush force-flushes the process tracer provider, so a batch exporter sends
// its queued spans now. A provider without ForceFlush, such as the no-op
// default, makes it a no-op.
func Flush(ctx context.Context) error {
	if f, ok := otel.GetTracerProvider().(interface{ ForceFlush(context.Context) error }); ok {
		return f.ForceFlush(ctx)
	}
	return nil
}
