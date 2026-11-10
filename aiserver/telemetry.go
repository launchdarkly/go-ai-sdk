package aiserver

import (
	"bytes"
	"context"
	"encoding/json"
	"sort"
	"strings"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/launchdarkly/go-sdk-common/v3/ldvalue"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"

	"github.com/launchdarkly/go-ai-sdk/internal/kitbase"
	"github.com/launchdarkly/go-ai-sdk/internal/telemetry"
)

type conversationKey struct{}

// WithConversationID binds a conversation id on ctx. Go's explicit
// context.Context replaces TS AsyncLocalStorage / Python contextvars, so
// "stream() binds at call time" holds by construction: Stream takes ctx.
func WithConversationID(ctx context.Context, id string) context.Context {
	if id == "" {
		return ctx
	}
	return context.WithValue(ctx, conversationKey{}, id)
}

func conversationID(ctx context.Context) string {
	id, _ := ctx.Value(conversationKey{}).(string)
	return id
}

// ConversationIDSpanProcessor stamps gen_ai.conversation.id and any
// attributes the SDK bound on the span's context write-if-absent, and counts
// the span for any SDK span counter bound there, on SDK spans only. With nothing bound it does
// nothing.
type ConversationIDSpanProcessor struct{}

func (ConversationIDSpanProcessor) OnStart(parent context.Context, s sdktrace.ReadWriteSpan) {
	if !strings.HasPrefix(s.InstrumentationScope().Name, kitbase.TracerPrefix) {
		return
	}
	telemetry.CountSpan(parent)
	var stamp []attribute.KeyValue
	if id := conversationID(parent); id != "" {
		stamp = append(stamp, attribute.String("gen_ai.conversation.id", id))
	}
	stamp = append(stamp, telemetry.BoundAttributes(parent)...)
	if len(stamp) == 0 {
		return
	}
	has := map[attribute.Key]bool{}
	for _, a := range s.Attributes() {
		has[a.Key] = true
	}
	var add []attribute.KeyValue
	for _, a := range stamp {
		if !has[a.Key] {
			has[a.Key] = true
			add = append(add, a)
		}
	}
	if len(add) > 0 {
		s.SetAttributes(add...)
	}
}
func (ConversationIDSpanProcessor) OnEnd(sdktrace.ReadOnlySpan)      {}
func (ConversationIDSpanProcessor) Shutdown(context.Context) error   { return nil }
func (ConversationIDSpanProcessor) ForceFlush(context.Context) error { return nil }

// compactSortedJSON renders per-kind keys byte-identically to TS/Python:
// sorted keys, no HTML escaping, literal non-ASCII.
func compactSortedJSON(m map[string]string) string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var buf bytes.Buffer
	buf.WriteByte('{')
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	for i, k := range keys {
		if i > 0 {
			buf.WriteByte(',')
		}
		_ = enc.Encode(k)
		buf.Truncate(buf.Len() - 1) // Encode appends '\n'
		buf.WriteByte(':')
		_ = enc.Encode(m[k])
		buf.Truncate(buf.Len() - 1)
	}
	buf.WriteByte('}')
	return buf.String()
}

// Unexported variables entries beside kitbase.VarLD and kitbase.VarLDContext.
const (
	// ldContextValue holds the typed context alongside the template map; Go
	// templates need a map but identity needs the typed ldcontext.Context.
	varLDContextTyped = "__ldContextTyped"
	// varLDSpanAttributes holds extra root span attributes. Its unexported
	// type means template variables cannot set it.
	varLDSpanAttributes = "__ldSpanAttributes"
)

type spanAttributes []attribute.KeyValue

// varLDNoFlagEvent marks a run whose root span gets no feature_flag event.
// Its unexported value type keeps template variables from setting it.
const varLDNoFlagEvent = "__ldNoFlagEvent"

type noFlagEvent struct{}

// varLDValue holds the evaluated variation for feature_flag.result.value. Its
// unexported value type keeps template variables from setting it.
const varLDValue = "__ldValue"

type flagValue struct{ v ldvalue.Value }

// setLDSpanAttributesFromVars writes the LaunchDarkly config-association and context
// identity attributes on a handler's root span from variables. Without an
// `__ld` entry (a handler called directly) only launchdarkly.operation.type
// is written, so the span is still found by the LLM dashboard filter.
func setLDSpanAttributesFromVars(span trace.Span, variables map[string]any) {
	td, ok := variables[kitbase.VarLD].(TrackData)
	if !ok {
		span.SetAttributes(attribute.String("launchdarkly.operation.type", "gen_ai"))
		return
	}
	c, _ := variables[varLDContextTyped].(ldcontext.Context)
	_, skipFlagEvent := variables[varLDNoFlagEvent].(noFlagEvent)
	value, _ := variables[varLDValue].(flagValue)
	setLDSpanAttributes(span, td, c, !skipFlagEvent, value.v)
	if extra, ok := variables[varLDSpanAttributes].(spanAttributes); ok {
		span.SetAttributes(extra...)
	}
}

// kitSetLDSpanAttributesFor writes the same attributes for a span that has no
// handler variables, such as a native graph runner's graph span. A zero
// Context writes no identity.
func kitSetLDSpanAttributesFor(span trace.Span, td TrackData, userCtx ldcontext.Context) {
	setLDSpanAttributes(span, td, userCtx, true, ldvalue.Null())
}

// value is the evaluated variation, or Null when the span may not carry it.
func setLDSpanAttributes(span trace.Span, td TrackData, userCtx ldcontext.Context, flagEvent bool, value ldvalue.Value) {
	attrs := []attribute.KeyValue{
		attribute.String("launchdarkly.operation.type", "gen_ai"),
		attribute.String("launchdarkly.config.key", td.ConfigKey),
		attribute.String("launchdarkly.variation.key", td.VariationKey),
		attribute.String("launchdarkly.run.id", td.RunID),
	}
	if td.GraphKey != "" {
		attrs = append(attrs, attribute.String("launchdarkly.graph.key", td.GraphKey))
	}
	eventAttrs := []attribute.KeyValue{
		attribute.String("feature_flag.key", td.ConfigKey),
		attribute.String("feature_flag.provider.name", "LaunchDarkly"),
	}
	if td.EnvironmentID != "" {
		eventAttrs = append(eventAttrs, attribute.String("feature_flag.set.id", td.EnvironmentID))
	}
	if td.VariationKey != "" {
		eventAttrs = append(eventAttrs, attribute.String("feature_flag.result.variant", td.VariationKey))
	}
	// Serialized as the SDK tracing hook does with value inclusion on; without
	// the variation, the key stands in as a JSON string.
	switch {
	case !value.IsNull():
		eventAttrs = append(eventAttrs, attribute.String("feature_flag.result.value", value.JSONString()))
	case td.VariationKey != "":
		eventAttrs = append(eventAttrs, attribute.String("feature_flag.result.value", ldvalue.String(td.VariationKey).JSONString()))
	}
	if userCtx.IsDefined() {
		if canonical, perKind, ok := kitbase.ContextIdentity(userCtx); ok {
			eventAttrs = append(eventAttrs,
				attribute.String("feature_flag.context.id", canonical),
				attribute.String("feature_flag.contextKeys", compactSortedJSON(perKind)))
			for k, v := range perKind {
				attrs = append(attrs, attribute.String("context.contextKeys."+k, v))
			}
		}
	}
	span.SetAttributes(attrs...)
	if flagEvent {
		span.AddEvent("feature_flag", trace.WithAttributes(eventAttrs...))
	}
}
