package aiserver

import (
	"bytes"
	"context"
	"encoding/json"
	"sort"
	"strings"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"
)

// TracerPrefix scopes the tracers whose spans get a conversation id.
const TracerPrefix = "github.com/launchdarkly/go-ai-sdk"

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

// ConversationIDSpanProcessor stamps gen_ai.conversation.id write-if-absent
// on SDK spans only.
type ConversationIDSpanProcessor struct{}

func (ConversationIDSpanProcessor) OnStart(parent context.Context, s sdktrace.ReadWriteSpan) {
	id := conversationID(parent)
	if id == "" || !strings.HasPrefix(s.InstrumentationScope().Name, TracerPrefix) {
		return
	}
	for _, a := range s.Attributes() {
		if a.Key == "gen_ai.conversation.id" {
			return
		}
	}
	s.SetAttributes(attribute.String("gen_ai.conversation.id", id))
}
func (ConversationIDSpanProcessor) OnEnd(sdktrace.ReadOnlySpan)      {}
func (ConversationIDSpanProcessor) Shutdown(context.Context) error   { return nil }
func (ConversationIDSpanProcessor) ForceFlush(context.Context) error { return nil }

// ContextIdentity returns the canonical key and per-kind keys, or
// ok=false. ldcontext.Context.FullyQualifiedKey already implements the
// canonical-key rules (sorting, %/: escaping, bare user key).
func ContextIdentity(c ldcontext.Context) (canonical string, perKind map[string]string, ok bool) {
	if c.Err() != nil {
		return "", nil, false
	}
	perKind = map[string]string{}
	for i := 0; i < c.IndividualContextCount(); i++ {
		ic := c.IndividualContextByIndex(i)
		if ic.Key() != "" {
			perKind[string(ic.Kind())] = ic.Key()
		}
	}
	if len(perKind) == 0 {
		return "", nil, false
	}
	return c.FullyQualifiedKey(), perKind, true
}

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

// LDMeta is the `__ld` variables entry handlers read.
const (
	VarLD        = "__ld"
	VarLDContext = "ldContext"
	// ldContextValue holds the typed context alongside the template map; Go
	// templates need a map but identity needs the typed ldcontext.Context.
	varLDContextTyped = "__ldContextTyped"
)

// SetLDSpanAttributes writes the LaunchDarkly config-association and context
// identity attributes on a handler's root span from variables. Without an
// `__ld` entry (a handler called directly) only launchdarkly.operation.type
// is written, so the span is still found by the LLM dashboard filter.
func SetLDSpanAttributes(span trace.Span, variables map[string]any) {
	td, ok := variables[VarLD].(TrackData)
	if !ok {
		span.SetAttributes(attribute.String("launchdarkly.operation.type", "gen_ai"))
		return
	}
	c, _ := variables[varLDContextTyped].(ldcontext.Context)
	SetLDSpanAttributesFor(span, td, c)
}

// SetLDSpanAttributesFor writes the same attributes for a span that has no
// handler variables, such as a native graph runner's graph span. A zero
// Context writes no identity.
func SetLDSpanAttributesFor(span trace.Span, td TrackData, userCtx ldcontext.Context) {
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
	if userCtx.IsDefined() {
		if canonical, perKind, ok := ContextIdentity(userCtx); ok {
			eventAttrs = append(eventAttrs,
				attribute.String("feature_flag.context.id", canonical),
				attribute.String("feature_flag.contextKeys", compactSortedJSON(perKind)))
			for k, v := range perKind {
				attrs = append(attrs, attribute.String("context.contextKeys."+k, v))
			}
		}
	}
	span.SetAttributes(attrs...)
	span.AddEvent("feature_flag", trace.WithAttributes(eventAttrs...))
}
