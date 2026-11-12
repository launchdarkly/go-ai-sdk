// Package kitbase holds the type-free handler helpers shared by aiserver and
// the handler packages. It imports nothing from aiserver.
package kitbase

import (
	"context"
	"encoding/json"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/launchdarkly/go-ai-sdk/internal/judging"
)

// GraphSpanName is the graph-level span.
const GraphSpanName = "launchdarkly.graph"

// OutputFormatInstruction is the best-effort structured-output suffix
// appended to a system prompt by handlers without native structured output.
func OutputFormatInstruction(schema map[string]any) string {
	b, _ := json.Marshal(schema)
	return "Respond with valid JSON matching this schema:\n" + string(b)
}

// WithOutputFormat appends OutputFormatInstruction to system when schema is set.
func WithOutputFormat(system string, schema map[string]any) string {
	if schema == nil {
		return system
	}
	if system == "" {
		return OutputFormatInstruction(schema)
	}
	return system + "\n\n" + OutputFormatInstruction(schema)
}

// SpanUsage is the token counts a span reports. Input is cache-inclusive: the
// caller folds provider cache buckets in before calling SetUsageSpanAttributes.
type SpanUsage struct {
	Input, Output, CacheRead, CacheCreation float64
	// Unreported marks a turn whose response carried no usage. Its chat span
	// still writes zeros, but the root writes usage only once a turn reported
	// some: zeros there would claim the run cost nothing.
	Unreported bool
}

// Add accumulates o into u.
func (u *SpanUsage) Add(o SpanUsage) {
	u.Input += o.Input
	u.Output += o.Output
	u.CacheRead += o.CacheRead
	u.CacheCreation += o.CacheCreation
}

// SetUsageSpanAttributes writes the token usage span attributes and their
// OpenLLMetry aliases. Total is always input + output.
func SetUsageSpanAttributes(span trace.Span, u SpanUsage) {
	in, out := int64(u.Input), int64(u.Output)
	span.SetAttributes(
		attribute.Int64("gen_ai.usage.input_tokens", in),
		attribute.Int64("gen_ai.usage.output_tokens", out),
		attribute.Int64("gen_ai.usage.total_tokens", in+out),
		attribute.Int64("gen_ai.usage.prompt_tokens", in),
		attribute.Int64("gen_ai.usage.completion_tokens", out),
		attribute.Int64("gen_ai.usage.cache_read.input_tokens", int64(u.CacheRead)),
		attribute.Int64("gen_ai.usage.cache_creation.input_tokens", int64(u.CacheCreation)),
	)
}

// HandlerSpanOptions describe a handler's root span.
type HandlerSpanOptions struct {
	Tracer       string // instrumentation scope; must start with TracerPrefix
	Name         string // root span name: "invoke_agent" for every handler
	System       string // gen_ai.system, e.g. "anthropic"
	Operation    string // gen_ai.operation.name; "" means "invoke_agent"
	RequestModel string
	ProviderName string // gen_ai.provider.name; "" means System
}

// VarLD and VarLDContext are the handler variables entries: the run's track
// data and the template view of its LaunchDarkly context.
const (
	VarLD        = "__ld"
	VarLDContext = "ldContext"
)

// TracerPrefix scopes the tracers whose spans get a conversation id.
const TracerPrefix = "github.com/launchdarkly/go-ai-sdk"

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

// ContextToMap is the template view of an LD context, injected as
// variables.ldContext.
func ContextToMap(c ldcontext.Context) map[string]any {
	b, err := json.Marshal(c)
	if err != nil {
		return map[string]any{}
	}
	var m map[string]any
	_ = json.Unmarshal(b, &m)
	return m
}

// WildcardProvider matches any provider name.
const WildcardProvider = "*"

// ParseJSONWithPossibleFences parses JSON optionally wrapped in markdown
// fences. ok is false when the text is not JSON; it never logs.
func ParseJSONWithPossibleFences(text string) (value any, ok bool) {
	return judging.ParseJSONWithPossibleFences(text)
}

// Run is one handler call's span tree: a root plus sibling child spans for
// each model turn and tool call. handlerkit.StartHandlerRun opens one.
type Run interface {
	// Context is the root span's context; child spans and provider calls use it.
	Context() context.Context
	// RootSpan is the handler's root span.
	RootSpan() trace.Span
	// StartChat opens a `chat <model>` span for one provider turn.
	StartChat(model string) trace.Span
	// EndChat records one turn's usage on its chat span and accumulates it
	// for the root.
	EndChat(s trace.Span, responseModel string, u SpanUsage, finishReason string)
	// StartTool opens an `execute_tool <name>` span, a sibling of the chat spans.
	StartTool(name, callID string) trace.Span
	// EndTool ends a tool span; a non-nil err marks it ERROR.
	EndTool(s trace.Span, err error)
	// ToolArguments records a tool call's arguments, subject to the content gate.
	ToolArguments(s trace.Span, arguments any)
	// ToolResult records a tool call's result, subject to the content gate.
	ToolResult(s trace.Span, result any)
	// AddUsage accumulates usage that has no chat span of its own.
	AddUsage(u SpanUsage)
	// Finish ends the root on success.
	Finish(responseModel, output string)
	// Fail records err on every open span and the root.
	Fail(err error)
	// EndOnPanic, deferred right after StartHandlerRun, ends every open span
	// ERROR on a panic, then re-panics.
	EndOnPanic()
	// Abandon ends every open span as abandoned, with UNSET status.
	Abandon()
}
