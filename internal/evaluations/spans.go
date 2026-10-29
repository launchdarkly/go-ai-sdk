package evaluations

import (
	"context"
	"fmt"
	"maps"
	"reflect"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/bridge"
	"github.com/launchdarkly/go-ai-sdk/internal/trajectory"
)

// Evaluation spans. The core offers only feature-neutral hooks
// (WithConversationID, WithSpanAttributes, WithSpanCounter and the content
// helpers); every evaluation name and the row binding live here.

const (
	// spanTracer is the SDK client tracer, so the processor treats these as
	// SDK spans.
	spanTracer   = aiserver.TracerPrefix + "/aiserver"
	rowSpanName  = "launchdarkly.dataset_row"
	attrRunID    = "launchdarkly.evaluation_run.id"
	attrRowIndex = "launchdarkly.evaluation_run.row_index"
)

// rowNamespace is the wire contract shared with ai-evaluator's
// conversationId; there is no second one.
var rowNamespace = uuid.MustParse("6f0c2a4e-3b6d-4d0e-9c1a-2e6b7f8a9d01")

// rowConversationID is the UUID5 of "{evaluationRunId}:{rowIndex}"; row 0 is
// the suffix ":0".
func rowConversationID(runID string, rowIndex int) string {
	return uuid.NewSHA1(rowNamespace, fmt.Appendf(nil, "%s:%d", runID, rowIndex)).String()
}

func rowAttributes(runID string, rowIndex int) []attribute.KeyValue {
	return []attribute.KeyValue{
		attribute.String("gen_ai.conversation.id", rowConversationID(runID, rowIndex)),
		attribute.String(attrRunID, runID),
		attribute.Int(attrRowIndex, rowIndex),
	}
}

// bindRow makes every SDK span started under ctx carry the row's
// conversation id (shadowing any outer one) and both run attributes.
func bindRow(ctx context.Context, runID string, rowIndex int) context.Context {
	ctx = aiserver.WithConversationID(ctx, rowConversationID(runID, rowIndex))
	return aiserver.WithSpanAttributes(ctx, rowAttributes(runID, rowIndex)[1:]...)
}

func tracer() trace.Tracer { return otel.Tracer(spanTracer) }

// rowTrace is one row's trace: a launchdarkly.dataset_row root that opens
// before generation and ends after the row's last criterion.
type rowTrace struct {
	ctx     context.Context // row binding on, root current
	root    trace.Span
	attrs   []attribute.KeyValue
	pending atomic.Int64
	once    sync.Once
	// processorActive is whether the SDK span processor saw the root. Only
	// then can a span count of zero mean the handler opened no span; under a
	// provider without it the count is always zero, and writing on zero would
	// trace a handler's call twice.
	processorActive bool
}

// startRow opens the row root as a new trace, so each row is its own trace
// whatever span the caller has open.
func startRow(ctx context.Context, runID string, rowIndex int) *rowTrace {
	attrs := rowAttributes(runID, rowIndex)
	probe, counter := aiserver.WithSpanCounter(bindRow(ctx, runID, rowIndex))
	ctx, root := tracer().Start(probe, rowSpanName, trace.WithNewRoot(), trace.WithAttributes(attrs...))
	return &rowTrace{ctx: ctx, root: root, attrs: attrs, processorActive: counter.Count() > 0}
}

// finishGeneration sets the root status, then ends the root now unless
// criteria remain to run on the row.
func (t *rowTrace) finishGeneration(err error, criteria int) {
	if err != nil {
		t.root.SetAttributes(attribute.String("error.type", errorType(err)))
		t.root.SetStatus(codes.Error, err.Error())
	} else {
		t.root.SetStatus(codes.Ok, "")
	}
	if err != nil || criteria == 0 {
		t.end()
		return
	}
	t.pending.Store(int64(criteria))
}

// criterionDone ends the root after the row's last criterion.
func (t *rowTrace) criterionDone() {
	if t.pending.Add(-1) == 0 {
		t.end()
	}
}

func (t *rowTrace) end() { t.once.Do(func() { t.root.End() }) }

// errorType is the error's type name without package or pointer, the Go
// form of Python's type(error).__name__: "errorString" for errors.New.
func errorType(err error) string {
	if err == nil {
		return ""
	}
	t := reflect.TypeOf(err)
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if t.Name() == "" {
		return t.String()
	}
	return t.Name()
}

// handlerCall brackets one generation or judge call. The context binds the
// handler's capture setting, so runner-written content follows the same
// gate as the handler's own spans, and a span counter that tells whether the
// handler opened any SDK span.
type handlerCall struct {
	ctx     context.Context
	counter *aiserver.SpanCounter
	row     *rowTrace
}

func beginCall(parent context.Context, h *aiserver.ProviderHandler, row *rowTrace) handlerCall {
	ctx, counter := aiserver.WithSpanCounter(bridge.StartInvocation(parent, uint8(h.CaptureContent)))
	return handlerCall{ctx: ctx, counter: counter, row: row}
}

// shouldDescribe reports whether the runner writes the call's spans: only
// when the handler opened no SDK span, so a call is never traced twice, and
// only when the SDK span processor is active to have counted them.
func (c handlerCall) shouldDescribe() bool {
	return c.row != nil && c.row.processorActive && c.counter.Count() == 0
}

// chatCall is what a plain function was handed and what came back.
type chatCall struct {
	config     aiserver.AiConfig
	variables  map[string]any
	finalUser  *string                 // nil when there is no user input
	toolCalls  []trajectory.Invocation // generation only
	output     any
	usage      map[string]any
	err        error
	start, end time.Time
}

// writeChat writes a `chat {model}` span under c.ctx timed to the call, and
// for a generation an `execute_tool` sibling per completed tool call.
func (c handlerCall) writeChat(call chatCall, attrs []attribute.KeyValue) {
	cfg := call.config
	span := c.startSpan(strings.TrimSpace("chat "+cfg.Model.Name), call.start, append(slices.Clone(attrs),
		append([]attribute.KeyValue{attribute.String("gen_ai.operation.name", "chat")}, modelIdentity(cfg)...)...))
	content := promptContent(cfg, call.variables, call.finalUser)
	if call.err == nil {
		if call.usage != nil {
			n := aiserver.ParseUsage(call.usage)
			u := aiserver.SpanUsage{Input: n.Input, Output: n.Output}
			if n.InputDetails != nil {
				u.CacheRead, u.CacheCreation = n.InputDetails.CacheRead, n.InputDetails.CacheCreation
			}
			aiserver.SetUsageSpanAttributes(span, u)
		}
		var parts []aiserver.MessagePart
		for i, inv := range call.toolCalls {
			parts = append(parts, aiserver.ToolCallPart(toolCallID(i), inv.Name, inv.Arguments))
		}
		if call.output != nil {
			parts = append(parts, aiserver.TextPart(aiserver.ToolResultString(call.output)))
		}
		if len(parts) > 0 {
			content.OutputMessages = []aiserver.ChatMessage{{Role: "assistant", Parts: parts}}
		}
	} else {
		span.RecordError(call.err)
	}
	aiserver.SetChatContent(c.ctx, span, content)
	endSpan(span, call.err, errorType(call.err), call.end)

	for i, inv := range call.toolCalls {
		if !inv.Done {
			continue // still open: there is no end to stamp
		}
		tool := c.startSpan("execute_tool "+inv.Name, inv.Started, append(slices.Clone(attrs),
			attribute.String("gen_ai.operation.name", "execute_tool"),
			attribute.String("gen_ai.tool.name", inv.Name),
			attribute.String("gen_ai.tool.call.id", toolCallID(i)),
			attribute.String("gen_ai.tool.type", "function")))
		aiserver.SetToolCallArguments(c.ctx, tool, inv.Arguments)
		var terr error
		if inv.Failed {
			terr = fmt.Errorf("%s", inv.Error)
		} else {
			aiserver.SetToolCallResult(c.ctx, tool, inv.Result)
		}
		endSpan(tool, terr, "tool_error", inv.Ended)
	}
}

func (c handlerCall) startSpan(name string, start time.Time, attrs []attribute.KeyValue) trace.Span {
	_, span := tracer().Start(c.ctx, name, trace.WithTimestamp(start), trace.WithAttributes(attrs...))
	return span
}

func endSpan(span trace.Span, err error, errType string, end time.Time) {
	if err != nil {
		span.SetAttributes(attribute.String("error.type", errType))
		span.SetStatus(codes.Error, err.Error())
	} else {
		span.SetStatus(codes.Ok, "")
	}
	span.End(trace.WithTimestamp(end))
}

// toolCallID numbers the recorded calls call_1, call_2, … in start order;
// the handler reports no provider call id.
func toolCallID(i int) string { return fmt.Sprintf("call_%d", i+1) }

// modelIdentity is the provider and model a config names, under both
// spellings of the provider key: gen_ai.system (pre-1.37 semconv, which
// dashboards still read) and gen_ai.provider.name.
func modelIdentity(cfg aiserver.AiConfig) []attribute.KeyValue {
	provider := strings.ToLower(cfg.Provider.Name)
	return []attribute.KeyValue{
		attribute.String("gen_ai.system", provider),
		attribute.String("gen_ai.provider.name", provider),
		attribute.String("gen_ai.request.model", cfg.Model.Name),
	}
}

// promptContent is the config's instructions and system messages as the
// system instructions, and its other messages then finalUser as the input,
// all rendered with variables.
func promptContent(cfg aiserver.AiConfig, variables map[string]any, finalUser *string) aiserver.ChatContent {
	var system []string
	if cfg.Instructions != "" {
		system = append(system, aiserver.ParseTemplate(cfg.Instructions, variables))
	}
	var input []aiserver.ChatMessage
	for _, m := range cfg.Messages {
		text := aiserver.ParseTemplate(aiserver.ContentText(m), variables)
		if m.Role == "system" {
			system = append(system, text)
			continue
		}
		input = append(input, aiserver.ChatMessage{Role: m.Role, Parts: []aiserver.MessagePart{aiserver.TextPart(text)}})
	}
	if finalUser != nil {
		input = append(input, aiserver.ChatMessage{Role: "user", Parts: []aiserver.MessagePart{aiserver.TextPart(*finalUser)}})
	}
	var tools []aiserver.Tool
	for _, k := range slices.Sorted(maps.Keys(cfg.Tools)) {
		tools = append(tools, cfg.Tools[k])
	}
	return aiserver.ChatContent{SystemInstructions: strings.Join(system, "\n\n"), InputMessages: input, ToolDefinitions: tools}
}

// startEvaluate opens `evaluate {criterionType}` under the row root, before
// the judge or scorer runs.
func startEvaluate(t *rowTrace, c Criterion, j *resolvedJudge) (context.Context, trace.Span) {
	attrs := append(slices.Clone(t.attrs), attribute.String("gen_ai.operation.name", "evaluate"))
	if j != nil {
		attrs = append(attrs, modelIdentity(j.config)...)
	}
	return tracer().Start(t.ctx, "evaluate "+c.criterionType(), trace.WithAttributes(attrs...))
}

// finishEvaluate records a COMPLETE result as a gen_ai.evaluation.result
// event mirrored onto the span, with no verdict, or an ERROR result as the
// span's status and error.type, then ends the span.
func finishEvaluate(span trace.Span, c *critResult) {
	if c.status != statusComplete {
		span.SetAttributes(attribute.String("error.type", c.cause))
		span.SetStatus(codes.Error, c.errMsg)
		span.End()
		return
	}
	attrs := []attribute.KeyValue{
		attribute.String("gen_ai.evaluation.name", c.criterion.criterionType()),
		attribute.Float64("gen_ai.evaluation.score.value", c.score),
	}
	if c.reason != "" && c.captureOn {
		attrs = append(attrs, attribute.String("gen_ai.evaluation.explanation", c.reason))
	}
	span.AddEvent("gen_ai.evaluation.result", trace.WithAttributes(attrs...))
	span.SetAttributes(attrs...)
	span.SetStatus(codes.Ok, "")
	span.End()
}

// spanAttrs is the row's span identity, or none for a row with no trace.
func (g *genResult) spanAttrs() []attribute.KeyValue {
	if g.trace == nil {
		return nil
	}
	return g.trace.attrs
}

// flushTelemetry ends any row root a failed phase left open, flushes the
// LaunchDarkly client, then force-flushes the tracer provider so a batch
// exporter sends the run's spans. It runs even when a phase failed.
func (r *runState) flushTelemetry(ctx context.Context, em *emitter) {
	for _, t := range r.rowTraces {
		if t != nil {
			t.end()
		}
	}
	em.flush()
	fctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), flushWait)
	defer cancel()
	if err := aiserver.FlushTelemetry(fctx); err != nil {
		r.m.logger.Warn("tracer provider flush failed; evaluation spans may not have been exported", "error", err)
	}
}
