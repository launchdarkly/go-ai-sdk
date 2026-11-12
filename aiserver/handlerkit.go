package aiserver

import (
	"context"
	"fmt"
	"strings"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"

	"github.com/launchdarkly/go-ai-sdk/internal/kitbase"
	"github.com/launchdarkly/go-ai-sdk/internal/telemetry"
)

// prompt is a handler's provider-neutral request: a system prompt plus the
// ordered user/assistant turns. Handlers map it to their
// provider's wire format.
type prompt struct {
	System string
	Turns  []Message
}

// kitComposePrompt applies the shared prompt-construction rules so every handler
// builds the same conversation:
//
//	[system from config] → [config conversation messages] → [history] → [userInput?]
//
// Messages mode prefers config.messages over instructions; agent mode prefers
// instructions. Without history, userInput is appended unless the
// last config turn is already a user turn. With history, a
// non-empty userInput is always appended and an empty one never is.
// An appended empty userInput is "" — never absent.
func kitComposePrompt(req HandlerRequest, mode Mode) prompt {
	cfg := req.Config
	var p prompt
	var systemParts []string
	for _, m := range cfg.Messages {
		if m.Blocks == nil {
			m.Text = ParseTemplate(m.Text, req.Variables)
		}
		if m.Role == "system" {
			systemParts = append(systemParts, m.Text)
			continue
		}
		p.Turns = append(p.Turns, m)
	}
	instructions := ParseTemplate(cfg.Instructions, req.Variables)
	switch {
	case mode == ModeAgent && cfg.Instructions != "":
		p.System = instructions
	case len(cfg.Messages) > 0:
		p.System = strings.Join(systemParts, "\n")
	default:
		p.System = instructions
	}

	if len(req.History) > 0 {
		for _, m := range req.History {
			if m.Role == "user" || m.Role == "assistant" {
				p.Turns = append(p.Turns, m)
			}
		}
		if req.UserInput != "" {
			p.Turns = append(p.Turns, Message{Role: "user", Text: req.UserInput})
		}
		return p
	}
	if len(p.Turns) == 0 || p.Turns[len(p.Turns)-1].Role != "user" {
		p.Turns = append(p.Turns, Message{Role: "user", Text: req.UserInput})
	}
	return p
}

// kitContentText flattens a message to text: Text, or the joined text blocks.
func kitContentText(m Message) string {
	if m.Blocks == nil {
		return m.Text
	}
	var parts []string
	for _, b := range m.Blocks {
		if b.Type == "text" {
			parts = append(parts, b.Text)
		}
	}
	return strings.Join(parts, "")
}

// kitImageURL returns a URL-sourced image's URL, or a data: URL for base64 data
// (the OpenAI / LangChain `image_url` form).
func kitImageURL(s ImageSource) string {
	if s.Type == "url" {
		return s.URL
	}
	return "data:" + s.MediaType + ";base64," + s.Data
}

// trackedSpan gives every span the kit opens a pointer identity. Span values
// from some tracers (notably the global no-op tracer used before a provider is
// registered) are structs with uncomparable fields, so using them as map keys
// or comparing them with == panics.
type trackedSpan struct{ trace.Span }

func trackSpan(s trace.Span) trace.Span { return &trackedSpan{s} }

// kitHandlerRun is one handler call's span tree: a root plus sibling child spans
// for each model turn and tool call.
type kitHandlerRun struct {
	opts    kitbase.HandlerSpanOptions
	ctx     context.Context
	Root    trace.Span
	tracer  trace.Tracer
	open    map[trace.Span]bool
	usage   kitbase.SpanUsage
	usageOK bool
	model   string      // last response model a chat turn reported
	judge   *judgeScope // non-nil when this run is a judge call
}

// kitStartHandlerRun opens the root span, writes model identity, the LD
// config-association and context-identity attributes (from req.Variables),
// and, when capture is enabled for ctx, the prompt content (OpenLLMetry
// attributes + gen_ai.content.prompt).
func kitStartHandlerRun(ctx context.Context, opts kitbase.HandlerSpanOptions, req HandlerRequest, p prompt) *kitHandlerRun {
	if opts.Operation == "" {
		opts.Operation = "invoke_agent"
	}
	if opts.ProviderName == "" {
		opts.ProviderName = opts.System
	}
	tracer := otel.Tracer(opts.Tracer)
	rctx, rawRoot := tracer.Start(ctx, opts.Name)
	root := trackSpan(rawRoot)
	r := &kitHandlerRun{opts: opts, ctx: rctx, Root: root, tracer: tracer, open: map[trace.Span]bool{root: true}, judge: judgeScopeFrom(ctx)}
	root.SetAttributes(attribute.String("gen_ai.operation.name", opts.Operation))
	root.SetAttributes(r.identity()...)
	setLDSpanAttributesFromVars(root, req.Variables)
	if telemetry.Enabled(ctx) {
		writePromptContent(root, p)
	}
	return r
}

func (r *kitHandlerRun) identity() []attribute.KeyValue {
	return []attribute.KeyValue{
		attribute.String("gen_ai.system", r.opts.System),
		attribute.String("gen_ai.provider.name", r.opts.ProviderName),
		attribute.String("gen_ai.request.model", r.opts.RequestModel),
	}
}

// Context is the root span's context; child spans and provider calls use it.
func (r *kitHandlerRun) Context() context.Context { return r.ctx }

func writePromptContent(span trace.Span, p prompt) {
	i := 0
	var lines []string
	add := func(role, text string) {
		span.SetAttributes(
			attribute.String(fmt.Sprintf("gen_ai.prompt.%d.role", i), role),
			attribute.String(fmt.Sprintf("gen_ai.prompt.%d.content", i), text))
		lines = append(lines, role+": "+text)
		i++
	}
	if p.System != "" {
		add("system", p.System)
	}
	for _, t := range p.Turns {
		add(t.Role, kitContentText(t))
	}
	span.AddEvent("gen_ai.content.prompt", trace.WithAttributes(attribute.String("gen_ai.prompt", strings.Join(lines, "\n"))))
}

// StartChat opens a `chat <model>` span for one provider turn.
func (r *kitHandlerRun) StartChat(model string) trace.Span {
	if model == "" {
		model = r.opts.RequestModel
	}
	_, raw := r.tracer.Start(r.ctx, "chat "+model)
	s := trackSpan(raw)
	s.SetAttributes(attribute.String("gen_ai.operation.name", "chat"))
	s.SetAttributes(r.identity()...)
	r.open[s] = true
	return s
}

// EndChat records one turn's usage on its chat span (always, zeros included)
// and accumulates it for the root unless u is Unreported.
func (r *kitHandlerRun) EndChat(s trace.Span, responseModel string, u kitbase.SpanUsage, finishReason string) {
	if responseModel == "" {
		responseModel = r.opts.RequestModel
	}
	s.SetAttributes(attribute.String("gen_ai.response.model", responseModel))
	r.model = responseModel
	kitbase.SetUsageSpanAttributes(s, u)
	if finishReason != "" {
		s.SetAttributes(attribute.StringSlice("gen_ai.response.finish_reasons", []string{finishReason}))
	}
	r.addUsage(u)
	r.end(s, nil)
}

// StartTool opens an `execute_tool <name>` span, a sibling of the chat spans.
func (r *kitHandlerRun) StartTool(name, callID string) trace.Span {
	_, raw := r.tracer.Start(r.ctx, "execute_tool "+name)
	s := trackSpan(raw)
	s.SetAttributes(attribute.String("gen_ai.operation.name", "execute_tool"),
		attribute.String("gen_ai.tool.name", name), attribute.String("gen_ai.tool.call.id", callID))
	r.open[s] = true
	return s
}

// EndTool ends a tool span; a non-nil err marks it ERROR.
func (r *kitHandlerRun) EndTool(s trace.Span, err error) { r.end(s, err) }

// ToolArguments records a tool call's arguments on its execute_tool span,
// subject to the content gate.
func (r *kitHandlerRun) ToolArguments(s trace.Span, arguments any) {
	telemetry.SetToolCallArguments(r.ctx, s, arguments)
}

// ToolResult records a tool call's result on its execute_tool span, subject
// to the content gate.
func (r *kitHandlerRun) ToolResult(s trace.Span, result any) {
	telemetry.SetToolCallResult(r.ctx, s, result)
}

func (r *kitHandlerRun) end(s trace.Span, err error) {
	if !r.open[s] {
		return
	}
	delete(r.open, s)
	if err != nil {
		s.RecordError(err)
		s.SetStatus(codes.Error, err.Error())
	} else {
		s.SetStatus(codes.Ok, "")
	}
	if s == r.Root && r.judge != nil && r.judge.hold(s) {
		return // the judge scope ends it after scoring, at this moment's timestamp
	}
	s.End()
}

// AddUsage accumulates usage that has no chat span of its own.
func (r *kitHandlerRun) AddUsage(u kitbase.SpanUsage) { r.addUsage(u) }

func (r *kitHandlerRun) addUsage(u kitbase.SpanUsage) {
	r.usage.Add(u)
	if !u.Unreported {
		r.usageOK = true
	}
}

// Finish ends the root on success: response model, run usage once reported, completion
// content (gen_ai.completion.0.* + gen_ai.content.completion) when
// capture is enabled, status OK.
func (r *kitHandlerRun) Finish(responseModel, output string) {
	if responseModel == "" {
		responseModel = r.opts.RequestModel
	}
	r.Root.SetAttributes(attribute.String("gen_ai.response.model", responseModel))
	if r.usageOK {
		kitbase.SetUsageSpanAttributes(r.Root, r.usage)
	}
	if telemetry.Enabled(r.ctx) {
		r.Root.SetAttributes(attribute.String("gen_ai.completion.0.role", "assistant"),
			attribute.String("gen_ai.completion.0.content", output))
		r.Root.AddEvent("gen_ai.content.completion", trace.WithAttributes(attribute.String("gen_ai.completion", output)))
	}
	r.end(r.Root, nil)
}

// responseModel is the last model a turn reported, else the request model.
func (r *kitHandlerRun) responseModel() string {
	if r.model != "" {
		return r.model
	}
	return r.opts.RequestModel
}

// Fail records err on every open span and the root, writing the response
// model and any usage already reported so partial spend survives.
func (r *kitHandlerRun) Fail(err error) {
	for s := range r.open {
		if s != r.Root {
			r.end(s, err)
		}
	}
	if r.open[r.Root] {
		r.Root.SetAttributes(attribute.String("gen_ai.response.model", r.responseModel()))
		if r.usageOK {
			kitbase.SetUsageSpanAttributes(r.Root, r.usage)
		}
	}
	r.end(r.Root, err)
}

// EndOnPanic, deferred right after kitStartHandlerRun, ends every open span
// ERROR when the handler, a provider call or a tool panics, then re-panics.
// Spans already ended are untouched.
func (r *kitHandlerRun) EndOnPanic() {
	if p := recover(); p != nil {
		r.Fail(fmt.Errorf("panic: %v", p))
		panic(p)
	}
}

// Abandon ends every open span with launchdarkly.stream.abandoned=true and
// UNSET status: an early consumer exit is neither success nor error.
func (r *kitHandlerRun) Abandon() {
	if r.open[r.Root] {
		r.Root.SetAttributes(attribute.String("gen_ai.response.model", r.responseModel()))
		if r.usageOK {
			kitbase.SetUsageSpanAttributes(r.Root, r.usage)
		}
	}
	for s := range r.open {
		s.SetAttributes(attribute.Bool("launchdarkly.stream.abandoned", true))
		s.End()
	}
	clear(r.open)
}

// ToolResultString renders a tool result for the provider: strings pass
// through, anything else is JSON.
func ToolResultString(v any) string { return telemetry.ValueString(v) }

// kitCallableTools returns the config tools that have a local ToolFunc,
// keyed by the config map key — the name the model sees.
func kitCallableTools(cfg AiConfig, handlers map[string]ToolHandler) map[string]Tool {
	out := map[string]Tool{}
	for key, t := range cfg.Tools {
		if _, ok := handlers[key].(ToolFunc); ok {
			out[key] = t
		}
	}
	return out
}
