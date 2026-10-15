package aiserver

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

// Prompt is a handler's provider-neutral request: a system prompt plus the
// ordered user/assistant turns. Handlers map it to their
// provider's wire format.
type Prompt struct {
	System string
	Turns  []Message
}

// ComposePrompt applies the shared prompt-construction rules so every handler
// builds the same conversation:
//
//	[system from config] → [config conversation messages] → [history] → [userInput?]
//
// Messages mode prefers config.messages over instructions; agent mode prefers
// instructions. Without history, userInput is appended unless the
// last config turn is already a user turn. With history, a
// non-empty userInput is always appended and an empty one never is.
// An appended empty userInput is "" — never absent.
func ComposePrompt(req HandlerRequest, mode Mode) Prompt {
	cfg := req.Config
	var p Prompt
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

// ContentText flattens a message to text: Text, or the joined text blocks.
func ContentText(m Message) string {
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

// ImageURL returns a URL-sourced image's URL, or a data: URL for base64 data
// (the OpenAI / LangChain `image_url` form).
func ImageURL(s ImageSource) string {
	if s.Type == "url" {
		return s.URL
	}
	return "data:" + s.MediaType + ";base64," + s.Data
}

// SpanUsage is the token counts a span reports. Input is cache-inclusive: the
// caller folds provider cache buckets in before calling SetUsageSpanAttributes.
type SpanUsage struct {
	Input, Output, CacheRead, CacheCreation float64
}

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

// trackedSpan gives every span the kit opens a pointer identity. Span values
// from some tracers (notably the global no-op tracer used before a provider is
// registered) are structs with uncomparable fields, so using them as map keys
// or comparing them with == panics.
type trackedSpan struct{ trace.Span }

func trackSpan(s trace.Span) trace.Span { return &trackedSpan{s} }

// HandlerSpanOptions describe a handler's root span.
type HandlerSpanOptions struct {
	Tracer       string // instrumentation scope; must start with TracerPrefix
	Name         string // documented span name, e.g. "claude.messages"
	System       string // gen_ai.system, e.g. "anthropic"
	Operation    string // gen_ai.operation.name; "" means "chat"
	RequestModel string
	ProviderName string // gen_ai.provider.name; "" means System
}

// HandlerRun is one handler call's span tree: a root plus sibling child spans
// for each model turn and tool call.
type HandlerRun struct {
	opts    HandlerSpanOptions
	ctx     context.Context
	Root    trace.Span
	tracer  trace.Tracer
	open    map[trace.Span]bool
	usage   SpanUsage
	usageOK bool
	model   string      // last response model a chat turn reported
	judge   *judgeScope // non-nil when this run is a judge call
}

// StartHandlerRun opens the root span, writes model identity, the LD
// config-association and context-identity attributes (from req.Variables),
// and the prompt content (OpenLLMetry attributes + gen_ai.content.prompt).
func StartHandlerRun(ctx context.Context, opts HandlerSpanOptions, req HandlerRequest, p Prompt) *HandlerRun {
	if opts.Operation == "" {
		opts.Operation = "chat"
	}
	if opts.ProviderName == "" {
		opts.ProviderName = opts.System
	}
	tracer := otel.Tracer(opts.Tracer)
	rctx, rawRoot := tracer.Start(ctx, opts.Name)
	root := trackSpan(rawRoot)
	r := &HandlerRun{opts: opts, ctx: rctx, Root: root, tracer: tracer, open: map[trace.Span]bool{root: true}, judge: judgeScopeFrom(ctx)}
	root.SetAttributes(attribute.String("gen_ai.operation.name", opts.Operation))
	root.SetAttributes(r.identity()...)
	SetLDSpanAttributes(root, req.Variables)
	writePromptContent(root, p)
	return r
}

func (r *HandlerRun) identity() []attribute.KeyValue {
	return []attribute.KeyValue{
		attribute.String("gen_ai.system", r.opts.System),
		attribute.String("gen_ai.provider.name", r.opts.ProviderName),
		attribute.String("gen_ai.request.model", r.opts.RequestModel),
	}
}

// Context is the root span's context; child spans and provider calls use it.
func (r *HandlerRun) Context() context.Context { return r.ctx }

func writePromptContent(span trace.Span, p Prompt) {
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
		add(t.Role, ContentText(t))
	}
	span.AddEvent("gen_ai.content.prompt", trace.WithAttributes(attribute.String("gen_ai.prompt", strings.Join(lines, "\n"))))
}

// StartChat opens a `chat <model>` span for one provider turn.
func (r *HandlerRun) StartChat(model string) trace.Span {
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
// and accumulates it for the root.
func (r *HandlerRun) EndChat(s trace.Span, responseModel string, u SpanUsage, finishReason string) {
	if responseModel == "" {
		responseModel = r.opts.RequestModel
	}
	s.SetAttributes(attribute.String("gen_ai.response.model", responseModel))
	r.model = responseModel
	SetUsageSpanAttributes(s, u)
	if finishReason != "" {
		s.SetAttributes(attribute.StringSlice("gen_ai.response.finish_reasons", []string{finishReason}))
	}
	r.usage.Add(u)
	r.usageOK = true
	r.end(s, nil)
}

// StartTool opens an `execute_tool <name>` span, a sibling of the chat spans.
func (r *HandlerRun) StartTool(name, callID string) trace.Span {
	_, raw := r.tracer.Start(r.ctx, "execute_tool "+name)
	s := trackSpan(raw)
	s.SetAttributes(attribute.String("gen_ai.operation.name", "execute_tool"),
		attribute.String("gen_ai.tool.name", name), attribute.String("gen_ai.tool.call.id", callID))
	r.open[s] = true
	return s
}

// EndTool ends a tool span; a non-nil err marks it ERROR.
func (r *HandlerRun) EndTool(s trace.Span, err error) { r.end(s, err) }

func (r *HandlerRun) end(s trace.Span, err error) {
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
func (r *HandlerRun) AddUsage(u SpanUsage) {
	r.usage.Add(u)
	r.usageOK = true
}

// Finish ends the root on success: response model, run usage, completion
// content (gen_ai.completion.0.* + gen_ai.content.completion), status OK.
func (r *HandlerRun) Finish(responseModel, output string) {
	if responseModel == "" {
		responseModel = r.opts.RequestModel
	}
	r.Root.SetAttributes(attribute.String("gen_ai.response.model", responseModel))
	SetUsageSpanAttributes(r.Root, r.usage)
	r.Root.SetAttributes(attribute.String("gen_ai.completion.0.role", "assistant"),
		attribute.String("gen_ai.completion.0.content", output))
	r.Root.AddEvent("gen_ai.content.completion", trace.WithAttributes(attribute.String("gen_ai.completion", output)))
	r.end(r.Root, nil)
}

// responseModel is the last model a turn reported, else the request model.
func (r *HandlerRun) responseModel() string {
	if r.model != "" {
		return r.model
	}
	return r.opts.RequestModel
}

// Fail records err on every open span and the root, writing the response
// model and any usage already reported so partial spend survives.
func (r *HandlerRun) Fail(err error) {
	for s := range r.open {
		if s != r.Root {
			r.end(s, err)
		}
	}
	if r.open[r.Root] {
		r.Root.SetAttributes(attribute.String("gen_ai.response.model", r.responseModel()))
		if r.usageOK {
			SetUsageSpanAttributes(r.Root, r.usage)
		}
	}
	r.end(r.Root, err)
}

// EndOnPanic, deferred right after StartHandlerRun, ends every open span
// ERROR when the handler, a provider call or a tool panics, then re-panics.
// Spans already ended are untouched.
func (r *HandlerRun) EndOnPanic() {
	if p := recover(); p != nil {
		r.Fail(fmt.Errorf("panic: %v", p))
		panic(p)
	}
}

// Abandon ends every open span with launchdarkly.stream.abandoned=true and
// UNSET status: an early consumer exit is neither success nor error.
func (r *HandlerRun) Abandon() {
	if r.open[r.Root] {
		r.Root.SetAttributes(attribute.String("gen_ai.response.model", r.responseModel()))
		if r.usageOK {
			SetUsageSpanAttributes(r.Root, r.usage)
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
func ToolResultString(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	if v == nil {
		return ""
	}
	b, err := json.Marshal(v)
	if err != nil {
		return fmt.Sprint(v)
	}
	return string(b)
}

// CallableTools returns the config tools that have a local ToolFunc,
// keyed by the config map key — the name the model sees.
func CallableTools(cfg AiConfig, handlers map[string]ToolHandler) map[string]Tool {
	out := map[string]Tool{}
	for key, t := range cfg.Tools {
		if _, ok := handlers[key].(ToolFunc); ok {
			out[key] = t
		}
	}
	return out
}
