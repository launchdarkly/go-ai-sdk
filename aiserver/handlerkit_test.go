package aiserver

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace/noop"
)

func roleTexts(p Prompt) []string {
	var out []string
	for _, t := range p.Turns {
		out = append(out, t.Role+":"+ContentText(t))
	}
	return out
}

func TestComposePrompt(t *testing.T) {
	img := Message{Role: "user", Blocks: []ContentBlock{{Type: "image", Source: &ImageSource{Type: "url", URL: "u"}}}}
	cases := []struct {
		name       string
		cfg        AiConfig
		mode       Mode
		input      string
		history    []Message
		wantSystem string
		wantTurns  []string
	}{
		{"instructions, input appended", AiConfig{Instructions: "Hi {{n}} {{missing}}"}, ModeMessages, "q", nil, "Hi Al {{missing}}", []string{"user:q"}},
		{"instructions, empty input still appended", AiConfig{Instructions: "s"}, ModeMessages, "", nil, "s", []string{"user:"}},
		{"messages ending user: not appended", AiConfig{Messages: []Message{{Role: "system", Text: "sys {{n}}"}, {Role: "user", Text: "u {{n}}"}}}, ModeMessages, "q", nil, "sys Al", []string{"user:u Al"}},
		{"messages ending assistant: appended", AiConfig{Messages: []Message{{Role: "assistant", Text: "a"}}}, ModeMessages, "q", nil, "", []string{"assistant:a", "user:q"}},
		{"both present, messages mode: messages win", AiConfig{Instructions: "I", Messages: []Message{{Role: "system", Text: "M"}, {Role: "assistant", Text: "a"}}}, ModeMessages, "q", nil, "M", []string{"assistant:a", "user:q"}},
		{"both present, agent mode: instructions win", AiConfig{Instructions: "I", Messages: []Message{{Role: "system", Text: "M"}}}, ModeAgent, "q", nil, "I", []string{"user:q"}},
		{"history between config and input, system filtered", AiConfig{Messages: []Message{{Role: "assistant", Text: "a"}}}, ModeMessages, "q", []Message{{Role: "system", Text: "x"}, {Role: "user", Text: "h"}}, "", []string{"assistant:a", "user:h", "user:q"}},
		{"history ending user, empty input: not appended", AiConfig{Instructions: "s"}, ModeMessages, "", []Message{img}, "s", []string{"user:"}},
		{"image-only history + question: both", AiConfig{Instructions: "s"}, ModeAgent, "what colour?", []Message{img}, "s", []string{"user:", "user:what colour?"}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			p := ComposePrompt(HandlerRequest{Config: c.cfg, UserInput: c.input, History: c.history, Variables: map[string]any{"n": "Al"}}, c.mode)
			if p.System != c.wantSystem || !reflect.DeepEqual(roleTexts(p), c.wantTurns) {
				t.Errorf("got system=%q turns=%v", p.System, roleTexts(p))
			}
		})
	}
	// Empty history behaves exactly like no history.
	cfg := AiConfig{Messages: []Message{{Role: "user", Text: "u"}}}
	if a, b := ComposePrompt(HandlerRequest{Config: cfg, UserInput: "q"}, ModeMessages), ComposePrompt(HandlerRequest{Config: cfg, UserInput: "q", History: []Message{}}, ModeMessages); !reflect.DeepEqual(a, b) {
		t.Errorf("empty history changed prompt: %v vs %v", a, b)
	}
	// Multimodal history keeps its blocks.
	p := ComposePrompt(HandlerRequest{Config: AiConfig{Instructions: "s"}, History: []Message{img}, UserInput: "q"}, ModeMessages)
	if p.Turns[0].Blocks == nil || p.Turns[0].Blocks[0].Source.URL != "u" {
		t.Error("image block lost")
	}
}

func TestWithOutputFormat(t *testing.T) {
	if got := WithOutputFormat("base", nil); got != "base" {
		t.Errorf("no schema changed system: %q", got)
	}
	if got := WithOutputFormat("base", map[string]any{"type": "object"}); got != "base\n\nRespond with valid JSON matching this schema:\n{\"type\":\"object\"}" {
		t.Errorf("got %q", got)
	}
}

// TestHandlerRunWithoutTracerProvider: with no tracer provider installed, OTel
// hands out no-op span values that cannot be map keys or compared with ==.
// A handler must still run, end its spans and fail cleanly.
func TestHandlerRunWithoutTracerProvider(t *testing.T) {
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(noop.NewTracerProvider())
	t.Cleanup(func() { otel.SetTracerProvider(prev) })
	req := HandlerRequest{Config: AiConfig{Model: Model{Name: "m"}}}
	for _, fail := range []bool{false, true} {
		r := StartHandlerRun(context.Background(), HandlerSpanOptions{Tracer: TracerPrefix + "/t", Name: "x", System: "s"}, req, Prompt{})
		chat := r.StartChat("m")
		r.EndChat(chat, "m", SpanUsage{Input: 1}, "stop")
		tool := r.StartTool("t", "id")
		if fail {
			r.Fail(errors.New("boom"))
			continue
		}
		r.EndTool(tool, nil)
		r.Finish("m", "out")
	}
	r := StartHandlerRun(context.Background(), HandlerSpanOptions{Tracer: TracerPrefix + "/t", Name: "x"}, req, Prompt{})
	r.StartChat("m")
	r.Abandon()
}

// TestHandlerRunContentEvents pins the content carriers: the prompt event
// holds the system prompt and user input, the completion event the output,
// and the OpenLLMetry indexed attributes mirror both.
func TestHandlerRunContentEvents(t *testing.T) {
	exp := tracetest.NewInMemoryExporter()
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(sdktrace.NewTracerProvider(sdktrace.WithSyncer(exp)))
	t.Cleanup(func() { otel.SetTracerProvider(prev) })

	p := Prompt{System: "Be terse", Turns: []Message{{Role: "user", Text: "what is a flag?"}}}
	r := StartHandlerRun(context.Background(), HandlerSpanOptions{Tracer: TracerPrefix + "/t", Name: "x", System: "s", RequestModel: "m"},
		HandlerRequest{}, p)
	r.Finish("m", "a toggle")

	span := exp.GetSpans().Snapshots()[0]
	events := map[string]string{}
	for _, e := range span.Events() {
		for _, a := range e.Attributes {
			events[e.Name+"/"+string(a.Key)] = a.Value.AsString()
		}
	}
	if got := events["gen_ai.content.prompt/gen_ai.prompt"]; got != "system: Be terse\nuser: what is a flag?" {
		t.Errorf("prompt event = %q", got)
	}
	if got := events["gen_ai.content.completion/gen_ai.completion"]; got != "a toggle" {
		t.Errorf("completion event = %q", got)
	}
	attrs := map[string]string{}
	for _, a := range span.Attributes() {
		attrs[string(a.Key)] = a.Value.Emit()
	}
	for k, want := range map[string]string{
		"gen_ai.prompt.0.role": "system", "gen_ai.prompt.0.content": "Be terse",
		"gen_ai.prompt.1.role": "user", "gen_ai.prompt.1.content": "what is a flag?",
		"gen_ai.completion.0.role": "assistant", "gen_ai.completion.0.content": "a toggle",
	} {
		if attrs[k] != want {
			t.Errorf("%s = %q, want %q", k, attrs[k], want)
		}
	}
}

// TestHandlerRunSpanTree pins the kit's span tree: chat and tool spans
// are siblings under the root, every span ends with the right status on
// success, failure and abandonment, and the root carries run usage.
func TestHandlerRunSpanTree(t *testing.T) {
	exp := withSpans(t)
	opts := HandlerSpanOptions{Tracer: TracerPrefix + "/t", Name: "claude.messages", System: "anthropic", RequestModel: "m"}
	req := HandlerRequest{Variables: map[string]any{VarLD: TrackData{ConfigKey: "cfg", VariationKey: "v", RunID: "r"}}}
	byName := func() map[string]tracetest.SpanStub {
		out := map[string]tracetest.SpanStub{}
		for _, s := range exp.GetSpans() {
			out[s.Name] = s
		}
		return out
	}
	attr := func(s tracetest.SpanStub, k string) string {
		for _, a := range s.Attributes {
			if string(a.Key) == k {
				return a.Value.Emit()
			}
		}
		return ""
	}

	r := StartHandlerRun(bg, opts, req, Prompt{})
	chat := r.StartChat("")
	r.EndChat(chat, "m-resp", SpanUsage{Input: 3, Output: 2}, "tool_use")
	tool := r.StartTool("lookup", "call-1")
	r.EndTool(tool, nil)
	r.EndChat(r.StartChat("m"), "", SpanUsage{Input: 1, Output: 1}, "")
	r.Finish("m-resp", "done")
	s := byName()
	root := s["claude.messages"]
	if len(exp.GetSpans()) != 4 {
		t.Fatalf("spans = %d, want root + 2 chat + 1 tool", len(exp.GetSpans()))
	}
	for _, c := range exp.GetSpans() {
		if c.Name == root.Name {
			continue
		}
		if c.Parent.SpanID() != root.SpanContext.SpanID() || c.Status.Code != codes.Ok || c.EndTime.IsZero() {
			t.Errorf("%s: parent=%v status=%v", c.Name, c.Parent.SpanID(), c.Status.Code)
		}
		if attr(c, "launchdarkly.config.key") != "" {
			t.Errorf("%s carries LD config attributes; only the root may", c.Name)
		}
	}
	for k, want := range map[string]string{
		"gen_ai.system": "anthropic", "gen_ai.provider.name": "anthropic", "gen_ai.operation.name": "chat",
		"gen_ai.request.model": "m", "gen_ai.response.model": "m-resp",
		"gen_ai.usage.input_tokens": "4", "gen_ai.usage.output_tokens": "3", "gen_ai.usage.total_tokens": "7",
		"gen_ai.usage.prompt_tokens": "4", "gen_ai.usage.completion_tokens": "3",
		"launchdarkly.operation.type": "gen_ai", "launchdarkly.config.key": "cfg",
		"launchdarkly.variation.key": "v", "launchdarkly.run.id": "r",
	} {
		if got := attr(root, k); got != want {
			t.Errorf("root %s = %q, want %q", k, got, want)
		}
	}
	if root.Status.Code != codes.Ok {
		t.Errorf("root status = %v", root.Status.Code)
	}
	if c := s["execute_tool lookup"]; attr(c, "gen_ai.tool.name") != "lookup" || attr(c, "gen_ai.tool.call.id") != "call-1" || attr(c, "gen_ai.operation.name") != "execute_tool" {
		t.Errorf("tool span attributes = %v", c.Attributes)
	}
	if c := s["chat m"]; attr(c, "gen_ai.usage.input_tokens") == "" || attr(c, "gen_ai.operation.name") != "chat" {
		t.Errorf("chat span attributes = %v", c.Attributes)
	}

	// Failure: every open span and the root end ERROR; usage reported so far survives.
	exp.Reset()
	r = StartHandlerRun(bg, opts, req, Prompt{})
	r.AddUsage(SpanUsage{Input: 5})
	r.StartChat("m")
	r.StartTool("lookup", "call-2")
	r.Fail(errors.New("provider down"))
	for _, c := range exp.GetSpans() {
		if c.Status.Code != codes.Error || c.EndTime.IsZero() {
			t.Errorf("failed run: %s status=%v ended=%v", c.Name, c.Status.Code, !c.EndTime.IsZero())
		}
	}
	if got := attr(byName()["claude.messages"], "gen_ai.usage.input_tokens"); got != "5" || len(exp.GetSpans()) != 3 {
		t.Errorf("failed root input tokens = %q, spans = %d", got, len(exp.GetSpans()))
	}

	// Abandonment: every open span ends UNSET with the abandoned flag.
	exp.Reset()
	r = StartHandlerRun(bg, opts, req, Prompt{})
	r.StartChat("m")
	r.Abandon()
	r.Fail(errors.New("late")) // ends nothing twice
	if len(exp.GetSpans()) != 2 {
		t.Fatalf("abandoned spans = %d", len(exp.GetSpans()))
	}
	for _, c := range exp.GetSpans() {
		if c.Status.Code != codes.Unset || attr(c, "launchdarkly.stream.abandoned") != "true" {
			t.Errorf("abandoned: %s status=%v", c.Name, c.Status.Code)
		}
	}
}

// The root reports a response model on failure and abandonment too: the last
// model a turn reported, else the request model.
func TestHandlerRunResponseModelOnFailAndAbandon(t *testing.T) {
	exp := withSpans(t)
	opts := HandlerSpanOptions{Tracer: TracerPrefix + "/t", Name: "root", System: "s", RequestModel: "req-model"}
	rootModel := func() string {
		for _, s := range exp.GetSpans() {
			if s.Name == "root" {
				for _, a := range s.Attributes {
					if a.Key == "gen_ai.response.model" {
						return a.Value.AsString()
					}
				}
			}
		}
		return ""
	}
	for name, end := range map[string]func(*HandlerRun){
		"fail":    func(r *HandlerRun) { r.Fail(errors.New("x")) },
		"abandon": func(r *HandlerRun) { r.Abandon() },
	} {
		exp.Reset()
		r := StartHandlerRun(bg, opts, HandlerRequest{}, Prompt{})
		end(r)
		if got := rootModel(); got != "req-model" {
			t.Errorf("%s without a turn: response model = %q", name, got)
		}
		exp.Reset()
		r = StartHandlerRun(bg, opts, HandlerRequest{}, Prompt{})
		r.EndChat(r.StartChat(""), "resp-model", SpanUsage{}, "")
		r.StartChat("")
		end(r)
		if got := rootModel(); got != "resp-model" {
			t.Errorf("%s after a turn: response model = %q", name, got)
		}
	}
}

// EndOnPanic ends every open span ERROR and lets the panic continue.
func TestHandlerRunEndOnPanic(t *testing.T) {
	exp := withSpans(t)
	opts := HandlerSpanOptions{Tracer: TracerPrefix + "/t", Name: "root", System: "s", RequestModel: "m"}
	var got any
	func() {
		defer func() { got = recover() }()
		r := StartHandlerRun(bg, opts, HandlerRequest{}, Prompt{})
		defer r.EndOnPanic()
		r.StartChat("m")
		r.StartTool("lookup", "c1")
		panic("tool bug")
	}()
	if got != "tool bug" {
		t.Fatalf("panic value = %v", got)
	}
	if n := len(exp.GetSpans()); n != 3 {
		t.Fatalf("ended spans = %d, want 3", n)
	}
	for _, s := range exp.GetSpans() {
		if s.Status.Code != codes.Error {
			t.Errorf("%s status = %v", s.Name, s.Status.Code)
		}
	}
	// No panic: a finished run is left as it was.
	exp.Reset()
	func() {
		r := StartHandlerRun(bg, opts, HandlerRequest{}, Prompt{})
		defer r.EndOnPanic()
		r.Finish("m", "ok")
	}()
	if s := exp.GetSpans(); len(s) != 1 || s[0].Status.Code != codes.Ok {
		t.Errorf("finished run spans = %v", s)
	}
}
