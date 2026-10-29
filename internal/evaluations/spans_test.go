package evaluations

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/sdkapi"
)

// flushLogger is the process tracer provider for a test: the SDK span
// processor ahead of an in-memory exporter, as InitClient installs it, with
// ForceFlush logged in the env's ordered log.
type flushLogger struct {
	*sdktrace.TracerProvider
	env *fakeEnv
}

func (f flushLogger) ForceFlush(ctx context.Context) error {
	f.env.mu.Lock()
	f.env.log = append(f.env.log, "tracer flush")
	f.env.mu.Unlock()
	return f.TracerProvider.ForceFlush(ctx)
}

func evalSpans(t *testing.T, e *fakeEnv) (*tracetest.InMemoryExporter, flushLogger) {
	t.Helper()
	// Shutdown restores the built-in capture default and clears the
	// Configure lock other tests' handler calls set.
	_ = aiserver.Shutdown(context.Background())
	t.Cleanup(func() { _ = aiserver.Shutdown(context.Background()) })
	exp := tracetest.NewInMemoryExporter()
	tp := flushLogger{sdktrace.NewTracerProvider(
		sdktrace.WithResource(resource.NewSchemaless(attribute.String("launchdarkly.project_id", "sdk-key"))),
		sdktrace.WithSpanProcessor(aiserver.ConversationIDSpanProcessor{}),
		sdktrace.WithSyncer(exp)), e}
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(tp)
	t.Cleanup(func() { otel.SetTracerProvider(prev) })
	return exp, tp
}

type spanSet []sdktrace.ReadOnlySpan

func (s spanSet) named(name string) []sdktrace.ReadOnlySpan {
	var out []sdktrace.ReadOnlySpan
	for _, sp := range s {
		if sp.Name() == name {
			out = append(out, sp)
		}
	}
	return out
}

func (s spanSet) one(t *testing.T, name string) sdktrace.ReadOnlySpan {
	t.Helper()
	got := s.named(name)
	if len(got) != 1 {
		t.Fatalf("want one %q span, got %d of %v", name, len(got), s.names())
	}
	return got[0]
}

func (s spanSet) names() []string {
	var out []string
	for _, sp := range s {
		out = append(out, sp.Name())
	}
	return out
}

func spanAttrs(sp sdktrace.ReadOnlySpan) map[string]attribute.Value {
	m := map[string]attribute.Value{}
	for _, a := range sp.Attributes() {
		m[string(a.Key)] = a.Value
	}
	return m
}

func jsonAttr(t *testing.T, sp sdktrace.ReadOnlySpan, key string) []map[string]any {
	t.Helper()
	v, ok := spanAttrs(sp)[key]
	if !ok {
		t.Fatalf("%s: no %s", sp.Name(), key)
	}
	var out []map[string]any
	if err := json.Unmarshal([]byte(v.AsString()), &out); err != nil {
		t.Fatalf("%s: %s is not a JSON array: %v", sp.Name(), key, err)
	}
	return out
}

func childOf(child, parent sdktrace.ReadOnlySpan) bool {
	return child.Parent().SpanID() == parent.SpanContext().SpanID()
}

// within reports that inner's window lies inside outer's.
func within(inner, outer sdktrace.ReadOnlySpan) bool {
	return !inner.StartTime().Before(outer.StartTime()) && !inner.EndTime().After(outer.EndTime())
}

// spanRun routes one hosted row with variables and returns the env and
// args for a plain-function generation that calls tool "lookup", plus a
// judge whose user message expands {{message_history}}.
func spanRun(t *testing.T, gen func(ctx context.Context, req aiserver.HandlerRequest) (aiserver.HandlerResult, error), judgeCalls *int) (*fakeEnv, RunArgs) {
	t.Helper()
	e := newEnv(t, 1)
	e.route("GET", "projects/proj/datasets/ds/rows", resp{body: map[string]any{"items": []any{
		map[string]any{"rowIndex": 0, "input": "What is 2+2?", "variables": map[string]any{"tone": "brief"}},
	}, "totalCount": 1}})
	e.judges["acc"] = judgeVariation("OpenAI", "completion", map[string]any{"messages": []any{
		map[string]any{"role": "system", "content": "You grade answers."},
		map[string]any{"role": "user", "content": "Transcript:\n{{message_history}}"},
	}})
	h := handler("OpenAI", aiserver.ModeMessages, func(ctx context.Context, req aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
		if _, isJudge := req.Variables["message_history"]; isJudge {
			*judgeCalls++
			return verdict(0.9, "correct")(ctx, req)
		}
		return gen(ctx, req)
	})
	args := baseArgs(h)
	args.Concurrency = 1
	args.Generation = &GenerationConfig{Provider: "OpenAI", Model: "gpt-4o", Mode: ModeCompletion,
		Messages: []aiserver.Message{{Role: "system", Text: "Answer in a {{tone}} way."}}}
	args.InlineTools = map[string]InlineTool{"lookup": {Schema: map[string]any{"type": "object"},
		Implementation: aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) { return "4", nil })}}
	args.Criteria = []Criterion{mustJudge(t, "acc", nil)}
	return e, args
}

func callLookup(ctx context.Context, req aiserver.HandlerRequest) {
	if fn, ok := req.ToolHandlers["lookup"].(aiserver.ToolFunc); ok {
		_, _ = fn(ctx, map[string]any{"q": "2+2"})
	}
}

func plainGeneration(ctx context.Context, req aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
	callLookup(ctx, req)
	return aiserver.HandlerResult{Output: "The answer is 4.", Usage: map[string]any{"inputTokens": 7.0, "outputTokens": 3.0}}, nil
}

func TestRowConversationIDVectors(t *testing.T) {
	if got := rowConversationID("run-id", 3); got != "faa03308-8fe7-5805-8c07-f82a3dbdb3f1" {
		t.Errorf("row 3 = %s", got)
	}
	if got := rowConversationID("run-id", 0); got != "8b90044e-ac6d-5af1-88a3-b75fbb9c8268" {
		t.Errorf("row 0 = %s", got)
	}
}

func TestEvaluationSpansForPlainFunctions(t *testing.T) {
	judgeCalls := 0
	e, args := spanRun(t, plainGeneration, &judgeCalls)
	exp, tp := evalSpans(t, e)
	mustRun(t, e.module(), args)
	if otel.GetTracerProvider() != tp {
		t.Fatal("run() must not install a tracer provider")
	}
	spans := spanSet(exp.GetSpans().Snapshots())
	if len(spans) != 5 {
		t.Fatalf("spans %v", spans.names())
	}
	root := spans.one(t, rowSpanName)
	genChat := spans.one(t, "chat gpt-4o")
	tool := spans.one(t, "execute_tool lookup")
	eval := spans.one(t, "evaluate acc")
	judgeChat := spans.one(t, "chat judge-model")

	// One trace, every span OK and carrying the row identity, timed by
	// containment rather than a duration floor.
	conv := rowConversationID("run-1", 0)
	for _, sp := range spans {
		a := spanAttrs(sp)
		if sp.SpanContext().TraceID() != root.SpanContext().TraceID() {
			t.Errorf("%s is in another trace", sp.Name())
		}
		if sp.Status().Code != codes.Ok {
			t.Errorf("%s status %v", sp.Name(), sp.Status())
		}
		if a["gen_ai.conversation.id"].AsString() != conv || a[attrRunID].AsString() != "run-1" ||
			a[attrRowIndex].Type() != attribute.INT64 || a[attrRowIndex].AsInt64() != 0 {
			t.Errorf("%s row identity %v", sp.Name(), a)
		}
		if _, has := a["launchdarkly.run.id"]; has {
			t.Errorf("%s has launchdarkly.run.id", sp.Name())
		}
		if !sp.EndTime().After(sp.StartTime()) {
			t.Errorf("%s is zero-length", sp.Name())
		}
		if r, _ := sp.Resource().Set().Value("launchdarkly.project_id"); r.AsString() != "sdk-key" {
			t.Errorf("%s resource project id %v", sp.Name(), r)
		}
	}
	for _, c := range []struct{ child, parent sdktrace.ReadOnlySpan }{
		{genChat, root}, {tool, root}, {eval, root}, {judgeChat, eval},
	} {
		if !childOf(c.child, c.parent) || !within(c.child, c.parent) {
			t.Errorf("%s must be a child inside %s", c.child.Name(), c.parent.Name())
		}
	}
	if !within(tool, genChat) {
		t.Error("the tool call must lie inside the generation chat")
	}
	if _, has := spanAttrs(root)["gen_ai.operation.name"]; has || strings.Contains(root.Name(), "evaluat") {
		t.Error("the row root is an envelope: no operation name, no 'evaluat' in its name")
	}

	// Generation chat.
	g := spanAttrs(genChat)
	if g["gen_ai.operation.name"].AsString() != "chat" || g["gen_ai.provider.name"].AsString() != "openai" ||
		g["gen_ai.request.model"].AsString() != "gpt-4o" || g["gen_ai.usage.input_tokens"].AsInt64() != 7 || g["gen_ai.usage.output_tokens"].AsInt64() != 3 {
		t.Errorf("generation chat identity/usage %v", g)
	}
	if sys := jsonAttr(t, genChat, "gen_ai.system_instructions"); sys[0]["content"] != "Answer in a brief way." {
		t.Errorf("system instructions %v", sys)
	}
	in := jsonAttr(t, genChat, "gen_ai.input.messages")
	if last := in[len(in)-1]; last["role"] != "user" || last["parts"].([]any)[0].(map[string]any)["content"] != "What is 2+2?" {
		t.Errorf("input messages %v", in)
	}
	if defs := jsonAttr(t, genChat, "gen_ai.tool.definitions"); len(defs) != 1 || defs[0]["name"] != "lookup" {
		t.Errorf("tool definitions %v", defs)
	}
	out := jsonAttr(t, genChat, "gen_ai.output.messages")
	parts := out[0]["parts"].([]any)
	call, text := parts[0].(map[string]any), parts[1].(map[string]any)
	if call["type"] != "tool_call" || call["name"] != "lookup" || call["id"] != spanAttrs(tool)["gen_ai.tool.call.id"].AsString() ||
		call["arguments"] != `{"q":"2+2"}` || text["type"] != "text" || text["content"] != "The answer is 4." {
		t.Errorf("output parts %v (arguments are the recorder's rendered string, as in Python)", parts)
	}
	// The OpenLLMetry carrier the trace view reads: system prompt as message 0.
	if g["gen_ai.system"].AsString() != "openai" || g["gen_ai.prompt.0.role"].AsString() != "system" ||
		g["gen_ai.prompt.0.content"].AsString() != "Answer in a brief way." || g["gen_ai.prompt.1.content"].AsString() != "What is 2+2?" ||
		g["gen_ai.completion.0.role"].AsString() != "assistant" ||
		g["gen_ai.completion.0.content"].AsString() != `{"name":"lookup","arguments":"{\"q\":\"2+2\"}"}`+"\nThe answer is 4." {
		t.Errorf("generation chat OpenLLMetry carrier %v", g)
	}
	if evs := genChat.Events(); len(evs) != 2 || evs[0].Name != "gen_ai.content.prompt" || evs[1].Name != "gen_ai.content.completion" {
		t.Errorf("generation chat content events %v", evs)
	}

	// Tool.
	ta := spanAttrs(tool)
	if ta["gen_ai.tool.name"].AsString() != "lookup" || ta["gen_ai.tool.call.id"].AsString() != "call_1" || ta["gen_ai.tool.type"].AsString() != "function" ||
		ta["gen_ai.tool.call.arguments"].AsString() != `{"q":"2+2"}` || ta["gen_ai.tool.call.result"].AsString() != "4" {
		t.Errorf("tool attrs %v", ta)
	}

	// Judge chat: rendered system prompt, the user message with the history
	// expanded, the judged output last, and the raw completion.
	if sys := jsonAttr(t, judgeChat, "gen_ai.system_instructions"); sys[0]["content"] != "You grade answers." {
		t.Errorf("judge system %v", sys)
	}
	jin := jsonAttr(t, judgeChat, "gen_ai.input.messages")
	history := jin[0]["parts"].([]any)[0].(map[string]any)["content"].(string)
	for _, want := range []string{"Transcript:", "What is 2+2?", "lookup", "The answer is 4."} {
		if !strings.Contains(history, want) {
			t.Errorf("judge user message lacks %q: %s", want, history)
		}
	}
	if last := jin[len(jin)-1]; last["role"] != "user" || last["parts"].([]any)[0].(map[string]any)["content"] != "The answer is 4." {
		t.Errorf("judged output must be the last input message: %v", jin)
	}
	jout := jsonAttr(t, judgeChat, "gen_ai.output.messages")
	if raw := jout[0]["parts"].([]any)[0].(map[string]any)["content"]; raw != `{"reasoning":"correct","score":0.9}` {
		t.Errorf("judge completion %v", raw)
	}

	// evaluate: the result event with no verdict, mirrored on the span.
	ea := spanAttrs(eval)
	if ea["gen_ai.operation.name"].AsString() != "evaluate" || ea["gen_ai.system"].AsString() != "openai" ||
		ea["gen_ai.provider.name"].AsString() != "openai" || ea["gen_ai.request.model"].AsString() != "judge-model" {
		t.Errorf("evaluate judge identity %v", ea)
	}
	events := eval.Events()
	if len(events) != 1 || events[0].Name != "gen_ai.evaluation.result" {
		t.Fatalf("evaluate events %v", events)
	}
	ev := map[string]attribute.Value{}
	for _, a := range events[0].Attributes {
		ev[string(a.Key)] = a.Value
	}
	for _, m := range []map[string]attribute.Value{ev, ea} {
		if m["gen_ai.evaluation.name"].AsString() != "acc" || m["gen_ai.evaluation.score.value"].AsFloat64() != 0.9 ||
			m["gen_ai.evaluation.explanation"].AsString() != "correct" {
			t.Errorf("result %v", m)
		}
		for _, k := range []string{"gen_ai.evaluation.score.label", "verdict", "passed", "successDirection"} {
			if _, has := m[k]; has {
				t.Errorf("result carries %s", k)
			}
		}
	}
	for _, k := range []string{"gen_ai.input.messages", "gen_ai.output.messages", "gen_ai.tool.call.arguments"} {
		if _, has := ea[k]; has {
			t.Errorf("evaluate carries %s; content belongs on its chat child", k)
		}
	}

	// Tracked payloads are unchanged.
	for _, name := range []string{GenerationEventName, CriterionEventName} {
		for _, ev := range e.ld.events(name) {
			for _, k := range []string{"tool_calls", "toolCalls", "trajectory", "arguments", "messages", "prompt", "input"} {
				if _, has := ev.Data[k]; has {
					t.Errorf("%s gained %s", name, k)
				}
			}
		}
	}

	// The tracer flush follows the client's last flush.
	log := e.logSnapshot()
	lastFlush, tracerFlush := -1, -1
	for i, l := range log {
		switch l {
		case "flush":
			lastFlush = i
		case "tracer flush":
			tracerFlush = i
		}
	}
	if tracerFlush < 0 || tracerFlush < lastFlush {
		t.Errorf("tracer flush must follow the client flush: %v", log)
	}
}

func TestEvaluationSpansWhenGenerationRaises(t *testing.T) {
	judgeCalls := 0
	e, args := spanRun(t, func(ctx context.Context, req aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
		callLookup(ctx, req)
		return aiserver.HandlerResult{}, errors.New("model unavailable")
	}, &judgeCalls)
	exp, _ := evalSpans(t, e)
	mustRun(t, e.module(), args)
	spans := spanSet(exp.GetSpans().Snapshots())
	if len(spans) != 3 || len(spans.named("evaluate acc")) != 0 || len(spans.named("chat judge-model")) != 0 || judgeCalls != 0 {
		t.Fatalf("spans %v, judge calls %d: a row that raised is never scored", spans.names(), judgeCalls)
	}
	root, chat, tool := spans.one(t, rowSpanName), spans.one(t, "chat gpt-4o"), spans.one(t, "execute_tool lookup")
	// error.type is the bare type name, as Python's type(error).__name__.
	for _, sp := range []sdktrace.ReadOnlySpan{root, chat} {
		if sp.Status().Code != codes.Error || spanAttrs(sp)["error.type"].AsString() != "errorString" {
			t.Errorf("%s must be ERROR with error.type errorString: %v %v", sp.Name(), sp.Status(), spanAttrs(sp))
		}
	}
	recorded := false
	for _, ev := range chat.Events() {
		recorded = recorded || ev.Name == "exception"
	}
	if !recorded {
		t.Error("a failed chat records the exception")
	}
	if _, has := spanAttrs(chat)["gen_ai.usage.input_tokens"]; has {
		t.Error("a failed chat has no usage")
	}
	ca := spanAttrs(chat)
	if _, has := ca["gen_ai.input.messages"]; !has {
		t.Error("a failed chat keeps its input")
	}
	if _, has := ca["gen_ai.output.messages"]; has {
		t.Error("a failed chat has no output")
	}
	if tool.Status().Code != codes.Ok || spanAttrs(tool)["gen_ai.tool.call.result"].AsString() != "4" ||
		tool.SpanContext().TraceID() != root.SpanContext().TraceID() {
		t.Errorf("tool span %v %v", tool.Status(), spanAttrs(tool))
	}
}

func TestEvaluationSpansForAHandlerThatOpensItsOwn(t *testing.T) {
	judgeCalls := 0
	e, args := spanRun(t, func(ctx context.Context, req aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
		run := aiserver.StartHandlerRun(ctx, aiserver.HandlerSpanOptions{Tracer: aiserver.TracerPrefix + "/test",
			Name: "invoke_agent test", System: "openai", Operation: "invoke_agent", RequestModel: "gpt-4o"}, req, aiserver.Prompt{})
		chat := run.StartChat("gpt-4o")
		run.EndChat(chat, "gpt-4o", aiserver.SpanUsage{Input: 1, Output: 1}, "tool_calls")
		tool := run.StartTool("lookup", "provider_call_9")
		run.ToolArguments(tool, map[string]any{"q": "2+2"})
		res, _ := req.ToolHandlers["lookup"].(aiserver.ToolFunc)(run.Context(), map[string]any{"q": "2+2"})
		run.ToolResult(tool, res)
		run.EndTool(tool, nil)
		_, foreign := otel.Tracer("example.com/other").Start(run.Context(), "third party")
		foreign.End()
		run.Finish("gpt-4o", "The answer is 4.")
		return aiserver.HandlerResult{Output: "The answer is 4."}, nil
	}, &judgeCalls)
	exp, _ := evalSpans(t, e)
	mustRun(t, e.module(), args)
	spans := spanSet(exp.GetSpans().Snapshots())
	root := spans.one(t, rowSpanName)
	agent, chat, tool := spans.one(t, "invoke_agent test"), spans.one(t, "chat gpt-4o"), spans.one(t, "execute_tool lookup")
	conv := rowConversationID("run-1", 0)
	for _, sp := range []sdktrace.ReadOnlySpan{agent, chat, tool} {
		a := spanAttrs(sp)
		if a["gen_ai.conversation.id"].AsString() != conv || a[attrRunID].AsString() != "run-1" || a[attrRowIndex].AsInt64() != 0 {
			t.Errorf("%s row identity %v", sp.Name(), a)
		}
		if v, has := a["launchdarkly.run.id"]; has && v.AsString() == "run-1" {
			t.Errorf("%s: launchdarkly.run.id must not be the evaluation run id", sp.Name())
		}
		if sp.SpanContext().TraceID() != root.SpanContext().TraceID() {
			t.Errorf("%s is in another trace", sp.Name())
		}
	}
	if !childOf(agent, root) {
		t.Error("the handler's root span must be a child of the row root")
	}
	if ta := spanAttrs(tool); ta["gen_ai.tool.name"].AsString() != "lookup" || ta["gen_ai.tool.call.id"].AsString() != "provider_call_9" ||
		ta["gen_ai.tool.call.arguments"].AsString() != `{"q":"2+2"}` || ta["gen_ai.tool.call.result"].AsString() != "4" {
		t.Errorf("handler tool span %v", ta)
	}
	foreign := spans.one(t, "third party")
	for _, k := range []string{"gen_ai.conversation.id", attrRunID, attrRowIndex} {
		if _, has := spanAttrs(foreign)[k]; has {
			t.Errorf("a non-SDK span got %s", k)
		}
	}
}

func TestEvaluationSpansFollowTheContentGate(t *testing.T) {
	gated := []string{"gen_ai.system_instructions", "gen_ai.input.messages", "gen_ai.output.messages",
		"gen_ai.tool.definitions", "gen_ai.tool.call.arguments", "gen_ai.tool.call.result", "gen_ai.evaluation.explanation"}
	cases := []struct {
		name      string
		configure aiserver.ContentCapture
		handler   aiserver.ContentCapture
		want      bool
	}{
		{"built-in default", aiserver.CaptureDefault, aiserver.CaptureDefault, true},
		{"configured off", aiserver.CaptureOff, aiserver.CaptureDefault, false},
		{"handler off", aiserver.CaptureDefault, aiserver.CaptureOff, false},
		{"handler on over configured off", aiserver.CaptureOff, aiserver.CaptureOn, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			judgeCalls := 0
			e, args := spanRun(t, plainGeneration, &judgeCalls)
			args.Handler.CaptureContent = tc.handler
			exp, _ := evalSpans(t, e)
			if err := aiserver.Configure(aiserver.SDKOptions{CaptureContent: tc.configure}); err != nil {
				t.Fatal(err)
			}
			mustRun(t, e.module(), args)
			// A non-evaluation span written through the same helpers, by a
			// handler with the same setting, follows the same gate.
			outside := sdkapi.CreateHandler(aiserver.ProvidesFor{Provider: "OpenAI", Mode: aiserver.ModeMessages},
				func(ctx context.Context, _ aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
					_, sp := otel.Tracer(aiserver.TracerPrefix+"/test").Start(ctx, "non-evaluation")
					aiserver.SetChatContent(ctx, sp, aiserver.ChatContent{SystemInstructions: "s",
						InputMessages: []aiserver.ChatMessage{{Role: "user", Parts: []aiserver.MessagePart{aiserver.TextPart("hi")}}}})
					sp.End()
					return aiserver.HandlerResult{}, nil
				}, nil, tc.handler)
			_, _ = outside.Invoke(context.Background(), aiserver.HandlerRequest{})

			got := map[string]bool{}
			var scored, nonEval bool
			for _, sp := range exp.GetSpans().Snapshots() {
				for _, a := range sp.Attributes() {
					got[string(a.Key)] = true
				}
				for _, ev := range sp.Events() {
					for _, a := range ev.Attributes {
						got[string(a.Key)] = true
						scored = scored || a.Key == "gen_ai.evaluation.score.value"
					}
				}
				if sp.Name() == "non-evaluation" {
					_, nonEval = spanAttrs(sp)["gen_ai.input.messages"]
				}
			}
			for _, k := range gated {
				if got[k] != tc.want {
					t.Errorf("%s present=%v, want %v", k, got[k], tc.want)
				}
			}
			if nonEval != tc.want {
				t.Errorf("non-evaluation span content present=%v, want %v", nonEval, tc.want)
			}
			if !scored || !got["gen_ai.evaluation.name"] || !got["gen_ai.request.model"] || !got["gen_ai.tool.name"] {
				t.Error("name, score, model and tool name are written whatever the gate")
			}
		})
	}
}

func TestErroredCriterionSpan(t *testing.T) {
	e := newEnv(t, 1)
	exp, _ := evalSpans(t, e)
	args := baseArgs(handler("OpenAI", aiserver.ModeMessages, nil))
	args.Criteria = []Criterion{mustScorer(t, "s", func(context.Context, DatasetRow, any) (float64, error) {
		return 0, errors.New("scorer broke")
	}, nil)}
	mustRun(t, e.module(), args)
	spans := spanSet(exp.GetSpans().Snapshots())
	eval := spans.one(t, "evaluate s")
	if eval.Status().Code != codes.Error || spanAttrs(eval)["error.type"].AsString() != causeScorerRaised || len(eval.Events()) != 0 {
		t.Errorf("errored criterion: %v %v events %v", eval.Status(), spanAttrs(eval), eval.Events())
	}
	if root := spans.one(t, rowSpanName); root.Status().Code != codes.Ok || !within(eval, root) {
		t.Error("the row generated, so its root is OK and wraps the criterion")
	}
}

func TestRowIDShadowsAnOuterConversationID(t *testing.T) {
	e := newEnv(t, 1)
	exp, _ := evalSpans(t, e)
	ctx := aiserver.WithConversationID(context.Background(), "caller-conversation")
	if _, err := e.module().Run(ctx, baseArgs(handler("OpenAI", aiserver.ModeMessages, nil))); err != nil {
		t.Fatal(err)
	}
	for _, sp := range exp.GetSpans().Snapshots() {
		if got := spanAttrs(sp)["gen_ai.conversation.id"].AsString(); got != rowConversationID("run-1", 0) {
			t.Errorf("%s conversation id %q", sp.Name(), got)
		}
	}
}

func TestTracerFlushRunsWhenTheEvaluationPhaseFails(t *testing.T) {
	e := newEnv(t, 2)
	exp, _ := evalSpans(t, e)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	args := baseArgs(handler("OpenAI", aiserver.ModeMessages, nil))
	args.Concurrency = 1
	args.Criteria = []Criterion{mustScorer(t, "s", func(context.Context, DatasetRow, any) (float64, error) {
		cancel()
		return 1, nil
	}, nil)}
	if _, err := e.module().Run(ctx, args); !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v", err)
	}
	if log := e.logSnapshot(); log[len(log)-1] != "tracer flush" {
		t.Fatalf("tracer flush must run last even on failure: %v", log)
	}
	if roots := spanSet(exp.GetSpans().Snapshots()).named(rowSpanName); len(roots) != 2 {
		t.Fatalf("every started row root must end, got %d", len(roots))
	}
}

// Each row is its own trace, whatever span the caller has open.
func TestRowRootStartsANewTrace(t *testing.T) {
	e := newEnv(t, 1)
	exp, _ := evalSpans(t, e)
	ctx, caller := otel.Tracer("example.com/app").Start(context.Background(), "caller")
	if _, err := e.module().Run(ctx, baseArgs(handler("OpenAI", aiserver.ModeMessages, nil))); err != nil {
		t.Fatal(err)
	}
	caller.End()
	root := spanSet(exp.GetSpans().Snapshots()).one(t, rowSpanName)
	if root.Parent().IsValid() || root.SpanContext().TraceID() == caller.SpanContext().TraceID() {
		t.Fatal("the row root must start a new trace")
	}
}

// Under a provider without the SDK span processor a zero span count proves
// nothing, so the runner writes no chat or execute_tool spans of its own.
func TestNoRunnerSpansWithoutTheSDKProcessor(t *testing.T) {
	judgeCalls := 0
	e, args := spanRun(t, plainGeneration, &judgeCalls)
	_ = aiserver.Shutdown(context.Background())
	t.Cleanup(func() { _ = aiserver.Shutdown(context.Background()) })
	exp := tracetest.NewInMemoryExporter()
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(sdktrace.NewTracerProvider(sdktrace.WithSyncer(exp)))
	t.Cleanup(func() { otel.SetTracerProvider(prev) })
	mustRun(t, e.module(), args)
	spans := spanSet(exp.GetSpans().Snapshots())
	for _, name := range []string{"chat gpt-4o", "execute_tool lookup", "chat judge-model"} {
		if len(spans.named(name)) != 0 {
			t.Errorf("runner wrote %q without the SDK processor", name)
		}
	}
	spans.one(t, rowSpanName)
	spans.one(t, "evaluate acc")
}

// A hosted row with no string input gets no final user turn.
func TestNoUserTurnWithoutRowInput(t *testing.T) {
	judgeCalls := 0
	e, args := spanRun(t, plainGeneration, &judgeCalls)
	e.route("GET", "projects/proj/datasets/ds/rows", resp{body: map[string]any{"items": []any{
		map[string]any{"rowIndex": 0, "variables": map[string]any{"tone": "brief"}},
	}, "totalCount": 1}})
	args.Criteria = nil
	exp, _ := evalSpans(t, e)
	mustRun(t, e.module(), args)
	chat := spanSet(exp.GetSpans().Snapshots()).one(t, "chat gpt-4o")
	if _, has := spanAttrs(chat)["gen_ai.input.messages"]; has {
		t.Errorf("no config messages besides system and no input: there is no input message, got %v", spanAttrs(chat)["gen_ai.input.messages"])
	}
}
