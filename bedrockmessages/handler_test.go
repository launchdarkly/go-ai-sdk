package bedrockmessages

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/bedrockwire"
)

var bg = context.Background()

func tracer(t *testing.T) *tracetest.InMemoryExporter {
	t.Helper()
	exp := tracetest.NewInMemoryExporter()
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(sdktrace.NewTracerProvider(sdktrace.WithSyncer(exp)))
	t.Cleanup(func() { otel.SetTracerProvider(prev) })
	return exp
}

// setup returns a handler with an injected client pointed at a fake that
// replays replies.
func setup(t *testing.T, opts Options, replies ...reply) (*aiserver.ProviderHandler, *fakeBedrock, *tracetest.InMemoryExporter) {
	t.Helper()
	t.Setenv("AWS_BEARER_TOKEN_BEDROCK", "")
	f := newFakeBedrock(t, replies...)
	if opts.Client == nil {
		opts.Client = f.client()
	}
	h, err := build(opts)
	if err != nil {
		t.Fatal(err)
	}
	return h.provider(), f, tracer(t)
}

func cfg(extra func(*aiserver.AiConfig)) aiserver.AiConfig {
	c := aiserver.AiConfig{Model: aiserver.Model{Name: "anthropic.claude-test"}}
	c.Provider.Name = "Bedrock"
	if extra != nil {
		extra(&c)
	}
	return c
}

func instructions(s string) func(*aiserver.AiConfig) {
	return func(c *aiserver.AiConfig) { c.Instructions = s }
}

func messages(r recorded) []map[string]any {
	var out []map[string]any
	for _, m := range r.body["messages"].([]any) {
		out = append(out, m.(map[string]any))
	}
	return out
}

func roles(r recorded) []string {
	var out []string
	for _, m := range messages(r) {
		out = append(out, m["role"].(string))
	}
	return out
}

func content(m map[string]any) []map[string]any {
	var out []map[string]any
	for _, c := range m["content"].([]any) {
		out = append(out, c.(map[string]any))
	}
	return out
}

func systemText(r recorded) string {
	s, _ := r.body["system"].([]any)
	if len(s) == 0 {
		return ""
	}
	return s[0].(map[string]any)["text"].(string)
}

func lastText(r recorded) string {
	m := messages(r)
	return content(m[len(m)-1])[0]["text"].(string)
}

func attrs(s sdktrace.ReadOnlySpan) map[string]string {
	out := map[string]string{}
	for _, a := range s.Attributes() {
		out[string(a.Key)] = a.Value.Emit()
	}
	return out
}

func spanNamed(t *testing.T, exp *tracetest.InMemoryExporter, name string) sdktrace.ReadOnlySpan {
	t.Helper()
	for _, s := range exp.GetSpans().Snapshots() {
		if s.Name() == name {
			return s
		}
	}
	t.Fatalf("no span %q", name)
	return nil
}

func spanCounts(exp *tracetest.InMemoryExporter) map[string]int {
	out := map[string]int{}
	for _, s := range exp.GetSpans().Snapshots() {
		out[s.Name()]++
	}
	return out
}

// eventAttr returns attribute key of the first event called name on s.
func eventAttr(s sdktrace.ReadOnlySpan, name, key string) string {
	for _, e := range s.Events() {
		if e.Name != name {
			continue
		}
		for _, kv := range e.Attributes {
			if string(kv.Key) == key {
				return kv.Value.Emit()
			}
		}
	}
	return ""
}

// checkContextIdentity asserts the context identity of c (kinds user
// u-1 and org o-1) on every root span and on no child span.
func checkContextIdentity(t *testing.T, exp *tracetest.InMemoryExporter, c ldcontext.Context) {
	t.Helper()
	roots := 0
	for _, s := range exp.GetSpans().Snapshots() {
		a, ev := attrs(s), map[string]string{}
		for _, e := range s.Events() {
			if e.Name == "feature_flag" {
				for _, kv := range e.Attributes {
					ev[string(kv.Key)] = kv.Value.Emit()
				}
			}
		}
		if s.Name() != SpanName {
			for k := range a {
				if strings.HasPrefix(k, "context.") || strings.HasPrefix(k, "launchdarkly.") {
					t.Errorf("child %s carries %s", s.Name(), k)
				}
			}
			if len(ev) > 0 {
				t.Errorf("child %s has a feature_flag event %v", s.Name(), ev)
			}
			continue
		}
		roots++
		if ev["feature_flag.context.id"] != c.FullyQualifiedKey() || ev["feature_flag.contextKeys"] != `{"org":"o-1","user":"u-1"}` {
			t.Errorf("root feature_flag event = %v, want context.id %q", ev, c.FullyQualifiedKey())
		}
		if a["context.contextKeys.user"] != "u-1" || a["context.contextKeys.org"] != "o-1" {
			t.Errorf("root context keys = user %q org %q", a["context.contextKeys.user"], a["context.contextKeys.org"])
		}
	}
	if roots == 0 {
		t.Error("no root span")
	}
}

// metadata, captureContent default, independent instances
// and independently owned clients.
func TestFactoryMetadata(t *testing.T) {
	f := newFakeBedrock(t)
	isolateAWSEnv(t, f)
	a, err := NewHandler(Options{})
	if err != nil {
		t.Fatal(err)
	}
	b, _ := NewHandler(Options{CaptureContent: true})
	if a.ProvidesFor != (aiserver.ProvidesFor{Provider: "Bedrock", Mode: aiserver.ModeMessages}) {
		t.Errorf("providesFor = %+v", a.ProvidesFor)
	}
	if a.CaptureContent || !b.CaptureContent || a == b || a.Invoke == nil || a.Stream == nil {
		t.Errorf("metadata wrong: a=%+v b=%+v", a, b)
	}
	h1, _ := build(Options{})
	h2, _ := build(Options{})
	if h1.Client == nil || h1.Client == h2.Client {
		t.Error("factory calls must own independent clients")
	}
}

// prompt paths, history, multimodal mapping.
func TestPromptPaths(t *testing.T) {
	var replies []reply
	for range 10 {
		replies = append(replies, textReply("ok", 1, 1))
	}
	h, f, _ := setup(t, Options{}, replies...)
	invoke := func(req aiserver.HandlerRequest) {
		t.Helper()
		if _, err := h.Invoke(bg, req); err != nil {
			t.Fatal(err)
		}
	}

	// 0. Path A: instructions → system; placeholders resolved or left as-is.
	invoke(aiserver.HandlerRequest{Config: cfg(instructions("Hi {{name}} {{missing}}")), UserInput: "q", Variables: map[string]any{"name": "Al"}})
	// 1. Path B with instructions too: messages win, last user turn → no append.
	both := cfg(func(c *aiserver.AiConfig) {
		c.Instructions = "IGNORED"
		c.Messages = []aiserver.Message{{Role: "system", Text: "sys {{x}}"}, {Role: "user", Text: "u {{x}}"}}
	})
	invoke(aiserver.HandlerRequest{Config: both, UserInput: "dup", Variables: map[string]any{"x": "1"}})
	// 2. Last config message assistant → userInput appended even when empty.
	asst := cfg(func(c *aiserver.AiConfig) { c.Messages = []aiserver.Message{{Role: "assistant", Text: "hello"}} })
	invoke(aiserver.HandlerRequest{Config: asst})
	// 3. History between config messages and userInput; system history dropped; image native.
	hist := []aiserver.Message{
		{Role: "system", Text: "drop me"},
		{Role: "user", Blocks: []aiserver.ContentBlock{{Type: "image", Source: &aiserver.ImageSource{Type: "base64", MediaType: "image/png", Data: "AAEC"}}, {Type: "text", Text: "see"}}},
		{Role: "assistant", Text: "seen"},
	}
	invoke(aiserver.HandlerRequest{Config: asst, UserInput: "next", History: hist})
	// 4. Empty userInput with history ending in user → no extra turn.
	invoke(aiserver.HandlerRequest{Config: cfg(instructions("x")), History: hist[:2]})
	// 5/6. Empty history behaves like no history.
	invoke(aiserver.HandlerRequest{Config: cfg(instructions("x")), UserInput: "q", History: []aiserver.Message{}})
	invoke(aiserver.HandlerRequest{Config: cfg(instructions("x")), UserInput: "q"})
	// 7. Instructions, empty userInput, no history → one empty user turn.
	invoke(aiserver.HandlerRequest{Config: cfg(instructions("x"))})
	// 8. Instructions + history + userInput → history, then the input.
	invoke(aiserver.HandlerRequest{Config: cfg(instructions("inst")), UserInput: "next", History: hist})

	r := f.reqs()
	if systemText(r[0]) != "Hi Al {{missing}}" || lastText(r[0]) != "q" || !reflect.DeepEqual(roles(r[0]), []string{"user"}) {
		t.Errorf("path A: %s", r[0].raw)
	}
	if systemText(r[1]) != "sys 1" || !reflect.DeepEqual(roles(r[1]), []string{"user"}) || lastText(r[1]) != "u 1" || strings.Contains(r[1].raw, "IGNORED") {
		t.Errorf("path B: %s", r[1].raw)
	}
	if _, has := r[2].body["system"]; has || !reflect.DeepEqual(roles(r[2]), []string{"assistant", "user"}) || lastText(r[2]) != bedrockwire.EmptyText {
		t.Errorf("assistant-last append: %s", r[2].raw)
	}
	if got := roles(r[3]); !reflect.DeepEqual(got, []string{"assistant", "user", "assistant", "user"}) || lastText(r[3]) != "next" || strings.Contains(r[3].raw, "drop me") {
		t.Errorf("history order: %v %s", got, r[3].raw)
	}
	img := content(messages(r[3])[1])[0]["image"].(map[string]any)
	if img["format"] != "png" || img["source"].(map[string]any)["bytes"] != "AAEC" {
		t.Errorf("image not mapped to Converse bytes: %v", img)
	}
	if got := roles(r[4]); !reflect.DeepEqual(got, []string{"user"}) {
		t.Errorf("empty input after user history appended a turn: %v", got)
	}
	if r[5].raw != r[6].raw {
		t.Errorf("empty history differs:\n%s\n%s", r[5].raw, r[6].raw)
	}
	if got, want := r[7].body["messages"], []any{map[string]any{"role": "user", "content": []any{map[string]any{"text": bedrockwire.EmptyText}}}}; !reflect.DeepEqual(got, want) {
		t.Errorf("instructions + empty input: messages = %v", got)
	}
	if got := roles(r[8]); !reflect.DeepEqual(got, []string{"user", "assistant", "user"}) || lastText(r[8]) != "next" || systemText(r[8]) != "inst" ||
		content(messages(r[8])[1])[0]["text"] != "seen" {
		t.Errorf("instructions + history order: %v %s", got, r[8].raw)
	}

	// URL images are never fetched: an error, not a silent drop.
	_, err := h.Invoke(bg, aiserver.HandlerRequest{Config: cfg(instructions("x")), History: []aiserver.Message{{Role: "user",
		Blocks: []aiserver.ContentBlock{{Type: "image", Source: &aiserver.ImageSource{Type: "url", URL: "https://x/y.png"}}}}}})
	if !errors.Is(err, ErrUnsupportedContent) || len(f.reqs()) != 9 {
		t.Errorf("URL image: err=%v requests=%d", err, len(f.reqs()))
	}
}

// wire shape: tool filtering, toolSpec, toolUse →
// toolResult conversation.
func TestToolConversionAndLoop(t *testing.T) {
	h, f, exp := setup(t, Options{}, toolReply("a", `{"city":"Paris","n":2}`), toolReply("b", `{}`), textReply("final", 2, 3), textReply("plain", 1, 1), textReply("none", 1, 1))
	params := map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}, "required": []any{"city"}, "additionalProperties": false}
	c := cfg(func(c *aiserver.AiConfig) {
		c.Instructions = "x"
		c.Tools = map[string]aiserver.Tool{
			"a": {Name: "a", Type: "function", Description: "tool a", Parameters: params},
			"b": {Name: "b", Type: "function", Parameters: map[string]any{"type": "object"}},
			"c": {Name: "c", Type: "function", Parameters: map[string]any{"type": "object"}},
		}
	})
	var calls []string
	var gotArgs map[string]any
	tools := map[string]aiserver.ToolHandler{
		"a": aiserver.ToolFunc(func(_ context.Context, args map[string]any) (any, error) {
			calls, gotArgs = append(calls, "a"), args
			return "sunny", nil
		}),
		"b": aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) {
			calls = append(calls, "b")
			return map[string]any{"ok": true}, nil
		}),
	}
	res, err := h.Invoke(bg, aiserver.HandlerRequest{Config: c, UserInput: "q", ToolHandlers: tools})
	if err != nil {
		t.Fatal(err)
	}
	r := f.reqs()
	sent := r[0].body["toolConfig"].(map[string]any)["tools"].([]any)
	if len(sent) != 2 {
		t.Fatalf("want 2 tools sent, got %d: %s", len(sent), r[0].raw)
	}
	spec := sent[0].(map[string]any)["toolSpec"].(map[string]any)
	if spec["name"] != "a" || spec["description"] != "tool a" || !reflect.DeepEqual(spec["inputSchema"].(map[string]any)["json"], params) {
		t.Errorf("toolSpec = %v", spec)
	}
	if !reflect.DeepEqual(calls, []string{"a", "b"}) || gotArgs["city"] != "Paris" || gotArgs["n"] != 2.0 || res.Output != "final" {
		t.Errorf("calls=%v args=%v out=%v", calls, gotArgs, res.Output)
	}
	// Turn 2 carries the assistant toolUse and a user toolResult.
	m := messages(r[1])
	if len(m) != 3 || m[1]["role"] != "assistant" || m[2]["role"] != "user" {
		t.Fatalf("follow-up messages: %s", r[1].raw)
	}
	use := content(m[1])[0]["toolUse"].(map[string]any)
	result := content(m[2])[0]["toolResult"].(map[string]any)
	if use["toolUseId"] != "tu-a" || use["name"] != "a" || result["toolUseId"] != "tu-a" ||
		result["content"].([]any)[0].(map[string]any)["text"] != "sunny" {
		t.Errorf("toolUse/toolResult: %v / %v", use, result)
	}
	if _, has := r[1].body["toolConfig"]; !has {
		t.Error("follow-up turns must keep toolConfig")
	}
	if got := content(messages(r[2])[4])[0]["toolResult"].(map[string]any)["content"].([]any)[0].(map[string]any)["text"]; got != `{"ok":true}` {
		t.Errorf("non-string tool result = %v", got)
	}
	if n := spanCounts(exp); n["chat anthropic.claude-test"] != 3 || n["execute_tool a"] != 1 || n["execute_tool b"] != 1 {
		t.Errorf("spans = %v", n)
	}

	// No tools in config: one call, handlers never invoked, no toolConfig.
	calls = nil
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: cfg(instructions("x")), ToolHandlers: tools}); err != nil {
		t.Fatal(err)
	}
	// A config tool without a handler is not sent.
	only := cfg(func(c *aiserver.AiConfig) {
		c.Instructions = "x"
		c.Tools = map[string]aiserver.Tool{"c": {Name: "c", Type: "function", Parameters: map[string]any{}}}
	})
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: only, ToolHandlers: tools}); err != nil {
		t.Fatal(err)
	}
	r = f.reqs()
	for _, i := range []int{3, 4} {
		if _, has := r[i].body["toolConfig"]; has {
			t.Errorf("request %d sent toolConfig: %s", i, r[i].raw)
		}
	}
	if len(calls) != 0 || len(r) != 5 {
		t.Errorf("calls=%v requests=%d", calls, len(r))
	}
}

// Bedrock telemetry: invoke_agent root, aws.bedrock identity, resolved
// modelId, cache folding, LD attributes on the root only.
func TestTelemetry(t *testing.T) {
	cached := reply{body: `{"output":{"message":{"role":"assistant","content":[{"text":"done"}]}},"stopReason":"end_turn",
"usage":{"inputTokens":10,"outputTokens":5,"totalTokens":20,"cacheReadInputTokens":3,"cacheWriteInputTokens":2},"metrics":{"latencyMs":1}}`}
	h, f, exp := setup(t, Options{}, toolReply("a", `{}`), cached)
	c := cfg(func(c *aiserver.AiConfig) {
		c.Instructions = "sys"
		c.Model = aiserver.Model{Name: "anthropic.claude-sonnet-4-5", Region: "us"}
		c.Tools = map[string]aiserver.Tool{"a": {Name: "a", Type: "function", Parameters: map[string]any{"type": "object"}}}
	})
	res, err := h.Invoke(bg, aiserver.HandlerRequest{Config: c, UserInput: "weather in Paris?",
		ToolHandlers: map[string]aiserver.ToolHandler{"a": aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) { return "r", nil })},
		Variables: map[string]any{"__ld": aiserver.TrackData{ConfigKey: "test-config", VariationKey: "variation-a", RunID: "run-123", Version: 1,
			ModelName: "test-model", ProviderName: "TestProvider"}}})
	if err != nil {
		t.Fatal(err)
	}
	if u := aiserver.ParseUsage(res.Usage); u.Input != 16 || u.Output != 6 || u.Total != 22 || u.InputDetails == nil || u.InputDetails.CacheRead != 3 || u.InputDetails.CacheCreation != 2 {
		t.Errorf("usage = %+v %+v", u, u.InputDetails)
	}
	const id = "us.anthropic.claude-sonnet-4-5"
	if f.reqs()[0].modelID != id {
		t.Errorf("modelId = %q", f.reqs()[0].modelID)
	}
	if n := spanCounts(exp); n[SpanName] != 1 || n["chat "+id] != 2 || n["execute_tool a"] != 1 || len(n) != 3 {
		t.Errorf("span tree = %v", n)
	}
	root := spanNamed(t, exp, SpanName)
	got := attrs(root)
	for k, want := range map[string]string{
		"gen_ai.operation.name": "invoke_agent", "gen_ai.system": "aws.bedrock", "gen_ai.provider.name": "aws.bedrock",
		"gen_ai.request.model": id, "gen_ai.response.model": id,
		"gen_ai.usage.input_tokens": "16", "gen_ai.usage.output_tokens": "6", "gen_ai.usage.total_tokens": "22",
		"gen_ai.usage.prompt_tokens": "16", "gen_ai.usage.completion_tokens": "6",
		"gen_ai.usage.cache_read.input_tokens": "3", "gen_ai.usage.cache_creation.input_tokens": "2",
		"launchdarkly.operation.type": "gen_ai", "launchdarkly.config.key": "test-config",
		"launchdarkly.variation.key": "variation-a", "launchdarkly.run.id": "run-123",
		"gen_ai.prompt.0.role": "system", "gen_ai.prompt.0.content": "sys", "gen_ai.prompt.1.role": "user", "gen_ai.prompt.1.content": "weather in Paris?",
		"gen_ai.completion.0.role": "assistant", "gen_ai.completion.0.content": "done",
	} {
		if got[k] != want {
			t.Errorf("root %s = %q, want %q", k, got[k], want)
		}
	}
	if _, has := got["launchdarkly.graph.key"]; has || root.Status().Code != codes.Ok {
		t.Errorf("graph key present or status %v", root.Status())
	}
	events := map[string]bool{}
	for _, e := range root.Events() {
		events[e.Name] = true
	}
	if !events["gen_ai.content.prompt"] || !events["gen_ai.content.completion"] {
		t.Errorf("events = %v", events)
	}
	if p := eventAttr(root, "gen_ai.content.prompt", "gen_ai.prompt"); !strings.Contains(p, "weather in Paris?") || !strings.Contains(p, "sys") {
		t.Errorf("gen_ai.content.prompt = %q", p)
	}
	if c := eventAttr(root, "gen_ai.content.completion", "gen_ai.completion"); c != "done" {
		t.Errorf("gen_ai.content.completion = %q", c)
	}
	for _, s := range exp.GetSpans().Snapshots() {
		if s.Name() == SpanName {
			continue
		}
		if s.Parent().SpanID() != root.SpanContext().SpanID() || s.Status().Code != codes.Ok {
			t.Errorf("%s: not an OK child of the root", s.Name())
		}
		for k := range attrs(s) {
			if strings.HasPrefix(k, "launchdarkly.") || strings.HasPrefix(k, "context.") {
				t.Errorf("child %s carries %s", s.Name(), k)
			}
		}
	}
	if a := attrs(spanNamed(t, exp, "chat "+id)); a["gen_ai.system"] != "aws.bedrock" || a["gen_ai.request.model"] != id {
		t.Errorf("chat attrs = %v", a)
	}
}

// provider and tool errors propagate and mark every span ERROR.
func TestErrors(t *testing.T) {
	h, _, exp := setup(t, Options{}, errorReply())
	_, err := h.Invoke(bg, aiserver.HandlerRequest{Config: cfg(instructions("x"))})
	var ve *types.ValidationException
	if !errors.As(err, &ve) {
		t.Fatalf("want ValidationException, got %v", err)
	}
	if len(exp.GetSpans().Snapshots()) != 2 {
		t.Errorf("spans = %v", spanCounts(exp))
	}
	for _, s := range exp.GetSpans().Snapshots() {
		if s.Status().Code != codes.Error || len(s.Events()) == 0 || s.EndTime().IsZero() {
			t.Errorf("%s: status %v events %d", s.Name(), s.Status().Code, len(s.Events()))
		}
	}

	h, _, _ = setup(t, Options{}, toolReply("a", `{}`))
	boom := errors.New("tool boom")
	c := cfg(func(c *aiserver.AiConfig) {
		c.Instructions = "x"
		c.Tools = map[string]aiserver.Tool{"a": {Name: "a", Type: "function", Parameters: map[string]any{}}}
	})
	_, err = h.Invoke(bg, aiserver.HandlerRequest{Config: c, ToolHandlers: map[string]aiserver.ToolHandler{
		"a": aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) { return nil, boom })}})
	if !errors.Is(err, boom) {
		t.Errorf("tool error not propagated: %v", err)
	}

	// A toolUse for a tool with no function fails, naming it, instead of
	// ending OK with no output or sending fewer toolResults than toolUses.
	h, f, exp := setup(t, Options{}, toolReply("zzz", `{}`), textReply("never", 1, 1))
	called := false
	_, err = h.Invoke(bg, aiserver.HandlerRequest{Config: c, ToolHandlers: map[string]aiserver.ToolHandler{
		"a": aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) { called = true; return "r", nil })}})
	if !errors.Is(err, ErrToolNotFound) || !strings.Contains(err.Error(), `"zzz"`) || called || len(f.reqs()) != 1 {
		t.Errorf("unknown tool: err=%v called=%v requests=%d", err, called, len(f.reqs()))
	}
	if root := spanNamed(t, exp, SpanName); root.Status().Code != codes.Error || root.EndTime().IsZero() {
		t.Errorf("unknown tool root status = %v", root.Status())
	}

	h, _, exp = setup(t, Options{}, errorReply())
	var got error
	for _, e := range h.Stream(bg, aiserver.HandlerRequest{Config: cfg(instructions("x"))}) {
		got = e
	}
	if !errors.As(got, &ve) || spanNamed(t, exp, SpanName).Status().Code != codes.Error {
		t.Errorf("stream error: %v", got)
	}
}

// model ID construction, including the negative double-prefix test.
func TestModelID(t *testing.T) {
	for _, tc := range []struct{ region, name, want string }{
		{"us", "anthropic.claude-sonnet-4-5", "us.anthropic.claude-sonnet-4-5"},
		{"us", "us.anthropic.claude-sonnet-4-5", "us.anthropic.claude-sonnet-4-5"},
		{"", "anthropic.claude-sonnet-4-5", "anthropic.claude-sonnet-4-5"},
		{"global", "anthropic.claude-sonnet-4-5", "global.anthropic.claude-sonnet-4-5"},
		{"us", "usa.model", "us.usa.model"}, // prefix must be exactly "<region>."
		{"eu", "us.anthropic.claude", "eu.us.anthropic.claude"},
	} {
		if got := bedrockwire.ModelID(aiserver.Model{Name: tc.name, Region: tc.region}); got != tc.want {
			t.Errorf("modelID(%q, %q) = %q, want %q", tc.region, tc.name, got, tc.want)
		}
	}
	h, f, _ := setup(t, Options{}, textReply("ok", 1, 1))
	c := cfg(func(c *aiserver.AiConfig) {
		c.Instructions = "x"
		c.Model = aiserver.Model{Name: "us.anthropic.claude-sonnet-4-5", Region: "us"}
	})
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: c}); err != nil {
		t.Fatal(err)
	}
	if got := f.reqs()[0].modelID; got != "us.anthropic.claude-sonnet-4-5" || strings.Contains(got, "us.us.") {
		t.Errorf("outbound modelId = %q", got)
	}
}

// escape hatch: called every turn (blocking, tool follow-up and
// streaming), merged after standard fields, handler-critical fields win.
func TestConverseOptions(t *testing.T) {
	var calls atomic.Int32
	opts := Options{ConverseOptions: func(c aiserver.AiConfig) *bedrockruntime.ConverseInput {
		calls.Add(1)
		return &bedrockruntime.ConverseInput{
			ModelId:  aws.String("hijacked"),
			Messages: []types.Message{{Role: types.ConversationRoleUser, Content: []types.ContentBlock{&types.ContentBlockMemberText{Value: "hijacked"}}}},
			System:   []types.SystemContentBlock{&types.SystemContentBlockMemberText{Value: "hijacked"}},
			ToolConfig: &types.ToolConfiguration{ToolChoice: &types.ToolChoiceMemberAny{},
				Tools: []types.Tool{&types.ToolMemberToolSpec{Value: types.ToolSpecification{Name: aws.String("hijacked")}}}},
			GuardrailConfig:              &types.GuardrailConfiguration{GuardrailIdentifier: aws.String("gr-1"), GuardrailVersion: aws.String("1")},
			AdditionalModelRequestFields: document.NewLazyDocument(map[string]any{"top_k": 5}),
			RequestMetadata:              map[string]string{"team": "ai"},
			PerformanceConfig:            &types.PerformanceConfiguration{Latency: types.PerformanceConfigLatencyOptimized},
			InferenceConfig:              &types.InferenceConfiguration{MaxTokens: aws.Int32(77)},
		}
	}}
	streamTool := frames(t, messageStart,
		[2]string{"contentBlockStart", `{"contentBlockIndex":0,"start":{"toolUse":{"toolUseId":"t1","name":"a"}}}`},
		[2]string{"contentBlockDelta", `{"contentBlockIndex":0,"delta":{"toolUse":{"input":"{}"}}}`},
		blockStop(0), messageStop("tool_use"), metadata(1, 1, 0, 0))
	h, f, _ := setup(t, opts, toolReply("a", `{}`), textReply("ok", 1, 1), streamTool, streamText(t, []string{"s"}, 1, 1))
	c := cfg(func(c *aiserver.AiConfig) {
		c.Instructions = "sys"
		c.Model.Parameters = map[string]any{"maxTokens": 10.0}
		c.Tools = map[string]aiserver.Tool{"a": {Name: "a", Type: "function", Parameters: map[string]any{"type": "object"}}}
	})
	tools := map[string]aiserver.ToolHandler{"a": aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) { return "r", nil })}
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: c, UserInput: "q", ToolHandlers: tools}); err != nil {
		t.Fatal(err)
	}
	for _, err := range h.Stream(bg, aiserver.HandlerRequest{Config: c, UserInput: "q", ToolHandlers: tools}) {
		if err != nil {
			t.Fatal(err)
		}
	}
	if calls.Load() != 4 {
		t.Errorf("callback calls = %d, want 4 (2 blocking + 2 streaming turns)", calls.Load())
	}
	for i, r := range f.reqs() {
		wantOp := "converse"
		if i >= 2 {
			wantOp = "converse-stream"
		}
		if r.op != wantOp || r.modelID != "anthropic.claude-test" || systemText(r) != "sys" || strings.Contains(r.raw, "hijacked") {
			t.Errorf("request %d: handler-critical field overridden: %s %s", i, r.op, r.raw)
		}
		if roles(r)[0] != "user" || lastText(recorded{body: map[string]any{"messages": r.body["messages"].([]any)[:1]}}) != "q" {
			t.Errorf("request %d messages replaced: %s", i, r.raw)
		}
		g := r.body["guardrailConfig"].(map[string]any)
		if g["guardrailIdentifier"] != "gr-1" || r.body["requestMetadata"].(map[string]any)["team"] != "ai" ||
			r.body["additionalModelRequestFields"].(map[string]any)["top_k"] != 5.0 ||
			r.body["performanceConfig"].(map[string]any)["latency"] != "optimized" ||
			r.body["inferenceConfig"].(map[string]any)["maxTokens"] != 77.0 {
			t.Errorf("request %d missing merged options: %s", i, r.raw)
		}
		tc := r.body["toolConfig"].(map[string]any)
		if _, isAny := tc["toolChoice"].(map[string]any)["any"]; !isAny || len(tc["tools"].([]any)) != 1 {
			t.Errorf("request %d toolConfig: %v", i, tc)
		}
	}
}

// model.parameters map only documented inference params;
// model.custom is never read (sentinel).
func TestParametersAndCustomSentinel(t *testing.T) {
	var returnCustom bool
	opts := Options{ConverseOptions: func(c aiserver.AiConfig) *bedrockruntime.ConverseInput {
		if !returnCustom {
			return nil
		}
		return &bedrockruntime.ConverseInput{AdditionalModelRequestFields: document.NewLazyDocument(c.Model.Custom)}
	}}
	h, f, _ := setup(t, opts, textReply("ok", 1, 1), textReply("ok", 1, 1), textReply("ok", 1, 1))
	c := cfg(func(c *aiserver.AiConfig) {
		c.Instructions = "x"
		c.OutputFormat = map[string]any{"type": "object"}
		c.Model.Parameters = map[string]any{"max_tokens": 100.0, "temperature": 0.5, "top_p": 0.9, "stopSequences": []any{"END"}, "unknownParam": "UNKNOWN_SENTINEL"}
		c.Model.Custom = map[string]any{"guardrailConfig": map[string]any{"guardrailIdentifier": "CUSTOM_SENTINEL"},
			"additionalModelRequestFields": map[string]any{"k": "CUSTOM_SENTINEL"}, "CUSTOM_SENTINEL": true}
	})
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: c, UserInput: "q"}); err != nil {
		t.Fatal(err)
	}
	r := f.reqs()[0]
	ic := r.body["inferenceConfig"].(map[string]any)
	want := map[string]any{"maxTokens": 100.0, "temperature": 0.5, "topP": 0.9, "stopSequences": []any{"END"}}
	if !reflect.DeepEqual(ic, want) {
		t.Errorf("inferenceConfig = %v", ic)
	}
	for _, s := range []string{"CUSTOM_SENTINEL", "UNKNOWN_SENTINEL", "guardrailConfig", "additionalModelRequestFields"} {
		if strings.Contains(r.raw, s) {
			t.Errorf("request carries %s: %s", s, r.raw)
		}
	}
	// outputFormat: best-effort system injection, never structured fields from custom.
	if s := systemText(r); !strings.HasPrefix(s, "x\n\n") || !strings.Contains(s, `{"type":"object"}`) {
		t.Errorf("outputFormat system = %q", s)
	}
	// No parameters → no inferenceConfig.
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: cfg(instructions("x"))}); err != nil {
		t.Fatal(err)
	}
	if _, has := f.reqs()[1].body["inferenceConfig"]; has {
		t.Errorf("inferenceConfig sent without parameters: %s", f.reqs()[1].raw)
	}
	// The callback may forward custom explicitly.
	returnCustom = true
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: c, UserInput: "q"}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(f.reqs()[2].raw, "CUSTOM_SENTINEL") {
		t.Errorf("explicit callback value missing: %s", f.reqs()[2].raw)
	}
}

// messages-only system content gets the instruction; absent schema adds none.
func TestOutputFormatMessagesPath(t *testing.T) {
	h, f, _ := setup(t, Options{}, textReply("ok", 1, 1), textReply("ok", 1, 1))
	c := cfg(func(c *aiserver.AiConfig) {
		c.Messages = []aiserver.Message{{Role: "system", Text: "from messages"}}
		c.OutputFormat = map[string]any{"type": "object"}
	})
	_, _ = h.Invoke(bg, aiserver.HandlerRequest{Config: c, UserInput: "q"})
	_, _ = h.Invoke(bg, aiserver.HandlerRequest{Config: cfg(instructions("plain")), UserInput: "q"})
	if s := systemText(f.reqs()[0]); !strings.HasPrefix(s, "from messages\n\n") || !strings.Contains(s, "JSON") {
		t.Errorf("messages-path system = %q", s)
	}
	if s := systemText(f.reqs()[1]); s != "plain" {
		t.Errorf("no-schema system = %q", s)
	}
}

// auth: explicit APIKey > AWS_BEARER_TOKEN_BEDROCK > credential chain;
// Region configures only the client.
func TestAuthAndRegion(t *testing.T) {
	f := newFakeBedrock(t, textReply("ok", 1, 1), textReply("ok", 1, 1), textReply("ok", 1, 1), textReply("ok", 1, 1))
	isolateAWSEnv(t, f)
	tracer(t)
	c := cfg(func(c *aiserver.AiConfig) {
		c.Instructions = "x"
		c.Model = aiserver.Model{Name: "anthropic.claude", Region: "us"}
	})
	call := func(opts Options) recorded {
		t.Helper()
		h, err := NewHandler(opts)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: c}); err != nil {
			t.Fatal(err)
		}
		r := f.reqs()
		return r[len(r)-1]
	}
	if r := call(Options{Region: "eu-west-1"}); !strings.HasPrefix(r.auth, "AWS4-HMAC-SHA256 Credential=AKIDENV/") ||
		!strings.Contains(r.auth, "/eu-west-1/bedrock/aws4_request") || r.modelID != "us.anthropic.claude" {
		t.Errorf("credential chain + factory region: auth=%q modelId=%q", r.auth, r.modelID)
	}
	if r := call(Options{}); !strings.Contains(r.auth, "/us-east-1/bedrock/") || strings.Contains(r.modelID, "us-east-1") {
		t.Errorf("AWS_REGION: auth=%q modelId=%q", r.auth, r.modelID)
	}
	t.Setenv("AWS_BEARER_TOKEN_BEDROCK", "env-token")
	if r := call(Options{}); r.auth != "Bearer env-token" {
		t.Errorf("env bearer: auth=%q", r.auth)
	}
	if r := call(Options{APIKey: "explicit-key"}); r.auth != "Bearer explicit-key" {
		t.Errorf("explicit APIKey must beat env: auth=%q", r.auth)
	}
}

// an injected client is used as-is; APIKey, Region and env never
// reconfigure it.
func TestInjectedClientAuthoritative(t *testing.T) {
	t.Setenv("AWS_BEARER_TOKEN_BEDROCK", "")
	f := newFakeBedrock(t, textReply("ok", 1, 1), streamText(t, []string{"x"}, 1, 1))
	injected := f.client()
	before := injected.Options()
	t.Setenv("AWS_BEARER_TOKEN_BEDROCK", "env-token")
	t.Setenv("AWS_REGION", "ap-south-1")
	tracer(t)
	h, err := build(Options{Client: injected, APIKey: "explicit-key", Region: "eu-west-1"})
	if err != nil {
		t.Fatal(err)
	}
	if h.Client != injected {
		t.Fatal("injected client replaced")
	}
	p := h.provider()
	if _, err := p.Invoke(bg, aiserver.HandlerRequest{Config: cfg(instructions("x"))}); err != nil {
		t.Fatal(err)
	}
	for _, err := range p.Stream(bg, aiserver.HandlerRequest{Config: cfg(instructions("x"))}) {
		if err != nil {
			t.Fatal(err)
		}
	}
	for _, r := range f.reqs() {
		if !strings.HasPrefix(r.auth, "AWS4-HMAC-SHA256 Credential=AKIDINJECTED/") || !strings.Contains(r.auth, "/us-west-2/") {
			t.Errorf("injected client reconfigured: auth=%q", r.auth)
		}
	}
	after := injected.Options()
	if after.Region != before.Region || after.BearerAuthTokenProvider != nil || !reflect.DeepEqual(after.AuthSchemePreference, before.AuthSchemePreference) {
		t.Errorf("injected client options mutated: %+v", after)
	}
}

// ConverseStream deltas, metadata usage, tool loop across
// streaming turns, abandonment.
func TestStream(t *testing.T) {
	toolTurn := frames(t, messageStart, textDelta(0, "Let me "), textDelta(0, "check. "), blockStop(0),
		[2]string{"contentBlockStart", `{"contentBlockIndex":1,"start":{"toolUse":{"toolUseId":"t1","name":"weather"}}}`},
		[2]string{"contentBlockDelta", `{"contentBlockIndex":1,"delta":{"toolUse":{"input":"{\"city\":"}}}`},
		[2]string{"contentBlockDelta", `{"contentBlockIndex":1,"delta":{"toolUse":{"input":"\"Paris\"}"}}}`},
		blockStop(1), messageStop("tool_use"), metadata(4, 2, 1, 0))
	h, f, exp := setup(t, Options{}, toolTurn, streamText(t, []string{"Sun", "ny"}, 3, 2), streamText(t, []string{"never"}, 1, 1))
	var gotArgs map[string]any
	c := cfg(func(c *aiserver.AiConfig) {
		c.Instructions = "x"
		c.OutputFormat = map[string]any{"type": "object"} // ignored when streaming
		c.Tools = map[string]aiserver.Tool{"weather": {Name: "weather", Type: "function", Parameters: map[string]any{"type": "object"}}}
	})
	tools := map[string]aiserver.ToolHandler{"weather": aiserver.ToolFunc(func(_ context.Context, a map[string]any) (any, error) {
		gotArgs = a
		return "sunny", nil
	})}
	var events []aiserver.HandlerStreamEvent
	for ev, err := range h.Stream(bg, aiserver.HandlerRequest{Config: c, UserInput: "q", ToolHandlers: tools,
		Variables: map[string]any{"__ld": aiserver.TrackData{ConfigKey: "test-config", VariationKey: "variation-a", RunID: "run-123"}}}) {
		if err != nil {
			t.Fatal(err)
		}
		events = append(events, ev)
	}
	var chunks []string
	for _, e := range events[:len(events)-1] {
		if e.Type != "chunk" {
			t.Fatalf("non-chunk before done: %+v", e)
		}
		chunks = append(chunks, e.Text)
	}
	done := events[len(events)-1]
	if !reflect.DeepEqual(chunks, []string{"Let me ", "check. ", "Sun", "ny"}) || done.Type != "done" || *done.Output != "Let me check. Sunny" {
		t.Fatalf("events = %+v", events)
	}
	if u := aiserver.ParseUsage(done.Usage); u.Input != 8 || u.Output != 4 {
		t.Errorf("usage = %+v", u)
	}
	if gotArgs["city"] != "Paris" {
		t.Errorf("tool args = %v", gotArgs)
	}
	r := f.reqs()
	if r[0].op != "converse-stream" || strings.Contains(systemText(r[0]), "JSON") {
		t.Errorf("stream request: %s %s", r[0].op, r[0].raw)
	}
	m := messages(r[1])
	asst := content(m[1])
	if len(m) != 3 || asst[0]["text"] != "Let me check. " || asst[1]["toolUse"].(map[string]any)["input"].(map[string]any)["city"] != "Paris" ||
		content(m[2])[0]["toolResult"].(map[string]any)["toolUseId"] != "t1" {
		t.Errorf("streamed tool-use not preserved: %s", r[1].raw)
	}
	root := spanNamed(t, exp, SpanName)
	a := attrs(root)
	for k, want := range map[string]string{
		"gen_ai.system": "aws.bedrock", "gen_ai.operation.name": "invoke_agent", "gen_ai.response.model": "anthropic.claude-test",
		"gen_ai.usage.input_tokens": "8", "gen_ai.usage.total_tokens": "12", "gen_ai.usage.cache_read.input_tokens": "1",
		"launchdarkly.config.key": "test-config", "gen_ai.completion.0.content": "Let me check. Sunny",
	} {
		if a[k] != want {
			t.Errorf("stream root %s = %q, want %q", k, a[k], want)
		}
	}
	rootEvents := map[string]bool{}
	for _, e := range root.Events() {
		rootEvents[e.Name] = true
	}
	if root.Status().Code != codes.Ok || !rootEvents["gen_ai.content.prompt"] || !rootEvents["gen_ai.content.completion"] {
		t.Errorf("stream root status %v events %v", root.Status(), rootEvents)
	}

	// Abandonment: break on the first chunk → spans abandoned, status unset.
	exp.Reset()
	for range h.Stream(bg, aiserver.HandlerRequest{Config: cfg(instructions("x"))}) {
		break
	}
	if len(exp.GetSpans().Snapshots()) != 2 {
		t.Errorf("abandon spans = %v", spanCounts(exp))
	}
	for _, s := range exp.GetSpans().Snapshots() {
		if attrs(s)["launchdarkly.stream.abandoned"] != "true" || s.Status().Code != codes.Unset {
			t.Errorf("%s: attrs=%v status=%v", s.Name(), attrs(s), s.Status().Code)
		}
	}
}

// context identity: the wrapper calls through
// config().invoke with variables, reports only its own helper, and the
// factory reports only its own.
func TestConvenienceExportAndUsage(t *testing.T) {
	f := newFakeBedrock(t, textReply("ok", 1, 1), textReply("ok", 1, 1), textReply("ok", 1, 1))
	t.Setenv("AWS_BEARER_TOKEN_BEDROCK", "")
	ld := &fakeLD{variation: map[string]any{
		"model": map[string]any{"name": "anthropic.claude", "region": "us"}, "provider": map[string]any{"name": "Bedrock"},
		"instructions": "Hi {{name}}", "_ldMeta": map[string]any{"enabled": true, "variationKey": "v"},
	}}
	// Flush helpers other tests reported before any client existed, so this
	// client starts with nothing reported.
	initOpts := aiserver.InitOptions{OTLPEndpoint: "http://127.0.0.1:1/v1/traces"}
	if _, err := aiserver.InitWithClient(bg, &fakeLD{}, initOpts); err != nil {
		t.Fatal(err)
	}
	_ = aiserver.Shutdown(bg)
	if _, err := aiserver.InitWithClient(bg, ld, initOpts); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = aiserver.Shutdown(bg) })
	exp := tracer(t) // InitWithClient installed its own provider

	multi := ldcontext.NewMulti(ldcontext.New("u-1"), ldcontext.NewWithKind("org", "o-1"))
	resp, err := BedrockMessages(bg, "cfg", "q", multi, &InvokeOptions{Variables: map[string]any{"name": "Al"}, Handler: Options{Client: f.client()}})
	if err != nil {
		t.Fatal(err)
	}
	r := f.reqs()[0]
	if resp.Response != "ok" || systemText(r) != "Hi Al" || lastText(r) != "q" || r.modelID != "us.anthropic.claude" {
		t.Errorf("wrapper call-through: resp=%v req=%s", resp.Response, r.raw)
	}
	if got := ld.usageHelpers(); !reflect.DeepEqual(got, []string{"bedrock-messages.bedrockMessages"}) {
		t.Errorf("wrapper usage helpers = %v", got)
	}
	if got := ld.requestedKeys(); !reflect.DeepEqual(got, []string{"cfg"}) {
		t.Errorf("wrapper evaluated flag keys %v, want [cfg]", got)
	}
	checkContextIdentity(t, exp, multi)

	// A repeat sends nothing; the factory reports itself only.
	if _, err := BedrockMessages(bg, "cfg", "q", multi, &InvokeOptions{Handler: Options{Client: f.client()}}); err != nil {
		t.Fatal(err)
	}
	if got := systemText(f.reqs()[1]); got != "Hi {{name}}" {
		t.Errorf("without Variables the placeholder must stay unrendered: %q", got)
	}
	if _, err := NewHandler(Options{Client: f.client()}); err != nil {
		t.Fatal(err)
	}
	_, _ = NewHandler(Options{Client: f.client()})
	if got := ld.usageHelpers(); !reflect.DeepEqual(got, []string{"bedrock-messages.bedrockMessages", "bedrock-messages.createBedrockMessagesHandler"}) {
		t.Errorf("usage helpers = %v", got)
	}
	// options may be nil: the wrapper builds its own client from the environment.
	isolateAWSEnv(t, f)
	if resp, err := BedrockMessages(bg, "cfg", "q", multi, nil); err != nil || resp.Response != "ok" {
		t.Errorf("nil options: %v %v", resp.Response, err)
	}
	ld.checkUsagePayloads(t) // wrapper and factory rows
}

// launchdarkly.graph.key is set when the __ld fixture has a graph key.
func TestGraphKeyAttribute(t *testing.T) {
	h, _, exp := setup(t, Options{}, textReply("ok", 1, 1))
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: cfg(instructions("x")),
		Variables: map[string]any{"__ld": aiserver.TrackData{ConfigKey: "k", GraphKey: "g"}}}); err != nil {
		t.Fatal(err)
	}
	if got := attrs(spanNamed(t, exp, SpanName))["launchdarkly.graph.key"]; got != "g" {
		t.Errorf("graph key = %q", got)
	}
}

// two handlers with their own injected clients each use only their
// own client, blocking and streaming.
func TestHandlersUseOwnClient(t *testing.T) {
	tracer(t)
	t.Setenv("AWS_BEARER_TOKEN_BEDROCK", "")
	f1 := newFakeBedrock(t, textReply("one", 1, 1), streamText(t, []string{"s1"}, 1, 1))
	f2 := newFakeBedrock(t, textReply("two", 1, 1), streamText(t, []string{"s2"}, 1, 1))
	h1, err := build(Options{Client: f1.client()})
	if err != nil {
		t.Fatal(err)
	}
	h2, err := build(Options{Client: f2.client()})
	if err != nil {
		t.Fatal(err)
	}
	for i, h := range []*aiserver.ProviderHandler{h1.provider(), h2.provider()} {
		want := []string{"one", "two"}[i]
		if res, err := h.Invoke(bg, aiserver.HandlerRequest{Config: cfg(instructions("x"))}); err != nil || res.Output != want {
			t.Errorf("handler %d: output %v err %v", i, res.Output, err)
		}
		for _, err := range h.Stream(bg, aiserver.HandlerRequest{Config: cfg(instructions("x"))}) {
			if err != nil {
				t.Errorf("handler %d stream: %v", i, err)
			}
		}
	}
	for i, f := range []*fakeBedrock{f1, f2} {
		if r := f.reqs(); len(r) != 2 || r[0].op != "converse" || r[1].op != "converse-stream" {
			t.Errorf("fake %d got %d requests", i, len(r))
		}
	}
}

// TestMaxSteps: a model that keeps asking for tools past the 10-turn bound
// fails the call with ErrMaxSteps instead of returning partial text.
func TestMaxSteps(t *testing.T) {
	var replies []reply
	for range bedrockwire.DefaultMaxSteps + 1 {
		replies = append(replies, toolReply("a", `{}`))
	}
	h, f, exp := setup(t, Options{}, replies...)
	c := cfg(func(c *aiserver.AiConfig) {
		c.Instructions = "x"
		c.Tools = map[string]aiserver.Tool{"a": {Name: "a", Type: "function", Parameters: map[string]any{}}}
	})
	tools := map[string]aiserver.ToolHandler{"a": aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) { return "r", nil })}
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: c, ToolHandlers: tools}); !errors.Is(err, ErrMaxSteps) {
		t.Fatalf("err = %v, want ErrMaxSteps", err)
	}
	if got := len(f.reqs()); got != bedrockwire.DefaultMaxSteps {
		t.Errorf("requests = %d, want %d", got, bedrockwire.DefaultMaxSteps)
	}
	if s := spanNamed(t, exp, SpanName); s.Status().Code != codes.Error {
		t.Errorf("root status = %v", s.Status())
	}
}
