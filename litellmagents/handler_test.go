package litellmagents

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/kitbase"
)

func cfg(extra func(*aiserver.AiConfig)) aiserver.AiConfig {
	c := aiserver.AiConfig{Model: aiserver.Model{Name: "anthropic/claude-sonnet-4"}}
	c.Provider.Name = "Anthropic"
	if extra != nil {
		extra(&c)
	}
	return c
}

func instructions(s string) func(*aiserver.AiConfig) {
	return func(c *aiserver.AiConfig) { c.Instructions = s }
}

// msg is one outbound Chat Completions message, flattened: content is the
// string content, the JSON of a content-part array, or "" for null.
type msg struct {
	role, content string
	calls         string // tool_calls function names, comma-joined
	callID        string // tool_call_id
}

func messages(req map[string]any) []msg {
	var out []msg
	items, _ := req["messages"].([]any)
	for _, it := range items {
		m := it.(map[string]any)
		g := msg{role: m["role"].(string)}
		switch c := m["content"].(type) {
		case string:
			g.content = c
		case nil:
		default:
			b, _ := json.Marshal(c)
			g.content = string(b)
		}
		if tc, ok := m["tool_calls"].([]any); ok {
			var names []string
			for _, x := range tc {
				names = append(names, x.(map[string]any)["function"].(map[string]any)["name"].(string))
			}
			g.calls = strings.Join(names, ",")
		}
		g.callID, _ = m["tool_call_id"].(string)
		out = append(out, g)
	}
	return out
}

func roles(req map[string]any) []string {
	var out []string
	for _, m := range messages(req) {
		out = append(out, m.role)
	}
	return out
}

func toolNames(req map[string]any) []string {
	var out []string
	tools, _ := req["tools"].([]any)
	for _, t := range tools {
		out = append(out, t.(map[string]any)["function"].(map[string]any)["name"].(string))
	}
	slices.Sort(out)
	return out
}

func spansByName(exp *tracetest.InMemoryExporter) map[string][]sdktrace.ReadOnlySpan {
	out := map[string][]sdktrace.ReadOnlySpan{}
	for _, s := range exp.GetSpans().Snapshots() {
		out[s.Name()] = append(out[s.Name()], s)
	}
	return out
}

func attrs(s sdktrace.ReadOnlySpan) map[string]string {
	out := map[string]string{}
	for _, a := range s.Attributes() {
		out[string(a.Key)] = a.Value.Emit()
	}
	return out
}

func eventNames(s sdktrace.ReadOnlySpan) []string {
	var out []string
	for _, e := range s.Events() {
		out = append(out, e.Name)
	}
	return out
}

func tool(fn func(map[string]any) (any, error)) aiserver.ToolFunc {
	return func(_ context.Context, args map[string]any) (any, error) { return fn(args) }
}

func funcTool(desc string) aiserver.Tool {
	return aiserver.Tool{Type: "function", Description: desc, Parameters: map[string]any{"type": "object"}}
}

func invoke(t *testing.T, h *aiserver.ProviderHandler, req aiserver.HandlerRequest) aiserver.HandlerResult {
	t.Helper()
	res, err := h.Invoke(bg, req)
	if err != nil {
		t.Fatal(err)
	}
	return res
}

func collect(t *testing.T, h *aiserver.ProviderHandler, req aiserver.HandlerRequest) ([]aiserver.HandlerStreamEvent, error) {
	t.Helper()
	var evs []aiserver.HandlerStreamEvent
	for ev, err := range h.Stream(bg, req) {
		if err != nil {
			return evs, err
		}
		evs = append(evs, ev)
	}
	return evs, nil
}

// The handler is the */agent wildcard and keeps CaptureContent.
func TestFactoryMetadata(t *testing.T) {
	_, c := fakeClient(t)
	h := handler(t, Options{Client: c, CaptureContent: aiserver.CaptureOff})
	if h.ProvidesFor != (aiserver.ProvidesFor{Provider: kitbase.WildcardProvider, Mode: aiserver.ModeAgent}) {
		t.Errorf("ProvidesFor = %+v, want {*, agent}", h.ProvidesFor)
	}
	if h.CaptureContent != aiserver.CaptureOff || h.Stream == nil {
		t.Errorf("CaptureContent = %v, Stream set = %v", h.CaptureContent, h.Stream != nil)
	}
	if d := handler(t, Options{Client: c}); d.CaptureContent != aiserver.CaptureDefault {
		t.Errorf("omitted CaptureContent = %v, want CaptureDefault", d.CaptureContent)
	}
	if SpanName != "invoke_agent" || PackageName != "github.com/launchdarkly/go-ai-sdk/litellmagents" {
		t.Error("package constants changed")
	}
}

// With no base URL, no LITELLM_BASE_URL and no injected client, every
// constructor refuses instead of falling back to the default OpenAI client.
func TestNoProxyIsRefused(t *testing.T) {
	tripwire(t)
	if h, err := NewHandler(Options{}); !errors.Is(err, ErrNoBaseURL) || h != nil {
		t.Errorf("NewHandler = %v, %v; want nil, ErrNoBaseURL", h, err)
	}
	if h, err := NewHandler(Options{APIKey: "k"}); !errors.Is(err, ErrNoBaseURL) || h != nil {
		t.Errorf("NewHandler(APIKey only) = %v, %v; want nil, ErrNoBaseURL", h, err)
	}
	if g, err := LiteLLMGraph("g", aiserver.GraphOptions{}, Options{}); !errors.Is(err, ErrNoBaseURL) || g != nil {
		t.Errorf("LiteLLMGraph = %v, %v; want nil, ErrNoBaseURL", g, err)
	}
	ld := &fakeLD{fallback: variation}
	installClient(t, ld)
	if _, err := LiteLLMAgents(bg, "cfg", "q", ldcontext.New("u"), nil); !errors.Is(err, ErrNoBaseURL) {
		t.Errorf("LiteLLMAgents err = %v, want ErrNoBaseURL", err)
	}
}

// Requests go to the proxy's Chat Completions endpoint. Options win over
// LITELLM_*; LITELLM_* is the fallback; OPENAI_* is never read.
func TestProxyConnection(t *testing.T) {
	cases := []struct {
		name           string
		opts           func(url string) Options
		envURL, envKey bool
		wantAuth       string // "" accepts any
	}{
		{"options", func(u string) Options { return Options{BaseURL: u, APIKey: "opt-key"} }, false, false, "Bearer opt-key"},
		{"environment", func(string) Options { return Options{} }, true, true, "Bearer env-key"},
		{"options win over environment", func(u string) Options { return Options{BaseURL: u, APIKey: "opt-key"} }, true, true, "Bearer opt-key"},
		{"no key for an open proxy", func(u string) Options { return Options{BaseURL: u} }, false, false, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tripwire(t)
			t.Setenv("OPENAI_ORG_ID", "org-x")
			t.Setenv("OPENAI_PROJECT_ID", "proj-x")
			t.Setenv("OPENAI_CUSTOM_HEADERS", "X-Leak: yes")
			f, url := proxy(t, textResp("ok", 1, 1))
			if tc.envURL {
				t.Setenv("LITELLM_BASE_URL", url)
			}
			if tc.envKey {
				t.Setenv("LITELLM_API_KEY", "env-key")
			}
			h := handler(t, tc.opts(url))
			invoke(t, h, aiserver.HandlerRequest{Config: cfg(nil), UserInput: "q"})
			if f.count() != 1 {
				t.Fatalf("proxy requests = %d, want 1", f.count())
			}
			r := f.rec(0)
			if !strings.HasSuffix(r.path, "/chat/completions") {
				t.Errorf("path = %q, want the Chat Completions endpoint", r.path)
			}
			if tc.wantAuth != "" && r.auth != tc.wantAuth {
				t.Errorf("Authorization = %q, want %q", r.auth, tc.wantAuth)
			}
			if strings.Contains(r.auth, "sk-from-openai-env") {
				t.Errorf("OPENAI_API_KEY reached the proxy: %q", r.auth)
			}
			for _, hdr := range []string{"OpenAI-Organization", "OpenAI-Project", "X-Leak"} {
				if v := r.header.Get(hdr); v != "" {
					t.Errorf("%s = %q reached the proxy", hdr, v)
				}
			}
		})
	}
}

// Two handlers with different proxies each use their own.
func TestHandlersWithDifferentProxiesCoexist(t *testing.T) {
	tripwire(t)
	fa, a := proxy(t, textResp("a", 1, 1))
	fb, b := proxy(t, textResp("b", 1, 1))
	ha, hb := handler(t, Options{BaseURL: a}), handler(t, Options{BaseURL: b})
	ra := invoke(t, ha, aiserver.HandlerRequest{Config: cfg(nil), UserInput: "q"})
	rb := invoke(t, hb, aiserver.HandlerRequest{Config: cfg(nil), UserInput: "q"})
	if ra.Output != "a" || rb.Output != "b" || fa.count() != 1 || fb.count() != 1 {
		t.Errorf("outputs %v / %v, requests %d / %d", ra.Output, rb.Output, fa.count(), fb.count())
	}
}

// ClientFactory runs per call, after evaluation, with the evaluated config;
// it wins over Client; its error is the call's error and no request is sent.
func TestClientFactory(t *testing.T) {
	tripwire(t)
	f, c := fakeClient(t, textResp("one", 1, 1), textResp("two", 1, 1))
	_, other := fakeClient(t)
	var seen []string
	h := handler(t, Options{Client: other, ClientFactory: func(cfg aiserver.AiConfig) (*openai.Client, error) {
		seen = append(seen, cfg.Model.Name)
		return c, nil
	}})
	invoke(t, h, aiserver.HandlerRequest{Config: cfg(nil), UserInput: "q"})
	invoke(t, h, aiserver.HandlerRequest{Config: cfg(func(c *aiserver.AiConfig) { c.Model.Name = "proxy-alias" }), UserInput: "q"})
	if !reflect.DeepEqual(seen, []string{"anthropic/claude-sonnet-4", "proxy-alias"}) || f.count() != 2 {
		t.Errorf("factory saw %v, proxy requests %d", seen, f.count())
	}
	if f.request(1)["model"] != "proxy-alias" {
		t.Errorf("model = %v, want the evaluated alias", f.request(1)["model"])
	}

	boom := errors.New("no client for you")
	exp := tracer(t)
	h = handler(t, Options{ClientFactory: func(aiserver.AiConfig) (*openai.Client, error) { return nil, boom }})
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: cfg(nil)}); !errors.Is(err, boom) {
		t.Errorf("err = %v, want the factory's", err)
	}
	if root := spansByName(exp)[SpanName]; len(root) != 1 || root[0].Status().Code != codes.Error {
		t.Errorf("factory error not recorded on the root span: %v", spansByName(exp))
	}
}

// The evaluated model name reaches the request unchanged: provider-qualified
// names and proxy aliases alike.
func TestModelNameUnchanged(t *testing.T) {
	for _, name := range []string{"anthropic/claude-sonnet-4", "gemini/gemini-2.5-pro", "proxy-alias", "gpt-4o"} {
		t.Run(name, func(t *testing.T) {
			h, f, exp := setup(t, textResp("ok", 1, 1))
			invoke(t, h, aiserver.HandlerRequest{Config: cfg(func(c *aiserver.AiConfig) { c.Model.Name = name })})
			if f.request(0)["model"] != name {
				t.Errorf("model = %v, want %q", f.request(0)["model"], name)
			}
			if a := attrs(spansByName(exp)[SpanName][0]); a["gen_ai.request.model"] != name {
				t.Errorf("gen_ai.request.model = %q", a["gen_ai.request.model"])
			}
		})
	}
}

// A Bedrock target gets the model.region inference-profile prefix after the
// bedrock/ route; spans report the sent id, in the chat span name too.
func TestBedrockRegionPrefix(t *testing.T) {
	cases := []struct{ provider, name, region, want string }{
		{"Bedrock", "bedrock/anthropic.claude-x", "us", "bedrock/us.anthropic.claude-x"},
		{"Bedrock", "bedrock/anthropic.claude-x", "global", "bedrock/global.anthropic.claude-x"},
		{"Bedrock", "bedrock/converse/anthropic.claude-x", "us", "bedrock/converse/us.anthropic.claude-x"},
		{"Bedrock", "bedrock/invoke/anthropic.claude-x", "us", "bedrock/invoke/us.anthropic.claude-x"},
		{"Bedrock", "bedrock/llama/meta.llama-x", "us", "bedrock/llama/us.meta.llama-x"},
		{"Bedrock", "bedrock/us.anthropic.claude-x", "us", "bedrock/us.anthropic.claude-x"},
		{"Bedrock", "bedrock/global.anthropic.claude-x", "us", "bedrock/global.anthropic.claude-x"},
		{"Bedrock", "bedrock/us-gov.anthropic.claude-x", "us", "bedrock/us-gov.anthropic.claude-x"},
		{"Bedrock", "bedrock/eu.anthropic.claude-x", "us", "bedrock/eu.anthropic.claude-x"},
		{"Bedrock", "bedrock/apac.anthropic.claude-x", "us", "bedrock/apac.anthropic.claude-x"},
		{"Bedrock", "bedrock/au.anthropic.claude-x", "us", "bedrock/au.anthropic.claude-x"},
		{"Bedrock", "bedrock/jp.anthropic.claude-x", "us", "bedrock/jp.anthropic.claude-x"},
		{"Bedrock", "bedrock/ca.anthropic.claude-x", "us", "bedrock/ca.anthropic.claude-x"},
		{"Bedrock", "bedrock/in.anthropic.claude-x", "us", "bedrock/in.anthropic.claude-x"},
		{"Bedrock", "bedrock/us.anthropic.claude-x", "eu", "bedrock/us.anthropic.claude-x"},
		{"Bedrock", "bedrock/arn:aws:bedrock:us-east-1:1:inference-profile/x", "us", "bedrock/arn:aws:bedrock:us-east-1:1:inference-profile/x"},
		{"Bedrock", "bedrock/anthropic.claude-x", "", "bedrock/anthropic.claude-x"},
		{"Bedrock", "bedrock/anthropic.claude-x", "   ", "bedrock/anthropic.claude-x"},
		{"Bedrock", "bedrock/anthropic.claude-x", " us ", "bedrock/us.anthropic.claude-x"},
		{"Bedrock", "bedrock/anthropic.claude-x", "us.", "bedrock/us.anthropic.claude-x"},
		{"OpenAI", "bedrock/anthropic.claude-x", "us", "bedrock/us.anthropic.claude-x"},
		{"Bedrock", "anthropic/claude-x", "us", "anthropic/claude-x"},
		{"bedrock", "my-alias", "us", "us.my-alias"},
		{"OpenAI", "my-alias", "us", "my-alias"},
		{"OpenAI", "gpt-4o", "us", "gpt-4o"},
	}
	for _, tc := range cases {
		t.Run(tc.provider+"/"+tc.name+"/region="+strconv.Quote(tc.region), func(t *testing.T) {
			h, f, exp := setup(t, textResp("ok", 1, 1))
			invoke(t, h, aiserver.HandlerRequest{Config: cfg(func(c *aiserver.AiConfig) {
				c.Provider.Name, c.Model.Name, c.Model.Region = tc.provider, tc.name, tc.region
			})})
			if f.request(0)["model"] != tc.want {
				t.Errorf("model = %v, want %q", f.request(0)["model"], tc.want)
			}
			if a := attrs(spansByName(exp)[SpanName][0]); a["gen_ai.request.model"] != tc.want {
				t.Errorf("gen_ai.request.model = %q, want %q", a["gen_ai.request.model"], tc.want)
			}
			if n := len(spansByName(exp)["chat "+tc.want]); n != 1 {
				t.Errorf("want one %q span, spans = %v", "chat "+tc.want, spansByName(exp))
			}
		})
	}
}

// Instructions are the system message; history then user input follow, with
// images as image_url parts and no duplicated final user turn.
func TestPromptAndHistory(t *testing.T) {
	h, f, _ := setup(t, textResp("ok", 1, 1))
	invoke(t, h, aiserver.HandlerRequest{Config: cfg(func(c *aiserver.AiConfig) {
		c.Instructions = "Be {{tone}}"
		c.Messages = []aiserver.Message{{Role: "system", Text: "ignored: instructions win"}}
	}), UserInput: "now", Variables: map[string]any{"tone": "terse"}, History: []aiserver.Message{
		{Role: "user", Blocks: []aiserver.ContentBlock{{Type: "text", Text: "see"}, {Type: "image", Source: &aiserver.ImageSource{Type: "base64", MediaType: "image/png", Data: "AAA"}}}},
		{Role: "assistant", Text: "a cat"},
	}})
	got := messages(f.request(0))
	if len(got) != 4 || got[0] != (msg{role: "system", content: "Be terse"}) || got[2] != (msg{role: "assistant", content: "a cat"}) ||
		got[3] != (msg{role: "user", content: "now"}) {
		t.Fatalf("messages = %+v", got)
	}
	if got[1].role != "user" || !strings.Contains(got[1].content, `"image_url"`) || !strings.Contains(got[1].content, "data:image/png;base64,AAA") ||
		!strings.Contains(got[1].content, `"see"`) {
		t.Errorf("multimodal turn = %+v", got[1])
	}

	t.Run("empty input adds no user turn", func(t *testing.T) {
		h, f, _ := setup(t, textResp("ok", 1, 1))
		invoke(t, h, aiserver.HandlerRequest{Config: cfg(instructions("sys")), History: []aiserver.Message{{Role: "user", Text: "only"}}})
		if r := roles(f.request(0)); !reflect.DeepEqual(r, []string{"system", "user"}) {
			t.Errorf("roles = %v", r)
		}
	})
}

// Only config tools with a ToolFunc are offered, as Chat Completions
// function tools.
func TestToolConversion(t *testing.T) {
	h, f, _ := setup(t, textResp("ok", 1, 1))
	invoke(t, h, aiserver.HandlerRequest{Config: cfg(func(c *aiserver.AiConfig) {
		c.Tools = map[string]aiserver.Tool{"lookup": funcTool("find it"), "nohandler": funcTool("x")}
	}), ToolHandlers: map[string]aiserver.ToolHandler{"lookup": tool(func(map[string]any) (any, error) { return "", nil })}})
	tools, _ := f.request(0)["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("tools = %v", tools)
	}
	want := map[string]any{"type": "function", "function": map[string]any{"name": "lookup", "description": "find it", "parameters": map[string]any{"type": "object"}}}
	got := tools[0].(map[string]any)
	if fn, _ := got["function"].(map[string]any); fn != nil {
		delete(fn, "strict")
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("tool = %v, want %v", got, want)
	}

	t.Run("handlers without config tools offer nothing", func(t *testing.T) {
		h, f, _ := setup(t, textResp("ok", 1, 1))
		invoke(t, h, aiserver.HandlerRequest{Config: cfg(nil), UserInput: "q",
			ToolHandlers: map[string]aiserver.ToolHandler{"lookup": tool(func(map[string]any) (any, error) { return "", nil })}})
		if f.count() != 1 {
			t.Fatalf("proxy requests = %d, want 1", f.count())
		}
		if tools, ok := f.request(0)["tools"]; ok {
			t.Errorf("tools = %v, want none sent", tools)
		}
	})
}

// The loop runs every tool call, appends the assistant tool-call message and
// one role:tool result per call, resends the full conversation, and sums
// usage over turns.
func TestToolLoop(t *testing.T) {
	var ran []string
	h, f, exp := setup(t, toolResp(call{"a", `{"q":"x"}`}, call{"b", `{}`}), textResp("done", 5, 3))
	res := invoke(t, h, aiserver.HandlerRequest{Config: cfg(func(c *aiserver.AiConfig) {
		c.Instructions = "sys"
		c.Tools = map[string]aiserver.Tool{"a": funcTool(""), "b": funcTool("")}
	}), UserInput: "go", ToolHandlers: map[string]aiserver.ToolHandler{
		"a": tool(func(args map[string]any) (any, error) { ran = append(ran, "a:"+args["q"].(string)); return "A", nil }),
		"b": tool(func(map[string]any) (any, error) { ran = append(ran, "b"); return map[string]any{"n": 1}, nil }),
	}})
	if res.Output != "done" || !reflect.DeepEqual(ran, []string{"a:x", "b"}) || f.count() != 2 {
		t.Fatalf("output %v ran %v requests %d", res.Output, ran, f.count())
	}
	got := messages(f.request(1))
	if len(got) != 5 || got[0].role != "system" || got[1] != (msg{role: "user", content: "go"}) ||
		got[2].role != "assistant" || got[2].calls != "a,b" ||
		got[3] != (msg{role: "tool", content: "A", callID: "call_a"}) || got[4] != (msg{role: "tool", content: `{"n":1}`, callID: "call_b"}) {
		t.Errorf("second request messages = %+v", got)
	}
	if res.Usage["input_tokens"] != 6.0 || res.Usage["output_tokens"] != 4.0 {
		t.Errorf("usage = %v, want summed over both turns", res.Usage)
	}
	byName := spansByName(exp)
	if len(byName["chat anthropic/claude-sonnet-4"]) != 2 || len(byName["execute_tool a"]) != 1 || len(byName["execute_tool b"]) != 1 {
		t.Errorf("span tree = %v", byName)
	}

	t.Run("tool error propagates", func(t *testing.T) {
		boom := errors.New("tool failed")
		h, _, _ := setup(t, toolResp(call{"a", `{}`}))
		_, err := h.Invoke(bg, aiserver.HandlerRequest{Config: cfg(func(c *aiserver.AiConfig) {
			c.Tools = map[string]aiserver.Tool{"a": funcTool("")}
		}), ToolHandlers: map[string]aiserver.ToolHandler{"a": tool(func(map[string]any) (any, error) { return nil, boom })}})
		if !errors.Is(err, boom) {
			t.Errorf("err = %v", err)
		}
	})
}

// A call to a tool that was not offered fails the run and runs nothing.
func TestToolsFailClosed(t *testing.T) {
	for _, path := range []string{"invoke", "stream"} {
		for _, name := range []string{"ghost", "privileged"} {
			t.Run(path+"/"+name, func(t *testing.T) {
				turn := toolResp(call{name, `{}`})
				if path == "stream" {
					turn = sseToolCall(name, `{}`)
				}
				h, _, _ := setup(t, turn)
				ran := false
				req := aiserver.HandlerRequest{Config: cfg(func(c *aiserver.AiConfig) {
					c.Tools = map[string]aiserver.Tool{"safe": funcTool("")}
				}), UserInput: "q", ToolHandlers: map[string]aiserver.ToolHandler{
					"safe":       tool(func(map[string]any) (any, error) { return "ok", nil }),
					"privileged": tool(func(map[string]any) (any, error) { ran = true; return "secret", nil }),
				}}
				var err error
				if path == "invoke" {
					_, err = h.Invoke(bg, req)
				} else {
					_, err = collect(t, h, req)
				}
				if !errors.Is(err, ErrToolNotOffered) || !strings.Contains(err.Error(), strconv.Quote(name)) || ran {
					t.Errorf("err=%v ran=%v", err, ran)
				}
			})
		}
	}
}

// Chat Completions has no hosted tools: a NativeTool is never sent and
// warns once per tool key.
func TestNativeToolSkipped(t *testing.T) {
	var buf bytes.Buffer
	aiserver.SetLogger(slog.New(slog.NewTextHandler(&buf, nil)))
	t.Cleanup(func() { aiserver.SetLogger(nil) })
	key := "ld-search-" + t.Name()
	h, f, _ := setup(t, textResp("ok", 1, 1), textResp("ok", 1, 1))
	req := aiserver.HandlerRequest{Config: cfg(func(c *aiserver.AiConfig) {
		c.Tools = map[string]aiserver.Tool{key: funcTool(""), "lookup": funcTool("")}
	}), ToolHandlers: map[string]aiserver.ToolHandler{
		key:      &aiserver.NativeTool{ToolName: "web_search"},
		"lookup": tool(func(map[string]any) (any, error) { return "", nil }),
	}}
	invoke(t, h, req)
	invoke(t, h, req)
	if got := toolNames(f.request(0)); !reflect.DeepEqual(got, []string{"lookup"}) {
		t.Errorf("tools = %v, want only the function tool", got)
	}
	if strings.Contains(f.rec(0).raw, "web_search") {
		t.Error("the native tool reached the request")
	}
	if n := strings.Count(buf.String(), key); n != 1 || !strings.Contains(buf.String(), "level=WARN") {
		t.Errorf("want one WARN naming %s, log = %q", key, buf.String())
	}
}

// outputFormat is a json_schema response_format on every blocking turn and
// the parsed object is returned; streaming sends none.
func TestStructuredOutput(t *testing.T) {
	format := map[string]any{"properties": map[string]any{"score": map[string]any{"type": "number"}}}
	h, f, _ := setup(t, toolResp(call{"a", `{}`}), textResp(`{"score":7}`, 1, 1))
	res := invoke(t, h, aiserver.HandlerRequest{Config: cfg(func(c *aiserver.AiConfig) {
		c.OutputFormat = format
		c.Tools = map[string]aiserver.Tool{"a": funcTool("")}
	}), ToolHandlers: map[string]aiserver.ToolHandler{"a": tool(func(map[string]any) (any, error) { return "", nil })}})
	if !reflect.DeepEqual(res.Output, map[string]any{"score": 7.0}) {
		t.Errorf("output = %#v", res.Output)
	}
	for i := range 2 {
		rf, _ := f.request(i)["response_format"].(map[string]any)
		js, _ := rf["json_schema"].(map[string]any)
		schema, _ := js["schema"].(map[string]any)
		if rf["type"] != "json_schema" || js["name"] != "output" || js["strict"] != false || !reflect.DeepEqual(schema["properties"], format["properties"]) {
			t.Errorf("turn %d response_format = %v", i, f.request(i)["response_format"])
		}
	}

	h, f, _ = setup(t, textResp("plain", 1, 1))
	if res := invoke(t, h, aiserver.HandlerRequest{Config: cfg(nil)}); res.Output != "plain" {
		t.Errorf("output = %#v", res.Output)
	}
	if _, ok := f.request(0)["response_format"]; ok {
		t.Error("response_format sent without outputFormat")
	}

	h, f, _ = setup(t, sse([]string{"x"}, 1, 1))
	if _, err := collect(t, h, aiserver.HandlerRequest{Config: cfg(func(c *aiserver.AiConfig) { c.OutputFormat = format })}); err != nil {
		t.Fatal(err)
	}
	if _, ok := f.request(0)["response_format"]; ok {
		t.Error("streaming must ignore outputFormat")
	}
}

// max_turns (or Options.MaxTurns) caps the run.
func TestMaxTurns(t *testing.T) {
	loop := toolResp(call{"a", `{}`})
	for _, tc := range []struct {
		name   string
		opts   Options
		params map[string]any
		want   int
	}{
		{"parameter", Options{}, map[string]any{"max_turns": 2.0}, 2},
		{"option", Options{MaxTurns: 3}, nil, 3},
		{"parameter wins", Options{MaxTurns: 5}, map[string]any{"max_turns": 1.0}, 1},
		{"camelCase parameter", Options{}, map[string]any{"maxTurns": 2.0}, 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f, c := fakeClient(t, loop, loop, loop, loop, loop, loop)
			tc.opts.Client = c
			h := handler(t, tc.opts)
			_, err := h.Invoke(bg, aiserver.HandlerRequest{Config: cfg(func(c *aiserver.AiConfig) {
				c.Tools = map[string]aiserver.Tool{"a": funcTool("")}
				c.Model.Parameters = tc.params
			}), ToolHandlers: map[string]aiserver.ToolHandler{"a": tool(func(map[string]any) (any, error) { return "", nil })}})
			if !errors.Is(err, ErrMaxTurns) || f.count() != tc.want {
				t.Errorf("err = %v, requests = %d, want ErrMaxTurns after %d", err, f.count(), tc.want)
			}
		})
	}
}

var ldFixture = aiserver.TrackData{ConfigKey: "test-config", VariationKey: "variation-a", RunID: "run-123", Version: 1, ModelName: "anthropic/claude-sonnet-4", ProviderName: "Anthropic"}

// The root invoke_agent span has chat and execute_tool children, provider
// identity, usage and (by default) content.
func TestTelemetry(t *testing.T) {
	h, _, exp := setup(t, completion(`{"role":"assistant","content":"done"}`, "stop", 10, 4, 3))
	invoke(t, h, aiserver.HandlerRequest{Config: cfg(instructions("Be terse")), UserInput: "hi",
		Variables: map[string]any{kitbase.VarLD: ldFixture}})
	roots := spansByName(exp)[SpanName]
	if len(roots) != 1 {
		t.Fatalf("spans = %v", spansByName(exp))
	}
	a := attrs(roots[0])
	want := map[string]string{
		"gen_ai.operation.name": "invoke_agent", "gen_ai.provider.name": "anthropic",
		"gen_ai.request.model": "anthropic/claude-sonnet-4", "gen_ai.response.model": servedModel,
		"gen_ai.usage.input_tokens": "10", "gen_ai.usage.output_tokens": "3", "gen_ai.usage.cache_read.input_tokens": "4",
		"gen_ai.prompt.0.content": "Be terse", "gen_ai.prompt.1.content": "hi", "gen_ai.completion.0.content": "done",
		"launchdarkly.config.key": "test-config", "launchdarkly.run.id": "run-123",
	}
	for k, v := range want {
		if a[k] != v {
			t.Errorf("%s = %q, want %q", k, a[k], v)
		}
	}
	if roots[0].Status().Code != codes.Ok {
		t.Errorf("status = %v", roots[0].Status())
	}
	chat := spansByName(exp)["chat anthropic/claude-sonnet-4"]
	if len(chat) != 1 || chat[0].Parent().SpanID() != roots[0].SpanContext().SpanID() ||
		attrs(chat[0])["gen_ai.response.finish_reasons"] != `["stop"]` {
		t.Errorf("chat span = %v", chat)
	}

	t.Run("provider name defaults to litellm", func(t *testing.T) {
		h, _, exp := setup(t, textResp("ok", 1, 1))
		invoke(t, h, aiserver.HandlerRequest{Config: cfg(func(c *aiserver.AiConfig) { c.Provider.Name = "" })})
		if p := attrs(spansByName(exp)[SpanName][0])["gen_ai.provider.name"]; p != "litellm" {
			t.Errorf("gen_ai.provider.name = %q, want litellm", p)
		}
	})

	t.Run("CaptureOff keeps content off spans", func(t *testing.T) {
		_, c := fakeClient(t, textResp("secret answer", 1, 1))
		exp := tracer(t)
		invoke(t, handler(t, Options{Client: c, CaptureContent: aiserver.CaptureOff}), aiserver.HandlerRequest{Config: cfg(instructions("secret prompt")), UserInput: "secret input"})
		for _, s := range exp.GetSpans().Snapshots() {
			for k, v := range attrs(s) {
				if strings.Contains(v, "secret") {
					t.Errorf("%s carries content in %s", s.Name(), k)
				}
			}
		}
	})
}

// A proxy error is returned as the openai-go error and every span ends ERROR.
func TestProviderError(t *testing.T) {
	h, _, exp := setup(t, "STATUS400")
	_, err := h.Invoke(bg, aiserver.HandlerRequest{Config: cfg(nil), UserInput: "hi"})
	var apiErr *openai.Error
	if !errors.As(err, &apiErr) || apiErr.StatusCode != 400 {
		t.Fatalf("err = %v", err)
	}
	if len(spansByName(exp)[SpanName]) != 1 {
		t.Fatalf("spans = %v", spansByName(exp))
	}
	for _, s := range exp.GetSpans().Snapshots() {
		if s.Status().Code != codes.Error || !slices.Contains(eventNames(s), "exception") || s.EndTime().IsZero() {
			t.Errorf("%s: status %v events %v", s.Name(), s.Status().Code, eventNames(s))
		}
	}
}

func TestStream(t *testing.T) {
	t.Run("stream with usage, chunks, one done", func(t *testing.T) {
		h, f, exp := setup(t, sse([]string{"do", "ne"}, 10, 3))
		evs, err := collect(t, h, aiserver.HandlerRequest{Config: cfg(instructions("Be terse")), UserInput: "hi"})
		if err != nil {
			t.Fatal(err)
		}
		if len(evs) != 3 || evs[0].Text != "do" || evs[1].Text != "ne" || evs[2].Type != aiserver.EventDone || *evs[2].Output != "done" ||
			evs[2].Usage["input_tokens"] != 10.0 || evs[2].Usage["output_tokens"] != 3.0 {
			t.Fatalf("events = %+v", evs)
		}
		r := f.request(0)
		if r["stream"] != true || !reflect.DeepEqual(r["stream_options"], map[string]any{"include_usage": true}) {
			t.Errorf("stream = %v stream_options = %v", r["stream"], r["stream_options"])
		}
		if roots := spansByName(exp)[SpanName]; len(roots) != 1 || roots[0].Status().Code != codes.Ok {
			t.Errorf("root = %v", roots)
		}
	})

	t.Run("fragmented tool call arguments are joined by index", func(t *testing.T) {
		var got map[string]any
		h, f, exp := setup(t, sseToolCall("a", `{"q"`, `:"x"`, `}`), sse([]string{"ok"}, 2, 3))
		evs, err := collect(t, h, aiserver.HandlerRequest{Config: cfg(func(c *aiserver.AiConfig) {
			c.Tools = map[string]aiserver.Tool{"a": funcTool("")}
		}), ToolHandlers: map[string]aiserver.ToolHandler{"a": tool(func(args map[string]any) (any, error) { got = args; return "r", nil })}})
		if err != nil {
			t.Fatal(err)
		}
		if len(evs) == 0 || evs[len(evs)-1].Type != aiserver.EventDone || evs[len(evs)-1].Usage["input_tokens"] != 3.0 || evs[len(evs)-1].Usage["output_tokens"] != 4.0 {
			t.Errorf("events = %+v, want done with usage summed over both turns", evs)
		}
		if byName := spansByName(exp); len(byName["chat anthropic/claude-sonnet-4"]) != 2 || len(byName["execute_tool a"]) != 1 {
			t.Errorf("stream span tree = %v", byName)
		}
		if got["q"] != "x" || f.count() != 2 {
			t.Errorf("args = %v requests = %d", got, f.count())
		}
		if m := messages(f.request(1)); len(m) < 2 || m[len(m)-1] != (msg{role: "tool", content: "r", callID: "call_a"}) {
			t.Errorf("tool result message = %+v", m)
		}
	})

	t.Run("provider error ends the span with ERROR", func(t *testing.T) {
		h, _, exp := setup(t, "STATUS400")
		if _, err := collect(t, h, aiserver.HandlerRequest{Config: cfg(nil)}); err == nil {
			t.Fatal("want error")
		}
		if s := spansByName(exp)[SpanName]; len(s) != 1 || s[0].Status().Code != codes.Error || s[0].EndTime().IsZero() {
			t.Errorf("root = %v", s)
		}
	})

	t.Run("a failure in the middle of the stream closes the body, returns the error and ends the root with ERROR", func(t *testing.T) {
		mid := chunk(`{"content":"a"}`, "") + `data: {"error":{"message":"upstream broke","type":"server_error"}}` + "\n\n"
		h, f, exp := setup(t, "HANG:"+mid)
		evs, err := collect(t, h, aiserver.HandlerRequest{Config: cfg(nil)})
		if err == nil {
			t.Fatalf("want an error, got events %+v", evs)
		}
		for _, ev := range evs {
			if ev.Type == aiserver.EventDone {
				t.Errorf("a failed stream must not end with done: %+v", evs)
			}
		}
		select {
		case <-f.closed:
		case <-time.After(5 * time.Second):
			t.Fatal("the response body was not closed after the failure")
		}
		if s := spansByName(exp)[SpanName]; len(s) != 1 || s[0].Status().Code != codes.Error || s[0].EndTime().IsZero() {
			t.Errorf("root = %v, want one ended ERROR root span", s)
		}
	})

	t.Run("cancellation closes the response body and ends every span", func(t *testing.T) {
		h, f, exp := setup(t, "HANG:"+chunk(`{"content":"a"}`, ""))
		ctx, cancel := context.WithCancel(bg)
		defer cancel()
		var gotErr error
		for _, err := range h.Stream(ctx, aiserver.HandlerRequest{Config: cfg(nil)}) {
			if err != nil {
				gotErr = err
				break
			}
			cancel()
		}
		if !errors.Is(gotErr, context.Canceled) {
			t.Errorf("err = %v, want context.Canceled", gotErr)
		}
		select {
		case <-f.closed:
		case <-time.After(5 * time.Second):
			t.Fatal("the response body was not closed after cancellation")
		}
		if s := spansByName(exp)[SpanName]; len(s) != 1 || s[0].EndTime().IsZero() {
			t.Errorf("root = %v, want one ended root span", s)
		}
	})

	t.Run("consumer break closes the response body and ends every span", func(t *testing.T) {
		h, f, exp := setup(t, "HANG:"+chunk(`{"content":"a"}`, ""))
		for range h.Stream(bg, aiserver.HandlerRequest{Config: cfg(nil)}) {
			break
		}
		select {
		case <-f.closed:
		case <-time.After(5 * time.Second):
			t.Fatal("the response body was not closed after the consumer stopped")
		}
		spans := exp.GetSpans().Snapshots()
		if len(spans) != 2 {
			t.Fatalf("want root + chat ended, got %d", len(spans))
		}
		for _, s := range spans {
			if attrs(s)["launchdarkly.stream.abandoned"] != "true" {
				t.Errorf("%s not marked abandoned", s.Name())
			}
		}
	})
}

var variation = map[string]any{
	"model": map[string]any{"name": "anthropic/claude-sonnet-4"}, "provider": map[string]any{"name": "Anthropic"},
	"instructions": "say {{user_input}}",
	"_ldMeta":      map[string]any{"enabled": true, "variationKey": "v", "mode": "agent"},
}

// LiteLLMAgents wires a fresh wildcard handler from opts.Handler and
// forwards variables; nil opts needs LITELLM_BASE_URL.
func TestConvenienceExport(t *testing.T) {
	tripwire(t)
	f, url := proxy(t, textResp("one", 1, 1), textResp("two", 1, 1))
	installClient(t, &fakeLD{fallback: map[string]any{
		"model": map[string]any{"name": "proxy-alias"}, "provider": map[string]any{"name": "Anthropic"},
		"instructions": "tone {{tone}}", "_ldMeta": map[string]any{"enabled": true, "variationKey": "v", "mode": "agent"},
	}})
	res, err := LiteLLMAgents(bg, "cfg", "q", ldcontext.New("u"), &InvokeOptions{Handler: Options{BaseURL: url}, Variables: map[string]any{"tone": "dry"}})
	if err != nil {
		t.Fatal(err)
	}
	if m := messages(f.request(0)); res.Response != "one" || f.request(0)["model"] != "proxy-alias" || len(m) == 0 || m[0].content != "tone dry" {
		t.Errorf("res = %+v request = %v", res, f.request(0))
	}
	t.Setenv("LITELLM_BASE_URL", url)
	if res, err := LiteLLMAgents(bg, "cfg", "q", ldcontext.New("u"), nil); err != nil || res.Response != "two" {
		t.Errorf("nil opts: %+v, %v", res, err)
	}
}

// Through aiserver.Config, the wildcard handler serves any provider.
func TestThroughConfig(t *testing.T) {
	_, c := fakeClient(t, toolResp(call{"a", `{}`}), textResp("ok", 1, 1))
	v := map[string]any{
		"model": map[string]any{"name": "gemini/gemini-2.5-pro"}, "provider": map[string]any{"name": "Gemini"},
		"instructions": "I", "tools": map[string]any{"a": map[string]any{"name": "a", "type": "function", "parameters": map[string]any{"type": "object"}}},
		"_ldMeta": map[string]any{"enabled": true, "variationKey": "v", "mode": "agent"},
	}
	exp := installClient(t, &fakeLD{fallback: v})
	ran := false
	res, err := aiserver.Config(aiserver.ConfigArgs{Key: "cfg", Handlers: []*aiserver.ProviderHandler{handler(t, Options{Client: c})},
		ToolHandlers: map[string]aiserver.ToolHandler{"a": tool(func(map[string]any) (any, error) { ran = true; return "r", nil })},
	}).Invoke(bg, "q", ldcontext.New("u"))
	if err != nil || res.Response != "ok" || !ran {
		t.Fatalf("res %+v err %v ran %v", res, err, ran)
	}
	if s := spansByName(exp); len(s[SpanName]) != 1 || len(s["execute_tool a"]) != 1 {
		t.Errorf("span tree = %v", s)
	}
}

// LiteLLMAgents and ToLiteLLMAgents, called first, report only themselves.
func TestSdkUsageOnlyItself(t *testing.T) {
	tripwire(t)
	c := openai.NewClient(option.WithBaseURL("http://127.0.0.1:1"), option.WithMaxRetries(0))
	ld := &fakeLD{fallback: variation}
	installClient(t, ld)
	_, _ = LiteLLMAgents(bg, "cfg", "q", ldcontext.New("u"), &InvokeOptions{Handler: Options{Client: &c}})
	if got := ld.usageHelpers(); !reflect.DeepEqual(got, []string{"litellm-agents.litellmAgents"}) {
		t.Errorf("LiteLLMAgents usage helpers = %v", got)
	}
	ld = &fakeLD{fallback: variation}
	installClient(t, ld)
	_, _ = ToLiteLLMAgents(nil, &NativeGraphOptions{Handler: Options{Client: &c}})
	if got := ld.usageHelpers(); !reflect.DeepEqual(got, []string{"litellm-agents.toLiteLLMAgents"}) {
		t.Errorf("ToLiteLLMAgents usage helpers = %v", got)
	}
}

// Each helper reports its usage ID once, with this package's identity, and
// init registers the package for $ld:ai:sdk:info.
func TestSdkUsage(t *testing.T) {
	tripwire(t)
	t.Setenv("LITELLM_BASE_URL", "http://127.0.0.1:1")
	ld := &fakeLD{fallback: variation}
	installClient(t, ld)
	_, _ = NewHandler(Options{})
	_, _ = NewHandler(Options{})
	_, _ = LiteLLMGraph("g", aiserver.GraphOptions{}, Options{})
	_, _ = ToLiteLLMAgents(nil, nil)
	c := openai.NewClient(option.WithBaseURL("http://127.0.0.1:1"), option.WithMaxRetries(0))
	_, _ = LiteLLMAgents(bg, "cfg", "q", ldcontext.New("u"), &InvokeOptions{Handler: Options{Client: &c}})
	want := []string{"litellm-agents.createLiteLLMAgentsHandler", "litellm-agents.litellmGraph", "litellm-agents.toLiteLLMAgents", "litellm-agents.litellmAgents"}
	if got := ld.usageHelpers(); !reflect.DeepEqual(got, want) {
		t.Fatalf("usage helpers = %v, want %v", got, want)
	}
	for _, tr := range ld.events("$ld:ai:sdk:usage") {
		if tr.data["helperPackageName"] != PackageName || tr.data["helperPackageVersion"] != PackageVersion || tr.data["aiSdkLanguage"] != "go" {
			t.Errorf("usage event = %+v", tr)
		}
	}
	var info bool
	for _, tr := range ld.events("$ld:ai:sdk:info") {
		info = info || (tr.data["aiSdkName"] == PackageName && tr.data["aiSdkVersion"] == PackageVersion)
	}
	if !info {
		t.Error("init() must register the package for $ld:ai:sdk:info")
	}
}
