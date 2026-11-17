package litellmmessages

import (
	"encoding/json"
	"errors"
	"maps"
	"net/http"
	"net/http/httptest"
	"reflect"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

func toJSON(v any) string { b, _ := json.Marshal(v); return string(b) }

// --- factory, proxy ownership -------------------------------------------------

func TestFactoryMetadata(t *testing.T) {
	a, err := NewHandler(Options{BaseURL: "http://127.0.0.1:1", CaptureContent: aiserver.CaptureOff})
	if err != nil {
		t.Fatal(err)
	}
	if a.ProvidesFor != (aiserver.ProvidesFor{Provider: "*", Mode: aiserver.ModeMessages}) || a.CaptureContent != aiserver.CaptureOff || a.Stream == nil {
		t.Errorf("handler = %+v", a)
	}
	b, _ := NewHandler(Options{BaseURL: "http://127.0.0.1:1"})
	if a == b {
		t.Error("NewHandler must return independent handlers")
	}
}

func TestNoBaseURLIsRefused(t *testing.T) {
	t.Setenv("LITELLM_BASE_URL", "")
	t.Setenv("OPENAI_BASE_URL", "http://127.0.0.1:1")
	if h, err := NewHandler(Options{}); !errors.Is(err, ErrNoBaseURL) || h != nil {
		t.Errorf("NewHandler(Options{}) = %v, %v; want nil, ErrNoBaseURL", h, err)
	}
	if _, err := LiteLLMMessages(bg, "cfg", "q", ldcontext.New("u"), nil); !errors.Is(err, ErrNoBaseURL) {
		t.Errorf("LiteLLMMessages: err = %v, want ErrNoBaseURL", err)
	}
	c := openai.NewClient(option.WithBaseURL("http://127.0.0.1:1"))
	for name, o := range map[string]Options{
		"client":  {Client: &c},
		"factory": {ClientFactory: func(aiserver.AiConfig) (*openai.Client, error) { return &c, nil }},
	} {
		if _, err := NewHandler(o); err != nil {
			t.Errorf("%s: an injected client needs no base URL: %v", name, err)
		}
	}
}

// The built client talks only to the proxy: the OPENAI_* variables
// openai-go reads by default never reach the request.
func TestBuiltClientTargetsOnlyTheProxy(t *testing.T) {
	decoy := newProxy(t, text("decoy", 1, 1))
	t.Setenv("OPENAI_BASE_URL", decoy.url)
	t.Setenv("OPENAI_API_KEY", "sk-openai")
	t.Setenv("OPENAI_ORG_ID", "org-x")
	t.Setenv("OPENAI_PROJECT_ID", "proj-x")
	t.Setenv("OPENAI_CUSTOM_HEADERS", "X-Leak: yes")

	cases := []struct {
		name            string
		baseEnv, keyEnv string // LITELLM_BASE_URL ("proxy" = the fake proxy), LITELLM_API_KEY
		opts            func(url string) Options
		wantKey         string
	}{
		{"options beat the environment", "http://127.0.0.1:1", "env-key",
			func(url string) Options { return Options{BaseURL: url, APIKey: "opt-key"} }, "opt-key"},
		{"environment", "proxy", "env-key", func(string) Options { return Options{} }, "env-key"},
		{"no key", "proxy", "", func(string) Options { return Options{} }, "not-needed"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := newProxy(t, text("ok", 1, 1))
			base := tc.baseEnv
			if base == "proxy" {
				base = p.url
			}
			t.Setenv("LITELLM_BASE_URL", base)
			t.Setenv("LITELLM_API_KEY", tc.keyEnv)
			h, err := NewHandler(tc.opts(p.url))
			if err != nil {
				t.Fatal(err)
			}
			if _, err := h.Invoke(bg, req(cfg("anthropic/claude-sonnet-4", "Anthropic", nil), "q")); err != nil {
				t.Fatal(err)
			}
			if p.count() != 1 || decoy.count() != 0 {
				t.Fatalf("proxy got %d requests, OPENAI_BASE_URL got %d", p.count(), decoy.count())
			}
			r := p.request(0)
			if r.path != "/chat/completions" {
				t.Errorf("path = %s", r.path)
			}
			if got := r.header.Get("Authorization"); got != "Bearer "+tc.wantKey {
				t.Errorf("Authorization = %q, want Bearer %s", got, tc.wantKey)
			}
			for _, hdr := range []string{"OpenAI-Organization", "OpenAI-Project", "X-Leak"} {
				if v := r.header.Get(hdr); v != "" {
					t.Errorf("%s = %q reached the proxy", hdr, v)
				}
			}
		})
	}
}

func TestInjectedClientAndFactory(t *testing.T) {
	p := newProxy(t, text("one", 1, 1), text("two", 1, 1), text("three", 1, 1))
	c := openai.NewClient(option.WithBaseURL(p.url), option.WithAPIKey("k"), option.WithMaxRetries(0))
	h, err := NewHandler(Options{Client: &c})
	if err != nil {
		t.Fatal(err)
	}
	if res, err := h.Invoke(bg, req(cfg("m", "X", nil), "q")); err != nil || res.Output != "one" {
		t.Fatalf("injected client: %v %v", res, err)
	}

	var seen []string
	other := newProxy(t)
	oc := openai.NewClient(option.WithBaseURL(other.url), option.WithMaxRetries(0))
	f, err := NewHandler(Options{Client: &oc, ClientFactory: func(ac aiserver.AiConfig) (*openai.Client, error) {
		seen = append(seen, ac.Model.Name)
		return &c, nil
	}})
	if err != nil {
		t.Fatal(err)
	}
	if res, err := f.Invoke(bg, req(cfg("evaluated-model", "X", nil), "q")); err != nil || res.Output != "two" {
		t.Fatalf("factory: %v %v", res, err)
	}
	if !reflect.DeepEqual(seen, []string{"evaluated-model"}) || other.count() != 0 {
		t.Errorf("factory saw %v; Client got %d requests (the factory wins)", seen, other.count())
	}

	boom := errors.New("no client for you")
	bad, _ := NewHandler(Options{ClientFactory: func(aiserver.AiConfig) (*openai.Client, error) { return nil, boom }})
	if _, err := bad.Invoke(bg, req(cfg("m", "X", nil), "q")); !errors.Is(err, boom) {
		t.Errorf("factory error: %v", err)
	}
}

// --- model routing ----------------------------------------------------------

func TestModelReachesTheProxyUnchanged(t *testing.T) {
	for _, name := range []string{"anthropic/claude-sonnet-4", "gemini/gemini-2.5-pro", "company-alias", "openai/gpt-4o"} {
		h, p, _ := setup(t, text("ok", 1, 1))
		if _, err := h.Invoke(bg, req(cfg(name, "Anthropic", func(c *aiserver.AiConfig) { c.Model.Region = "us" }), "q")); err != nil {
			t.Fatal(err)
		}
		if got := p.request(0).body["model"]; got != name {
			t.Errorf("model = %v, want %q unchanged", got, name)
		}
	}
}

func TestBedrockInferenceProfilePrefix(t *testing.T) {
	const id = "anthropic.claude-opus-5-5"
	cases := []struct{ provider, region, name, want string }{
		{"Bedrock", "us", id, "us." + id},
		{"Bedrock", "global", id, "global." + id},
		{"bedrock", "us", id, "us." + id},
		{"BEDROCK", "us", id, "us." + id},
		{"Bedrock", "us", "us." + id, "us." + id},
		{"Bedrock", "eu", "us." + id, "us." + id},
		{"Bedrock", "us", "global." + id, "global." + id},
		{"Bedrock", "us", "us-gov." + id, "us-gov." + id},
		{"Bedrock", "us", "eu." + id, "eu." + id},
		{"Bedrock", "us", "apac." + id, "apac." + id},
		{"Bedrock", "us", "au." + id, "au." + id},
		{"Bedrock", "us", "jp." + id, "jp." + id},
		{"Bedrock", "us", "ca." + id, "ca." + id},
		{"Bedrock", "us", "in." + id, "in." + id},
		{"Bedrock", "us", "arn:aws:bedrock:us-east-1:1:inference-profile/x", "arn:aws:bedrock:us-east-1:1:inference-profile/x"},
		{"Bedrock", "", id, id},
		{"Bedrock", "   ", id, id},
		{"Bedrock", " us ", id, "us." + id},
		{"Bedrock", "us.", id, "us." + id},
		{"OpenAI", "us", "gpt-4o", "gpt-4o"},
		{"Bedrock", "us", "bedrock/" + id, "bedrock/us." + id},
		{"Bedrock", "us", "bedrock/converse/" + id, "bedrock/converse/us." + id},
		{"Bedrock", "us", "bedrock/invoke/" + id, "bedrock/invoke/us." + id},
		{"Bedrock", "us", "bedrock/llama/meta.llama3-3-70b-instruct-v1:0", "bedrock/llama/us.meta.llama3-3-70b-instruct-v1:0"},
		{"Bedrock", "us", "bedrock/us." + id, "bedrock/us." + id},
		{"Bedrock", "us", "bedrock/invoke/arn:aws:bedrock:us-east-1:1:x/y", "bedrock/invoke/arn:aws:bedrock:us-east-1:1:x/y"},
		{"Anthropic", "us", "bedrock/" + id, "bedrock/us." + id},
		{"Bedrock", "us", "anthropic/claude-x", "anthropic/claude-x"},
		{"Bedrock", "us", "company-alias", "us.company-alias"},
		{"OpenAI", "us", "company-alias", "company-alias"},
	}
	for _, tc := range cases {
		h, p, exp := setup(t, text("ok", 1, 1))
		c := cfg(tc.name, tc.provider, func(c *aiserver.AiConfig) { c.Model.Region = tc.region })
		if _, err := h.Invoke(bg, req(c, "q")); err != nil {
			t.Fatal(err)
		}
		if got := p.request(0).body["model"]; got != tc.want {
			t.Errorf("%s / region %q / %q: model = %v, want %q", tc.provider, tc.region, tc.name, got, tc.want)
		}
		// Bedrock telemetry reports the id sent, on the root and chat spans.
		spans := spansByName(exp)
		if len(spans[SpanName]) != 1 || attrs(spans[SpanName][0])["gen_ai.request.model"] != tc.want || len(spans["chat "+tc.want]) != 1 {
			t.Errorf("%q: spans do not report the sent id %q: %v", tc.name, tc.want, slices.Collect(maps.Keys(spans)))
		}
	}
}

// --- parameters -------------------------------------------------------------

var allowed = []string{"frequency_penalty", "max_completion_tokens", "max_tokens", "parallel_tool_calls", "presence_penalty", "reasoning_effort", "seed", "stop", "temperature", "tool_choice", "top_p"}

// sent is the request body for params on invoke or stream.
func sent(t *testing.T, params map[string]any, streaming bool) map[string]any {
	t.Helper()
	answer := text("ok", 1, 1)
	if streaming {
		answer = stream(1, 1, chunk(`{"role":"assistant","content":"ok"}`, "stop"))
	}
	h, p, _ := setup(t, answer)
	r := req(cfg("anthropic/claude-sonnet-4", "Anthropic", func(c *aiserver.AiConfig) { c.Model.Parameters = params }), "q")
	var err error
	if streaming {
		_, err = drain(h, r)
	} else {
		_, err = h.Invoke(bg, r)
	}
	if err != nil {
		t.Fatal(err)
	}
	return p.request(0).body
}

func TestParametersAllowlist(t *testing.T) {
	params := map[string]any{}
	for i, k := range allowed {
		params[k] = float64(i + 1)
	}
	params["stop"] = []any{"END"}
	params["tool_choice"] = "auto"
	params["parallel_tool_calls"] = true
	for _, streaming := range []bool{false, true} {
		body := sent(t, params, streaming)
		for _, k := range allowed {
			if !reflect.DeepEqual(body[k], params[k]) {
				t.Errorf("stream=%v: %s = %v, want %v", streaming, k, body[k], params[k])
			}
		}
		base := sent(t, nil, streaming)
		for k := range body {
			if _, ok := base[k]; !ok && !slices.Contains(allowed, k) {
				t.Errorf("stream=%v: unexpected key %q forwarded", streaming, k)
			}
		}
	}
}

func TestNeverForwardedParameters(t *testing.T) {
	never := map[string]any{
		"api_base": "http://evil", "api_key": "k", "base_url": "http://evil", "extra_headers": map[string]any{"x": "y"},
		"aws_access_key_id": "a", "aws_secret_access_key": "s", "aws_region_name": "us-east-1", "vertex_credentials": "c", "vertex_project": "p",
		"custom_llm_provider": "openai", "api_version": "v", "deployment_id": "d", "model_list": []any{}, "fallbacks": []any{"x"},
		"num_retries": 9, "max_retries": 9, "mock_response": "hi", "shared_session": "s", "timeout": 1, "user": "u",
		"store": true, "metadata": map[string]any{"a": "b"}, "logit_bias": map[string]any{}, "top_k": 3, "unknown_key": 1,
		"model": "other-model", "messages": []any{}, "tools": []any{}, "stream": true, "stream_options": map[string]any{}, "response_format": map[string]any{"type": "text"},
	}
	for _, streaming := range []bool{false, true} {
		base := sent(t, nil, streaming)
		body := sent(t, never, streaming)
		if !reflect.DeepEqual(body, base) {
			t.Errorf("stream=%v: never-forwarded keys changed the request:\n got %s\nwant %s", streaming, toJSON(body), toJSON(base))
		}
	}
}

// --- prompt, history, multimodal ----------------------------------------------

func messages(body map[string]any) string { return toJSON(body["messages"]) }

func TestPromptAndHistory(t *testing.T) {
	h, p, _ := setup(t, text("a", 1, 1), text("b", 1, 1), text("c", 1, 1))
	c := cfg("m", "X", func(c *aiserver.AiConfig) { c.Instructions = "Be {{tone}}." })
	r := req(c, "hello")
	r.Variables = map[string]any{"tone": "brief"}
	if _, err := h.Invoke(bg, r); err != nil {
		t.Fatal(err)
	}
	if got, want := messages(p.request(0).body), `[{"content":"Be brief.","role":"system"},{"content":"hello","role":"user"}]`; got != want {
		t.Errorf("instructions:\n got %s\nwant %s", got, want)
	}

	c = cfg("m", "X", func(c *aiserver.AiConfig) {
		c.Messages = []aiserver.Message{{Role: "system", Text: "sys"}, {Role: "user", Text: "configured"}}
	})
	if _, err := h.Invoke(bg, req(c, "")); err != nil {
		t.Fatal(err)
	}
	if got, want := messages(p.request(1).body), `[{"content":"sys","role":"system"},{"content":"configured","role":"user"}]`; got != want {
		t.Errorf("config messages:\n got %s\nwant %s", got, want)
	}

	r = req(cfg("m", "X", func(c *aiserver.AiConfig) { c.Instructions = "sys" }), "now")
	r.History = []aiserver.Message{
		{Role: "user", Blocks: []aiserver.ContentBlock{{Type: "text", Text: "look"}, {Type: "image", Source: &aiserver.ImageSource{Type: "base64", MediaType: "image/png", Data: "QUJD"}}}},
		{Role: "assistant", Text: "a cat"},
	}
	if _, err := h.Invoke(bg, r); err != nil {
		t.Fatal(err)
	}
	want := `[{"content":"sys","role":"system"},{"content":[{"text":"look","type":"text"},{"image_url":{"url":"data:image/png;base64,QUJD"},"type":"image_url"}],"role":"user"},{"content":"a cat","role":"assistant"},{"content":"now","role":"user"}]`
	if got := messages(p.request(2).body); got != want {
		t.Errorf("history:\n got %s\nwant %s", got, want)
	}
}

// --- tools, loop, usage, structured output -------------------------------------

func toolConfig(edit func(*aiserver.AiConfig)) aiserver.AiConfig {
	return cfg("anthropic/claude-sonnet-4", "Anthropic", func(c *aiserver.AiConfig) {
		c.Instructions = "sys"
		c.Tools = map[string]aiserver.Tool{
			"lookup":  funcTool("Look it up", map[string]any{"type": "object", "properties": map[string]any{"q": map[string]any{"type": "string"}}}),
			"weather": funcTool("", map[string]any{"type": "object"}),
			"no_fn":   funcTool("", map[string]any{"type": "object"}),
			"web":     {Type: "function"},
		}
		if edit != nil {
			edit(c)
		}
	})
}

func toolHandlers(log *[]string) map[string]aiserver.ToolHandler {
	return map[string]aiserver.ToolHandler{
		"lookup": tool(func(a map[string]any) (any, error) { *log = append(*log, "lookup:"+toJSON(a)); return "42", nil }),
		"weather": tool(func(a map[string]any) (any, error) {
			*log = append(*log, "weather")
			return map[string]any{"t": 20}, nil
		}),
		"web": &aiserver.NativeTool{ToolName: "web_search"},
	}
}

func TestToolsOffered(t *testing.T) {
	h, p, _ := setup(t, text("ok", 1, 1))
	var log []string
	r := req(toolConfig(nil), "q")
	r.ToolHandlers = toolHandlers(&log)
	if _, err := h.Invoke(bg, r); err != nil {
		t.Fatal(err)
	}
	want := `[{"function":{"description":"Look it up","name":"lookup","parameters":{"properties":{"q":{"type":"string"}},"type":"object"}},"type":"function"},{"function":{"description":"","name":"weather","parameters":{"type":"object"}},"type":"function"}]`
	if got := toJSON(p.request(0).body["tools"]); got != want {
		t.Errorf("tools (only callable config tools; no NativeTool, none without a function):\n got %s\nwant %s", got, want)
	}

	h, p, _ = setup(t, text("ok", 1, 1))
	if _, err := h.Invoke(bg, req(cfg("m", "X", nil), "q")); err != nil {
		t.Fatal(err)
	}
	if _, ok := p.request(0).body["tools"]; ok {
		t.Error("no tools must omit the tools field")
	}
}

func TestToolLoop(t *testing.T) {
	h, p, _ := setup(t,
		completion("", []call{{"c1", "lookup", `{"q":"a"}`}, {"c2", "weather", `{}`}}, 10, 2),
		completion("", []call{{"c3", "lookup", `{"q":"b"}`}}, 20, 3),
		text("done", 30, 4))
	var log []string
	r := req(toolConfig(nil), "q")
	r.ToolHandlers = toolHandlers(&log)
	res, err := h.Invoke(bg, r)
	if err != nil {
		t.Fatal(err)
	}
	if res.Output != "done" || !reflect.DeepEqual(log, []string{`lookup:{"q":"a"}`, "weather", `lookup:{"q":"b"}`}) {
		t.Fatalf("output %v, tool calls %v", res.Output, log)
	}
	if res.Usage["input_tokens"] != 60.0 || res.Usage["output_tokens"] != 9.0 {
		t.Errorf("usage = %v, want summed over every turn (60 / 9)", res.Usage)
	}
	if p.count() != 3 {
		t.Fatalf("requests = %d", p.count())
	}
	want := `[{"content":"sys","role":"system"},{"content":"q","role":"user"},` +
		`{"content":null,"role":"assistant","tool_calls":[{"function":{"arguments":"{\"q\":\"a\"}","name":"lookup"},"id":"c1","type":"function"},{"function":{"arguments":"{}","name":"weather"},"id":"c2","type":"function"}]},` +
		`{"content":"42","role":"tool","tool_call_id":"c1"},{"content":"{\"t\":20}","role":"tool","tool_call_id":"c2"}]`
	if got := messages(p.request(1).body); got != want {
		t.Errorf("second turn messages:\n got %s\nwant %s", got, want)
	}
}

func TestToolLoopBound(t *testing.T) {
	var answers []string
	for range 11 {
		answers = append(answers, completion("", []call{{"c", "lookup", `{}`}}, 1, 1))
	}
	h, _, _ := setup(t, answers...)
	var log []string
	r := req(toolConfig(nil), "q")
	r.ToolHandlers = toolHandlers(&log)
	if _, err := h.Invoke(bg, r); !errors.Is(err, ErrMaxSteps) {
		t.Errorf("err = %v, want ErrMaxSteps", err)
	}
}

func TestToolsFailClosed(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		for _, name := range []string{"ghost", "no_fn", "web", "privileged"} {
			answer := completion("", []call{{"c", name, `{}`}}, 1, 1)
			if streaming {
				answer = stream(1, 1, chunk(`{"role":"assistant","tool_calls":[{"index":0,"id":"c","type":"function","function":{"name":"`+name+`","arguments":"{}"}}]}`, "tool_calls"))
			}
			h, _, _ := setup(t, answer)
			var log []string
			ran := false
			r := req(toolConfig(nil), "q")
			r.ToolHandlers = toolHandlers(&log)
			r.ToolHandlers["privileged"] = tool(func(map[string]any) (any, error) { ran = true; return "x", nil })
			var err error
			if streaming {
				_, err = drain(h, r)
			} else {
				_, err = h.Invoke(bg, r)
			}
			if err == nil || !strings.Contains(err.Error(), `"`+name+`"`) || ran || len(log) != 0 {
				t.Errorf("stream=%v %s: err=%v ran=%v log=%v; want an error naming the tool and nothing run", streaming, name, err, ran, log)
			}
		}
	}
}

func TestToolErrorPropagates(t *testing.T) {
	h, _, _ := setup(t, completion("", []call{{"c", "lookup", `{}`}}, 1, 1))
	boom := errors.New("tool failed")
	r := req(toolConfig(nil), "q")
	r.ToolHandlers = map[string]aiserver.ToolHandler{"lookup": tool(func(map[string]any) (any, error) { return nil, boom })}
	if _, err := h.Invoke(bg, r); !errors.Is(err, boom) {
		t.Errorf("err = %v, want the tool's error", err)
	}
}

func TestStructuredOutput(t *testing.T) {
	format := map[string]any{"type": "json_schema", "properties": map[string]any{"a": map[string]any{"type": "number"}}}
	wantFormat := `{"json_schema":{"name":"output","schema":{"properties":{"a":{"type":"number"}},"type":"object"},"strict":false},"type":"json_schema"}`

	h, p, _ := setup(t, completion("", []call{{"c", "lookup", `{}`}}, 1, 1), text(`{"a":1}`, 1, 1))
	var log []string
	r := req(toolConfig(func(c *aiserver.AiConfig) { c.OutputFormat = format }), "q")
	r.ToolHandlers = toolHandlers(&log)
	res, err := h.Invoke(bg, r)
	if err != nil {
		t.Fatal(err)
	}
	for i := range 2 {
		if got := toJSON(p.request(i).body["response_format"]); got != wantFormat {
			t.Errorf("turn %d response_format:\n got %s\nwant %s", i, got, wantFormat)
		}
	}
	if res.Output != `{"a":1}` {
		t.Errorf("output = %#v, want the raw JSON string (the client layer parses it)", res.Output)
	}

	sh, sp, _ := setup(t, stream(1, 1, chunk(`{"role":"assistant","content":"x"}`, "stop")))
	if _, err := drain(sh, req(cfg("m", "X", func(c *aiserver.AiConfig) { c.OutputFormat = format }), "q")); err != nil {
		t.Fatal(err)
	}
	if _, ok := sp.request(0).body["response_format"]; ok {
		t.Error("streaming must ignore outputFormat")
	}
}

// --- streaming --------------------------------------------------------------

func TestStream(t *testing.T) {
	h, p, _ := setup(t, stream(5, 2,
		chunk(`{"role":"assistant","content":"Hel"}`, ""),
		chunk(`{"content":"lo"}`, ""),
		chunk(`{}`, "stop")))
	evs, err := drain(h, req(cfg("anthropic/claude-sonnet-4", "Anthropic", nil), "q"))
	if err != nil {
		t.Fatal(err)
	}
	var chunks []string
	dones := 0
	for _, ev := range evs {
		switch ev.Type {
		case aiserver.EventChunk:
			chunks = append(chunks, ev.Text)
		case aiserver.EventDone:
			dones++
			if ev.Output == nil || *ev.Output != "Hello" || ev.Usage["input_tokens"] != 5.0 || ev.Usage["output_tokens"] != 2.0 {
				t.Errorf("done = %+v", ev)
			}
		}
	}
	if !reflect.DeepEqual(chunks, []string{"Hel", "lo"}) || dones != 1 || evs[len(evs)-1].Type != aiserver.EventDone {
		t.Errorf("chunks %q, %d done events (want exactly one, last)", chunks, dones)
	}
	b := p.request(0).body
	if b["stream"] != true || !reflect.DeepEqual(b["stream_options"], map[string]any{"include_usage": true}) {
		t.Errorf("stream=%v stream_options=%v", b["stream"], b["stream_options"])
	}
}

func TestStreamToolCallFragments(t *testing.T) {
	h, p, _ := setup(t,
		stream(1, 1,
			chunk(`{"role":"assistant","tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"lookup","arguments":""}}]}`, ""),
			chunk(`{"tool_calls":[{"index":1,"id":"c2","type":"function","function":{"name":"weather","arguments":"{"}}]}`, ""),
			chunk(`{"tool_calls":[{"index":0,"function":{"arguments":"{\"q\":"}}]}`, ""),
			chunk(`{"tool_calls":[{"index":1,"function":{"arguments":"}"}}]}`, ""),
			chunk(`{"tool_calls":[{"index":0,"function":{"arguments":"\"a\"}"}}]}`, ""),
			chunk(`{}`, "tool_calls")),
		stream(1, 1, chunk(`{"role":"assistant","content":"done"}`, "stop")))
	var log []string
	r := req(toolConfig(nil), "q")
	r.ToolHandlers = toolHandlers(&log)
	if _, err := drain(h, r); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(log, []string{`lookup:{"q":"a"}`, "weather"}) || p.count() != 2 {
		t.Errorf("tool calls %v, requests %d", log, p.count())
	}
}

func TestStreamAbandonmentClosesTheBody(t *testing.T) {
	h, p, exp := setup(t, "HOLD")
	for ev, err := range h.Stream(bg, req(cfg("m", "X", nil), "q")) {
		if err != nil {
			t.Fatal(err)
		}
		if ev.Type == aiserver.EventChunk {
			break
		}
	}
	select {
	case <-p.closed:
	case <-time.After(3 * time.Second):
		t.Fatal("the response body was not closed after the consumer stopped")
	}
	for _, s := range exp.GetSpans().Snapshots() {
		if s.EndTime().IsZero() || s.Status().Code.String() == "Error" {
			t.Errorf("span %s: ended=%v status=%v; abandonment ends every span without an error", s.Name(), !s.EndTime().IsZero(), s.Status())
		}
	}
	if len(spansByName(exp)[SpanName]) != 1 {
		t.Errorf("root span not ended: %v", slices.Collect(maps.Keys(spansByName(exp))))
	}
}

func TestProviderErrors(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		h, _, exp := setup(t, "STATUS500")
		r := req(cfg("m", "X", nil), "q")
		var err error
		if streaming {
			_, err = drain(h, r)
		} else {
			_, err = h.Invoke(bg, r)
		}
		var apiErr *openai.Error
		if !errors.As(err, &apiErr) || apiErr.StatusCode != 500 {
			t.Errorf("stream=%v: err = %v, want the proxy's 500", streaming, err)
		}
		if root := spansByName(exp)[SpanName]; len(root) != 1 || root[0].Status().Code.String() != "Error" {
			t.Errorf("stream=%v: root span must record the error", streaming)
		}
	}
}

// --- retries ------------------------------------------------------------------

func TestMaxRetries(t *testing.T) {
	cases := []struct {
		name string
		max  *int
		want int
	}{{"unset keeps the SDK default of 2", nil, 3}, {"zero", retries(0), 1}, {"three", retries(3), 4}}
	for _, tc := range cases {
		for _, streaming := range []bool{false, true} {
			p := newProxy(t)
			h, err := NewHandler(Options{BaseURL: p.url, MaxRetries: tc.max})
			if err != nil {
				t.Fatal(err)
			}
			r := req(cfg("m", "X", func(c *aiserver.AiConfig) { c.Model.Parameters = map[string]any{"max_retries": 9, "num_retries": 9} }), "q")
			if streaming {
				_, _ = drain(h, r)
			} else {
				_, _ = h.Invoke(bg, r)
			}
			if p.count() != tc.want {
				t.Errorf("%s stream=%v: requests = %d, want %d", tc.name, streaming, p.count(), tc.want)
			}
		}
	}
	c := openai.NewClient(option.WithBaseURL("http://127.0.0.1:1"))
	for name, o := range map[string]Options{
		"negative":     {BaseURL: "http://127.0.0.1:1", MaxRetries: retries(-1)},
		"with Client":  {Client: &c, MaxRetries: retries(1)},
		"with factory": {ClientFactory: func(aiserver.AiConfig) (*openai.Client, error) { return &c, nil }, MaxRetries: retries(1)},
	} {
		if h, err := NewHandler(o); err == nil || h != nil || !strings.Contains(err.Error(), "MaxRetries") {
			t.Errorf("%s: %v, %v; want a refusal naming MaxRetries", name, h, err)
		}
		if _, err := LiteLLMMessages(bg, "cfg", "q", ldcontext.New("u"), &InvokeOptions{Handler: o}); err == nil || !strings.Contains(err.Error(), "MaxRetries") {
			t.Errorf("%s: LiteLLMMessages must return the refusal, got %v", name, err)
		}
	}
}

// --- telemetry ------------------------------------------------------------------

func TestTelemetry(t *testing.T) {
	h, _, exp := setup(t, completion("", []call{{"c1", "lookup", `{"q":"a"}`}}, 10, 2), text("secret answer", 5, 1))
	var log []string
	r := req(toolConfig(nil), "secret question")
	r.ToolHandlers = toolHandlers(&log)
	if _, err := h.Invoke(bg, r); err != nil {
		t.Fatal(err)
	}
	spans := spansByName(exp)
	chats := spans["chat anthropic/claude-sonnet-4"]
	if len(spans[SpanName]) != 1 || len(chats) != 2 || len(spans["execute_tool lookup"]) != 1 {
		t.Fatalf("span tree = %v", slices.Collect(maps.Keys(spans)))
	}
	root := spans[SpanName][0]
	for _, s := range exp.GetSpans().Snapshots() {
		if s.SpanContext().SpanID() != root.SpanContext().SpanID() && s.Parent().SpanID() != root.SpanContext().SpanID() {
			t.Errorf("%s is not a direct child of the root span", s.Name())
		}
	}
	a := attrs(chats[0])
	if a["gen_ai.system"] != "litellm" || a["gen_ai.provider.name"] != "anthropic" || a["gen_ai.request.model"] != "anthropic/claude-sonnet-4" || a["gen_ai.response.model"] != servedModel {
		t.Errorf("chat span attributes = %v", a)
	}
	if ra := attrs(root); ra["gen_ai.usage.input_tokens"] != "15" || ra["gen_ai.usage.output_tokens"] != "3" {
		t.Errorf("root usage = %v", ra)
	}

	h, _, exp = setup(t, text("ok", 1, 1))
	if _, err := h.Invoke(bg, req(cfg("alias", "", nil), "q")); err != nil {
		t.Fatal(err)
	}
	if got := attrs(spansByName(exp)[SpanName][0])["gen_ai.provider.name"]; got != "litellm" {
		t.Errorf("gen_ai.provider.name with no config provider = %q, want litellm", got)
	}
}

func TestContentCapture(t *testing.T) {
	for _, capture := range []aiserver.ContentCapture{aiserver.CaptureOff, aiserver.CaptureOn} {
		p := newProxy(t, text("secret answer", 1, 1))
		exp := withTracer(t)
		h, err := NewHandler(Options{BaseURL: p.url, CaptureContent: capture})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := h.Invoke(bg, req(cfg("m", "X", nil), "secret question")); err != nil {
			t.Fatal(err)
		}
		var all strings.Builder
		for _, s := range exp.GetSpans().Snapshots() {
			all.WriteString(toJSON(attrs(s)))
			for _, e := range s.Events() {
				for _, a := range e.Attributes {
					all.WriteString(a.Value.Emit())
				}
			}
		}
		got := strings.Contains(all.String(), "secret question") && strings.Contains(all.String(), "secret answer")
		if got != (capture == aiserver.CaptureOn) {
			t.Errorf("capture %v: content on spans = %v", capture, got)
		}
	}
}

// --- convenience wrapper and usage reporting ------------------------------------

func TestConvenienceExport(t *testing.T) {
	p := newProxy(t, text("one", 1, 1))
	ld := &fakeLD{variation: map[string]any{
		"model": map[string]any{"name": "anthropic/claude-sonnet-4"}, "provider": map[string]any{"name": "Anthropic"},
		"messages": []any{map[string]any{"role": "user", "content": "say {{word}}"}},
		"_ldMeta":  map[string]any{"enabled": true, "variationKey": "v"},
	}}
	installClient(t, ld)
	res, err := LiteLLMMessages(bg, "cfg", "", ldcontext.New("u-1"), &InvokeOptions{
		Variables: map[string]any{"word": "hello"}, Handler: Options{BaseURL: p.url, MaxRetries: retries(0)},
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.Response != "one" || !strings.Contains(messages(p.request(0).body), "say hello") {
		t.Errorf("response %+v, messages %s", res, messages(p.request(0).body))
	}
	if !reflect.DeepEqual(ld.helpers, []string{"litellm-messages.litellmMessages"}) {
		t.Errorf("usage helpers = %v, want only the wrapper's", ld.helpers)
	}
}

func TestSdkUsage(t *testing.T) {
	ld := &fakeLD{}
	installClient(t, ld)
	_, _ = NewHandler(Options{BaseURL: "http://127.0.0.1:1"})
	_, _ = NewHandler(Options{BaseURL: "http://127.0.0.1:1"})
	if !reflect.DeepEqual(ld.helpers, []string{"litellm-messages.createLiteLLMMessagesHandler"}) {
		t.Errorf("usage helpers = %v", ld.helpers)
	}
}

// A failing proxy is not a reason to fall back: the call fails against
// LITELLM_BASE_URL and OPENAI_BASE_URL is never tried.
func TestNoRequestWithoutProxy(t *testing.T) {
	var hits atomic.Int32
	decoy := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { hits.Add(1) }))
	t.Cleanup(decoy.Close)
	t.Setenv("OPENAI_BASE_URL", decoy.URL)
	t.Setenv("LITELLM_BASE_URL", "http://127.0.0.1:1")
	h, err := NewHandler(Options{MaxRetries: retries(0)})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := h.Invoke(bg, req(cfg("m", "X", nil), "q")); err == nil {
		t.Error("want a connection error from the unreachable proxy")
	}
	if hits.Load() != 0 {
		t.Errorf("OPENAI_BASE_URL got %d requests", hits.Load())
	}
}
