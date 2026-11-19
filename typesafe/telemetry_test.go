package typesafe

import (
	"context"
	"encoding/json"
	"math"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

var ldFixture = aiserver.TrackData{ConfigKey: "test-config", VariationKey: "variation-a", RunID: "run-123", Version: 1,
	ModelName: "jev-latest", ProviderName: "TypeSafe"}

// invoke_agent → chat <model>, with model identity, usage and the
// LaunchDarkly attributes on the root, both spans OK and ended.
func TestTelemetry(t *testing.T) {
	newJev(t, ok(threeAnswers, 100, 20))
	exp := tracetest.NewInMemoryExporter()
	installClient(t, &fakeLD{}, exp)
	if _, err := invoke(t, aiserver.HandlerRequest{Config: configOf(t, judgeVariation("jev-latest", classifiers)),
		UserInput: "assistant output", Variables: map[string]any{"__ld": ldFixture}}); err != nil {
		t.Fatal(err)
	}
	spans := spansByName(exp)
	if len(exp.GetSpans().Snapshots()) != 2 || len(spans[SpanName]) != 1 || len(spans["chat jev-latest"]) != 1 {
		t.Fatalf("span tree = %v, want invoke_agent and chat jev-latest", spans)
	}
	root, chat := spans[SpanName][0], spans["chat jev-latest"][0]
	if chat.Parent().SpanID() != root.SpanContext().SpanID() || root.Parent().IsValid() {
		t.Error("chat must be a direct child of the root invoke_agent span")
	}
	usage := map[string]string{"gen_ai.usage.input_tokens": "100", "gen_ai.usage.output_tokens": "20", "gen_ai.usage.total_tokens": "120",
		"gen_ai.usage.prompt_tokens": "100", "gen_ai.usage.completion_tokens": "20"}
	want := map[string]string{
		"gen_ai.system": "typesafe", "gen_ai.operation.name": "invoke_agent", "gen_ai.request.model": "jev-latest",
		"gen_ai.response.model":       "jev-2026-01-01",
		"launchdarkly.operation.type": "gen_ai", "launchdarkly.config.key": "test-config",
		"launchdarkly.variation.key": "variation-a", "launchdarkly.run.id": "run-123",
	}
	for k, v := range usage {
		want[k] = v
	}
	a := attrs(root)
	for k, v := range want {
		if a[k] != v {
			t.Errorf("root %s = %q, want %q", k, a[k], v)
		}
	}
	c := attrs(chat)
	for k, v := range map[string]string{"gen_ai.operation.name": "chat", "gen_ai.request.model": "jev-latest"} {
		if c[k] != v {
			t.Errorf("chat %s = %q, want %q", k, c[k], v)
		}
	}
	for k, v := range usage {
		if c[k] != v {
			t.Errorf("chat %s = %q, want %q", k, c[k], v)
		}
	}
	if _, has := a["launchdarkly.graph.key"]; has {
		t.Error("graph key set without graphKey")
	}
	if v, ok := eventAttr(root, "feature_flag", "feature_flag.result.variant"); !ok || v != "variation-a" {
		t.Errorf("root feature_flag variant = %q, %v; want variation-a", v, ok)
	}
	if slices.Contains(eventNames(chat), "feature_flag") {
		t.Error("chat span carries the feature_flag event")
	}
	if _, has := c["launchdarkly.config.key"]; has {
		t.Error("chat span carries LaunchDarkly config attributes")
	}
	for _, s := range []string{SpanName, "chat jev-latest"} {
		sp := spans[s][0]
		if sp.Status().Code != codes.Ok || sp.EndTime().IsZero() {
			t.Errorf("%s: status %v, ended %v", s, sp.Status().Code, !sp.EndTime().IsZero())
		}
	}
}

// A client failure records the exception, sets ERROR and ends both spans,
// and returns the error.
func TestClientErrorSpans(t *testing.T) {
	newJev(t, reply{status: 400, body: `{"detail":"bad request"}`})
	exp := tracetest.NewInMemoryExporter()
	installClient(t, &fakeLD{}, exp)
	_, err := invoke(t, aiserver.HandlerRequest{Config: configOf(t, judgeVariation("jev-latest", classifiers))})
	if err == nil {
		t.Fatal("want the client error")
	}
	snaps := exp.GetSpans().Snapshots()
	if len(snaps) != 2 {
		t.Fatalf("spans = %d, want 2", len(snaps))
	}
	for _, s := range snaps {
		if s.Status().Code != codes.Error || !slices.Contains(eventNames(s), "exception") || s.EndTime().IsZero() {
			t.Errorf("%s: status %v events %v ended %v", s.Name(), s.Status().Code, eventNames(s), !s.EndTime().IsZero())
		}
	}
}

// The judged state and Jev's answers reach spans only when content capture
// is on. The setting is resolved when the handler runs.
func TestCaptureContent(t *testing.T) {
	for name, tc := range map[string]struct {
		opt       aiserver.ContentCapture
		configure aiserver.ContentCapture // applied after NewHandler; CaptureDefault = not called
		want      bool
	}{
		"unset follows the default, on":         {aiserver.CaptureDefault, aiserver.CaptureDefault, true},
		"explicit off":                          {aiserver.CaptureOff, aiserver.CaptureDefault, false},
		"unset reads Configure at run time":     {aiserver.CaptureDefault, aiserver.CaptureOff, false},
		"explicit on wins over a Configure off": {aiserver.CaptureOn, aiserver.CaptureOff, true},
	} {
		t.Run(name, func(t *testing.T) {
			newJev(t, ok(threeAnswers, 1, 1))
			exp := tracetest.NewInMemoryExporter()
			installClient(t, &fakeLD{}, exp)
			h := must(NewHandler(Options{CaptureContent: tc.opt}))
			if tc.configure != aiserver.CaptureDefault {
				if err := aiserver.Configure(aiserver.SDKOptions{CaptureContent: tc.configure}); err != nil {
					t.Fatal(err)
				}
			}
			if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: configOf(t, judgeVariation("jev-latest", classifiers)),
				UserInput: "assistant output", Variables: map[string]any{"input": "the secret question"}}); err != nil {
				t.Fatal(err)
			}
			var state, answers bool
			for _, s := range exp.GetSpans().Snapshots() {
				for _, v := range spanText(s) {
					state = state || strings.Contains(v, "the secret question")
					answers = answers || strings.Contains(v, "Harmful")
				}
			}
			if state != tc.want || answers != tc.want {
				t.Errorf("state on spans %v, answers on spans %v; want %v", state, answers, tc.want)
			}
		})
	}
}

// Through the core: a judge variation whose provider is TypeSafe selects this
// handler, and its labels come back as <judge>.<label> results.
func TestThroughJudge(t *testing.T) {
	jev := newJev(t, ok(threeAnswers, 10, 4))
	parentVariation := map[string]any{
		"model": map[string]any{"name": "test-model"}, "provider": map[string]any{"name": "P"},
		"instructions":       "be helpful",
		"judgeConfiguration": map[string]any{"judges": []any{map[string]any{"key": "jev-judge", "samplingRate": 1.0}}},
		"_ldMeta":            map[string]any{"enabled": true, "variationKey": "v1", "version": 1},
	}
	ld := &fakeLD{variations: map[string]map[string]any{"cfg": parentVariation, "jev-judge": judgeVariation("jev-latest", classifiers)}}
	installClient(t, ld, tracetest.NewInMemoryExporter())
	parent := aiserver.CreateHandler(aiserver.ProvidesFor{Provider: "P", Mode: aiserver.ModeMessages},
		func(context.Context, aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
			return aiserver.HandlerResult{Output: "answer"}, nil
		}, nil, aiserver.CaptureDefault)
	resp, err := aiserver.Config(aiserver.ConfigArgs{Key: "cfg", Handlers: []*aiserver.ProviderHandler{parent, must(NewHandler(Options{}))}}).
		Invoke(bg, "q", ldcontext.New("u-1"))
	if err != nil {
		t.Fatal(err)
	}
	reqs := jev.reqs()
	if len(reqs) != 1 {
		t.Fatalf("TypeSafe requests = %d, want 1", len(reqs))
	}
	if s, _ := reqs[0].body["state"].(map[string]any); s["output"] != "answer" || s["input"] != "q" {
		t.Errorf("state = %v, want the parent's answer as output and its input", s)
	}
	keys := slices.Sorted(func(yield func(string) bool) {
		for k := range resp.JudgeResults {
			if !yield(k) {
				return
			}
		}
	})
	if !reflect.DeepEqual(keys, []string{"jev-judge.accuracy", "jev-judge.route", "jev-judge.tone"}) {
		t.Fatalf("judge results = %v", resp.JudgeResults)
	}
	for k, want := range map[string]float64{"jev-judge.tone": 0.25, "jev-judge.accuracy": 0.5, "jev-judge.route": 0.8} {
		if got := resp.JudgeResults[k].Score; math.Abs(got-want) > 1e-9 {
			t.Errorf("%s score = %v, want %v", k, got, want)
		}
	}
	if got := ld.events("$ld:ai:judge:jev:tone"); len(got) != 1 || got[0].data["judgeConfigKey"] != "jev-judge.tone" {
		t.Errorf("tone track = %+v", got)
	}
}

// NewHandler reports one usage event however often it is called, and init
// registers the package for $ld:ai:sdk:info.
func TestSdkUsage(t *testing.T) {
	ld := &fakeLD{}
	installClient(t, ld, tracetest.NewInMemoryExporter())
	must(NewHandler(Options{}))
	must(NewHandler(Options{}))
	usage := ld.events("$ld:ai:sdk:usage")
	if len(usage) != 1 || usage[0].data["helper"] != "typesafe.createTypesafeHandler" {
		t.Fatalf("usage events = %+v, want one typesafe.createTypesafeHandler", usage)
	}
	u := usage[0]
	if u.data["helperPackageName"] != PackageName || u.data["helperPackageVersion"] != PackageVersion ||
		u.data["aiSdkName"] != aiserver.PackageName || u.data["aiSdkLanguage"] != "go" || u.value != 1 ||
		u.ctx.Kind() != "ld_ai" || !u.ctx.Anonymous() {
		t.Errorf("usage event = %+v", u)
	}
	var info bool
	for _, tr := range ld.events("$ld:ai:sdk:info") {
		info = info || (tr.data["aiSdkName"] == PackageName && tr.data["aiSdkVersion"] == PackageVersion)
	}
	if !info {
		t.Error("init() must register the package for $ld:ai:sdk:info")
	}
}

// The reduced telemetry a TypeSafe judge call writes: one chat span carrying
// the provider id, the response model, usage and, with capture on, the state
// as one user prompt and the answer as the completion.
func TestChatSpan(t *testing.T) {
	run := func(t *testing.T, model string, capture aiserver.ContentCapture, rp reply) (*fakeJev, aiserver.HandlerResult, *tracetest.InMemoryExporter) {
		t.Helper()
		jev := newJev(t, rp)
		exp := tracetest.NewInMemoryExporter()
		installClient(t, &fakeLD{}, exp)
		res, err := must(NewHandler(Options{CaptureContent: capture})).Invoke(bg, aiserver.HandlerRequest{
			Config: configOf(t, judgeVariation(model, classifiers)), UserInput: "assistant output",
			Variables: map[string]any{"input": "the question", "__ld": ldFixture}})
		if err != nil {
			t.Fatal(err)
		}
		return jev, res, exp
	}
	chatOf := func(t *testing.T, exp *tracetest.InMemoryExporter, name string) sdktrace.ReadOnlySpan {
		t.Helper()
		spans := spansByName(exp)
		var chats []string
		for n := range spans {
			if strings.HasPrefix(n, "chat") {
				chats = append(chats, n)
			}
		}
		if len(chats) != 1 || len(spans[name]) != 1 {
			t.Fatalf("chat spans = %v, want one %q", chats, name)
		}
		return spans[name][0]
	}

	t.Run("provider id, response model and usage", func(t *testing.T) {
		_, _, exp := run(t, "jev-latest", aiserver.CaptureDefault, ok(threeAnswers, 100, 20))
		c := attrs(chatOf(t, exp, "chat jev-latest"))
		for k, v := range map[string]string{"gen_ai.system": "typesafe", "gen_ai.response.model": "jev-2026-01-01",
			"gen_ai.usage.input_tokens": "100", "gen_ai.usage.output_tokens": "20"} {
			if c[k] != v {
				t.Errorf("chat %s = %q, want %q", k, c[k], v)
			}
		}
	})

	t.Run("capture on: the state is the one user prompt, the answer the completion", func(t *testing.T) {
		jev, res, exp := run(t, "jev-latest", aiserver.CaptureOn, ok(threeAnswers, 1, 1))
		chat := chatOf(t, exp, "chat jev-latest")
		c := attrs(chat)
		if c["gen_ai.prompt.0.role"] != "user" {
			t.Errorf("gen_ai.prompt.0.role = %q, want user", c["gen_ai.prompt.0.role"])
		}
		var onSpan map[string]any
		if err := json.Unmarshal([]byte(c["gen_ai.prompt.0.content"]), &onSpan); err != nil ||
			!reflect.DeepEqual(onSpan, jev.reqs()[0].body["state"]) {
			t.Errorf("gen_ai.prompt.0.content = %q, want the state sent (%v)", c["gen_ai.prompt.0.content"], jev.reqs()[0].body["state"])
		}
		for k := range c {
			if strings.HasPrefix(k, "gen_ai.prompt.") && !strings.HasPrefix(k, "gen_ai.prompt.0.") {
				t.Errorf("extra prompt message %s: the state is a single user prompt", k)
			}
		}
		if p, has := eventAttr(chat, "gen_ai.content.prompt", "gen_ai.prompt"); !has || !strings.Contains(p, "the question") {
			t.Errorf("gen_ai.content.prompt = %q, %v; want the state", p, has)
		}
		out := res.Output.(string)
		if c["gen_ai.completion.0.role"] != "assistant" || c["gen_ai.completion.0.content"] != out {
			t.Errorf("completion = %q %q, want assistant %q", c["gen_ai.completion.0.role"], c["gen_ai.completion.0.content"], out)
		}
		if got, has := eventAttr(chat, "gen_ai.content.completion", "gen_ai.completion"); !has || got != out {
			t.Errorf("gen_ai.content.completion = %q, %v; want the answer", got, has)
		}
	})

	t.Run("capture off: no prompt or completion on any span", func(t *testing.T) {
		_, _, exp := run(t, "jev-latest", aiserver.CaptureOff, ok(threeAnswers, 1, 1))
		for _, s := range exp.GetSpans().Snapshots() {
			for k := range attrs(s) {
				if strings.HasPrefix(k, "gen_ai.prompt.") || strings.HasPrefix(k, "gen_ai.completion.") {
					t.Errorf("%s carries %s with capture off", s.Name(), k)
				}
			}
			if n := eventNames(s); slices.Contains(n, "gen_ai.content.prompt") || slices.Contains(n, "gen_ai.content.completion") {
				t.Errorf("%s content events %v with capture off", s.Name(), n)
			}
		}
	})

	t.Run("an empty model name still names the span chat jev", func(t *testing.T) {
		_, _, exp := run(t, "", aiserver.CaptureDefault, ok(threeAnswers, 1, 1))
		chatOf(t, exp, "chat jev")
	})

	t.Run("no usage reported: none on the root", func(t *testing.T) {
		_, _, exp := run(t, "jev-latest", aiserver.CaptureDefault, reply{body: `{"model":"jev-2026-01-01","answers":` + threeAnswers + `}`})
		chatOf(t, exp, "chat jev-latest")
		for k := range attrs(spansByName(exp)[SpanName][0]) {
			if strings.HasPrefix(k, "gen_ai.usage.") {
				t.Errorf("root carries %s with no usage reported", k)
			}
		}
	})
}

// Context identity follows the shared rules: on the root span and its
// feature_flag event, never on the chat span.
func TestContextIdentity(t *testing.T) {
	newJev(t, ok(threeAnswers, 1, 1))
	exp := tracetest.NewInMemoryExporter()
	v := judgeVariation("jev-latest", classifiers)
	delete(v["_ldMeta"].(map[string]any), "mode")
	installClient(t, &fakeLD{variations: map[string]map[string]any{"cfg": v}}, exp)
	multi := ldcontext.NewMulti(ldcontext.New("u-1"), ldcontext.NewWithKind("org", "o-1"))
	if _, err := aiserver.Config(aiserver.ConfigArgs{Key: "cfg", Handlers: []*aiserver.ProviderHandler{must(NewHandler(Options{}))}}).
		Invoke(bg, "q", multi); err != nil {
		t.Fatal(err)
	}
	snaps := exp.GetSpans().Snapshots()
	if len(snaps) != 2 {
		t.Fatalf("spans = %d, want invoke_agent and one chat", len(snaps))
	}
	for _, s := range snaps {
		var keys []string
		for k := range attrs(s) {
			if strings.HasPrefix(k, "context.contextKeys.") {
				keys = append(keys, k)
			}
		}
		if (s.Name() == SpanName) != (len(keys) == 2) {
			t.Errorf("%s context keys = %v", s.Name(), keys)
		}
		if s.Name() != SpanName {
			continue
		}
		id, _ := eventAttr(s, "feature_flag", "feature_flag.context.id")
		ck, _ := eventAttr(s, "feature_flag", "feature_flag.contextKeys")
		if id != multi.FullyQualifiedKey() || ck != `{"org":"o-1","user":"u-1"}` {
			t.Errorf("feature_flag context id %q keys %q", id, ck)
		}
	}
}
