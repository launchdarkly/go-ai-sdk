package aiserver

import (
	"bytes"
	"context"
	"errors"
	"log/slog"
	"slices"
	"strings"
	"testing"
)

// scoped is a "*" handler of mode that answers out, scoped to providers
// (nil = unscoped). It sets the field directly so the routing tests do not
// depend on WithProviders.
func scoped(mode Mode, out string, providers ...string) *ProviderHandler {
	h := stubHandler("*", mode, out, nil, nil)
	h.Providers = providers
	return h
}

func captureLog(t *testing.T) *bytes.Buffer {
	t.Helper()
	var buf bytes.Buffer
	SetLogger(slog.New(slog.NewTextHandler(&buf, nil)))
	t.Cleanup(func() { SetLogger(slog.Default()) })
	return &buf
}

func TestWithProviders(t *testing.T) {
	base := stubHandler("*", ModeMessages, "x", nil, nil)
	names := []string{"Bedrock", "Anthropic", "Bedrock"}
	h, err := base.WithProviders(names...)
	if err != nil {
		t.Fatalf("WithProviders: %v", err)
	}
	if !slices.Equal(h.Providers, []string{"Bedrock", "Anthropic"}) {
		t.Errorf("Providers = %v, want each name once in first-seen order", h.Providers)
	}
	if h == base || base.Providers != nil {
		t.Error("WithProviders must return a copy and leave the receiver unscoped")
	}
	if h.ProvidesFor != base.ProvidesFor || h.Invoke == nil {
		t.Errorf("copy lost ProvidesFor or Invoke: %+v", h)
	}
	names[0] = "Gemini"
	if h.Providers[0] != "Bedrock" {
		t.Error("the handler must own its list: mutating the caller's slice changed it")
	}
	again, err := h.WithProviders("OpenAI")
	if err != nil || !slices.Equal(again.Providers, []string{"OpenAI"}) || !slices.Equal(h.Providers, []string{"Bedrock", "Anthropic"}) {
		t.Errorf("rescoping: %v %v (original %v)", again, err, h.Providers)
	}

	for name, bad := range map[string][]string{"empty": {}, "nil": nil, "blank": {""}, "padded": {" Bedrock"}, "trailing space": {"Bedrock "}, "one bad of two": {"Bedrock", ""}} {
		if got, err := base.WithProviders(bad...); !errors.Is(err, ErrInvalidProviders) || got != nil || !strings.Contains(err.Error(), "providers") {
			t.Errorf("%s: got %v, %v; want nil and ErrInvalidProviders naming providers", name, got, err)
		}
	}
}

func TestRegistryScopedWildcardIdentity(t *testing.T) {
	t.Run("same set in another order or with repeats collides and keeps the slot", func(t *testing.T) {
		buf := captureLog(t)
		first := scoped(ModeAgent, "1", "Bedrock", "Anthropic")
		other := scoped(ModeAgent, "other", "Gemini")
		second := scoped(ModeAgent, "2", "Anthropic", "Bedrock", "Anthropic")
		r := NewRegistry(RegistryEntries{Handlers: []*ProviderHandler{first, other, second}})
		if hs := r.Handlers(); len(hs) != 2 || hs[0] != second || hs[1] != other {
			t.Fatalf("handlers = %v, want [second other]", hs)
		}
		log := buf.String()
		if strings.Count(log, "level=WARN") != 1 {
			t.Fatalf("want one warning, got: %s", log)
		}
		if !strings.Contains(log, "Anthropic,Bedrock") && !strings.Contains(log, "[Anthropic Bedrock]") {
			t.Errorf("warning must name the provider set in sorted order: %s", log)
		}
	})
	t.Run("a repeated name is the same set", func(t *testing.T) {
		captureLog(t)
		a, b := scoped(ModeAgent, "a", "Bedrock"), scoped(ModeAgent, "b", "Bedrock", "Bedrock")
		if hs := NewRegistry(RegistryEntries{Handlers: []*ProviderHandler{a, b}}).Handlers(); len(hs) != 1 || hs[0] != b {
			t.Errorf("handlers = %v, want [b]", hs)
		}
	})
	t.Run("replacement keeps the original slot", func(t *testing.T) {
		captureLog(t)
		bg := scoped(ModeAgent, "bg", "Bedrock", "Gemini")
		ba := scoped(ModeAgent, "ba", "Bedrock", "Anthropic")
		gb := scoped(ModeAgent, "gb", "Gemini", "Bedrock")
		if hs := NewRegistry(RegistryEntries{Handlers: []*ProviderHandler{bg, ba, gb}}).Handlers(); len(hs) != 2 || hs[0] != gb || hs[1] != ba {
			t.Errorf("handlers = %v, want [gb ba]", hs)
		}
	})
	t.Run("an exact handler replaced in place keeps its slot", func(t *testing.T) {
		captureLog(t)
		a1, b, a2 := stubHandler("A", ModeMessages, "1", nil, nil), stubHandler("B", ModeMessages, "b", nil, nil), stubHandler("A", ModeMessages, "2", nil, nil)
		if hs := NewRegistry(RegistryEntries{Handlers: []*ProviderHandler{a1, b, a2}}).Handlers(); len(hs) != 2 || hs[0] != a2 || hs[1] != b {
			t.Errorf("handlers = %v, want [a2 b]", hs)
		}
	})
	t.Run("different sets and an unscoped wildcard all stay", func(t *testing.T) {
		buf := captureLog(t)
		bed, anth, any := scoped(ModeAgent, "b", "Bedrock"), scoped(ModeAgent, "a", "Anthropic"), scoped(ModeAgent, "any")
		if hs := NewRegistry(RegistryEntries{Handlers: []*ProviderHandler{bed, anth, any}}).Handlers(); len(hs) != 3 {
			t.Errorf("handlers = %v, want all three", hs)
		}
		if strings.Contains(buf.String(), "level=WARN") {
			t.Errorf("no warning expected: %s", buf.String())
		}
	})
	t.Run("two unscoped wildcards still collide", func(t *testing.T) {
		captureLog(t)
		a, b := scoped(ModeAgent, "a"), scoped(ModeAgent, "b")
		if hs := NewRegistry(RegistryEntries{Handlers: []*ProviderHandler{a, b}}).Handlers(); len(hs) != 1 || hs[0] != b {
			t.Errorf("handlers = %v, want [b]", hs)
		}
	})
	t.Run("a list on an exact handler is not part of its identity", func(t *testing.T) {
		captureLog(t)
		a, b := stubHandler("OpenAI", ModeMessages, "a", nil, nil), stubHandler("OpenAI", ModeMessages, "b", nil, nil)
		a.Providers = []string{"Bedrock"}
		if hs := NewRegistry(RegistryEntries{Handlers: []*ProviderHandler{a, b}}).Handlers(); len(hs) != 1 || hs[0] != b {
			t.Errorf("handlers = %v, want [b]", hs)
		}
	})
}

func TestScopedWildcardRouting(t *testing.T) {
	exactOpenAI := stubHandler("OpenAI", ModeMessages, "exact", nil, nil)
	cases := []struct {
		name     string
		handlers []*ProviderHandler
		provider string
		mode     Mode
		want     string
	}{
		{"a listed provider selects that wildcard", []*ProviderHandler{scoped(ModeMessages, "bedrock", "Bedrock")}, "Bedrock", ModeMessages, "bedrock"},
		{"two scoped wildcards divide the providers: Bedrock", []*ProviderHandler{scoped(ModeMessages, "bedrock", "Bedrock"), scoped(ModeMessages, "anthropic", "Anthropic")}, "Bedrock", ModeMessages, "bedrock"},
		{"two scoped wildcards divide the providers: Anthropic", []*ProviderHandler{scoped(ModeMessages, "bedrock", "Bedrock"), scoped(ModeMessages, "anthropic", "Anthropic")}, "Anthropic", ModeMessages, "anthropic"},
		{"exact provider match still wins", []*ProviderHandler{scoped(ModeMessages, "scoped", "OpenAI"), exactOpenAI}, "OpenAI", ModeMessages, "exact"},
		{"scoped beats an earlier unscoped wildcard", []*ProviderHandler{scoped(ModeMessages, "any"), scoped(ModeMessages, "bedrock", "Bedrock")}, "Bedrock", ModeMessages, "bedrock"},
		{"an unscoped wildcard is the fallback", []*ProviderHandler{scoped(ModeMessages, "bedrock", "Bedrock"), scoped(ModeMessages, "any")}, "OpenAI", ModeMessages, "any"},
		{"the shorter list wins", []*ProviderHandler{scoped(ModeMessages, "two", "Bedrock", "Anthropic"), scoped(ModeMessages, "one", "Bedrock")}, "Bedrock", ModeMessages, "one"},
		{"length counts distinct names", []*ProviderHandler{scoped(ModeMessages, "two", "Bedrock", "Anthropic"), scoped(ModeMessages, "one", "Bedrock", "Bedrock")}, "Bedrock", ModeMessages, "one"},
		{"equal-length overlap keeps the earlier handler", []*ProviderHandler{scoped(ModeMessages, "first", "Bedrock", "Gemini"), scoped(ModeMessages, "second", "Bedrock", "Anthropic")}, "Bedrock", ModeMessages, "first"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h, err := kitSelectHandler(tc.handlers, tc.provider, tc.mode)
			if err != nil {
				t.Fatalf("kitSelectHandler: %v", err)
			}
			if res, _ := h.Invoke(bg, HandlerRequest{}); res.Output != tc.want {
				t.Errorf("selected %q, want %q", res.Output, tc.want)
			}
		})
	}

	t.Run("an unlisted provider is the plain not-found error", func(t *testing.T) {
		_, err := kitSelectHandler([]*ProviderHandler{scoped(ModeMessages, "bedrock", "Bedrock")}, "OpenAI", ModeMessages)
		if !errors.Is(err, ErrNoHandler) || err.Error() != ErrNoHandler.Error()+" OpenAI" {
			t.Errorf("err = %v, want exactly %q", err, ErrNoHandler.Error()+" OpenAI")
		}
	})
	t.Run("mode still has to match", func(t *testing.T) {
		if h, err := kitSelectHandler([]*ProviderHandler{scoped(ModeMessages, "bedrock", "Bedrock")}, "Bedrock", ModeAgent); err == nil {
			t.Errorf("selected %v for an agent variation", h)
		}
	})
	t.Run("through Config the unlisted handler is not called", func(t *testing.T) {
		withFake(t, map[string]any{"oa": enabledConfig("OpenAI", "", nil)})
		called := false
		h := createHandler(ProvidesFor{"*", ModeMessages}, func(context.Context, HandlerRequest) (HandlerResult, error) {
			called = true
			return HandlerResult{Output: "x"}, nil
		}, nil, CaptureDefault)
		h.Providers = []string{"Bedrock"}
		_, err := Config(ConfigArgs{Key: "oa", Handlers: []*ProviderHandler{h}}).Invoke(bg, "", userCtx)
		if err == nil || called || err.Error() != ErrNoHandler.Error()+" OpenAI" {
			t.Errorf("err = %v, called = %v", err, called)
		}
	})
}

func TestScopedWildcardGraphNode(t *testing.T) {
	node := func(provider string) *GraphNode {
		n := &GraphNode{Key: "n", Meta: VariationMeta{Mode: "agent"}}
		n.Config.Provider.Name = provider
		return n
	}
	t.Run("the sole-handler fallback does not bypass the list", func(t *testing.T) {
		_, err := selectNodeHandler(node("OpenAI"), []*ProviderHandler{scoped(ModeAgent, "bedrock", "Bedrock")})
		if !errors.Is(err, ErrNoHandler) || !strings.Contains(err.Error(), "OpenAI") {
			t.Errorf("err = %v, want ErrNoHandler naming OpenAI", err)
		}
	})
	t.Run("the sole unscoped handler still runs any node", func(t *testing.T) {
		only := stubHandler("Anthropic", ModeAgent, "only", nil, nil)
		if h, err := selectNodeHandler(node("OpenAI"), []*ProviderHandler{only}); err != nil || h != only {
			t.Errorf("got %v, %v", h, err)
		}
	})
	t.Run("exact, then the shortest scoped list, then unscoped", func(t *testing.T) {
		any, two, one := scoped(ModeAgent, "any"), scoped(ModeAgent, "two", "Bedrock", "Anthropic"), scoped(ModeAgent, "one", "Bedrock")
		exact := stubHandler("Bedrock", ModeAgent, "exact", nil, nil)
		for _, tc := range []struct {
			hs   []*ProviderHandler
			want *ProviderHandler
		}{
			{[]*ProviderHandler{any, two, one, exact}, exact},
			{[]*ProviderHandler{any, two, one}, one},
			{[]*ProviderHandler{any, two}, two},
		} {
			if h, err := selectNodeHandler(node("Bedrock"), tc.hs); err != nil || h != tc.want {
				t.Errorf("selected %v (%v), want %v", h, err, tc.want)
			}
		}
		if h, err := selectNodeHandler(node("OpenAI"), []*ProviderHandler{two, any}); err != nil || h != any {
			t.Errorf("OpenAI node: selected %v (%v), want the unscoped wildcard", h, err)
		}
	})
}

func TestScopedWildcardJudgeSelection(t *testing.T) {
	bedrock := scoped(ModeMessages, "bedrock", "Bedrock")
	t.Run("pickJudgeHandler", func(t *testing.T) {
		if _, _, ok := pickJudgeHandler(nil, []*ProviderHandler{bedrock}, "Anthropic", ModeMessages); ok {
			t.Error("a scoped wildcard must not run an unlisted judge")
		}
		if h, _, ok := pickJudgeHandler(nil, []*ProviderHandler{bedrock}, "Bedrock", ModeMessages); !ok || h != bedrock {
			t.Errorf("listed judge: %v %v", h, ok)
		}
		if _, _, ok := pickJudgeHandler(bedrock, nil, "Anthropic", ModeMessages); ok {
			t.Error("a scoped parent must not cover an unlisted judge")
		}
		if _, _, ok := pickJudgeHandler(bedrock, []*ProviderHandler{}, "Anthropic", ModeMessages); ok {
			t.Error("an empty handlers list still checks the parent against the judge provider")
		}
		agentScoped := scoped(ModeAgent, "agent", "Bedrock")
		if _, _, ok := pickJudgeHandler(nil, []*ProviderHandler{agentScoped}, "Anthropic", ModeMessages); ok {
			t.Error("the agent fallback must honour the list too")
		}
		unscoped := scoped(ModeMessages, "any")
		if h, _, ok := pickJudgeHandler(nil, []*ProviderHandler{unscoped, bedrock}, "Bedrock", ModeMessages); !ok || h != bedrock {
			t.Errorf("scoped must beat unscoped for its listed provider: got %v", h)
		}
		if h, _, ok := pickJudgeHandler(nil, []*ProviderHandler{unscoped, bedrock}, "Anthropic", ModeMessages); !ok || h != unscoped {
			t.Errorf("unscoped still covers any provider: got %v", h)
		}
		two := scoped(ModeMessages, "two", "Bedrock", "Anthropic")
		if h, _, ok := pickJudgeHandler(nil, []*ProviderHandler{two, bedrock}, "Bedrock", ModeMessages); !ok || h != bedrock {
			t.Errorf("the shorter list wins: got %v", h)
		}
	})
	t.Run("RunJudge skips an unlisted judge without calling the handler", func(t *testing.T) {
		called := false
		h := createHandler(ProvidesFor{"*", ModeMessages}, func(context.Context, HandlerRequest) (HandlerResult, error) {
			called = true
			return HandlerResult{Output: `{"score":1}`}, nil
		}, nil, CaptureDefault)
		h.Providers = []string{"Bedrock"}
		res, err := RunJudge(bg, JudgeTask{ConfigKey: "judge", JudgeProvider: "Anthropic", JudgeMode: ModeMessages}, []*ProviderHandler{h})
		if res != nil || err != nil || called {
			t.Errorf("res=%v err=%v called=%v", res, err, called)
		}
	})
	t.Run("Config skips the judge with a warning naming the judge and provider", func(t *testing.T) {
		buf := captureLog(t)
		parent := enabledConfig("Bedrock", "", judgedConfig(nil))
		withFake(t, map[string]any{"cfg": parent, "judge": enabledConfig("Anthropic", "judge", nil)})
		judged := 0
		h := createHandler(ProvidesFor{"*", ModeMessages}, func(_ context.Context, req HandlerRequest) (HandlerResult, error) {
			if req.Variables["response_to_evaluate"] != nil {
				judged++
			}
			return HandlerResult{Output: "answer"}, nil
		}, nil, CaptureDefault)
		h.Providers = []string{"Bedrock"}
		args := ConfigArgs{Key: "cfg", Handlers: []*ProviderHandler{h}}
		if _, err := Config(args).Invoke(bg, "q", userCtx); err != nil {
			t.Fatal(err)
		}
		if judged != 0 {
			t.Errorf("the scoped handler ran an Anthropic judge %d time(s)", judged)
		}
		log := buf.String()
		if !strings.Contains(log, "level=WARN") || !strings.Contains(log, "judge") || !strings.Contains(log, "Anthropic") {
			t.Errorf("want a warning naming judge key and provider, got: %s", log)
		}
		args.SkipJudges = true
		resp, err := Config(args).Invoke(bg, "q", userCtx)
		if err != nil || len(resp.JudgeTasks) != 0 {
			t.Errorf("tasks = %v (err %v), want the Anthropic judge omitted", resp.JudgeTasks, err)
		}
	})
	t.Run("a judge skipped for an exact handler warns too", func(t *testing.T) {
		buf := captureLog(t)
		withFake(t, map[string]any{"cfg": enabledConfig("OpenAI", "", judgedConfig(nil)), "judge": enabledConfig("Anthropic", "judge", nil)})
		if _, err := Config(ConfigArgs{Key: "cfg", Handlers: []*ProviderHandler{stubHandler("OpenAI", ModeMessages, "a", nil, nil)}}).Invoke(bg, "q", userCtx); err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(buf.String(), "level=WARN") || !strings.Contains(buf.String(), "Anthropic") {
			t.Errorf("want a skip warning, got: %s", buf.String())
		}
	})
}
