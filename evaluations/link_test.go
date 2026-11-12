package evaluations

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
	"testing"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/launchdarkly/go-sdk-common/v3/ldvalue"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

// servedFlags is the LaunchDarkly client InspectConfig reads. Evaluations
// tests stub their own client for events, so this one is installed on the
// aiserver singleton for the served-variation cases only.
type servedFlags struct {
	mu   sync.Mutex
	vars map[string]any
	ctxs []ldcontext.Context
}

func (s *servedFlags) JSONVariation(key string, c ldcontext.Context, def ldvalue.Value) (ldvalue.Value, error) {
	s.mu.Lock()
	s.ctxs = append(s.ctxs, c)
	raw, ok := s.vars[key]
	s.mu.Unlock()
	if !ok {
		return def, aiserver.ErrConfigNotFound
	}
	b, _ := json.Marshal(raw)
	return ldvalue.Parse(b), nil
}

func (s *servedFlags) TrackMetric(string, ldcontext.Context, float64, ldvalue.Value) error {
	return nil
}
func (s *servedFlags) Flush()       {}
func (s *servedFlags) Close() error { return nil }

func installServedFlags(t *testing.T, vars map[string]any) *servedFlags {
	t.Helper()
	_ = aiserver.Shutdown(context.Background())
	f := &servedFlags{vars: vars}
	if _, err := aiserver.InitWithClient(context.Background(), f, aiserver.InitOptions{}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = aiserver.Shutdown(context.Background()) })
	return f
}

func servedFlag(variationKey string, version any) map[string]any {
	meta := map[string]any{"enabled": true, "variationKey": variationKey, "mode": "completion"}
	if version != nil {
		meta["version"] = version
	}
	return map[string]any{
		"model":        map[string]any{"name": "m"},
		"provider":     map[string]any{"name": "OpenAI"},
		"instructions": "from the flag",
		"_ldMeta":      meta,
	}
}

func servedArgs() RunOptions {
	a := baseArgs(handler("OpenAI", aiserver.ModeMessages, nil))
	a.Generation = nil
	a.AIConfig = &AIConfigRef{Key: "cfg"}
	a.Overrides = &GenerationOverrides{Provider: "OpenAI"}
	return a
}

func routeServedVariation(e *fakeEnv) {
	e.route("GET", "projects/proj/ai-configs/cfg", resp{body: map[string]any{"mode": "completion"}})
	newest := map[string]any{
		"version":  9,
		"messages": []any{map[string]any{"role": "system", "content": "NEWEST"}},
		"model":    map[string]any{"modelName": "gpt-newest"},
	}
	served := map[string]any{
		"version":  7,
		"messages": []any{map[string]any{"role": "system", "content": "SERVED"}},
		"model":    map[string]any{"modelName": "gpt-served"},
	}
	e.route("GET", "projects/proj/ai-configs/cfg/variations/concise", resp{body: map[string]any{"items": []any{newest, served}}})
}

func TestServedVariationUsesMetaVersionNotNewest(t *testing.T) {
	e := newEnv(t, 1)
	routeServedVariation(e)
	flags := installServedFlags(t, map[string]any{"cfg": servedFlag("concise", 7)})
	args := servedArgs()
	args.Overrides.Model = "gpt-override"
	mustRun(t, e.module(), args)
	if len(flags.ctxs) != 1 || flags.ctxs[0].Kind() != "evaluation" || flags.ctxs[0].Key() != "eval-key" {
		t.Fatalf("inspect context %+v", flags.ctxs)
	}
	body := e.body("POST", "projects/proj/offline-evaluations").(map[string]any)
	raw := toJSON(body)
	if body["configKey"] != "cfg" || body["variationKey"] != "concise" || body["variationVersion"] != 7.0 {
		t.Fatalf("link %s", raw)
	}
	if body["generationModel"] != "gpt-override" {
		t.Fatalf("override model %v", body["generationModel"])
	}
	if !strings.Contains(raw, "SERVED") || strings.Contains(raw, "NEWEST") {
		t.Fatalf("content %s", raw)
	}
}

func TestServedVariationLinkIsSentOnPatch(t *testing.T) {
	e := newEnv(t, 1)
	routeServedVariation(e)
	installServedFlags(t, map[string]any{"cfg": servedFlag("concise", 7)})
	e.route("GET", "projects/proj/offline-evaluations/eval-key", resp{body: existingEval(3, nil)})
	e.route("PATCH", "projects/proj/offline-evaluations/eval-key", resp{body: existingEval(4, nil)})
	mustRun(t, e.module(), servedArgs())
	body := e.body("PATCH", "projects/proj/offline-evaluations/eval-key").(map[string]any)
	if body["configKey"] != "cfg" || body["variationKey"] != "concise" || body["variationVersion"] != 7.0 {
		t.Fatalf("patch link %s", toJSON(body))
	}
	if _, has := body["name"]; has {
		t.Fatal("PATCH must still omit name")
	}
}

func TestMissingServedVersionErrorsBeforeWrite(t *testing.T) {
	e := newEnv(t, 1)
	installServedFlags(t, map[string]any{"cfg": servedFlag("concise", nil)})
	runErr(t, e.module(), servedArgs(), ErrInvalidConfig, "_ldMeta.version")
	e.assertNoRequests()
}

func TestDisabledServedVariationErrorsBeforeWrite(t *testing.T) {
	e := newEnv(t, 1)
	flag := servedFlag("concise", 7)
	flag["_ldMeta"].(map[string]any)["enabled"] = false
	installServedFlags(t, map[string]any{"cfg": flag})
	runErr(t, e.module(), servedArgs(), ErrInvalidConfig, "not delivered")
	e.assertNoRequests()
}

func TestUndeliveredConfigErrorsBeforeWrite(t *testing.T) {
	e := newEnv(t, 1)
	installServedFlags(t, map[string]any{})
	runErr(t, e.module(), servedArgs(), ErrNotFound, "not delivered", `"cfg"`)
	e.assertNoRequests()
}

func TestWhitespaceVariationIsServed(t *testing.T) {
	e := newEnv(t, 1)
	routeServedVariation(e)
	installServedFlags(t, map[string]any{"cfg": servedFlag("concise", 7)})
	args := servedArgs()
	args.AIConfig.Variation = " "
	mustRun(t, e.module(), args)
	body := e.body("POST", "projects/proj/offline-evaluations").(map[string]any)
	if body["variationKey"] != "concise" || body["variationVersion"] != 7.0 {
		t.Fatalf("link %s", toJSON(body))
	}
}

func TestServedVersionMissingFromListErrorsBeforeWrite(t *testing.T) {
	e := newEnv(t, 1)
	e.route("GET", "projects/proj/ai-configs/cfg", resp{body: map[string]any{"mode": "completion"}})
	e.route("GET", "projects/proj/ai-configs/cfg/variations/concise", resp{body: map[string]any{"items": []any{
		map[string]any{"version": 9, "messages": []any{map[string]any{"role": "system", "content": "NEWEST"}}, "model": map[string]any{"modelName": "gpt-newest"}},
	}}})
	installServedFlags(t, map[string]any{"cfg": servedFlag("concise", 7)})
	runErr(t, e.module(), servedArgs(), ErrInvalidResponse, "has no version 7")
	if e.count("POST ") != 0 || e.count("PATCH ") != 0 {
		t.Fatalf("wrote before the version check: %v", e.sequence())
	}
}
