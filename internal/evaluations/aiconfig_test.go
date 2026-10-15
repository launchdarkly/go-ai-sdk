package evaluations

import (
	"reflect"
	"strings"
	"testing"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

const (
	cfgGET   = "GET projects/proj/ai-configs/cfg"
	varGET   = "GET projects/proj/ai-configs/cfg/variations/v1"
	modelGET = "GET projects/proj/ai-configs/model-configs/OpenAI.gpt-4o?version=3"
)

// aiConfigEnv serves an AI Config "cfg" in mode, whose variation "v1" has
// an older version 1 and the latest version 2 built from latest.
func aiConfigEnv(t *testing.T, mode any, latest map[string]any) *fakeEnv {
	e := newEnv(t, 1)
	cfg := map[string]any{}
	if mode != nil {
		cfg["mode"] = mode
	}
	e.route("GET", "projects/proj/ai-configs/cfg", resp{body: cfg})
	old := map[string]any{"version": 1, "instructions": "OLD", "messages": []any{map[string]any{"role": "system", "content": "OLD"}},
		"model": map[string]any{"modelName": "old-model", "parameters": map[string]any{"old": true}}, "modelConfigKey": "Old.model"}
	latest["version"] = 2
	e.route("GET", "projects/proj/ai-configs/cfg/variations/v1", resp{body: map[string]any{"items": []any{latest, old, "junk"}}})
	e.route("GET", "projects/proj/ai-configs/model-configs/OpenAI.gpt-4o",
		resp{body: map[string]any{"provider": "OpenAI", "id": "gpt-4o", "params": map[string]any{"max_tokens": 100, "temperature": 1}}})
	return e
}

func linked(extra map[string]any) map[string]any {
	v := map[string]any{"modelConfigKey": "OpenAI.gpt-4o", "modelConfigVersion": 3,
		"model": map[string]any{"modelName": "OpenAI.gpt-4o", "parameters": map[string]any{"temperature": 0.5}}}
	for k, val := range extra {
		v[k] = val
	}
	return v
}

func aiArgs(h ...*aiserver.ProviderHandler) RunArgs {
	a := baseArgs(nil)
	a.Handler, a.Handlers, a.Generation = nil, h, nil
	a.AIConfig = &AIConfigRef{Key: "cfg", Variation: "v1"}
	return a
}

var sysMsgs = []any{map[string]any{"role": "system", "content": "SYS"}}

func TestAIConfigHappyPathSequence(t *testing.T) {
	e := aiConfigEnv(t, "completion", linked(map[string]any{"messages": sysMsgs, "tools": []any{map[string]any{"key": "lookup", "version": 4}}}))
	e.route("GET", "projects/proj/ai-tools/lookup", resp{body: map[string]any{"version": 4}})
	var rec recorder
	args := aiArgs(handler("OpenAI", aiserver.ModeMessages, rec.wrap(nil)))
	args.Tools = map[string]aiserver.ToolHandler{"lookup": noopTool}
	mustRun(t, e.module(), args)
	assertSequence(t, e, cfgGET, varGET, modelGET, "GET projects/proj/ai-tools/lookup",
		"GET projects/proj/datasets/ds", rowsPath, evalPath, runPath, summaryPath)
	cfg := rec.all()[0].Config
	// The model config's id wins over modelName; params layer variation over model config.
	if cfg.Model.Name != "gpt-4o" || !reflect.DeepEqual(cfg.Model.Parameters, map[string]any{"max_tokens": 100.0, "temperature": 0.5}) {
		t.Fatalf("model %+v", cfg.Model)
	}
	if strings.Contains(toJSON(e.body("POST", "projects/proj/evaluations")), "OLD") || strings.Contains(toJSON(cfg), "OLD") {
		t.Fatal("an older variation version leaked into the run")
	}
	body := e.body("POST", "projects/proj/evaluations").(map[string]any)
	if _, has := body["mode"]; has {
		t.Fatal("mode must not be sent to LaunchDarkly")
	}
}

func TestAIConfigWithoutModelConfig(t *testing.T) {
	e := aiConfigEnv(t, nil, map[string]any{"messages": sysMsgs, "model": map[string]any{"modelName": "gpt-x"}})
	args := aiArgs(handler("OpenAI", aiserver.ModeMessages, nil))
	args.Overrides = &GenerationOverrides{Provider: "OpenAI"}
	mustRun(t, e.module(), args)
	assertNoURL(t, e, "model-configs")
	t.Run("provider is required", func(t *testing.T) {
		e := aiConfigEnv(t, nil, map[string]any{"messages": sysMsgs, "model": map[string]any{"modelName": "gpt-x"}})
		runErr(t, e.module(), aiArgs(handler("OpenAI", aiserver.ModeMessages, nil)), ErrInvalidArgument, "generation.provider is required")
		assertSequence(t, e, cfgGET, varGET)
	})
	// after the merge the model must be a non-empty string, with the
	// generation source's error, and the run stops before any other request.
	t.Run("model is required", func(t *testing.T) {
		e := aiConfigEnv(t, nil, map[string]any{"messages": sysMsgs, "model": map[string]any{"modelName": ""}})
		args := aiArgs(handler("OpenAI", aiserver.ModeMessages, nil))
		args.Overrides = &GenerationOverrides{Provider: "OpenAI"}
		runErr(t, e.module(), args, ErrInvalidArgument, "generation.model is required")
		assertSequence(t, e, cfgGET, varGET)
	})
}

func TestAIConfigModeChecks(t *testing.T) {
	cases := []struct {
		name  string
		mode  any
		pool  []*aiserver.ProviderHandler
		want  []string
		reqs  []string
		judge bool
	}{
		{"unsupported mode", "chat", nil, []string{`"cfg"`, `"chat"`}, []string{cfgGET}, false},
		{"non-string mode", 3, nil, []string{`"cfg"`, "3"}, []string{cfgGET}, false},
		{"agent config, messages pool", "agent", nil, []string{`"cfg"`, `"agent"`, "[OpenAI, messages]"}, []string{cfgGET}, false},
		{"absent mode is completion", nil, []*aiserver.ProviderHandler{handler("OpenAI", aiserver.ModeAgent, nil)}, []string{`"completion"`, "[OpenAI, agent]"}, []string{cfgGET}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			e := aiConfigEnv(t, tc.mode, linked(map[string]any{"messages": sysMsgs}))
			pool := tc.pool
			if pool == nil {
				pool = []*aiserver.ProviderHandler{handler("OpenAI", aiserver.ModeMessages, nil)}
			}
			runErr(t, e.module(), aiArgs(pool...), ErrInvalidConfig, tc.want...)
			assertSequence(t, e, tc.reqs...)
		})
	}
	t.Run("judge mode selects a messages handler", func(t *testing.T) {
		e := aiConfigEnv(t, "judge", linked(map[string]any{"messages": sysMsgs}))
		mustRun(t, e.module(), aiArgs(handler("OpenAI", aiserver.ModeMessages, nil)))
	})
}

func TestAIConfigNotFound(t *testing.T) {
	cases := []struct {
		name, method, path string
		want               []string
	}{
		{"config", "GET", "projects/proj/ai-configs/cfg", []string{`"cfg"`, `"proj"`}},
		{"variation", "GET", "projects/proj/ai-configs/cfg/variations/v1", []string{`"cfg"`, `"v1"`, `"proj"`}},
		{"model config", "GET", "projects/proj/ai-configs/model-configs/OpenAI.gpt-4o", []string{`"OpenAI.gpt-4o"`, `"proj"`}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			e := aiConfigEnv(t, "completion", linked(map[string]any{"messages": sysMsgs}))
			e.route(tc.method, tc.path, resp{status: 404})
			runErr(t, e.module(), aiArgs(handler("OpenAI", aiserver.ModeMessages, nil)), ErrNotFound, tc.want...)
		})
	}
}

func TestAIConfigVariationShape(t *testing.T) {
	t.Run("no integer version", func(t *testing.T) {
		e := aiConfigEnv(t, "completion", linked(nil))
		e.route("GET", "projects/proj/ai-configs/cfg/variations/v1", resp{body: map[string]any{"items": []any{map[string]any{"version": "2"}}}})
		runErr(t, e.module(), aiArgs(handler("OpenAI", aiserver.ModeMessages, nil)), ErrInvalidResponse, "no versions")
	})
	t.Run("non-string modelConfigKey", func(t *testing.T) {
		e := aiConfigEnv(t, "completion", map[string]any{"modelConfigKey": 7})
		runErr(t, e.module(), aiArgs(handler("OpenAI", aiserver.ModeMessages, nil)), ErrInvalidResponse, `"cfg"/"v1"`, "modelConfigKey")
	})
	t.Run("version query only for an integer pin", func(t *testing.T) {
		e := aiConfigEnv(t, "completion", linked(map[string]any{"messages": sysMsgs, "modelConfigVersion": "3"}))
		mustRun(t, e.module(), aiArgs(handler("OpenAI", aiserver.ModeMessages, nil)))
		if e.count("model-configs/OpenAI.gpt-4o?") != 0 || e.count("model-configs/OpenAI.gpt-4o") != 1 {
			t.Fatalf("sequence %v", e.sequence())
		}
	})
	t.Run("outputFormat copied", func(t *testing.T) {
		of := map[string]any{"type": "object"}
		e := aiConfigEnv(t, "completion", linked(map[string]any{"messages": sysMsgs, "outputFormat": of}))
		var rec recorder
		mustRun(t, e.module(), aiArgs(handler("OpenAI", aiserver.ModeMessages, rec.wrap(nil))))
		if !reflect.DeepEqual(rec.all()[0].Config.OutputFormat, of) {
			t.Fatalf("outputFormat %v", rec.all()[0].Config.OutputFormat)
		}
	})
	t.Run("provider uncovered after the model config read", func(t *testing.T) {
		e := aiConfigEnv(t, "completion", linked(map[string]any{"messages": sysMsgs}))
		runErr(t, e.module(), aiArgs(handler("Anthropic", aiserver.ModeMessages, nil)), ErrInvalidArgument, `"OpenAI"`, "[Anthropic, messages]")
		assertSequence(t, e, cfgGET, varGET, modelGET)
	})
}

func TestAIConfigPromptConversion(t *testing.T) {
	both := map[string]any{"instructions": "INSTR", "messages": sysMsgs}
	cases := []struct {
		name      string
		mode      string
		prompt    map[string]any
		overrides *GenerationOverrides
		pool      aiserver.Mode
		wantInstr string
		wantMsgs  int
		wantErr   []string
	}{
		{"completion with both runs on messages", "completion", both, nil, aiserver.ModeMessages, "", 1, nil},
		{"agent with both runs on instructions", "agent", both, nil, aiserver.ModeAgent, "INSTR", 0, nil},
		{"completion with only instructions throws", "completion", map[string]any{"instructions": "INSTR"}, nil, aiserver.ModeMessages, "", 0,
			[]string{`"cfg"`, `"v1"`, `"completion"`, "needs messages"}},
		{"messages override fixes it", "completion", map[string]any{"instructions": "INSTR"},
			&GenerationOverrides{Messages: []aiserver.Message{{Role: "user", Text: "OVR"}}}, aiserver.ModeMessages, "", 1, nil},
		{"neither field runs with an empty prompt", "completion", map[string]any{}, nil, aiserver.ModeMessages, "", 0, nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			e := aiConfigEnv(t, tc.mode, linked(tc.prompt))
			var rec recorder
			args := aiArgs(handler("OpenAI", tc.pool, rec.wrap(nil)))
			args.Overrides = tc.overrides
			if tc.wantErr != nil {
				runErr(t, e.module(), args, ErrInvalidConfig, tc.wantErr...)
				assertSequence(t, e, cfgGET, varGET)
				return
			}
			mustRun(t, e.module(), args)
			cfg := rec.all()[0].Config
			if cfg.Instructions != tc.wantInstr || len(cfg.Messages) != tc.wantMsgs {
				t.Fatalf("instructions %q messages %v", cfg.Instructions, cfg.Messages)
			}
		})
	}
}

func TestAIConfigOverrides(t *testing.T) {
	e := aiConfigEnv(t, "agent", linked(map[string]any{"instructions": "INSTR"}))
	var rec recorder
	args := aiArgs(handler("Anthropic", aiserver.ModeAgent, rec.wrap(nil)))
	args.Overrides = &GenerationOverrides{Provider: "Anthropic", Model: "claude", Parameters: map[string]any{"temperature": 0.1},
		Instructions: "NEW", PromptSnippets: map[string]string{"s": "v"}}
	mustRun(t, e.module(), args)
	cfg := rec.all()[0].Config
	if cfg.Provider.Name != "Anthropic" || cfg.Model.Name != "claude" || cfg.Instructions != "NEW" ||
		!reflect.DeepEqual(cfg.Model.Parameters, map[string]any{"max_tokens": 100.0, "temperature": 0.1}) {
		t.Fatalf("config %+v", cfg)
	}
}

func TestAIConfigVariationTools(t *testing.T) {
	tools := []any{map[string]any{"key": "lookup", "version": 4}, map[string]any{"key": "refund", "version": 1}}
	t.Run("omitted tools throws naming every variation tool", func(t *testing.T) {
		e := aiConfigEnv(t, "completion", linked(map[string]any{"messages": sysMsgs, "tools": tools}))
		runErr(t, e.module(), aiArgs(handler("OpenAI", aiserver.ModeMessages, nil)), ErrInvalidConfig, "lookup", "refund", "pass Tools")
		assertSequence(t, e, cfgGET, varGET, modelGET)
	})
	t.Run("empty tools runs with none", func(t *testing.T) {
		e := aiConfigEnv(t, "completion", linked(map[string]any{"messages": sysMsgs, "tools": tools}))
		args := aiArgs(handler("OpenAI", aiserver.ModeMessages, nil))
		args.Tools = map[string]aiserver.ToolHandler{}
		mustRun(t, e.module(), args)
		assertNoURL(t, e, "ai-tools")
		if !strings.Contains(e.logs.String(), "lookup") || !strings.Contains(e.logs.String(), "refund") {
			t.Fatalf("missing tool warnings: %s", e.logs)
		}
	})
	t.Run("version drift warns and uses the resolved version", func(t *testing.T) {
		e := aiConfigEnv(t, "completion", linked(map[string]any{"messages": sysMsgs, "tools": tools[:1]}))
		e.route("GET", "projects/proj/ai-tools/lookup", resp{body: map[string]any{"version": 9}})
		args := aiArgs(handler("OpenAI", aiserver.ModeMessages, nil))
		args.Tools = map[string]aiserver.ToolHandler{"lookup": noopTool}
		mustRun(t, e.module(), args)
		logs := e.logs.String()
		if !strings.Contains(logs, "pinnedVersion=4") || !strings.Contains(logs, "resolvedVersion=9") {
			t.Fatalf("drift warning: %s", logs)
		}
		if !strings.Contains(toJSON(e.body("POST", "projects/proj/evaluations")), `"version":9`) {
			t.Fatal("run must use the resolved version")
		}
	})
}

func TestAIConfigVariationJudges(t *testing.T) {
	judges := map[string]any{"judges": []any{map[string]any{"judgeConfigKey": "accuracy"}}}
	t.Run("default criteria fail when the judge fails to resolve", func(t *testing.T) {
		e := aiConfigEnv(t, "completion", linked(map[string]any{"messages": sysMsgs, "judgeConfiguration": judges}))
		runErr(t, e.module(), aiArgs(handler("OpenAI", aiserver.ModeMessages, nil)), ErrJudgeResolution, `"accuracy"`)
	})
	t.Run("explicit empty criteria never resolves it", func(t *testing.T) {
		e := aiConfigEnv(t, "completion", linked(map[string]any{"messages": sysMsgs, "judgeConfiguration": judges}))
		args := aiArgs(handler("OpenAI", aiserver.ModeMessages, nil))
		args.Criteria = []Criterion{}
		mustRun(t, e.module(), args)
		if len(e.judgeCalls) != 0 {
			t.Fatal("judge resolved despite Criteria = []")
		}
	})
	t.Run("default criteria use default thresholds in order", func(t *testing.T) {
		two := map[string]any{"judges": []any{map[string]any{"judgeConfigKey": "b"}, map[string]any{"judgeConfigKey": "a"}}}
		e := aiConfigEnv(t, "completion", linked(map[string]any{"messages": sysMsgs, "judgeConfiguration": two}))
		e.judges["a"], e.judges["b"] = judgeVariation("OpenAI", "completion", nil), judgeVariation("OpenAI", "completion", nil)
		mustRun(t, e.module(), aiArgs(handler("OpenAI", aiserver.ModeMessages, verdict(0.9, "ok"))))
		crit := e.body("POST", "projects/proj/evaluations").(map[string]any)["criteria"].([]any)
		if len(crit) != 2 || crit[0].(map[string]any)["judgeKey"] != "b" ||
			!reflect.DeepEqual(crit[1].(map[string]any)["options"], map[string]any{"threshold": 0.5, "passRateThreshold": 0.95}) {
			t.Fatalf("criteria %s", toJSON(crit))
		}
	})
}
