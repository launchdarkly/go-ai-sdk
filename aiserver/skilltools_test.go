package aiserver

import (
	"bytes"
	"context"
	"log/slog"
	"maps"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/launchdarkly/go-ai-sdk/internal/bridge"
)

// TESTING.md §3.21a "Offering skill tools to the provider": core adds the
// tools of the skills a variation attaches to the config.tools a handler
// receives. The skills package supplies them through bridge.SkillTools; these
// tests install a fake. Go's Tools is a map, so the order bullets do not apply
// (A.12).

func skillTool(name, desc string) Tool {
	return Tool{Name: name, Type: "function", Description: desc, Parameters: map[string]any{"type": "object"}}
}

type skillPin struct {
	key     string
	version int
}

func withSkillToolsFake(t *testing.T, byPin map[skillPin][]Tool) *[]skillPin {
	t.Helper()
	var asked []skillPin
	prev := bridge.SkillTools
	bridge.SkillTools = func(key string, version int) []Tool {
		asked = append(asked, skillPin{key, version})
		return byPin[skillPin{key, version}]
	}
	t.Cleanup(func() { bridge.SkillTools = prev; resetSkillToolWarnings() })
	resetSkillToolWarnings()
	return &asked
}

func captureLogs(t *testing.T) *bytes.Buffer {
	t.Helper()
	var buf bytes.Buffer
	SetLogger(slog.New(slog.NewTextHandler(&buf, nil)))
	t.Cleanup(func() { SetLogger(nil) })
	return &buf
}

func skillRefs(pins ...skillPin) []any {
	out := []any{}
	for _, p := range pins {
		out = append(out, map[string]any{"key": p.key, "version": float64(p.version)})
	}
	return out
}

func parsedWithSkills(t *testing.T, tools map[string]any, pins ...skillPin) AiConfig {
	t.Helper()
	raw := enabledConfig("P", "", map[string]any{"skills": skillRefs(pins...)})
	delete(raw, "_ldMeta")
	if tools != nil {
		raw["tools"] = tools
	}
	cfg, err := kitParseAiConfig(raw)
	if err != nil {
		t.Fatal(err)
	}
	return cfg
}

func ownTools(names ...string) map[string]any { return toolDefs(names...)["tools"].(map[string]any) }

func keys(m map[string]Tool) []string { return slices.Sorted(maps.Keys(m)) }

var s1, s2 = skillPin{"s1", 1}, skillPin{"s2", 1}

func TestSkillToolsAreAddedToConfigTools(t *testing.T) {
	asked := withSkillToolsFake(t, map[skillPin][]Tool{s2: {skillTool("c", "c")}, s1: {skillTool("d", "d")}})
	cfg := withSkillTools("cfg", parsedWithSkills(t, ownTools("a", "b"), s2, s1))
	if got := keys(cfg.Tools); !slices.Equal(got, []string{"a", "b", "c", "d"}) {
		t.Fatalf("tools = %v", got)
	}
	if cfg.Tools["c"].Description != "c" {
		t.Errorf("skill tool definition not kept: %+v", cfg.Tools["c"])
	}
	if !slices.Equal(*asked, []skillPin{s2, s1}) {
		t.Errorf("skills read = %v, want the variation's skills order", *asked)
	}
}

func TestEachSkillIsReadAtThePinnedVersion(t *testing.T) {
	asked := withSkillToolsFake(t, map[skillPin][]Tool{{"s1", 3}: {skillTool("v3", "")}, {"s1", 4}: {skillTool("v4", "")}})
	cfg := withSkillTools("cfg", parsedWithSkills(t, nil, skillPin{"s1", 3}))
	if got := keys(cfg.Tools); !slices.Equal(got, []string{"v3"}) || !slices.Equal(*asked, []skillPin{{"s1", 3}}) {
		t.Fatalf("tools = %v, read %v", got, *asked)
	}
}

func TestSkillToolsExactDuplicateIsOfferedOnce(t *testing.T) {
	logs := captureLogs(t)
	shared := skillTool("shared", "same")
	withSkillToolsFake(t, map[skillPin][]Tool{s1: {shared}, s2: {shared, skillTool("other", "")}})
	cfg := withSkillTools("cfg", parsedWithSkills(t, nil, s1, s2))
	if got := keys(cfg.Tools); !slices.Equal(got, []string{"other", "shared"}) {
		t.Fatalf("tools = %v", got)
	}
	if logs.Len() != 0 {
		t.Errorf("an exact duplicate must not warn: %s", logs)
	}
}

func TestVariationAndSkillShareAToolKeyVariationWins(t *testing.T) {
	logs := captureLogs(t)
	withSkillToolsFake(t, map[skillPin][]Tool{{"s1", 7}: {skillTool("get-weather", "from skill")}})
	for range 2 { // each conflict warns once per process
		cfg := withSkillTools("cfg", parsedWithSkills(t, ownTools("get-weather"), skillPin{"s1", 7}))
		if got := cfg.Tools["get-weather"]; got.Description != "" || len(cfg.Tools) != 1 {
			t.Fatalf("the variation's definition must win: %+v", cfg.Tools)
		}
	}
	out := logs.String()
	if strings.Count(out, "level=WARN") != 1 {
		t.Fatalf("want one warning, got %q", out)
	}
	for _, want := range []string{"'cfg'", "'s1' version 7", "'get-weather'"} {
		if !strings.Contains(out, want) {
			t.Errorf("warning does not name %s: %q", want, out)
		}
	}
}

// AIC-3975 rejects any key a variation shares with an attached skill, so an
// identical definition warns too.
func TestVariationAndSkillWithAnIdenticalToolStillWarn(t *testing.T) {
	logs := captureLogs(t)
	same := Tool{Name: "own", Type: "function", Parameters: map[string]any{}}
	withSkillToolsFake(t, map[skillPin][]Tool{s1: {same}})
	cfg := withSkillTools("cfg", parsedWithSkills(t, ownTools("own"), s1))
	if !reflect.DeepEqual(cfg.Tools["own"], same) || len(cfg.Tools) != 1 {
		t.Fatalf("tools = %+v", cfg.Tools)
	}
	if strings.Count(logs.String(), "level=WARN") != 1 {
		t.Fatalf("want one warning, got %q", logs)
	}
}

func TestTwoSkillsDefineAToolDifferentlyFirstWins(t *testing.T) {
	logs := captureLogs(t)
	withSkillToolsFake(t, map[skillPin][]Tool{s1: {skillTool("t", "v1")}, s2: {skillTool("t", "v2")}})
	for range 2 {
		cfg := withSkillTools("cfg", parsedWithSkills(t, nil, s1, s2))
		if cfg.Tools["t"].Description != "v1" || len(cfg.Tools) != 1 {
			t.Fatalf("the first skill's definition must be offered once: %+v", cfg.Tools)
		}
	}
	out := logs.String()
	if strings.Count(out, "level=WARN") != 1 || !strings.Contains(out, "s1@1") || !strings.Contains(out, "s2@1") || !strings.Contains(out, "'t'") {
		t.Errorf("want one warning naming both skills and the tool, got %q", out)
	}
}

func TestNoSkillsReadsNoStoreAndNoSkillsPackageAddsNothing(t *testing.T) {
	asked := withSkillToolsFake(t, nil)
	cfg := parsedWithSkills(t, ownTools("own"))
	if got := withSkillTools("cfg", cfg); !slices.Equal(keys(got.Tools), []string{"own"}) || len(*asked) != 0 {
		t.Fatalf("tools = %v, store read for %v", keys(got.Tools), *asked)
	}
	bridge.SkillTools = nil
	if got := withSkillTools("cfg", parsedWithSkills(t, nil, s1)); got.Tools != nil {
		t.Fatalf("without the skills package no skill tools are added: %v", got.Tools)
	}
}

// §3.7 "Skill tools are attached tools" and §3.21a "The allowlist is the
// variation's tools plus the tools of the skills it attaches", on Invoke and
// Stream.
func TestSkillToolHandlersReachTheHandler(t *testing.T) {
	withSkillToolsFake(t, map[skillPin][]Tool{s1: {skillTool("get-weather", "")}, s2: {skillTool("not-attached", "")}})
	all := map[string]ToolHandler{"safe": okTool(), "get-weather": okTool(), "not-attached": okTool()}
	for _, path := range []string{"invoke", "stream"} {
		t.Run(path, func(t *testing.T) {
			cfg := enabledConfig("P", "", nil)
			cfg["skills"] = skillRefs(s1)
			if got := runPath(t, path, cfg, all); !slices.Equal(got, []string{"get-weather"}) {
				t.Errorf("no config.tools: tools = %v", got)
			}
			cfg = enabledConfig("P", "", toolDefs("safe"))
			cfg["skills"] = skillRefs(s1)
			if got := runPath(t, path, cfg, all); !slices.Equal(got, []string{"get-weather", "safe"}) {
				t.Errorf("with config.tools: tools = %v", got)
			}
		})
	}
}

// InspectConfig returns the variation's own tools; the handler's Config has
// the skill tools, and the variation's raw payload is not changed.
func TestInspectConfigKeepsTheVariationsOwnTools(t *testing.T) {
	withSkillToolsFake(t, map[skillPin][]Tool{s1: {skillTool("from-skill", "")}})
	cfg := enabledConfig("P", "", toolDefs("own"))
	cfg["skills"] = skillRefs(s1)
	withFake(t, map[string]any{"cfg": cfg})
	var seen HandlerRequest
	inst := Config(ConfigArgs{Key: "cfg", Handlers: []*ProviderHandler{stubHandler("P", ModeMessages, "ok", nil, &seen)}})
	if _, err := inst.Invoke(context.Background(), "q", userCtx); err != nil {
		t.Fatal(err)
	}
	if got := keys(seen.Config.Tools); !slices.Equal(got, []string{"from-skill", "own"}) {
		t.Fatalf("handler config tools = %v", got)
	}
	if _, ok := seen.Config.Raw["tools"].(map[string]any)["from-skill"]; ok {
		t.Error("skill tools must not be written into the variation's raw payload")
	}
	res, err := InspectConfig(context.Background(), "cfg", userCtx)
	if err != nil {
		t.Fatal(err)
	}
	if got := keys(res.Config.Tools); !slices.Equal(got, []string{"own"}) {
		t.Fatalf("InspectConfig tools = %v", got)
	}
}

// Every graph node gets its own skills' tools, so native graph adapters
// (which read node configs from ResolveGraph) offer them too.
func TestGraphNodesGetTheirSkillTools(t *testing.T) {
	withSkillToolsFake(t, map[skillPin][]Tool{s1: {skillTool("leaf-tool", "")}})
	leaf := nodeCfg("leaf")
	leaf["skills"] = skillRefs(s1)
	withFake(t, map[string]any{
		"g":    map[string]any{"root": "root", "edges": map[string]any{"root": []any{map[string]any{"key": "leaf"}}}},
		"root": nodeCfg("r"), "leaf": leaf,
	})
	def, err := ResolveGraph(bg, "g", userCtx, GraphOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if got := keys(def.GetNode("leaf").Config.Tools); !slices.Equal(got, []string{"leaf-tool"}) {
		t.Errorf("leaf tools = %v", got)
	}
	if got := def.GetNode("root").Config.Tools; len(got) != 0 {
		t.Errorf("root tools = %v", got)
	}
}
