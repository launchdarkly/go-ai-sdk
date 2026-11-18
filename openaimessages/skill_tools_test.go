package openaimessages

import (
	"crypto/sha256"
	"encoding/hex"
	"maps"
	"slices"
	"strings"
	"testing"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/skills"
)

func skillToolDef(name string) map[string]any {
	return map[string]any{"type": "function", "name": name, "parameters": map[string]any{"type": "object"}}
}

func skillEnvelope(key string, version int, tools ...any) map[string]any {
	sum := sha256.Sum256([]byte("# " + key))
	return map[string]any{"key": key, "version": version, "content": "# " + key, "contentHash": hex.EncodeToString(sum[:]), "tools": tools}
}

// TESTING.md §3.21a "Offering skill tools to the provider", end to end from
// hand-built skill envelopes through Config().Invoke to the provider request.
func TestSkillToolsReachTheProviderAndDispatch(t *testing.T) {
	f, client := fakeClient(t, toolResp("r1", "get-weather", `{}`), textResp("done", 1, 1))
	v := maps.Clone(variation)
	v["tools"] = map[string]any{"own": skillToolDef("own")}
	v["skills"] = []any{map[string]any{"key": "weather", "version": 1}}
	ld := &fakeLD{variation: v}
	exp := tracetest.NewInMemoryExporter()
	installClient(t, ld, exp)
	skills.SetSkillStore(skills.NewInMemorySkillStore(map[string]any{
		"a": skillEnvelope("weather", 1, skillToolDef("get-weather")),
		"b": skillEnvelope("not-attached", 1, skillToolDef("other-skill-tool")),
	}))

	ran := 0
	handlers := map[string]aiserver.ToolHandler{
		"own":              tool(func(map[string]any) (any, error) { return "ok", nil }),
		"get-weather":      tool(func(map[string]any) (any, error) { ran++; return "sunny", nil }),
		"other-skill-tool": tool(func(map[string]any) (any, error) { return "no", nil }),
		"unnamed":          tool(func(map[string]any) (any, error) { return "no", nil }),
	}
	h := must(newHandler(Options{Client: client}))
	inst := aiserver.Config(aiserver.ConfigArgs{Key: "cfg", Handlers: []*aiserver.ProviderHandler{h}, ToolHandlers: handlers})
	if _, err := inst.Invoke(bg, "hi", ldcontext.New("u-1")); err != nil {
		t.Fatal(err)
	}

	var offered []string
	for _, tl := range f.requests[0]["tools"].([]any) {
		offered = append(offered, tl.(map[string]any)["name"].(string))
	}
	if !slices.Equal(offered, []string{"get-weather", "own"}) {
		t.Fatalf("offered tools = %v; want the variation's and the attached skill's only", offered)
	}
	if ran != 1 {
		t.Fatalf("the skill tool ran %d times", ran)
	}
	var toolKeys []string
	for _, c := range ld.tracks {
		if c.event == "$ld:ai:tool_call" {
			toolKeys = append(toolKeys, c.data["toolKey"].(string))
		}
	}
	if !slices.Equal(toolKeys, []string{"get-weather"}) {
		t.Errorf("$ld:ai:tool_call toolKeys = %v", toolKeys)
	}
	if spans := spansByName(exp)["execute_tool get-weather"]; len(spans) != 1 {
		t.Errorf("execute_tool spans = %d", len(spans))
	}
	// §3.27: core reads skills through core retrieval, not the accessors.
	for _, helper := range ld.usageHelpers() {
		if strings.HasPrefix(helper, "client.getSkill") {
			t.Errorf("reading skills for their tools reported %s", helper)
		}
	}
}
