package openaiagents

import (
	"crypto/sha256"
	"encoding/hex"
	"maps"
	"reflect"
	"slices"
	"testing"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/skills"
)

// TESTING.md §3.21a "Native graph adapters offer a node's skill tools".
func TestToOpenAIAgentsOffersANodesSkillTools(t *testing.T) {
	triage := node("Route", "triage-model", "lookup")
	triage["skills"] = []any{map[string]any{"key": "weather", "version": 1}}
	ld := twoNode(map[string]map[string]any{"triage": triage})
	installClient(t, ld)
	sum := sha256.Sum256([]byte("# w"))
	skills.SetSkillStore(skills.NewInMemorySkillStore(map[string]any{"w": map[string]any{
		"key": "weather", "version": 1, "content": "# w", "contentHash": hex.EncodeToString(sum[:]),
		"tools": []any{map[string]any{"type": "function", "name": "get-weather", "parameters": map[string]any{"type": "object"}}},
	}}))
	var calls []string
	handlers := handlersFor(&calls)
	handlers["get-weather"] = tool(func(map[string]any) (any, error) { return "sunny", nil })
	g := must(ToOpenAIAgents(resolve(t, ld, "g"), &NativeGraphOptions{ToolHandlers: handlers}))
	agents, err := g.buildAgents(bg, nil, "run", false)
	if err != nil {
		t.Fatal(err)
	}
	root, leaf := agents["triage"], agents["billing"]
	if got := slices.Sorted(maps.Keys(root.tools)); !reflect.DeepEqual(got, []string{"get-weather", "lookup"}) || len(root.handoffs) != 1 {
		t.Errorf("triage tools = %v, handoffs = %d", got, len(root.handoffs))
	}
	if got := slices.Sorted(maps.Keys(leaf.tools)); !reflect.DeepEqual(got, []string{"refund"}) {
		t.Errorf("billing tools = %v", got)
	}
}

// §3.21a "A skill tool needs a registered handler": a NativeTool under a skill
// tool's key follows §1.3 and becomes a hosted tool, not a function tool.
func TestNativeToolUnderASkillToolKeyIsHosted(t *testing.T) {
	triage := node("Route", "triage-model")
	triage["skills"] = []any{map[string]any{"key": "search", "version": 1}}
	ld := twoNode(map[string]map[string]any{"triage": triage})
	installClient(t, ld)
	sum := sha256.Sum256([]byte("# s"))
	skills.SetSkillStore(skills.NewInMemorySkillStore(map[string]any{"s": map[string]any{
		"key": "search", "version": 1, "content": "# s", "contentHash": hex.EncodeToString(sum[:]),
		"tools": []any{map[string]any{"type": "function", "name": "web", "parameters": map[string]any{"type": "object"}}},
	}}))
	handlers := map[string]aiserver.ToolHandler{"web": &aiserver.NativeTool{ToolName: "web_search"}}
	agents, err := must(ToOpenAIAgents(resolve(t, ld, "g"), &NativeGraphOptions{ToolHandlers: handlers})).buildAgents(bg, nil, "run", false)
	if err != nil {
		t.Fatal(err)
	}
	root := agents["triage"]
	if _, fn := root.tools["web"]; fn {
		t.Error("a NativeTool must not be offered as a function tool")
	}
	if _, hosted := root.native["web"]; !hosted {
		t.Errorf("native = %v, want the skill tool hosted", root.native)
	}
}
