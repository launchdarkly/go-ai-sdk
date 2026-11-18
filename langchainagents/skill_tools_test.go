package langchainagents

import (
	"crypto/sha256"
	"encoding/hex"
	"reflect"
	"testing"

	"github.com/launchdarkly/go-ai-sdk/skills"
)

// TESTING.md §3.21a "Native graph adapters offer a node's skill tools": a node
// whose variation attaches a skill pinning get-weather offers it beside its own
// and handoff tools; a node that does not attach the skill does not get it.
func TestToLangGraphOffersANodesSkillTools(t *testing.T) {
	ld := twoNode()
	ld.variations["triage"]["skills"] = []any{map[string]any{"key": "weather", "version": 1}}
	installClient(t, ld)
	sum := sha256.Sum256([]byte("# w"))
	skills.SetSkillStore(skills.NewInMemorySkillStore(map[string]any{"w": map[string]any{
		"key": "weather", "version": 1, "content": "# w", "contentHash": hex.EncodeToString(sum[:]),
		"tools": []any{map[string]any{"type": "function", "name": "get-weather", "parameters": map[string]any{"type": "object"}}},
	}}))
	var calls []string
	handlers := handlersFor(&calls)
	handlers["get-weather"] = fn(func(map[string]any) (any, error) { return "sunny", nil })
	fm := map[string]*fakeModel{
		"triage":  {turns: []scripted{textTurn("routed", 1, 1)}},
		"billing": {turns: []scripted{textTurn("billed", 1, 1)}},
	}
	if _, err := must(ToLangGraph(resolve("g"), &NativeGraphOptions{ToolHandlers: handlers, ModelFactory: models(fm)})).Invoke(bg, "hi"); err != nil {
		t.Fatal(err)
	}
	if got := toolNames(fm["triage"].call(0).opts); !reflect.DeepEqual(got, []string{"get-weather", "lookup", "transfer_to_billing"}) {
		t.Errorf("triage tools = %v", got)
	}
	if got := toolNames(fm["billing"].call(0).opts); !reflect.DeepEqual(got, []string{"refund"}) {
		t.Errorf("billing tools = %v", got)
	}
}
