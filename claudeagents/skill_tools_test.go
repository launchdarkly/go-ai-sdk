package claudeagents

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"reflect"
	"testing"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/skills"
)

func weatherSkillStore() *skills.InMemorySkillStore {
	sum := sha256.Sum256([]byte("# w"))
	return skills.NewInMemorySkillStore(map[string]any{"w": map[string]any{
		"key": "weather", "version": 1, "content": "# w", "contentHash": hex.EncodeToString(sum[:]),
		"tools": []any{map[string]any{"type": "function", "name": "get-weather", "parameters": map[string]any{"type": "object"}}},
	}})
}

// TESTING.md §3.21a "Native graph adapters offer a node's skill tools".
func TestToClaudeAgentsOffersANodesSkillTools(t *testing.T) {
	flags := graphFlags(nil)
	flags["leaf"] = agentVariation("leaf inst", map[string]any{
		"tools":  map[string]any{"lookup": map[string]any{"name": "lookup", "type": "function", "parameters": map[string]any{"type": "object"}}},
		"skills": []any{map[string]any{"key": "weather", "version": 1}},
	})
	e := setup(t, flags, toolResp("mcp__subagents__leaf", `{"input":"q"}`), textResp("leaf done", 1, 1), textResp("done", 1, 1))
	skills.SetSkillStore(weatherSkillStore())
	ok := aiserver.ToolFunc(func(_ context.Context, _ map[string]any) (any, error) { return "ok", nil })
	tools := map[string]aiserver.ToolHandler{"lookup": ok, "get-weather": ok}
	if _, err := must(ToClaudeAgents(resolve(t), &AdapterOptions{Handler: e.opts, ToolHandlers: tools})).Invoke(bg, "q"); err != nil {
		t.Fatal(err)
	}
	r := e.api.reqs()
	if len(r) != 3 || !reflect.DeepEqual(toolNames(r[0]), []string{"mcp__subagents__leaf"}) {
		t.Fatalf("calls=%d root tools=%v", len(r), toolNames(r[0]))
	}
	if got := toolNames(r[1]); !reflect.DeepEqual(got, []string{"get-weather", "lookup"}) {
		t.Errorf("leaf tools = %v", got)
	}
}
