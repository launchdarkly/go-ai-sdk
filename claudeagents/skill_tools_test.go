package claudeagents

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"log/slog"
	"reflect"
	"strings"
	"testing"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/handlerkit"
	"github.com/launchdarkly/go-ai-sdk/skills"
)

func weatherSkillStore() *skills.InMemorySkillStore {
	sum := sha256.Sum256([]byte("# w"))
	return skills.NewInMemorySkillStore(map[string]any{"w": map[string]any{
		"key": "weather", "version": 1, "content": "# w", "contentHash": hex.EncodeToString(sum[:]),
		"tools": []any{map[string]any{"type": "function", "name": "get-weather", "parameters": map[string]any{"type": "object"}}},
	}})
}

// TESTING.md §3.21a "Native graph adapters offer a node's skill tools", and
// "Skill tools are merged once": the adapter always calls the merge, which
// leaves a config ResolveGraph merged unchanged (no second read, no conflict
// warning) and merges an unmerged one.
func TestToClaudeAgentsOffersANodesSkillTools(t *testing.T) {
	for _, unmerged := range []bool{false, true} {
		t.Run(map[bool]string{false: "merged by ResolveGraph", true: "unmerged"}[unmerged], func(t *testing.T) {
			flags := graphFlags(nil)
			flags["leaf"] = agentVariation("leaf inst", map[string]any{
				"tools":  map[string]any{"lookup": map[string]any{"name": "lookup", "type": "function", "parameters": map[string]any{"type": "object"}}},
				"skills": []any{map[string]any{"key": "weather", "version": 1}},
			})
			e := setup(t, flags, toolResp("mcp__subagents__leaf", `{"input":"q"}`), textResp("leaf done", 1, 1), textResp("done", 1, 1))
			store := &countingStore{SkillStore: weatherSkillStore()}
			skills.SetSkillStore(store)
			def := resolve(t)
			if unmerged {
				n := def.GetNode("leaf")
				n.Config = must(handlerkit.ParseAiConfig(n.Config.Raw))
				store.reads = 0
			}
			logs := captureLogs(t)
			ok := aiserver.ToolFunc(func(_ context.Context, _ map[string]any) (any, error) { return "ok", nil })
			tools := map[string]aiserver.ToolHandler{"lookup": ok, "get-weather": ok}
			if _, err := must(ToClaudeAgents(def, &AdapterOptions{Handler: e.opts, ToolHandlers: tools})).Invoke(bg, "q"); err != nil {
				t.Fatal(err)
			}
			r := e.api.reqs()
			if len(r) != 3 || !reflect.DeepEqual(toolNames(r[0]), []string{"mcp__subagents__leaf"}) {
				t.Fatalf("calls=%d root tools=%v", len(r), toolNames(r[0]))
			}
			if got := toolNames(r[1]); !reflect.DeepEqual(got, []string{"get-weather", "lookup"}) {
				t.Errorf("leaf tools = %v", got)
			}
			if store.reads != 1 {
				t.Errorf("the skill was read %d times, want once", store.reads)
			}
			if strings.Contains(logs.String(), "level=WARN") {
				t.Errorf("unexpected warning: %s", logs)
			}
		})
	}
}

type countingStore struct {
	skills.SkillStore
	reads int
}

func (c *countingStore) GetObject(kind, key string, version int) (any, error) {
	c.reads++
	return c.SkillStore.GetObject(kind, key, version)
}

func captureLogs(t *testing.T) *bytes.Buffer {
	t.Helper()
	var buf bytes.Buffer
	aiserver.SetLogger(slog.New(slog.NewTextHandler(&buf, nil)))
	t.Cleanup(func() { aiserver.SetLogger(nil) })
	return &buf
}
