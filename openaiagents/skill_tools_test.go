package openaiagents

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"log/slog"
	"maps"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/handlerkit"
	"github.com/launchdarkly/go-ai-sdk/skills"
)

// TESTING.md §3.21a "Native graph adapters offer a node's skill tools", and
// "Skill tools are merged once": the adapter always calls the merge, which
// leaves a config ResolveGraph merged unchanged (no second read, no conflict
// warning) and merges an unmerged one.
func TestToOpenAIAgentsOffersANodesSkillTools(t *testing.T) {
	for _, unmerged := range []bool{false, true} {
		t.Run(map[bool]string{false: "merged by ResolveGraph", true: "unmerged"}[unmerged], func(t *testing.T) {
			triage := node("Route", "triage-model", "lookup")
			triage["skills"] = []any{map[string]any{"key": "weather", "version": 1}}
			ld := twoNode(map[string]map[string]any{"triage": triage})
			installClient(t, ld)
			sum := sha256.Sum256([]byte("# w"))
			store := &countingStore{SkillStore: skills.NewInMemorySkillStore(map[string]any{"w": map[string]any{
				"key": "weather", "version": 1, "content": "# w", "contentHash": hex.EncodeToString(sum[:]),
				"tools": []any{map[string]any{"type": "function", "name": "get-weather", "parameters": map[string]any{"type": "object"}}},
			}})}
			skills.SetSkillStore(store)
			def := resolve(t, ld, "g")
			if unmerged {
				n := def.GetNode("triage")
				n.Config = must(handlerkit.ParseAiConfig(n.Config.Raw))
				store.reads = 0
			}
			logs := captureLogs(t)
			var calls []string
			handlers := handlersFor(&calls)
			handlers["get-weather"] = tool(func(map[string]any) (any, error) { return "sunny", nil })
			g := must(ToOpenAIAgents(def, &NativeGraphOptions{ToolHandlers: handlers}))
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
