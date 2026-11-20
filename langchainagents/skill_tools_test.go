package langchainagents

import (
	"bytes"
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

// TESTING.md §3.21a "Native graph adapters offer a node's skill tools": a node
// whose variation attaches a skill pinning get-weather offers it beside its own
// and handoff tools; a node that does not attach the skill does not get it.
// "Skill tools are merged once": the adapter always calls the merge, which
// leaves a config ResolveGraph merged unchanged (no second read, no conflict
// warning) and merges an unmerged one.
func TestToLangGraphOffersANodesSkillTools(t *testing.T) {
	for _, unmerged := range []bool{false, true} {
		t.Run(map[bool]string{false: "merged by ResolveGraph", true: "unmerged"}[unmerged], func(t *testing.T) {
			ld := twoNode()
			ld.variations["triage"]["skills"] = []any{map[string]any{"key": "weather", "version": 1}}
			installClient(t, ld)
			sum := sha256.Sum256([]byte("# w"))
			store := &countingStore{SkillStore: skills.NewInMemorySkillStore(map[string]any{"w": map[string]any{
				"key": "weather", "version": 1, "content": "# w", "contentHash": hex.EncodeToString(sum[:]),
				"tools": []any{map[string]any{"type": "function", "name": "get-weather", "parameters": map[string]any{"type": "object"}}},
			}})}
			skills.SetSkillStore(store)
			def := resolve("g")
			if unmerged {
				n := def.GetNode("triage")
				n.Config = must(handlerkit.ParseAiConfig(n.Config.Raw))
				store.reads = 0
			}
			logs := captureLogs(t)
			var calls []string
			handlers := handlersFor(&calls)
			handlers["get-weather"] = fn(func(map[string]any) (any, error) { return "sunny", nil })
			fm := map[string]*fakeModel{
				"triage":  {turns: []scripted{textTurn("routed", 1, 1)}},
				"billing": {turns: []scripted{textTurn("billed", 1, 1)}},
			}
			if _, err := must(ToLangGraph(def, &NativeGraphOptions{ToolHandlers: handlers, ModelFactory: models(fm)})).Invoke(bg, "hi"); err != nil {
				t.Fatal(err)
			}
			if got := toolNames(fm["triage"].call(0).opts); !reflect.DeepEqual(got, []string{"get-weather", "lookup", "transfer_to_billing"}) {
				t.Errorf("triage tools = %v", got)
			}
			if got := toolNames(fm["billing"].call(0).opts); !reflect.DeepEqual(got, []string{"refund"}) {
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

func captureLogs(t *testing.T) *bytes.Buffer {
	t.Helper()
	var buf bytes.Buffer
	aiserver.SetLogger(slog.New(slog.NewTextHandler(&buf, nil)))
	t.Cleanup(func() { aiserver.SetLogger(nil) })
	return &buf
}

type countingStore struct {
	skills.SkillStore
	reads int
}

func (c *countingStore) GetObject(kind, key string, version int) (any, error) {
	c.reads++
	return c.SkillStore.GetObject(kind, key, version)
}
