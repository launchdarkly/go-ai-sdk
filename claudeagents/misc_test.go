package claudeagents

import (
	"errors"
	"reflect"
	"slices"
	"testing"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

func TestBuiltins(t *testing.T) {
	want := map[*aiserver.NativeTool]string{
		ClaudeBash: "Bash", ClaudeRead: "Read", ClaudeEdit: "Edit", ClaudeWrite: "Write", ClaudeGlob: "Glob",
		ClaudeGrep: "Grep", ClaudeWebFetch: "WebFetch", ClaudeWebSearch: "WebSearch", ClaudeTodoWrite: "TodoWrite",
		ClaudeNotebookEdit: "NotebookEdit",
	}
	if len(want) != 10 { // map keys are the sentinel pointers: 10 distinct identities
		t.Fatalf("sentinels share identity: %d distinct", len(want))
	}
	for n, name := range want {
		var h aiserver.ToolHandler = n // each is a NativeTool tool handler
		if _, ok := h.(*aiserver.NativeTool); !ok || n.ToolName != name {
			t.Errorf("%s: ToolName=%q", name, n.ToolName)
		}
	}
}

// A NativeTool this package cannot provide fails loudly instead of vanishing.
func TestUnsupportedNativeTool(t *testing.T) {
	e := setup(t, nil, textResp("x", 1, 1))
	c := cfg(func(c *aiserver.AiConfig) {
		c.Tools = map[string]aiserver.Tool{"shell": {Name: "shell", Parameters: map[string]any{}}}
	})
	_, err := e.handler().Invoke(bg, aiserver.HandlerRequest{Config: c, ToolHandlers: map[string]aiserver.ToolHandler{"shell": &aiserver.NativeTool{ToolName: "Teleport"}}})
	if !errors.Is(err, ErrUnsupportedNativeTool) || len(e.api.reqs()) != 0 {
		t.Errorf("err=%v calls=%d", err, len(e.api.reqs()))
	}
}

// convenience export, plus handler-level NativeTool tracking.
func TestClaudeAgentsConvenience(t *testing.T) {
	tools := map[string]any{"web-search": map[string]any{"name": "web-search", "type": "function", "parameters": map[string]any{}}}
	e := setup(t, map[string]any{"cfg": agentVariation("Answer {{user_input}}", map[string]any{"tools": tools})},
		msgJSON("end_turn", serverSearch, 1, 1), textResp("bare", 1, 1))
	res, err := ClaudeAgents(bg, "cfg", "hi", user, &InvokeOptions{Handler: e.opts, Variables: map[string]any{"user_input": "hello"},
		ToolHandlers: map[string]aiserver.ToolHandler{"web-search": ClaudeWebSearch}})
	if err != nil || res.Response != "found" || res.TrackData.ConfigKey != "cfg" {
		t.Fatalf("res=%+v err=%v", res, err)
	}
	if s := system(e.api.reqs()[0]); s != "Answer hello" {
		t.Errorf("variables not forwarded: %q", s)
	}
	tc := e.ld.events("$ld:ai:tool_call")
	if len(tc) != 1 || tc[0].Data["toolKey"] != "web-search" {
		t.Errorf("native tool_call = %+v", tc)
	}
	// options are optional; the default handler uses the env API key, so only check it routes.
	if _, err := ClaudeAgents(bg, "missing", "hi", user, nil); err == nil {
		t.Error("want a disabled-config error")
	}
}

// Usage events: each helper reports once with this package's identity.
func TestSDKUsage(t *testing.T) {
	cases := []struct {
		helper string
		call   func(e *env)
	}{
		{"claude-agents.createClaudeAgentsHandler", func(*env) { NewHandler(Options{}) }},
		{"claude-agents.claudeAgents", func(e *env) {
			_, _ = ClaudeAgents(bg, "cfg", "q", user, &InvokeOptions{Handler: e.opts})
		}},
		{"claude-agents.claudeGraph", func(e *env) { _, _ = ClaudeGraph("g", GraphOptions{}, e.opts).Invoke(bg, "q", user) }},
		{"claude-agents.toClaudeAgents", func(e *env) { ToClaudeAgents(nil, nil) }},
	}
	for _, tc := range cases {
		t.Run(tc.helper, func(t *testing.T) {
			e := setup(t, map[string]any{"cfg": agentVariation("x", nil), "g": map[string]any{"root": "cfg", "_ldMeta": map[string]any{"enabled": true}}},
				textResp("a", 1, 1), textResp("b", 1, 1), sse([]string{"c"}, "", 1, 1), sse([]string{"d"}, "", 1, 1))
			tc.call(e)
			tc.call(e) // a repeat sends nothing
			if got := e.ld.helpers(); !reflect.DeepEqual(got, []string{tc.helper}) {
				t.Errorf("usage helpers = %v", got)
			}
			ev := e.ld.events("$ld:ai:sdk:usage")[0]
			if ev.Data["helperPackageName"] != PackageName || ev.Data["helperPackageVersion"] != PackageVersion ||
				ev.Data["aiSdkName"] != aiserver.PackageName || ev.Data["aiSdkVersion"] != aiserver.PackageVersion ||
				ev.Data["aiSdkLanguage"] != "go" || ev.Metric != 1 {
				t.Errorf("usage payload = %v", ev.Data)
			}
			if tc.helper == "claude-agents.claudeAgents" && len(e.ld.events("$ld:ai:generation:success")) == 0 {
				t.Error("wrapper did not reach config().invoke")
			}
		})
	}
	// sdk-info registered under the package identity
	e := setup(t, nil)
	var pkgs []any
	for _, ev := range e.ld.events("$ld:ai:sdk:info") {
		pkgs = append(pkgs, ev.Data["aiSdkName"])
	}
	if !slices.Contains(pkgs, any(PackageName)) {
		t.Errorf("sdk info = %v", pkgs)
	}
}
