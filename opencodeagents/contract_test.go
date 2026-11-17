package opencodeagents

import (
	"context"
	"errors"
	"reflect"
	"regexp"
	"strings"
	"testing"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

func TestSentinels(t *testing.T) {
	want := map[*aiserver.NativeTool]string{
		OpenCodeBash: "bash", OpenCodeRead: "read", OpenCodeEdit: "edit", OpenCodeWrite: "write",
		OpenCodeGlob: "glob", OpenCodeGrep: "grep", OpenCodeTask: "task", OpenCodeSkill: "skill",
		OpenCodeTodoWrite: "todowrite", OpenCodeWebFetch: "webfetch", OpenCodeWebSearch: "websearch",
		OpenCodeApplyPatch: "apply_patch",
	}
	if len(want) != 12 {
		t.Fatalf("sentinels share identity: %d distinct", len(want))
	}
	for n, name := range want {
		if n == nil || n.ToolName != name {
			t.Errorf("%s: %+v", name, n)
		}
	}
}

// runTools runs one call with the given config tool keys and handlers and
// returns the fake's state.
func runTools(t *testing.T, keys []string, handlers map[string]aiserver.ToolHandler, spec fakeSpec) fakeState {
	t.Helper()
	if spec.Events == nil {
		spec.Events = reply("ok", 1, 1)
	}
	f := newFake(t, spec)
	c := cfg(nil)
	if keys != nil {
		c = cfg(withTools(keys...))
	}
	if _, err := invoke(t, f.handler(t), aiserver.HandlerRequest{Config: c, UserInput: "q", ToolHandlers: handlers}); err != nil {
		t.Fatal(err)
	}
	return f.state(t)
}

// The offered set is the intersection of the config's tools and the tool
// handlers, in both directions.
func TestOfferedSetIsTheIntersection(t *testing.T) {
	setup(t, nil)
	s := runTools(t, []string{"shell", "reader"}, map[string]aiserver.ToolHandler{
		"shell":  OpenCodeBash, // offered
		"extra":  OpenCodeGrep, // not in the config
		"absent": aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) { return "x", nil }),
	}, fakeSpec{}) // "reader" has no handler
	if got := permission(t, s); !reflect.DeepEqual(got, []string{"deny *", "allow bash"}) {
		t.Errorf("permission = %v", got)
	}
	if len(s.MCP) != 0 {
		t.Errorf("a bridge was registered for a tool the config does not declare: %v", s.MCP)
	}
}

// An empty offered set, and a call with no tool handlers at all, send only
// the deny-all rule.
func TestEmptyOfferedSetDeniesEverything(t *testing.T) {
	setup(t, nil)
	for name, run := range map[string]func() fakeState{
		"no config tools": func() fakeState {
			return runTools(t, nil, map[string]aiserver.ToolHandler{"bash": OpenCodeBash}, fakeSpec{})
		},
		"no tool handlers": func() fakeState { return runTools(t, []string{"bash", "skill"}, nil, fakeSpec{}) },
	} {
		t.Run(name, func(t *testing.T) {
			if got := permission(t, run()); !reflect.DeepEqual(got, []string{"deny *"}) {
				t.Errorf("permission = %v", got)
			}
		})
	}
}

// skill is denied unless the variation lists it and the caller passes
// OpenCodeSkill.
func TestSkillIsDeniedUnlessOffered(t *testing.T) {
	setup(t, nil)
	if got := permission(t, runTools(t, []string{"skill"}, map[string]aiserver.ToolHandler{"other": OpenCodeSkill}, fakeSpec{})); !reflect.DeepEqual(got, []string{"deny *"}) {
		t.Errorf("unoffered: %v", got)
	}
	if got := permission(t, runTools(t, []string{"skill"}, map[string]aiserver.ToolHandler{"skill": OpenCodeSkill}, fakeSpec{})); !reflect.DeepEqual(got, []string{"deny *", "allow skill"}) {
		t.Errorf("offered: %v", got)
	}
}

// edit, write and apply_patch are one grant: any of them sends one edit allow.
func TestEditFamilyIsOneGrant(t *testing.T) {
	setup(t, nil)
	for name, tools := range map[string]map[string]aiserver.ToolHandler{
		"edit":        {"t": OpenCodeEdit},
		"write":       {"t": OpenCodeWrite},
		"apply_patch": {"t": OpenCodeApplyPatch},
		"all three":   {"t": OpenCodeEdit, "u": OpenCodeWrite, "v": OpenCodeApplyPatch},
	} {
		t.Run(name, func(t *testing.T) {
			var keys []string
			for k := range tools {
				keys = append(keys, k)
			}
			if got := permission(t, runTools(t, keys, tools, fakeSpec{})); !reflect.DeepEqual(got, []string{"deny *", "allow edit"}) {
				t.Errorf("permission = %v", got)
			}
		})
	}
}

// A NativeTool fires $ld:ai:tool_call once per tool call id, under its
// config key, when its execute_tool span opens.
func TestNativeToolIsTrackedOncePerCall(t *testing.T) {
	tools := map[string]any{"run-shell": map[string]any{"name": "run-shell", "type": "function", "parameters": map[string]any{}}}
	e := setup(t, map[string]any{"cfg": agentVariation("x", map[string]any{"tools": tools})})
	f := newFake(t, fakeSpec{Events: []map[string]any{
		assistant("m1"), delta("m1", "p1", "running"),
		toolPart("m1", "c1", "bash", "pending", nil, ""),
		toolPart("m1", "c1", "bash", "running", map[string]any{"command": "ls"}, ""),
		toolPart("m1", "c1", "bash", "running", map[string]any{"command": "ls"}, ""),
		toolPart("m1", "c1", "bash", "completed", map[string]any{"command": "ls"}, "a.txt"),
		toolPart("m1", "c2", "bash", "completed", map[string]any{"command": "pwd"}, "/w"),
		completed("m1", 1, 1), idle("ses"),
	}})
	if _, err := OpenCodeAgents(bg, "cfg", "q", user, &InvokeOptions{Handler: f.options(),
		ToolHandlers: map[string]aiserver.ToolHandler{"run-shell": OpenCodeBash}}); err != nil {
		t.Fatal(err)
	}
	tc := e.ld.events("$ld:ai:tool_call")
	if len(tc) != 2 || tc[0].Data["toolKey"] != "run-shell" || tc[1].Data["toolKey"] != "run-shell" {
		t.Errorf("tool_call events = %+v", tc)
	}
	tools2 := e.spans("execute_tool")
	if len(tools2) != 2 {
		t.Fatalf("execute_tool spans = %d", len(tools2))
	}
	for _, s := range tools2 {
		if a := attrs(s); a["gen_ai.tool.name"] != "run-shell" || (a["gen_ai.tool.call.id"] != "c1" && a["gen_ai.tool.call.id"] != "c2") {
			t.Errorf("tool span attrs = %v", a)
		}
	}
}

// A ToolFunc is served from a per-call loopback MCP bridge, registered with
// POST /mcp, allowed as ld_<run>_<key>, and disconnected after the turn.
func TestBridgedToolEndToEnd(t *testing.T) {
	tools := map[string]any{"get_weather": map[string]any{"name": "get_weather", "type": "function", "description": "Weather",
		"parameters": map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}}}}
	e := setup(t, map[string]any{"cfg": agentVariation("x", map[string]any{"tools": tools})})
	var got map[string]any
	weather := aiserver.ToolFunc(func(_ context.Context, args map[string]any) (any, error) {
		got = args
		return "sunny in " + args["city"].(string), nil
	})
	f := newFake(t, fakeSpec{
		BridgeCalls: []bridgeCall{{Name: "get_weather", Arguments: map[string]any{"city": "Paris"}}},
		Events: []map[string]any{
			assistant("m1"), delta("m1", "p1", "checking"),
			toolPart("m1", "c1", "{{bridge:get_weather}}", "running", map[string]any{"city": "Paris"}, ""),
			toolPart("m1", "c1", "{{bridge:get_weather}}", "completed", map[string]any{"city": "Paris"}, "sunny in Paris"),
			completed("m1", 1, 1), idle("ses"),
		},
	})
	if _, err := OpenCodeAgents(bg, "cfg", "q", user, &InvokeOptions{Handler: f.options(),
		ToolHandlers: map[string]aiserver.ToolHandler{"get_weather": weather}}); err != nil {
		t.Fatal(err)
	}
	s := f.state(t)
	if len(s.MCP) != 1 {
		t.Fatalf("mcp registrations = %v", s.MCP)
	}
	reg := s.MCP[0]
	name, _ := reg["name"].(string)
	if !regexp.MustCompile(`^ld_[0-9a-f]{16}$`).MatchString(name) {
		t.Errorf("bridge name = %q", name)
	}
	c, _ := reg["config"].(map[string]any)
	url, _ := c["url"].(string)
	auth, _ := c["headers"].(map[string]any)["Authorization"].(string)
	if c["type"] != "remote" || c["oauth"] != false || !strings.HasPrefix(url, "http://127.0.0.1:") || !strings.HasPrefix(auth, "Bearer ") || len(auth) < 40 {
		t.Errorf("registration = %v", reg)
	}
	if !strings.HasPrefix(s.BridgeStream, "text/event-stream|: connected") {
		t.Errorf("bridge GET stream = %q", s.BridgeStream)
	}
	if len(s.BridgeTools) != 1 || s.BridgeTools[0]["name"] != "get_weather" || s.BridgeTools[0]["description"] != "Weather" ||
		!reflect.DeepEqual(s.BridgeTools[0]["inputSchema"], tools["get_weather"].(map[string]any)["parameters"]) {
		t.Errorf("tools/list = %v", s.BridgeTools)
	}
	if len(s.BridgeResults) != 1 || s.BridgeResults[0]["isError"] != false ||
		!reflect.DeepEqual(s.BridgeResults[0]["content"], []any{map[string]any{"type": "text", "text": "sunny in Paris"}}) {
		t.Errorf("tools/call = %v", s.BridgeResults)
	}
	if got["city"] != "Paris" {
		t.Errorf("tool args = %v", got)
	}
	if p := permission(t, s); !reflect.DeepEqual(p, []string{"deny *", "allow " + name + "_get_weather"}) {
		t.Errorf("permission = %v", p)
	}
	if !reflect.DeepEqual(s.Disconnects, []string{name}) {
		t.Errorf("disconnects = %v", s.Disconnects)
	}
	if tc := e.ld.events("$ld:ai:tool_call"); len(tc) != 1 || tc[0].Data["toolKey"] != "get_weather" {
		t.Errorf("tool_call events = %+v", tc)
	}
	if ts := e.spans("execute_tool"); len(ts) != 1 || attrs(ts[0])["gen_ai.tool.name"] != "get_weather" {
		t.Errorf("execute_tool spans = %v", ts)
	}
	assertGone(t, s)
}

// A ToolFunc error comes back to OpenCode as an MCP tool error, not a
// failed call.
func TestBridgedToolErrorIsAToolResult(t *testing.T) {
	setup(t, nil)
	boom := aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) { return nil, errors.New("tool broke") })
	s := runTools(t, []string{"flaky"}, map[string]aiserver.ToolHandler{"flaky": boom},
		fakeSpec{BridgeCalls: []bridgeCall{{Name: "flaky", Arguments: map[string]any{}}}})
	if len(s.BridgeResults) != 1 || s.BridgeResults[0]["isError"] != true ||
		!strings.Contains(jsonString(s.BridgeResults[0]["content"]), "tool broke") {
		t.Errorf("tools/call = %v", s.BridgeResults)
	}
}

// POST /mcp answering 200 with a status other than "connected" fails the
// call before the message is posted.
func TestBridgeThatDoesNotConnectFailsTheCall(t *testing.T) {
	setup(t, nil)
	f := newFake(t, fakeSpec{MCPStatus: "failed", Events: reply("ok", 1, 1)})
	fn := aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) { return "x", nil })
	_, err := invoke(t, f.handler(t), aiserver.HandlerRequest{Config: cfg(withTools("t")), UserInput: "q",
		ToolHandlers: map[string]aiserver.ToolHandler{"t": fn}})
	if err == nil || !strings.Contains(err.Error(), "failed") {
		t.Errorf("err = %v", err)
	}
	s := f.state(t)
	if len(s.Messages) != 0 {
		t.Error("the message was posted")
	}
	assertGone(t, s)
}

// Native and bridged tools can be offered together.
func TestNativeAndBridgedTogether(t *testing.T) {
	setup(t, nil)
	fn := aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) { return "x", nil })
	s := runTools(t, []string{"shell", "lookup"}, map[string]aiserver.ToolHandler{"shell": OpenCodeBash, "lookup": fn}, fakeSpec{})
	name, _ := s.MCP[0]["name"].(string)
	got := permission(t, s)
	if len(got) != 3 || got[0] != "deny *" || !contains(got, "allow bash") || !contains(got, "allow "+name+"_lookup") {
		t.Errorf("permission = %v", got)
	}
}

// Two calls get different bridge names, tokens and ports.
func TestEachCallGetsItsOwnBridge(t *testing.T) {
	setup(t, nil)
	fn := aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) { return "x", nil })
	a := runTools(t, []string{"t"}, map[string]aiserver.ToolHandler{"t": fn}, fakeSpec{})
	b := runTools(t, []string{"t"}, map[string]aiserver.ToolHandler{"t": fn}, fakeSpec{})
	if jsonString(a.MCP[0]) == jsonString(b.MCP[0]) || a.MCP[0]["name"] == b.MCP[0]["name"] {
		t.Errorf("registrations repeat: %v / %v", a.MCP[0], b.MCP[0])
	}
}

// A tool OpenCode reports that this call did not offer is not tracked.
func TestUnofferedToolIsNotTracked(t *testing.T) {
	tools := map[string]any{"shell": map[string]any{"name": "shell", "type": "function", "parameters": map[string]any{}}}
	e := setup(t, map[string]any{"cfg": agentVariation("x", map[string]any{"tools": tools})})
	f := newFake(t, fakeSpec{Events: []map[string]any{
		assistant("m1"), delta("m1", "p1", "x"),
		toolPart("m1", "c1", "read", "completed", map[string]any{"path": "a"}, "data"),
		completed("m1", 1, 1), idle("ses"),
	}})
	if _, err := OpenCodeAgents(bg, "cfg", "q", user, &InvokeOptions{Handler: f.options(),
		ToolHandlers: map[string]aiserver.ToolHandler{"shell": OpenCodeBash}}); err != nil {
		t.Fatal(err)
	}
	if tc := e.ld.events("$ld:ai:tool_call"); len(tc) != 0 {
		t.Errorf("tool_call events = %+v", tc)
	}
}

func contains(s []string, v string) bool {
	for _, x := range s {
		if x == v {
			return true
		}
	}
	return false
}
