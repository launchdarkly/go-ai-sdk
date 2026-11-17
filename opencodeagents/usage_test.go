package opencodeagents

import (
	"testing"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

// Each public helper reports its usage id once, with this package's identity.
func TestSDKUsage(t *testing.T) {
	cases := []struct {
		helper string
		call   func(t *testing.T, f *fake)
	}{
		{"opencode-agents.createOpenCodeHandler", func(t *testing.T, f *fake) { _, _ = NewHandler(f.options()) }},
		{"opencode-agents.opencodeAgents", func(t *testing.T, f *fake) {
			_, _ = OpenCodeAgents(bg, "cfg", "q", user, &InvokeOptions{Handler: f.options()})
		}},
	}
	for _, tc := range cases {
		t.Run(tc.helper, func(t *testing.T) {
			e := setup(t, map[string]any{"cfg": agentVariation("x", nil)})
			f := newFake(t, fakeSpec{Events: reply("ok", 1, 1)})
			tc.call(t, f)
			tc.call(t, f)
			var got []map[string]any
			for _, ev := range e.ld.events("$ld:ai:sdk:usage") {
				got = append(got, ev.Data)
			}
			n := 0
			for _, d := range got {
				if d["helper"] == tc.helper {
					n++
					if d["helperPackageName"] != PackageName || d["helperPackageVersion"] != PackageVersion {
						t.Errorf("usage identity = %v", d)
					}
				}
				if d["helper"] == "opencode-agents.createOpenCodeHandler" && tc.helper != d["helper"] {
					t.Errorf("the wrapper also reported %v", d["helper"])
				}
			}
			if n != 1 {
				t.Errorf("%s reported %d times: %v", tc.helper, n, got)
			}
		})
	}
}

// The wrapper forwards variables and tool handlers and routes through config.
func TestOpenCodeAgentsConvenience(t *testing.T) {
	setup(t, map[string]any{"cfg": agentVariation("Answer {{who}}", map[string]any{
		"tools": map[string]any{"shell": map[string]any{"name": "shell", "type": "function", "parameters": map[string]any{}}},
	})})
	f := newFake(t, fakeSpec{Events: reply("done", 1, 1)})
	res, err := OpenCodeAgents(bg, "cfg", "q", user, &InvokeOptions{Handler: f.options(), Variables: map[string]any{"who": "Ada"},
		ToolHandlers: map[string]aiserver.ToolHandler{"shell": OpenCodeBash}})
	if err != nil || res.Response != "done" || res.TrackData.ConfigKey != "cfg" {
		t.Fatalf("res = %+v, err = %v", res, err)
	}
	s := f.state(t)
	if s.Messages[0]["system"] != "Answer Ada" {
		t.Errorf("system = %v", s.Messages[0]["system"])
	}
	if p := permission(t, s); len(p) != 2 || p[1] != "allow bash" {
		t.Errorf("permission = %v", p)
	}
	if _, err := OpenCodeAgents(bg, "missing", "q", user, &InvokeOptions{Handler: f.options()}); err == nil {
		t.Error("want a disabled-config error")
	}
	if _, err := OpenCodeAgents(bg, "cfg", "q", user, nil); err == nil {
		t.Error("nil options (no Env) must fail")
	}
}
