package aiserver

import (
	"context"
	"iter"
	"maps"
	"slices"
	"strings"
	"testing"
)

// A handler receives only the tools the active config attaches. ToolHandlers
// (and a Registry) hold the tools of every variation, so the core narrows the
// map to Config.Tools before any handler runs, on Invoke, on Stream and on
// every graph node.

func toolDefs(names ...string) map[string]any {
	tools := map[string]any{}
	for _, n := range names {
		tools[n] = map[string]any{"name": n, "type": "function", "parameters": map[string]any{}}
	}
	return map[string]any{"tools": tools}
}

// capturingHandler records the tool keys of every request it gets, on both paths.
func capturingHandler(seen *[][]string) *ProviderHandler {
	record := func(req HandlerRequest) {
		*seen = append(*seen, slices.Sorted(maps.Keys(req.ToolHandlers)))
	}
	invoke := func(_ context.Context, req HandlerRequest) (HandlerResult, error) {
		record(req)
		return HandlerResult{Output: "ok"}, nil
	}
	stream := func(_ context.Context, req HandlerRequest) iter.Seq2[HandlerStreamEvent, error] {
		return func(yield func(HandlerStreamEvent, error) bool) {
			record(req)
			out := "ok"
			yield(HandlerStreamEvent{Type: EventDone, Output: &out}, nil)
		}
	}
	return createHandler(ProvidesFor{"P", ModeMessages}, invoke, stream, CaptureDefault)
}

func runPath(t *testing.T, path string, cfg map[string]any, tools map[string]ToolHandler) []string {
	t.Helper()
	withFake(t, map[string]any{"cfg": cfg})
	var seen [][]string
	inst := Config(ConfigArgs{Key: "cfg", Handlers: []*ProviderHandler{capturingHandler(&seen)}, ToolHandlers: tools})
	if path == "invoke" {
		if _, err := inst.Invoke(bg, "q", userCtx); err != nil {
			t.Fatal(err)
		}
	} else {
		for _, err := range inst.Stream(bg, "q", userCtx) {
			if err != nil {
				t.Fatal(err)
			}
		}
	}
	if len(seen) != 1 {
		t.Fatalf("handler calls = %d", len(seen))
	}
	return seen[0]
}

func okTool() ToolFunc {
	return func(context.Context, map[string]any) (any, error) { return "ok", nil }
}

func TestScopeUnattachedToolsNeverReachTheHandler(t *testing.T) {
	for _, path := range []string{"invoke", "stream"} {
		t.Run(path, func(t *testing.T) {
			ran := false
			privileged := ToolFunc(func(context.Context, map[string]any) (any, error) { ran = true; return "ran", nil })
			got := runPath(t, path, enabledConfig("P", "", toolDefs("safe")), map[string]ToolHandler{"safe": okTool(), "privileged": privileged})
			if !slices.Equal(got, []string{"safe"}) || ran {
				t.Errorf("tools = %v, privileged ran = %v", got, ran)
			}
		})
	}
}

func TestScopeNoConfigToolsMeansNoTools(t *testing.T) {
	for _, path := range []string{"invoke", "stream"} {
		for name, extra := range map[string]map[string]any{"absent": nil, "empty": toolDefs()} {
			t.Run(path+"/"+name, func(t *testing.T) {
				got := runPath(t, path, enabledConfig("P", "", extra), map[string]ToolHandler{"privileged": okTool()})
				if len(got) != 0 {
					t.Errorf("tools = %v, want none", got)
				}
			})
		}
	}
}

func TestScopeNativeToolsTheSameWay(t *testing.T) {
	for _, path := range []string{"invoke", "stream"} {
		t.Run(path, func(t *testing.T) {
			bash := &NativeTool{ToolName: "Bash"}
			if got := runPath(t, path, enabledConfig("P", "", toolDefs("safe")), map[string]ToolHandler{"safe": okTool(), "bash": bash}); !slices.Equal(got, []string{"safe"}) {
				t.Errorf("unattached native: tools = %v", got)
			}
			if got := runPath(t, path, enabledConfig("P", "", toolDefs("bash")), map[string]ToolHandler{"bash": bash}); !slices.Equal(got, []string{"bash"}) {
				t.Errorf("attached native: tools = %v", got)
			}
		})
	}
}

// A routed node keeps its own tools and the synthetic handoff tools (they are
// on the routed config's Tools), and gets no tool registered for another node.
func TestScopeGraphNodesKeepTheirHandoffTools(t *testing.T) {
	router := nodeCfg("route it")
	for k, v := range toolDefs("safe") {
		router[k] = v
	}
	billing := nodeCfg("billing work")
	for k, v := range toolDefs("ledger") {
		billing[k] = v
	}
	withFake(t, map[string]any{
		"route": map[string]any{"root": "router", "edges": map[string]any{"router": []any{
			map[string]any{"key": "billing"}, map[string]any{"key": "support"},
		}}},
		"router":  router,
		"billing": billing,
		"support": nodeCfg("support work"),
	})
	rh := &recordingHandler{}
	tools := map[string]ToolHandler{"safe": okTool(), "ledger": okTool(), "privileged": okTool()}
	if _, err := Graph("route", GraphOptions{Handlers: []*ProviderHandler{rh.handler()}, ToolHandlers: tools}).Invoke(bg, "q", userCtx); err != nil {
		t.Fatal(err)
	}
	got := rh.calls[0].Req.ToolHandlers
	var handoffs int
	for k := range got {
		if strings.HasPrefix(k, "__handoff_") {
			handoffs++
		}
	}
	_, safe := got["safe"]
	_, ledger := got["ledger"]
	_, privileged := got["privileged"]
	if !safe || handoffs != 2 || ledger || privileged {
		t.Errorf("router tools = %v", slices.Sorted(maps.Keys(got)))
	}
}
