package stategraph

import (
	"context"
	"errors"
	"slices"
	"strings"
	"testing"
)

// appendSchema appends string updates to a []string state.
type appendSchema struct{}

func (appendSchema) Init() any { return []string(nil) }

func (appendSchema) Update(current, update any) (any, error) {
	c, _ := current.([]string)
	switch u := update.(type) {
	case nil:
		return c, nil
	case []string:
		return append(slices.Clip(c), u...), nil
	case string:
		return append(slices.Clip(c), u), nil
	default:
		return nil, errors.New("unexpected update")
	}
}

func emit(s string) func(context.Context, any) (any, error) {
	return func(context.Context, any) (any, error) { return s, nil }
}

func run(t *testing.T, g *Graph, initial any) (any, error) {
	t.Helper()
	r, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	return r.Invoke(context.Background(), initial)
}

func TestStaticEdgesRunInOrderAndMerge(t *testing.T) {
	g := New()
	g.SetSchema(appendSchema{})
	g.AddNode("a", "", emit("a"))
	g.AddNode("b", "", emit("b"))
	g.AddEdge("a", "b")
	g.AddEdge("b", END)
	g.SetEntryPoint("a")
	got, err := run(t, g, []string{"in"})
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"in", "a", "b"}; !slices.Equal(got.([]string), want) {
		t.Fatalf("state = %v, want %v", got, want)
	}
}

func TestConditionalEdgeSeesMergedState(t *testing.T) {
	g := New()
	g.SetSchema(appendSchema{})
	calls := 0
	g.AddNode("agent", "", func(context.Context, any) (any, error) {
		calls++
		return "agent", nil
	})
	g.AddNode("tools", "", emit("tools"))
	g.AddConditionalEdge("agent", func(_ context.Context, state any) string {
		s := state.([]string)
		if s[len(s)-1] != "agent" {
			t.Errorf("condition saw %v before the agent update was merged", s)
		}
		if calls < 2 {
			return "tools"
		}
		return END
	})
	g.AddEdge("tools", "agent")
	g.SetEntryPoint("agent")
	got, err := run(t, g, nil)
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"agent", "tools", "agent"}; !slices.Equal(got.([]string), want) {
		t.Fatalf("state = %v, want %v", got, want)
	}
}

func TestCommandGotoOverridesEdge(t *testing.T) {
	g := New()
	g.SetSchema(appendSchema{})
	g.AddNode("a", "", func(context.Context, any) (any, error) {
		return &Command{Update: "a", Goto: "c"}, nil
	})
	g.AddNode("b", "", emit("b"))
	g.AddNode("c", "", emit("c"))
	g.AddEdge("a", "b")
	g.AddEdge("b", END)
	g.AddEdge("c", END)
	g.SetEntryPoint("a")
	got, err := run(t, g, nil)
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"a", "c"}; !slices.Equal(got.([]string), want) {
		t.Fatalf("state = %v, want %v (Goto must override the static edge)", got, want)
	}
}

func TestCommandGotoEndAndNilUpdate(t *testing.T) {
	g := New()
	g.SetSchema(appendSchema{})
	g.AddNode("a", "", func(context.Context, any) (any, error) { return &Command{Goto: END}, nil })
	g.AddNode("b", "", emit("b"))
	g.AddEdge("a", "b")
	g.SetEntryPoint("a")
	got, err := run(t, g, []string{"in"})
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"in"}; !slices.Equal(got.([]string), want) {
		t.Fatalf("state = %v, want %v", got, want)
	}
}

func TestNoSchemaReplacesState(t *testing.T) {
	g := New()
	g.AddNode("a", "", emit("replaced"))
	g.AddEdge("a", END)
	g.SetEntryPoint("a")
	got, err := run(t, g, "initial")
	if err != nil {
		t.Fatal(err)
	}
	if got != "replaced" {
		t.Fatalf("state = %v, want replaced", got)
	}
}

func TestNodeErrorIsWrappedWithItsName(t *testing.T) {
	boom := errors.New("boom")
	g := New()
	g.AddNode("agent", "", func(context.Context, any) (any, error) { return nil, boom })
	g.AddEdge("agent", END)
	g.SetEntryPoint("agent")
	_, err := run(t, g, nil)
	if !errors.Is(err, boom) || !strings.Contains(err.Error(), "error in node agent") {
		t.Fatalf("err = %v, want boom wrapped with the node name", err)
	}
}

func TestNodePanicBecomesError(t *testing.T) {
	g := New()
	g.AddNode("agent", "", func(context.Context, any) (any, error) { panic("kaboom") })
	g.AddEdge("agent", END)
	g.SetEntryPoint("agent")
	_, err := run(t, g, nil)
	if err == nil || !strings.Contains(err.Error(), "panic in node agent: kaboom") {
		t.Fatalf("err = %v, want the panic as an error", err)
	}
}

func TestCancelledContextStopsBeforeTheNextNode(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	ran := []string{}
	g := New()
	g.AddNode("a", "", func(context.Context, any) (any, error) {
		ran = append(ran, "a")
		cancel()
		return nil, nil
	})
	g.AddNode("b", "", func(context.Context, any) (any, error) {
		ran = append(ran, "b")
		return nil, nil
	})
	g.AddEdge("a", "b")
	g.AddEdge("b", END)
	g.SetEntryPoint("a")
	r, err := g.Compile()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := r.Invoke(ctx, nil); !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	if !slices.Equal(ran, []string{"a"}) {
		t.Fatalf("ran %v, want only a", ran)
	}
}

func TestRunErrors(t *testing.T) {
	t.Run("no outgoing edge", func(t *testing.T) {
		g := New()
		g.AddNode("a", "", emit("a"))
		g.SetEntryPoint("a")
		if _, err := run(t, g, nil); !errors.Is(err, ErrNoOutgoingEdge) {
			t.Fatalf("err = %v, want ErrNoOutgoingEdge", err)
		}
	})
	t.Run("goto unknown node", func(t *testing.T) {
		g := New()
		g.AddNode("a", "", func(context.Context, any) (any, error) { return &Command{Goto: "nope"}, nil })
		g.SetEntryPoint("a")
		if _, err := run(t, g, nil); !errors.Is(err, ErrNodeNotFound) {
			t.Fatalf("err = %v, want ErrNodeNotFound", err)
		}
	})
	t.Run("condition returns empty", func(t *testing.T) {
		g := New()
		g.AddNode("a", "", emit("a"))
		g.AddConditionalEdge("a", func(context.Context, any) string { return "" })
		g.SetEntryPoint("a")
		if _, err := run(t, g, nil); err == nil {
			t.Fatal("want an error for an empty conditional route")
		}
	})
}

func TestCompileErrors(t *testing.T) {
	cases := map[string]func(*Graph){
		"no entry point":      func(g *Graph) {},
		"unknown entry point": func(g *Graph) { g.SetEntryPoint("nope") },
		"edge to unknown":     func(g *Graph) { g.SetEntryPoint("a"); g.AddEdge("a", "nope") },
		"edge from unknown":   func(g *Graph) { g.SetEntryPoint("a"); g.AddEdge("nope", END) },
		"cond from unknown": func(g *Graph) {
			g.SetEntryPoint("a")
			g.AddConditionalEdge("nope", func(context.Context, any) string { return END })
		},
	}
	for name, setup := range cases {
		t.Run(name, func(t *testing.T) {
			g := New()
			g.AddNode("a", "", emit("a"))
			setup(g)
			if _, err := g.Compile(); err == nil {
				t.Fatal("Compile succeeded, want an error")
			}
		})
	}
}
