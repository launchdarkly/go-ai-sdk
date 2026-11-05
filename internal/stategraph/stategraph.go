// Package stategraph is the small state-graph runner behind langchainagents:
// named nodes, one static or conditional edge per node, an entry point, and
// a schema that merges each node's update into the state. A node can return
// a *Command to merge an update and jump to a node of its choice instead of
// following its edge. It replaces github.com/smallnest/langgraphgo/graph and
// keeps the semantics langchainagents relied on there.
package stategraph

import (
	"context"
	"errors"
	"fmt"
)

// END is the stop sentinel: an edge or Goto to END finishes the run.
const END = "END"

var (
	// ErrNoEntryPoint means Compile was called before SetEntryPoint.
	ErrNoEntryPoint = errors.New("stategraph: entry point not set")
	// ErrNodeNotFound means an entry point, edge or Goto names no node.
	ErrNodeNotFound = errors.New("stategraph: node not found")
	// ErrNoOutgoingEdge means a node finished without an edge or a Goto.
	ErrNoOutgoingEdge = errors.New("stategraph: no outgoing edge")
)

// NodeFunc runs one node over the current state. It returns the update to
// merge, or a *Command to merge an update and route explicitly.
type NodeFunc func(ctx context.Context, state any) (any, error)

// Command is a node result that routes the run. Goto overrides the node's
// edge; END finishes the run. A nil Update leaves the state unchanged.
type Command struct {
	Update any
	Goto   string
}

// Schema builds the initial state and merges node updates into it.
type Schema interface {
	Init() any
	Update(current, update any) (any, error)
}

// Graph is built with AddNode / AddEdge / AddConditionalEdge /
// SetEntryPoint / SetSchema, then compiled into a Runnable.
type Graph struct {
	nodes  map[string]NodeFunc
	edges  map[string]string
	conds  map[string]func(context.Context, any) string
	entry  string
	schema Schema
}

// New returns an empty graph.
func New() *Graph {
	return &Graph{
		nodes: map[string]NodeFunc{},
		edges: map[string]string{},
		conds: map[string]func(context.Context, any) string{},
	}
}

// AddNode registers fn under name, replacing any node of the same name.
// The description is accepted for parity with LangGraph and not used.
func (g *Graph) AddNode(name, _ string, fn func(context.Context, any) (any, error)) {
	g.nodes[name] = fn
}

// AddEdge routes from to to after from runs.
func (g *Graph) AddEdge(from, to string) { g.edges[from] = to }

// AddConditionalEdge routes from to whatever condition returns, evaluated
// on the state after from's update is merged.
func (g *Graph) AddConditionalEdge(from string, condition func(context.Context, any) string) {
	g.conds[from] = condition
}

// SetEntryPoint names the first node to run.
func (g *Graph) SetEntryPoint(name string) { g.entry = name }

// SetSchema sets how updates merge into the state. Without a schema each
// update replaces the state.
func (g *Graph) SetSchema(schema Schema) { g.schema = schema }

// Compile checks the topology and returns a runnable graph.
func (g *Graph) Compile() (*Runnable, error) {
	if g.entry == "" {
		return nil, ErrNoEntryPoint
	}
	if _, ok := g.nodes[g.entry]; !ok {
		return nil, fmt.Errorf("%w: entry point %s", ErrNodeNotFound, g.entry)
	}
	for from, to := range g.edges {
		if _, ok := g.nodes[from]; !ok {
			return nil, fmt.Errorf("%w: edge from %s", ErrNodeNotFound, from)
		}
		if _, ok := g.nodes[to]; !ok && to != END {
			return nil, fmt.Errorf("%w: edge %s -> %s", ErrNodeNotFound, from, to)
		}
	}
	for from := range g.conds {
		if _, ok := g.nodes[from]; !ok {
			return nil, fmt.Errorf("%w: conditional edge from %s", ErrNodeNotFound, from)
		}
	}
	return &Runnable{g: g}, nil
}

// Runnable is a compiled graph.
type Runnable struct{ g *Graph }

// Invoke runs the graph from the entry point until a node routes to END,
// and returns the final state. A node error is returned wrapped with the
// node's name; a node panic is returned as an error.
func (r *Runnable) Invoke(ctx context.Context, initial any) (any, error) {
	state := initial
	if s := r.g.schema; s != nil {
		var err error
		if state, err = s.Update(s.Init(), initial); err != nil {
			return nil, fmt.Errorf("stategraph: initialize state: %w", err)
		}
	}
	for name := r.g.entry; name != END; {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		fn, ok := r.g.nodes[name]
		if !ok {
			return nil, fmt.Errorf("%w: %s", ErrNodeNotFound, name)
		}
		result, err := runNode(ctx, name, fn, state)
		if err != nil {
			return nil, err
		}
		update, next := result, ""
		if cmd, ok := result.(*Command); ok {
			update, next = cmd.Update, cmd.Goto
		}
		if state, err = r.merge(state, update); err != nil {
			return nil, fmt.Errorf("stategraph: merge update from node %s: %w", name, err)
		}
		if next == "" {
			if next, err = r.edge(ctx, name, state); err != nil {
				return nil, err
			}
		}
		name = next
	}
	return state, nil
}

func (r *Runnable) merge(state, update any) (any, error) {
	if r.g.schema == nil {
		if update == nil {
			return state, nil
		}
		return update, nil
	}
	return r.g.schema.Update(state, update)
}

func (r *Runnable) edge(ctx context.Context, name string, state any) (string, error) {
	if to, ok := r.g.edges[name]; ok {
		return to, nil
	}
	if cond, ok := r.g.conds[name]; ok {
		to := cond(ctx, state)
		if to == "" {
			return "", fmt.Errorf("stategraph: conditional edge from %s returned no node", name)
		}
		return to, nil
	}
	return "", fmt.Errorf("%w: %s", ErrNoOutgoingEdge, name)
}

func runNode(ctx context.Context, name string, fn NodeFunc, state any) (result any, err error) {
	defer func() {
		if p := recover(); p != nil {
			result, err = nil, fmt.Errorf("stategraph: panic in node %s: %v", name, p)
		}
	}()
	result, err = fn(ctx, state)
	if err != nil {
		return nil, fmt.Errorf("stategraph: error in node %s: %w", name, err)
	}
	return result, nil
}
