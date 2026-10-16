package langchainagents

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"

	"go.opentelemetry.io/otel/trace"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/langchainwire"
	"github.com/launchdarkly/go-ai-sdk/internal/stategraph"
	"github.com/launchdarkly/go-ai-sdk/llms"
)

// This file is the Go equivalent of LangGraph's create_react_agent /
// LangChain's createAgent, built on internal/stategraph: an "agent" node
// calls the model, a "tools" node (the ToolNode equivalent) runs the
// requested tools, and a tools condition loops between them until the model
// answers without tool calls.

const (
	messagesKey = "messages"
	gotoKey     = "goto"
	agentNode   = "agent"
	toolsNode   = "tools"
	// recursionLimit bounds node executions per graph run, LangGraph's
	// default recursion_limit; stategraph has none of its own.
	recursionLimit = 25
)

// errStopped means the stream consumer stopped iterating mid-turn.
var errStopped = errors.New("langchainagents: stream abandoned by consumer")

// messagesSchema is the WorkflowState equivalent: `messages` is appended to
// (LangGraph's add_messages, for messages without ids) and any other key is
// overwritten. Each update returns a new map, so a state a node was given is
// never changed under it. State is map[string]any carried as `any`.
type messagesSchema struct{}

func newMessagesSchema() messagesSchema { return messagesSchema{} }

func (messagesSchema) Init() any { return map[string]any{} }

func (messagesSchema) Update(current, update any) (any, error) {
	u, ok := update.(map[string]any)
	if !ok { // a Command without an Update
		return current, nil
	}
	c, _ := current.(map[string]any)
	next := make(map[string]any, len(c)+len(u))
	maps.Copy(next, c)
	for k, v := range u {
		if k != messagesKey {
			next[k] = v
			continue
		}
		var add []llms.MessageContent
		switch m := v.(type) {
		case []llms.MessageContent:
			add = m
		case llms.MessageContent:
			add = []llms.MessageContent{m}
		default:
			return nil, fmt.Errorf("langchainagents: %q update is %T, not []llms.MessageContent", messagesKey, v)
		}
		if cur, ok := next[messagesKey].([]llms.MessageContent); ok {
			next[messagesKey] = append(cur, add...)
		} else {
			next[messagesKey] = add
		}
	}
	return next, nil
}

func messagesOf(state any) []llms.MessageContent {
	m, _ := state.(map[string]any)
	msgs, _ := m[messagesKey].([]llms.MessageContent)
	return msgs
}

// stateGraph is the part of *stategraph.Graph this package uses; tests
// wrap the real constructor to observe the topology it builds.
type stateGraph interface {
	AddNode(name, description string, fn func(context.Context, any) (any, error))
	AddEdge(from, to string)
	AddConditionalEdge(from string, condition func(context.Context, any) string)
	SetEntryPoint(name string)
	SetSchema(schema stategraph.Schema)
	Compile() (*stategraph.Runnable, error)
}

var newStateGraph = func() stateGraph { return stategraph.New() }

// toolExec runs one tool call and returns the tool message; a handoff also
// returns the Command that routes the enclosing graph.
type toolExec func(ctx context.Context, tc llms.ToolCall) (llms.MessageContent, *stategraph.Command, error)

// reactAgent is what createReactAgent compiles: model, tools and system
// prompt (create_react_agent's model, tools, prompt), plus the run's hooks.
// The node functions write the run's results back into it, so one
// reactAgent serves one run.
type reactAgent struct {
	model     llms.Model
	tools     []llms.Tool
	exec      toolExec
	system    string
	opts      []llms.CallOption
	modelName string
	// run receives chat and execute_tool spans (the handler); nil in a
	// native graph node.
	run *aiserver.HandlerRun
	// onText streams text deltas; nil when blocking.
	onText func(string) bool

	usage    aiserver.SpanUsage
	streamed strings.Builder
	steps    int
	stopped  bool
	err      error // first node error, returned unwrapped
}

// createAgent is the agent factory; tests replace it to observe its input.
var createAgent = createReactAgent

func createReactAgent(a *reactAgent) (*stategraph.Runnable, error) {
	g := newStateGraph()
	g.SetSchema(newMessagesSchema())
	g.AddNode(agentNode, "Calls the model", a.agentStep)
	g.SetEntryPoint(agentNode)
	if len(a.tools) == 0 {
		g.AddEdge(agentNode, stategraph.END)
		return g.Compile()
	}
	g.AddNode(toolsNode, "Runs the requested tools", a.toolsStep)
	g.AddConditionalEdge(agentNode, toolsCondition)
	g.AddEdge(toolsNode, agentNode)
	return g.Compile()
}

func (a *reactAgent) fail(err error) (any, error) {
	if a.err == nil {
		a.err = err
	}
	return nil, err
}

func (a *reactAgent) step() error {
	a.steps++
	if a.steps > recursionLimit {
		return fmt.Errorf("%w (%d)", ErrRecursionLimit, recursionLimit)
	}
	return nil
}

func (a *reactAgent) agentStep(ctx context.Context, state any) (any, error) {
	if err := a.step(); err != nil {
		return a.fail(err)
	}
	msgs := messagesOf(state)
	if a.system != "" {
		msgs = append([]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeSystem, a.system)}, msgs...)
	}
	opts := a.opts
	if len(a.tools) > 0 {
		opts = append(slices.Clone(opts), llms.WithTools(a.tools))
	}
	var chat trace.Span
	if a.run != nil {
		chat = a.run.StartChat(a.modelName)
	}
	resp, text, stopped, err := langchainwire.Turn(ctx, a.model, msgs, opts, a.onText)
	if langchainwire.IsEmptyResponse(err) && len(msgs) > 0 && msgs[len(msgs)-1].Role == llms.ChatMessageTypeTool {
		// Claude can end its turn with no content right after a tool result (seen
		// live after a handoff); Anthropic's advice is a new user turn, not a resend.
		retry := append(slices.Clone(msgs), llms.TextParts(llms.ChatMessageTypeHuman, "Please continue."))
		resp, text, stopped, err = langchainwire.Turn(ctx, a.model, retry, opts, a.onText)
	}
	if stopped {
		a.stopped = true
		return nil, errStopped
	}
	if err != nil {
		return a.fail(err)
	}
	u := langchainwire.ResponseUsage(resp)
	a.usage.Add(u)
	if a.run != nil {
		a.run.EndChat(chat, a.modelName, u, langchainwire.FinishReason(resp))
	}
	a.streamed.WriteString(text)
	return map[string]any{messagesKey: []llms.MessageContent{resp.AssistantMessage()}}, nil
}

func (a *reactAgent) toolsStep(ctx context.Context, state any) (any, error) {
	if err := a.step(); err != nil {
		return a.fail(err)
	}
	msgs := messagesOf(state)
	var out []llms.MessageContent
	target := ""
	for _, tc := range toolCalls(msgs[len(msgs)-1]) {
		m, cmd, err := a.exec(ctx, tc)
		if err != nil {
			return a.fail(err)
		}
		out = append(out, m)
		if cmd != nil && target == "" {
			target = cmd.Goto
		}
	}
	update := map[string]any{messagesKey: out}
	if target == "" {
		return update, nil
	}
	// A handoff ends this agent; the caller routes the enclosing graph.
	update[gotoKey] = target
	return &stategraph.Command{Update: update, Goto: stategraph.END}, nil
}

// toolsCondition is LangGraph's tools_condition: run the tools when the
// last message requests any, otherwise finish.
func toolsCondition(_ context.Context, state any) string {
	msgs := messagesOf(state)
	if len(msgs) > 0 && len(toolCalls(msgs[len(msgs)-1])) > 0 {
		return toolsNode
	}
	return stategraph.END
}

func toolCalls(m llms.MessageContent) []llms.ToolCall {
	if m.Role != llms.ChatMessageTypeAI {
		return nil
	}
	var out []llms.ToolCall
	for _, p := range m.Parts {
		if tc, ok := p.(llms.ToolCall); ok {
			out = append(out, tc)
		}
	}
	return out
}

// agentResult is a finished agent run: every message (input first), and
// the node a handoff routed to ("" when none).
type agentResult struct {
	messages []llms.MessageContent
	next     string
}

// invoke builds the agent and runs it from msgs.
func (a *reactAgent) invoke(ctx context.Context, msgs []llms.MessageContent) (agentResult, error) {
	runnable, err := createAgent(a)
	if err != nil {
		return agentResult{}, fmt.Errorf("langchainagents: compile agent: %w", err)
	}
	// Clip so the reducer's appends never write into the caller's array.
	final, err := runnable.Invoke(ctx, map[string]any{messagesKey: slices.Clip(msgs)})
	if a.stopped {
		return agentResult{}, errStopped
	}
	if a.err != nil {
		return agentResult{}, a.err
	}
	if err != nil {
		return agentResult{}, err
	}
	m, _ := final.(map[string]any)
	next, _ := m[gotoKey].(string)
	return agentResult{messages: messagesOf(final), next: next}, nil
}

// lastAIText is the text of the last AI message: its text parts joined,
// so array content is concatenated.
func lastAIText(msgs []llms.MessageContent) string {
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role != llms.ChatMessageTypeAI {
			continue
		}
		var b strings.Builder
		for _, p := range msgs[i].Parts {
			if t, ok := p.(llms.TextContent); ok {
				b.WriteString(t.Text)
			}
		}
		return b.String()
	}
	return ""
}
