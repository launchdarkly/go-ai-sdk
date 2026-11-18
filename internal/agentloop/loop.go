// Package agentloop is the agent run loop shared by the agent handlers that
// stand in for an agents SDK: it drives a model turn by turn, executes
// function tools, switches agents on a handoff call, and stops at the first
// turn with no function calls. Hooks mirror the agents SDKs' agent_start /
// agent_end / agent_handoff run events.
//
// The wire is the Model's: what one turn sends and how a follow-up turn
// carries the conversation (a previous response id, or the full message list)
// are not the loop's concern.
package agentloop

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"strings"
	"unicode"

	"go.opentelemetry.io/otel/trace"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/handlerkit"
	"github.com/launchdarkly/go-ai-sdk/internal/kitbase"
)

// DefaultMaxTurns matches the Agents SDK's DEFAULT_MAX_TURNS.
const DefaultMaxTurns = 10

// Agent is one agent in a run, as the loop sees it. The handler package owns
// the concrete type and how its requests are built.
type Agent interface {
	// ModelName is the requested model, for the chat span.
	ModelName() string
	// StructuredOutput reports that the final text is JSON to decode.
	StructuredOutput() bool
	// Handoff returns the agent the transfer tool switches to and the name
	// the handoff output reports, or ok=false when tool is not a handoff.
	Handoff(tool string) (target Agent, name string, ok bool)
	// Tool returns the function behind an offered function tool, or
	// ok=false when name was not offered or has no local function.
	Tool(name string) (fn aiserver.ToolFunc, ok bool)
}

// ToolCall is one function call the model requested.
type ToolCall struct {
	ID, Name, Arguments string
}

// ToolOutput answers one function call.
type ToolOutput struct {
	CallID, Output string
}

// Input is what the run added since the previous turn: the seed turns on the
// first turn, then the outputs answering the previous turn's calls.
type Input struct {
	Turns   []aiserver.Message
	Outputs []ToolOutput
}

// Tokens are a turn's token counts, summed exactly over a run.
type Tokens struct{ Input, Output int64 }

// Raw is the handler-result usage. It omits cache keys: the input count
// already includes them and handlerkit.ParseUsage would fold them in a
// second time.
func (u Tokens) Raw() map[string]any {
	return map[string]any{"input_tokens": float64(u.Input), "output_tokens": float64(u.Output)}
}

// Turn is one successful model call.
type Turn struct {
	Text   string     // the turn's output text
	Calls  []ToolCall // function calls, in output order
	Tokens Tokens
	Usage  kitbase.SpanUsage // for the chat span
	Model  string            // the model the API reported, else the requested one
	Finish string            // semconv finish reason
	// Native are the built-in tools the provider ran this turn; each is
	// tracked as a tool call.
	Native []*aiserver.NativeTool
}

// Model performs one model call for the active agent. A Model serves one run
// and may keep state between turns.
type Model interface {
	// Turn calls the model once for agent a, sending what the run added
	// since the previous turn (in). onText is nil for a blocking call;
	// otherwise it receives each text delta, and stopped reports that it
	// returned false.
	Turn(ctx context.Context, a Agent, in Input, onText func(string) bool) (t Turn, stopped bool, err error)
}

// Hooks are the run lifecycle events; nil members are skipped.
type Hooks struct {
	AgentStart func(a Agent)
	AgentEnd   func(a Agent, output any)
	Handoff    func(from, to Agent)
	// TurnEnd sees every successful model turn of the active agent.
	TurnEnd func(a Agent, t Turn)
}

// Runner drives one run.
type Runner struct {
	Model    Model
	MaxTurns int
	Hooks    Hooks
	// Span records chat / execute_tool child spans; nil records none.
	Span kitbase.Run
	// OnText streams text deltas; nil makes blocking calls.
	OnText func(string) bool
	// ErrMaxTurns and ErrToolNotOffered are the handler package's sentinels,
	// and Prefix starts its tool error messages ("openaiagents").
	ErrMaxTurns, ErrToolNotOffered error
	Prefix                         string
}

// Result is a finished run.
type Result struct {
	Output any    // final text, or the decoded JSON for structured output
	Text   string // final text; when streaming, every streamed delta
	Tokens Tokens
	Model  string // model the API reported on the last turn
}

// Run executes start until a turn produces no function calls. stopped
// reports that the stream consumer stopped iterating.
func (r *Runner) Run(ctx context.Context, start Agent, turns []aiserver.Message) (res Result, stopped bool, err error) {
	current := start
	r.emitStart(current)
	in := Input{Turns: turns}
	var streamed strings.Builder
	var onText func(string) bool
	if r.OnText != nil {
		onText = func(t string) bool {
			streamed.WriteString(t)
			return r.OnText(t)
		}
	}
	for turn := 1; ; turn++ {
		if turn > r.MaxTurns {
			return res, false, fmt.Errorf("%w (%d)", r.ErrMaxTurns, r.MaxTurns)
		}
		var chat trace.Span
		if r.Span != nil {
			chat = r.Span.StartChat(current.ModelName())
		}
		t, stop, terr := r.Model.Turn(ctx, current, in, onText)
		if stop {
			return res, true, nil
		}
		if terr != nil {
			return res, false, terr
		}
		res.Tokens.Input += t.Tokens.Input
		res.Tokens.Output += t.Tokens.Output
		res.Model = t.Model
		if r.Span != nil {
			r.Span.EndChat(chat, res.Model, t.Usage, t.Finish)
		}
		if r.Hooks.TurnEnd != nil {
			r.Hooks.TurnEnd(current, t)
		}
		for _, nt := range t.Native {
			handlerkit.TrackNativeTool(nt)
		}

		if len(t.Calls) == 0 {
			res.Text = t.Text
			res.Output = res.Text
			if r.OnText != nil {
				res.Text = streamed.String() // done.output spans every streamed turn
				res.Output = res.Text
			} else if current.StructuredOutput() {
				var v any
				if json.Unmarshal([]byte(res.Text), &v) == nil {
					res.Output = v // the parsed object, not a re-serialized string
				} // else best effort: the raw text
			}
			if r.Hooks.AgentEnd != nil {
				r.Hooks.AgentEnd(current, res.Output)
			}
			return res, false, nil
		}
		outputs, next, terr := r.runCalls(ctx, current, t.Calls)
		if terr != nil {
			return res, false, terr
		}
		if next != nil {
			if r.Hooks.Handoff != nil {
				r.Hooks.Handoff(current, next)
			}
			current = next
			r.emitStart(current)
		}
		in = Input{Outputs: outputs}
	}
}

func (r *Runner) emitStart(a Agent) {
	if r.Hooks.AgentStart != nil {
		r.Hooks.AgentStart(a)
	}
}

// runCalls answers every function call of one turn. The first handoff call
// selects the next agent; later handoff calls in the same turn are ignored,
// as in the Agents SDK. A call for a tool that was not offered is an error.
func (r *Runner) runCalls(ctx context.Context, a Agent, calls []ToolCall) ([]ToolOutput, Agent, error) {
	var outputs []ToolOutput
	var next Agent
	for _, call := range calls {
		if target, name, ok := a.Handoff(call.Name); ok {
			out := "Multiple handoffs detected, ignoring this one."
			if next == nil {
				next = target
				b, _ := json.Marshal(map[string]string{"assistant": name})
				out = string(b)
			}
			outputs = append(outputs, ToolOutput{CallID: call.ID, Output: out})
			continue
		}
		var span trace.Span
		if r.Span != nil {
			span = r.Span.StartTool(call.Name, call.ID)
			r.Span.ToolArguments(span, call.Arguments)
		}
		res, err := r.callTool(ctx, a, call)
		if err != nil {
			return nil, nil, err // Run.Fail ends the open tool span
		}
		if r.Span != nil {
			r.Span.ToolResult(span, res)
			r.Span.EndTool(span, nil)
		}
		outputs = append(outputs, ToolOutput{CallID: call.ID, Output: aiserver.ToolResultString(res)})
	}
	return outputs, next, nil
}

// callTool runs one function tool. Only tools offered to this agent run;
// the raw handler map is never consulted for anything else.
func (r *Runner) callTool(ctx context.Context, a Agent, call ToolCall) (any, error) {
	fn, offered := a.Tool(call.Name)
	if !offered || fn == nil {
		return nil, fmt.Errorf("%w: %q", r.ErrToolNotOffered, call.Name)
	}
	args, err := parseArguments(call.Arguments)
	if err != nil {
		return nil, fmt.Errorf("%s: tool %q arguments: %w", r.Prefix, call.Name, err)
	}
	res, err := fn(ctx, args)
	if err != nil {
		return nil, fmt.Errorf("%s: tool %q: %w", r.Prefix, call.Name, err) // propagate
	}
	return res, nil
}

// parseArguments decodes a function call's JSON argument object; blank
// arguments decode to nil.
func parseArguments(raw string) (map[string]any, error) {
	if strings.TrimSpace(raw) == "" {
		return nil, nil
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(raw), &args); err != nil {
		return nil, err
	}
	return args, nil
}

// SnakeParameters adds each camelCase key (`topP`, `maxTokens`) under its
// snake_case name, so a config written for the JavaScript packages'
// ModelSettings spelling still forwards. An explicit snake_case key wins.
func SnakeParameters(in map[string]any) map[string]any {
	out := maps.Clone(in)
	for k, v := range in {
		if strings.ToLower(k) == k {
			continue
		}
		var b strings.Builder
		for _, c := range k {
			if unicode.IsUpper(c) {
				b.WriteByte('_')
				c = unicode.ToLower(c)
			}
			b.WriteRune(c)
		}
		if _, set := in[b.String()]; !set {
			out[b.String()] = v
		}
	}
	return out
}

// MaxTurns reads `max_turns` / `maxTurns` from model parameters, else def,
// else DefaultMaxTurns.
func MaxTurns(parameters map[string]any, def int) int {
	for _, k := range []string{"max_turns", "maxTurns"} {
		if v, ok := parameters[k].(float64); ok && v >= 1 {
			return int(v)
		}
	}
	if def < 1 {
		return DefaultMaxTurns
	}
	return def
}
