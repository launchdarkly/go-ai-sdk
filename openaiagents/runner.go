package openaiagents

import (
	"context"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"maps"
	"regexp"
	"slices"
	"strings"
	"unicode"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"go.opentelemetry.io/otel/trace"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/openaiwire"
)

// This file is the Go stand-in for the OpenAI Agents SDK's Agent / Runner:
// an agent is instructions + model + tools + handoffs + output type, and the
// runner drives the Responses API turn by turn, executing function tools,
// switching agents on a handoff call, and stopping at the first turn with no
// function calls. Hooks mirror the SDK's agent_start / agent_end /
// agent_handoff run events.

// defaultMaxTurns matches the Agents SDK's DEFAULT_MAX_TURNS.
const defaultMaxTurns = 10

// handoffPrefix is the Agents SDK's default handoff tool-name prefix.
const handoffPrefix = "transfer_to_"

// forwardedParameters are the config.model.parameters keys sent verbatim:
// the Responses API fields behind the JavaScript package's ModelSettings
// allowlist. Handler-owned fields (model, input, instructions, tools, text,
// previous_response_id) are deliberately absent; `verbosity` is mapped to
// text.verbosity and `max_turns` caps the run instead.
var forwardedParameters = []string{
	"context_management", "max_output_tokens", "parallel_tool_calls", "prompt_cache_retention",
	"reasoning", "store", "temperature", "tool_choice", "top_p", "truncation",
}

// agent is one agent in a run.
type agent struct {
	name         string // sanitized; what the model sees in handoff tool names
	key          string // graph node key; "" for the single-config handler
	instructions string
	model        string
	parameters   map[string]any
	tools        map[string]aiserver.Tool        // function tools with a local ToolFunc
	native       map[string]aiserver.Tool        // hosted tools, keyed like tools
	handlers     map[string]aiserver.ToolHandler // ToolFunc / NativeTool by tool key
	handoffs     []handoff
	outputType   *outputType
}

// handoff is a transfer tool that switches the run to target.
type handoff struct {
	tool        string
	description string
	target      *agent
}

var unsafeNameChars = regexp.MustCompile(`[^a-zA-Z0-9_-]`)

// sanitizeName makes a node key a valid agent / function-name fragment.
func sanitizeName(key string) string {
	s := unsafeNameChars.ReplaceAllString(key, "_")
	if len(s) > 64 {
		s = s[:64]
	}
	return s
}

// newAgent builds an agent from a config. Function tools are the config
// tools with a local ToolFunc; config tools whose handler is a
// NativeTool become hosted Responses API tools of type NativeTool.ToolName.
func newAgent(name, instructions string, cfg aiserver.AiConfig, handlers map[string]aiserver.ToolHandler) *agent {
	a := &agent{
		name: name, instructions: instructions, model: cfg.Model.Name,
		parameters: snakeParameters(cfg.Model.Parameters),
		tools:      aiserver.CallableTools(cfg, handlers),
		native:     map[string]aiserver.Tool{},
		handlers:   handlers,
	}
	for key, t := range cfg.Tools {
		if _, ok := handlers[key].(*aiserver.NativeTool); ok {
			a.native[key] = t
		}
	}
	return a
}

// addHandoff offers target as a transfer tool. The description names the
// target and appends the edge's handoff description when set.
func (a *agent) addHandoff(target *agent, targetKey, edgeDescription string) {
	tool := handoffPrefix + target.name
	taken := slices.ContainsFunc(a.handoffs, func(h handoff) bool { return h.tool == tool })
	if len(tool) > 64 || taken {
		// Over the 64-character limit, or two keys sanitize alike (a.b, a_b):
		// cut and suffix a hash of the key so each target keeps its own tool.
		h := fnv.New32a()
		h.Write([]byte(targetKey))
		suffix := fmt.Sprintf("_%08x", h.Sum32())
		tool = tool[:min(len(tool), 64-len(suffix))] + suffix
	}
	desc := "Handoff to the " + targetKey + " agent to handle the request."
	if edgeDescription != "" {
		desc += " " + edgeDescription
	}
	a.handoffs = append(a.handoffs, handoff{tool: tool, description: desc, target: target})
}

// request builds one turn's parameters. Instructions, tools and text.format
// are sent every turn because a handoff changes the active agent.
func (a *agent) request(input responses.ResponseInputParam, previousResponseID string) responses.ResponseNewParams {
	p := responses.ResponseNewParams{
		Model: a.model,
		Input: responses.ResponseNewParamsInputUnion{OfInputItemList: input},
	}
	if previousResponseID != "" {
		p.PreviousResponseID = param.NewOpt(previousResponseID)
	}
	if a.instructions != "" {
		p.Instructions = param.NewOpt(a.instructions)
	}
	p.Tools = openaiwire.FunctionTools(a.tools)
	for _, key := range slices.Sorted(maps.Keys(a.native)) {
		p.Tools = append(p.Tools, openaiwire.HostedTool(a.handlers[key].(*aiserver.NativeTool).ToolName, a.native[key].CustomParameters))
	}
	for _, h := range a.handoffs {
		p.Tools = append(p.Tools, openaiwire.FunctionTool(h.tool, h.description,
			map[string]any{"type": "object", "properties": map[string]any{}, "additionalProperties": false}))
	}
	if a.outputType != nil {
		p.Text.Format = responses.ResponseFormatTextConfigUnionParam{OfJSONSchema: &responses.ResponseFormatTextJSONSchemaConfigParam{
			Name: a.outputType.Name, Schema: a.outputType.Schema, Strict: param.NewOpt(a.outputType.Strict),
		}}
	}
	verbosity, _ := a.parameters["verbosity"].(string)
	if text, ok := a.parameters["text"].(map[string]any); ok {
		if v, ok := text["verbosity"].(string); ok {
			verbosity = v // an explicit text.verbosity wins, as in the JavaScript package
		}
	}
	if verbosity != "" {
		p.Text.Verbosity = responses.ResponseTextConfigVerbosity(verbosity)
	}
	if extra := openaiwire.ModelParameters(a.parameters, forwardedParameters); len(extra) > 0 {
		p.SetExtraFields(extra)
	}
	return p
}

// snakeParameters adds each camelCase key (`topP`, `maxTokens`) under its
// snake_case name, so a config written for the JavaScript package's
// ModelSettings spelling still forwards. An explicit snake_case key wins.
func snakeParameters(in map[string]any) map[string]any {
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

// maxTurns reads `max_turns` / `maxTurns` from model parameters, else def.
func maxTurns(parameters map[string]any, def int) int {
	for _, k := range []string{"max_turns", "maxTurns"} {
		if v, ok := parameters[k].(float64); ok && v >= 1 {
			return int(v)
		}
	}
	if def < 1 {
		return defaultMaxTurns
	}
	return def
}

// hooks are the run lifecycle events; nil members are skipped.
type hooks struct {
	agentStart func(a *agent)
	agentEnd   func(a *agent, output any)
	handoff    func(from, to *agent)
	// turnEnd sees every successful model turn of the active agent.
	turnEnd func(a *agent, resp *responses.Response)
}

// runner drives one run.
type runner struct {
	client   *openai.Client
	maxTurns int
	hooks    hooks
	// span records chat / execute_tool child spans; nil records none.
	span *aiserver.HandlerRun
	// onText streams text deltas; nil makes blocking calls.
	onText func(string) bool
}

// runResult is a finished run.
type runResult struct {
	output any    // final text, or the decoded JSON for an output type
	text   string // final text; when streaming, every streamed delta
	usage  openaiwire.RunUsage
	model  string // model the API reported on the last turn
}

// run executes start until a turn produces no function calls. stopped
// reports that the stream consumer stopped iterating.
func (r *runner) run(ctx context.Context, start *agent, input responses.ResponseInputParam) (res runResult, stopped bool, err error) {
	current := start
	r.emitStart(current)
	params := current.request(input, "")
	var streamed strings.Builder
	for turn := 1; ; turn++ {
		if turn > r.maxTurns {
			return res, false, fmt.Errorf("%w (%d)", ErrMaxTurns, r.maxTurns)
		}
		var chat trace.Span
		if r.span != nil {
			chat = r.span.StartChat(current.model)
		}
		resp, stop, terr := r.turn(ctx, params, &streamed)
		if stop {
			return res, true, nil
		}
		if terr != nil {
			return res, false, terr
		}
		res.usage.Add(resp.Usage)
		res.model = openaiwire.ResponseModel(resp, current.model)
		if r.span != nil {
			r.span.EndChat(chat, res.model, openaiwire.Usage(resp.Usage), openaiwire.FinishReason(resp))
		}
		if r.hooks.turnEnd != nil {
			r.hooks.turnEnd(current, resp)
		}
		trackNative(current, resp)

		var calls []responses.ResponseFunctionToolCall
		for _, item := range resp.Output {
			if item.Type == "function_call" {
				calls = append(calls, item.AsFunctionCall())
			}
		}
		if len(calls) == 0 {
			res.text = resp.OutputText()
			res.output = res.text
			if r.onText != nil {
				res.text = streamed.String() // done.output spans every streamed turn
				res.output = res.text
			} else if current.outputType != nil {
				var v any
				if json.Unmarshal([]byte(res.text), &v) == nil {
					res.output = v // the parsed object, not a re-serialized string
				} // else best effort: the raw text
			}
			if r.hooks.agentEnd != nil {
				r.hooks.agentEnd(current, res.output)
			}
			return res, false, nil
		}
		outputs, next, terr := r.runCalls(ctx, current, calls)
		if terr != nil {
			return res, false, terr
		}
		if next != nil {
			if r.hooks.handoff != nil {
				r.hooks.handoff(current, next)
			}
			current = next
			r.emitStart(current)
		}
		// previous_response_id carries the conversation; the new turn adds
		// only the call outputs.
		params = current.request(outputs, resp.ID)
	}
}

func (r *runner) emitStart(a *agent) {
	if r.hooks.agentStart != nil {
		r.hooks.agentStart(a)
	}
}

// turn performs one provider call, streaming when onText is set.
func (r *runner) turn(ctx context.Context, params responses.ResponseNewParams, streamed *strings.Builder) (*responses.Response, bool, error) {
	if r.onText == nil {
		resp, err := r.client.Responses.New(ctx, params)
		if err != nil {
			return nil, false, fmt.Errorf("openaiagents: responses.create: %w", err)
		}
		return resp, false, nil
	}
	resp, stopped, err := openaiwire.StreamTurn(ctx, r.client, params, func(t string) bool {
		streamed.WriteString(t)
		return r.onText(t)
	})
	if err != nil {
		return nil, false, fmt.Errorf("openaiagents: responses.stream: %w", err)
	}
	return resp, stopped, nil
}

// runCalls answers every function call of one turn. The first handoff call
// selects the next agent; later handoff calls in the same turn are ignored,
// as in the Agents SDK. A call for a tool that was not offered is an error.
func (r *runner) runCalls(ctx context.Context, a *agent, calls []responses.ResponseFunctionToolCall) (responses.ResponseInputParam, *agent, error) {
	var outputs responses.ResponseInputParam
	var next *agent
	for _, call := range calls {
		if h := a.findHandoff(call.Name); h != nil {
			out := "Multiple handoffs detected, ignoring this one."
			if next == nil {
				next = h.target
				b, _ := json.Marshal(map[string]string{"assistant": h.target.name})
				out = string(b)
			}
			outputs = append(outputs, openaiwire.FunctionCallOutput(call.CallID, out))
			continue
		}
		var span trace.Span
		if r.span != nil {
			span = r.span.StartTool(call.Name, call.CallID)
		}
		res, err := callTool(ctx, a, call)
		if err != nil {
			return nil, nil, err // HandlerRun.Fail ends the open tool span
		}
		if r.span != nil {
			r.span.EndTool(span, nil)
		}
		outputs = append(outputs, openaiwire.FunctionCallOutput(call.CallID, aiserver.ToolResultString(res)))
	}
	return outputs, next, nil
}

func (a *agent) findHandoff(tool string) *handoff {
	for i := range a.handoffs {
		if a.handoffs[i].tool == tool {
			return &a.handoffs[i]
		}
	}
	return nil
}

// callTool runs one function tool. Only tools offered to this agent run;
// the raw handler map is never consulted for anything else.
func callTool(ctx context.Context, a *agent, call responses.ResponseFunctionToolCall) (any, error) {
	fn, _ := a.handlers[call.Name].(aiserver.ToolFunc)
	if _, offered := a.tools[call.Name]; !offered || fn == nil {
		return nil, fmt.Errorf("%w: %q", ErrToolNotOffered, call.Name)
	}
	args, err := openaiwire.ParseArguments(call.Arguments)
	if err != nil {
		return nil, fmt.Errorf("openaiagents: tool %q arguments: %w", call.Name, err)
	}
	res, err := fn(ctx, args)
	if err != nil {
		return nil, fmt.Errorf("openaiagents: tool %q: %w", call.Name, err) // propagate
	}
	return res, nil
}

// trackNative calls Track on the NativeTool behind each hosted-tool call in
// resp (for example a web_search_call for a web_search tool).
func trackNative(a *agent, resp *responses.Response) {
	for _, item := range resp.Output {
		kind, ok := strings.CutSuffix(item.Type, "_call")
		if !ok || item.Type == "function_call" {
			continue
		}
		for _, key := range slices.Sorted(maps.Keys(a.native)) {
			if nt := a.handlers[key].(*aiserver.NativeTool); strings.HasPrefix(nt.ToolName, kind) {
				nt.Track()
				break
			}
		}
	}
}
