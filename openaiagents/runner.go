package openaiagents

import (
	"context"
	"fmt"
	"hash/fnv"
	"maps"
	"regexp"
	"slices"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/agentloop"
	"github.com/launchdarkly/go-ai-sdk/internal/handlerkit"
	"github.com/launchdarkly/go-ai-sdk/internal/openaiwire"
)

// This file is the Go stand-in for the OpenAI Agents SDK's Agent: an agent
// is instructions + model + tools + handoffs + output type. The run loop is
// internal/agentloop; responsesModel is its Responses API wire.

// handoffPrefix is the Agents SDK's default handoff tool-name prefix.
const handoffPrefix = "transfer_to_"

// forwardedParameters are the config.model.parameters keys forwarded: the
// Responses API fields behind the JavaScript package's ModelSettings
// allowlist. Handler-owned fields (model, input, instructions, tools, text,
// previous_response_id) are deliberately absent; `verbosity` is mapped to
// text.verbosity and `max_turns` caps the run instead. store,
// prompt_cache_retention, truncation and context_management (data retention
// and server-side state) are never forwarded. tool_choice is forwarded only
// as a string.
var forwardedParameters = []string{
	"max_output_tokens", "parallel_tool_calls", "reasoning", "temperature", "tool_choice", "top_p",
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
		parameters: agentloop.SnakeParameters(cfg.Model.Parameters),
		tools:      handlerkit.CallableTools(cfg, handlers),
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
	extra := openaiwire.ModelParameters(a.parameters, forwardedParameters)
	if _, ok := extra["tool_choice"].(string); !ok {
		delete(extra, "tool_choice")
	}
	if len(extra) > 0 {
		p.SetExtraFields(extra)
	}
	return p
}

// ModelName implements agentloop.Agent.
func (a *agent) ModelName() string { return a.model }

// StructuredOutput implements agentloop.Agent.
func (a *agent) StructuredOutput() bool { return a.outputType != nil }

// Handoff implements agentloop.Agent: the target of the transfer tool, named
// as the handoff output reports it.
func (a *agent) Handoff(tool string) (agentloop.Agent, string, bool) {
	if h := a.findHandoff(tool); h != nil {
		return h.target, h.target.name, true
	}
	return nil, "", false
}

// Tool implements agentloop.Agent: only tools offered to this agent run;
// the raw handler map is never consulted for anything else.
func (a *agent) Tool(name string) (aiserver.ToolFunc, bool) {
	fn, _ := a.handlers[name].(aiserver.ToolFunc)
	_, offered := a.tools[name]
	return fn, offered
}

func (a *agent) findHandoff(tool string) *handoff {
	for i := range a.handoffs {
		if a.handoffs[i].tool == tool {
			return &a.handoffs[i]
		}
	}
	return nil
}

// newRunner is the agent loop over the Responses API for one run.
func newRunner(client *openai.Client, maxTurns int) *agentloop.Runner {
	return &agentloop.Runner{
		Model: &responsesModel{client: client}, MaxTurns: maxTurns,
		ErrMaxTurns: ErrMaxTurns, ErrToolNotOffered: ErrToolNotOffered, Prefix: "openaiagents",
	}
}

// responsesModel is the Responses API wire for one run. Follow-up turns
// chain with previous_response_id and send only the call outputs; one id is
// threaded through the whole run, across handoffs.
type responsesModel struct {
	client         *openai.Client
	lastResponseID string
}

// Turn implements agentloop.Model: one provider call, streaming when onText
// is set.
func (m *responsesModel) Turn(ctx context.Context, la agentloop.Agent, in agentloop.Input, onText func(string) bool) (agentloop.Turn, bool, error) {
	a := la.(*agent)
	input := openaiwire.Turns(in.Turns)
	for _, o := range in.Outputs {
		input = append(input, openaiwire.FunctionCallOutput(o.CallID, o.Output))
	}
	params := a.request(input, m.lastResponseID)
	var resp *responses.Response
	if onText == nil {
		r, err := m.client.Responses.New(ctx, params)
		if err != nil {
			return agentloop.Turn{}, false, fmt.Errorf("openaiagents: responses.create: %w", err)
		}
		resp = r
	} else {
		r, stopped, err := openaiwire.StreamTurn(ctx, m.client, params, onText)
		if err != nil {
			return agentloop.Turn{}, false, fmt.Errorf("openaiagents: responses.stream: %w", err)
		}
		if stopped {
			return agentloop.Turn{}, true, nil
		}
		resp = r
	}
	m.lastResponseID = resp.ID
	t := agentloop.Turn{
		Text:   resp.OutputText(),
		Tokens: agentloop.Tokens{Input: resp.Usage.InputTokens, Output: resp.Usage.OutputTokens},
		Usage:  openaiwire.Usage(resp.Usage),
		Model:  openaiwire.ResponseModel(resp, a.model),
		Finish: openaiwire.FinishReason(resp),
		Native: nativeCalls(a, resp),
	}
	for _, item := range resp.Output {
		if item.Type == "function_call" {
			c := item.AsFunctionCall()
			t.Calls = append(t.Calls, agentloop.ToolCall{ID: c.CallID, Name: c.Name, Arguments: c.Arguments})
		}
	}
	return t, false, nil
}

// nativeCalls is the NativeTool behind each hosted-tool call in resp (for
// example a web_search_call for a web_search tool).
func nativeCalls(a *agent, resp *responses.Response) []*aiserver.NativeTool {
	var out []*aiserver.NativeTool
	for _, item := range resp.Output {
		kind, ok := strings.CutSuffix(item.Type, "_call")
		if !ok || item.Type == "function_call" {
			continue
		}
		for _, key := range slices.Sorted(maps.Keys(a.native)) {
			if nt := a.handlers[key].(*aiserver.NativeTool); strings.HasPrefix(nt.ToolName, kind) {
				out = append(out, nt)
				break
			}
		}
	}
	return out
}
