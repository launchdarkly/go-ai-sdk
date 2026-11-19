package litellmagents

import (
	"cmp"
	"context"
	"fmt"
	"hash/fnv"
	"maps"
	"regexp"
	"slices"
	"strings"
	"sync"

	"github.com/openai/openai-go/v3"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/agentloop"
	"github.com/launchdarkly/go-ai-sdk/internal/bridge"
	"github.com/launchdarkly/go-ai-sdk/internal/chatwire"
	"github.com/launchdarkly/go-ai-sdk/internal/handlerkit"
)

// This file is the Go stand-in for the Agents SDK's Agent on its Chat
// Completions model: an agent is instructions + model + tools + handoffs +
// output schema. The run loop is internal/agentloop; chatModel is its wire.

// handoffPrefix is the Agents SDK's default handoff tool-name prefix.
const handoffPrefix = "transfer_to_"

// forwardedParameters are the config.model.parameters keys sent as written:
// the Chat Completions fields behind the Agents SDK ModelSettings allowlist.
// `reasoning` (effort only, as reasoning_effort) and `verbosity` are mapped
// separately, `max_turns` caps the run, and tool_choice is sent only as a
// string. Everything else is dropped.
var forwardedParameters = []string{
	"frequency_penalty", "max_tokens", "parallel_tool_calls", "presence_penalty", "temperature", "tool_choice", "top_p",
}

// agent is one agent in a run.
type agent struct {
	name         string // sanitized; what the model sees in handoff tool names
	key          string // graph node key; "" for the single-config handler
	instructions string
	model        string // the id sent to the proxy
	parameters   map[string]any
	tools        map[string]aiserver.Tool // function tools with a local ToolFunc
	handlers     map[string]aiserver.ToolHandler
	handoffs     []handoff
	schema       map[string]any // structured output; nil for text
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

// newAgent builds an agent from a config. Only config tools with a local
// ToolFunc are offered; Chat Completions has no hosted tools, so a
// NativeTool is skipped with a warning.
func newAgent(name, instructions string, cfg aiserver.AiConfig, handlers map[string]aiserver.ToolHandler) *agent {
	for key := range cfg.Tools {
		if _, ok := handlers[key].(*aiserver.NativeTool); ok {
			warnNativeSkipped(key)
		}
	}
	return &agent{
		name: name, instructions: instructions, model: requestModel(cfg),
		parameters: agentloop.SnakeParameters(cfg.Model.Parameters),
		tools:      handlerkit.CallableTools(cfg, handlers),
		handlers:   handlers,
	}
}

// nativeWarned holds the tool keys already warned about, once per process.
var nativeWarned sync.Map

func warnNativeSkipped(key string) {
	if _, seen := nativeWarned.LoadOrStore(key, struct{}{}); !seen {
		bridge.Logger().Warn("litellmagents: skipping NativeTool: Chat Completions has no provider built-ins", "tool", key)
	}
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

// request builds one turn: this agent's instructions as the system message,
// then the run's conversation so far.
func (a *agent) request(conversation []openai.ChatCompletionMessageParamUnion) openai.ChatCompletionNewParams {
	p := openai.ChatCompletionNewParams{
		Model:    a.model,
		Messages: append(chatwire.Messages(a.instructions, nil), conversation...),
		Tools:    chatwire.FunctionTools(a.tools),
	}
	for _, h := range a.handoffs {
		p.Tools = append(p.Tools, chatwire.FunctionTool(h.tool, h.description,
			map[string]any{"type": "object", "properties": map[string]any{}, "additionalProperties": false}))
	}
	if a.schema != nil {
		p.ResponseFormat = chatwire.ResponseFormat(a.schema)
	}
	if extra := modelParameters(a.parameters); len(extra) > 0 {
		p.SetExtraFields(extra)
	}
	return p
}

// modelParameters maps snake_case parameters to the request fields they
// forward.
func modelParameters(in map[string]any) map[string]any {
	out := chatwire.ModelParameters(in, forwardedParameters)
	if _, ok := out["tool_choice"].(string); !ok {
		delete(out, "tool_choice")
	}
	if r, ok := in["reasoning"].(map[string]any); ok {
		if effort, ok := r["effort"].(string); ok {
			out["reasoning_effort"] = effort
		}
	}
	verbosity, _ := in["verbosity"].(string)
	if text, ok := in["text"].(map[string]any); ok {
		if v, ok := text["verbosity"].(string); ok {
			verbosity = v // an explicit text.verbosity wins, as in the JavaScript package
		}
	}
	if verbosity != "" {
		out["verbosity"] = verbosity
	}
	return out
}

// outputSchema is the JSON Schema sent for a variation's outputFormat, with
// a root type of "object", as the JavaScript package sends it; nil without
// one. The input map is not modified.
func outputSchema(outputFormat map[string]any) map[string]any {
	if len(outputFormat) == 0 {
		return nil
	}
	schema := maps.Clone(outputFormat)
	if t, _ := schema["type"].(string); t == "" || t == "json" || t == "json_schema" {
		schema["type"] = "object"
	}
	return schema
}

// crossRegionPrefixes are the Bedrock cross-region inference-profile
// prefixes an id may already carry.
var crossRegionPrefixes = []string{"global.", "us.", "us-gov.", "eu.", "apac.", "au.", "jp.", "ca.", "in."}

// requestModel is the model sent to the proxy. A `bedrock/` name gets the
// model.region inference-profile prefix on its last segment, whatever the
// provider, so every `bedrock/<route>/` form works; a bare name gets it only
// when the provider is Bedrock. Any other name, and any ARN, is sent as
// configured.
func requestModel(cfg aiserver.AiConfig) string {
	name := cfg.Model.Name
	if strings.HasPrefix(name, "bedrock/") {
		if strings.Contains(name, "arn:") {
			return name
		}
		cut := strings.LastIndex(name, "/")
		return name[:cut+1] + bedrockModelID(name[cut+1:], cfg.Model.Region)
	}
	if strings.EqualFold(cfg.Provider.Name, "bedrock") && !strings.Contains(name, "/") {
		return bedrockModelID(name, cfg.Model.Region)
	}
	return name
}

// bedrockModelID prepends region once to a Bedrock model id. An id that
// already carries region or a known prefix, and an ARN, is unchanged.
func bedrockModelID(id, region string) string {
	prefix := strings.TrimRight(strings.TrimSpace(region), ".")
	if prefix == "" || id == "" || strings.HasPrefix(id, prefix+".") || strings.HasPrefix(id, "arn:") {
		return id
	}
	if slices.ContainsFunc(crossRegionPrefixes, func(p string) bool { return strings.HasPrefix(id, p) }) {
		return id
	}
	return prefix + "." + id
}

// ModelName implements agentloop.Agent.
func (a *agent) ModelName() string { return a.model }

// StructuredOutput implements agentloop.Agent.
func (a *agent) StructuredOutput() bool { return a.schema != nil }

// Handoff implements agentloop.Agent: the target of the transfer tool, named
// as the handoff output reports it.
func (a *agent) Handoff(tool string) (agentloop.Agent, string, bool) {
	for _, h := range a.handoffs {
		if h.tool == tool {
			return h.target, h.target.name, true
		}
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

// newRunner is the agent loop over Chat Completions for one run.
func newRunner(client *openai.Client, maxTurns int) *agentloop.Runner {
	return &agentloop.Runner{
		Model: &chatModel{client: client}, MaxTurns: maxTurns,
		ErrMaxTurns: ErrMaxTurns, ErrToolNotOffered: ErrToolNotOffered, Prefix: "litellmagents",
	}
}

// chatModel is the Chat Completions wire for one run. Chat Completions keeps
// no server-side state, so every turn resends the whole conversation, across
// handoffs too; only the system message changes with the active agent.
type chatModel struct {
	client       *openai.Client
	conversation []openai.ChatCompletionMessageParamUnion
}

// Turn implements agentloop.Model: one proxy call, streaming when onText is
// set.
func (m *chatModel) Turn(ctx context.Context, la agentloop.Agent, in agentloop.Input, onText func(string) bool) (agentloop.Turn, bool, error) {
	a := la.(*agent)
	m.conversation = append(m.conversation, chatwire.Messages("", in.Turns)...)
	for _, o := range in.Outputs {
		m.conversation = append(m.conversation, chatwire.ToolMessage(o.CallID, o.Output))
	}
	params := a.request(m.conversation)
	var t chatwire.Turn
	if onText == nil {
		resp, err := m.client.Chat.Completions.New(ctx, params)
		if err != nil {
			return agentloop.Turn{}, false, fmt.Errorf("litellmagents: chat.completions.create: %w", err)
		}
		if t, err = chatwire.TurnOf(resp); err != nil {
			return agentloop.Turn{}, false, fmt.Errorf("litellmagents: %w", err)
		}
	} else {
		st, stopped, err := chatwire.StreamTurn(ctx, m.client, params, onText)
		if err != nil {
			return agentloop.Turn{}, false, fmt.Errorf("litellmagents: chat.completions.stream: %w", err)
		}
		if stopped {
			return agentloop.Turn{}, true, nil
		}
		t = st
	}
	m.conversation = append(m.conversation, chatwire.AssistantMessage(t))
	out := agentloop.Turn{
		Text:   t.Text,
		Tokens: agentloop.Tokens{Input: int64(t.Usage.Input), Output: int64(t.Usage.Output)},
		Usage:  t.Usage,
		Model:  cmp.Or(t.Model, a.model),
		Finish: t.Finish,
	}
	for _, c := range t.ToolCalls {
		out.Calls = append(out.Calls, agentloop.ToolCall{ID: c.ID, Name: c.Name, Arguments: c.Arguments})
	}
	return out, false, nil
}
