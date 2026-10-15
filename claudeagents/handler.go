// Package claudeagents is the Anthropic agent-mode handler for aiserver
// (providesFor Anthropic/agent), plus the claudeGraph convenience wrapper and
// the toClaudeAgents native graph adapter.
//
// Go has no Claude Agent SDK, so the agent loop runs on the Anthropic
// Messages API: function tools execute locally, the WebSearch / WebFetch
// NativeTool sentinels map to Anthropic server tools, the other Claude Code
// built-ins (Bash, Read, Edit, ...) execute on the host confined to
// Options.WorkDir, and graph children become sub-agent tools of their parent.
package claudeagents

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"go.opentelemetry.io/otel/trace"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/anthropicwire"
)

const (
	// PackageName is the identity this package registers for $ld:ai:sdk:info
	// and $ld:ai:sdk:usage.
	PackageName = "github.com/launchdarkly/go-ai-sdk/claudeagents"
	// PackageVersion is this package's release version.
	PackageVersion = "0.1.0"
	// SpanName is the documented root span name.
	SpanName = "claude.query"
	// DefaultMaxSteps bounds the agent loop when neither Options.MaxSteps nor
	// the config's max_turns parameter sets a bound.
	DefaultMaxSteps = 10
)

// ProvidesFor is the variation provider/mode this handler serves.
var ProvidesFor = aiserver.ProvidesFor{Provider: "Anthropic", Mode: aiserver.ModeAgent}

var (
	// ErrToolNotFound is returned when the model calls a tool that has no
	// local executor. The error message names the tool.
	ErrToolNotFound = errors.New("claudeagents: no handler registered for tool")
	// ErrUnsupportedNativeTool is returned when a NativeTool in the config
	// names a tool this package neither maps to an Anthropic server tool nor
	// executes on the host.
	ErrUnsupportedNativeTool = errors.New("claudeagents: unsupported native tool")
	// ErrMaxSteps is returned when the model still requests tools after the
	// step bound is reached (the Agent SDK's error_max_turns).
	ErrMaxSteps = errors.New("claudeagents: agent loop reached its maximum number of turns")
)

func init() { aiserver.RegisterAiSdkPackage(PackageName, PackageVersion) }

// Options configure NewHandler.
type Options struct {
	// Client is used for every call when set. Otherwise a client is built from
	// APIKey, falling back to the ANTHROPIC_API_KEY environment variable.
	Client *anthropic.Client
	// APIKey is the Anthropic API key used when Client is nil.
	APIKey string
	// CaptureContent is attached to the handler so the client core can gate
	// judge explanations on it.
	CaptureContent bool
	// MaxSteps bounds model turns per run. Zero means DefaultMaxSteps. A
	// config's model.parameters max_turns / maxTurns takes precedence.
	MaxSteps int
	// WorkDir is the directory the host-executed built-ins (ClaudeBash,
	// ClaudeRead, ...) run in and are confined to. Empty means the process
	// working directory when the handler is constructed. A relative path is
	// made absolute at construction.
	WorkDir string
}

// NewHandler returns the Anthropic agent handler (createClaudeAgentsHandler).
func NewHandler(opts Options) *aiserver.ProviderHandler {
	aiserver.ReportUsage("claude-agents.createClaudeAgentsHandler", PackageName, PackageVersion)
	return newHandler(opts)
}

func newHandler(opts Options) *aiserver.ProviderHandler {
	h := newAgent(opts)
	return aiserver.CreateHandlerInternal(ProvidesFor, h.invoke, h.stream, opts.CaptureContent)
}

// InvokeOptions are the optional arguments to ClaudeAgents.
type InvokeOptions struct {
	// ToolHandlers are the config's tool executors and NativeTool sentinels.
	ToolHandlers map[string]aiserver.ToolHandler
	// Registry supplies further tools and handlers.
	Registry *aiserver.Registry
	// SkipJudges returns judge tasks instead of running judges.
	SkipJudges bool
	// Variables are forwarded to Invoke, not to the config.
	Variables map[string]any
	// Handler configures the handler built for this call.
	Handler Options
}

// ClaudeAgents invokes configKey through a fresh Anthropic agent handler.
// opts may be nil.
func ClaudeAgents(ctx context.Context, configKey, userInput string, userCtx ldcontext.Context, opts *InvokeOptions) (aiserver.ProviderResponse, error) {
	aiserver.ReportUsage("claude-agents.claudeAgents", PackageName, PackageVersion)
	if opts == nil {
		opts = &InvokeOptions{}
	}
	return aiserver.ConfigInternal(aiserver.ConfigArgs{
		Key: configKey, Handlers: []*aiserver.ProviderHandler{newHandler(opts.Handler)},
		ToolHandlers: opts.ToolHandlers, Registry: opts.Registry, SkipJudges: opts.SkipJudges,
	}).Invoke(ctx, userInput, userCtx, aiserver.WithVariables(opts.Variables))
}

type agent struct {
	client   *anthropic.Client
	maxSteps int
	workDir  string // absolute; "" if it could not be determined
}

func newAgent(opts Options) *agent {
	c := opts.Client
	if c == nil {
		var reqOpts []option.RequestOption
		if opts.APIKey != "" {
			reqOpts = append(reqOpts, option.WithAPIKey(opts.APIKey))
		}
		nc := anthropic.NewClient(reqOpts...)
		c = &nc
	}
	steps := opts.MaxSteps
	if steps <= 0 {
		steps = DefaultMaxSteps
	}
	dir := opts.WorkDir
	if dir == "" {
		dir, _ = os.Getwd()
	}
	if dir != "" {
		if abs, err := filepath.Abs(dir); err == nil {
			dir = abs
		}
	}
	return &agent{client: c, maxSteps: steps, workDir: dir}
}

// localTool is a tool the loop executes in-process: a config function tool
// or a graph sub-agent.
type localTool struct {
	name  string
	param anthropic.ToolUnionParam
	fn    aiserver.ToolFunc
}

// toolset is everything offered to the model for one run.
type toolset struct {
	params  []anthropic.ToolUnionParam
	local   map[string]aiserver.ToolFunc
	natives map[string]*aiserver.NativeTool // by server-tool name
	host    map[string]*aiserver.NativeTool // by Claude Code tool name
	ws      *workspace                      // set when host is non-empty
}

// buildTools offers config tools that have a handler: a ToolFunc as a
// custom tool, a NativeTool as its Anthropic server tool or as a
// host-executed function tool under its Claude Code name. Extra tools (graph
// sub-agents) are appended. The caller closes ts.ws.
func buildTools(cfg aiserver.AiConfig, handlers map[string]aiserver.ToolHandler, extra []localTool, workDir string) (toolset, error) {
	ts := toolset{local: map[string]aiserver.ToolFunc{}, natives: map[string]*aiserver.NativeTool{}, host: map[string]*aiserver.NativeTool{}}
	for _, key := range slices.Sorted(maps.Keys(cfg.Tools)) {
		t := cfg.Tools[key]
		switch h := handlers[key].(type) {
		case aiserver.ToolFunc:
			ts.params = append(ts.params, anthropicwire.FunctionTool(key, t.Description, t.Parameters))
			ts.local[key] = h
		case *aiserver.NativeTool:
			if ht, ok := hostTools[h.ToolName]; ok {
				if ts.host[h.ToolName] == nil { // first LD key wins, like server tools
					ts.host[h.ToolName] = h
					ts.params = append(ts.params, anthropicwire.FunctionTool(h.ToolName, ht.description, ht.schema))
				}
				continue
			}
			name, param, ok := serverTool(h.ToolName)
			if !ok {
				return toolset{}, fmt.Errorf("%w: %s (tool %q)", ErrUnsupportedNativeTool, h.ToolName, key)
			}
			if ts.natives[name] == nil {
				ts.natives[name] = h
				ts.params = append(ts.params, param)
			}
		}
	}
	for _, x := range extra {
		ts.params = append(ts.params, x.param)
		ts.local[x.name] = x.fn
	}
	for name := range ts.host {
		if ts.local[name] != nil {
			return toolset{}, fmt.Errorf("claudeagents: tool name %q is used by both a function tool and a built-in", name)
		}
	}
	if len(ts.host) > 0 {
		ws, err := openWorkspace(workDir)
		if err != nil {
			return toolset{}, err
		}
		ts.ws = ws
	}
	return ts, nil
}

// runHost executes a host built-in. Its failures become is_error results the
// model can react to; only the span records them.
func (r *toolset) runHost(ctx context.Context, run *aiserver.HandlerRun, b anthropic.ContentBlockUnion) anthropic.ContentBlockParamUnion {
	span := run.StartTool(b.Name, b.ID)
	r.host[b.Name].Track()
	out, err := hostTools[b.Name].run(trace.ContextWithSpan(ctx, span), r.ws, b.Input)
	if err != nil && out == "" {
		out = err.Error()
	}
	run.EndTool(span, err)
	return anthropic.NewToolResultBlock(b.ID, out, err != nil)
}

// serverTool maps a Claude Code built-in name to the Anthropic server tool
// that provides the same capability on the Messages API.
func serverTool(toolName string) (string, anthropic.ToolUnionParam, bool) {
	switch toolName {
	case "WebSearch":
		return "web_search", anthropic.ToolUnionParam{OfWebSearchTool20250305: &anthropic.WebSearchTool20250305Param{}}, true
	case "WebFetch":
		return "web_fetch", anthropic.ToolUnionParam{OfWebFetchTool20250910: &anthropic.WebFetchTool20250910Param{}}, true
	}
	return "", anthropic.ToolUnionParam{}, false
}

// maxStepsFor lets the config's max_turns (forwarded by the JS handler as the
// Agent SDK's maxTurns) override the handler bound.
func (a *agent) maxStepsFor(cfg aiserver.AiConfig) int {
	for _, k := range []string{"max_turns", "maxTurns"} {
		if v, ok := cfg.Model.Parameters[k].(float64); ok && v >= 1 {
			return int(v)
		}
	}
	return a.maxSteps
}

// runResult is one agent run's outcome. final is the last model response's
// text (the Agent SDK `result`); all is every turn's text, which the stream
// done event reports.
type runResult struct {
	final, all string
	usage      anthropicwire.Usage
	abandoned  bool
}

// spanIdentity is the LaunchDarkly association a ToClaudeAgents node writes
// on its root span. The core only reads the typed context from variables it
// builds itself, so the adapter passes it here instead of through __ld.
type spanIdentity struct {
	td      aiserver.TrackData
	userCtx ldcontext.Context
}

// run is the agent loop shared by both paths; onText is nil when blocking.
// ident is nil for handler calls, whose __ld variable carries the association.
func (a *agent) run(ctx context.Context, req aiserver.HandlerRequest, extra []localTool, ident *spanIdentity, onText func(string) bool) (runResult, error) {
	var res runResult
	model := req.Config.Model.Name
	p := aiserver.ComposePrompt(req, aiserver.ModeAgent)
	if onText == nil { // streaming ignores outputFormat
		p.System = aiserver.WithOutputFormat(p.System, req.Config.OutputFormat)
	}
	r := aiserver.StartHandlerRun(ctx, aiserver.HandlerSpanOptions{
		Tracer: PackageName, Name: SpanName, System: "anthropic", Operation: "invoke_agent", RequestModel: model,
	}, req, p)
	defer r.EndOnPanic()
	if ident != nil {
		aiserver.SetLDSpanAttributesFor(r.Root, ident.td, ident.userCtx)
	}

	ts, err := buildTools(req.Config, req.ToolHandlers, extra, a.workDir)
	if err != nil {
		r.Fail(err)
		return res, err
	}
	defer ts.ws.close()
	params := anthropic.MessageNewParams{
		Model: anthropic.Model(model), MaxTokens: anthropicwire.MaxTokens(req.Config),
		Messages: anthropicwire.Messages(p.Turns), Tools: ts.params,
	}
	q := buildQueryOptions(req.Config.Model.Parameters)
	if len(q.body) > 0 {
		params.SetExtraFields(q.body)
	}
	if p.System != "" {
		params.System = []anthropic.TextBlockParam{{Text: p.System}}
	}

	var final, all strings.Builder
	var responseModel string
	steps := a.maxStepsFor(req.Config)
	for step := 0; ; step++ {
		if step == steps {
			err := fmt.Errorf("%w (%d)", ErrMaxSteps, steps)
			r.Fail(err)
			return res, err
		}
		chat := r.StartChat(model)
		msg, stopped, terr := a.turn(r.Context(), params, q.reqOpts, onText)
		if terr != nil && q.fallbackModel != "" && model != q.fallbackModel && len(msg.Content) == 0 && shouldFallBack(terr) {
			// fallbackModel: the rest of the run uses it, as in the Agent SDK.
			r.EndTool(chat, terr) // ends the failed chat span as ERROR
			model = q.fallbackModel
			params.Model = anthropic.Model(model)
			chat = r.StartChat(model)
			msg, stopped, terr = a.turn(r.Context(), params, q.reqOpts, onText)
		}
		res.usage.Add(msg.Usage)
		if stopped {
			r.Abandon()
			res.abandoned = true
			return res, nil
		}
		if terr != nil {
			r.Fail(terr)
			return res, terr
		}
		responseModel = string(msg.Model)
		r.EndChat(chat, responseModel, anthropicwire.SpanUsage(msg.Usage), anthropicwire.FinishReason(string(msg.StopReason)))

		var results []anthropic.ContentBlockParamUnion
		for _, b := range msg.Content {
			switch b.Type {
			case "text":
				final.WriteString(b.Text)
				all.WriteString(b.Text)
			case "server_tool_use":
				// Anthropic ran the tool; record it like the Agent SDK's PreToolUse hook.
				if n := ts.natives[b.Name]; n != nil {
					n.Track()
				}
				r.EndTool(r.StartTool(b.Name, b.ID), nil)
			case "tool_use":
				if ts.host[b.Name] != nil {
					results = append(results, ts.runHost(r.Context(), r, b))
					continue
				}
				fn := ts.local[b.Name]
				if fn == nil {
					err := fmt.Errorf("%w: %q", ErrToolNotFound, b.Name)
					r.Fail(err)
					return res, err
				}
				span := r.StartTool(b.Name, b.ID)
				var args map[string]any
				_ = json.Unmarshal(b.Input, &args)
				out, ferr := fn(trace.ContextWithSpan(r.Context(), span), args)
				if ferr != nil {
					r.Fail(ferr)
					return res, ferr // tool errors propagate
				}
				r.EndTool(span, nil)
				results = append(results, anthropic.NewToolResultBlock(b.ID, aiserver.ToolResultString(out), false))
			}
		}
		switch {
		case msg.StopReason == anthropic.StopReasonToolUse && len(results) > 0:
			params.Messages = append(params.Messages, msg.ToParam(), anthropic.NewUserMessage(results...))
			final.Reset() // the result is the last response's text
			continue
		case msg.StopReason == anthropic.StopReasonPauseTurn:
			// A long server-tool turn paused; resend so the model continues it.
			params.Messages = append(params.Messages, msg.ToParam())
			continue
		}
		break
	}
	res.final, res.all = final.String(), all.String()
	if onText == nil {
		r.Finish(responseModel, res.final)
	} else {
		r.Finish(responseModel, res.all) // what the stream's done event reports
	}
	return res, nil
}

// turn performs one provider call. When onText is set it streams, calling
// onText per text delta; stopped reports that onText returned false.
func (a *agent) turn(ctx context.Context, params anthropic.MessageNewParams, reqOpts []option.RequestOption, onText func(string) bool) (msg anthropic.Message, stopped bool, err error) {
	if onText == nil {
		m, err := a.client.Messages.New(ctx, params, reqOpts...)
		if err != nil {
			return anthropic.Message{}, false, err
		}
		return *m, false, nil
	}
	s := a.client.Messages.NewStreaming(ctx, params, reqOpts...)
	defer s.Close() // releases the HTTP body on early exit
	for s.Next() {
		ev := s.Current()
		if err := msg.Accumulate(ev); err != nil {
			return msg, false, err
		}
		if ev.Type == "content_block_delta" && ev.Delta.Type == "text_delta" && !onText(ev.Delta.Text) {
			return msg, true, nil
		}
	}
	return msg, false, s.Err()
}

func (a *agent) invoke(ctx context.Context, req aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
	res, err := a.run(ctx, req, nil, nil, nil)
	if err != nil {
		return aiserver.HandlerResult{}, err
	}
	return aiserver.HandlerResult{Output: res.final, Usage: res.usage.Raw()}, nil
}

func (a *agent) stream(ctx context.Context, req aiserver.HandlerRequest) iter.Seq2[aiserver.HandlerStreamEvent, error] {
	return func(yield func(aiserver.HandlerStreamEvent, error) bool) {
		res, err := a.run(ctx, req, nil, nil, func(t string) bool {
			return yield(aiserver.HandlerStreamEvent{Type: aiserver.EventChunk, Text: t}, nil)
		})
		if res.abandoned {
			return
		}
		if err != nil {
			yield(aiserver.HandlerStreamEvent{}, err)
			return
		}
		yield(aiserver.HandlerStreamEvent{Type: aiserver.EventDone, Output: &res.all, Usage: res.usage.Raw()}, nil)
	}
}
