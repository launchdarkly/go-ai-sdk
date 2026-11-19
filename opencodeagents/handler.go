// Package opencodeagents is the OpenCode agent handler. Each call starts its
// own `opencode serve` child on loopback, runs one session over its HTTP API
// and stops the child when the call ends.
package opencodeagents

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"iter"
	"maps"
	"slices"
	"strings"
	"time"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/handlerkit"
	"github.com/launchdarkly/go-ai-sdk/internal/kitbase"
	"github.com/launchdarkly/go-ai-sdk/internal/sdkapi"
	"github.com/launchdarkly/go-ai-sdk/internal/telemetry"
)

const (
	// PackageName is the identity this package registers for $ld:ai:sdk:info
	// and $ld:ai:sdk:usage.
	PackageName = "github.com/launchdarkly/go-ai-sdk/opencodeagents"
	// PackageVersion is this package's release version.
	PackageVersion = "1.0.0" // x-release-please-version
	// SpanName is the root span name for both the blocking and streaming paths.
	SpanName = "invoke_agent"
)

// providesFor routes any provider's agent-mode config here.
var providesFor = aiserver.ProvidesFor{Provider: kitbase.WildcardProvider, Mode: aiserver.ModeAgent}

// Timings the tests shorten.
var (
	startupTimeout    = 45 * time.Second
	shutdownGrace     = 5 * time.Second
	drainFallback     = 5 * time.Second
	keepaliveInterval = 15 * time.Second
)

func init() { handlerkit.RegisterAiSdkPackage(PackageName, PackageVersion) }

// Options configure NewHandler.
type Options struct {
	// WorkDir is the child's working directory. Required. It is checked to
	// exist on each call.
	WorkDir string
	// ConfigPath is the OpenCode config file, passed as OPENCODE_CONFIG.
	// Required. It is checked to exist on each call.
	ConfigPath string
	// Env is the child's whole environment, as "KEY=value" entries. Required:
	// a nil Env is refused, because a nil exec.Cmd.Env inherits the parent's.
	// An empty, non-nil Env is an empty environment. The handler sets
	// OPENCODE_CONFIG, OPENCODE_SERVER_USERNAME and OPENCODE_SERVER_PASSWORD,
	// replacing any value Env has for them.
	Env []string
	// Command is the executable and leading arguments. Empty means
	// {"opencode"}. It is a command, not a URL.
	Command []string
	// Model maps a config onto an OpenCode provider id and model id. Nil
	// means the default mapping (see NewHandler).
	Model func(cfg *aiserver.AiConfig) (providerID, modelID string, err error)
	// Agent is the OpenCode agent name sent with each message, if set.
	Agent string
	// CaptureContent gates prompt, completion and tool content on spans.
	// CaptureDefault follows aiserver.Configure, else on.
	CaptureContent aiserver.ContentCapture
}

// NewHandler returns the OpenCode agent handler (createOpenCodeHandler). It
// serves agent mode for any provider. It returns an error when WorkDir or
// ConfigPath is empty, Env is nil, or Command has an empty part or a URL.
//
// The default model mapping splits a "provider/model" name; otherwise
// Bedrock maps to amazon-bedrock, Gemini to google, and any other provider
// name is lowercased. An amazon-bedrock model id gets model.region as its
// inference-profile prefix.
func NewHandler(opts Options) (*aiserver.ProviderHandler, error) {
	handlerkit.ReportUsage("opencode-agents.createOpenCodeHandler", PackageName, PackageVersion)
	return newHandler(opts)
}

func newHandler(opts Options) (*aiserver.ProviderHandler, error) {
	if err := validate(opts); err != nil {
		return nil, err
	}
	opts.Env = slices.Clone(opts.Env)
	opts.Command = slices.Clone(opts.Command)
	if opts.Model == nil {
		opts.Model = defaultModel
	}
	a := &agent{opts: opts}
	return sdkapi.CreateHandler(providesFor, a.invoke, a.stream, opts.CaptureContent), nil
}

func validate(opts Options) error {
	switch {
	case strings.TrimSpace(opts.WorkDir) == "":
		return errors.New("opencodeagents: Options.WorkDir is required")
	case strings.TrimSpace(opts.ConfigPath) == "":
		return errors.New("opencodeagents: Options.ConfigPath is required")
	case opts.Env == nil:
		return errors.New("opencodeagents: Options.Env is required; pass an empty slice for an empty environment")
	}
	for _, part := range opts.Command {
		if part == "" || strings.Contains(part, "://") {
			return errors.New("opencodeagents: Options.Command takes a command, not a URL, and no empty part")
		}
	}
	return nil
}

// InvokeOptions are the optional arguments to OpenCodeAgents.
type InvokeOptions struct {
	// ToolHandlers are the config's tool executors and NativeTool sentinels.
	ToolHandlers map[string]aiserver.ToolHandler
	// Registry supplies further tools and handlers.
	Registry *aiserver.Registry
	// SkipJudges returns judge tasks instead of running judges.
	SkipJudges bool
	// Variables are forwarded to Invoke, not to the config.
	Variables map[string]any
	// Handler configures the handler built for this call. Its required
	// fields are required here too.
	Handler Options
}

// OpenCodeAgents invokes configKey through a fresh OpenCode handler. opts may
// be nil, but the handler options are required, so a nil opts fails.
func OpenCodeAgents(ctx context.Context, configKey, userInput string, userCtx ldcontext.Context, opts *InvokeOptions) (aiserver.ProviderResponse, error) {
	handlerkit.ReportUsage("opencode-agents.opencodeAgents", PackageName, PackageVersion)
	if opts == nil {
		opts = &InvokeOptions{}
	}
	h, err := newHandler(opts.Handler)
	if err != nil {
		return aiserver.ProviderResponse{}, err
	}
	return sdkapi.Config(aiserver.ConfigArgs{
		Key: configKey, Handlers: []*aiserver.ProviderHandler{h},
		ToolHandlers: opts.ToolHandlers, Registry: opts.Registry, SkipJudges: opts.SkipJudges,
	}).Invoke(ctx, userInput, userCtx, aiserver.WithVariables(opts.Variables))
}

// SessionError reports that the OpenCode session failed on its event bus.
// It carries the error name, never the provider's error body.
type SessionError struct {
	// Name is the bus error's name, or "" when it had none.
	Name string
}

func (e *SessionError) Error() string {
	if e.Name == "" {
		return "opencodeagents: the OpenCode session failed"
	}
	return fmt.Sprintf("opencodeagents: the OpenCode session failed (%s)", e.Name)
}

type agent struct{ opts Options }

type result struct {
	output    string
	usage     map[string]any
	abandoned bool
}

func (a *agent) invoke(ctx context.Context, req aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
	res, err := a.run(ctx, req, nil)
	if err != nil {
		return aiserver.HandlerResult{}, err
	}
	return aiserver.HandlerResult{Output: res.output, Usage: res.usage}, nil
}

func (a *agent) stream(ctx context.Context, req aiserver.HandlerRequest) iter.Seq2[aiserver.HandlerStreamEvent, error] {
	return func(yield func(aiserver.HandlerStreamEvent, error) bool) {
		res, err := a.run(ctx, req, func(t string) bool {
			return yield(aiserver.HandlerStreamEvent{Type: aiserver.EventChunk, Text: t}, nil)
		})
		if res.abandoned {
			return
		}
		if err != nil {
			yield(aiserver.HandlerStreamEvent{}, err)
			return
		}
		yield(aiserver.HandlerStreamEvent{Type: aiserver.EventDone, Output: &res.output, Usage: res.usage}, nil)
	}
}

// toolPlan is what one call offers OpenCode: the intersection of the
// config's tools and the tool handlers.
type toolPlan struct {
	offered   []string                        // OpenCode tool names to allow
	names     map[string]string               // OpenCode tool name → config key
	native    map[string]*aiserver.NativeTool // OpenCode tool name → sentinel
	callables map[string]aiserver.ToolFunc    // config key → bridged function
}

func planTools(cfg aiserver.AiConfig, handlers map[string]aiserver.ToolHandler) toolPlan {
	p := toolPlan{names: map[string]string{}, native: map[string]*aiserver.NativeTool{}, callables: map[string]aiserver.ToolFunc{}}
	for _, key := range slices.Sorted(maps.Keys(cfg.Tools)) {
		switch h := handlers[key].(type) {
		case *aiserver.NativeTool:
			if p.native[h.ToolName] == nil { // first config key wins
				p.native[h.ToolName] = h
				p.names[h.ToolName] = key
				p.offered = append(p.offered, h.ToolName)
			}
		case aiserver.ToolFunc:
			p.callables[key] = h
		}
	}
	return p
}

// ruleset denies every tool, then allows each offered one. OpenCode applies
// the last matching rule and appends this after the agent's own config, so
// it always wins. write and apply_patch are checked against edit, so any of
// the three sends one edit allow.
func ruleset(offered []string) []rule {
	rules := []rule{{Permission: "*", Action: "deny", Pattern: "*"}}
	seen := map[string]bool{}
	for _, name := range offered {
		if name == "write" || name == "apply_patch" {
			name = "edit"
		}
		if !seen[name] {
			seen[name] = true
			rules = append(rules, rule{Permission: name, Action: "allow", Pattern: "*"})
		}
	}
	return rules
}

func (a *agent) run(ctx context.Context, req aiserver.HandlerRequest, onText func(string) bool) (result, error) {
	providerID, modelID, err := a.opts.Model(&req.Config)
	if err != nil {
		return result{}, fmt.Errorf("opencodeagents: model: %w", err)
	}
	p := handlerkit.ComposePrompt(req, aiserver.ModeAgent)
	system := telemetry.ProviderName(providerName(req.Config))
	r := handlerkit.StartHandlerRun(ctx, kitbase.HandlerSpanOptions{
		Tracer: PackageName, Name: SpanName, System: system, Operation: "invoke_agent", RequestModel: modelID,
	}, req, p)
	defer r.EndOnPanic()
	fail := func(err error) (result, error) {
		r.Fail(err)
		return result{}, err
	}

	srv, err := startServer(r.Context(), a.opts)
	if err != nil {
		return fail(err)
	}
	defer srv.stop()

	plan := planTools(req.Config, req.ToolHandlers)
	if len(plan.callables) > 0 {
		b := newBridge()
		for key, fn := range plan.callables {
			t := req.Config.Tools[key]
			b.register(key, t.Description, t.Parameters, fn)
		}
		url, err := b.start(r.Context())
		if err != nil {
			return fail(err)
		}
		defer b.stop()
		runID := make([]byte, 8)
		_, _ = rand.Read(runID)
		name := "ld_" + hex.EncodeToString(runID)
		if err := registerBridge(r.Context(), srv, name, url+"/mcp/"+hex.EncodeToString(runID), b.token); err != nil {
			disconnectBridge(srv, name)
			return fail(err)
		}
		defer disconnectBridge(srv, name)
		for _, key := range slices.Sorted(maps.Keys(plan.callables)) {
			composed := bridgeToolName(name, key)
			plan.offered = append(plan.offered, composed)
			plan.names[composed] = key
		}
	}

	msg := message{Parts: []textPart{{Type: "text", Text: promptText(req, p)}},
		Model: modelRef{ProviderID: providerID, ModelID: modelID}, System: p.System, Agent: a.opts.Agent}
	red := newReducer(r, modelID, plan.names, plan.native, onText)
	_, err = runSession(r.Context(), srv, ruleset(plan.offered), msg, func(ev map[string]any) (bool, error) {
		err := red.handle(ev)
		return red.stopped, err
	})
	if red.stopped {
		r.Abandon()
		return result{abandoned: true}, nil
	}
	if err != nil {
		return fail(err)
	}
	red.close()
	out := red.out.String()
	r.Finish(modelID, out)
	return result{output: out, usage: red.usageMap()}, nil
}

// promptText flattens the composed turns into one user message. A lone user
// turn with no config conversation and no history is sent as it is.
func promptText(req aiserver.HandlerRequest, p handlerkit.Prompt) string {
	var lines []string
	for _, t := range p.Turns {
		text := handlerkit.ContentText(t)
		if text == "" {
			continue
		}
		label := "User"
		if t.Role == "assistant" {
			label = "Assistant"
		}
		lines = append(lines, label+": "+text)
	}
	if len(lines) == 1 && len(p.Turns) == 1 && p.Turns[0].Role == "user" && len(req.History) == 0 {
		return strings.TrimPrefix(lines[0], "User: ")
	}
	return strings.Join(lines, "\n\n")
}

// providerName is who served the model, from the variation, or "opencode".
func providerName(cfg aiserver.AiConfig) string {
	if n := strings.ToLower(strings.TrimSpace(cfg.Provider.Name)); n != "" {
		return n
	}
	return "opencode"
}

// defaultModel is the default Options.Model.
func defaultModel(cfg *aiserver.AiConfig) (providerID, modelID string, err error) {
	name := cfg.Model.Name
	if before, after, ok := strings.Cut(name, "/"); ok {
		providerID, modelID = before, after
		if strings.EqualFold(providerID, "bedrock") {
			providerID = "amazon-bedrock"
		}
	} else {
		modelID = name
		switch p := strings.ToLower(cfg.Provider.Name); p {
		case "bedrock":
			providerID = "amazon-bedrock"
		case "gemini":
			providerID = "google"
		default:
			providerID = p
		}
	}
	if providerID == "amazon-bedrock" {
		modelID = handlerkit.BedrockModelID(modelID, cfg.Model.Region)
	}
	return providerID, modelID, nil
}
