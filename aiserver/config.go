package aiserver

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"math"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/launchdarkly/go-sdk-common/v3/ldvalue"

	"github.com/launchdarkly/go-ai-sdk/internal/sdkcall"
)

// ConfigArgs are the options for Config.
type ConfigArgs struct {
	Key          string
	Handlers     []*ProviderHandler // JS `handler` (one or many); Go has no union, so always a slice
	ToolHandlers map[string]ToolHandler
	Registry     *Registry
	SkipJudges   bool
	// SDKCall suppresses $ld:ai:sdk:usage for this value. Only internal/sdkapi
	// can build a non-zero token. Applications leave it zero.
	SDKCall sdkcall.Token
}

// AIConfig is the value Config returns.
type AIConfig struct {
	args ConfigArgs
}

// Config is the unified AI Config entry point.
func Config(args ConfigArgs) *AIConfig { return &AIConfig{args: args} }

func variationToAny(v ldvalue.Value) any {
	var out any
	_ = json.Unmarshal([]byte(v.JSONString()), &out)
	return out
}

func readVariation(ctx context.Context, key string, userCtx ldcontext.Context) (raw map[string]any, meta *VariationMeta, err error) {
	c, err := InitClient(ctx, InitOptions{})
	if err != nil {
		return nil, nil, err
	}
	v, evalErr := c.JSONVariation(key, userCtx, ldvalue.ObjectBuild().SetBool("enabled", false).Build())
	raw, _ = variationToAny(v).(map[string]any)
	if m, ok := raw["_ldMeta"].(map[string]any); ok {
		meta = parseMeta(m)
	}
	if meta == nil {
		// No AI Config came back. Keep the SDK's reason (client not ready,
		// unknown flag, wrong type) so the error names the real cause.
		if evalErr != nil {
			return nil, nil, fmt.Errorf("%w: variation %s returned null: %v", ErrConfigNotFound, key, evalErr)
		}
		return nil, nil, fmt.Errorf("%w: variation %s returned null: the flag does not exist or is not an AI Config", ErrConfigNotFound, key)
	}
	return raw, meta, nil
}

// parseMeta reads `_ldMeta` field by field, so one malformed field (for
// example a non-integer modelVersion) drops only that field.
func parseMeta(m map[string]any) *VariationMeta {
	meta := &VariationMeta{}
	meta.Enabled, _ = m["enabled"].(bool)
	meta.VariationKey, _ = m["variationKey"].(string)
	meta.Mode, _ = m["mode"].(string)
	meta.ModelKey, _ = m["modelKey"].(string)
	if v, ok := m["version"].(float64); ok && v == math.Trunc(v) {
		meta.Version = int(v)
	}
	if v, ok := m["modelVersion"].(float64); ok && v == math.Trunc(v) {
		n := int(v)
		meta.ModelVersion = &n
	}
	return meta
}

func extractVariation(ctx context.Context, key string, userCtx ldcontext.Context) (AiConfig, VariationMeta, error) {
	raw, meta, err := readVariation(ctx, key, userCtx)
	if err != nil {
		return AiConfig{}, VariationMeta{}, err
	}
	if !meta.Enabled {
		return AiConfig{}, VariationMeta{}, fmt.Errorf("%w: variation %s is not enabled", ErrConfigDisabled, key)
	}
	cfg, err := ParseAiConfig(raw)
	if err != nil {
		return AiConfig{}, VariationMeta{}, errorf("invalid AI config for %q: %w", key, err)
	}
	return cfg, *meta, nil
}

func selectHandler(cfg AiConfig, meta VariationMeta, handlers []*ProviderHandler) (*ProviderHandler, error) {
	if cfg.Provider.Name == "" {
		return nil, errorf("provider not found")
	}
	return SelectHandler(handlers, cfg.Provider.Name, NormalizeMode(meta.Mode))
}

// SelectHandler is the config() routing rule: the handler whose
// ProvidesFor is exactly {provider, mode}, else {"*", mode}, with the
// handler's mode normalized like the config's ("completion" is messages).
// Every path that routes a config by provider uses it, so the rule lives in
// one place.
func SelectHandler(handlers []*ProviderHandler, provider string, mode Mode) (*ProviderHandler, error) {
	for _, want := range []ProvidesFor{{provider, mode}, {WildcardProvider, mode}} {
		for _, h := range handlers {
			if h.ProvidesFor.Provider == want.Provider && handlerMode(h) == want.Mode {
				return h, nil
			}
		}
	}
	for _, h := range handlers {
		if h.ProvidesFor.Provider == provider || h.ProvidesFor.Provider == WildcardProvider {
			return nil, fmt.Errorf("%w %s with mode %s", ErrNoHandler, provider, mode)
		}
	}
	return nil, fmt.Errorf("%w %s", ErrNoHandler, provider)
}

type resolved struct {
	cfg      AiConfig
	meta     VariationMeta
	handler  *ProviderHandler
	handlers []*ProviderHandler
	tools    map[string]ToolHandler
}

func (c *AIConfig) resolve(ctx context.Context, userCtx ldcontext.Context) (resolved, error) {
	handlers := resolveHandlers(c.args.Registry, c.args.Handlers)
	tools := resolveTools(c.args.Registry, c.args.ToolHandlers)
	cfg, meta, err := extractVariation(ctx, c.args.Key, userCtx)
	if err != nil {
		return resolved{}, err
	}
	h, err := selectHandler(cfg, meta, handlers)
	if err != nil {
		return resolved{}, err
	}
	return resolved{cfg, meta, h, handlers, tools}, nil
}

func resolveOutputFormat(raw any, outputFormat map[string]any) any {
	s, isStr := raw.(string)
	if outputFormat == nil || !isStr {
		return raw
	}
	if v, ok := ParseJSONWithPossibleFences(s); ok {
		return v
	}
	return s // best effort
}

// Invoke runs the config once. variables and history may be nil.
func (c *AIConfig) Invoke(ctx context.Context, userInput string, userCtx ldcontext.Context, opts ...CallOption) (ProviderResponse, error) {
	co := NewCallOptions(opts...)
	variables, history := co.Variables, co.History
	if !c.args.SDKCall.Marked() {
		reportUsage("client.config.invoke", "", "")
	}
	r, err := c.resolve(ctx, userCtx)
	if err != nil {
		return ProviderResponse{}, err
	}
	res, err := executeAndTrack(ctx, execArgs{
		configKey: c.args.Key, config: r.cfg, meta: r.meta, userCtx: userCtx, handler: r.handler,
		userInput: userInput, tools: r.tools, variables: variables, history: history,
	})
	if err != nil {
		return ProviderResponse{}, err
	}
	parsed := resolveOutputFormat(res.Output, r.cfg.OutputFormat)
	resp := ProviderResponse{Response: parsed, Usage: res.Usage, TrackData: res.TrackData}
	ja := judgeArgs{cfg: r.cfg, userCtx: userCtx, handler: r.handler, handlers: r.handlers,
		userInput: userInput, output: outputString(parsed), parentTD: res.TrackData, trajectory: res.Trajectory}
	if c.args.SkipJudges {
		resp.JudgeTasks, err = buildJudgeTasks(ctx, ja)
	} else {
		resp.JudgeResults, err = runJudges(ctx, ja)
	}
	return resp, err
}

// Stream yields chunk events then one done event. Usage is reported and the
// ctx (conversation id, parent span) is captured when Stream is called.
func (c *AIConfig) Stream(ctx context.Context, userInput string, userCtx ldcontext.Context, opts ...CallOption) iter.Seq2[StreamEvent, error] {
	co := NewCallOptions(opts...)
	variables, history := co.Variables, co.History
	if !c.args.SDKCall.Marked() {
		reportUsage("client.config.stream", "", "")
	}
	return func(yield func(StreamEvent, error) bool) {
		r, err := c.resolve(ctx, userCtx)
		if err != nil {
			yield(StreamEvent{}, err)
			return
		}
		var done streamDone
		for ev, err := range executeAndStream(ctx, execArgs{
			configKey: c.args.Key, config: r.cfg, meta: r.meta, userCtx: userCtx, handler: r.handler,
			userInput: userInput, tools: r.tools, variables: variables, history: history,
		}, &done) {
			if err != nil {
				yield(StreamEvent{}, err)
				return
			}
			if !yield(StreamEvent{Type: EventChunk, Text: ev.Text}, nil) {
				return
			}
		}
		out := StreamEvent{Type: EventDone, Response: done.response, Usage: done.usage}
		if !c.args.SkipJudges {
			jr, err := runJudges(ctx, judgeArgs{cfg: r.cfg, userCtx: userCtx, handler: r.handler, handlers: r.handlers,
				userInput: userInput, output: done.response, parentTD: done.td, trajectory: done.trajectory})
			if err != nil {
				yield(StreamEvent{}, err)
				return
			}
			if len(jr) > 0 {
				out.JudgeResults = jr
			}
		}
		yield(out, nil)
	}
}

// InspectResult is the result of InspectConfig.
type InspectResult struct {
	Enabled bool
	Config  *AiConfig
	Meta    *VariationMeta
}

// InspectConfig reads a variation without invoking a model. It never errors.
func InspectConfig(ctx context.Context, key string, userCtx ldcontext.Context) InspectResult {
	reportUsage("client.inspectConfig", "", "")
	raw, meta, err := readVariation(ctx, key, userCtx)
	if err != nil || meta == nil {
		return InspectResult{}
	}
	res := InspectResult{Enabled: meta.Enabled, Meta: meta}
	if meta.Enabled {
		if cfg, err := ParseAiConfig(raw); err == nil {
			res.Config = &cfg
		}
	}
	return res
}
