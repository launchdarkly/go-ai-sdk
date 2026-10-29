package experimental

import (
	"context"
	"crypto/sha1" //nolint:gosec // a sampling bucket, not a security check
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"maps"
	"slices"
	"strconv"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/launchdarkly/go-sdk-common/v3/ldreason"
	"github.com/launchdarkly/go-sdk-common/v3/ldvalue"
	"go.opentelemetry.io/otel/attribute"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/bridge"
	"github.com/launchdarkly/go-ai-sdk/internal/sdkcall"
)

// Shadow requests. Experimental: every name in this file may change in a
// minor release.
//
// ShadowConfig is Config that also runs the shadow variations LaunchDarkly
// attaches to the matched targeting rule or the fallthrough. The served
// variation runs exactly as Config runs it. Once it succeeds, each sampled
// shadow runs on the same input in the background, or is handed to OnTask
// to run later with RunShadow. A shadow's response is discarded; it is
// tracked on the normal $ld:ai:* events with isShadow.
//
//	shadowed, err := experimental.ShadowConfig(
//		aiserver.ConfigArgs{Key: "my-ai-config", Handlers: handlers}, experimental.ShadowOptions{})
//	resp, err := shadowed.Invoke(ctx, "Summarize this document", userCtx)

const (
	defaultShadowTimeout = 30 * time.Second
	fallthroughSlot      = "fallthrough"
	bucketScale          = float64(0xFFFFFFFFFFFFFFF)
)

var errShadowTimeout = errors.New("timeout")

// ShadowOptions are ShadowConfig's options beyond Config's. Experimental.
type ShadowOptions struct {
	// Timeout bounds each shadow run. Zero means 30 seconds.
	Timeout time.Duration
	// OnTask, when set, receives each sampled shadow instead of the SDK
	// running it. It should only enqueue the task; run it with RunShadow.
	OnTask func(ShadowTask)
}

// ShadowAIConfig is the value ShadowConfig returns. Its Invoke and Stream
// return what Config's would. Experimental.
type ShadowAIConfig struct {
	args aiserver.ConfigArgs
	opts ShadowOptions
}

// ShadowTask is everything needed to run and track one sampled shadow, with
// no flag evaluation. It survives a JSON round trip, except Config.Raw.
// Experimental.
type ShadowTask struct {
	ConfigKey       string                 `json:"configKey"`
	ShadowRequestID string                 `json:"shadowRequestId"`
	Config          aiserver.AiConfig      `json:"config"`
	Meta            aiserver.VariationMeta `json:"meta"`
	UserInput       string                 `json:"userInput"`
	History         []aiserver.Message     `json:"history,omitempty"`
	Variables       map[string]any         `json:"variables,omitempty"`
	UserContext     ldcontext.Context      `json:"userContext"`
	Timeout         time.Duration          `json:"timeout"`
	TrackData       map[string]any         `json:"trackData"`
	SpanAttributes  map[string]any         `json:"spanAttributes"`
}

// ShadowResult is the outcome of one shadow run. It has no response text:
// the answer is discarded. Experimental.
type ShadowResult struct {
	Usage     aiserver.TokenUsage `json:"usage"`
	TrackData map[string]any      `json:"trackData"`
	Error     string              `json:"error,omitempty"`
}

// ShadowConfig is Config, and also runs the shadows of the matched rule or
// fallthrough. Shadows start only after the served variation succeeds and
// never delay or change its response. A running shadow ends at its timeout
// or when aiserver.Shutdown cancels it. It returns an error for a negative Timeout.
// Experimental.
func ShadowConfig(args aiserver.ConfigArgs, opts ShadowOptions) (*ShadowAIConfig, error) {
	aiserver.ReportUsage("experimental.shadowConfig", "", "")
	if opts.Timeout < 0 {
		return nil, errors.New("ShadowOptions.Timeout must be greater than 0")
	}
	if opts.Timeout == 0 {
		opts.Timeout = defaultShadowTimeout
	}
	return &ShadowAIConfig{args: args, opts: opts}, nil
}

// RunShadow runs one ShadowTask. It never fails: a failure is returned in
// ShadowResult.Error. Experimental.
func RunShadow(ctx context.Context, task ShadowTask, handlers []*aiserver.ProviderHandler) ShadowResult {
	aiserver.ReportUsage("experimental.runShadow", "", "")
	return runShadow(ctx, task, handlers)
}

// Invoke runs the served variation like Config(...).Invoke, then starts its
// shadows. Experimental.
func (s *ShadowAIConfig) Invoke(ctx context.Context, userInput string, userCtx ldcontext.Context, opts ...aiserver.CallOption) (aiserver.ProviderResponse, error) {
	e := &evaluation{runID: uuid.NewString()}
	resp, err := s.primary(e).Invoke(ctx, userInput, userCtx, opts...)
	if err == nil {
		s.startShadows(ctx, e, userInput, userCtx, opts)
	}
	return resp, err
}

// Stream runs the served variation like Config(...).Stream, and starts its
// shadows at the done event. Experimental.
func (s *ShadowAIConfig) Stream(ctx context.Context, userInput string, userCtx ldcontext.Context, opts ...aiserver.CallOption) iter.Seq2[aiserver.StreamEvent, error] {
	return func(yield func(aiserver.StreamEvent, error) bool) {
		e := &evaluation{runID: uuid.NewString()}
		for ev, err := range s.primary(e).Stream(ctx, userInput, userCtx, opts...) {
			if err == nil && ev.Type == aiserver.EventDone {
				s.startShadows(ctx, e, userInput, userCtx, opts)
			}
			if !yield(ev, err) {
				return
			}
		}
	}
}

func (s *ShadowAIConfig) primary(e *evaluation) *aiserver.AIConfig {
	args := s.args
	args.SDKCall = sdkcall.Mark()
	return bridge.ConfigWithOptions(args, bridge.RunOptions{Variation: e.variation, RunID: e.runID}).(*aiserver.AIConfig)
}

func (s *ShadowAIConfig) handlers() []*aiserver.ProviderHandler {
	handlers := slices.Clone(s.args.Handlers)
	if s.args.Registry != nil {
		handlers = append(handlers, s.args.Registry.Handlers()...)
	}
	return handlers
}

func (s *ShadowAIConfig) startShadows(ctx context.Context, e *evaluation, userInput string, userCtx ldcontext.Context, opts []aiserver.CallOption) {
	defer func() {
		if r := recover(); r != nil {
			bridge.Logger().Warn("Skipping shadow requests: they could not be prepared", "panic", r)
		}
	}()
	handlers := s.handlers()
	call := aiserver.NewCallOptions(opts...)
	for _, task := range selectTasks(s.args.Key, e, handlers, userInput, userCtx, call, s.opts.Timeout) {
		if s.opts.OnTask != nil {
			handOver(s.opts.OnTask, task)
			continue
		}
		startBackground(ctx, task, handlers)
	}
}

func handOver(onTask func(ShadowTask), task ShadowTask) {
	defer func() {
		if r := recover(); r != nil {
			bridge.Logger().Warn("The shadow requests OnTask callback failed", "panic", r)
		}
	}()
	onTask(task)
}

var (
	// running counts background shadows, so tests can wait for them.
	running   sync.WaitGroup
	cancelsMu sync.Mutex
	cancels   = map[*context.CancelFunc]struct{}{}
)

func init() { bridge.OnShutdown(cancelRunning) }

func cancelRunning(context.Context) {
	cancelsMu.Lock()
	defer cancelsMu.Unlock()
	for cancel := range cancels {
		(*cancel)()
	}
	clear(cancels)
}

// startBackground detaches the shadow from the caller's cancellation, so it
// outlives the request; it keeps the context's values and trace. Shutdown
// cancels it.
func startBackground(ctx context.Context, task ShadowTask, handlers []*aiserver.ProviderHandler) {
	ctx, cancel := context.WithCancel(context.WithoutCancel(ctx))
	cancelsMu.Lock()
	cancels[&cancel] = struct{}{}
	cancelsMu.Unlock()
	running.Go(func() {
		defer func() {
			cancelsMu.Lock()
			delete(cancels, &cancel)
			cancelsMu.Unlock()
			cancel()
		}()
		runShadow(ctx, task, handlers)
	})
}

type detailClient interface {
	JSONVariationDetail(key string, c ldcontext.Context, def ldvalue.Value) (ldvalue.Value, ldreason.EvaluationDetail, error)
}

// evaluation records the served variation's one flag evaluation.
type evaluation struct {
	runID               string
	reason              ldreason.EvaluationReason
	shadows             any
	primaryVariationKey string
}

// variation replaces Config's JSONVariation call: it reads the reason too,
// and removes shadowConfiguration before the served variation runs.
func (e *evaluation) variation(client bridge.VariationClient, key string, c ldcontext.Context, def ldvalue.Value) (ldvalue.Value, error) {
	detailed, ok := client.(detailClient)
	if !ok {
		return client.JSONVariation(key, c, def)
	}
	v, detail, err := detailed.JSONVariationDetail(key, c, def)
	e.reason = detail.Reason
	var value map[string]any
	if json.Unmarshal([]byte(v.JSONString()), &value) != nil || value == nil {
		return v, err
	}
	meta, _ := value["_ldMeta"].(map[string]any)
	e.primaryVariationKey, _ = meta["variationKey"].(string)
	shadows, present := value["shadowConfiguration"]
	if !present {
		return v, err
	}
	e.shadows = shadows
	delete(value, "shadowConfiguration")
	return ldvalue.FromJSONMarshal(value), err
}

func slotFor(reason ldreason.EvaluationReason) (string, bool) {
	switch reason.GetKind() {
	case ldreason.EvalReasonFallthrough:
		return fallthroughSlot, true
	case ldreason.EvalReasonRuleMatch:
		return reason.GetRuleID(), reason.GetRuleID() != ""
	}
	return "", false
}

type shadowEntry struct {
	id, variationKey, ownerKind, ownerID string
	samplingRate, endsAt                 float64
}

func parseEntry(raw any) (shadowEntry, bool) {
	m, _ := raw.(map[string]any)
	e := shadowEntry{}
	e.id, _ = m["id"].(string)
	e.variationKey, _ = m["variationKey"].(string)
	e.ownerKind, _ = m["ownerKind"].(string)
	e.ownerID, _ = m["ownerId"].(string)
	rate, rateOK := m["samplingRate"].(float64)
	endsAt, endsOK := m["endsAt"].(float64)
	e.samplingRate, e.endsAt = rate, endsAt
	return e, e.id != "" && e.variationKey != "" && rateOK && endsOK
}

func inSample(shadowID string, c ldcontext.Context, samplingRate float64) bool {
	canonical, _, ok := aiserver.ContextIdentity(c)
	if !ok {
		return false
	}
	sum := sha1.Sum([]byte(shadowID + "." + canonical)) //nolint:gosec // a sampling bucket
	n, _ := strconv.ParseUint(hex.EncodeToString(sum[:])[:15], 16, 64)
	return float64(n)/bucketScale < samplingRate
}

// shadowVariation parses an enabled shadow variation without its tools,
// judges and shadows.
func shadowVariation(raw any) (aiserver.AiConfig, aiserver.VariationMeta, bool) {
	value, _ := deepCopy(raw).(map[string]any)
	metaRaw, _ := value["_ldMeta"].(map[string]any)
	if metaRaw["enabled"] != true {
		return aiserver.AiConfig{}, aiserver.VariationMeta{}, false
	}
	var meta aiserver.VariationMeta
	b, _ := json.Marshal(metaRaw)
	if json.Unmarshal(b, &meta) != nil {
		return aiserver.AiConfig{}, aiserver.VariationMeta{}, false
	}
	for _, k := range []string{"_ldMeta", "tools", "judgeConfiguration", "shadowConfiguration"} {
		delete(value, k)
	}
	cfg, err := aiserver.ParseAiConfig(value)
	return cfg, meta, err == nil
}

func handlerFor(cfg aiserver.AiConfig, meta aiserver.VariationMeta, handlers []*aiserver.ProviderHandler) *aiserver.ProviderHandler {
	if cfg.Provider.Name == "" {
		return nil
	}
	h, err := aiserver.SelectHandler(handlers, cfg.Provider.Name, aiserver.NormalizeMode(meta.Mode))
	if err != nil {
		return nil
	}
	return h
}

func selectTasks(configKey string, e *evaluation, handlers []*aiserver.ProviderHandler, userInput string, userCtx ldcontext.Context, call aiserver.CallOptions, timeout time.Duration) []ShadowTask {
	slot, ok := slotFor(e.reason)
	if !ok {
		return nil
	}
	configuration, _ := e.shadows.(map[string]any)
	shadows, _ := configuration["shadows"].(map[string]any)
	variations, _ := configuration["variations"].(map[string]any)
	entries, _ := shadows[slot].([]any)
	nowMs := float64(time.Now().UnixMilli())
	var tasks []ShadowTask
	for _, raw := range entries {
		entry, ok := parseEntry(raw)
		if !ok || entry.endsAt <= nowMs || !inSample(entry.id, userCtx, entry.samplingRate) {
			continue
		}
		cfg, meta, ok := shadowVariation(variations[entry.variationKey])
		if !ok {
			continue
		}
		if handlerFor(cfg, meta, handlers) == nil {
			bridge.Logger().Debug("Skipping shadow request: no handler for its provider", "shadowRequestId", entry.id)
			continue
		}
		vars, _ := deepCopy(call.Variables).(map[string]any)
		tasks = append(tasks, ShadowTask{
			ConfigKey:       configKey,
			ShadowRequestID: entry.id,
			Config:          cfg,
			Meta:            meta,
			UserInput:       userInput,
			History:         copyHistory(call.History),
			Variables:       vars,
			UserContext:     userCtx,
			Timeout:         timeout,
			TrackData: map[string]any{
				"runId":               e.runID,
				"isShadow":            true,
				"primaryVariationKey": e.primaryVariationKey,
				"shadowRequestId":     entry.id,
				"ownerKind":           entry.ownerKind,
				"ownerId":             entry.ownerID,
			},
			SpanAttributes: map[string]any{
				"launchdarkly.is_shadow":                    true,
				"launchdarkly.shadow.request.id":            entry.id,
				"launchdarkly.shadow.primary.variation.key": e.primaryVariationKey,
			},
		})
	}
	return tasks
}

func runShadow(ctx context.Context, task ShadowTask, handlers []*aiserver.ProviderHandler) (res ShadowResult) {
	res = ShadowResult{TrackData: maps.Clone(task.TrackData)}
	defer func() {
		if r := recover(); r != nil {
			res = ShadowResult{TrackData: maps.Clone(task.TrackData), Error: fmt.Sprintf("panic: %v", r)}
		}
	}()
	if err := ctx.Err(); err != nil {
		res.Error = err.Error()
		return res
	}
	h := handlerFor(task.Config, task.Meta, handlers)
	if h == nil {
		res.Error = "no handler for provider " + task.Config.Provider.Name
		return res
	}
	timeout := task.Timeout
	if timeout <= 0 {
		timeout = defaultShadowTimeout
	}
	runID, _ := task.TrackData["runId"].(string)
	extra := maps.Clone(task.TrackData)
	delete(extra, "runId")
	value := shadowValue(task.Config, task.Meta)
	run := bridge.ConfigWithOptions(aiserver.ConfigArgs{
		Key:      task.ConfigKey,
		Handlers: []*aiserver.ProviderHandler{withDeadline(h, timeout)},
		SDKCall:  sdkcall.Mark(),
	}, bridge.RunOptions{
		Variation: func(bridge.VariationClient, string, ldcontext.Context, ldvalue.Value) (ldvalue.Value, error) {
			return value, nil
		},
		RunID:          runID,
		TrackData:      extra,
		SpanAttributes: toAttributes(task.SpanAttributes),
	}).(*aiserver.AIConfig)
	vars, _ := deepCopy(task.Variables).(map[string]any)
	resp, err := run.Invoke(ctx, task.UserInput, task.UserContext,
		aiserver.WithVariables(vars), aiserver.WithHistory(copyHistory(task.History)))
	if err != nil {
		res.Error = err.Error()
		return res
	}
	res.Usage = resp.Usage
	res.TrackData = trackDataMap(resp.TrackData)
	maps.Copy(res.TrackData, task.TrackData)
	return res
}

// shadowValue rebuilds the variation value Config parses: the config's own
// fields plus its _ldMeta.
func shadowValue(cfg aiserver.AiConfig, meta aiserver.VariationMeta) ldvalue.Value {
	value, _ := deepCopy(cfg.Raw).(map[string]any)
	if value == nil {
		b, _ := json.Marshal(cfg)
		_ = json.Unmarshal(b, &value)
	}
	value["_ldMeta"] = meta
	return ldvalue.FromJSONMarshal(value)
}

// withDeadline makes the handler fail with errShadowTimeout once timeout
// passes, so the normal tracking path records the timeout once. A late
// result is discarded.
func withDeadline(h *aiserver.ProviderHandler, timeout time.Duration) *aiserver.ProviderHandler {
	invoke := func(ctx context.Context, req aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
		ctx, cancel := context.WithTimeoutCause(ctx, timeout, errShadowTimeout)
		defer cancel()
		type outcome struct {
			res aiserver.HandlerResult
			err error
		}
		done := make(chan outcome, 1)
		go func() {
			defer func() {
				if r := recover(); r != nil {
					done <- outcome{err: fmt.Errorf("handler panicked: %v", r)}
				}
			}()
			res, err := h.Invoke(ctx, req)
			done <- outcome{res, err}
		}()
		select {
		case o := <-done:
			if errors.Is(context.Cause(ctx), errShadowTimeout) {
				return aiserver.HandlerResult{}, errShadowTimeout
			}
			return o.res, o.err
		case <-ctx.Done():
			return aiserver.HandlerResult{}, context.Cause(ctx)
		}
	}
	return &aiserver.ProviderHandler{ProvidesFor: h.ProvidesFor, CaptureContent: h.CaptureContent, Invoke: invoke}
}

func trackDataMap(td aiserver.TrackData) map[string]any {
	var m map[string]any
	b, _ := json.Marshal(td)
	_ = json.Unmarshal(b, &m)
	return m
}

func toAttributes(m map[string]any) []attribute.KeyValue {
	var attrs []attribute.KeyValue
	for _, k := range slices.Sorted(maps.Keys(m)) {
		switch v := m[k].(type) {
		case string:
			attrs = append(attrs, attribute.String(k, v))
		case bool:
			attrs = append(attrs, attribute.Bool(k, v))
		case float64:
			attrs = append(attrs, attribute.Float64(k, v))
		case int:
			attrs = append(attrs, attribute.Int(k, v))
		}
	}
	return attrs
}

func deepCopy(v any) any {
	switch t := v.(type) {
	case map[string]any:
		if t == nil {
			return t
		}
		out := make(map[string]any, len(t))
		for k, val := range t {
			out[k] = deepCopy(val)
		}
		return out
	case []any:
		if t == nil {
			return t
		}
		out := make([]any, len(t))
		for i, val := range t {
			out[i] = deepCopy(val)
		}
		return out
	}
	return v
}

func copyHistory(history []aiserver.Message) []aiserver.Message {
	if history == nil {
		return nil
	}
	out := make([]aiserver.Message, len(history))
	for i, m := range history {
		if m.Blocks != nil {
			blocks := make([]aiserver.ContentBlock, len(m.Blocks))
			for j, b := range m.Blocks {
				if b.Source != nil {
					src := *b.Source
					b.Source = &src
				}
				blocks[j] = b
			}
			m.Blocks = blocks
		}
		out[i] = m
	}
	return out
}
