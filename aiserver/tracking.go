package aiserver

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"maps"
	"slices"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/launchdarkly/go-sdk-common/v3/ldvalue"

	"github.com/launchdarkly/go-ai-sdk/internal/trajectory"
)

func track(event string, ctx ldcontext.Context, td TrackData, metric float64) {
	if !ctx.IsDefined() { // no context, no tracking (an undefined context cannot be sent)
		return
	}
	c, err := GetClient()
	if err != nil {
		return
	}
	b, _ := json.Marshal(td)
	safeTrack(c, event, ctx, metric, ldvalue.Parse(b))
}

// Track sends a LaunchDarkly AI tracking event with td as its data. It is
// for handler and adapter packages; a missing client or a failing track
// never returns an error, because telemetry must not fail an AI call.
func Track(event string, ctx ldcontext.Context, td TrackData, metric float64) {
	track(event, ctx, td, metric)
}

// WrapToolHandlers wraps each tool so a call emits `$ld:ai:tool_call`.
// A NativeTool keeps its ToolName and gains a working Track. Synthetic
// `__handoff_` routing tools are passed through without tracking.
func WrapToolHandlers(tools map[string]ToolHandler, userCtx ldcontext.Context, td TrackData) map[string]ToolHandler {
	out := make(map[string]ToolHandler, len(tools))
	for name, h := range tools {
		toolTD := td
		toolTD.ToolKey = name
		switch t := h.(type) {
		case *NativeTool:
			out[name] = &NativeTool{ToolName: t.ToolName, track: func() { track("$ld:ai:tool_call", userCtx, toolTD, 1) }}
		case ToolFunc:
			if strings.HasPrefix(name, trajectory.HandoffPrefix) {
				out[name] = t
				continue
			}
			out[name] = ToolFunc(func(ctx context.Context, args map[string]any) (any, error) {
				track("$ld:ai:tool_call", userCtx, toolTD, 1)
				return t(ctx, args)
			})
		}
	}
	return out
}

// ContextToMap is the template view of an LD context, injected as
// variables.ldContext.
func ContextToMap(c ldcontext.Context) map[string]any {
	b, err := json.Marshal(c)
	if err != nil {
		return map[string]any{}
	}
	var m map[string]any
	_ = json.Unmarshal(b, &m)
	return m
}

type execArgs struct {
	configKey string
	config    AiConfig
	meta      VariationMeta
	userCtx   ldcontext.Context
	handler   *ProviderHandler
	userInput string
	tools     map[string]ToolHandler
	variables map[string]any
	graphKey  string
	history   []Message
}

// execResult is one tracked invocation's outcome.
type execResult struct {
	Output     any
	Usage      TokenUsage
	TrackData  TrackData
	Trajectory string // rendered tool-call trajectory; "" when no tool was observable
}

func newTrackData(configKey string, cfg AiConfig, meta VariationMeta, graphKey string) TrackData {
	td := TrackData{
		RunID:        uuid.NewString(),
		ConfigKey:    configKey,
		VariationKey: meta.VariationKey,
		Version:      meta.Version,
		ModelName:    cfg.Model.Name,
		ProviderName: cfg.Provider.Name,
		GraphKey:     graphKey,
		ModelKey:     meta.ModelKey,
		ModelVersion: meta.ModelVersion,
		// The Go server SDK does not expose the environment id, so it is
		// configured, as in Python; it becomes feature_flag.set.id.
		EnvironmentID: envValue("LD_ENVIRONMENT_ID"),
	}
	if td.Version == 0 {
		td.Version = 1
	}
	return td
}

// prepare builds the handler request: tools are first wrapped by the
// trajectory recorder (on the original map, so a native tool is still seen
// as native) and then by the tool-call tracking wrapper.
func (a execArgs) prepare() (TrackData, HandlerRequest, *trajectory.Recorder) {
	td := newTrackData(a.configKey, a.config, a.meta, a.graphKey)
	rec := trajectory.NewRecorder()
	exposed := make(map[string]bool, len(a.config.Tools))
	for k := range a.config.Tools {
		exposed[k] = true
	}
	recorded := make(map[string]ToolHandler, len(a.tools))
	// Sorted, so the trajectory's "Tools available" line is byte-stable.
	for _, name := range slices.Sorted(maps.Keys(a.tools)) {
		h := a.tools[name]
		if fn, ok := h.(ToolFunc); ok && trajectory.ShouldObserve(name, exposed) {
			recorded[name] = ToolFunc(rec.Observe(name, fn))
			continue
		}
		recorded[name] = h
	}
	vars := maps.Clone(a.variables)
	if vars == nil {
		vars = map[string]any{}
	}
	vars[VarLDContext] = ContextToMap(a.userCtx) // always overwrites a caller's ldContext
	vars[varLDContextTyped] = a.userCtx
	vars[VarLD] = td
	return td, HandlerRequest{
		Config:       a.config,
		UserInput:    a.userInput,
		ToolHandlers: WrapToolHandlers(recorded, a.userCtx, td),
		Variables:    vars,
		History:      a.history,
	}, rec
}

func trackOutcome(userCtx ldcontext.Context, td TrackData, start time.Time, err error) {
	track("$ld:ai:duration:total", userCtx, td, float64(time.Since(start).Milliseconds()))
	if err != nil {
		track("$ld:ai:generation:error", userCtx, td, 1)
		return
	}
	track("$ld:ai:generation:success", userCtx, td, 1)
}

// trackPanic, deferred around a handler call, tracks a panicking handler as
// a failed generation and re-panics: Go's panic is the handler's other
// error path, and the caller's own recovery must still see it.
func trackPanic(userCtx ldcontext.Context, td TrackData, start time.Time) {
	if r := recover(); r != nil {
		trackOutcome(userCtx, td, start, fmt.Errorf("handler panicked: %v", r))
		panic(r)
	}
}

func trackTokens(userCtx ldcontext.Context, td TrackData, u TokenUsage) {
	if u.Total > 0 {
		track("$ld:ai:tokens:total", userCtx, td, u.Total)
	}
	if u.Input > 0 {
		track("$ld:ai:tokens:input", userCtx, td, u.Input)
	}
	if u.Output > 0 {
		track("$ld:ai:tokens:output", userCtx, td, u.Output)
	}
}

func executeAndTrack(ctx context.Context, a execArgs) (execResult, error) {
	td, req, rec := a.prepare()
	ctx = startInvocation(ctx, a.handler.CaptureContent)
	start := time.Now()
	res, err := func() (HandlerResult, error) {
		defer trackPanic(a.userCtx, td, start)
		return a.handler.Invoke(ctx, req)
	}()
	trackOutcome(a.userCtx, td, start, err)
	if err != nil {
		return execResult{TrackData: td, Trajectory: rec.Render()}, err
	}
	usage := ParseUsage(res.Usage)
	trackTokens(a.userCtx, td, usage)
	if res.Output == nil {
		res.Output = ""
	}
	return execResult{Output: res.Output, Usage: usage, TrackData: td, Trajectory: rec.Render()}, nil
}

func outputString(out any) string {
	switch t := out.(type) {
	case nil:
		return ""
	case string:
		return t
	default:
		b, _ := json.Marshal(t)
		return string(b)
	}
}

type streamDone struct {
	response   string
	usage      TokenUsage
	td         TrackData
	trajectory string
}

// executeAndStream yields chunk events; done is filled in after the stream
// completes. It falls back to Invoke when the handler has no Stream. An
// abandoned stream tracks neither success nor error.
func executeAndStream(ctx context.Context, a execArgs, done *streamDone) iter.Seq2[HandlerStreamEvent, error] {
	return func(yield func(HandlerStreamEvent, error) bool) {
		td, req, rec := a.prepare()
		ctx := startInvocation(ctx, a.handler.CaptureContent)
		start := time.Now()
		var full strings.Builder
		var rawUsage map[string]any
		var err error
		// A panic raised by the consumer's loop body passes through emit; it
		// is not a handler failure, so only other panics are tracked.
		inYield, tracked := false, false
		emit := func(text string) bool {
			inYield = true
			ok := yield(HandlerStreamEvent{Type: EventChunk, Text: text}, nil)
			inYield = false
			return ok
		}
		defer func() {
			if tracked || inYield {
				return // recover is not called, so the panic continues
			}
			if r := recover(); r != nil {
				trackOutcome(a.userCtx, td, start, fmt.Errorf("handler panicked: %v", r))
				panic(r)
			}
		}()
		if a.handler.Stream != nil {
			for ev, e := range a.handler.Stream(ctx, req) {
				if e != nil {
					err = e
					break
				}
				switch ev.Type {
				case EventChunk:
					full.WriteString(ev.Text)
					if !emit(ev.Text) {
						return
					}
				case EventDone:
					if ev.Output != nil {
						full.Reset()
						full.WriteString(*ev.Output)
					}
					rawUsage = ev.Usage
				}
			}
		} else {
			var res HandlerResult
			res, err = a.handler.Invoke(ctx, req)
			if err == nil {
				full.WriteString(outputString(res.Output)) // streaming ignores outputFormat
				rawUsage = res.Usage
				if full.Len() > 0 && !emit(full.String()) {
					return
				}
			}
		}
		trackOutcome(a.userCtx, td, start, err)
		tracked = true
		if err != nil {
			yield(HandlerStreamEvent{}, err)
			return
		}
		usage := ParseUsage(rawUsage)
		trackTokens(a.userCtx, td, usage)
		*done = streamDone{response: full.String(), usage: usage, td: td, trajectory: rec.Render()}
	}
}
