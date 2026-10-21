package aiserver

import (
	"context"
	"math/rand/v2"
	"strings"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"

	"github.com/launchdarkly/go-ai-sdk/internal/judging"
)

// FormattingInstructions is appended to every judge message_history.
const FormattingInstructions = judging.FormattingInstructions

// sampled is a seam so tests can make sampling deterministic.
var sampled = func(rate float64) bool { return rand.Float64() < rate }

type judgeArgs struct {
	cfg        AiConfig
	userCtx    ldcontext.Context
	handler    *ProviderHandler
	handlers   []*ProviderHandler
	userInput  string
	output     string
	parentTD   TrackData
	graphKey   string
	trajectory string
}

func collapseMessages(cfg AiConfig) AiConfig {
	if len(cfg.Messages) == 0 || cfg.Instructions != "" {
		return cfg
	}
	texts := make([]string, len(cfg.Messages))
	for i, m := range cfg.Messages {
		texts[i] = ContentText(m)
	}
	cfg.Instructions = strings.Join(texts, "\n\n")
	cfg.Messages = nil
	return cfg
}

func providerMatches(p ProvidesFor, provider string) bool {
	return p.Provider == provider || p.Provider == WildcardProvider
}

// pickJudgeHandler applies the judge handler priority: exact provider+mode
// (or wildcard), then an agent handler for a messages judge (collapsing its
// messages), then the parent handler when it covers the provider. ok=false
// skips the judge rather than send a model to the wrong provider's API.
func pickJudgeHandler(parent *ProviderHandler, handlers []*ProviderHandler, provider string, mode Mode) (h *ProviderHandler, collapse, ok bool) {
	if handlers == nil {
		if parent == nil || !providerMatches(parent.ProvidesFor, provider) {
			return nil, false, false
		}
		return parent, mode == ModeMessages && handlerMode(parent) == ModeAgent, true
	}
	for _, c := range handlers {
		if providerMatches(c.ProvidesFor, provider) && handlerMode(c) == mode {
			return c, false, true
		}
	}
	if mode == ModeMessages {
		for _, c := range handlers {
			if providerMatches(c.ProvidesFor, provider) && handlerMode(c) == ModeAgent {
				return c, true, true
			}
		}
	}
	if parent != nil && providerMatches(parent.ProvidesFor, provider) {
		return parent, mode == ModeMessages && handlerMode(parent) == ModeAgent, true
	}
	return nil, false, false
}

func activeJudges(cfg AiConfig) []JudgeRef {
	if cfg.JudgeConfiguration == nil {
		return nil
	}
	var out []JudgeRef
	for _, j := range cfg.JudgeConfiguration.Judges {
		if j.SamplingRate > 0 && sampled(j.SamplingRate) {
			out = append(out, j)
		}
	}
	return out
}

// resolveJudge fetches a judge's config. A judge runs after the response is
// already produced and paid for, so a judge that cannot be resolved (most
// often one toggled off in LaunchDarkly) is logged and skipped rather than
// failing the call, as in the JavaScript SDK.
func resolveJudge(ctx context.Context, key string, userCtx ldcontext.Context) (AiConfig, VariationMeta, bool) {
	cfg, meta, err := extractVariation(ctx, key, userCtx)
	if err != nil {
		sdkLogger().Error("judge skipped: its config could not be resolved", "judge", key, "error", err)
		return AiConfig{}, VariationMeta{}, false
	}
	return cfg, meta, true
}

// runJudges executes the configured judges inline. Judges never get
// the caller's tools. A provider failure is returned; an unparseable verdict
// is logged and that judge is skipped.
func runJudges(ctx context.Context, a judgeArgs) (map[string]JudgeResult, error) {
	results := map[string]JudgeResult{}
	for _, j := range activeJudges(a.cfg) {
		jcfg, jmeta, ok := resolveJudge(ctx, j.Key, a.userCtx)
		if !ok {
			continue
		}
		h, collapse, ok := pickJudgeHandler(a.handler, a.handlers, jcfg.Provider.Name, NormalizeMode(jmeta.Mode))
		if !ok {
			continue
		}
		effective := jcfg
		if collapse {
			effective = collapseMessages(jcfg)
		}
		history := judging.BuildMessageHistory(a.userInput, a.trajectory, a.output)
		err := withJudgeEvaluation(ctx, j.Key, func(ctx context.Context, record func(float64, string)) error {
			res, err := executeAndTrack(ctx, execArgs{
				configKey: j.Key, config: effective, meta: jmeta, userCtx: a.userCtx, handler: h,
				userInput: a.output, graphKey: a.graphKey,
				variables: map[string]any{"message_history": history, "response_to_evaluate": a.output},
			})
			if err != nil {
				return err
			}
			rawScore, reasoning, perr := judging.ParseVerdict(res.Output)
			score, numeric := judging.NumericScore(rawScore)
			if perr != nil || !numeric {
				sdkLogger().Error("judge returned invalid output", "judge", j.Key)
				return nil
			}
			results[j.Key] = JudgeResult{Score: score, Response: reasoning, Usage: res.Usage}
			explanation := ""
			if h.CaptureContent {
				explanation = reasoning
			}
			record(score, explanation)
			if jcfg.EvaluationMetricKey != "" {
				td := a.parentTD
				td.JudgeConfigKey = j.Key
				track(jcfg.EvaluationMetricKey, a.userCtx, td, score)
			}
			return nil
		})
		if err != nil {
			return results, err
		}
	}
	return results, nil
}

// BuildJudgeTasksArgs are the inputs to BuildJudgeTasks: the parent
// invocation's config, context, handlers, input, output and track data.
type BuildJudgeTasksArgs struct {
	Config      AiConfig
	UserContext ldcontext.Context
	Handler     *ProviderHandler
	Handlers    []*ProviderHandler
	UserInput   string
	Output      string
	TrackData   TrackData
	Trajectory  string
}

// BuildJudgeTasks resolves the configured judges into serialisable tasks
// without running them (prepareAsyncJudge in JS and Python). Sampling applies.
func BuildJudgeTasks(ctx context.Context, args BuildJudgeTasksArgs) ([]JudgeTask, error) {
	reportUsage("client.buildJudgeTasks", "", "")
	return buildJudgeTasks(ctx, judgeArgs{
		cfg: args.Config, userCtx: args.UserContext, handler: args.Handler, handlers: args.Handlers,
		userInput: args.UserInput, output: args.Output, parentTD: args.TrackData, trajectory: args.Trajectory,
	})
}

func buildJudgeTasks(ctx context.Context, a judgeArgs) ([]JudgeTask, error) {
	tasks := []JudgeTask{}
	for _, j := range activeJudges(a.cfg) {
		jcfg, jmeta, ok := resolveJudge(ctx, j.Key, a.userCtx)
		if !ok {
			continue
		}
		mode := NormalizeMode(jmeta.Mode)
		_, collapse, ok := pickJudgeHandler(a.handler, a.handlers, jcfg.Provider.Name, mode)
		if !ok {
			continue
		}
		tasks = append(tasks, JudgeTask{
			ConfigKey: j.Key, JudgeConfig: jcfg, JudgeMeta: jmeta, ActualOutput: a.output,
			UserInput: a.userInput, Trajectory: a.trajectory, UserContext: a.userCtx,
			JudgeProvider: jcfg.Provider.Name, JudgeMode: mode, CollapseMessages: collapse,
			EvaluationMetricKey: jcfg.EvaluationMetricKey, ParentTrackData: a.parentTD,
		})
	}
	return tasks, nil
}

// JudgeRunResult is a deferred judge's verdict. TrackData has
// judgeConfigKey merged in, ready for Track on the caller's goroutine.
type JudgeRunResult struct {
	Score     float64
	Response  string
	Usage     TokenUsage
	TrackData TrackData
}

// RunJudge executes a deferred judge task with an explicit handler list and
// no global registry. It returns (nil, nil) when no handler fits or the
// verdict is not a finite numeric score, and an error for provider failures.
func RunJudge(ctx context.Context, task JudgeTask, handlers []*ProviderHandler) (*JudgeRunResult, error) {
	reportUsage("client.runJudge", "", "")
	var h *ProviderHandler
	for _, c := range handlers {
		if providerMatches(c.ProvidesFor, task.JudgeProvider) && handlerMode(c) == task.JudgeMode {
			h = c
			break
		}
	}
	if h == nil && task.JudgeMode == ModeMessages {
		for _, c := range handlers {
			if providerMatches(c.ProvidesFor, task.JudgeProvider) && handlerMode(c) == ModeAgent {
				h = c
				break
			}
		}
	}
	if h == nil {
		return nil, nil
	}
	cfg := task.JudgeConfig
	if task.CollapseMessages {
		cfg = collapseMessages(cfg)
	}
	vars := map[string]any{}
	vars["message_history"] = judging.BuildMessageHistory(task.UserInput, task.Trajectory, task.ActualOutput)
	vars["response_to_evaluate"] = task.ActualOutput

	var out *JudgeRunResult
	err := withJudgeEvaluation(ctx, task.ConfigKey, func(ctx context.Context, record func(float64, string)) error {
		res, err := executeAndTrack(ctx, execArgs{
			configKey: task.ConfigKey, config: cfg, meta: task.JudgeMeta, userCtx: task.UserContext,
			handler: h, userInput: task.ActualOutput, variables: vars,
		})
		if err != nil {
			return err
		}
		rawScore, reasoning, perr := judging.ParseVerdict(res.Output)
		score, numeric := judging.NumericScore(rawScore)
		if perr != nil || !numeric {
			return nil
		}
		explanation := ""
		if h.CaptureContent {
			explanation = reasoning
		}
		record(score, explanation)
		// The score belongs to the evaluated (parent) config, exactly as in the inline path.
		td := task.ParentTrackData
		td.JudgeConfigKey = task.ConfigKey
		out = &JudgeRunResult{Score: score, Response: reasoning, Usage: res.Usage, TrackData: td}
		return nil
	})
	return out, err
}
