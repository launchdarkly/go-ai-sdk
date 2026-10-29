package evaluations

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"maps"
	"math"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/judging"
	"github.com/launchdarkly/go-ai-sdk/internal/trajectory"
)

const (
	datasetPageSize = 200
	uploadBatchSize = 500

	statusComplete = "COMPLETE"
	statusError    = "ERROR"

	// generationErrorCode is the numeric code on a failed generation.
	generationErrorCode = 5001
)

// Criterion failure causes. They classify results for
// tests and logs; they are not wire values.
const (
	causeInvalidJudgeOutput = "invalid_judge_output"
	causeInvalidScore       = "invalid_score"
	causeHandlerRaised      = "handler_raised"
	causeScorerRaised       = "scorer_raised"
)

// runState is one Run: validated arguments, then what each step resolves.
type runState struct {
	m           *Module
	args        RunArgs
	pool        []*aiserver.ProviderHandler
	gen         GenerationConfig
	genHandler  *aiserver.ProviderHandler
	tools       []runTool
	criteria    []Criterion
	judges      map[string]*resolvedJudge
	concurrency int
	interval    time.Duration
	timeout     time.Duration
	client      aiserver.LDClient

	inline     bool
	rawRows    []rawRow
	rows       []row
	datasetKey string
	datasetID  string

	pinnedTools   map[string]int
	evaluationID  string
	evaluationKey string
	evalVersion   int
	runID         string

	// rowTraces holds each started row's trace by row position, so the
	// deferred cleanup can end any root a failed phase left open.
	rowTraces []*rowTrace
}

// row is a rendered dataset row. input/expected are nil when the
// hosted row held no string there.
type row struct {
	index     int
	input     *string
	expected  *string
	variables map[string]any
	metadata  map[string]any
}

type resolvedJudge struct {
	config       aiserver.AiConfig
	variationKey string
	version      *int
	handler      *aiserver.ProviderHandler
}

type genResult struct {
	row         row
	status      string
	output      any
	usage       map[string]any
	errMsg      string
	started     time.Time
	elapsed     time.Duration
	trajectory  string
	invocations []trajectory.Invocation
	trace       *rowTrace
}

type critResult struct {
	rowIndex  int
	criterion Criterion
	judge     *resolvedJudge
	status    string
	score     float64
	reason    string
	cause     string
	errMsg    string
	usage     map[string]any
	started   time.Time
	elapsed   time.Duration
	// captureOn is the judge handler's effective capture flag, which gates
	// the explanation on the evaluate span.
	captureOn bool
}

func deref(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}

func orNil(s *string) any {
	if s == nil {
		return nil
	}
	return *s
}

func (r *runState) project() string { return segment(r.args.ProjectKey) }

// --- validation before any I/O ------------------------------------

func (m *Module) prepare(a RunArgs) (*runState, error) {
	if strings.TrimSpace(a.ProjectKey) == "" {
		return nil, invalid("ProjectKey must not be blank")
	}
	if strings.TrimSpace(a.Key) == "" {
		return nil, invalid("Key must not be blank")
	}
	r := &runState{m: m, args: a, judges: map[string]*resolvedJudge{}}
	switch d := a.Dataset.(type) {
	case nil:
		return nil, invalid("dataset is required: pass a DatasetKey or InlineRows")
	case DatasetKey:
		if strings.TrimSpace(string(d)) == "" {
			return nil, invalid("dataset must not be blank")
		}
		r.datasetKey = string(d)
	case InlineRows:
		rows, err := normalizeInlineRows(d)
		if err != nil {
			return nil, err
		}
		r.inline, r.rawRows = true, rows
	default:
		return nil, invalid("dataset must be a dataset key or a sequence of inline rows, got %T", a.Dataset)
	}
	var err error
	if r.pool, err = validatePool(a.Handler, a.Handlers); err != nil {
		return nil, err
	}
	switch {
	case a.Generation == nil && a.AIConfig == nil:
		return nil, invalid("pass Generation, or AIConfig to evaluate an existing AI Config variation")
	case a.Generation != nil && a.AIConfig != nil:
		return nil, invalid("pass Generation or AIConfig, not both; override AI Config fields with Overrides")
	case a.Overrides != nil && a.AIConfig == nil:
		return nil, invalid("Overrides apply only with AIConfig")
	}
	if a.AIConfig != nil {
		if strings.TrimSpace(a.AIConfig.Key) == "" {
			return nil, invalid("AIConfig.Key must not be blank")
		}
		if strings.TrimSpace(a.AIConfig.Variation) == "" {
			return nil, invalid("AIConfig.Variation must not be blank")
		}
	} else if err := r.checkGeneration(*a.Generation); err != nil {
		return nil, err
	}
	if r.tools, err = validateTools(a.Tools, a.InlineTools); err != nil {
		return nil, err
	}
	switch {
	case a.Concurrency < 0:
		return nil, invalid("concurrency must be at least 1")
	case a.Concurrency == 0:
		r.concurrency = defaultConcurrency
	default:
		r.concurrency = a.Concurrency
	}
	if a.PollInterval < 0 {
		return nil, invalid("PollInterval must not be negative")
	}
	if a.PollTimeout < 0 {
		return nil, invalid("PollTimeout must not be negative")
	}
	r.interval, r.timeout = cmp.Or(a.PollInterval, defaultPollInterval), cmp.Or(a.PollTimeout, defaultPollTimeout)
	if a.Criteria != nil {
		if err := validateCriteria(a.Criteria); err != nil {
			return nil, err
		}
		r.criteria = a.Criteria
	}
	return r, nil
}

// checkGeneration runs the mode, coverage and prompt checks for settings
// written in code.
func (r *runState) checkGeneration(g GenerationConfig) error {
	if err := validateGenerationFields(&g); err != nil {
		return err
	}
	mode, err := validateGenerationMode(g.Mode)
	if err != nil {
		return err
	}
	h := selectHandler(r.pool, g.Provider, mode)
	if h == nil {
		return uncovered(g.Provider, mode, r.pool)
	}
	if p := promptProblem(mode, g.Instructions, g.Messages); p != "" {
		return invalid("generation prompt: %s %s", p, describe(h))
	}
	r.gen, r.genHandler = g, h
	return nil
}

// --- execution -----------------------------------------------------------

func (r *runState) execute(ctx context.Context) (*EvalRunResult, error) {
	if r.args.AIConfig != nil {
		if err := r.readAIConfig(ctx); err != nil {
			return nil, err
		}
	}
	if err := r.resolveTools(ctx); err != nil {
		return nil, err
	}
	if err := r.resolveJudges(ctx); err != nil {
		return nil, err
	}
	if r.inline {
		for _, raw := range r.rawRows {
			var expected any
			if raw.expected != nil {
				expected = *raw.expected
			}
			r.rows = append(r.rows, renderRow(raw.index, raw.input, expected, raw.variables, raw.metadata))
		}
	} else if err := r.fetchDataset(ctx); err != nil {
		return nil, err
	}
	if err := r.createEvaluation(ctx); err != nil {
		return nil, err
	}
	if err := r.createRun(ctx); err != nil {
		return nil, err
	}
	if r.inline {
		if err := r.upload(ctx); err != nil {
			return nil, err
		}
	}
	em := newEmitter(r.client, r.m.flushBound, r.m.logger, r.args.ProjectKey, r.evaluationID, r.runID,
		r.evaluationKey, r.evalVersion, r.datasetID, r.datasetKey)
	if err := r.runPhases(ctx, em); err != nil {
		return nil, err
	}
	summary, err := r.poll(ctx)
	if err != nil {
		return nil, err
	}
	return &EvalRunResult{
		// A failed row is one LaunchDarkly ruled against its criteria, so it
		// fails the gate as an error row does.
		Passed: summary.ErrorRows == 0 && summary.FailedRows == 0,
		URL: fmt.Sprintf("%s/projects/%s/ai/evaluations/%s/runs/%s", UIBaseURI, r.project(),
			segment(r.evaluationID), segment(r.runID)),
		RunID:   r.runID,
		Summary: summary,
	}, nil
}

// runPhases generates, emits, scores and emits. The final flush is
// deferred so queued generation events survive an evaluation-phase failure.
func (r *runState) runPhases(ctx context.Context, em *emitter) error {
	defer r.flushTelemetry(ctx, em)
	gens, err := r.generate(ctx)
	if err != nil {
		return err
	}
	for _, g := range gens {
		em.track(GenerationEventName, g.row.index, func() (map[string]any, error) {
			return em.payload(g.row.index, nil, generationFields(g))
		})
	}
	if len(r.criteria) == 0 {
		return nil
	}
	em.flush() // phase boundary
	results, err := r.evaluate(ctx, gens)
	if err != nil {
		return err
	}
	for _, c := range results {
		em.track(CriterionEventName, c.rowIndex, func() (map[string]any, error) {
			return em.payload(c.rowIndex, map[string]any{"criterionType": c.criterion.criterionType()}, criterionFields(c))
		})
	}
	return nil
}

// forEachBounded runs fn(0..n-1) on at most limit goroutines and waits for
// all of them. It stops dispatching when ctx is done.
func forEachBounded(ctx context.Context, n, limit int, fn func(i int)) error {
	jobs := make(chan int)
	var wg sync.WaitGroup
	for range min(limit, n) {
		wg.Go(func() {
			for i := range jobs {
				fn(i)
			}
		})
	}
	defer wg.Wait()
	defer close(jobs)
	for i := range n {
		select {
		case jobs <- i:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return ctx.Err()
}

// invoke calls a handler once and never retries it; a panic is a
// handler error, so one row cannot take down the run.
func invoke(ctx context.Context, h *aiserver.ProviderHandler, req aiserver.HandlerRequest) (res aiserver.HandlerResult, err error) {
	defer func() {
		if p := recover(); p != nil {
			err = fmt.Errorf("panic: %v", p)
		}
	}()
	return h.Invoke(ctx, req)
}

// --- generation phase ---------------------------------------------

func renderRow(index int, input, expected, variables, metadata any) row {
	vars := map[string]any{}
	if m, ok := variables.(map[string]any); ok {
		maps.Copy(vars, m)
	}
	out := row{index: index, variables: vars}
	if s, ok := input.(string); ok {
		v := aiserver.ParseTemplate(s, vars)
		out.input = &v
	}
	if s, ok := expected.(string); ok {
		v := aiserver.ParseTemplate(s, vars)
		out.expected = &v
	}
	vars["input"] = orNil(out.input)
	vars["expected_output"] = orNil(out.expected)
	out.metadata, _ = metadata.(map[string]any)
	return out
}

// handlerConfig is the one config every row receives; tools come from one
// resolved shape whatever their source.
func (r *runState) handlerConfig() aiserver.AiConfig {
	g := r.gen
	var cfg aiserver.AiConfig
	cfg.Provider.Name = g.Provider
	cfg.Model = aiserver.Model{Name: g.Model, Parameters: g.Parameters}
	if len(r.tools) > 0 {
		cfg.Tools = make(map[string]aiserver.Tool, len(r.tools))
		for _, t := range r.tools {
			cfg.Tools[t.key] = aiserver.Tool{Name: t.key, Type: "function", Description: t.description, Parameters: t.schema}
		}
	}
	snippets := map[string]any{}
	for k, v := range g.PromptSnippets {
		snippets[k] = v
	}
	vars := map[string]any{"snippet": snippets}
	switch {
	case g.Instructions != "":
		cfg.Instructions = aiserver.ParseTemplate(g.Instructions, vars)
	case len(g.Messages) > 0:
		cfg.Messages = slices.Clone(g.Messages)
		for i, msg := range cfg.Messages {
			if msg.Blocks == nil {
				cfg.Messages[i].Text = aiserver.ParseTemplate(msg.Text, vars)
			}
		}
	}
	cfg.OutputFormat = g.OutputFormat
	return cfg
}

// rowTools wraps each local tool with the row's own recorder, under its
// original key; native tools pass through unrecorded.
func (r *runState) rowTools(rec *trajectory.Recorder) map[string]aiserver.ToolHandler {
	out := make(map[string]aiserver.ToolHandler, len(r.tools))
	for _, t := range r.tools {
		if fn, ok := t.impl.(aiserver.ToolFunc); ok && trajectory.ShouldObserve(t.key, nil) {
			out[t.key] = aiserver.ToolFunc(rec.Observe(t.key, fn))
			continue
		}
		out[t.key] = t.impl
	}
	return out
}

func (r *runState) generate(ctx context.Context) ([]*genResult, error) {
	cfg := r.handlerConfig()
	results := make([]*genResult, len(r.rows))
	r.rowTraces = make([]*rowTrace, len(r.rows))
	err := forEachBounded(ctx, len(r.rows), r.concurrency, func(i int) {
		row := r.rows[i]
		rt := startRow(ctx, r.runID, row.index)
		r.rowTraces[i] = rt
		rec := trajectory.NewRecorder()
		call := beginCall(rt.ctx, r.genHandler, rt)
		started := time.Now()
		res, err := invoke(call.ctx, r.genHandler, aiserver.HandlerRequest{
			Config: cfg, UserInput: deref(row.input), ToolHandlers: r.rowTools(rec), Variables: maps.Clone(row.variables),
		})
		ended := time.Now()
		g := &genResult{row: row, started: started, elapsed: ended.Sub(started),
			trajectory: rec.Render(), invocations: rec.Invocations(), trace: rt}
		if call.shouldDescribe() {
			call.writeChat(chatCall{config: cfg, variables: row.variables, finalUser: row.input,
				toolCalls: g.invocations, output: res.Output, usage: res.Usage, err: err, start: started, end: ended}, rt.attrs)
		}
		rt.finishGeneration(err, len(r.criteria))
		if err != nil {
			g.status, g.errMsg = statusError, "Unknown error"
			if msg := err.Error(); msg != "" {
				g.errMsg = "handler raised: " + msg
			}
		} else {
			g.status, g.output, g.usage = statusComplete, res.Output, res.Usage
		}
		results[i] = g
	})
	if err != nil {
		return nil, fmt.Errorf("evaluations: generation phase: %w", err)
	}
	return results, nil
}

func generationFields(g *genResult) map[string]any {
	f := map[string]any{
		"status":      g.status,
		"startedAt":   timestamp(g.started),
		"generatedAt": timestamp(g.started.Add(g.elapsed)),
		"latencyMs":   g.elapsed.Milliseconds(),
		"output":      g.output,
	}
	if g.status == statusError {
		f["error"] = map[string]any{"code": generationErrorCode, "message": g.errMsg}
		f["errorMessage"] = g.errMsg
	}
	if g.usage != nil {
		f["usage"] = usageWire(g.usage)
	}
	return f
}

// --- evaluation phase ---------------------------------------------

func (r *runState) resolveJudges(ctx context.Context) error {
	lctx := ldcontext.NewBuilder(r.args.ProjectKey).Kind("evaluation").Build()
	var problems []string
	for _, c := range r.criteria {
		j, ok := c.(*Judge)
		if !ok {
			continue
		}
		raw, err := resolveJudgeConfig(ctx, j.key, lctx)
		if err == nil && raw == nil {
			err = errors.New("the variation is not an object")
		}
		var cfg aiserver.AiConfig
		if err == nil {
			cfg, err = aiserver.ParseAiConfig(raw)
		}
		if err != nil {
			return fmt.Errorf("evaluations: %w: failed to resolve LaunchDarkly judge %q: %w; if the judge does not exist in project %q, create it in the LaunchDarkly UI and try again",
				ErrJudgeResolution, j.key, err, r.args.ProjectKey)
		}
		meta, _ := raw["_ldMeta"].(map[string]any)
		rj := &resolvedJudge{}
		rj.variationKey, _ = meta["variationKey"].(string)
		if v, ok := asInt(meta["version"]); ok {
			rj.version = &v
		}
		modeStr, _ := meta["mode"].(string)
		mode := aiserver.NormalizeMode(modeStr)
		// Judges cannot use tools, even when their AI Config defines some.
		cfg.Tools = nil
		cfg.Raw = maps.Clone(cfg.Raw)
		delete(cfg.Raw, "tools")
		rj.config = cfg
		rj.handler = selectHandler(r.pool, cfg.Provider.Name, mode)
		switch {
		case rj.handler == nil:
			problems = append(problems, fmt.Sprintf("judge %q (provider %q, %s mode) has no handler", j.key, cfg.Provider.Name, mode))
		default:
			if p := promptProblem(mode, cfg.Instructions, cfg.Messages); p != "" {
				problems = append(problems, fmt.Sprintf("judge %q: %s %s", j.key, p, describe(rj.handler)))
			}
		}
		r.judges[j.key] = rj
	}
	if len(problems) > 0 {
		return fmt.Errorf("evaluations: %w: %s; the pool provides for: %s",
			ErrJudgeResolution, strings.Join(problems, "; "), describePool(r.pool))
	}
	return nil
}

func (r *runState) evaluate(ctx context.Context, gens []*genResult) ([]*critResult, error) {
	type pair struct {
		g *genResult
		c Criterion
	}
	var pairs []pair
	for _, g := range gens {
		if g.status != statusComplete {
			continue // a row that did not generate is never scored
		}
		for _, c := range r.criteria {
			pairs = append(pairs, pair{g, c})
		}
	}
	results := make([]*critResult, len(pairs))
	err := forEachBounded(ctx, len(pairs), r.concurrency, func(i int) {
		rt := pairs[i].g.trace
		switch c := pairs[i].c.(type) {
		case *Judge:
			ectx, span := startEvaluate(rt, c, r.judges[c.key])
			results[i] = r.runJudge(ectx, pairs[i].g, c)
			finishEvaluate(span, results[i])
		case *Scorer:
			ectx, span := startEvaluate(rt, c, nil)
			results[i] = runScorer(ectx, pairs[i].g, c)
			finishEvaluate(span, results[i])
		}
		rt.criterionDone()
	})
	if err != nil {
		return nil, fmt.Errorf("evaluations: evaluation phase: %w", err)
	}
	return results, nil
}

func (c *critResult) fail(cause, msg string) *critResult {
	c.elapsed = time.Since(c.started)
	c.status, c.cause, c.errMsg = statusError, cause, msg
	return c
}

func (r *runState) runJudge(ctx context.Context, g *genResult, j *Judge) *critResult {
	rj := r.judges[j.key]
	res := &critResult{rowIndex: g.row.index, criterion: j, judge: rj, started: time.Now()}
	output := aiserver.ToolResultString(g.output)
	input := deref(g.row.input)
	groundTruth := deref(g.row.expected)
	if j.groundTruthContext != "" {
		groundTruth = aiserver.ParseTemplate(j.groundTruthContext, g.row.variables)
	}
	vars := maps.Clone(g.row.variables)
	// Absent values are "", never nil: a nil leaves literal mustache text.
	vars["input"] = input
	vars["response_to_evaluate"] = output
	vars["expected_output"] = deref(g.row.expected)
	vars["ground_truth_context"] = groundTruth
	vars["message_history"] = judging.BuildMessageHistory(input, g.trajectory, output)
	vars["formatting_instructions"] = judging.FormattingInstructions
	call := beginCall(ctx, rj.handler, g.trace)
	res.captureOn = aiserver.CaptureContentEnabled(call.ctx)
	started := time.Now()
	hr, err := invoke(call.ctx, rj.handler, aiserver.HandlerRequest{
		Config: rj.config, UserInput: output, ToolHandlers: map[string]aiserver.ToolHandler{}, Variables: vars,
	})
	if call.shouldDescribe() {
		call.writeChat(chatCall{config: rj.config, variables: vars, finalUser: &output,
			output: hr.Output, err: err, start: started, end: time.Now()}, g.spanAttrs())
	}
	if err != nil {
		return res.fail(causeHandlerRaised, err.Error()) // errorMessage equals the error message
	}
	raw, reasoning, err := judging.ParseVerdict(hr.Output)
	if err != nil {
		return res.fail(causeInvalidJudgeOutput, err.Error())
	}
	if reasoning == "" {
		return res.fail(causeInvalidJudgeOutput, "judge response has no reasoning")
	}
	score, ok := judgeScore(raw)
	if !ok || score < 0 || score > 1 {
		return res.fail(causeInvalidScore, fmt.Sprintf("judge score must be a finite number between 0 and 1, got %#v", raw))
	}
	res.elapsed = time.Since(res.started)
	res.status, res.score, res.reason, res.usage = statusComplete, score, reasoning, hr.Usage
	return res
}

// judgeScore accepts a finite JSON number, or a Go integer from a handler
// that returned a literal map; bool and strings are not scores.
func judgeScore(v any) (float64, bool) {
	if f, ok := judging.NumericScore(v); ok {
		return f, true
	}
	if _, isFloat := v.(float64); isFloat {
		return 0, false
	}
	n, ok := intValue(v)
	return float64(n), ok
}

func runScorer(ctx context.Context, g *genResult, s *Scorer) *critResult {
	res := &critResult{rowIndex: g.row.index, criterion: s, started: time.Now()}
	public := DatasetRow{RowIndex: g.row.index, Input: deref(g.row.input), ExpectedOutput: deref(g.row.expected),
		Variables: maps.Clone(g.row.variables), Metadata: maps.Clone(g.row.metadata)}
	score, err := callScorer(ctx, s.fn, public, g.output)
	if err != nil {
		return res.fail(causeScorerRaised, err.Error())
	}
	if math.IsNaN(score) || math.IsInf(score, 0) || score < 0 || score > 1 {
		return res.fail(causeInvalidScore, fmt.Sprintf("scorer %q must return a finite number between 0 and 1, got %v", s.name, score))
	}
	res.elapsed = time.Since(res.started)
	res.status, res.score = statusComplete, score
	return res
}

func callScorer(ctx context.Context, fn ScorerFunc, row DatasetRow, output any) (score float64, err error) {
	defer func() {
		if p := recover(); p != nil {
			err = fmt.Errorf("panic: %v", p)
		}
	}()
	return fn(ctx, row, output)
}

func criterionFields(c *critResult) map[string]any {
	f := map[string]any{
		"kind":        "scorer",
		"status":      c.status,
		"startedAt":   timestamp(c.started),
		"evaluatedAt": timestamp(c.started.Add(c.elapsed)),
		"latencyMs":   c.elapsed.Milliseconds(),
	}
	if c.status == statusComplete {
		f["score"] = c.score
	} else {
		f["errorMessage"] = cmp.Or(c.errMsg, "Unknown error")
	}
	if c.judge != nil {
		f["kind"] = "judge"
		f["judgeKey"] = c.criterion.criterionType()
		f["variationKey"] = c.judge.variationKey
		if c.judge.version != nil {
			f["version"] = *c.judge.version
		}
		if c.status == statusComplete {
			f["reason"] = c.reason
		}
		if c.usage != nil {
			f["usage"] = usageWire(c.usage)
		}
	}
	return f
}
