package evaluations

import (
	"context"
	"errors"
	"math"
	"reflect"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/judging"
)

func TestNoCriteriaSkipsThePhase(t *testing.T) {
	e := newEnv(t, 2)
	mustRun(t, e.module(), baseArgs(handler("OpenAI", aiserver.ModeMessages, nil)))
	if len(e.ld.events(CriterionEventName)) != 0 || len(e.judgeCalls) != 0 {
		t.Fatal("criterion phase ran without criteria")
	}
	assertSequence(t, e, "GET projects/proj/datasets/ds", rowsPath, evalPath, runPath, summaryPath)
	if _, has := e.body("POST", "projects/proj/evaluations").(map[string]any)["criteria"]; has {
		t.Fatal("criteria must be absent from the create body")
	}
}

// TestSDKNeverComputesAVerdict fixes the score and moves the threshold to
// both sides of it, with the judge inverted, not inverted and unset.
func TestSDKNeverComputesAVerdict(t *testing.T) {
	for _, threshold := range []float64{0.2, 0.9} {
		for _, inverted := range []any{true, false, nil} {
			e := newEnv(t, 1)
			extra := map[string]any{}
			if inverted != nil {
				extra["isInverted"] = inverted
			}
			e.judges["acc"] = judgeVariation("OpenAI", "completion", extra)
			args := baseArgs(handler("OpenAI", aiserver.ModeMessages, verdict(0.6, "fine")))
			args.Criteria = []Criterion{mustJudge(t, "acc", &JudgeOptions{Threshold: ptr(threshold)}),
				mustScorer(t, "sc", constScorer(0.6), &ScorerOptions{Threshold: ptr(threshold)})}
			mustRun(t, e.module(), args)
			for _, ev := range e.ld.tracks {
				if ev.Event == CriterionEventName && ev.Data["score"] != 0.6 {
					t.Fatalf("score %v", ev.Data["score"])
				}
				for _, k := range []string{"verdict", "passed", "successDirection"} {
					if _, has := ev.Data[k]; has {
						t.Fatalf("%s on %s", k, ev.Event)
					}
				}
			}
		}
	}
}

func TestJudgeResolution(t *testing.T) {
	e := newEnv(t, 3)
	e.route("GET", "projects/proj/ai-tools/lib", resp{body: map[string]any{"version": 1}})
	e.judges["acc"] = judgeVariation("OpenAI", "completion", map[string]any{"_ldMeta": map[string]any{"enabled": true}})
	args := baseArgs(handler("OpenAI", aiserver.ModeMessages, verdict(1, "great")))
	args.Tools = map[string]aiserver.ToolHandler{"lib": noopTool}
	args.Criteria = []Criterion{mustJudge(t, "acc", nil)}
	mustRun(t, e.module(), args)
	if len(e.judgeCalls) != 1 {
		t.Fatalf("resolved %d times; must be once per run", len(e.judgeCalls))
	}
	if c := e.judgeCalls[0].Ctx; c.Kind() != "evaluation" || c.Key() != "proj" {
		t.Fatalf("resolution context %v", c)
	}
	log := e.logSnapshot()
	if !(slices.Index(log, "GET projects/proj/ai-tools/lib") < slices.Index(log, "judge acc") &&
		slices.Index(log, "judge acc") < slices.Index(log, "GET projects/proj/datasets/ds")) {
		t.Fatalf("resolution must sit between tool verification and dataset resolution: %v", log)
	}
	ev := e.ld.events(CriterionEventName)[0].Data
	if ev["variationKey"] != "" {
		t.Fatalf("variationKey %v", ev["variationKey"])
	}
	if _, has := ev["version"]; has {
		t.Fatal("unknown judge version must be omitted")
	}
}

func TestJudgeResolutionFailures(t *testing.T) {
	disabled := errors.New("variation accuracy is not enabled")
	cases := []struct {
		name  string
		serve any // nil: unknown flag; struct{}: the seam returns no object
		cause string
	}{
		{"unknown judge", nil, `flag "accuracy" not found`},
		{"disabled judge", disabled, "is not enabled"},
		{"not an object", struct{}{}, "not an object"},
		{"invalid config", map[string]any{"model": "x"}, "model.name"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			e := newEnv(t, 1)
			if tc.serve != nil {
				e.judges["accuracy"] = tc.serve
			}
			args := baseArgs(handler("OpenAI", aiserver.ModeMessages, nil))
			args.Criteria = []Criterion{mustJudge(t, "accuracy", nil)}
			err := runErr(t, e.module(), args, ErrJudgeResolution, `"accuracy"`, "create it", tc.cause)
			if tc.serve == disabled && !errors.Is(err, disabled) {
				t.Fatal("must wrap the underlying cause")
			}
			e.assertNoRequests()
		})
	}
}

func TestPinnedCriteria(t *testing.T) {
	e := newEnv(t, 1)
	e.judges["acc"] = judgeVariation("OpenAI", "completion", nil)
	args := baseArgs(handler("OpenAI", aiserver.ModeMessages, verdict(1, "ok")))
	args.Criteria = []Criterion{mustJudge(t, "acc", nil),
		mustScorer(t, "regex_hits", constScorer(0), &ScorerOptions{Threshold: ptr(0.2), SuccessDirection: LowerIsBetter})}
	mustRun(t, e.module(), args)
	want := []any{
		map[string]any{"criterionType": "acc", "kind": "judge", "judgeKey": "acc", "options": map[string]any{"threshold": 0.5, "passRateThreshold": 0.95}},
		map[string]any{"criterionType": "regex_hits", "kind": "scorer", "successDirection": "lower_is_better", "options": map[string]any{"threshold": 0.2, "passRateThreshold": 0.95}},
	}
	if got := e.body("POST", "projects/proj/evaluations").(map[string]any)["criteria"]; !reflect.DeepEqual(got, want) {
		t.Fatalf("criteria\n got %s\nwant %s", toJSON(got), toJSON(want))
	}
	t.Run("ground truth context is pinned", func(t *testing.T) {
		j := mustJudge(t, "g", &JudgeOptions{GroundTruthContext: "ctx"})
		if j.wire()["options"].(map[string]any)["groundTruthContext"] != "ctx" {
			t.Fatal(j.wire())
		}
	})
}

func TestJudgeHandlerSelection(t *testing.T) {
	cases := []struct {
		name               string
		pool               []*aiserver.ProviderHandler
		genProvider, genMd string
		judgeProv, judgeMd string
		wantGen, wantJudge int // pool indexes
	}{
		{"mixed providers", []*aiserver.ProviderHandler{handler("OpenAI", "messages", nil), handler("Anthropic", "messages", nil)}, "OpenAI", ModeCompletion, "Anthropic", "completion", 0, 1},
		{"mixed modes", []*aiserver.ProviderHandler{handler("OpenAI", "agent", nil), handler("OpenAI", "messages", nil)}, "OpenAI", ModeAgent, "OpenAI", "completion", 0, 1},
		{"one handler for both phases", []*aiserver.ProviderHandler{handler("OpenAI", "messages", nil)}, "OpenAI", ModeCompletion, "OpenAI", "judge", 0, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			e := newEnv(t, 2)
			e.judges["acc"] = judgeVariation(tc.judgeProv, tc.judgeMd, nil)
			gens, judges := make([]atomic.Int32, len(tc.pool)), make([]atomic.Int32, len(tc.pool))
			for i, h := range tc.pool {
				h.Invoke = func(_ context.Context, req aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
					if _, isJudge := req.Variables["message_history"]; isJudge {
						judges[i].Add(1)
						return verdict(1, "ok")(nil, req)
					}
					gens[i].Add(1)
					return aiserver.HandlerResult{Output: "o"}, nil
				}
			}
			args := baseArgs(nil)
			args.Handler, args.Handlers = nil, tc.pool
			args.Generation = &GenerationConfig{Provider: tc.genProvider, Model: "m", Mode: tc.genMd}
			if tc.genMd == ModeAgent {
				args.Generation.Instructions = "act"
			}
			args.Criteria = []Criterion{mustJudge(t, "acc", nil)}
			mustRun(t, e.module(), args)
			for i := range tc.pool {
				wantG, wantJ := int32(0), int32(0)
				if i == tc.wantGen {
					wantG = 2
				}
				if i == tc.wantJudge {
					wantJ = 2
				}
				if gens[i].Load() != wantG || judges[i].Load() != wantJ {
					t.Fatalf("handler %d: %d generations, %d judgings", i, gens[i].Load(), judges[i].Load())
				}
			}
		})
	}
}

func TestJudgeCoverageAndPromptFailFast(t *testing.T) {
	t.Run("no agent fallback", func(t *testing.T) {
		e := newEnv(t, 1)
		e.judges["acc"] = judgeVariation("OpenAI", "completion", nil)
		var rec recorder
		args := baseArgs(handler("OpenAI", aiserver.ModeAgent, rec.wrap(nil)))
		args.Generation = &GenerationConfig{Provider: "OpenAI", Model: "m", Mode: ModeAgent}
		args.Criteria = []Criterion{mustJudge(t, "acc", nil)}
		runErr(t, e.module(), args, ErrJudgeResolution, `"acc"`, "[OpenAI, agent]")
		if rec.n() != 0 || e.count("POST") != 0 {
			t.Fatal("nothing may run or be created")
		}
	})
	t.Run("every problem in one error", func(t *testing.T) {
		e := newEnv(t, 1)
		e.judges["a"] = judgeVariation("Anthropic", "completion", nil)
		e.judges["b"] = judgeVariation("Bedrock", "agent", nil)
		e.judges["c"] = judgeVariation("OpenAI", "completion", map[string]any{"messages": nil, "instructions": "rubric"})
		args := baseArgs(handler("OpenAI", aiserver.ModeMessages, nil))
		args.Criteria = []Criterion{mustJudge(t, "a", nil), mustJudge(t, "b", nil), mustJudge(t, "c", nil)}
		runErr(t, e.module(), args, ErrJudgeResolution,
			`judge "a" (provider "Anthropic", messages mode)`, `judge "b" (provider "Bedrock", agent mode)`,
			`judge "c": the prompt needs messages`, "the pool provides for: [OpenAI, messages]")
		if e.count("POST") != 0 {
			t.Fatal("a record was created")
		}
	})
	t.Run("agent judge with messages", func(t *testing.T) {
		e := newEnv(t, 1)
		e.judges["a"] = judgeVariation("OpenAI", "agent", nil)
		args := baseArgs(handler("OpenAI", aiserver.ModeMessages, nil))
		args.Handler, args.Handlers = nil, []*aiserver.ProviderHandler{args.Handler, handler("OpenAI", aiserver.ModeAgent, nil)}
		args.Criteria = []Criterion{mustJudge(t, "a", nil)}
		runErr(t, e.module(), args, ErrJudgeResolution, `judge "a": the prompt needs instructions`)
	})
}

// judgeEnv runs one judge "acc" over rows with a pool of one messages
// handler; judge receives every judge call.
func judgeEnv(t *testing.T, rows int, judge func(context.Context, aiserver.HandlerRequest) (aiserver.HandlerResult, error), extra map[string]any) (*fakeEnv, RunArgs, *recorder) {
	e := newEnv(t, rows)
	e.judges["acc"] = judgeVariation("OpenAI", "completion", extra)
	rec := &recorder{}
	h := handler("OpenAI", aiserver.ModeMessages, func(ctx context.Context, req aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
		if _, isJudge := req.Variables["message_history"]; isJudge {
			return rec.wrap(judge)(ctx, req)
		}
		return aiserver.HandlerResult{Output: "out:" + req.UserInput}, nil
	})
	args := baseArgs(h)
	args.Criteria = []Criterion{mustJudge(t, "acc", nil)}
	return e, args, rec
}

func TestJudgesGetNoTools(t *testing.T) {
	var genTools map[string]aiserver.ToolHandler
	e, args, rec := judgeEnv(t, 1, verdict(1, "ok"), map[string]any{"tools": map[string]any{
		"t": map[string]any{"name": "t", "type": "function", "parameters": map[string]any{}}}})
	e.route("GET", "projects/proj/ai-tools/lib", resp{body: map[string]any{"version": 1}})
	inner := args.Handler.Invoke
	args.Handler.Invoke = func(ctx context.Context, req aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
		if _, isJudge := req.Variables["message_history"]; !isJudge {
			genTools = req.ToolHandlers
		}
		return inner(ctx, req)
	}
	args.Tools = map[string]aiserver.ToolHandler{"lib": noopTool}
	args.InlineTools = map[string]InlineTool{"inl": {Implementation: noopTool, Schema: map[string]any{}}}
	mustRun(t, e.module(), args)
	if len(genTools) != 2 {
		t.Fatalf("generation tools %v", genTools)
	}
	jc := rec.all()[0]
	if jc.ToolHandlers == nil || len(jc.ToolHandlers) != 0 || jc.Config.Tools != nil {
		t.Fatalf("judge tools %v / %v", jc.ToolHandlers, jc.Config.Tools)
	}
	if _, has := jc.Config.Raw["tools"]; has {
		t.Fatal("judge Raw still carries tools")
	}
}

func TestJudgeConfigUnrenderedAndVariables(t *testing.T) {
	var rendered string
	judge := func(_ context.Context, req aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
		// The handler owns the one template pass.
		rendered = aiserver.ParseTemplate(aiserver.ContentText(req.Config.Messages[0]), req.Variables)
		return verdict(0.5, "ok")(nil, req)
	}
	e, args, rec := judgeEnv(t, 1, judge, nil)
	e.route("GET", "projects/proj/datasets/ds/rows", resp{body: map[string]any{"totalCount": 1, "items": []any{
		map[string]any{"rowIndex": 0, "input": "Q {{x}}", "variables": map[string]any{"x": "1"}}}}})
	inner := args.Handler.Invoke
	args.Handler.Invoke = func(ctx context.Context, req aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
		if _, isJudge := req.Variables["message_history"]; !isJudge {
			return aiserver.HandlerResult{Output: "says {{expected_output}}"}, nil
		}
		return inner(ctx, req)
	}
	mustRun(t, e.module(), args)
	jc := rec.all()[0]
	if aiserver.ContentText(jc.Config.Messages[0]) != "Grade {{message_history}}" {
		t.Fatal("the harness rendered the judge config")
	}
	if jc.UserInput != "says {{expected_output}}" || !strings.Contains(rendered, "says {{expected_output}}") {
		t.Fatalf("generated text was expanded: %q", rendered)
	}
	var keys []string
	for k := range jc.Variables {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	want := []string{"expected_output", "formatting_instructions", "ground_truth_context", "input", "message_history", "response_to_evaluate", "x"}
	if !reflect.DeepEqual(keys, want) {
		t.Fatalf("judge variables %v", keys)
	}
	v := jc.Variables
	if v["expected_output"] != "" || v["ground_truth_context"] != "" || v["input"] != "Q 1" || v["response_to_evaluate"] != "says {{expected_output}}" ||
		v["formatting_instructions"] != aiserver.FormattingInstructions {
		t.Fatalf("judge variables %v", v)
	}
	if v["message_history"] != judging.BuildMessageHistory("Q 1", "", "says {{expected_output}}") ||
		!strings.Contains(v["message_history"].(string), judging.FormattingInstructions) {
		t.Fatalf("message_history %q", v["message_history"])
	}
}

// a judge built from the AI Library's default templates references
// only {{message_history}}; the formatting instructions ride inside it, so
// the judge scores a row without edits.
func TestDefaultTemplateJudgeScoresWithoutEdits(t *testing.T) {
	var rendered string
	judge := func(_ context.Context, req aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
		rendered = aiserver.ParseTemplate(aiserver.ContentText(req.Config.Messages[0]), req.Variables)
		return verdict(0.8, "grounded")(nil, req)
	}
	e, args, rec := judgeEnv(t, 1, judge, map[string]any{
		"messages": []any{map[string]any{"role": "system", "content": "Rate the relevance of the response.\n\n{{message_history}}"}}})
	mustRun(t, e.module(), args)
	if rec.n() != 1 {
		t.Fatalf("judge invoked %d times", rec.n())
	}
	if !strings.Contains(rendered, judging.FormattingInstructions) || strings.Contains(rendered, "{{") {
		t.Fatalf("the rendered rubric lacks the formatting instructions or left a placeholder: %q", rendered)
	}
	ev := e.ld.events(CriterionEventName)[0].Data
	if ev["status"] != "COMPLETE" || ev["score"] != 0.8 || ev["reason"] != "grounded" {
		t.Fatalf("event %v", ev)
	}
}

func TestGroundTruthContext(t *testing.T) {
	cases := []struct {
		name     string
		gtc      string
		expected bool
		want     string
	}{
		{"criterion context rendered", "Answer: {{x}}", true, "Answer: 1"},
		{"falls back to expected output", "", true, "a0"},
		{"empty when neither", "", false, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			e, args, rec := judgeEnv(t, 1, verdict(1, "ok"), nil)
			item := map[string]any{"rowIndex": 0, "input": "q", "variables": map[string]any{"x": "1"}}
			if tc.expected {
				item["expectedOutput"] = "a0"
			}
			e.route("GET", "projects/proj/datasets/ds/rows", resp{body: map[string]any{"totalCount": 1, "items": []any{item}}})
			args.Criteria = []Criterion{mustJudge(t, "acc", &JudgeOptions{GroundTruthContext: tc.gtc})}
			mustRun(t, e.module(), args)
			if got := rec.all()[0].Variables["ground_truth_context"]; got != tc.want {
				t.Fatalf("ground_truth_context %q", got)
			}
		})
	}
}

func TestMessageHistoryCarriesTheTrajectory(t *testing.T) {
	e, args, rec := judgeEnv(t, 1, verdict(1, "ok"), nil)
	inner := args.Handler.Invoke
	args.Handler.Invoke = func(ctx context.Context, req aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
		if _, isJudge := req.Variables["message_history"]; !isJudge {
			_, _ = req.ToolHandlers["lookup"].(aiserver.ToolFunc)(ctx, map[string]any{"id": 1})
			return aiserver.HandlerResult{Output: "answer"}, nil
		}
		return inner(ctx, req)
	}
	args.InlineTools = map[string]InlineTool{"lookup": {Schema: map[string]any{}, Implementation: aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) {
		return "{{expected_output}} café 東京", nil
	})}}
	mustRun(t, e.module(), args)
	h := rec.all()[0].Variables["message_history"].(string)
	iIn, iTool, iOut, iFmt := strings.Index(h, "q0"), strings.Index(h, "1. lookup"), strings.Index(h, "answer"), strings.Index(h, "Your response MUST")
	if !(iIn >= 0 && iIn < iTool && iTool < iOut && iOut < iFmt) {
		t.Fatalf("history order: %q", h)
	}
	if !strings.Contains(h, "{{expected_output}} café 東京") || strings.Contains(h, `\u`) {
		t.Fatalf("tool result must stay literal: %q", h)
	}
	for k := range rec.all()[0].Variables {
		if strings.Contains(k, "trajectory") {
			t.Fatal("no separate trajectory variable")
		}
	}
}

func TestJudgeResponseParsing(t *testing.T) {
	out := func(o any) func(context.Context, aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
		return func(context.Context, aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
			return aiserver.HandlerResult{Output: o}, nil
		}
	}
	cases := []struct {
		name  string
		judge func(context.Context, aiserver.HandlerRequest) (aiserver.HandlerResult, error)
		cause string
		score float64
	}{
		{"prose", out("the answer looks right to me"), causeInvalidJudgeOutput, 0},
		{"non-object result", out(42.0), causeInvalidJudgeOutput, 0},
		{"empty object", out("{}"), causeInvalidJudgeOutput, 0},
		{"no reasoning", out(`{"score": 0.9}`), causeInvalidJudgeOutput, 0},
		{"empty reasoning", out(`{"score": 0.9, "reasoning": ""}`), causeInvalidJudgeOutput, 0},
		{"string score", out(`{"score": "high (0.9)", "reasoning": "r"}`), causeInvalidScore, 0},
		{"out of range", out(`{"score": 3, "reasoning": "r"}`), causeInvalidScore, 0},
		{"negative", out(`{"score": -0.1, "reasoning": "r"}`), causeInvalidScore, 0},
		{"boolean", out(`{"score": true, "reasoning": "r"}`), causeInvalidScore, 0},
		{"NaN", out(map[string]any{"score": math.NaN(), "reasoning": "r"}), causeInvalidScore, 0},
		{"handler raised", func(context.Context, aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
			return aiserver.HandlerResult{}, errors.New("judge down")
		}, causeHandlerRaised, 0},
		{"fenced JSON", out("```json\n{\"score\": 0.7, \"reasoning\": \"r\"}\n```"), "", 0.7},
		{"reason alias", out(`{"score": 0.4, "reason": "r"}`), "", 0.4},
		{"decoded object with a Go int", out(map[string]any{"score": 1, "reasoning": "r"}), "", 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			e, args, rec := judgeEnv(t, 1, tc.judge, nil)
			res := mustRun(t, e.module(), args)
			if res == nil || e.count("/summary") != 1 || e.ld.flushes != 2 {
				t.Fatal("run must complete, flush and poll")
			}
			if rec.n() != 1 {
				t.Fatalf("judge invoked %d times; never retried", rec.n())
			}
			ev := e.ld.events(CriterionEventName)[0].Data
			if tc.cause == "" {
				if ev["status"] != "COMPLETE" || ev["score"] != tc.score || ev["reason"] != "r" {
					t.Fatalf("event %v", ev)
				}
				return
			}
			if ev["status"] != "ERROR" || ev["errorMessage"] == nil || ev["errorMessage"] == "" {
				t.Fatalf("event %v", ev)
			}
			if msg, _ := ev["errorMessage"].(string); tc.cause == causeHandlerRaised && msg != "judge down" {
				t.Fatalf("errorMessage %q does not carry the handler's error", msg)
			}
			for _, k := range []string{"score", "reason", "error", "errorCode"} {
				if _, has := ev[k]; has {
					t.Fatalf("ERROR result has %s", k)
				}
			}
		})
	}
}

// TestCriterionCauses asserts the failure taxonomy on the result records.
func TestCriterionCauses(t *testing.T) {
	g := &genResult{row: renderRow(0, "q", nil, nil, nil), status: statusComplete, output: "o"}
	judgeWith := func(o any, err error) *critResult {
		r := &runState{judges: map[string]*resolvedJudge{"j": {handler: handler("OpenAI", "messages",
			func(context.Context, aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
				return aiserver.HandlerResult{Output: o}, err
			})}}}
		return r.runJudge(context.Background(), g, mustJudge(t, "j", nil))
	}
	cases := []struct {
		got  *critResult
		want string
	}{
		{judgeWith("prose", nil), causeInvalidJudgeOutput},
		{judgeWith(`{"score":0.5}`, nil), causeInvalidJudgeOutput},
		{judgeWith(`{"score":"0.5","reasoning":"r"}`, nil), causeInvalidScore},
		{judgeWith(nil, errors.New("x")), causeHandlerRaised},
		{runScorer(context.Background(), g, mustScorer(t, "s", func(context.Context, DatasetRow, any) (float64, error) { return 0, errors.New("x") }, nil)), causeScorerRaised},
		{runScorer(context.Background(), g, mustScorer(t, "s", func(context.Context, DatasetRow, any) (float64, error) { panic("bug") }, nil)), causeScorerRaised},
		{runScorer(context.Background(), g, mustScorer(t, "s", constScorer(math.NaN()), nil)), causeInvalidScore},
		{runScorer(context.Background(), g, mustScorer(t, "s", constScorer(1.5), nil)), causeInvalidScore},
		{runScorer(context.Background(), g, mustScorer(t, "s", constScorer(math.Inf(1)), nil)), causeInvalidScore},
	}
	for i, tc := range cases {
		if tc.got.status != statusError || tc.got.cause != tc.want || tc.got.errMsg == "" {
			t.Errorf("case %d: %+v, want cause %s", i, tc.got, tc.want)
		}
	}
	if r := runScorer(context.Background(), g, mustScorer(t, "s", constScorer(1.5), nil)); !strings.Contains(r.errMsg, "1.5") {
		t.Errorf("invalid_score must name the value: %q", r.errMsg)
	}
}

func TestJudgeUsageAndEventFields(t *testing.T) {
	e, args, _ := judgeEnv(t, 2, func(context.Context, aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
		return aiserver.HandlerResult{Output: `{"score":0.8,"reasoning":"good"}`, Usage: map[string]any{"inputTokens": 5.0, "outputTokens": 6.0}}, nil
	}, nil)
	args.Criteria = append(args.Criteria, mustScorer(t, "len", constScorer(1), nil))
	mustRun(t, e.module(), args)
	evs := e.ld.events(CriterionEventName)
	if len(evs) != 4 {
		t.Fatalf("%d criterion events", len(evs))
	}
	ids := map[string]bool{}
	for _, ev := range evs {
		d := ev.Data
		for k, want := range map[string]any{"projectKey": "proj", "evaluationId": "eval-1", "evaluationRunId": "run-1", "runId": "run-1",
			"datasetId": "ds-id", "evaluationKey": "eval-key", "evaluationVersion": 3.0, "datasetKey": "ds", "status": "COMPLETE"} {
			if d[k] != want {
				t.Errorf("%s = %v", k, d[k])
			}
		}
		started, _ := time.Parse(time.RFC3339Nano, d["startedAt"].(string))
		evaluated, _ := time.Parse(time.RFC3339Nano, d["evaluatedAt"].(string))
		if evaluated.Before(started) {
			t.Error("evaluatedAt precedes startedAt")
		}
		for _, k := range []string{"input", "expectedOutput", "metadata", "variables", "output", "inputTokens", "outputTokens", "verdict", "passed", "successDirection", "trajectory"} {
			if _, has := d[k]; has {
				t.Errorf("%s must be absent", k)
			}
		}
		id := toJSON(map[string]any{"row": d["rowIndex"], "c": d["criterionType"]})
		want, _ := eventID(map[string]any{"projectKey": "proj", "evaluationId": "eval-1", "evaluationRunId": "run-1", "runId": "run-1",
			"datasetId": "ds-id", "rowIndex": int(d["rowIndex"].(float64)), "criterionType": d["criterionType"]})
		if d["eventId"] != want {
			t.Errorf("eventId for %s", id)
		}
		ids[d["eventId"].(string)] = true
		switch d["kind"] {
		case "judge":
			if d["judgeKey"] != "acc" || d["variationKey"] != "jv" || d["version"] != 7.0 || d["reason"] != "good" ||
				!sameJSON(d["usage"], map[string]any{"inputTokens": 5, "outputTokens": 6}) {
				t.Errorf("judge event %v", d)
			}
		case "scorer":
			for _, k := range []string{"judgeKey", "variationKey", "version", "reason", "usage"} {
				if _, has := d[k]; has {
					t.Errorf("scorer event has %s", k)
				}
			}
			if d["score"] != 1.0 || d["criterionType"] != "len" {
				t.Errorf("scorer event %v", d)
			}
		default:
			t.Errorf("kind %v", d["kind"])
		}
	}
	if len(ids) != 4 {
		t.Fatal("criteria on one row must have distinct eventIds")
	}
}

func TestScorerInvocation(t *testing.T) {
	e := newEnv(t, 2)
	var rec recorder
	var mu sync.Mutex
	var rows []DatasetRow
	var outputs []any
	args := baseArgs(handler("OpenAI", aiserver.ModeMessages, rec.wrap(nil)))
	args.Criteria = []Criterion{mustScorer(t, "s", func(_ context.Context, r DatasetRow, out any) (float64, error) {
		mu.Lock()
		rows, outputs = append(rows, r), append(outputs, out)
		mu.Unlock()
		return 0.25, nil
	}, nil)}
	mustRun(t, e.module(), args)
	if rec.n() != 2 {
		t.Fatalf("handler invoked %d times; a scorer makes no model call", rec.n())
	}
	slices.SortFunc(rows, func(a, b DatasetRow) int { return a.RowIndex - b.RowIndex })
	if rows[1].RowIndex != 1 || rows[1].Input != "q1" || rows[1].ExpectedOutput != "a1" || rows[1].Variables["input"] != "q1" {
		t.Fatalf("scorer row %+v", rows[1])
	}
	if !slices.Contains(outputs, any("out:q0")) {
		t.Fatalf("outputs %v", outputs)
	}
}

func TestPhaseSequencingAndIsolation(t *testing.T) {
	e := newEnv(t, 3)
	var inGen atomic.Int32
	var overlap atomic.Bool
	var genCalls sync.Map
	h := handler("OpenAI", aiserver.ModeMessages, func(_ context.Context, req aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
		if _, isJudge := req.Variables["message_history"]; isJudge {
			if inGen.Load() != 0 {
				overlap.Store(true)
			}
			e.mu.Lock()
			e.log = append(e.log, "judge-invoke")
			e.mu.Unlock()
			return verdict(1, "ok")(nil, req)
		}
		inGen.Add(1)
		defer inGen.Add(-1)
		n, _ := genCalls.LoadOrStore(req.UserInput, new(atomic.Int32))
		n.(*atomic.Int32).Add(1)
		if req.UserInput == "q1" {
			return aiserver.HandlerResult{}, errors.New("generation failed")
		}
		return aiserver.HandlerResult{Output: "ok"}, nil
	})
	e.judges["acc"] = judgeVariation("OpenAI", "completion", nil)
	var scored atomic.Int32
	args := baseArgs(h)
	args.Criteria = []Criterion{mustJudge(t, "acc", nil), mustScorer(t, "s", func(context.Context, DatasetRow, any) (float64, error) {
		scored.Add(1)
		return 1, nil
	}, nil)}
	mustRun(t, e.module(), args)
	if overlap.Load() {
		t.Fatal("a judge ran while generation was in flight")
	}
	log := e.logSnapshot()
	lastGen, firstJudge := -1, slices.Index(log, "judge-invoke")
	for i, s := range log {
		if s == "track "+GenerationEventName {
			lastGen = i
		}
	}
	if lastGen > firstJudge || slices.Index(log, "flush") > firstJudge {
		t.Fatalf("generation events and the boundary flush must precede the first judge: %v", log)
	}
	for _, ev := range e.ld.events(CriterionEventName) {
		if ev.Data["rowIndex"] == 1.0 {
			t.Fatal("a row that did not generate got a criterion event")
		}
	}
	if len(e.ld.events(CriterionEventName)) != 4 || scored.Load() != 2 {
		t.Fatalf("%d criterion events, %d scorer calls", len(e.ld.events(CriterionEventName)), scored.Load())
	}
	genCalls.Range(func(k, v any) bool {
		if v.(*atomic.Int32).Load() != 1 {
			t.Errorf("row %v generated %d times", k, v.(*atomic.Int32).Load())
		}
		return true
	})
	if e.count("/summary") != 1 {
		t.Fatal("run must poll")
	}
}

func TestCriterionConcurrencyAcrossPairs(t *testing.T) {
	var live, peak atomic.Int32
	judge := func(_ context.Context, req aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
		n := live.Add(1)
		for p := peak.Load(); n > p && !peak.CompareAndSwap(p, n); p = peak.Load() {
		}
		time.Sleep(10 * time.Millisecond)
		live.Add(-1)
		return verdict(1, "ok")(nil, req)
	}
	e, args, _ := judgeEnv(t, 3, judge, nil)
	args.Concurrency = 2
	mustRun(t, e.module(), args)
	if peak.Load() != 2 {
		t.Fatalf("peak in-flight judges %d, want 2", peak.Load())
	}
}

// A judge variation reporting no version or variation key: version is
// omitted and variationKey is "".
func TestJudgeEventWithoutResolvedMetadata(t *testing.T) {
	e, args, _ := judgeEnv(t, 1, verdict(0.5, "r"), map[string]any{"_ldMeta": map[string]any{"enabled": true, "mode": "completion"}})
	mustRun(t, e.module(), args)
	d := e.ld.events(CriterionEventName)[0].Data
	if _, has := d["version"]; has || d["variationKey"] != "" || d["status"] != "COMPLETE" {
		t.Fatalf("event %v", d)
	}
}
