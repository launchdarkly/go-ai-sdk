package evaluations

import (
	"context"
	"errors"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

var noopTool = aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) { return "ok", nil })

// TestRunValidationBeforeIO covers every pre-I/O validation check: each must fail with
// ErrInvalidArgument, name what is wrong, and send zero requests.
func TestRunValidationBeforeIO(t *testing.T) {
	msgs := []aiserver.Message{{Role: "user", Text: "hi"}}
	messages := handler("OpenAI", aiserver.ModeMessages, nil)
	agent := handler("OpenAI", aiserver.ModeAgent, nil)
	inlineRow := func(m map[string]any) InlineRows { return InlineRows{map[string]any{"input": "ok"}, m} }
	cases := []struct {
		name   string
		mutate func(*RunArgs)
		want   []string
	}{
		{"blank project", func(a *RunArgs) { a.ProjectKey = " " }, []string{"ProjectKey"}},
		{"blank key", func(a *RunArgs) { a.Key = "" }, []string{"Key must not be blank"}},
		{"nil dataset", func(a *RunArgs) { a.Dataset = nil }, []string{"dataset is required", "DatasetKey or InlineRows"}},
		{"blank dataset", func(a *RunArgs) { a.Dataset = DatasetKey("  ") }, []string{"dataset must not be blank"}},
		{"empty inline dataset", func(a *RunArgs) { a.Dataset = InlineRows{} }, []string{"Inline dataset is empty"}},
		{"bare string row", func(a *RunArgs) { a.Dataset = InlineRows{"just text"} }, []string{"row 0"}},
		{"snake_case near miss", func(a *RunArgs) { a.Dataset = inlineRow(map[string]any{"input": "x", "expected_output": "y"}) },
			[]string{"row 1", `"expected_output"`, "expectedOutput"}},
		{"rowIdx differs", func(a *RunArgs) { a.Dataset = inlineRow(map[string]any{"input": "x", "rowIdx": 5}) }, []string{"row 1", "rowIdx 5"}},
		{"bool rowIdx", func(a *RunArgs) { a.Dataset = inlineRow(map[string]any{"input": "x", "rowIdx": true}) }, []string{"row 1", "true"}},
		{"public row index differs", func(a *RunArgs) { a.Dataset = InlineRows{DatasetRow{Input: "a"}, DatasetRow{RowIndex: 3, Input: "b"}} }, []string{"row 1", "RowIndex 3"}},
		{"missing input", func(a *RunArgs) { a.Dataset = inlineRow(map[string]any{}) }, []string{"row 1 input"}},
		{"empty input", func(a *RunArgs) { a.Dataset = inlineRow(map[string]any{"input": ""}) }, []string{"row 1 input"}},
		{"non-string input", func(a *RunArgs) { a.Dataset = inlineRow(map[string]any{"input": 3}) }, []string{"row 1 input"}},
		{"non-string expectedOutput", func(a *RunArgs) { a.Dataset = inlineRow(map[string]any{"input": "x", "expectedOutput": 1}) }, []string{"row 1 expectedOutput"}},
		{"non-mapping variables", func(a *RunArgs) { a.Dataset = inlineRow(map[string]any{"input": "x", "variables": []any{1}}) }, []string{"row 1 variables"}},
		{"NaN in variables of row 1", func(a *RunArgs) {
			a.Dataset = inlineRow(map[string]any{"input": "x", "variables": map[string]any{"n": math.NaN()}})
		}, []string{"row 1 variables", "NaN"}},
		{"Infinity in metadata", func(a *RunArgs) {
			a.Dataset = inlineRow(map[string]any{"input": "x", "metadata": map[string]any{"n": math.Inf(1)}})
		}, []string{"row 1 metadata", "+Inf"}},
		{"-Infinity nested in a list", func(a *RunArgs) {
			a.Dataset = inlineRow(map[string]any{"input": "x", "variables": map[string]any{"l": []any{1, math.Inf(-1)}}})
		}, []string{"row 1 variables", "-Inf"}},
		{"a set", func(a *RunArgs) {
			a.Dataset = inlineRow(map[string]any{"input": "x", "variables": map[string]any{"s": map[int]struct{}{1: {}}}})
		}, []string{"row 1 variables", "non-string keys"}},
		{"a date", func(a *RunArgs) {
			a.Dataset = inlineRow(map[string]any{"input": "x", "metadata": map[string]any{"at": time.Now()}})
		}, []string{"row 1 metadata", "time.Time"}},
		{"non-string key", func(a *RunArgs) {
			a.Dataset = InlineRows{DatasetRow{Input: "x", Variables: map[string]any{"m": map[int]string{1: "a"}}}}
		},
			[]string{"row 0 variables", "non-string keys"}},
		{"handler and handlers", func(a *RunArgs) { a.Handlers = []*aiserver.ProviderHandler{agent} }, []string{"Handler and Handlers"}},
		{"neither handler", func(a *RunArgs) { a.Handler = nil }, []string{"Handler and Handlers"}},
		{"empty handlers", func(a *RunArgs) { a.Handler, a.Handlers = nil, []*aiserver.ProviderHandler{} }, []string{"Handlers must be a non-empty list"}},
		{"nil handlers entry", func(a *RunArgs) { a.Handler, a.Handlers = nil, []*aiserver.ProviderHandler{messages, nil} }, []string{"Handlers[1]", "CreateHandler"}},
		{"not callable", func(a *RunArgs) {
			a.Handler = &aiserver.ProviderHandler{ProvidesFor: aiserver.ProvidesFor{Provider: "OpenAI", Mode: "messages"}}
		},
			[]string{"Handler must be callable", "CreateHandler"}},
		{"no providesFor", func(a *RunArgs) {
			a.Handler, a.Handlers = nil, []*aiserver.ProviderHandler{messages, {Invoke: messages.Invoke}}
		}, []string{"Handlers[1]", "ProvidesFor", "CreateHandler"}},
		{"duplicate providesFor after normalizeMode", func(a *RunArgs) {
			a.Handler, a.Handlers = nil, []*aiserver.ProviderHandler{handler("OpenAI", "completion", nil), handler("OpenAI", "messages", nil)}
		}, []string{`"OpenAI"`, `"messages"`}},
		{"no source", func(a *RunArgs) { a.Generation = nil }, []string{"pass Generation, or AIConfig"}},
		{"both sources", func(a *RunArgs) { a.AIConfig = &AIConfigRef{Key: "k", Variation: "v"} }, []string{"not both"}},
		{"overrides without AI Config", func(a *RunArgs) { a.Overrides = &GenerationOverrides{} }, []string{"Overrides apply only with AIConfig"}},
		{"blank aiConfig.key", func(a *RunArgs) { a.Generation, a.AIConfig = nil, &AIConfigRef{Variation: "v"} }, []string{"AIConfig.Key"}},
		{"blank aiConfig.variation", func(a *RunArgs) { a.Generation, a.AIConfig = nil, &AIConfigRef{Key: "k", Variation: " "} }, []string{"AIConfig.Variation"}},
		{"missing provider", func(a *RunArgs) { a.Generation.Provider = "" }, []string{"generation.provider is required"}},
		{"missing model", func(a *RunArgs) { a.Generation.Model = "" }, []string{"generation.model is required"}},
		{"missing mode", func(a *RunArgs) { a.Generation.Mode = "" }, []string{"generation.mode is required. Set it to 'completion' or 'agent'."}},
		{"mode messages", func(a *RunArgs) { a.Generation.Mode = "messages" }, []string{`"messages"`, "'completion' or 'agent'"}},
		{"mode judge", func(a *RunArgs) { a.Generation.Mode = "judge" }, []string{`"judge"`}},
		{"mode wrong case", func(a *RunArgs) { a.Generation.Mode = "Completion" }, []string{`"Completion"`}},
		{"mode typo", func(a *RunArgs) { a.Generation.Mode = "agnet" }, []string{`"agnet"`}},
		{"instructions and messages", func(a *RunArgs) { a.Generation.Instructions, a.Generation.Messages = "x", msgs }, []string{"generation.instructions and generation.messages"}},
		{"pool does not cover", func(a *RunArgs) { a.Handler = agent }, []string{`"OpenAI"`, `"messages" mode`, `"agent"`, "[OpenAI, agent]"}},
		{"messages handler with instructions", func(a *RunArgs) { a.Generation.Instructions = "x" }, []string{"needs messages", "[OpenAI, messages]"}},
		{"agent handler with messages", func(a *RunArgs) { a.Handler, a.Generation.Mode, a.Generation.Messages = agent, ModeAgent, msgs }, []string{"needs instructions", "[OpenAI, agent]"}},
		{"negative concurrency", func(a *RunArgs) { a.Concurrency = -1 }, []string{"concurrency must be at least 1"}},
		{"negative poll interval", func(a *RunArgs) { a.PollInterval = -time.Second }, []string{"PollInterval"}},
		{"negative poll timeout", func(a *RunArgs) { a.PollTimeout = -time.Second }, []string{"PollTimeout"}},
		{"duplicate criteria", func(a *RunArgs) {
			a.Criteria = []Criterion{mustJudge(t, "accuracy", nil), mustScorer(t, "accuracy", constScorer(1), nil)}
		}, []string{"duplicate", "accuracy"}},
		{"duplicate criteria by case", func(a *RunArgs) {
			a.Criteria = []Criterion{mustJudge(t, "Accuracy", nil), mustScorer(t, "accuracy", constScorer(1), nil)}
		}, []string{"duplicate", "accuracy"}},
		{"nil criterion", func(a *RunArgs) { a.Criteria = []Criterion{(*Judge)(nil)} }, []string{"Criteria[0]"}},
		{"zero-value judge", func(a *RunArgs) { a.Criteria = []Criterion{&Judge{}} }, []string{"judge key must not be blank"}},
		{"blank inline key", func(a *RunArgs) {
			a.InlineTools = map[string]InlineTool{" ": {Implementation: noopTool, Schema: map[string]any{}}}
		}, []string{"InlineTools"}},
		{"uppercase inline key", func(a *RunArgs) {
			a.InlineTools = map[string]InlineTool{"Lookup": {Implementation: noopTool, Schema: map[string]any{}}}
		}, []string{`"Lookup"`, "uppercase"}},
		{"inline schema missing", func(a *RunArgs) { a.InlineTools = map[string]InlineTool{"t": {Implementation: noopTool}} }, []string{`"t"`, "schema", "null"}},
		{"inline schema NaN", func(a *RunArgs) {
			a.InlineTools = map[string]InlineTool{"t": {Implementation: noopTool, Schema: map[string]any{"maximum": math.NaN()}}}
		}, []string{`"t"`, "NaN"}},
		{"inline schema with a date", func(a *RunArgs) {
			a.InlineTools = map[string]InlineTool{"t": {Implementation: noopTool, Schema: map[string]any{"d": time.Now()}}}
		}, []string{`"t"`, "time.Time"}},
		{"inline native tool", func(a *RunArgs) {
			a.InlineTools = map[string]InlineTool{"web": {Implementation: &aiserver.NativeTool{ToolName: "web"}, Schema: map[string]any{}}}
		}, []string{`"web"`, "native tool has no schema to define"}},
		{"inline without implementation", func(a *RunArgs) { a.InlineTools = map[string]InlineTool{"t": {Schema: map[string]any{}}} }, []string{`"t"`, "implementation"}},
		{"library nil implementation", func(a *RunArgs) { a.Tools = map[string]aiserver.ToolHandler{"t": aiserver.ToolFunc(nil)} }, []string{`"t"`, "implementation"}},
		{"library nil native", func(a *RunArgs) { a.Tools = map[string]aiserver.ToolHandler{"t": (*aiserver.NativeTool)(nil)} }, []string{`"t"`}},
		{"same key from both sources", func(a *RunArgs) {
			a.Tools = map[string]aiserver.ToolHandler{"lookup_order": noopTool}
			a.InlineTools = map[string]InlineTool{"lookup_order": {Implementation: noopTool, Schema: map[string]any{}}}
		}, []string{`"lookup_order"`, "library", "inline"}},
		{"keys differing only by case", func(a *RunArgs) {
			a.Tools = map[string]aiserver.ToolHandler{"Lookup_Order": noopTool}
			a.InlineTools = map[string]InlineTool{"lookup_order": {Implementation: noopTool, Schema: map[string]any{}}}
		}, []string{`"Lookup_Order"`, `"lookup_order"`}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			e := newEnv(t, 1)
			m := e.module()
			m.sdkKey = "sdk-key"
			e.byoc = false // validation must come first: a client resolution would be recorded in initCalls
			args := baseArgs(messages)
			args.Generation = gen()
			tc.mutate(&args)
			runErr(t, m, args, ErrInvalidArgument, tc.want...)
			e.assertNoRequests()
			if len(e.initCalls) != 0 {
				t.Fatal("validation must precede client resolution")
			}
		})
	}
}

func TestTwoLibraryKeysDifferingByCaseAreTwoLookups(t *testing.T) {
	e := newEnv(t, 1)
	for _, k := range []string{"Lookup", "lookup"} {
		e.route("GET", "projects/proj/ai-tools/"+k, resp{body: map[string]any{"version": 1}})
	}
	args := baseArgs(handler("OpenAI", aiserver.ModeMessages, nil))
	args.Tools = map[string]aiserver.ToolHandler{"Lookup": noopTool, "lookup": noopTool}
	mustRun(t, e.module(), args)
	if e.count("ai-tools/") != 2 {
		t.Fatalf("tool GETs: %v", e.sequence())
	}
}

func TestCriterionConstruction(t *testing.T) {
	bad := []float64{-0.5, 1.5, math.NaN(), math.Inf(1), math.Inf(-1)}
	for _, v := range bad {
		for _, field := range []string{"threshold", "passRateThreshold"} {
			jo, so := &JudgeOptions{}, &ScorerOptions{}
			if field == "threshold" {
				jo.Threshold, so.Threshold = ptr(v), ptr(v)
			} else {
				jo.PassRateThreshold, so.PassRateThreshold = ptr(v), ptr(v)
			}
			if _, err := NewJudge("j", jo); !errors.Is(err, ErrInvalidArgument) || !strings.Contains(err.Error(), field) {
				t.Errorf("judge %s=%v: %v", field, v, err)
			}
			if _, err := NewScorer("s", constScorer(1), so); !errors.Is(err, ErrInvalidArgument) || !strings.Contains(err.Error(), field) {
				t.Errorf("scorer %s=%v: %v", field, v, err)
			}
		}
	}
	if _, err := NewJudge(" ", nil); err == nil {
		t.Error("blank judge key accepted")
	}
	if _, err := NewScorer("", constScorer(1), nil); err == nil {
		t.Error("blank scorer name accepted")
	}
	if _, err := NewScorer("s", nil, nil); err == nil || !strings.Contains(err.Error(), "callable") {
		t.Errorf("nil scorer fn: %v", err)
	}
	if _, err := NewScorer("s", constScorer(1), &ScorerOptions{SuccessDirection: "sideways"}); err == nil {
		t.Error("unknown direction accepted")
	}
	j := mustJudge(t, "j", nil)
	if j.threshold != 0.5 || j.passRate != 0.95 {
		t.Errorf("judge defaults %v %v", j.threshold, j.passRate)
	}
	s := mustScorer(t, "s", constScorer(1), nil)
	if s.threshold != 1.0 || s.passRate != 0.95 || s.direction != HigherIsBetter {
		t.Errorf("scorer defaults %v %v %v", s.threshold, s.passRate, s.direction)
	}
	if z := mustJudge(t, "z", &JudgeOptions{Threshold: ptr(0.0), PassRateThreshold: ptr(1.0)}); z.threshold != 0 || z.passRate != 1 {
		t.Errorf("boundary thresholds %v %v", z.threshold, z.passRate)
	}
}
