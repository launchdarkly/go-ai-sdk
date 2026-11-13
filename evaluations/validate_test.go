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
// lib is a library tool as Tools().Get returns it for project "proj".
func lib(key string, impl aiserver.ToolHandler) EvalTool {
	return EvalTool{Key: key, Implementation: impl, Schema: map[string]any{},
		library: &libraryPin{key: key, schema: map[string]any{}, version: 1, projectKey: "proj"}}
}

func TestRunValidationBeforeIO(t *testing.T) {
	msgs := []aiserver.Message{{Role: "user", Text: "hi"}}
	messages := handler("OpenAI", aiserver.ModeMessages, nil)
	agent := handler("OpenAI", aiserver.ModeAgent, nil)
	inlineRow := func(m map[string]any) InlineRows { return InlineRows{map[string]any{"input": "ok"}, m} }
	cases := []struct {
		name   string
		mutate func(*RunOptions)
		want   []string
	}{
		{"blank key", func(a *RunOptions) { a.Key = "" }, []string{"Key must not be blank"}},
		{"uppercase key", func(a *RunOptions) { a.Key = "Eval-Key" }, []string{"Key must match"}},
		{"colon key", func(a *RunOptions) { a.Key = "eval:key" }, []string{"Key must match"}},
		{"space key", func(a *RunOptions) { a.Key = "eval key" }, []string{"Key must match"}},
		{"leading hyphen", func(a *RunOptions) { a.Key = "-eval" }, []string{"Key must match"}},
		{"leading dot", func(a *RunOptions) { a.Key = ".eval" }, []string{"Key must match"}},
		{"leading underscore", func(a *RunOptions) { a.Key = "_eval" }, []string{"Key must match"}},
		{"dotdot key", func(a *RunOptions) { a.Key = ".." }, []string{"Key must match"}},
		{"key too long", func(a *RunOptions) { a.Key = strings.Repeat("a", 129) }, []string{"128"}},
		{"nil dataset", func(a *RunOptions) { a.Dataset = nil }, []string{"dataset is required", "DatasetKey or InlineRows"}},
		{"blank dataset", func(a *RunOptions) { a.Dataset = DatasetKey("  ") }, []string{"dataset must not be blank"}},
		{"empty inline dataset", func(a *RunOptions) { a.Dataset = InlineRows{} }, []string{"Inline dataset is empty"}},
		{"bare string row", func(a *RunOptions) { a.Dataset = InlineRows{"just text"} }, []string{"row 0 must be a mapping"}},
		{"snake_case near miss", func(a *RunOptions) { a.Dataset = inlineRow(map[string]any{"input": "x", "expected_output": "y"}) },
			[]string{"row 1", `"expected_output"`, "expectedOutput"}},
		{"rowIdx differs", func(a *RunOptions) { a.Dataset = inlineRow(map[string]any{"input": "x", "rowIdx": 5}) }, []string{"row 1", "rowIdx 5"}},
		{"bool rowIdx", func(a *RunOptions) { a.Dataset = inlineRow(map[string]any{"input": "x", "rowIdx": true}) }, []string{"row 1", "true"}},
		{"public row type", func(a *RunOptions) { a.Dataset = InlineRows{map[string]any{"input": "a"}, DatasetRow{Input: "b"}} },
			[]string{"row 1 must be a mapping", "evaluations.DatasetRow"}},
		{"public row pointer", func(a *RunOptions) { a.Dataset = InlineRows{&DatasetRow{Input: "a"}} },
			[]string{"row 0 must be a mapping", "*evaluations.DatasetRow"}},
		{"nil public row pointer", func(a *RunOptions) { a.Dataset = InlineRows{(*DatasetRow)(nil)} },
			[]string{"row 0 must be a mapping", "*evaluations.DatasetRow"}},
		{"missing input", func(a *RunOptions) { a.Dataset = inlineRow(map[string]any{}) }, []string{"row 1 input"}},
		{"empty input", func(a *RunOptions) { a.Dataset = inlineRow(map[string]any{"input": ""}) }, []string{"row 1 input"}},
		{"non-string input", func(a *RunOptions) { a.Dataset = inlineRow(map[string]any{"input": 3}) }, []string{"row 1 input"}},
		{"non-string expectedOutput", func(a *RunOptions) { a.Dataset = inlineRow(map[string]any{"input": "x", "expectedOutput": 1}) }, []string{"row 1 expectedOutput"}},
		{"non-mapping variables", func(a *RunOptions) { a.Dataset = inlineRow(map[string]any{"input": "x", "variables": []any{1}}) }, []string{"row 1 variables"}},
		{"NaN in variables of row 1", func(a *RunOptions) {
			a.Dataset = inlineRow(map[string]any{"input": "x", "variables": map[string]any{"n": math.NaN()}})
		}, []string{"row 1 variables", "NaN"}},
		{"Infinity in metadata", func(a *RunOptions) {
			a.Dataset = inlineRow(map[string]any{"input": "x", "metadata": map[string]any{"n": math.Inf(1)}})
		}, []string{"row 1 metadata", "+Inf"}},
		{"-Infinity nested in a list", func(a *RunOptions) {
			a.Dataset = inlineRow(map[string]any{"input": "x", "variables": map[string]any{"l": []any{1, math.Inf(-1)}}})
		}, []string{"row 1 variables", "-Inf"}},
		{"a set", func(a *RunOptions) {
			a.Dataset = inlineRow(map[string]any{"input": "x", "variables": map[string]any{"s": map[int]struct{}{1: {}}}})
		}, []string{"row 1 variables", "non-string keys"}},
		{"a date", func(a *RunOptions) {
			a.Dataset = inlineRow(map[string]any{"input": "x", "metadata": map[string]any{"at": time.Now()}})
		}, []string{"row 1 metadata", "time.Time"}},
		{"non-string key", func(a *RunOptions) {
			a.Dataset = InlineRows{map[string]any{"input": "x", "variables": map[string]any{"m": map[int]string{1: "a"}}}}
		},
			[]string{"row 0 variables", "non-string keys"}},
		{"handler and handlers", func(a *RunOptions) { a.Handlers = []*aiserver.ProviderHandler{agent} }, []string{"Handler and Handlers"}},
		{"neither handler", func(a *RunOptions) { a.Handler = nil }, []string{"Handler and Handlers"}},
		{"empty handlers", func(a *RunOptions) { a.Handler, a.Handlers = nil, []*aiserver.ProviderHandler{} }, []string{"Handlers must be a non-empty list"}},
		{"nil handlers entry", func(a *RunOptions) { a.Handler, a.Handlers = nil, []*aiserver.ProviderHandler{messages, nil} }, []string{"Handlers[1]", "CreateHandler"}},
		{"not callable", func(a *RunOptions) {
			a.Handler = &aiserver.ProviderHandler{ProvidesFor: aiserver.ProvidesFor{Provider: "OpenAI", Mode: "messages"}}
		},
			[]string{"Handler must be callable", "CreateHandler"}},
		{"no providesFor", func(a *RunOptions) {
			a.Handler, a.Handlers = nil, []*aiserver.ProviderHandler{messages, {Invoke: messages.Invoke}}
		}, []string{"Handlers[1]", "ProvidesFor", "CreateHandler"}},
		{"duplicate providesFor after normalizeMode", func(a *RunOptions) {
			a.Handler, a.Handlers = nil, []*aiserver.ProviderHandler{handler("OpenAI", "completion", nil), handler("OpenAI", "messages", nil)}
		}, []string{`"OpenAI"`, `"messages"`}},
		{"no source", func(a *RunOptions) { a.Generation = nil }, []string{"pass Generation, or AIConfig"}},
		{"both sources", func(a *RunOptions) { a.AIConfig = &AIConfigRef{Key: "k", Variation: "v"} }, []string{"not both"}},
		{"overrides without AI Config", func(a *RunOptions) { a.Overrides = &GenerationOverrides{} }, []string{"Overrides apply only with AIConfig"}},
		{"blank aiConfig.key", func(a *RunOptions) { a.Generation, a.AIConfig = nil, &AIConfigRef{Variation: "v"} }, []string{"AIConfig.Key"}},
		{"missing provider", func(a *RunOptions) { a.Generation.Provider = "" }, []string{"generation.provider is required"}},
		{"missing model", func(a *RunOptions) { a.Generation.Model = "" }, []string{"generation.model is required"}},
		{"missing mode", func(a *RunOptions) { a.Generation.Mode = "" }, []string{"generation.mode is required. Set it to 'completion' or 'agent'."}},
		{"mode messages", func(a *RunOptions) { a.Generation.Mode = "messages" }, []string{`"messages"`, "'completion' or 'agent'"}},
		{"mode judge", func(a *RunOptions) { a.Generation.Mode = "judge" }, []string{`"judge"`}},
		{"mode wrong case", func(a *RunOptions) { a.Generation.Mode = "Completion" }, []string{`"Completion"`}},
		{"mode typo", func(a *RunOptions) { a.Generation.Mode = "agnet" }, []string{`"agnet"`}},
		{"instructions and messages", func(a *RunOptions) { a.Generation.Instructions, a.Generation.Messages = "x", msgs }, []string{"generation.instructions and generation.messages"}},
		{"pool does not cover", func(a *RunOptions) { a.Handler = agent }, []string{`"OpenAI"`, `"messages" mode`, `"agent"`, "[OpenAI, agent]"}},
		{"messages handler with instructions", func(a *RunOptions) { a.Generation.Instructions = "x" }, []string{"needs messages", "[OpenAI, messages]"}},
		{"agent handler with messages", func(a *RunOptions) {
			a.Handler, a.Generation.Mode, a.Generation.Messages = agent, GenerationModeAgent, msgs
		}, []string{"needs instructions", "[OpenAI, agent]"}},
		{"negative concurrency", func(a *RunOptions) { a.Concurrency = -1 }, []string{"concurrency must be at least 1"}},
		{"negative poll interval", func(a *RunOptions) { a.PollInterval = -time.Second }, []string{"PollInterval"}},
		{"negative poll timeout", func(a *RunOptions) { a.PollTimeout = -time.Second }, []string{"PollTimeout"}},
		{"duplicate criteria", func(a *RunOptions) {
			a.Criteria = []Criterion{mustJudge(t, "accuracy", nil), mustScorer(t, "accuracy", constScorer(1), nil)}
		}, []string{"duplicate", "accuracy"}},
		{"duplicate criteria by case", func(a *RunOptions) {
			a.Criteria = []Criterion{mustJudge(t, "Accuracy", nil), mustScorer(t, "accuracy", constScorer(1), nil)}
		}, []string{"duplicate", "accuracy"}},
		{"nil criterion", func(a *RunOptions) { a.Criteria = []Criterion{(*Judge)(nil)} }, []string{"Criteria[0]"}},
		{"zero-value judge", func(a *RunOptions) { a.Criteria = []Criterion{&Judge{}} }, []string{"judge key must not be blank"}},
		{"blank inline key", func(a *RunOptions) {
			a.Tools = []EvalTool{{Key: " ", Implementation: noopTool, Schema: map[string]any{}}}
		}, []string{"tool keys must not be blank"}},
		{"blank library key", func(a *RunOptions) { a.Tools = []EvalTool{lib(" ", noopTool)} }, []string{"tool keys must not be blank"}},
		{"uppercase library key", func(a *RunOptions) { a.Tools = []EvalTool{lib("Lookup", noopTool)} }, []string{`"Lookup"`, "uppercase", `"lookup"`}},
		{"uppercase inline key", func(a *RunOptions) {
			a.Tools = []EvalTool{{Key: "Lookup", Implementation: noopTool, Schema: map[string]any{}}}
		}, []string{`"Lookup"`, "uppercase"}},
		{"key checked before implementation", func(a *RunOptions) { a.Tools = []EvalTool{{Key: "Lookup"}} }, []string{`"Lookup"`, "uppercase"}},
		{"inline schema missing", func(a *RunOptions) { a.Tools = []EvalTool{{Key: "t", Implementation: noopTool}} }, []string{`"t"`, "schema", "null"}},
		{"inline schema NaN", func(a *RunOptions) {
			a.Tools = []EvalTool{{Key: "t", Implementation: noopTool, Schema: map[string]any{"maximum": math.NaN()}}}
		}, []string{`"t"`, "NaN"}},
		{"inline schema with a date", func(a *RunOptions) {
			a.Tools = []EvalTool{{Key: "t", Implementation: noopTool, Schema: map[string]any{"d": time.Now()}}}
		}, []string{`"t"`, "time.Time"}},
		{"inline schema with a non-string key", func(a *RunOptions) {
			a.Tools = []EvalTool{{Key: "t", Implementation: noopTool, Schema: map[string]any{"properties": map[int]any{1: "x"}}}}
		}, []string{`"t"`, "$.properties", "non-string keys"}},
		{"inline schema with a non-string key inside a list", func(a *RunOptions) {
			a.Tools = []EvalTool{{Key: "t", Implementation: noopTool, Schema: map[string]any{"anyOf": []any{map[bool]any{true: "x"}}}}}
		}, []string{`"t"`, "$.anyOf[0]", "non-string keys"}},
		{"inline native tool", func(a *RunOptions) {
			a.Tools = []EvalTool{{Key: "web", Implementation: &aiserver.NativeTool{ToolName: "web"}, Schema: map[string]any{}}}
		}, []string{`"web"`, "must not use a NativeTool", "Tools().Get"}},
		{"inline without implementation", func(a *RunOptions) { a.Tools = []EvalTool{{Key: "t", Schema: map[string]any{}}} }, []string{`"t"`, "implementation"}},
		{"library nil implementation", func(a *RunOptions) { a.Tools = []EvalTool{lib("t", aiserver.ToolFunc(nil))} }, []string{`"t"`, "implementation"}},
		{"library nil native", func(a *RunOptions) { a.Tools = []EvalTool{lib("t", (*aiserver.NativeTool)(nil))} }, []string{`"t"`, "implementation"}},
		{"library tool from another project", func(a *RunOptions) {
			other := lib("t", noopTool)
			other.library.projectKey = "other"
			a.Tools = []EvalTool{other}
		}, []string{`"t"`, `project "other"`, `cannot run in project "proj"`}},
		{"library tool renamed after Get", func(a *RunOptions) {
			t := lib("lookup", noopTool)
			t.Key = "refund"
			a.Tools = []EvalTool{t}
		}, []string{`library tool "refund" was changed`, `read as "lookup"`, "inline tool"}},
		{"library tool schema replaced after Get", func(a *RunOptions) {
			t := lib("lookup", noopTool)
			t.Schema = map[string]any{"type": "object"}
			a.Tools = []EvalTool{t}
		}, []string{`library tool "lookup" was changed`}},
		{"library tool schema edited in place after Get", func(a *RunOptions) {
			t := lib("lookup", noopTool)
			t.Schema["type"] = "object"
			a.Tools = []EvalTool{t}
		}, []string{`library tool "lookup" was changed`}},
		{"library tool description changed after Get", func(a *RunOptions) {
			t := lib("lookup", noopTool)
			t.Description = "other"
			a.Tools = []EvalTool{t}
		}, []string{`library tool "lookup" was changed`}},
		{"same key from both sources", func(a *RunOptions) {
			a.Tools = []EvalTool{lib("lookup_order", noopTool), {Key: "lookup_order", Implementation: noopTool, Schema: map[string]any{}}}
		}, []string{`"lookup_order" (inline) collides with "lookup_order" (library)`, "once in Tools"}},
		{"same library key twice", func(a *RunOptions) {
			a.Tools = []EvalTool{lib("lookup", noopTool), lib("lookup", noopTool)}
		}, []string{`"lookup" (library) collides with "lookup" (library)`}},
		{"library and inline keys equal under case folding", func(a *RunOptions) {
			a.Tools = []EvalTool{lib("straße", noopTool), {Key: "strasse", Implementation: noopTool, Schema: map[string]any{}}}
		}, []string{`"strasse" (inline) collides with "straße" (library)`, "without case"}},
		{"inline keys equal under case folding", func(a *RunOptions) {
			a.Tools = []EvalTool{
				{Key: "straße", Implementation: noopTool, Schema: map[string]any{}},
				{Key: "strasse", Implementation: noopTool, Schema: map[string]any{}},
			}
		}, []string{`"strasse" (inline) collides with "straße" (inline)`}},
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
