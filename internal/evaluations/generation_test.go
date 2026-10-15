package evaluations

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

func TestHostedRowRendering(t *testing.T) {
	e := newEnv(t, 0)
	e.route("GET", "projects/proj/datasets/ds/rows", resp{body: map[string]any{"totalCount": 3, "items": []any{
		map[string]any{"rowIndex": 0, "input": "Hi {{name}} {{missing}}", "expectedOutput": "Bye {{name}}", "variables": map[string]any{"name": "Ann"}, "metadata": map[string]any{"m": 1}},
		map[string]any{"rowIndex": 1, "input": 5, "expectedOutput": nil, "metadata": "not an object"},
		map[string]any{"rowIndex": 2},
	}}})
	e.route("GET", "projects/proj/evaluations/eval-1/runs/run-1/summary", resp{body: map[string]any{"total": 3, "passed": 3}})
	var rec recorder
	var mu sync.Mutex
	got := map[int]DatasetRow{}
	args := baseArgs(handler("OpenAI", aiserver.ModeMessages, rec.wrap(nil)))
	args.Criteria = []Criterion{mustScorer(t, "s", func(_ context.Context, r DatasetRow, _ any) (float64, error) {
		mu.Lock()
		got[r.RowIndex] = r
		mu.Unlock()
		return 1, nil
	}, nil)}
	mustRun(t, e.module(), args)
	byInput := map[string]aiserver.HandlerRequest{}
	for _, c := range rec.all() {
		byInput[c.UserInput] = c
	}
	first, ok := byInput["Hi Ann {{missing}}"]
	if !ok || first.Variables["input"] != "Hi Ann {{missing}}" || first.Variables["expected_output"] != "Bye Ann" || first.Variables["name"] != "Ann" {
		t.Fatalf("rendered row 0: %v", byInput)
	}
	if v, has := byInput[""].Variables["input"]; !has || v != nil {
		t.Fatalf("non-string input must be injected as nil: %v", byInput[""].Variables)
	}
	if !reflect.DeepEqual(got[0].Metadata, map[string]any{"m": 1.0}) || got[1].Metadata != nil || got[0].ExpectedOutput != "Bye Ann" {
		t.Fatalf("scorer rows %+v", got)
	}
	t.Run("row without integer rowIndex throws", func(t *testing.T) {
		e := newEnv(t, 1)
		e.route("GET", "projects/proj/datasets/ds/rows", resp{body: map[string]any{"totalCount": 1, "items": []any{map[string]any{"input": "x"}}}})
		runErr(t, e.module(), baseArgs(handler("OpenAI", aiserver.ModeMessages, nil)), ErrInvalidResponse, "rowIndex")
	})
}

func TestInlineRowsShareTheRenderer(t *testing.T) {
	e := newEnv(t, 1)
	var rec recorder
	args := baseArgs(handler("OpenAI", aiserver.ModeMessages, rec.wrap(nil)))
	args.Dataset = InlineRows{map[string]any{"input": "Hi {{name}}", "expectedOutput": "{{name}}!", "variables": map[string]any{"name": "Bo"}}}
	mustRun(t, e.module(), args)
	c := rec.all()[0]
	if c.UserInput != "Hi Bo" || c.Variables["input"] != "Hi Bo" || c.Variables["expected_output"] != "Bo!" {
		t.Fatalf("inline render %q %v", c.UserInput, c.Variables)
	}
}

func TestGenerationHandlerSelection(t *testing.T) {
	cases := []struct {
		name     string
		pool     []*aiserver.ProviderHandler
		provider string
		mode     string
		want     int
	}{
		{"provider decides", []*aiserver.ProviderHandler{handler("OpenAI", "messages", nil), handler("Anthropic", "messages", nil)}, "Anthropic", ModeCompletion, 1},
		{"mode decides", []*aiserver.ProviderHandler{handler("OpenAI", "messages", nil), handler("OpenAI", "agent", nil)}, "OpenAI", ModeAgent, 1},
		{"exact beats wildcard", []*aiserver.ProviderHandler{handler("*", "messages", nil), handler("OpenAI", "messages", nil)}, "OpenAI", ModeCompletion, 1},
		{"wildcard alone", []*aiserver.ProviderHandler{handler("*", "messages", nil)}, "OpenAI", ModeCompletion, 0},
		{"handler mode is normalized", []*aiserver.ProviderHandler{handler("OpenAI", "completion", nil)}, "OpenAI", ModeCompletion, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			e := newEnv(t, 3)
			recs := make([]*recorder, len(tc.pool))
			for i, h := range tc.pool {
				recs[i] = &recorder{}
				h.Invoke = recs[i].wrap(nil)
			}
			args := baseArgs(nil)
			args.Handler, args.Handlers = nil, tc.pool
			args.Generation = &GenerationConfig{Provider: tc.provider, Model: "m", Mode: tc.mode}
			mustRun(t, e.module(), args)
			for i, r := range recs {
				want := 0
				if i == tc.want {
					want = 3
				}
				if r.n() != want {
					t.Fatalf("handler %d invoked %d times, want %d", i, r.n(), want)
				}
			}
		})
	}
}

func TestHandlerConfigConstruction(t *testing.T) {
	e := newEnv(t, 1)
	var rec recorder
	args := baseArgs(handler("OpenAI", aiserver.ModeMessages, rec.wrap(nil)))
	args.Generation.PromptSnippets = map[string]string{"tone": "kind"}
	args.Generation.Messages = []aiserver.Message{
		{Role: "system", Text: "Be {{snippet.tone}} about {{input}}"},
		{Role: "user", Blocks: []aiserver.ContentBlock{{Type: "text", Text: "{{snippet.tone}}"}}},
	}
	args.Generation.OutputFormat = map[string]any{"type": "object"}
	mustRun(t, e.module(), args)
	cfg := rec.all()[0].Config
	if cfg.Provider.Name != "OpenAI" || cfg.Model.Name != "gpt-4o" || cfg.Model.Parameters != nil {
		t.Fatalf("provider/model %+v", cfg)
	}
	if cfg.Messages[0].Text != "Be kind about {{input}}" || cfg.Messages[1].Blocks[0].Text != "{{snippet.tone}}" {
		t.Fatalf("snippets %+v", cfg.Messages)
	}
	if !reflect.DeepEqual(cfg.OutputFormat, map[string]any{"type": "object"}) {
		t.Fatalf("outputFormat %v", cfg.OutputFormat)
	}
	t.Run("instructions snippets and no outputFormat", func(t *testing.T) {
		e := newEnv(t, 1)
		var rec recorder
		args := baseArgs(handler("OpenAI", aiserver.ModeAgent, rec.wrap(nil)))
		args.Generation = &GenerationConfig{Provider: "OpenAI", Model: "m", Mode: ModeAgent, Instructions: "{{snippet.a}} {{input}}",
			PromptSnippets: map[string]string{"a": "A"}, Parameters: map[string]any{"t": 1}}
		mustRun(t, e.module(), args)
		cfg := rec.all()[0].Config
		if cfg.Instructions != "A {{input}}" || cfg.OutputFormat != nil || cfg.Model.Parameters["t"] != 1 {
			t.Fatalf("config %+v", cfg)
		}
	})
}

// TestToolConfigIsSourceIndependent runs one handler with a library tool
// and with the same tool declared inline; the configs must be equal.
func TestToolConfigIsSourceIndependent(t *testing.T) {
	schema := map[string]any{"type": "object", "properties": map[string]any{"id": map[string]any{"type": "string"}}}
	configs := make([]aiserver.AiConfig, 2)
	for i, inline := range []bool{false, true} {
		e := newEnv(t, 1)
		e.route("GET", "projects/proj/ai-tools/lookup", resp{body: map[string]any{"version": 1, "description": "Find it", "schema": schema}})
		var rec recorder
		args := baseArgs(handler("OpenAI", aiserver.ModeMessages, rec.wrap(nil)))
		if inline {
			args.InlineTools = map[string]InlineTool{"lookup": {Implementation: noopTool, Schema: schema, Description: "Find it"}}
		} else {
			args.Tools = map[string]aiserver.ToolHandler{"lookup": noopTool}
		}
		mustRun(t, e.module(), args)
		configs[i] = rec.all()[0].Config
	}
	if !reflect.DeepEqual(configs[0], configs[1]) {
		t.Fatalf("library %+v\ninline  %+v", configs[0], configs[1])
	}
	if tool := configs[0].Tools["lookup"]; tool.Description != "Find it" || !reflect.DeepEqual(tool.Parameters, schema) {
		t.Fatalf("tool %+v", tool)
	}
}

func TestToolHandlersKeepKeysAndPassThrough(t *testing.T) {
	e := newEnv(t, 1)
	e.route("GET", "projects/proj/ai-tools/lib", resp{body: map[string]any{"version": 1}})
	wantErr := errors.New("tool failed")
	native := &aiserver.NativeTool{ToolName: "web_search"}
	var results []any
	var errs []error
	h := handler("OpenAI", aiserver.ModeMessages, func(ctx context.Context, req aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
		if req.ToolHandlers["web"] != native {
			t.Error("native tool must pass through unchanged")
		}
		for _, k := range []string{"lib", "inl"} {
			v, err := req.ToolHandlers[k].(aiserver.ToolFunc)(ctx, map[string]any{"k": k})
			results, errs = append(results, v), append(errs, err)
		}
		return aiserver.HandlerResult{Output: "done"}, nil
	})
	args := baseArgs(h)
	args.Tools = map[string]aiserver.ToolHandler{"lib": aiserver.ToolFunc(func(_ context.Context, a map[string]any) (any, error) {
		return map[string]any{"from": a["k"]}, nil
	}), "web": native}
	args.InlineTools = map[string]InlineTool{"inl": {Schema: map[string]any{}, Implementation: aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) {
		return nil, wantErr
	})}}
	e.route("GET", "projects/proj/ai-tools/web", resp{body: map[string]any{"version": 1}})
	mustRun(t, e.module(), args)
	if !reflect.DeepEqual(results[0], map[string]any{"from": "lib"}) || errs[0] != nil || errs[1] != wantErr {
		t.Fatalf("results %v errs %v", results, errs)
	}
}

func TestGenerationConcurrencyBounded(t *testing.T) {
	for _, tc := range []struct{ concurrency, want int }{{3, 3}, {0, 10}} {
		e := newEnv(t, 25)
		var live, peak atomic.Int32
		h := handler("OpenAI", aiserver.ModeMessages, func(context.Context, aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
			n := live.Add(1)
			for {
				p := peak.Load()
				if n <= p || peak.CompareAndSwap(p, n) {
					break
				}
			}
			time.Sleep(5 * time.Millisecond)
			live.Add(-1)
			return aiserver.HandlerResult{Output: "x"}, nil
		})
		args := baseArgs(h)
		args.Concurrency = tc.concurrency
		mustRun(t, e.module(), args)
		if int(peak.Load()) != tc.want || len(e.ld.events(GenerationEventName)) != 25 {
			t.Fatalf("concurrency %d: peak %d, events %d", tc.concurrency, peak.Load(), len(e.ld.events(GenerationEventName)))
		}
	}
}

func TestHandlerErrorsAreRowErrors(t *testing.T) {
	e := newEnv(t, 4)
	var calls sync.Map
	h := handler("OpenAI", aiserver.ModeMessages, func(_ context.Context, req aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
		n, _ := calls.LoadOrStore(req.UserInput, new(atomic.Int32))
		n.(*atomic.Int32).Add(1)
		switch req.UserInput {
		case "q0":
			return aiserver.HandlerResult{}, errors.New("provider exploded")
		case "q1":
			return aiserver.HandlerResult{}, errors.New("")
		case "q2":
			panic("handler bug")
		}
		return aiserver.HandlerResult{}, nil // nil output: no output field
	})
	res := mustRun(t, e.module(), baseArgs(h))
	if !res.Passed || e.count("/summary") != 1 {
		t.Fatal("run must proceed to polling")
	}
	byRow := map[float64]map[string]any{}
	for _, ev := range e.ld.events(GenerationEventName) {
		byRow[ev.Data["rowIndex"].(float64)] = ev.Data
	}
	checks := []struct {
		row     float64
		message string
	}{{0, "handler raised: provider exploded"}, {1, "Unknown error"}, {2, "handler raised: panic: handler bug"}}
	for _, c := range checks {
		d := byRow[c.row]
		if d["status"] != "ERROR" || d["errorMessage"] != c.message || !sameJSON(d["error"], map[string]any{"code": 5001, "message": c.message}) {
			t.Errorf("row %v: %v", c.row, d)
		}
		if _, has := d["output"]; has {
			t.Errorf("row %v: output on an ERROR row", c.row)
		}
	}
	if d := byRow[3]; d["status"] != "COMPLETE" {
		t.Errorf("row 3: %v", d)
	} else {
		for _, k := range []string{"output", "error", "errorMessage", "usage"} {
			if _, has := d[k]; has {
				t.Errorf("row 3 has %s", k)
			}
		}
	}
	calls.Range(func(k, v any) bool {
		if v.(*atomic.Int32).Load() != 1 {
			t.Errorf("row %v invoked %d times; handlers are never retried", k, v.(*atomic.Int32).Load())
		}
		return true
	})
}

func TestTrajectoryPerRow(t *testing.T) {
	e := newEnv(t, 2)
	gate := make(chan struct{})
	var started sync.WaitGroup
	started.Add(2)
	tool := aiserver.ToolFunc(func(_ context.Context, a map[string]any) (any, error) { return a["row"], nil })
	h := handler("OpenAI", aiserver.ModeMessages, func(ctx context.Context, req aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
		fn := req.ToolHandlers["lookup"].(aiserver.ToolFunc)
		_, _ = fn(ctx, map[string]any{"row": req.UserInput, "n": 1})
		started.Done()
		<-gate // both rows have made one call; interleave the second
		_, _ = fn(ctx, map[string]any{"row": req.UserInput, "n": 2})
		if req.UserInput == "q1" {
			return aiserver.HandlerResult{}, errors.New("late failure")
		}
		return aiserver.HandlerResult{Output: "ok"}, nil
	})
	go func() { started.Wait(); close(gate) }()
	args := baseArgs(h)
	args.InlineTools = map[string]InlineTool{"lookup": {Implementation: tool, Schema: map[string]any{}}}
	r, err := e.module().prepare(args)
	if err != nil {
		t.Fatal(err)
	}
	r.client = e.ld
	r.rows = []row{renderRow(0, "q0", nil, nil, nil), renderRow(1, "q1", nil, nil, nil)}
	gens, err := r.generate(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	for _, g := range gens {
		if len(g.invocations) != 2 {
			t.Fatalf("row %d recorded %d calls", g.row.index, len(g.invocations))
		}
		for _, inv := range g.invocations {
			if !strings.Contains(inv.Arguments, `"row":"`+deref(g.row.input)+`"`) {
				t.Fatalf("row %d trajectory has another row's call: %+v", g.row.index, inv)
			}
		}
		if !strings.Contains(g.trajectory, "Tools available: lookup") {
			t.Fatalf("trajectory %q", g.trajectory)
		}
	}
	if gens[1].status != statusError {
		t.Fatal("row 1 should have failed but kept its calls")
	}
	f := generationFields(gens[0])
	for _, k := range []string{"trajectory", "tools", "toolCalls"} {
		if _, has := f[k]; has {
			t.Fatalf("generation event carries %s", k)
		}
	}
}
