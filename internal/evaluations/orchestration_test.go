package evaluations

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

func assertSequence(t *testing.T, e *fakeEnv, want ...string) {
	t.Helper()
	if got := e.sequence(); !reflect.DeepEqual(got, want) {
		t.Fatalf("request sequence\n got: %q\nwant: %q", got, want)
	}
}

func assertNoURL(t *testing.T, e *fakeEnv, substr string) {
	t.Helper()
	for _, s := range e.sequence() {
		if strings.Contains(s, substr) {
			t.Fatalf("unexpected request %s", s)
		}
	}
}

const (
	evalPath    = "POST projects/proj/evaluations"
	runPath     = "POST projects/proj/evaluations/eval-1/runs"
	summaryPath = "GET projects/proj/evaluations/eval-1/runs/run-1/summary"
	rowsPath    = "GET projects/proj/datasets/ds/rows?mode=all&limit=200&offset=0"
	uploadPath  = "POST projects/proj/evaluations/eval-1/runs/run-1/dataset-rows"
)

func TestHappyPathHostedSequence(t *testing.T) {
	e := newEnv(t, 2)
	e.route("GET", "projects/proj/ai-tools/lookup", resp{body: map[string]any{"version": 4, "description": "d", "schema": map[string]any{"type": "object"}}})
	args := baseArgs(handler("OpenAI", aiserver.ModeMessages, nil))
	args.Tools = map[string]aiserver.ToolHandler{"lookup": noopTool}
	res := mustRun(t, e.module(), args)
	assertSequence(t, e, "GET projects/proj/ai-tools/lookup", "GET projects/proj/datasets/ds", rowsPath, evalPath, runPath, summaryPath)
	for _, bad := range []string{"/generation-results", "/evaluation-results", "dataset-rows", "/cancel"} {
		assertNoURL(t, e, bad)
	}
	if !res.Passed || res.RunID != "run-1" || res.Summary.TotalRows != 2 {
		t.Fatalf("result %+v", res)
	}
	// Both phases run in the gap: nothing but tracks and flushes between
	// run creation and the first summary poll.
	log := e.logSnapshot()
	i, j := slices.Index(log, runPath), slices.Index(log, summaryPath)
	for _, s := range log[i+1 : j] {
		if !strings.HasPrefix(s, "track ") && s != "flush" {
			t.Fatalf("request %s between run creation and summary poll", s)
		}
	}
}

func TestHappyPathAllInlineToolsSequence(t *testing.T) {
	e := newEnv(t, 1)
	args := baseArgs(handler("OpenAI", aiserver.ModeMessages, nil))
	args.InlineTools = map[string]InlineTool{"lookup": {Implementation: noopTool, Schema: map[string]any{}}}
	mustRun(t, e.module(), args)
	assertSequence(t, e, "GET projects/proj/datasets/ds", rowsPath, evalPath, runPath, summaryPath)
	assertNoURL(t, e, "/ai-tools")
}

func TestMixedToolsLookUpLibraryKeysOnly(t *testing.T) {
	e := newEnv(t, 1)
	for _, k := range []string{"alpha", "gamma"} {
		e.route("GET", "projects/proj/ai-tools/"+k, resp{body: map[string]any{"version": 1}})
	}
	args := baseArgs(handler("OpenAI", aiserver.ModeMessages, nil))
	args.Tools = map[string]aiserver.ToolHandler{"alpha": noopTool, "gamma": noopTool}
	args.InlineTools = map[string]InlineTool{"beta": {Implementation: noopTool, Schema: map[string]any{}}}
	mustRun(t, e.module(), args)
	var keys []string
	for _, s := range e.sequence() {
		if k, ok := strings.CutPrefix(s, "GET projects/proj/ai-tools/"); ok {
			keys = append(keys, k)
		}
	}
	slices.Sort(keys)
	if !reflect.DeepEqual(keys, []string{"alpha", "gamma"}) {
		t.Fatalf("tool GETs for %v", keys)
	}
}

func TestHappyPathInlineDatasetSequence(t *testing.T) {
	e := newEnv(t, 2)
	args := baseArgs(handler("OpenAI", aiserver.ModeMessages, nil))
	args.Dataset = InlineRows{map[string]any{"input": "a"}, map[string]any{"input": "b"}}
	args.InlineTools = map[string]InlineTool{"lookup": {Implementation: noopTool, Schema: map[string]any{}}}
	mustRun(t, e.module(), args)
	assertSequence(t, e, evalPath, runPath, uploadPath, summaryPath)
	assertNoURL(t, e, "/datasets/")
	if b := e.body("POST", "projects/proj/evaluations/eval-1/runs"); !reflect.DeepEqual(b, map[string]any{"source": "api"}) {
		t.Fatalf("inline run body %v", b)
	}
}

func TestInlineDatasetWithLibraryTool(t *testing.T) {
	e := newEnv(t, 1)
	e.route("GET", "projects/proj/ai-tools/lookup", resp{body: map[string]any{"version": 1}})
	args := baseArgs(handler("OpenAI", aiserver.ModeMessages, nil))
	args.Dataset = InlineRows{map[string]any{"input": "a"}}
	args.Tools = map[string]aiserver.ToolHandler{"lookup": noopTool}
	mustRun(t, e.module(), args)
	assertSequence(t, e, "GET projects/proj/ai-tools/lookup", evalPath, runPath, uploadPath, summaryPath)
}

func TestToolResolution(t *testing.T) {
	t.Run("404 names the tool and project and creates nothing", func(t *testing.T) {
		e := newEnv(t, 1)
		args := baseArgs(handler("OpenAI", aiserver.ModeMessages, nil))
		args.Tools = map[string]aiserver.ToolHandler{"lokup": noopTool}
		err := runErr(t, e.module(), args, ErrNotFound, `"lokup"`, `"proj"`)
		var apiErr *APIError
		if !errors.As(err, &apiErr) {
			t.Fatal("404 should wrap the APIError")
		}
		for _, s := range e.sequence() {
			if strings.HasPrefix(s, "POST") {
				t.Fatalf("POST after a tool 404: %v", e.sequence())
			}
		}
	})
	t.Run("missing version", func(t *testing.T) {
		e := newEnv(t, 1)
		e.route("GET", "projects/proj/ai-tools/t", resp{body: map[string]any{"schema": map[string]any{}}})
		args := baseArgs(handler("OpenAI", aiserver.ModeMessages, nil))
		args.Tools = map[string]aiserver.ToolHandler{"t": noopTool}
		runErr(t, e.module(), args, ErrInvalidResponse, `"t"`, "version")
	})
	t.Run("description and schema default", func(t *testing.T) {
		e := newEnv(t, 1)
		e.route("GET", "projects/proj/ai-tools/t", resp{body: map[string]any{"version": 2}})
		var rec recorder
		args := baseArgs(handler("OpenAI", aiserver.ModeMessages, rec.wrap(nil)))
		args.Tools = map[string]aiserver.ToolHandler{"t": noopTool}
		mustRun(t, e.module(), args)
		tool := rec.all()[0].Config.Tools["t"]
		if tool.Description != "" || tool.Parameters == nil || len(tool.Parameters) != 0 {
			t.Fatalf("defaults %+v", tool)
		}
	})
}

func TestDatasetResolutionAndPagination(t *testing.T) {
	t.Run("404", func(t *testing.T) {
		e := newEnv(t, 1)
		e.route("GET", "projects/proj/datasets/ds", resp{status: 404})
		runErr(t, e.module(), baseArgs(handler("OpenAI", aiserver.ModeMessages, nil)), ErrNotFound, `"ds"`, `"proj"`)
	})
	t.Run("missing id", func(t *testing.T) {
		e := newEnv(t, 1)
		e.route("GET", "projects/proj/datasets/ds", resp{body: map[string]any{"key": "ds"}})
		runErr(t, e.module(), baseArgs(handler("OpenAI", aiserver.ModeMessages, nil)), ErrInvalidResponse, `"id"`)
	})
	t.Run("pages of 200 until totalCount", func(t *testing.T) {
		e := newEnv(t, 0)
		page := func(from, n int) resp {
			items := make([]any, n)
			for i := range items {
				items[i] = map[string]any{"rowIndex": from + i, "input": "q"}
			}
			return resp{body: map[string]any{"items": items, "totalCount": 250}}
		}
		e.route("GET", "projects/proj/datasets/ds/rows", page(0, 200), page(200, 50))
		e.route("GET", "projects/proj/evaluations/eval-1/runs/run-1/summary", resp{body: map[string]any{"total": 250, "passed": 250}})
		mustRun(t, e.module(), baseArgs(handler("OpenAI", aiserver.ModeMessages, nil)))
		if e.count("datasets/ds/rows?mode=all&limit=200&offset=0") != 1 || e.count("datasets/ds/rows?mode=all&limit=200&offset=200") != 1 {
			t.Fatalf("pages: %v", e.sequence())
		}
		if n := len(e.ld.events(GenerationEventName)); n != 250 {
			t.Fatalf("generated %d rows", n)
		}
	})
	for name, body := range map[string]any{
		"items not a list":         map[string]any{"items": "x", "totalCount": 1},
		"totalCount not an int":    map[string]any{"items": []any{}, "totalCount": "1"},
		"empty dataset":            map[string]any{"items": []any{}, "totalCount": 0},
		"fewer rows than total":    map[string]any{"items": []any{}, "totalCount": 3},
		"row without a rowIndex":   map[string]any{"items": []any{map[string]any{"input": "x"}}, "totalCount": 1},
		"row index not an integer": map[string]any{"items": []any{map[string]any{"rowIndex": 1.5}}, "totalCount": 1},
	} {
		t.Run(name, func(t *testing.T) {
			e := newEnv(t, 1)
			e.route("GET", "projects/proj/datasets/ds/rows", resp{body: body})
			runErr(t, e.module(), baseArgs(handler("OpenAI", aiserver.ModeMessages, nil)), nil, "dataset")
			for _, s := range e.sequence() {
				if !strings.HasPrefix(s, "GET") {
					t.Fatalf("non-GET request %s", s)
				}
			}
		})
	}
}

func TestEvaluationCreateBody(t *testing.T) {
	libResp := resp{body: map[string]any{"version": 4}}
	cases := []struct {
		name   string
		mutate func(*RunArgs)
		want   map[string]any
	}{
		{"no tools", func(a *RunArgs) {}, map[string]any{
			"name": "eval-key", "generationProvider": "OpenAI", "generationModel": "gpt-4o", "messages": []any{}}},
		{"library only, with parameters and snippets", func(a *RunArgs) {
			a.Tools = map[string]aiserver.ToolHandler{"lib": noopTool}
			a.Generation.Parameters = map[string]any{"temperature": 0.2}
			a.Generation.PromptSnippets = map[string]string{"tone": "kind"}
			a.Generation.Messages = []aiserver.Message{{Role: "system", Text: "be {{snippet.tone}}"}}
		}, map[string]any{
			"name": "eval-key", "generationProvider": "OpenAI", "generationModel": "gpt-4o",
			"parameters": map[string]any{"temperature": 0.2}, "promptSnippets": map[string]any{"tone": "kind"},
			"messages": []any{map[string]any{"role": "system", "content": "be {{snippet.tone}}"}},
			"tools":    []any{map[string]any{"key": "lib", "version": 4.0, "source": "library"}}}},
		{"mixed, in key order, instructions wrapped", func(a *RunArgs) {
			a.Handler = handler("OpenAI", aiserver.ModeAgent, nil)
			a.Generation.Mode, a.Generation.Instructions = ModeAgent, "act"
			a.Tools = map[string]aiserver.ToolHandler{"lib": noopTool}
			a.InlineTools = map[string]InlineTool{"aaa": {Implementation: noopTool, Schema: map[string]any{"type": "object"}}}
		}, map[string]any{
			"name": "eval-key", "generationProvider": "OpenAI", "generationModel": "gpt-4o",
			"messages": []any{map[string]any{"role": "system", "content": "act"}},
			"tools": []any{
				map[string]any{"key": "aaa", "schema": map[string]any{"type": "object"}, "description": "", "source": "inline"},
				map[string]any{"key": "lib", "version": 4.0, "source": "library"}}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			e := newEnv(t, 1)
			e.route("GET", "projects/proj/ai-tools/lib", libResp)
			args := baseArgs(handler("OpenAI", aiserver.ModeMessages, nil))
			tc.mutate(&args)
			mustRun(t, e.module(), args)
			if got := e.body("POST", "projects/proj/evaluations"); !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("create body\n got: %s\nwant: %s", toJSON(got), toJSON(tc.want))
			}
		})
	}
}

func TestRecordResponsesAreValidated(t *testing.T) {
	cases := map[string]struct {
		path string
		body any
		want string
	}{
		"evaluation without id":      {"projects/proj/evaluations", map[string]any{"version": 1}, `"id"`},
		"evaluation with only an id": {"projects/proj/evaluations", map[string]any{"id": "eval-1"}, `"version"`},
		"run without id":             {"projects/proj/evaluations/eval-1/runs", map[string]any{"evaluationId": "eval-1", "state": "PENDING"}, `"id"`},
		"run without evaluationId":   {"projects/proj/evaluations/eval-1/runs", map[string]any{"id": "run-1", "state": "PENDING"}, `"evaluationId"`},
		"run without state":          {"projects/proj/evaluations/eval-1/runs", map[string]any{"id": "run-1", "evaluationId": "eval-1"}, `"state"`},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			e := newEnv(t, 1)
			e.route("POST", tc.path, resp{status: 201, body: tc.body})
			runErr(t, e.module(), baseArgs(handler("OpenAI", aiserver.ModeMessages, nil)), ErrInvalidResponse, tc.want)
		})
	}
}

func TestRunCreateBodyHosted(t *testing.T) {
	e := newEnv(t, 1)
	mustRun(t, e.module(), baseArgs(handler("OpenAI", aiserver.ModeMessages, nil)))
	if b := e.body("POST", "projects/proj/evaluations/eval-1/runs"); !reflect.DeepEqual(b, map[string]any{"source": "api", "datasetId": "ds-id"}) {
		t.Fatalf("hosted run body %v", b)
	}
}

func TestPathSegmentsAreEncoded(t *testing.T) {
	e := newEnv(t, 1)
	p, ev, run, ds, tool := "my proj", "ev/1", "run?1", "d s", "to#ol"
	enc := func(s string) string { return segment(s) }
	base := "projects/" + enc(p)
	e.route("GET", base+"/ai-tools/"+enc(tool), resp{body: map[string]any{"version": 1}})
	e.route("GET", base+"/datasets/"+enc(ds), resp{body: map[string]any{"id": "x"}})
	e.route("GET", base+"/datasets/"+enc(ds)+"/rows", resp{body: map[string]any{"items": []any{map[string]any{"rowIndex": 0, "input": "q"}}, "totalCount": 1}})
	e.route("POST", base+"/evaluations", resp{body: map[string]any{"id": ev, "version": 1}})
	runs := base + "/evaluations/" + enc(ev) + "/runs"
	e.route("POST", runs, resp{body: map[string]any{"id": run, "evaluationId": ev, "state": "PENDING"}})
	e.route("GET", runs+"/"+enc(run)+"/summary", resp{body: map[string]any{"total": 1, "passed": 1}})
	args := baseArgs(handler("OpenAI", aiserver.ModeMessages, nil))
	args.ProjectKey, args.Dataset = p, DatasetKey(ds)
	args.Tools = map[string]aiserver.ToolHandler{tool: noopTool}
	res := mustRun(t, e.module(), args)
	if !strings.HasSuffix(res.URL, "/projects/my%20proj/ai/evaluations/ev%2F1/runs/run%3F1") {
		t.Fatalf("url %s", res.URL)
	}
	// Inline upload and cancel paths too.
	e2 := newEnv(t, 1)
	e2.route("POST", base+"/evaluations", resp{body: map[string]any{"id": ev, "version": 1}})
	e2.route("POST", runs, resp{body: map[string]any{"id": run, "evaluationId": ev, "state": "PENDING"}})
	e2.route("POST", runs+"/"+enc(run)+"/dataset-rows", resp{status: 400})
	e2.route("POST", runs+"/"+enc(run)+"/cancel", resp{status: 200})
	args.Dataset, args.Tools = InlineRows{map[string]any{"input": "x"}}, nil
	runErr(t, e2.module(), args, ErrUploadFailed)
	if e2.count("run%3F1/dataset-rows") != 1 || e2.count("run%3F1/cancel") != 1 {
		t.Fatalf("encoded upload/cancel paths: %v", e2.sequence())
	}
}

func TestInlineUploadBodyAndRendering(t *testing.T) {
	e := newEnv(t, 2)
	var rec recorder
	args := baseArgs(handler("OpenAI", aiserver.ModeMessages, rec.wrap(nil)))
	args.Dataset = InlineRows{
		DatasetRow{Input: "Where is order {{order_id}}?", Variables: map[string]any{"order_id": "A-17"}},
		map[string]any{"input": "b", "expectedOutput": "B", "metadata": map[string]any{"m": 1}, "rowIdx": 1},
	}
	mustRun(t, e.module(), args)
	want := map[string]any{"rows": []any{
		map[string]any{"rowIdx": 0.0, "input": "Where is order {{order_id}}?", "expectedOutput": nil, "variables": map[string]any{"order_id": "A-17"}, "metadata": nil},
		map[string]any{"rowIdx": 1.0, "input": "b", "expectedOutput": "B", "variables": map[string]any{}, "metadata": map[string]any{"m": 1.0}},
	}}
	if got := e.body("POST", "projects/proj/evaluations/eval-1/runs/run-1/dataset-rows"); !reflect.DeepEqual(got, want) {
		t.Fatalf("upload body\n got %s\nwant %s", toJSON(got), toJSON(want))
	}
	var inputs []string
	for _, c := range rec.all() {
		inputs = append(inputs, c.UserInput)
		if c.UserInput == "Where is order A-17?" && (c.Variables["input"] != "Where is order A-17?" || c.Variables["expected_output"] != nil) {
			t.Fatalf("injected variables %v", c.Variables)
		}
	}
	slices.Sort(inputs)
	if !reflect.DeepEqual(inputs, []string{"Where is order A-17?", "b"}) {
		t.Fatalf("handler inputs %v", inputs)
	}
}

func TestInlineUploadBatches(t *testing.T) {
	e := newEnv(t, 1001)
	rows := make(InlineRows, 1001)
	for i := range rows {
		rows[i] = map[string]any{"input": fmt.Sprint(i)}
	}
	args := baseArgs(handler("OpenAI", aiserver.ModeMessages, nil))
	args.Dataset = rows
	mustRun(t, e.module(), args)
	var sizes []int
	var idx []int
	for _, r := range e.requests() {
		if strings.HasSuffix(r.Path, "/dataset-rows") {
			batch := r.Body.(map[string]any)["rows"].([]any)
			sizes = append(sizes, len(batch))
			for _, row := range batch {
				idx = append(idx, int(row.(map[string]any)["rowIdx"].(float64)))
			}
		}
	}
	if !reflect.DeepEqual(sizes, []int{500, 500, 1}) {
		t.Fatalf("batch sizes %v", sizes)
	}
	for i, v := range idx {
		if v != i {
			t.Fatalf("rowIdx %d at %d", v, i)
		}
	}
	log := e.logSnapshot()
	lastUpload := -1
	for i, s := range log {
		if strings.HasSuffix(s, "/dataset-rows") {
			lastUpload = i
		}
	}
	if first := slices.Index(log, "track "+GenerationEventName); first < lastUpload {
		t.Fatalf("a track (%d) preceded the last upload (%d)", first, lastUpload)
	}
}

func TestInlineUploadRetriedAsIdempotent(t *testing.T) {
	e := newEnv(t, 1)
	e.route("POST", "projects/proj/evaluations/eval-1/runs/run-1/dataset-rows", resp{status: 503}, resp{status: 200})
	args := baseArgs(handler("OpenAI", aiserver.ModeMessages, nil))
	args.Dataset = InlineRows{map[string]any{"input": "x"}}
	mustRun(t, e.module(), args)
	if e.count("/dataset-rows") != 2 {
		t.Fatalf("uploads: %v", e.sequence())
	}
}

func TestInlineUploadFailure(t *testing.T) {
	cases := []struct {
		name    string
		rows    int
		answers []resp
		cancel  []resp
		uploads int
		cancels int
		want    string
		warn    bool
	}{
		{"first batch", 1, []resp{{status: 400, body: "bad"}}, []resp{{status: 200}}, 1, 1, "rows 0-0 of 1", false},
		{"later batch", 501, []resp{{status: 200}, {status: 400}}, []resp{{status: 200}}, 2, 1, "rows 500-500 of 501", false},
		{"cancel fails too", 1, []resp{{status: 422}}, []resp{{status: 409, body: "already terminal"}}, 1, 1, "rows 0-0 of 1", true},
		// The cancel is an idempotent POST: a 5xx is retried before it counts.
		{"cancel 5xx is retried", 1, []resp{{status: 400, body: "bad"}}, []resp{{status: 503}, {status: 200}}, 1, 2, "rows 0-0 of 1", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			e := newEnv(t, tc.rows)
			e.route("POST", "projects/proj/evaluations/eval-1/runs/run-1/dataset-rows", tc.answers...)
			e.route("POST", "projects/proj/evaluations/eval-1/runs/run-1/cancel", tc.cancel...)
			var rec recorder
			args := baseArgs(handler("OpenAI", aiserver.ModeMessages, rec.wrap(nil)))
			rows := make(InlineRows, tc.rows)
			for i := range rows {
				rows[i] = map[string]any{"input": "x"}
			}
			args.Dataset = rows
			err := runErr(t, e.module(), args, ErrUploadFailed, tc.want, "run-1")
			var apiErr *APIError
			if !errors.As(err, &apiErr) || apiErr.Status != tc.answers[len(tc.answers)-1].status {
				t.Fatalf("must wrap the upload's APIError: %v", err)
			}
			if n := e.count("/cancel"); n != tc.cancels {
				t.Fatalf("cancel requests %d, want %d: %v", n, tc.cancels, e.sequence())
			}
			seq := e.sequence()
			if e.count("/dataset-rows") != tc.uploads || seq[len(seq)-1] != "POST projects/proj/evaluations/eval-1/runs/run-1/cancel" {
				t.Fatalf("sequence %v", seq)
			}
			if b := e.requests()[len(seq)-1]; b.Raw != nil && len(b.Raw) > 0 {
				t.Fatalf("cancel must have no body: %s", b.Raw)
			}
			if rec.n() != 0 || len(e.ld.tracks) != 0 || e.count("/summary") != 0 {
				t.Fatal("nothing may generate, track or poll after a failed upload")
			}
			if tc.warn != strings.Contains(e.logs.String(), "run-1") {
				t.Fatalf("cancel warning logged=%v: %s", !tc.warn, e.logs)
			}
		})
	}
}

func TestHostedRegressionEventID(t *testing.T) {
	e := newEnv(t, 1)
	mustRun(t, e.module(), baseArgs(handler("OpenAI", aiserver.ModeMessages, nil)))
	if e.count("dataset-rows") != 0 {
		t.Fatal("hosted run uploaded rows")
	}
	ev := e.ld.events(GenerationEventName)[0].Data
	// json.dumps(identity, sort_keys=True, separators=(",", ":")) in Python.
	if ev["eventId"] != "4de0c74ccd1dfc456a3f4024eca2a6cda2d80455eb80ddedd94146198b44c784" {
		t.Fatalf("eventId %v", ev["eventId"])
	}
}

func TestRunContextCancellation(t *testing.T) {
	e := newEnv(t, 20)
	ctx, cancel := context.WithCancel(context.Background())
	h := handler("OpenAI", aiserver.ModeMessages, func(ctx context.Context, req aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
		cancel()
		<-ctx.Done()
		return aiserver.HandlerResult{}, ctx.Err()
	})
	args := baseArgs(h)
	args.Concurrency = 3
	_, err := e.module().Run(ctx, args)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v", err)
	}
	if e.count("/summary") != 0 || e.ld.flushes != 1 {
		t.Fatalf("after cancel: %v, flushes %d", e.sequence(), e.ld.flushes)
	}
}

// TestNoGoroutineLeak runs a cancelled run and a completed run and waits,
// under a deadline, for the goroutine count to return to its baseline.
func TestNoGoroutineLeak(t *testing.T) {
	before := runtime.NumGoroutine()
	e := newEnv(t, 50)
	ctx, cancel := context.WithCancel(context.Background())
	block := handler("OpenAI", aiserver.ModeMessages, func(ctx context.Context, req aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
		cancel()
		<-ctx.Done()
		return aiserver.HandlerResult{}, ctx.Err()
	})
	args := baseArgs(block)
	args.Criteria = []Criterion{mustScorer(t, "s", constScorer(1), nil)}
	if _, err := e.module().Run(ctx, args); !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v", err)
	}
	args.Handler = handler("OpenAI", aiserver.ModeMessages, nil)
	mustRun(t, newEnv(t, 50).module(), args)
	deadline := time.Now().Add(2 * time.Second)
	for runtime.NumGoroutine() > before {
		if time.Now().After(deadline) {
			t.Fatalf("goroutines: %d before, %d after", before, runtime.NumGoroutine())
		}
		time.Sleep(10 * time.Millisecond)
	}
}
