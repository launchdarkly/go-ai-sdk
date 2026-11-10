package evaluations_test

import (
	"context"
	"errors"
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/launchdarkly/go-ai-sdk/evaluations"
)

// surface pins the exported names of package evaluations. A new or removed
// name fails here, so it shows in review.
var surface = []string{
	"AIConfigRef", "APIError", "Criterion", "CriterionSummary", "Dataset", "DatasetKey", "DatasetRow", "ErrAPI", "ErrInvalidArgument",
	"ErrInvalidConfig", "ErrInvalidResponse", "ErrJudgeResolution", "ErrMissingCredential", "ErrNoClient",
	"ErrNotFound", "ErrPollTimeout", "ErrTransport", "ErrUploadFailed", "EvalRunResult", "EvalTool",
	"EvaluationsModule", "GenerationConfig", "GenerationMode", "GenerationModeAgent", "GenerationModeCompletion",
	"GenerationOverrides", "HigherIsBetter", "InitEvaluations", "InitEvaluationsOptions", "InlineRows", "Judge",
	"JudgeOptions", "LowerIsBetter", "NewJudge", "NewScorer", "RunOptions", "RunSummary", "Scorer", "ScorerFunc",
	"ScorerOptions", "SuccessDirection", "ToolSource", "ToolSourceInline", "ToolSourceLibrary", "ToolsClient",
	"EvalVerdict", "EvalVerdictFail", "EvalVerdictPass",
}

// parseDir parses the non-test .go files matching glob.
func parseDir(t *testing.T, glob string) []*ast.File {
	t.Helper()
	files, _ := filepath.Glob(glob)
	fset := token.NewFileSet()
	var out []*ast.File
	for _, file := range files {
		if strings.HasSuffix(file, "_test.go") {
			continue
		}
		f, err := parser.ParseFile(fset, file, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, f)
	}
	if len(out) == 0 {
		t.Fatalf("no files match %s", glob)
	}
	return out
}

func exportedNames(t *testing.T, glob string) []string {
	t.Helper()
	var out []string
	for _, f := range parseDir(t, glob) {
		for name := range f.Scope.Objects {
			if ast.IsExported(name) {
				out = append(out, name)
			}
		}
	}
	slices.Sort(out)
	return out
}

func TestSurfaceIsPinned(t *testing.T) {
	got := exportedNames(t, "*.go")
	if want := slices.Sorted(slices.Values(surface)); !slices.Equal(got, want) {
		t.Fatalf("evaluations surface:\n got %v\nwant %v", got, want)
	}
	// The former package-level defaults, event names, host constants and usage ID stay
	// internal. (APIBaseURI / UIBaseURI are now InitEvaluationsOptions fields.)
	for _, name := range []string{"DefaultJudgeThreshold", "DefaultScorerThreshold", "DefaultPassRateThreshold",
		"GenerationEventName", "CriterionEventName", "APIBaseURI", "UIBaseURI", "HelperName"} {
		if slices.Contains(got, name) {
			t.Errorf("%s must not be exported", name)
		}
	}
}

// Evaluations is its own package. Neither the aiserver root nor the
// experimental entry point imports it, so neither can re-export its names.
func TestNotReexportedFromAiserver(t *testing.T) {
	for _, glob := range []string{"../aiserver/*.go", "../aiserver/experimental/*.go"} {
		for _, f := range parseDir(t, glob) {
			for _, imp := range f.Imports {
				if p, _ := strconv.Unquote(imp.Path.Value); p == "github.com/launchdarkly/go-ai-sdk/evaluations" {
					t.Errorf("%s imports %s", glob, p)
				}
			}
		}
	}
	// The experimental entry point keeps no evaluations alias.
	if slices.Contains(exportedNames(t, "../aiserver/experimental/*.go"), "InitEvaluations") {
		t.Error("aiserver/experimental still exports InitEvaluations")
	}
}

// Core never imports experimental code.
func TestDoesNotImportExperimentalCode(t *testing.T) {
	for _, f := range parseDir(t, "*.go") {
		for _, imp := range f.Imports {
			if p, _ := strconv.Unquote(imp.Path.Value); strings.Contains(p, "/experimental") {
				t.Errorf("evaluations imports %s", p)
			}
		}
	}
}

// The public names are callable from outside the package. InitEvaluations
// does no network I/O.
func TestPublicNamesAreCallable(t *testing.T) {
	t.Setenv("LD_PROJECT_KEY", "")
	t.Setenv("LD_API_TOKEN", "")
	if _, err := evaluations.InitEvaluations(evaluations.InitEvaluationsOptions{ProjectKey: " "}); !errors.Is(err, evaluations.ErrMissingCredential) ||
		!strings.Contains(err.Error(), "LD_PROJECT_KEY") {
		t.Fatalf("no project key: %v", err)
	}
	if _, err := evaluations.InitEvaluations(evaluations.InitEvaluationsOptions{ProjectKey: "proj"}); !errors.Is(err, evaluations.ErrMissingCredential) ||
		!strings.Contains(err.Error(), "LD_API_TOKEN") {
		t.Fatalf("no API token: %v", err)
	}
	evals, err := evaluations.InitEvaluations(evaluations.InitEvaluationsOptions{ProjectKey: "proj", APIToken: "api-test", SDKKey: "sdk-test"})
	if err != nil || evals == nil || evals.SDKKey() != "sdk-test" {
		t.Fatalf("InitEvaluations: %v %v", evals, err)
	}
	if evals.Tools() == nil || (evaluations.EvalTool{Key: "t"}).Source() != evaluations.ToolSourceInline {
		t.Fatal("Tools() must return a client, and a literal EvalTool must be inline")
	}
	judge, err := evaluations.NewJudge("accuracy", nil)
	if err != nil || judge.Key() != "accuracy" {
		t.Fatalf("NewJudge: %v %v", judge, err)
	}
	scorer, err := evaluations.NewScorer("exact", func(context.Context, evaluations.DatasetRow, any) (float64, error) {
		return 1, nil
	}, nil)
	if err != nil || scorer.Name() != "exact" {
		t.Fatalf("NewScorer: %v %v", scorer, err)
	}
	gen := evaluations.GenerationConfig{Provider: "OpenAI", Model: "m", Mode: evaluations.GenerationModeAgent}
	if criteria := []evaluations.Criterion{judge, scorer}; len(criteria) != 2 || gen.Mode != "agent" {
		t.Fatal("judge and scorer must both be criteria, and the mode constant typed")
	}
}
