package experimental_test

import (
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// evaluationsSurface is the documented evaluations surface of the
// experimental entry point.
var evaluationsSurface = []string{
	"AIConfigRef", "APIError", "Criterion", "Dataset", "DatasetKey", "DatasetRow", "DefaultJudgeThreshold",
	"DefaultPassRateThreshold", "DefaultScorerThreshold", "ErrEvalAPI", "ErrEvalInvalidArgument",
	"ErrEvalInvalidConfig", "ErrEvalInvalidResponse", "ErrEvalJudgeResolution", "ErrEvalMissingCredential",
	"ErrEvalNoClient", "ErrEvalNotFound", "ErrEvalPollTimeout", "ErrEvalTransport", "ErrEvalUploadFailed",
	"EvalRunResult", "EvaluationRunArgs", "Evaluations", "EvaluationsOptions", "GenerationConfig",
	"GenerationModeAgent", "GenerationModeCompletion", "GenerationOverrides", "HigherIsBetter", "InitEvaluations",
	"InlineRows", "InlineTool", "Judge", "JudgeOptions", "LowerIsBetter", "NewJudge", "NewScorer", "RunSummary",
	"Scorer", "ScorerFunc", "ScorerOptions", "SuccessDirection",
}

func TestEvaluationsSurfaceIsExactlyTheDocumentedSet(t *testing.T) {
	got := exportedDecls(t, "evaluations.go")
	want := slices.Sorted(slices.Values(evaluationsSurface))
	if !slices.Equal(got, want) {
		t.Fatalf("experimental evaluations surface:\n got %v\nwant %v", got, want)
	}
	root := exportedDecls(t, coreFiles(t)...)
	for _, name := range evaluationsSurface {
		if slices.Contains(root, name) {
			t.Errorf("%s leaked onto the aiserver package root", name)
		}
	}
	// The whole experimental entry point is skills plus evaluations; a new
	// file's names must be pinned too.
	all := exportedDecls(t, experimentalFiles(t)...)
	pinned := slices.Sorted(slices.Values(append(slices.Clone(skillsSurface), evaluationsSurface...)))
	if !slices.Equal(all, pinned) {
		t.Errorf("experimental entry point has unpinned names:\n got %v\nwant %v", all, pinned)
	}
}

func experimentalFiles(t *testing.T) []string {
	t.Helper()
	all, _ := filepath.Glob("*.go")
	var out []string
	for _, f := range all {
		if !strings.HasSuffix(f, "_test.go") {
			out = append(out, f)
		}
	}
	return out
}
