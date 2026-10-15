package experimental

import (
	"github.com/launchdarkly/go-ai-sdk/internal/evaluations"
)

// Evaluations from Code. Experimental: every name in
// this file may change in a minor release.
//
// A run reads a dataset (hosted in LaunchDarkly or inline), generates each
// row with a caller-supplied handler, scores every generated row against
// judges and scorers, reports each result to LaunchDarkly as a custom event,
// and polls the run summary for the verdict:
//
//	evals, err := experimental.InitEvaluations(experimental.EvaluationsOptions{})
//	res, err := evals.Run(ctx, experimental.EvaluationRunArgs{
//		ProjectKey: "my-project",
//		Key:        "support-bot-" + time.Now().Format("20060102-150405"),
//		Dataset:    experimental.DatasetKey("support-questions"),
//		Handler:    openaimessages.NewHandler(openaimessages.Options{}),
//		Generation: &experimental.GenerationConfig{Provider: "OpenAI", Model: "gpt-4o",
//			Mode: experimental.GenerationModeCompletion},
//	})
//	if err == nil && !res.Passed { os.Exit(1) }

// Evaluations runs evaluations from code. Create it with InitEvaluations.
// Experimental.
type Evaluations = evaluations.Module

// EvaluationsOptions configure InitEvaluations: the API access token
// (default LD_API_TOKEN), the SDK key (default LD_SDK_KEY), and the HTTP
// client, timeout, attempts and logger of the management API client. The
// API and UI hosts are fixed. Experimental.
type EvaluationsOptions = evaluations.Options

// EvaluationRunArgs are the arguments to Evaluations.Run. Experimental.
type EvaluationRunArgs = evaluations.RunArgs

// GenerationConfig holds generation settings written in code. Mode is
// required: GenerationModeCompletion or GenerationModeAgent. Experimental.
type GenerationConfig = evaluations.GenerationConfig

// GenerationOverrides replace fields of the settings read from an AI Config
// variation. Experimental.
type GenerationOverrides = evaluations.GenerationOverrides

// AIConfigRef names an AI Config and variation whose latest version seeds
// a run. Experimental.
type AIConfigRef = evaluations.AIConfigRef

// InlineTool defines a tool in code: implementation, schema, description.
// Experimental.
type InlineTool = evaluations.InlineTool

// DatasetRow is the public dataset row type a scorer receives, and one form
// of an inline row. Experimental.
type DatasetRow = evaluations.DatasetRow

// Dataset is a DatasetKey or InlineRows. Experimental.
type Dataset = evaluations.Dataset

// DatasetKey is the key of a dataset hosted in LaunchDarkly. Experimental.
type DatasetKey = evaluations.DatasetKey

// InlineRows is a dataset defined in code: DatasetRow values or maps in the
// upload wire shape. Experimental.
type InlineRows = evaluations.InlineRows

// Criterion is a *Judge or a *Scorer. Experimental.
type Criterion = evaluations.Criterion

// Judge references a LaunchDarkly AI Judge config. Experimental.
type Judge = evaluations.Judge

// JudgeOptions are a judge reference's optional thresholds and ground-truth
// context. Experimental.
type JudgeOptions = evaluations.JudgeOptions

// Scorer is a local deterministic criterion. Experimental.
type Scorer = evaluations.Scorer

// ScorerOptions are a scorer's optional thresholds and success direction.
// Experimental.
type ScorerOptions = evaluations.ScorerOptions

// ScorerFunc scores one generated output with a finite number in 0–1.
// Experimental.
type ScorerFunc = evaluations.ScorerFunc

// SuccessDirection is which way a scorer's score points. Experimental.
type SuccessDirection = evaluations.SuccessDirection

// EvalRunResult is a run's outcome. Experimental.
type EvalRunResult = evaluations.EvalRunResult

// RunSummary are a run's row counts. Experimental.
type RunSummary = evaluations.RunSummary

// APIError is a non-2xx management API response. Experimental.
type APIError = evaluations.APIError

// Generation modes, scorer directions and default thresholds. Experimental.
const (
	GenerationModeCompletion = evaluations.ModeCompletion
	GenerationModeAgent      = evaluations.ModeAgent

	HigherIsBetter = evaluations.HigherIsBetter
	LowerIsBetter  = evaluations.LowerIsBetter

	DefaultJudgeThreshold    = evaluations.DefaultJudgeThreshold
	DefaultScorerThreshold   = evaluations.DefaultScorerThreshold
	DefaultPassRateThreshold = evaluations.DefaultPassRateThreshold
)

// Evaluation sentinel errors, for errors.Is. Experimental.
var (
	ErrEvalInvalidArgument   = evaluations.ErrInvalidArgument
	ErrEvalMissingCredential = evaluations.ErrMissingCredential
	ErrEvalNoClient          = evaluations.ErrNoClient
	ErrEvalAPI               = evaluations.ErrAPI
	ErrEvalTransport         = evaluations.ErrTransport
	ErrEvalInvalidResponse   = evaluations.ErrInvalidResponse
	ErrEvalNotFound          = evaluations.ErrNotFound
	ErrEvalInvalidConfig     = evaluations.ErrInvalidConfig
	ErrEvalJudgeResolution   = evaluations.ErrJudgeResolution
	ErrEvalUploadFailed      = evaluations.ErrUploadFailed
	ErrEvalPollTimeout       = evaluations.ErrPollTimeout
)

// InitEvaluations resolves both credentials and returns the evaluations
// module. It performs no network I/O. Experimental.
func InitEvaluations(opts EvaluationsOptions) (*Evaluations, error) { return evaluations.Init(opts) }

// NewJudge validates a judge reference. Thresholds default to 0.5 (row) and
// 0.95 (pass rate). Experimental.
func NewJudge(key string, opts *JudgeOptions) (*Judge, error) { return evaluations.NewJudge(key, opts) }

// NewScorer validates a scorer. Thresholds default to 1.0 (row) and 0.95
// (pass rate); the direction defaults to HigherIsBetter. Experimental.
func NewScorer(name string, fn ScorerFunc, opts *ScorerOptions) (*Scorer, error) {
	return evaluations.NewScorer(name, fn, opts)
}
