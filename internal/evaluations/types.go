// Package evaluations implements Evaluations from Code: it
// runs a dataset through caller-supplied handlers, scores the outputs against
// judges and scorers, reports every result to LaunchDarkly as custom events,
// and polls the run summary for a verdict.
//
// Experimental. The public surface is re-exported from
// github.com/launchdarkly/go-ai-sdk/aiserver/experimental; names may change
// in a minor release.
package evaluations

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

// Sentinel errors; every error the package returns wraps one of them.
var (
	// ErrInvalidArgument: a run argument failed validation before any I/O.
	ErrInvalidArgument = errors.New("invalid argument")
	// ErrMissingCredential: Init could not resolve LD_API_TOKEN or LD_SDK_KEY.
	ErrMissingCredential = errors.New("missing credential")
	// ErrNoClient: no LaunchDarkly client is available to emit result events.
	ErrNoClient = errors.New("no LaunchDarkly client")
	// ErrAPI: the management API answered with a non-2xx status (see APIError).
	ErrAPI = errors.New("management API error")
	// ErrTransport: a management API request failed without a response.
	ErrTransport = errors.New("management API transport failure")
	// ErrInvalidResponse: a management API response had an unexpected shape.
	ErrInvalidResponse = errors.New("invalid management API response")
	// ErrNotFound: a referenced AI Config, variation, model config, tool or
	// dataset does not exist in the project.
	ErrNotFound = errors.New("not found")
	// ErrInvalidConfig: an AI Config read from LaunchDarkly cannot be run
	// with the supplied handlers or tools.
	ErrInvalidConfig = errors.New("invalid AI Config for this run")
	// ErrJudgeResolution: a judge could not be resolved or has no handler.
	ErrJudgeResolution = errors.New("judge resolution failed")
	// ErrUploadFailed: an inline dataset batch did not upload; the run was cancelled.
	ErrUploadFailed = errors.New("inline dataset upload failed")
	// ErrPollTimeout: the run summary never became terminal.
	ErrPollTimeout = errors.New("summary polling timed out")
)

func invalid(format string, args ...any) error {
	return fmt.Errorf("evaluations: %w: "+format, append([]any{ErrInvalidArgument}, args...)...)
}

// Generation modes accepted by GenerationConfig.Mode, as the LaunchDarkly UI
// shows them.
const (
	ModeCompletion = "completion"
	ModeAgent      = "agent"
)

// GenerationConfig holds the generation settings written in code.
type GenerationConfig struct {
	// Provider names the generation provider (for example "OpenAI"). Required.
	Provider string
	// Model is the provider's model ID. Required.
	Model string
	// Mode is required and has no default. Choose "completion" when the AI
	// Config uses a list of messages and the handler sends one request to a
	// chat or messages API. Choose "agent" when the AI Config uses one
	// instructions block and an agent framework runs the model, for example
	// OpenAI Agents, Claude Agents, or LangChain agents. The value must match
	// the mode of a handler in the pool; when unsure, use the AI Config's mode
	// in LaunchDarkly. Mode selects the handler only and is never sent to
	// LaunchDarkly.
	Mode string
	// Parameters are model parameters; nil when none.
	Parameters map[string]any
	// Instructions is the agent prompt. Mutually exclusive with Messages.
	Instructions string
	// Messages is the completion prompt. Mutually exclusive with Instructions.
	Messages []aiserver.Message
	// PromptSnippets fill {{snippet.<name>}} placeholders at config-build time.
	PromptSnippets map[string]string
	// OutputFormat is a JSON Schema forwarded to the handler config.
	OutputFormat map[string]any
}

// GenerationOverrides replace single fields of the settings read from an AI
// Config variation. There is no Mode: the AI Config supplies it. Setting
// Instructions or Messages replaces both fetched prompt fields; Parameters
// merge key by key.
type GenerationOverrides struct {
	Provider       string
	Model          string
	Parameters     map[string]any
	Instructions   string
	Messages       []aiserver.Message
	PromptSnippets map[string]string
	OutputFormat   map[string]any
}

// AIConfigRef identifies an existing AI Config variation whose latest
// version seeds the run's generation settings.
type AIConfigRef struct {
	// Key is the AI Config key.
	Key string
	// Variation is the variation key.
	Variation string
}

// InlineTool defines a tool in code: the executable plus the schema and
// description the model is shown. The schema is copied when the run is
// validated, so later changes to the caller's map do not reach the run.
type InlineTool struct {
	// Implementation is the aiserver.ToolFunc the handler calls. A
	// NativeTool is not allowed: a native tool has no schema to define.
	Implementation aiserver.ToolHandler
	// Schema is the tool's JSON Schema; required, {} for a no-argument tool.
	Schema map[string]any
	// Description defaults to "".
	Description string
}

// DatasetRow is the public dataset row type: what a scorer receives, and
// one of the two forms of an inline row. A zero RowIndex on an inline row
// means "use the position"; any other value must equal the position.
type DatasetRow struct {
	RowIndex       int
	Input          string
	ExpectedOutput string
	Variables      map[string]any
	Metadata       map[string]any
}

// Dataset is the run's row source: a DatasetKey or InlineRows.
type Dataset interface{ isDataset() }

// DatasetKey is the key of a dataset hosted in LaunchDarkly.
type DatasetKey string

func (DatasetKey) isDataset() {}

// InlineRows is a dataset defined in code. Each element is a DatasetRow
// (or *DatasetRow) or a map[string]any in the upload wire shape: input,
// expectedOutput, variables, metadata and an optional rowIdx.
type InlineRows []any

func (InlineRows) isDataset() {}

// SuccessDirection is which way a scorer's score points.
type SuccessDirection string

// Scorer success directions.
const (
	HigherIsBetter SuccessDirection = "higher_is_better"
	LowerIsBetter  SuccessDirection = "lower_is_better"
)

// Default thresholds.
const (
	DefaultJudgeThreshold    = 0.5
	DefaultScorerThreshold   = 1.0
	DefaultPassRateThreshold = 0.95
)

// Criterion is a Judge or a Scorer.
type Criterion interface {
	criterionType() string
	validate() error
	wire() map[string]any
}

// JudgeOptions are the optional fields of a judge reference. A nil
// threshold takes its default.
type JudgeOptions struct {
	Threshold          *float64
	PassRateThreshold  *float64
	GroundTruthContext string
}

// Judge references a LaunchDarkly AI Judge config. Its success direction is
// the judge's own (isInverted) and cannot be set here.
type Judge struct {
	key                string
	threshold          float64
	passRate           float64
	groundTruthContext string
}

// NewJudge validates and returns a judge reference.
func NewJudge(key string, opts *JudgeOptions) (*Judge, error) {
	if opts == nil {
		opts = &JudgeOptions{}
	}
	j := &Judge{key: key, threshold: orDefault(opts.Threshold, DefaultJudgeThreshold),
		passRate: orDefault(opts.PassRateThreshold, DefaultPassRateThreshold), groundTruthContext: opts.GroundTruthContext}
	if err := j.validate(); err != nil {
		return nil, err
	}
	return j, nil
}

// Key is the judge config key, which is also its criterion identity.
func (j *Judge) Key() string { return j.key }

func (j *Judge) criterionType() string { return j.key }

func (j *Judge) validate() error {
	if strings.TrimSpace(j.key) == "" {
		return invalid("judge key must not be blank")
	}
	return validateThresholds(j.threshold, j.passRate)
}

func (j *Judge) wire() map[string]any {
	opts := map[string]any{"threshold": j.threshold, "passRateThreshold": j.passRate}
	if j.groundTruthContext != "" {
		opts["groundTruthContext"] = j.groundTruthContext
	}
	return map[string]any{"criterionType": j.key, "kind": "judge", "judgeKey": j.key, "options": opts}
}

// ScorerFunc scores one generated output. It must return a finite number in
// 0–1; an error or panic becomes a scorer_raised result.
type ScorerFunc func(ctx context.Context, row DatasetRow, output any) (float64, error)

// ScorerOptions are the optional fields of a scorer. A nil threshold takes
// its default; an empty direction is HigherIsBetter.
type ScorerOptions struct {
	Threshold         *float64
	PassRateThreshold *float64
	SuccessDirection  SuccessDirection
}

// Scorer is a local deterministic criterion; it makes no model call.
type Scorer struct {
	name      string
	fn        ScorerFunc
	threshold float64
	passRate  float64
	direction SuccessDirection
}

// NewScorer validates and returns a scorer.
func NewScorer(name string, fn ScorerFunc, opts *ScorerOptions) (*Scorer, error) {
	if opts == nil {
		opts = &ScorerOptions{}
	}
	dir := opts.SuccessDirection
	if dir == "" {
		dir = HigherIsBetter
	}
	s := &Scorer{name: name, fn: fn, threshold: orDefault(opts.Threshold, DefaultScorerThreshold),
		passRate: orDefault(opts.PassRateThreshold, DefaultPassRateThreshold), direction: dir}
	if err := s.validate(); err != nil {
		return nil, err
	}
	return s, nil
}

// Name is the scorer's name, which is also its criterion identity.
func (s *Scorer) Name() string { return s.name }

func (s *Scorer) criterionType() string { return s.name }

func (s *Scorer) validate() error {
	if strings.TrimSpace(s.name) == "" {
		return invalid("scorer name must not be blank")
	}
	if s.fn == nil {
		return invalid("scorer %q fn must be callable", s.name)
	}
	if s.direction != HigherIsBetter && s.direction != LowerIsBetter {
		return invalid("scorer %q success direction must be %q or %q, got %q", s.name, HigherIsBetter, LowerIsBetter, s.direction)
	}
	return validateThresholds(s.threshold, s.passRate)
}

func (s *Scorer) wire() map[string]any {
	return map[string]any{"criterionType": s.name, "kind": "scorer", "successDirection": string(s.direction),
		"options": map[string]any{"threshold": s.threshold, "passRateThreshold": s.passRate}}
}

func orDefault(p *float64, def float64) float64 {
	if p == nil {
		return def
	}
	return *p
}

// validateThresholds rejects NaN and ±Inf explicitly: NaN compares false
// against both range bounds.
func validateThresholds(threshold, passRate float64) error {
	for _, f := range []struct {
		name string
		v    float64
	}{{"threshold", threshold}, {"passRateThreshold", passRate}} {
		if math.IsNaN(f.v) || math.IsInf(f.v, 0) || f.v < 0 || f.v > 1 {
			return invalid("%s must be a finite number between 0 and 1, got %v", f.name, f.v)
		}
	}
	return nil
}

// RunArgs are the arguments to Module.Run.
type RunArgs struct {
	// ProjectKey is the project holding the evaluation, tools and dataset.
	ProjectKey string
	// Key names the evaluation Run creates. Every Run creates a new
	// evaluation, so the key must be unique per run (e.g. timestamp-suffixed).
	Key string
	// Dataset is a DatasetKey or InlineRows. Required.
	Dataset Dataset
	// Handler and Handlers supply the pool; set exactly one. Handler h is
	// the same as Handlers []{h}. Every config selects its handler by
	// provider and mode.
	Handler  *aiserver.ProviderHandler
	Handlers []*aiserver.ProviderHandler
	// Generation (settings in code) or AIConfig (read from LaunchDarkly):
	// set exactly one. Overrides apply only with AIConfig.
	Generation *GenerationConfig
	AIConfig   *AIConfigRef
	Overrides  *GenerationOverrides
	// Tools are library tools, resolved by key against LaunchDarkly.
	// InlineTools are defined in code. Leaving both nil with AIConfig uses
	// the variation's tools; a non-nil empty map means "no tools".
	Tools       map[string]aiserver.ToolHandler
	InlineTools map[string]InlineTool
	// Criteria are judges and scorers. nil with AIConfig uses the
	// variation's judges; a non-nil empty slice runs generation only.
	Criteria []Criterion
	// Concurrency bounds in-flight handler, judge and scorer calls. 0 means 10.
	Concurrency int
	// PollInterval and PollTimeout govern summary polling. 0 means the
	// default (2s and 3m).
	PollInterval time.Duration
	PollTimeout  time.Duration
}

// RunSummary are the run's row counts.
type RunSummary struct {
	TotalRows   int
	PassedRows  int
	FailedRows  int
	ErrorRows   int
	PendingRows int
}

// EvalRunResult is the outcome of a run.
type EvalRunResult struct {
	// Passed is true iff the terminal summary has no error and no failed rows.
	Passed  bool
	URL     string
	RunID   string
	Summary RunSummary
}
