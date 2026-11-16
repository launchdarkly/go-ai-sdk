# evaluations

Evaluations from Code for the LaunchDarkly AI SDK for Go
(`github.com/launchdarkly/go-ai-sdk/evaluations`).

Runs a dataset through your own handler, scores each output with LaunchDarkly AI Judges
and/or local scorers, reports every result to LaunchDarkly as custom events, and waits for
the run verdict. Use it as a CI gate.

```bash
go get github.com/launchdarkly/go-ai-sdk/evaluations
go get github.com/launchdarkly/go-ai-sdk/openaimessages   # or any handler package for the generation step
```

## Environment variables

| Variable | Required | Description |
| --- | --- | --- |
| `LD_PROJECT_KEY` | yes, unless `InitEvaluationsOptions.ProjectKey` | The project holding the evaluations, runs, tools, datasets and judges. Checked first. |
| `LD_API_TOKEN` | yes, unless `InitEvaluationsOptions.APIToken` | LaunchDarkly API access token; authorizes every management API request. Checked second. |
| `LD_SDK_KEY` | yes, unless `InitEvaluationsOptions.SDKKey` or a client is already initialized (`aiserver.InitClient` / `InitWithClient`) | Server-side SDK key for the client that sends result events. When a client is already installed it is used and this key is ignored (with a warning). |
| `LD_API_BASE_URI` | no | Management API host; `InitEvaluationsOptions.APIBaseURI` wins over it. Default `https://app.launchdarkly.com`. `LD_BASE_URI` is never used: it configures flag delivery and may point at a relay proxy. |
| `LD_UI_BASE_URI` | no | Host of the run link in the result; `InitEvaluationsOptions.UIBaseURI` wins over it. It carries no API traffic and is not derived from the API host. Default `https://app.launchdarkly.com`. |

Handlers read their provider's own variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, AWS chain).
A blank value counts as unset. A missing project key or credential fails `InitEvaluations` with `ErrMissingCredential`, before any network I/O.

## What leaves the process on spans

`Run` also exports OpenTelemetry spans through the tracer provider `aiserver.InitClient`
registered: one `launchdarkly.dataset_row` trace per row, an `evaluate {criterion}` span per
criterion, and, for a handler that opens no spans of its own, `chat` and `execute_tool` spans
the runner writes. Every span carries the row's `gen_ai.conversation.id`,
`launchdarkly.evaluation_run.id` and `launchdarkly.evaluation_run.row_index`.

**By default those spans carry the transcript:** the row input and the config's messages, the
system and judge instructions rendered with the row's variables, the model output, tool
arguments and results, the judge's raw completion, and its reason
(`gen_ai.evaluation.explanation`). Setting capture off removes all of it, on evaluation spans as
on every other span: `CaptureContent: aiserver.CaptureOff` on the handler, or
`aiserver.Configure(aiserver.SDKOptions{CaptureContent: aiserver.CaptureOff})` before the first
handler call. Model, provider, usage, status, tool names and scores are written either way. The
`$ld:ai:offline-evals:*` result events never carry any of this content.

## Example

```go
evals, err := evaluations.InitEvaluations(evaluations.InitEvaluationsOptions{}) // LD_PROJECT_KEY, LD_API_TOKEN, LD_SDK_KEY
if err != nil { log.Fatal(err) }
judge, _ := evaluations.NewJudge("accuracy", nil)
// A library tool: read and pinned now, so Run issues no tool request.
lookup, err := evals.Tools().Get(ctx, "lookup-order", aiserver.ToolFunc(lookupOrder))
if err != nil { log.Fatal(err) }
handler, err := openaimessages.NewHandler(openaimessages.Options{})
if err != nil { log.Fatal(err) }
res, err := evals.Run(ctx, evaluations.RunOptions{
	Key:        "support-qa", // stable; a later Run reuses this evaluation
	Dataset:    evaluations.DatasetKey("support-questions"),
	Handler:    handler,
	Generation: &evaluations.GenerationConfig{
		Provider: "OpenAI", Model: "gpt-4o",
		Mode:     evaluations.GenerationModeCompletion, // required: "completion" or "agent"
		Messages: []aiserver.Message{{Role: "system", Text: "You are a support agent."}},
	},
	Tools:    []evaluations.EvalTool{lookup},
	Criteria: []evaluations.Criterion{judge},
})
if err != nil { log.Fatal(err) }
fmt.Println(res.URL)
if !res.Passed { os.Exit(1) }
```

| Argument | Meaning |
| --- | --- |
| `Key` | Stable offline-evaluation key: `^[a-z0-9][a-z0-9._-]*$`, at most 128 characters, checked before any network I/O. `Run` GETs `…/offline-evaluations/{key}`, POSTs (`key` and `name`) when the evaluation is missing, or PATCHes a full config snapshot when it exists, then creates a new run. PATCH omits `name`. The result `URL` stays the run link. Requires the offline-evaluations API. |
| `Dataset` | `DatasetKey("...")` for a hosted dataset, or `InlineRows{...}` (`map[string]any` values with `input`, `expectedOutput`, `variables`, `metadata`, optional `rowIdx`; a `DatasetRow` is rejected). |
| `Handler` / `Handlers` | Exactly one. Each config (generation and every judge) picks its handler by provider and mode; an exact provider beats `*`; there is no mode fallback. |
| `Generation` / `AIConfig` | Exactly one. `AIConfig: &AIConfigRef{Key, Variation}` reads the latest variation version; `Overrides` replace single fields. |
| `Tools` | One list of `EvalTool`. Define a tool in code with a literal (`EvalTool{Key, Implementation, Schema, Description}`; `Schema` is required, `{}` is valid). Use a library tool with `evals.Tools().Get(ctx, key, impl)`, which reads it and pins its version before `Run`; `Run` itself issues no tool request. Every key is non-blank and lowercase, may appear once, and must differ from every other under Unicode case folding (`straße` and `strasse` collide). A library tool must come from this module's project, and its `Key`, `Schema` and `Description` must stay as `Get` returned them (an edited copy fails `Run`; define an inline tool instead). With `AIConfig`, nil `Tools` fails when the variation lists tools, and `[]EvalTool{}` runs with no tools. |
| `Criteria` | `*Judge` (`NewJudge`) and `*Scorer` (`NewScorer`). With `AIConfig`, nil uses the variation's judges and `[]Criterion{}` runs none. |
| `Concurrency` | In-flight handler, judge and scorer calls (default 10). |
| `PollInterval` / `PollTimeout` | Summary polling (default 2s / 3m). Large datasets need a longer timeout. |

The SDK never computes a verdict: it reports scores, and LaunchDarkly rules on them against the
thresholds pinned on the evaluation. Once the run's rows are all accounted for and the summary
reports a criterion with a pass rate threshold, `Run` reads the run's `Verdict` (waiting up to 30s
for it to be recorded). A run fails when any criterion's pass rate (passed rows over all rows, so
errored rows count against it) is below its threshold. `Passed` follows the verdict when there is
one. Without one, `Passed` is true only when no row errored or failed. Every criterion sends a pass
rate threshold, 0.95 by default, so a run that sets none passes when 95% of each criterion's rows
pass. Set `PassRateThreshold: new(1.0)` in every criterion's `JudgeOptions` or `ScorerOptions` to
require every row to pass. Where
LaunchDarkly does not serve verdicts for the project, `Verdict` is `""` and `Passed` uses the row
counts.
Management API requests go to `APIBaseURI`, then `LD_API_BASE_URI`, then `https://app.launchdarkly.com`.
The run link uses `UIBaseURI`, then `LD_UI_BASE_URI`, then the same default. At either level a
blank value counts as unset, and a trailing `/` is stripped. Both hosts are resolved once, at
`InitEvaluations`.

## API reference

```go
func InitEvaluations(opts InitEvaluationsOptions) (*EvaluationsModule, error)
func (m *EvaluationsModule) Run(ctx context.Context, args RunOptions) (*EvalRunResult, error)
func (m *EvaluationsModule) Tools() *ToolsClient
func (c *ToolsClient) Get(ctx context.Context, key string, impl aiserver.ToolHandler) (EvalTool, error)
func NewJudge(key string, opts *JudgeOptions) (*Judge, error)
func NewScorer(name string, fn ScorerFunc, opts *ScorerOptions) (*Scorer, error)
```

| Type | Key fields |
| --- | --- |
| `InitEvaluationsOptions` | `ProjectKey` (default `LD_PROJECT_KEY`), `APIToken` (default `LD_API_TOKEN`), `SDKKey` (default `LD_SDK_KEY`), `HTTPClient` (default `http.DefaultClient`), `RequestTimeout` (30s per attempt), `MaxAttempts` (4), `Logger`, `APIBaseURI` (default `LD_API_BASE_URI`, then `https://app.launchdarkly.com`), `UIBaseURI` (default `LD_UI_BASE_URI`, then `https://app.launchdarkly.com`) |
| `EvalRunResult` | `Passed` (follows `Verdict`; without one, true iff the final summary has no error and no failed rows), `URL` (the run in the LaunchDarkly UI), `RunID`, `Summary`, `Verdict` |
| `RunSummary` | `TotalRows`, `PassedRows`, `FailedRows`, `ErrorRows`, `PendingRows`, `Criteria` |
| `CriterionSummary` | `CriterionType`, `PassRateThreshold` (`*float64`), `PassedRows`, `FailedRows`, `ErrorRows`, `PassRate` (`*float64`), `PassRateVerdict` |
| `EvalVerdict` | `EvalVerdictPass`, `EvalVerdictFail`, or `""` (none recorded, not served, or a value this release does not recognise) |

Errors are sentinels for `errors.Is`: `ErrInvalidArgument` (validation, before any I/O),
`ErrMissingCredential`, `ErrNoClient`, `ErrAPI` (non-2xx; `*APIError` carries the
response), `ErrTransport`, `ErrInvalidResponse`, `ErrNotFound`, `ErrInvalidConfig`,
`ErrJudgeResolution`, `ErrUploadFailed`, `ErrPollTimeout`.
