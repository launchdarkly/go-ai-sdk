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
res, err := evals.Run(ctx, evaluations.RunOptions{
	Key:        "support-bot-" + time.Now().Format("20060102-150405"), // unique per run
	Dataset:    evaluations.DatasetKey("support-questions"),
	Handler:    openaimessages.NewHandler(openaimessages.Options{}),
	Generation: &evaluations.GenerationConfig{
		Provider: "OpenAI", Model: "gpt-4o",
		Mode:     evaluations.GenerationModeCompletion, // required: "completion" or "agent"
		Messages: []aiserver.Message{{Role: "system", Text: "You are a support agent."}},
	},
	Criteria: []evaluations.Criterion{judge},
})
if err != nil { log.Fatal(err) }
fmt.Println(res.URL)
if !res.Passed { os.Exit(1) }
```

| Argument | Meaning |
| --- | --- |
| `Dataset` | `DatasetKey("...")` for a hosted dataset, or `InlineRows{...}` (`DatasetRow` values or `map[string]any` with `input`, `expectedOutput`, `variables`, `metadata`, optional `rowIdx`). |
| `Handler` / `Handlers` | Exactly one. Each config (generation and every judge) picks its handler by provider and mode; an exact provider beats `*`; there is no mode fallback. |
| `Generation` / `AIConfig` | Exactly one. `AIConfig: &AIConfigRef{Key, Variation}` reads the latest variation version; `Overrides` replace single fields. |
| `Tools` / `InlineTools` | Library tools (resolved and version-pinned by key) and tools defined in code (`EvalTool{Implementation, Schema, Description}`). Every key, library or inline, is non-blank and lowercase, and keys must differ under Unicode case folding (`straße` and `strasse` collide). |
| `Criteria` | `*Judge` (`NewJudge`) and `*Scorer` (`NewScorer`). With `AIConfig`, nil uses the variation's judges and `[]Criterion{}` runs none. |
| `Concurrency` | In-flight handler, judge and scorer calls (default 10). |
| `PollInterval` / `PollTimeout` | Summary polling (default 2s / 3m). Large datasets need a longer timeout. |

The SDK never computes a verdict: it reports scores, and LaunchDarkly rules on them against the
thresholds pinned on the evaluation. `Passed` is true only when no row errored or failed.
Management API requests go to `APIBaseURI`, then `LD_API_BASE_URI`, then `https://app.launchdarkly.com`.
The run link uses `UIBaseURI`, then `LD_UI_BASE_URI`, then the same default. At either level a
blank value counts as unset, and a trailing `/` is stripped. Both hosts are fixed at
`InitEvaluations`.

## API reference

```go
func InitEvaluations(opts InitEvaluationsOptions) (*EvaluationsModule, error)
func (m *EvaluationsModule) Run(ctx context.Context, args RunOptions) (*EvalRunResult, error)
func NewJudge(key string, opts *JudgeOptions) (*Judge, error)
func NewScorer(name string, fn ScorerFunc, opts *ScorerOptions) (*Scorer, error)
```

| Type | Key fields |
| --- | --- |
| `InitEvaluationsOptions` | `ProjectKey` (default `LD_PROJECT_KEY`), `APIToken` (default `LD_API_TOKEN`), `SDKKey` (default `LD_SDK_KEY`), `HTTPClient` (default `http.DefaultClient`), `RequestTimeout` (30s per attempt), `MaxAttempts` (4), `Logger`, `APIBaseURI` (default `LD_API_BASE_URI`, then `https://app.launchdarkly.com`), `UIBaseURI` (default `LD_UI_BASE_URI`, then `https://app.launchdarkly.com`) |
| `EvalRunResult` | `Passed` (true iff the final summary has no error and no failed rows), `URL` (the run in the LaunchDarkly UI), `RunID`, `Summary` |
| `RunSummary` | `TotalRows`, `PassedRows`, `FailedRows`, `ErrorRows`, `PendingRows` |

Errors are sentinels for `errors.Is`: `ErrInvalidArgument` (validation, before any I/O),
`ErrMissingCredential`, `ErrNoClient`, `ErrAPI` (non-2xx; `*APIError` carries the
response), `ErrTransport`, `ErrInvalidResponse`, `ErrNotFound`, `ErrInvalidConfig`,
`ErrJudgeResolution`, `ErrUploadFailed`, `ErrPollTimeout`.
