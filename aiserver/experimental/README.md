# experimental

Experimental entry point of the LaunchDarkly AI SDK for Go
(`github.com/launchdarkly/go-ai-sdk/aiserver/experimental`). Every name here is published
but may change in a minor release; none is exported from `aiserver`. Agent Skills is part of the
core client: see [`aiserver`](../README.md#agent-skills).

<!-- Each feature owns one section below. Append; do not rewrite another feature's section. -->

## Evaluations from Code

Runs a dataset through your own handler, scores each output with LaunchDarkly AI Judges
and/or local scorers, reports every result to LaunchDarkly as custom events, and waits for
the run verdict. Use it as a CI gate.

```bash
go get github.com/launchdarkly/go-ai-sdk/aiserver/experimental
go get github.com/launchdarkly/go-ai-sdk/openaimessages   # or any handler package for the generation step
```

### Environment variables

| Variable | Required | Description |
| --- | --- | --- |
| `LD_API_TOKEN` | yes, unless `EvaluationsOptions.APIToken` | LaunchDarkly API access token; authorizes every management API request. Checked first. |
| `LD_SDK_KEY` | yes, unless `EvaluationsOptions.SDKKey` or a client is already initialized (`aiserver.InitClient` / `InitWithClient`) | Server-side SDK key for the client that sends result events. When a client is already installed it is used and this key is ignored (with a warning). |

Handlers read their provider's own variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, AWS chain).
Missing credentials fail `InitEvaluations` with `ErrEvalMissingCredential`, before any network I/O.

### Example

```go
evals, err := experimental.InitEvaluations(experimental.EvaluationsOptions{}) // LD_API_TOKEN, LD_SDK_KEY
if err != nil { log.Fatal(err) }
judge, _ := experimental.NewJudge("accuracy", nil)
res, err := evals.Run(ctx, experimental.EvaluationRunArgs{
	ProjectKey: "my-project",
	Key:        "support-bot-" + time.Now().Format("20060102-150405"), // unique per run
	Dataset:    experimental.DatasetKey("support-questions"),
	Handler:    openaimessages.NewHandler(openaimessages.Options{}),
	Generation: &experimental.GenerationConfig{
		Provider: "OpenAI", Model: "gpt-4o",
		Mode:     experimental.GenerationModeCompletion, // required: "completion" or "agent"
		Messages: []aiserver.Message{{Role: "system", Text: "You are a support agent."}},
	},
	Criteria: []experimental.Criterion{judge},
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
| `Tools` / `InlineTools` | Library tools (resolved and version-pinned by key) and tools defined in code (`InlineTool{Implementation, Schema, Description}`). Inline keys are lowercase. |
| `Criteria` | `*Judge` (`NewJudge`) and `*Scorer` (`NewScorer`). With `AIConfig`, nil uses the variation's judges and `[]Criterion{}` runs none. |
| `Concurrency` | In-flight handler, judge and scorer calls (default 10). |
| `PollInterval` / `PollTimeout` | Summary polling (default 2s / 3m). Large datasets need a longer timeout. |

The SDK never computes a verdict: it reports scores, and LaunchDarkly rules on them against the
thresholds pinned on the evaluation. `Passed` is true only when no row errored or failed.
The management API host is fixed at `https://app.launchdarkly.com`.

### API reference

```go
func InitEvaluations(opts EvaluationsOptions) (*Evaluations, error)
func (m *Evaluations) Run(ctx context.Context, args EvaluationRunArgs) (*EvalRunResult, error)
func NewJudge(key string, opts *JudgeOptions) (*Judge, error)
func NewScorer(name string, fn ScorerFunc, opts *ScorerOptions) (*Scorer, error)
```

| Type | Key fields |
| --- | --- |
| `EvaluationsOptions` | `APIToken` (default `LD_API_TOKEN`), `SDKKey` (default `LD_SDK_KEY`), `HTTPClient` (default `http.DefaultClient`), `RequestTimeout` (30s per attempt), `MaxAttempts` (4), `Logger` |
| `EvalRunResult` | `Passed` (true iff the final summary has no error and no failed rows), `URL` (the run in the LaunchDarkly UI), `RunID`, `Summary` |
| `RunSummary` | `TotalRows`, `PassedRows`, `FailedRows`, `ErrorRows`, `PendingRows` |

Errors are sentinels for `errors.Is`: `ErrEvalInvalidArgument` (validation, before any I/O),
`ErrEvalMissingCredential`, `ErrEvalNoClient`, `ErrEvalAPI` (non-2xx; `*APIError` carries the
response), `ErrEvalTransport`, `ErrEvalInvalidResponse`, `ErrEvalNotFound`, `ErrEvalInvalidConfig`,
`ErrEvalJudgeResolution`, `ErrEvalUploadFailed`, `ErrEvalPollTimeout`.
