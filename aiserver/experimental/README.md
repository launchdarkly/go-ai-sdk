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
| `LD_PROJECT_KEY` | yes, unless `EvaluationsOptions.ProjectKey` | The project holding the evaluations, runs, tools, datasets and judges. Checked first. |
| `LD_API_TOKEN` | yes, unless `EvaluationsOptions.APIToken` | LaunchDarkly API access token; authorizes every management API request. Checked second. |
| `LD_SDK_KEY` | yes, unless `EvaluationsOptions.SDKKey` or a client is already initialized (`aiserver.InitClient` / `InitWithClient`) | Server-side SDK key for the client that sends result events. When a client is already installed it is used and this key is ignored (with a warning). |

Handlers read their provider's own variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, AWS chain).
A blank value counts as unset. A missing project key or credential fails `InitEvaluations` with `ErrEvalMissingCredential`, before any network I/O.

### What leaves the process on spans

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

### Example

```go
evals, err := experimental.InitEvaluations(experimental.EvaluationsOptions{}) // LD_PROJECT_KEY, LD_API_TOKEN, LD_SDK_KEY
if err != nil { log.Fatal(err) }
judge, _ := experimental.NewJudge("accuracy", nil)
res, err := evals.Run(ctx, experimental.EvaluationRunArgs{
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
| `Tools` / `InlineTools` | Library tools (resolved and version-pinned by key) and tools defined in code (`InlineTool{Implementation, Schema, Description}`). Every key, library or inline, is non-blank and lowercase, and keys must differ under Unicode case folding (`straße` and `strasse` collide). |
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
| `EvaluationsOptions` | `ProjectKey` (default `LD_PROJECT_KEY`), `APIToken` (default `LD_API_TOKEN`), `SDKKey` (default `LD_SDK_KEY`), `HTTPClient` (default `http.DefaultClient`), `RequestTimeout` (30s per attempt), `MaxAttempts` (4), `Logger` |
| `EvalRunResult` | `Passed` (true iff the final summary has no error and no failed rows), `URL` (the run in the LaunchDarkly UI), `RunID`, `Summary` |
| `RunSummary` | `TotalRows`, `PassedRows`, `FailedRows`, `ErrorRows`, `PendingRows` |

Errors are sentinels for `errors.Is`: `ErrEvalInvalidArgument` (validation, before any I/O),
`ErrEvalMissingCredential`, `ErrEvalNoClient`, `ErrEvalAPI` (non-2xx; `*APIError` carries the
response), `ErrEvalTransport`, `ErrEvalInvalidResponse`, `ErrEvalNotFound`, `ErrEvalInvalidConfig`,
`ErrEvalJudgeResolution`, `ErrEvalUploadFailed`, `ErrEvalPollTimeout`.

## Shadow requests

A shadow request runs a candidate variation on real traffic without serving it. Use
`ShadowConfig` in place of `aiserver.Config` at the call sites you want to shadow. It takes the
same `aiserver.ConfigArgs` plus `ShadowOptions`, and its `Invoke` and `Stream` return exactly what
`Config`'s return. When the context matches a targeting rule or the fallthrough that has shadows
attached, each sampled shadow runs the same input on its own variation once the served call
succeeds. Its response is discarded. Its events are the normal `$ld:ai:*` events for that
variation, with the served call's `runId` and `isShadow: true`, so you can compare the two on the
usual AI metrics. A shadow never delays or changes the served response, and its failures are only
tracked.

```go
shadowed, err := experimental.ShadowConfig(
	aiserver.ConfigArgs{Key: "my-ai-config", Handlers: []*aiserver.ProviderHandler{openai, anthropic}},
	experimental.ShadowOptions{},
)
if err != nil { log.Fatal(err) }
resp, err := shadowed.Invoke(ctx, "Summarize this document", userCtx)

// Or hand each shadow to your own queue, and run it from a worker.
// OnTask runs before Invoke returns, so it must never block.
queue := make(chan experimental.ShadowTask, 100)
shadowed, err = experimental.ShadowConfig(args, experimental.ShadowOptions{
	OnTask: func(task experimental.ShadowTask) {
		select {
		case queue <- task:
		default: // queue full: drop this shadow
		}
	},
})
result := experimental.RunShadow(ctx, <-queue, handlers) // never fails; see result.Error
```

| Option | Meaning |
| --- | --- |
| `Timeout` | How long each shadow may run before it is tracked as an error with `Error: "timeout"`. Zero means 30 seconds; a negative value makes `ShadowConfig` return an error. |
| `OnTask` | Run shadows yourself. It receives a `ShadowTask` per sampled shadow before `Invoke` returns, so it must not block: only enqueue it. A `ShadowTask` survives `encoding/json`, including unmodeled variation fields, and carries the caller's conversation id. `RunShadow` uses the client already set up and never starts one. |

Shadows run in a background goroutine by default, detached from the caller's cancellation, and
each one ends at its timeout. `aiserver.Shutdown` cancels any that are still running, and a
cancelled shadow records nothing. A shadow always runs as
a plain invoke, even when the served call streamed, with no tools and no judges. Pass handlers for
every provider your shadow variations use, since a shadow with no matching handler is skipped.
Shadows need a client with `JSONVariationDetail`, which `*ld.LDClient` has; with any other client
the served call still runs and no shadow does. Plain `aiserver.Config` never runs shadows.
