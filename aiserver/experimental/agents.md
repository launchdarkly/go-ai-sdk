# experimental — contributor and agent guide

Experimental entry point. The files here are thin: type aliases and
one-line functions over `internal/<feature>`, each documented as Experimental. A feature promoted
to core moves to `aiserver` and leaves this package.

<!-- Each feature owns one section below. Append; do not rewrite another feature's section. -->

## Evaluations from Code

Tier 0 (core client, experimental entry point). Not a handler: no `providesFor`. It consumes
handlers (`EvaluationRunArgs.Handler` / `Handlers`) and selects one per config by provider and mode.
All logic and tests live in `internal/evaluations`; `evaluations.go` here is aliases plus three
one-line constructors.

| File | Responsibility |
| --- | --- |
| `evaluations.go` | Public aliases (`Evaluations`, `EvaluationRunArgs`, `GenerationConfig`, `InlineTool`, `Judge`, `Scorer`, …), `InitEvaluations`, `NewJudge`, `NewScorer`, `ErrEval*` sentinels. Sentinels carry an `Eval` prefix because `aiserver.ErrInvalidArgument` belongs to Agent Skills. |
| `../../internal/evaluations/api.go` | Management API client: fixed host, headers, retry policy (429 always; 5xx/transport only for GET/HEAD and idempotent POSTs), `Retry-After`, injectable sleep/jitter. |
| `.../types.go` | Public types, sentinel errors, criterion constructors and wire shapes. |
| `.../validate.go` | Every pre-I/O check: pool, generation mode/coverage/prompt, tools, inline rows (strict JSON), criteria. Shared `selectHandler`. |
| `.../steps.go` | Management API steps, in order: AI Config reads, tool and dataset resolution, record creation, inline upload/cancel, summary polling. |
| `.../runner.go` | Validation entry, orchestration, bounded worker pool, generation phase (trajectory recorder per row), judge resolution and invocation, scorers. |
| `.../events.go` | Canonical JSON / `eventId` (byte-identical to Python), payload building, flush-bounded emitter with per-event failure isolation. |
| `.../*_test.go` | One file per feature area; `harness_test.go` holds the recording transport, stub LD client and seams. |

Seams (package vars in `internal/evaluations`, replaced by tests): `getClient`, `initClient`,
`resolveJudgeConfig` (flag delivery via `internal/bridge.ExtractVariationRaw`),
`defaultPollInterval`, `defaultPollTimeout`; `Module.flushBound`; `apiClient.sleep/random/now`.

Rules that are easy to break: judges never get tools; the judge config is passed unrendered;
`message_history` comes only from `internal/judging.BuildMessageHistory`; a row that failed
generation gets no criterion event; the final flush is deferred; an inline run omits
`datasetId`/`datasetKey` everywhere. Check: `gofmt -l internal/evaluations && go vet ./internal/evaluations/ && go test -race ./internal/evaluations/`.

```go
// evaluations exports (see evaluations.go; matches `go doc -short ./aiserver/experimental`)
type Evaluations = evaluations.Module          // method: Run(ctx, EvaluationRunArgs) (*EvalRunResult, error)
type EvaluationsOptions EvaluationRunArgs GenerationConfig GenerationOverrides AIConfigRef InlineTool
type DatasetRow Dataset DatasetKey InlineRows Criterion Judge JudgeOptions Scorer ScorerOptions
type ScorerFunc SuccessDirection EvalRunResult RunSummary APIError
const GenerationModeCompletion GenerationModeAgent HigherIsBetter LowerIsBetter
const DefaultJudgeThreshold DefaultScorerThreshold DefaultPassRateThreshold
var ErrEvalInvalidArgument ErrEvalMissingCredential ErrEvalNoClient ErrEvalAPI ErrEvalTransport
var ErrEvalInvalidResponse ErrEvalNotFound ErrEvalInvalidConfig ErrEvalJudgeResolution
var ErrEvalUploadFailed ErrEvalPollTimeout
func InitEvaluations(opts EvaluationsOptions) (*Evaluations, error)
func NewJudge(key string, opts *JudgeOptions) (*Judge, error)
func NewScorer(name string, fn ScorerFunc, opts *ScorerOptions) (*Scorer, error)
```

**OTel.** Evaluations configures no tracer provider. The provider is registered by
`aiserver.InitClient` / `InitWithClient` (see `../agents.md`); generation and judge calls go through
the handlers, which emit their usual spans under it. Results reach LaunchDarkly as SDK custom events,
not spans.

**`InitClient` — when to call it.** `InitEvaluations` resolves credentials only; it does no I/O. The
client that emits result events is resolved before the first management API request
(`Module.resolveClient`), on one of three paths:

- **Lazy (default)** — no client installed: `Run` calls `aiserver.InitClient` with the resolved SDK key
  (`EvaluationsOptions.SDKKey`, else `LD_SDK_KEY`).
- **Options** — call `aiserver.InitClient(ctx, InitOptions{...})` first to set base URIs, OTLP endpoint
  or service name; `Run` reuses that client.
- **BYOC** — call `aiserver.InitWithClient(ctx, ldClient, opts)` first; `LD_SDK_KEY` may then be
  omitted. If a client is installed *and* an SDK key was resolved, the installed client wins and a
  warning is logged: both must point at the project under evaluation.

| Dependency | Why |
| --- | --- |
| `aiserver` | `ProviderHandler`, `ToolHandler`, `ReportUsage`, client lifecycle (`LDClient.TrackMetric` / `Flush` emit results). |
| `internal/bridge` | Flag delivery for judge configs (`ExtractVariationRaw`) and the shared logger, without core naming an experimental type. |
| `internal/judging` | `BuildMessageHistory` and judge prompt helpers shared with core judges. |
| `internal/trajectory` | Per-row trajectory recorder fed to judges. |
| `go-sdk-common/v3` (`ldcontext`, `ldvalue`) | Event contexts and payload values. |
| standard library | Management API client (`net/http`), retries, canonical JSON, hashing. |
