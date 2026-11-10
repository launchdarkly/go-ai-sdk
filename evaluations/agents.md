# evaluations — contributor and agent guide

Evaluations from Code, imported as `github.com/launchdarkly/go-ai-sdk/evaluations`. Not a handler:
no `providesFor`. It consumes handlers (`RunOptions.Handler` / `Handlers`) and selects one per config
by provider and mode. Its exported names are pinned in `surface_test.go`.

| File | Responsibility |
| --- | --- |
| `module.go` | `InitEvaluationsOptions`, `EvaluationsModule`, `InitEvaluations` (project key, token, SDK key, in that order; reports `client.initEvaluations` before any check), client resolution, `Run`. |
| `api.go` | Management API client: host resolved at init (never `LD_BASE_URI`), headers, retry policy (429 always; 5xx/transport only for GET/HEAD and idempotent POSTs), `Retry-After`, injectable sleep/jitter. `APIError`. |
| `types.go` | Public types, sentinel errors, criterion constructors (`NewJudge`, `NewScorer`) and wire shapes. |
| `tools.go` | `ToolsClient` (`EvaluationsModule.Tools`): `Get` validates the key and implementation, reads `ai-tools/<key>` and returns a library `EvalTool` with its pinned version and project. The only tool request. |
| `validate.go` | Every pre-I/O check: pool, generation mode/coverage/prompt, tools (key, implementation, inline schema or library project, case-folded collisions, in list order), inline rows (strict JSON), criteria. Shared `selectHandler`. |
| `steps.go` | Management API steps, in order: AI Config reads (variation tool warnings), dataset resolution, record creation, inline upload/cancel, summary polling, run verdict wait. No tool request. |
| `runner.go` | Validation entry, orchestration, bounded worker pool, generation phase (trajectory recorder per row), judge resolution and invocation, scorers. |
| `spans.go` | Row, criterion, and fallback `chat` / `execute_tool` spans; content on them follows the capture setting. |
| `events.go` | Canonical JSON / `eventId` (byte-identical to Python), payload building, flush-bounded emitter with per-event failure isolation. |
| `*_test.go` | One file per feature area; `harness_test.go` holds the recording transport, stub LD client and seams; `surface_test.go` pins the exported names and calls them from outside the package. |

Seams (package vars, replaced by tests): `getClient`, `initClient`,
`resolveJudgeConfig` (flag delivery via `internal/bridge.ExtractVariationRaw`),
`defaultPollInterval`, `defaultPollTimeout`; `EvaluationsModule.flushBound`; `apiClient.sleep/random/now`.

Rules that are easy to break: `Run` issues no tool request (library tools come pinned from `Tools().Get`); only `Get` can make a library `EvalTool`, and `Run` rejects one whose `Key`, `Schema` or `Description` no longer match what `Get` pinned; judges never get tools; the judge config is passed unrendered;
`message_history` comes only from `internal/judging.BuildMessageHistory`; a row that failed
generation gets no criterion event; the final flush is deferred; an inline run omits
`datasetId`/`datasetKey` everywhere; the default thresholds, event names and hosts stay unexported.
Check: `gofmt -l evaluations && go vet ./evaluations/ && go test -race ./evaluations/`.

```go
// evaluations exports (matches `go doc -short ./evaluations`)
type EvaluationsModule           // methods: Run(ctx, RunOptions) (*EvalRunResult, error), SDKKey() string, Tools() *ToolsClient
type ToolsClient                 // method: Get(ctx, key string, impl aiserver.ToolHandler) (EvalTool, error)
type EvalTool                    // methods: Source() ToolSource, Version() int, ProjectKey() string
type InitEvaluationsOptions RunOptions GenerationConfig GenerationMode GenerationOverrides AIConfigRef ToolSource
type DatasetRow Dataset DatasetKey InlineRows Criterion Judge JudgeOptions Scorer ScorerOptions
type ScorerFunc SuccessDirection EvalRunResult RunSummary CriterionSummary Verdict APIError
const GenerationModeCompletion GenerationModeAgent ToolSourceInline ToolSourceLibrary HigherIsBetter LowerIsBetter VerdictPass VerdictFail
var ErrInvalidArgument ErrMissingCredential ErrNoClient ErrAPI ErrTransport
var ErrInvalidResponse ErrNotFound ErrInvalidConfig ErrJudgeResolution
var ErrUploadFailed ErrPollTimeout
func InitEvaluations(opts InitEvaluationsOptions) (*EvaluationsModule, error)
func NewJudge(key string, opts *JudgeOptions) (*Judge, error)
func NewScorer(name string, fn ScorerFunc, opts *ScorerOptions) (*Scorer, error)
```

**OTel.** Evaluations configures no tracer provider. The provider is registered by
`aiserver.InitClient` / `InitWithClient` (see `../aiserver/agents.md`); generation and judge calls go
through the handlers, which emit their usual spans under it. The runner also writes a
`launchdarkly.dataset_row` trace per row, an `evaluate {criterion}` span per criterion, and `chat` /
`execute_tool` spans for a handler that opens none (`spans.go`). Results reach LaunchDarkly as SDK
custom events, not spans.

**`InitClient` — when to call it.** `InitEvaluations` resolves the project key and credentials only;
it does no I/O. `Tools().Get` calls the management API without a client. The client that emits
result events is resolved when `Run` starts, before its first management API request
(`EvaluationsModule.resolveClient`), on one of three paths:

- **Lazy (default)** — no client installed: `Run` calls `aiserver.InitClient` with the resolved SDK key
  (`InitEvaluationsOptions.SDKKey`, else `LD_SDK_KEY`).
- **Options** — call `aiserver.InitClient(ctx, InitOptions{...})` first to set base URIs, OTLP endpoint
  or service name; `Run` reuses that client.
- **BYOC** — call `aiserver.InitWithClient(ctx, ldClient, opts)` first; `LD_SDK_KEY` may then be
  omitted. If a client is installed *and* an SDK key was resolved, the installed client wins and a
  warning is logged: both must point at the project under evaluation.

| Dependency | Why |
| --- | --- |
| `aiserver` | `ProviderHandler`, `ToolHandler`, `ReportUsage`, client lifecycle (`LDClient.TrackMetric` / `Flush` emit results). |
| `internal/bridge` | Flag delivery for judge configs (`ExtractVariationRaw`) and the shared logger. |
| `internal/judging` | `BuildMessageHistory` and judge prompt helpers shared with core judges. |
| `internal/trajectory` | Per-row trajectory recorder fed to judges. |
| `go-sdk-common/v3` (`ldcontext`, `ldvalue`) | Event contexts and payload values. |
| `golang.org/x/text` | Unicode case folding for the tool key collision check. |
| standard library | Management API client (`net/http`), retries, canonical JSON, hashing. |
