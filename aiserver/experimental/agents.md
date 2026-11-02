# experimental — contributor and agent guide

Experimental entry point. The files here are thin: type aliases and
one-line functions over `internal/<feature>`, each documented as Experimental.

<!-- Each feature owns one section below. Append; do not rewrite another feature's section. -->

## Evaluations from Code

Tier 0 (core client, experimental entry point). Not a handler: no `providesFor`. It consumes
handlers (`EvaluationRunArgs.Handler` / `Handlers`) and selects one per config by provider and mode.
All logic and tests live in `internal/evaluations`; `evaluations.go` here is aliases plus three
one-line constructors.

| File | Responsibility |
| --- | --- |
| `evaluations.go` | Public aliases (`Evaluations`, `EvaluationRunArgs`, `GenerationConfig`, `InlineTool`, `Judge`, `Scorer`, …), `InitEvaluations`, `NewJudge`, `NewScorer`, `ErrEval*` sentinels. Sentinels carry an `Eval` prefix because `ErrInvalidArgument` is taken by skills. |
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

## Agent Skills

Tier 0 (core client, experimental entry point). Not a handler: no `providesFor`. All logic and tests
live in `internal/skills`; `skills.go` here is aliases plus one-line wrappers. Skills report no
`$ld:ai:sdk:usage` helper: they have no usage ID, and JS and Python send none. `InitEvaluations` reports
`experimental.initEvaluations`. Add an ID only once it is shared with the JS and Python SDKs.

| File | Responsibility |
| --- | --- |
| `skills.go` | Public aliases, constants, sentinels, wrappers. Its exported set is pinned by `skills_test.go`. |
| `../../internal/skills/types.go` | Immutable value types (unexported fields + getters), closed sets, on-disk constants. |
| `.../store.go` | `SkillStore` seam and optional halves, `Listener`, module state, `SetSkillStore`, `ClearState` (registered with `bridge.OnShutdown`), readiness probe. |
| `.../telemetry.go` | The only place signals and integrity log records are built (three signals, ten reason codes). |
| `.../verify.go` | `verifyRawSkill` (shape → size → encoding → hash), `resolveFromStore` (the one retrieval path: key check, then version check), `newestByKey`. |
| `.../accessors.go`, `.../memstore.go` | `SkillRefs`, `GetSkill*`, `AllSkills`; `InMemorySkillStore` and the shared `objectSet`. |
| `.../writeskills.go` | Options, the availability gate, request resolution (`"*"` enumerates raw objects; never `AllSkills`). |
| `.../materialize_unix.go` | Root validation, manifest load/rewrite, write/adopt/update, prune. `materialize_other.go` refuses non-POSIX. |
| `.../safefs_unix.go` | Descriptor-pinned primitives on `golang.org/x/sys/unix` and the `fsOps` interception hooks. |
| `.../fdv2_reader.go`, `.../fdv2_store.go` | Pure FDv2 protocol reader; transport (HTTP, SSE, backoff, lifecycle). Standard library only. |
| `.../watch.go` | Watcher goroutine: trailing debounce, context cancellation, detach on close. |

```go
// skills exports (see skills.go)
SkillReference SkillKey SkillRequest SkillInput Skill SkillFields SkillOutcome SkillOutcomeReason
ReconcileAction ReconcileActionKind ReconcileReport OnUnavailable SkillStore ListenerStore
ListenerRemover InitializationProber Listener InMemorySkillStore LaunchDarklySkillStore
LaunchDarklySkillStoreOptions LaunchDarklySkillStoreMode StoreDiagnostics WriteSkillsOptions
WatchSkillsOptions SkillWatcher SkillFilename ManifestFilename ManifestVersion Reason* Action*
OnUnavailableKeep OnUnavailableRaise LaunchDarklySkillStoreModeStream LaunchDarklySkillStoreModePoll
ErrNoSkillStore ErrInvalidArgument ErrSkillsUnavailable ErrStoreNotWatchable ErrUnsupportedKind
ErrStoreClosed ErrUnsupportedPlatform NewSkillReference NewSkill NewSkillOutcome NewListener
NewInMemorySkillStore NewLaunchDarklySkillStore SetSkillStore SkillRefs GetSkill GetSkillResult
GetSkills AllSkills WriteSkills WriteAllSkills WatchSkills WatchAllSkills
```

**Lifecycle.** `InitClient` is not required for skills; a store is configured with `SetSkillStore`
only (never an `InitOptions` field).
`aiserver.Shutdown` clears the store and emitter through the bridge hook; it does not close a
`LaunchDarklySkillStore`.

**OTel.** Skills emit no spans and never call `client.track`, not even `$ld:ai:sdk:usage`. The
tracer provider is set up by `aiserver.InitClient` (see `../agents.md`).

**Security posture (non-relaxable).** The root is pinned once per reconcile; every read and write
under it is `*at()`-relative, so swapping a directory for a symlink cannot redirect anything. The
manifest read and the adoption comparison read share one no-follow, non-blocking, `fstat`-checked,
`len+1`-bounded primitive (`readRegularAt`). Only manifest-listed paths under a matching key are
overwritten or pruned; a corrupt manifest or incomplete retrieval suppresses all deletion. Content is
re-verified immediately before writing. The SDK never parses `Skill.Content`; do not add a reader.

**Telemetry seam.** Private emitter (`SetEmitterForTesting`), default no-op, `language` = `go`. Store
keys and hashes are shape-checked and redacted before reaching any signal or record.

| Dependency | Why |
| --- | --- |
| `golang.org/x/sys/unix` | `Openat`, `Mkdirat`, `Renameat`, `Unlinkat`, `Fstatat`, `Fchmod`, `Mkfifo` (tests). Already in `go.mod`. |
| standard library | Everything else, including the FDv2 transport. |

**Common pitfalls.**

- Calling `WriteAllSkills` before `WaitForSkills` returns true does not prune (it reports
  unavailable) — but a hand-written store without `IsInitialized` is treated as ready, so its empty
  answer reads as "every skill revoked".
- No `removed` action does not mean nothing is stale: a run with a manifest or retrieval error
  prunes nothing. Check `report.OK()`.
- One watcher (or one `WriteSkills` at a time) per root; interleaved reconciles lose manifest entries.
- `Listener` is a pointer so `RemoveListener` can find it; wrapping the same func twice makes two
  listeners.
- Tests that touch `fsOps`, the umask or module state must not run in parallel.

Check: `gofmt -l internal/skills && go vet ./internal/skills/ ./aiserver/experimental/ && go test -race ./internal/skills/ ./aiserver/experimental/`.
