# aiserver — agents.md

## Role and tier

Tier 0 — core client. Every handler package (Tier 1) depends on it; it imports
no handler package. Go has no barrel package (Tier 2): applications import the
handler packages they use.

## File map

| File | Responsibility |
|---|---|
| `types.go` | Public value types: `AiConfig`, `Message`, `ProviderHandler`, `HandlerRequest`, `ToolHandler`, `TrackData`, results |
| `parse.go` | `ParseAiConfig` — variation validation incl. fail-closed `skills` refs |
| `errors.go` | Exported sentinel errors for `errors.Is` |
| `utils.go` | `ParseTemplate`, `ParseUsage`, `NormalizeMode`, fence-tolerant JSON parse |
| `registry.go` | `Registry`, `Compose`, handler/tool resolution |
| `lifecycle.go` | `InitClient` / `InitWithClient` / `GetClient` / `Shutdown`, OTel setup, sdk-info and helper-usage events |
| `config.go` | `Config().Invoke/Stream`, handler selection, `InspectConfig`, variation reading |
| `tracking.go` | Tracked execution, `$ld:ai:*` events, tool-call tracking wrapper, trajectory capture |
| `judges.go` | Inline judges, `BuildJudgeTasks`, `RunJudge`, judge handler selection |
| `judgeeval.go` | Holds a judge's root span open to write `gen_ai.evaluation.result` |
| `graph.go` | `ResolveGraph`, `GraphDefinition`, `Graph().Invoke/Stream`, handoff routing |
| `telemetry.go` | Conversation-id processor, context identity, `SetLDSpanAttributes` |
| `handlerkit.go` | Shared handler building blocks: `ComposePrompt`, span tree (`HandlerRun`), usage attributes |
| `capture.go` | `Configure`, `SDKOptions`, `ContentCapture`, `ErrConfigureLocked`; binds a handler's capture setting per call |

Internal packages: `internal/trajectory`, `internal/judging`
(judge message history and verdict parsing), `internal/bridge` (shutdown hooks and
the shared logger, used by Agent Skills and the experimental entry point), `internal/telemetry` (the content gate's state and
per-call binding, GenAI chat/tool content writers, span-attribute and
span-counter bindings, tracer flush), `internal/sdkapi` (handler-package entry
points: `CreateHandler` without a usage report, `StartInvocation` for native
graph runners).

## Public exports

```go
// lifecycle
InitClient, InitWithClient, GetClient, Shutdown, SetLogger, InitOptions, LDClient
// configs
Config, ConfigArgs, AIConfig, InspectConfig, InspectResult
// graphs
Graph, ResolveGraph, AgentGraph, GraphDefinition, GraphNode, GraphEdge,
GraphOptions, GraphResponse, GraphStreamEvent, GraphTopology, RouteResult, RunNodeOptions,
TraverseVisitor, TrackGraphEvent
// content capture
Configure, SDKOptions, ContentCapture, CaptureDefault, CaptureOn, CaptureOff
// errors
ErrNotInitialized, ErrConfigNotFound, ErrConfigDisabled, ErrNoHandler, ErrGraphDisabled,
ErrConfigureLocked
// call options and events
CallOption, CallOptions, NewCallOptions, WithVariables, WithHistory, EventType,
EventChunk, EventDone, EventNodeStart, EventNodeDone, EventHandoff
// handlers
ProviderHandler, ProvidesFor, Mode, ModeMessages, ModeAgent, WildcardProvider,
CreateHandler, InvokeFunc, StreamFunc, HandlerRequest,
HandlerResult, HandlerStreamEvent, ToolHandler, ToolFunc, NativeTool, WrapToolHandlers
// handler kit
ComposePrompt, Prompt, WithOutputFormat, OutputFormatInstruction, ContentText, ImageURL,
StartHandlerRun, HandlerRun, HandlerSpanOptions, SpanUsage, SetUsageSpanAttributes,
CallableTools, ToolResultString, SetLDSpanAttributes
// judges
BuildJudgeTasks, BuildJudgeTasksArgs, RunJudge, JudgeTask, JudgeResult, JudgeRunResult,
FormattingInstructions
// registry, utils, telemetry
Registry, RegistryEntries, NewRegistry, Compose, ParseAiConfig, ParseTemplate,
ParseUsage, NormalizeMode, ParseJSONWithPossibleFences, WithConversationID,
ConversationIDSpanProcessor, ContextIdentity, ContextToMap, Track, RegisterAiSdkPackage,
ReportUsage, TracerPrefix, PackageName, PackageVersion
```

## Implementation details

### Lifecycle invariants

- One client per process. `InitClient` is idempotent and serialised by
  `lifecycleMu`; a failed call stores nothing and closes any client it built,
  so the next call retries. `stateMu` guards the client only briefly, so
  `GetClient` and tracking never wait on a slow initialization.
- `Shutdown` swaps state out under the lock, runs the bridge shutdown hooks
  (Agent Skills state, experimental features; panics recovered), shuts the
  tracer provider down, then flushes and closes the client. Calling it twice is a no-op.
- `$ld:ai:sdk:info` is sent once per registered package per client;
  `$ld:ai:sdk:usage` once per helper per client, held until a client exists.

### `Config().Invoke` flow

1. Resolve handlers (local first, then registry) and tools (local overlays registry).
2. `extractVariation`: variation with default `{enabled:false}`, require
   `_ldMeta.enabled`, `ParseAiConfig`. An `AIConfig` built through
   `bridge.ConfigWithOptions` calls its `RunOptions.Variation` instead of
   `JSONVariation`, and its tracked runs use the options' run id, extra track
   data and span attributes.
3. Select the handler: exact `[provider, mode]`, then `["*", mode]`, else error.
4. Tracked execution: mint `TrackData`, wrap tools (trajectory recorder inside
   the `$ld:ai:tool_call` wrapper), inject `variables.ldContext` and `__ld`,
   call the handler, track duration, success/error and tokens. A handler
   panic is tracked as an error and re-raised; a panic from the stream
   consumer's loop body is not tracked (it is abandonment).
5. Parse JSON output when `outputFormat` is set (best effort).
6. Run judges inline, or build `JudgeTasks` when `SkipJudges`.

### Content capture

Prompt, completion, tool argument/result and judge-explanation content is
written on spans only when capture is enabled for the call's `ctx`
(`telemetry.Enabled`). Precedence: the handler's `CaptureContent`
(`CaptureOn` / `CaptureOff`), else the process default from
`Configure(SDKOptions{CaptureContent: ...})`, else on.

- **Binding.** Each handler call binds its resolved setting on `ctx`. The core
  path (`executeAndTrack`, `executeAndStream`, evaluations) binds the value of
  the handler it routed to, marked pending; the `CreateHandler` wrapper
  consumes that mark and keeps the core's binding, so a copied handler with a
  different `CaptureContent` is honoured. A direct `Invoke`/`Stream` call (no
  mark) binds the handler's own setting. The `$ld:ai:tool_call` tool wrapper
  clears an unconsumed mark, so a handler called from a tool binds its own.
  A nested `CaptureDefault` handler follows the process default, not the
  enclosing handler's explicit value.
- **Native graph runners** (`openaiagents.ToOpenAIAgents`,
  `claudeagents.ToClaudeAgents`, `langchainagents.ToLangGraph`) run agent loops
  without a handler call, so they bind with `sdkapi.StartInvocation` before
  the graph span: `Handler.CaptureContent` for the OpenAI and Claude runners,
  `CaptureDefault` for LangGraph.
- **Configure lock.** The first binding latches `Configure`. After that,
  `Configure` compares resolved values (`CaptureDefault` == `CaptureOn`) and
  returns `ErrConfigureLocked` on a change; values above `CaptureOff` are
  rejected. `Shutdown` restores the built-in default and clears the lock.
- Model, provider, usage, status and tool names are written either way.

### `Graph().Invoke` / `Stream` flow

Both drain one walk: track `$ld:ai:graph:node` on entry, run the node (handoff
tools on multi-edge nodes), track `handoff_success`, thread
`[Original request]` / `[Previous agent response]` into the next node, stop on
a terminal answer, leaf, revisit, or 100 steps; then graph duration, tokens,
success, and the optional graph judge. History reaches the root only.

## OTel setup

The tracer provider is registered by `InitClient` / `InitWithClient` here;
handler packages only start spans. OTel SDK modules are regular Go
dependencies — nothing extra to install.

## InitClient — when to call it

Lazy initialization is the default. Call `InitClient` explicitly to pass
options, to fail fast at startup, or to pre-warm; call `InitWithClient` to
reuse an LD client you already built.

## Dependencies

| Module | Why |
|---|---|
| `github.com/launchdarkly/go-server-sdk/v7` | Default LaunchDarkly client |
| `github.com/launchdarkly/go-sdk-common/v3` | `ldcontext`, `ldvalue` |
| `go.opentelemetry.io/otel` (+ `sdk`, `otlptracehttp`) | Spans, tracer provider, OTLP export |
| `github.com/google/uuid` | Run ids |
| `golang.org/x/sys` | `*at()` filesystem calls for skill materialization |

## Agent Skills

Part of the core client. Not a handler: no `providesFor`. All logic and tests live in the
`skills_*.go` files of this package; `skills_public_test.go` exercises the exported names from
outside the package. The names are pinned in `surface_test.go`'s `rootSurface`.
Skills report no `$ld:ai:sdk:usage` helper yet. JS and Python report `client.*` IDs for the public
skills helpers; Go does not, and `TestSkillsHelpersReportNoUsage` pins that until it is added.

| File | Responsibility |
| --- | --- |
| `skills_types.go` | Immutable value types (unexported fields + getters), closed sets, on-disk constants. |
| `skills_store.go` | `SkillStore` seam and optional halves, `Listener`, module state, `SetSkillStore`, `clearSkillState` (registered with `bridge.OnShutdown`), readiness probe. |
| `skills_telemetry.go` | The only place signals and integrity log records are built (three signals, ten reason codes). |
| `skills_verify.go` | `verifyRawSkill` (shape → size → encoding → hash), `resolveFromStore` (the one retrieval path: key check, then version check), `newestByKey`. |
| `skills_accessors.go`, `skills_memstore.go` | `SkillRefs`, `GetSkill*`, `AllSkills`; `InMemorySkillStore` and the shared `objectSet`. |
| `skills_writeskills.go` | Options, the availability gate, request resolution (`"*"` enumerates raw objects; never `AllSkills`). |
| `skills_materialize_unix.go` | Root validation, manifest load/rewrite, write/adopt/update, prune. `skills_materialize_other.go` refuses non-POSIX. |
| `skills_safefs_unix.go` | Descriptor-pinned primitives on `golang.org/x/sys/unix` and the `fsOps` interception hooks. |
| `skills_fdv2_reader.go`, `skills_fdv2_store.go` | Pure FDv2 protocol reader; transport (HTTP, SSE, backoff, lifecycle). Standard library only. |
| `skills_watch.go` | Watcher goroutine: trailing debounce, context cancellation, detach on close. |

```go
// skills exports (skills_*.go)
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
`Shutdown` clears the store and emitter through the bridge hook; it does not close a
`LaunchDarklySkillStore`.

**OTel.** Skills emit no spans and never call `client.track`, not even `$ld:ai:sdk:usage`. The
tracer provider is set up by `InitClient`.

**Security posture (non-relaxable).** The root is pinned once per reconcile; every read and write
under it is `*at()`-relative, so swapping a directory for a symlink cannot redirect anything. The
manifest read and the adoption comparison read share one no-follow, non-blocking, `fstat`-checked,
`len+1`-bounded primitive (`readRegularAt`). Only manifest-listed paths under a matching key are
overwritten or pruned; a corrupt manifest or incomplete retrieval suppresses all deletion. Content is
re-verified immediately before writing. The SDK never parses `Skill.Content`; do not add a reader.

**Telemetry seam.** Private emitter (`setSkillEmitterForTesting`), default no-op, `language` = `go`. Store
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

Check: `gofmt -l aiserver && go vet ./aiserver/... && go test -race ./aiserver/...`.

## Common pitfalls

- **Defer `EndOnPanic` after `StartHandlerRun`.** Without it a panicking
  provider call or tool leaves the run's spans unended; with it every open
  span ends ERROR and the panic continues.
- **Tool keys are the contract.** The model sees the `tools` map key from the
  variation; a `ToolHandlers` key that differs by case is silently not offered.
- **Context is explicit.** Spans and conversation ids follow the `ctx` you
  pass. Starting a stream with `context.Background()` detaches it from your
  request span.
- **Content is on spans by default**. Set the handler's
  `CaptureContent: CaptureOff`, or call `Configure` before the first handler
  call, before exporting to a collector that must not hold prompt text.
- **Configure after the first call cannot change the effective value.** It
  returns `ErrConfigureLocked` until `Shutdown`.
- **Judges need a handler for their own provider.** A judge whose provider no
  handler covers is skipped, not run against the wrong API.
