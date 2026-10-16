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

Internal packages: `internal/trajectory`, `internal/judging`
(judge message history and verdict parsing), `internal/bridge` (hooks to the
experimental entry point).

## Public exports

```go
// lifecycle
InitClient, InitWithClient, GetClient, Shutdown, SetLogger, InitOptions, LDClient
// configs
Config, ConfigArgs, AIConfig, InspectConfig, InspectResult
// graphs
Graph, ResolveGraph, AgentGraph, GraphDefinition, GraphNode, GraphEdge,
GraphOptions, GraphResponse, GraphStreamEvent, GraphTopology, RouteResult, RunNodeOptions,
TraverseVisitor, MaxGraphCacheSize, TrackGraphEvent
// errors
ErrNotInitialized, ErrConfigNotFound, ErrConfigDisabled, ErrNoHandler, ErrGraphDisabled
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
- `Shutdown` swaps state out under the lock, runs experimental hooks (panics
  recovered), shuts the tracer provider down, then flushes and closes the
  client. Calling it twice is a no-op.
- `$ld:ai:sdk:info` is sent once per registered package per client;
  `$ld:ai:sdk:usage` once per helper per client, held until a client exists.

### `Config().Invoke` flow

1. Resolve handlers (local first, then registry) and tools (local overlays registry).
2. `extractVariation`: variation with default `{enabled:false}`, require
   `_ldMeta.enabled`, `ParseAiConfig`.
3. Select the handler: exact `[provider, mode]`, then `["*", mode]`, else error.
4. Tracked execution: mint `TrackData`, wrap tools (trajectory recorder inside
   the `$ld:ai:tool_call` wrapper), inject `variables.ldContext` and `__ld`,
   call the handler, track duration, success/error and tokens. A handler
   panic is tracked as an error and re-raised; a panic from the stream
   consumer's loop body is not tracked (it is abandonment).
5. Parse JSON output when `outputFormat` is set (best effort).
6. Run judges inline, or build `JudgeTasks` when `SkipJudges`.

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

## Common pitfalls

- **Defer `EndOnPanic` after `StartHandlerRun`.** Without it a panicking
  provider call or tool leaves the run's spans unended; with it every open
  span ends ERROR and the panic continues.
- **Tool keys are the contract.** The model sees the `tools` map key from the
  variation; a `ToolHandlers` key that differs by case is silently not offered.
- **Context is explicit.** Spans and conversation ids follow the `ctx` you
  pass. Starting a stream with `context.Background()` detaches it from your
  request span.
- **Content is on spans by default**. Do not export traces to a
  collector that must not hold prompt text.
- **Judges need a handler for their own provider.** A judge whose provider no
  handler covers is skipped, not run against the wrong API.
