# aiserver — LaunchDarkly AI SDK for Go (core client)

Resolves LaunchDarkly AI Config variations, routes each call to a provider
handler by the variation's provider and mode, and records LaunchDarkly metrics,
OpenTelemetry spans and judge evaluations around it.

```bash
go get github.com/launchdarkly/go-ai-sdk/aiserver
```

Pair it with one or more handler packages:
[`claudemessages`](../claudemessages), [`openaimessages`](../openaimessages),
[`bedrockmessages`](../bedrockmessages), [`langchainmessages`](../langchainmessages),
[`claudeagents`](../claudeagents), [`openaiagents`](../openaiagents),
[`bedrockagents`](../bedrockagents).

## Quick start

### Single handler

```go
import (
	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/claudemessages"
	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
)

ctx := context.Background()
if _, err := aiserver.InitClient(ctx, aiserver.InitOptions{}); err != nil { // reads LD_SDK_KEY
	log.Fatal(err)
}
defer aiserver.Shutdown(ctx)

resp, err := aiserver.Config(aiserver.ConfigArgs{
	Key:      "my-ai-config",
	Handlers: []*aiserver.ProviderHandler{claudemessages.NewHandler(claudemessages.Options{})},
}).Invoke(ctx, "Summarise feature flags", ldcontext.New("user-123"),
	aiserver.WithVariables(map[string]any{"tone": "brief"}), // optional: fills {{tone}} in the config's messages
	aiserver.WithHistory(previousMessages))                          // optional
if err != nil {
	log.Fatal(err) // errors.Is(err, aiserver.ErrConfigDisabled) etc. — see Errors
}
fmt.Println(resp.Text(), resp.Usage.Total) // Text: string response, or structured output as JSON
```

### Multiple providers

The variation's `provider.name` and `_ldMeta.mode` pick the handler. A `*`
provider (LangChain) is the fallback when no handler names the provider.

```go
oa := openaimessages.NewHandler(openaimessages.Options{})
cl := claudemessages.NewHandler(claudemessages.Options{})
cfg := aiserver.Config(aiserver.ConfigArgs{Key: "my-ai-config", Handlers: []*aiserver.ProviderHandler{oa, cl}})
```

### Streaming

```go
for ev, err := range cfg.Stream(ctx, "Hello", userCtx) {
	if err != nil {
		return err
	}
	if ev.Type == aiserver.EventChunk {
		fmt.Print(ev.Text)
	}
}
```

Breaking out of the loop abandons the stream: spans are marked
`launchdarkly.stream.abandoned` and neither a success nor an error is tracked.
A handler that panics is tracked as a generation error (and a graph run as a
failure) before the panic continues to your code; a panic in your own loop
body counts as abandonment.

### Multi-agent graph

```go
g := aiserver.Graph("travel-agent-flow", aiserver.GraphOptions{Handlers: handlers})
out, err := g.Invoke(ctx, "Plan a trip to Lisbon", userCtx)
```

`Graph(...).Stream` yields `node_start`, `chunk`, `node_done`, `handoff` and a
final `done`. Framework-native runners take `aiserver.ResolveGraph(...)`.

### Registry

```go
registry := aiserver.NewRegistry(aiserver.RegistryEntries{
	Handlers: []*aiserver.ProviderHandler{oa, cl},
	Tools:    map[string]aiserver.ToolHandler{"get-weather": aiserver.ToolFunc(getWeather)},
})
aiserver.Config(aiserver.ConfigArgs{Key: "my-ai-config", Registry: registry})
```

Unlike the JavaScript and Python SDKs, Go has no SDK-owned global registry:
packages cannot register into shared state on import, so you build a
`*Registry` and pass it explicitly. If you want one per process, keep it in a
variable in your own package.

Tool keys must match the keys in the variation's `tools` map exactly. A tool
with no registered function is never shown to the model.

## Configuration

Each variable fills the matching `InitOptions` field only when that field is
empty; a blank value counts as unset.

| Variable | Required | Description |
|---|---|---|
| `LD_SDK_KEY` | yes, unless `InitOptions.SDKKey` or `InitWithClient` | LaunchDarkly server-side SDK key; also the `highlight.project_id` resource attribute |
| `LD_BASE_URI` | no | LaunchDarkly polling/base URI (`InitOptions.BaseURI`) for non-production environments |
| `LD_STREAM_URI` | no | LaunchDarkly streaming URI (`InitOptions.StreamURI`) |
| `LD_EVENTS_URI` | no | LaunchDarkly events URI (`InitOptions.EventsURI`) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | no | OTLP/HTTP trace endpoint (`InitOptions.OTLPEndpoint`; `/v1/traces` is appended); default `https://otel.observability.app.launchdarkly.com` (`DefaultOTLPEndpoint`) |
| `LD_SERVICE_NAME` | no | `service.name` resource attribute (`InitOptions.ServiceName`); default `go-sdk` |
| `LD_ENVIRONMENT` | no | `deployment.environment` resource attribute (`InitOptions.Environment`); omitted when unset |
| `LD_ENVIRONMENT_ID` | no | LaunchDarkly environment ID, sent as `environmentId` on every `$ld:ai:*` event and as `feature_flag.set.id` on handler spans for AI Config Monitoring; omitted when unset (the Go server SDK does not expose it) |

`InitOptions` fields: `SDKKey`, `BaseURI`, `StreamURI`, `EventsURI` (non-production
LaunchDarkly endpoints), `OTLPEndpoint`, `ServiceName`, `Environment`, `WaitFor`
(initialization timeout, default 10s).

Initialization paths:

- **Lazy** — the first `Config(...)` or `Graph(...)` call, `ResolveGraph` or
  `InspectConfig` calls `InitClient` with an empty `InitOptions`, so every
  setting comes from the environment variables above.
- **Explicit** — `InitClient(ctx, opts)` to pass options or fail fast at startup.
- **Bring your own client** — `InitWithClient(ctx, ldClient, opts)` with any value
  that satisfies `aiserver.LDClient` (a `*ldclient.LDClient` does).

## Telemetry

`InitClient` / `InitWithClient` register a global OpenTelemetry tracer provider:
an OTLP/HTTP exporter with gzip compression, a resource carrying
`highlight.project_id` (the SDK key), `service.name` and
`deployment.environment` when set, W3C trace-context and baggage propagators,
and `ConversationIDSpanProcessor` ahead of the batch exporter.

Bind a conversation id with `aiserver.WithConversationID(ctx, id)`; SDK spans
started under that context get `gen_ai.conversation.id`.

> **Content is recorded on spans.** By design, handler spans
> carry prompt and completion text (`gen_ai.prompt.N.*`, `gen_ai.completion.0.*`,
> `gen_ai.content.*` events). `CaptureContent` only gates the judge explanation.

## Errors

Match failures with `errors.Is`; each is wrapped with the key or provider it
concerns, and keeps the LaunchDarkly SDK's own reason when there is one.

| Error | Meaning |
|---|---|
| `ErrNotInitialized` | `GetClient` before `InitClient` / `InitWithClient` |
| `ErrConfigNotFound` | the flag returned no AI Config (missing, wrong type, or not evaluable) |
| `ErrConfigDisabled` | the AI Config variation is turned off |
| `ErrNoHandler` | no handler serves the variation's provider and mode |
| `ErrGraphDisabled` | the graph flag, or a node it references, is not enabled |

```go
resp, err := cfg.Invoke(ctx, input, user)
if errors.Is(err, aiserver.ErrConfigDisabled) {
	return fallbackAnswer(input)
}
```

## API reference

| Function | Purpose |
|---|---|
| `Config(ConfigArgs) *AIConfig` | Bind a config key; `.Invoke(ctx, input, ldctx, ...CallOption) (ProviderResponse, error)` / `.Stream(ctx, input, ldctx, ...CallOption) iter.Seq2[StreamEvent, error]` |
| `Graph(key, GraphOptions) *AgentGraph` | Model-driven multi-agent router; `.Invoke` / `.Stream` |
| `ResolveGraph(ctx, key, ldctx, GraphOptions) *GraphDefinition` | Topology + node configs for native runners |
| `InspectConfig(ctx, key, ldctx) InspectResult` | Read a variation without invoking a model; never errors |
| `CreateHandler(ProvidesFor, invoke, stream, captureContent) *ProviderHandler` | Build a custom handler |
| `BuildJudgeTasks` / `RunJudge` | Deferred judge evaluation on your own goroutine |
| `InitClient` / `InitWithClient` / `GetClient` / `Shutdown` | Lifecycle |
| `ParseTemplate`, `ParseUsage`, `ParseAiConfig`, `NormalizeMode`, `ParseJSONWithPossibleFences` | Utilities |

Optional per-call inputs are functional options:
`Invoke(ctx, input, user, aiserver.WithVariables(vars), aiserver.WithHistory(history))`.
Stream events carry an `EventType` (`EventChunk`, `EventDone`, `EventNodeStart`,
`EventNodeDone`, `EventHandoff`), and `ProviderResponse.Text()` returns the
response as text whether it is a string or parsed structured output.

`ConfigArgs{Key, Handlers, ToolHandlers, Registry, SkipJudges}`. With
`SkipJudges`, `Invoke` returns `JudgeTasks` (JSON-serialisable) instead of
`JudgeResults`. A judge whose own AI Config cannot be resolved (missing, or
turned off) is skipped and logged, so it never fails a call whose answer is
already produced; a judge's provider failure is returned.

## Agent Skills

Version-pinned skill references on AI Configs, verified skill content, and materialization of that
content to disk as `<root>/<key>/SKILL.md`. Install with
`go get github.com/launchdarkly/go-ai-sdk`; POSIX only for materialization.

### Keep a directory in sync with LaunchDarkly

```go
store, err := aiserver.NewFDv2SkillStore(os.Getenv("LD_SDK_KEY"), nil) // server-side sdk- key only
if err != nil { log.Fatal(err) }
if _, err := store.Start(); err != nil { log.Fatal(err) }
defer store.Close()
// Gate boot on the first payload. Before it, WriteAllSkills reports
// "unavailable" and deletes nothing.
if !store.WaitForSkills(10 * time.Second) {
	log.Printf("no skill payload yet: failed=%q diagnostics=%+v", store.Failed(), store.Diagnostics())
}
aiserver.SetSkillStore(store)

report, watcher, err := aiserver.WatchAllSkills(ctx, "/etc/agent/skills", nil)
if err != nil { log.Fatal(err) } // a bad root or option fails here, on your goroutine
defer watcher.Close()
for _, a := range report.Errors() {
	log.Printf("skill %q: %s", a.Key(), a.ErrorMessage()) // Key() is "" for a run-level failure
}
```

A skill revoked in LaunchDarkly is removed from disk within the debounce window (500ms).

### The skills one AI Config uses

```go
res := aiserver.InspectConfig(ctx, "my-agent", ldcontext.New("user-1"))
refs, err := aiserver.SkillRefs(res.Config) // an error, never [], when the config failed to resolve
if err != nil { return err }
report, err := aiserver.WriteSkills(refs, "/etc/agent/skills", nil)
```

### Read content without touching disk

```go
outcome, err := aiserver.GetSkillResult("pdf-extraction", 0) // 0 = newest version
if err != nil { return err } // only ErrNoSkillStore
switch outcome.Reason() {
case aiserver.ReasonIntegrityFailure:
	return fmt.Errorf("refusing to run: %s", outcome.Detail()) // fail closed on tampering
case aiserver.ReasonOK:
	use(outcome.Skill().Content()) // opaque bytes: the SDK never parses or decodes them
}
```

For local development and tests, `aiserver.NewInMemorySkillStore(map[string]any{...})` holds raw
wire objects (`key`, `version`, `content`, `contentHash`, optional `name`, `description`).

### API reference

| Function | Returns | Notes |
| --- | --- | --- |
| `SetSkillStore(store)` | — | nil is ignored; `Shutdown` clears it. Does not close a replaced store; a running watcher keeps the store it started with. |
| `SkillRefs(*AiConfig)` | `[]SkillReference, error` | Pure: no client, store or telemetry. A nil config, a non-array `skills`, or any malformed entry is an error (never a partial list, which `WriteSkills` with pruning would act on). |
| `GetSkill(key, version)` | `*Skill, error` | nil when missing, at another version, or withheld. |
| `GetSkillResult(key, version)` | `SkillOutcome, error` | `absent`, `integrity_failure`, `ok`, `store_unavailable`, `wrong_version`. |
| `GetSkills(refs)` | `[]Skill, error` | `[]SkillKey`, `[]SkillReference` or mixed `[]SkillRequest`; input order, misses omitted. |
| `AllSkills()` | `[]Skill, error` | Newest version per key. |
| `WriteSkills(skills, root, *WriteSkillsOptions)` | `ReconcileReport, error` | Any slice of `Skill`, `*Skill`, `SkillReference`, `SkillKey` (or mixed `[]SkillInput`). |
| `WriteAllSkills(root, *WriteSkillsOptions)` | `ReconcileReport, error` | Every skill the store holds (the `"*"` key). |
| `WatchSkills` / `WatchAllSkills(ctx, …, *WatchSkillsOptions)` | `ReconcileReport, *SkillWatcher, error` | Needs a store with `AddListener`; stops on `Close` or when `ctx` is done. |
| `NewFDv2SkillStore(sdkKey, *FDv2SkillStoreOptions)` | `*FDv2SkillStore, error` | `Start`, `WaitForSkills`, `IsInitialized`, `Failed`, `Diagnostics`, `Close` (final). |

- `WriteSkillsOptions`: `NoPrune` (pruning is on by default), `Timeout *time.Duration` (nil = 10s,
  0 = already exhausted, negative is an error), `OnUnavailable` (`keep` default; `raise` returns
  `ErrSkillsUnavailable`). `WatchSkillsOptions` embeds it and adds `Debounce` (0 = 500ms) and
  `OnReconcile`.
- `FDv2SkillStoreOptions`: `BaseURI` (default `https://sdk.launchdarkly.com`; given alone it serves
  both endpoints), `StreamURI` (default `https://stream.launchdarkly.com`), `Mode` (`stream`
  default, or `poll`), `PollInterval` (30s), `ReadTimeout` (10s poll, 300s stream),
  `InitialBackoff` (1s), `MaxBackoff` (30s). URIs must be `https`; zero means default, negative is
  an error.
- Constants: `SkillFilename` (`SKILL.md`), `ManifestFilename` (`.launchdarkly-skills.json` — add it
  to `.gitignore`), `ManifestVersion` (1). Errors are sentinels for `errors.Is`.

### Environment variables

The skills API reads none. The examples read `LD_SDK_KEY` (required there) for the store.

### Integrity log record

An integrity failure is logged at ERROR through the SDK logger (`SetLogger`, default `slog.Default()`) as
`ld.skills.integrity_failure {sorted-key JSON}`, with the same mapping on the `ld_skills` attribute.
SIEM rules can match the event name and `reason_code`.

## Development

```bash
go test -race ./...
go vet ./...
```

Experimental features (Evaluations from Code) live in
[`aiserver/experimental`](experimental).
