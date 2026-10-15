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
aiserver.GlobalRegistry.Register(aiserver.RegistryEntries{
	Handlers: []*aiserver.ProviderHandler{oa, cl},
	Tools:    map[string]aiserver.ToolHandler{"get-weather": aiserver.ToolFunc(getWeather)},
})
aiserver.Config(aiserver.ConfigArgs{Key: "my-ai-config", Registry: aiserver.GlobalRegistry})
```

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

- **Lazy** — `Config(...).Invoke` and `InspectConfig` call `InitClient` with no
  options on first use.
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
`JudgeResults`.

## Development

```bash
go test -race ./...
go vet ./...
```

Experimental features (Agent Skills, Evaluations from Code) live in
[`aiserver/experimental`](experimental).
