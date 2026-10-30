# LaunchDarkly AI SDK for Go — agents.md

Architectural contract for contributors and coding agents. Behaviour follows
the LaunchDarkly AI SDK contract shared with the JavaScript and Python SDKs;
where it leaves a detail open, the JavaScript SDK decides, then the Python SDK.

## Package hierarchy

```
Tier 0  aiserver                 core: no provider SDK imports
        aiserver/experimental    experimental entry point (skills, evaluations)
        internal/{bridge,trajectory,judging}   shared internals
Tier 1  claudemessages openaimessages bedrockmessages langchainmessages
        claudeagents   openaiagents   bedrockagents
        internal/{anthropicwire,openaiwire,bedrockwire}  provider wire helpers shared by a pair
```

Tier 1 packages import `aiserver` and their provider SDK only. `aiserver`
never imports a Tier 1 package, and core never names an experimental type:
the two meet only through `internal/bridge`.

## The ProviderHandler contract

```go
type ProviderHandler struct {
	ProvidesFor    ProvidesFor // {Provider, Mode}; Provider "*" = any provider
	CaptureContent ContentCapture // CaptureOn / CaptureOff; CaptureDefault follows Configure, else on
	Invoke         InvokeFunc  // func(ctx, HandlerRequest) (HandlerResult, error)
	Stream         StreamFunc  // optional: func(ctx, HandlerRequest) iter.Seq2[HandlerStreamEvent, error]
}

type HandlerRequest struct {
	Config       AiConfig               // the resolved variation
	UserInput    string                 // "" when absent
	ToolHandlers map[string]ToolHandler // already wrapped for tracking and trajectory
	Variables    map[string]any         // caller variables + ldContext + __ld (TrackData)
	History      []Message              // runtime conversation; may hold image blocks
}
```

`ProvidesFor` is required for routing: the core matches the variation's
`provider.name` and normalised mode (`agent`, else `messages`). Build handlers
with `CreateHandler` (applications). SDK packages call `internal/sdkapi`,
which does not report usage; they report their own helper usage instead.

A handler must: compose its prompt with `aiserver.ComposePrompt`; offer only
config tools that have a `ToolFunc` (`CallableTools`); run the tool loop and
propagate tool errors; open its span tree with `StartHandlerRun`; return
provider usage in a shape `ParseUsage` reads (cache buckets unfolded for
Anthropic/Bedrock, cache-inclusive input for OpenAI/LangChain); and on
streaming, yield `chunk` events then exactly one `done`, stopping cleanly when
the consumer stops.

## NativeTool sentinel

A `*aiserver.NativeTool{ToolName}` value in `ToolHandlers` marks a provider
built-in instead of a local function. After the core wraps it, the handler
detects it with a type switch (`case *aiserver.NativeTool`), wires the
provider's built-in named `ToolName`, and calls `Track()` when the model uses
it so `$ld:ai:tool_call` is sent with the LaunchDarkly tool key. This replaces
the JS `NATIVE_TOOL_KEY` symbol.

Built-ins per provider (only the agent handlers wire `NativeTool`; every
`*messages` handler offers only `ToolFunc` tools via `CallableTools`, so a
`NativeTool` there is not sent to the model):

| Provider | Package | Sentinel → provider built-in | Unsupported |
|---|---|---|---|
| Anthropic | `claudeagents` (`builtins.go`, `handler.go` `serverTool`, `hosttools.go`) | `ClaudeWebSearch` (`WebSearch`) → `web_search_20250305`; `ClaudeWebFetch` (`WebFetch`) → `web_fetch_20250910` server tools. `ClaudeBash`, `ClaudeRead`, `ClaudeEdit`, `ClaudeWrite`, `ClaudeGlob`, `ClaudeGrep`, `ClaudeTodoWrite`, `ClaudeNotebookEdit` → function tools under the Claude Code name and schema, executed on the host; file tools confined to `Options.WorkDir` (`os.Root`), Bash not sandboxed. `Track()` fires per call. | Any other `ToolName`: the config fails with `ErrUnsupportedNativeTool` |
| OpenAI | `openaiagents` (`runner.go` `newAgent`, `openaiwire.HostedTool`) | No exported sentinels: `&aiserver.NativeTool{ToolName: "web_search"}` (or any hosted Responses tool type, e.g. `file_search`, `code_interpreter`) becomes a hosted tool `{"type": ToolName, ...customParameters}`. `Track()` fires on the matching `<type>_call` output item. | — (the type is passed through; the API rejects unknown types) |
| Bedrock | `bedrockagents` | none | `NativeTool` is not wired |
| LangChain | `langchainmessages` | none | `NativeTool` is not wired |

## Lifecycle

`InitClient` (lazy on first `Invoke`) → `Config` / `Graph` → `Shutdown`.
`InitWithClient` installs an existing `aiserver.LDClient` (BYOC) and still sets
up telemetry. A failed init is never cached.

## Agent graphs

A graph flag holds a `GraphTopology` (`root` + `edges`). `ResolveGraph`
evaluates every referenced node config; any disabled node disables the graph.
`GraphDefinition` exposes `Root`, `GetNode`, `GetChildNodes`,
`GetParentNodes`, `TerminalNodes`, `EdgesFrom`, `Traverse` (root first),
`ReverseTraverse` (leaves first, root last), `RunNode` and `Route`. Agent
packages build `XxxGraph` wrappers on `internal/sdkapi.Graph` and native runners on
`ResolveGraph`, emitting graph events with `TrackGraphEvent`.

## OTel setup

`aiserver.InitClient` / `InitWithClient` register the global tracer provider
(OTLP/HTTP + gzip, `launchdarkly.project_id` / `service.name` /
`deployment.environment` resource, W3C propagators,
`ConversationIDSpanProcessor`). Handler packages only start spans through
`StartHandlerRun`; they never configure OTel. The OTel SDK modules are ordinary
`go.mod` dependencies, so there is nothing extra to install. Endpoint and
resource configuration (`OTEL_EXPORTER_OTLP_ENDPOINT`, `LD_SERVICE_NAME`,
`LD_ENVIRONMENT`): see [aiserver/agents.md](aiserver/agents.md#otel-setup) and
[aiserver/README.md](aiserver/README.md#configuration).

## Dependencies

One module (`go.mod`); each package imports only what it needs.

| Module | Used by | Why |
|---|---|---|
| `github.com/launchdarkly/go-server-sdk/v7` | `aiserver` | Default LaunchDarkly client |
| `github.com/launchdarkly/go-sdk-common/v3` | all | `ldcontext`, `ldvalue` |
| `go.opentelemetry.io/otel` (+ `sdk`, `trace`, `otlptracehttp`) | `aiserver`, handlers | Tracer provider, spans, OTLP export |
| `github.com/google/uuid` | `aiserver` | Run ids |
| `github.com/anthropics/anthropic-sdk-go` | `claudemessages`, `claudeagents` | Anthropic Messages API |
| `github.com/openai/openai-go/v3` | `openaimessages`, `openaiagents` | OpenAI Responses API |
| `github.com/aws/aws-sdk-go-v2` (+ `config`, `credentials`, `service/bedrockruntime`, `eventstream`), `github.com/aws/smithy-go` | `bedrockmessages`, `bedrockagents` | Bedrock Converse / ConverseStream |
| `github.com/tmc/langchaingo` | `langchainmessages` | Community LangChain port |
| `golang.org/x/sys` | `internal/skills` | `*at()` filesystem calls for skill materialization |

## Common pitfalls

- **A `NativeTool` on a messages handler or Bedrock/LangChain is dropped
  silently.** Only `claudeagents` and `openaiagents` wire built-ins; elsewhere
  the tool is just not offered.
- **Tool keys must match the variation's `tools` map exactly.** A handler
  under a differently-cased key is never offered to the model.
- **Prompt, completion and tool content is on spans by default**. Set the handler's `CaptureContent: aiserver.CaptureOff`, or `aiserver.Configure` before the first call, to keep it off spans. Model, usage and status are written either way.
- **Pass the request `ctx`.** Spans, the conversation id and cancellation follow
  it; `context.Background()` detaches the call from the caller's trace.
- **Breaking out of a `Config(...).Stream` loop abandons the run**: no success or error is
  tracked. Drain to the `done` event if you need the metrics.
- **One disabled node disables the whole graph** (`ErrGraphDisabled`).
