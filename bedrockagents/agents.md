# bedrockagents — contributor and agent guide

## Role and tier

Tier 1 handler package. `ProvidesFor = {Provider: "Bedrock", Mode: "agent"}`. Depends on the Tier 0 core
`aiserver` and the internal Converse engine `internal/bedrockwire` (shared with `bedrockmessages`); nothing in
the SDK depends on it.

The JavaScript and Python `bedrock-agents` packages are built on Strands. Go has no Strands SDK, so
this package reproduces the observable Strands contract on Converse: same modelId rule, parameters, model-options
callback, client escape hatch, native message input, tools, streaming and span tree.

## File map

| File | Responsibility |
| --- | --- |
| `handler.go` | Public API (`Options`, `NewHandler`, `BedrockAgents`, `BedrockGraph`); builds a `bedrockwire.Engine` in agent mode. |
| `../internal/bedrockwire/engine.go` | Client construction and auth precedence, the tool loop, the Converse / ConverseStream turn, mode-dependent prompt, structured output and result rules. |
| `../internal/bedrockwire/convert.go` | Pure mapping: modelId prefixing, LD messages and images → Converse content, tools → `toolSpec`, `model.parameters` → `inferenceConfig`, options merge, `outputConfig`, ConverseStream input conversion and stream accumulation. |
| `handler_test.go` | Handler tests: agent-mode prompt, tools, telemetry, errors, streaming, structured output, model ID and auth, `BedrockGraph`, usage reporting. |
| `fake_test.go` | HTTPS fake Bedrock Runtime (JSON and event-stream replies), keyed recording fake LD client, AWS env isolation. |

## Public exports

```go
const (
	PackageName     = "github.com/launchdarkly/go-ai-sdk/bedrockagents"
	PackageVersion  = "0.1.0"
	SpanName        = "invoke_agent"
	GenAISystem     = "aws.bedrock"
	DefaultMaxSteps = 10
)
var ProvidesFor aiserver.ProvidesFor
var ErrUnsupportedContent error
var ErrToolNotFound error
type Options struct{ Client *bedrockruntime.Client; APIKey, Region string; ModelOptions func(aiserver.AiConfig) *bedrockruntime.ConverseInput; CaptureContent bool; MaxSteps int }
func NewHandler(Options) (*aiserver.ProviderHandler, error)
type InvokeOptions struct{ ToolHandlers map[string]aiserver.ToolHandler; Registry *aiserver.Registry; SkipJudges bool; Variables map[string]any; Handler Options }
func BedrockAgents(ctx context.Context, configKey, userInput string, userCtx ldcontext.Context, opts *InvokeOptions) (aiserver.ProviderResponse, error)
func BedrockGraph(key string, opts aiserver.GraphOptions, handlerOpts Options) (*aiserver.AgentGraph, error)
```

`init()` registers `PackageName`/`PackageVersion` for `$ld:ai:sdk:info`. Usage helpers:
`bedrock-agents.createBedrockAgentsHandler` (`NewHandler`), `bedrock-agents.bedrockAgents` (`BedrockAgents`),
`bedrock-agents.bedrockGraph` (`BedrockGraph`). Each reports first, then uses the non-reporting `build`,
`aiserver.ConfigInternal` or `aiserver.GraphInternal`, so one call emits only its own helper.

## Implementation details

- **config** — `aiserver.ComposePrompt(req, ModeAgent)`: `instructions` is the system prompt and wins over
  system-role `messages`; non-system config messages, then history (user/assistant only), then `userInput` are
  Converse `messages` (these rules live in the core). `model.name` + `model.region` → `modelId`.
  `model.parameters` → `inferenceConfig` (four documented fields). `model.custom` is never read.
  `outputFormat` → `outputConfig.textFormat {type: json_schema, structure.jsonSchema {name: "output", schema}}`
  on blocking turns only; no system-prompt instruction. The JSON text is returned and the core parses it.
- **userInput** — appended by `ComposePrompt` as a text block (`""` when empty and appended).
- **toolHandlers** — only `aiserver.ToolFunc` entries are offered. On `stopReason: tool_use` each
  `toolUse` runs under an `execute_tool <name>` span; the assistant message and a user message of `toolResult`
  blocks are appended and the loop continues, up to `MaxSteps` model turns (default 10). Tool errors abort the
  run and propagate wrapped (`errors.Is` works). A `toolUse` naming a tool that was not offered (not in
  `config.tools`, or without a `ToolFunc`, even when `ToolHandlers` has a function under that name) fails the
  run with `ErrToolNotFound` (the error names the tool) and leaves an ERROR `execute_tool` span.
- **variables** — template values for `ComposePrompt`, plus `__ld` / `ldContext` written to the root span by
  `aiserver.StartHandlerRun` (child spans never carry them).
- **history** — structured Converse turns, never system text; base64 images → `image {format, source.bytes}`.
- **Result** — blocking `Output` is the final model turn's text (the Strands agent result); streaming
  `done.Output` is every turn's text. Usage keeps Bedrock's raw keys so `aiserver.ParseUsage` folds once.
- **Telemetry** — root `invoke_agent` (operation `invoke_agent`, system/provider `aws.bedrock`, request and
  response model = resolved `modelId`), one `chat <modelId>` child per model turn, sibling
  `execute_tool <name>` per tool call. The AWS SDK tracer stays no-op, so there is no duplicate representation.
- **ModelOptions** — called once per handler call and merged into every turn (`copyFields`, by field name);
  `ModelId`, `Messages`, `System` are protected; `ToolConfig` only contributes `ToolChoice` when the handler sends
  tools; an `OutputConfig` without `TextFormat` keeps the handler's.
- **BedrockGraph** — replaces `opts.Handlers` with the one handler and calls `aiserver.GraphInternal` through the
  `graphInternal` variable, which tests swap to observe the forwarded options.

## OTel setup

The tracer provider is registered by `aiserver.InitClient` / `aiserver.InitWithClient` in the core package. This
package only creates spans through `otel.Tracer(PackageName)`; it never configures exporters. The OTel SDK is a
dependency of the core module, so there is no extra install step in Go. See `../aiserver` (core agents.md) for
the OTLP endpoint and resource attributes.

## `InitClient()` — when to call it

Not required: the core initialises lazily on the first `Config(...).Invoke` / `Stream`, `Graph(...).Invoke` /
`Stream`, or `BedrockAgents` call. Call `aiserver.InitClient` explicitly for custom options (SDK key, base URIs,
OTLP endpoint), use `aiserver.InitWithClient` to bring your own LaunchDarkly client (BYOC), or call it at startup
to pre-warm.

## Dependencies

| Dependency | Why |
| --- | --- |
| `github.com/launchdarkly/go-ai-sdk/aiserver` | Handler contract, prompt composition, graph, span helpers, usage reporting. |
| `github.com/launchdarkly/go-ai-sdk/internal/bedrockwire` | Converse engine shared with `bedrockmessages`. |
| `github.com/aws/aws-sdk-go-v2/service/bedrockruntime` | Converse / ConverseStream client and types. |
| `github.com/launchdarkly/go-sdk-common/v3` | `ldcontext.Context` for the convenience wrapper. |
| test: `aws-sdk-go-v2/credentials`, `aws/protocol/eventstream`, OTel SDK `tracetest` | Static test credentials, event-stream frame encoding, span capture. |

## Common pitfalls

1. **`model.region` is not an AWS region.** It is the inference-profile prefix (`us`, `eu`, `global`). Set the
   AWS region with `Options.Region` or `AWS_REGION`.
2. **Native structured output needs a model that supports it.** `outputFormat` is always sent as
   `outputConfig.textFormat`; a model without Converse structured-output support rejects the request with a
   validation error. Remove `outputFormat` from that variation.
3. **`ModelOptions` runs once per call, not per turn.** Unlike `bedrockmessages.ConverseOptions`, a value that must
   change between tool turns cannot be set here.
4. **`BedrockGraph` drops `opts.Handlers`.** Add other providers' handlers by calling `aiserver.Graph` directly.
5. **An injected client ignores `APIKey` and `Region`**, and `model.custom` is ignored on purpose.
