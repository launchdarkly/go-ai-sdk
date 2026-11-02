# bedrockmessages — contributor and agent guide

## Role and tier

Tier 1 handler package. `ProvidesFor = {Provider: "Bedrock", Mode: "messages"}`. Depends on the Tier 0 core
`aiserver`; nothing in the SDK depends on it.

## File map

| File | Responsibility |
| --- | --- |
| `handler.go` | Public API (`Options`, `NewHandler`, `BedrockMessages`); builds a `bedrockwire.Engine` in messages mode. |
| `../internal/bedrockwire/engine.go` | Shared with `bedrockagents`: client construction and auth precedence, the tool loop, and the Converse / ConverseStream turn. |
| `../internal/bedrockwire/convert.go` | Shared pure mapping: modelId prefixing, LD messages and images → Converse content, tools → `toolSpec`, `model.parameters` → `inferenceConfig`, `ConverseOptions` merge, ConverseStream input conversion and stream-event accumulation. |
| `handler_test.go` | Handler tests: prompt paths, tools, telemetry, errors, streaming, structured output, model ID and auth, usage reporting. |
| `fake_test.go` | HTTPS fake Bedrock Runtime (JSON and event-stream replies), recording fake LD client, AWS env isolation. |

## Public exports

```go
const (
	PackageName    = "github.com/launchdarkly/go-ai-sdk/bedrockmessages"
	PackageVersion = "0.1.0"
	SpanName       = "invoke_agent"
	GenAISystem    = "aws.bedrock"
)
var ProvidesFor aiserver.ProvidesFor
var ErrUnsupportedContent error
var ErrToolNotFound error
var ErrMaxSteps error
type Options struct{ Client *bedrockruntime.Client; APIKey, Region string; ConverseOptions func(aiserver.AiConfig) *bedrockruntime.ConverseInput; CaptureContent bool }
func NewHandler(Options) (*aiserver.ProviderHandler, error)
type InvokeOptions struct{ ToolHandlers map[string]aiserver.ToolHandler; Registry *aiserver.Registry; SkipJudges bool; Variables map[string]any; Handler Options }
func BedrockMessages(ctx context.Context, configKey, userInput string, userCtx ldcontext.Context, opts *InvokeOptions) (aiserver.ProviderResponse, error)
```

`init()` registers `PackageName`/`PackageVersion` for `$ld:ai:sdk:info`. `NewHandler` reports
`bedrock-messages.createBedrockMessagesHandler` and `BedrockMessages` reports `bedrock-messages.bedrockMessages`;
both then use the non-reporting `build` and `internal/sdkapi.Config`.

## Implementation details

- **config** — `aiserver.ComposePrompt(req, ModeMessages)` builds system + turns (messages win over instructions;
  these rules live in the core). `model.name` + `model.region` → `modelId` (prefix once, exact
  `<region>.` match). `model.parameters` → `inferenceConfig` for the four documented fields only.
  `model.custom` is never read. `outputFormat` appends the core's JSON instruction to the system prompt on the
  blocking path only. `tools` filtered by `aiserver.CallableTools` → `toolConfig.tools[].toolSpec` with the
  schema passed through as a document.
- **userInput** — appended by `ComposePrompt` as a text block (`""` when empty and appended).
- **toolHandlers** — only `aiserver.ToolFunc` entries are offered. On `stopReason: tool_use` each `toolUse` runs
  under an `execute_tool <name>` span; the assistant message and a user message of `toolResult` blocks are
  appended and the loop continues for at most 10 turns; a model still asking for tools after that fails the
  run with `ErrMaxSteps`. Tool errors abort the run and propagate (wrapped, `errors.Is` works). A `toolUse`
  naming a tool that was not offered fails the run with `ErrToolNotFound` (the error names the tool; its
  `execute_tool <name>` span ends ERROR), even when `ToolHandlers` has a function under that name.
- **variables** — template values for `ComposePrompt`, plus `__ld` / `ldContext` written to the root span by
  `aiserver.StartHandlerRun` (child spans never carry them).
- **Telemetry** — root `invoke_agent` (operation `invoke_agent`, system/provider `aws.bedrock`, request and
  response model = resolved `modelId`), one `chat <modelId>` child per turn, sibling `execute_tool <name>`.
  Usage on spans is cache-inclusive (`inputTokens + cacheReadInputTokens + cacheWriteInputTokens`); the
  handler's returned usage keeps Bedrock's raw camelCase keys so `aiserver.ParseUsage` folds exactly once.
  The AWS SDK's own tracer is left at its no-op default, so there is no second span representation.
- **ConverseOptions** — called per turn; `copyFields` (reflection, by field name) merges every non-zero field
  except `ModelId`, `Messages`, `System`, `ToolConfig`. `ToolConfig` is taken whole only when the handler has no
  tools, otherwise only `ToolChoice`. `toStreamInput` converts to `ConverseStreamInput`, translating
  `GuardrailConfig` to the stream type; `TestStreamInputCoversConverseInput` (in `internal/bedrockwire`) fails if a future SDK adds a field
  that cannot be copied.
- **Client** — injected client used as-is. Otherwise `config.LoadDefaultConfig` (+ `WithRegion`) per factory
  call; the SDK applies `AWS_BEARER_TOKEN_BEDROCK` and the option function then applies `APIKey`, so the
  explicit key wins; with neither, SigV4 via the default credential chain.

## OTel setup

The tracer provider is registered by `aiserver.InitClient` / `aiserver.InitWithClient` in the core package. This
package only creates spans through `otel.Tracer(PackageName)`; it never configures exporters. The OTel SDK is a
dependency of the core module, so there is no extra install step in Go. See `../aiserver` (core agents.md) for
the OTLP endpoint and resource attributes.

## `InitClient()` — when to call it

Not required: the core initialises lazily on the first `Config(...).Invoke` / `Stream` or `BedrockMessages` call.
Call `aiserver.InitClient` explicitly for custom options (SDK key, base URIs, OTLP endpoint), use
`aiserver.InitWithClient` to bring your own LaunchDarkly client (BYOC), or call it at startup to pre-warm.

## Dependencies

| Dependency | Why |
| --- | --- |
| `github.com/launchdarkly/go-ai-sdk/aiserver` | Handler contract, prompt composition, span helpers, usage reporting. |
| `github.com/launchdarkly/go-ai-sdk/internal/bedrockwire` | Converse engine shared with `bedrockagents`. |
| `github.com/aws/aws-sdk-go-v2/service/bedrockruntime` | Converse / ConverseStream client and types. |
| `github.com/aws/aws-sdk-go-v2/config` | Default AWS configuration and credential chain for constructed clients. |
| `github.com/aws/aws-sdk-go-v2/aws` | Pointer helpers. |
| `github.com/aws/smithy-go/auth/bearer` | Static bearer-token provider for `Options.APIKey`. |
| `github.com/launchdarkly/go-sdk-common/v3` | `ldcontext.Context` for the convenience wrapper. |
| test: `aws-sdk-go-v2/credentials`, `aws/protocol/eventstream`, OTel SDK `tracetest` | Static test credentials, event-stream frame encoding, span capture. |

## Common pitfalls

1. **`model.region` is not an AWS region.** It is the inference-profile prefix (`us`, `eu`, `global`). Putting
   `us-east-1` there yields `us-east-1.anthropic…` and a Bedrock validation error. Set the AWS region with
   `Options.Region` or `AWS_REGION`.
2. **Bearer auth needs HTTPS.** `APIKey` / `AWS_BEARER_TOKEN_BEDROCK` are refused by the SDK on a plain-HTTP
   endpoint override; tests use an HTTPS fake plus `AWS_CA_BUNDLE`.
3. **An injected client ignores `APIKey` and `Region`.** Configure its credentials and region when you build it.
   Also note `AWS_BEARER_TOKEN_BEDROCK` is read by `bedrockruntime.New` itself, so a client you build while that
   variable is set already uses bearer auth.
4. **`model.custom` is ignored on purpose.** Values there (guardrails, `additionalModelRequestFields`) do nothing
   unless your `ConverseOptions` callback returns them.
5. **URL history images fail.** The handler never downloads; convert to base64 first. Unknown block types are
   skipped silently, matching the other handlers.
