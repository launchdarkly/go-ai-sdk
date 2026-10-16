# langchainmessages — agent notes

## Role and tier

Tier 1 handler package. `ProvidesFor = {"*", "messages"}`: the wildcard fallback for any
messages-mode variation. Built on the SDK's own [`llms`](../llms) package (model interface plus
built-in OpenAI Chat Completions, Anthropic Messages and Bedrock Converse models). LangChain has
no official Go SDK, and the community ports were removed (AIC-3609).

## File map

| File | Responsibility |
|---|---|
| `handler.go` | Factory, convenience wrapper, tool loop, streaming. Model routing, prompt/tool mapping and usage extraction live in [`internal/langchainwire`](../internal/langchainwire), shared with `langchainagents`. |
| `bedrock_zero_test.go` | Explicit `temperature: 0` reaches a fake Bedrock Converse endpoint through the real handler. |
| `handler_test.go` | Handler tests: prompt paths, history and images, tools, tool loop, telemetry, errors, streaming, structured output, wrapper, usage reporting. |
| `fake_test.go` | Fake `llms.Model` (scripted turns, records messages and call options) and recording fake LD client. |
| `README.md` | User docs. |
| `agents.md` | This file. |

## Public exports

```go
const PackageName, PackageVersion, SpanName, StreamSpanName
var ProvidesFor
var ErrMaxSteps, ErrNoToolHandler, ErrEmptyResponse
type ModelFactory func(cfg aiserver.AiConfig) (llms.Model, error)
type Options struct{ Model llms.Model; ModelFactory ModelFactory; CaptureContent bool }
func NewHandler(opts Options) *aiserver.ProviderHandler
type InvokeOptions struct{ ToolHandlers; Registry; SkipJudges; Variables; Handler Options }
func LangChainMessages(ctx, configKey, userInput string, userCtx ldcontext.Context, opts *InvokeOptions) (aiserver.ProviderResponse, error)
```

`init()` calls `aiserver.RegisterAiSdkPackage(PackageName, PackageVersion)`. `NewHandler` reports
`langchain-messages.createLangChainHandler`, `LangChainMessages` reports
`langchain-messages.langchainMessages`; both then use the non-reporting `newHandler` and
`aiserver.ConfigInternal`.

## Implementation details

- **`config`** — `langchainwire.ModelConfig` applies the Bedrock `model.region` prefix. `provider.name`
  (lowercased, `openai` when empty) is `gen_ai.system` and picks the built-in model.
  `model.parameters` maps to `llms.CallOption`s; `model.custom` is never read. `tools` are
  filtered by `aiserver.CallableTools` and sent with `llms.WithTools`, sorted by name.
  `outputFormat` (blocking only) adds `llms.WithJSONMode()` plus the schema instruction on the
  system prompt, and for the built-in OpenAI model an `openai.WithResponseFormat` json_schema.
- **`userInput` / `history`** — `aiserver.ComposePrompt(req, ModeMessages)` decides the turns;
  `langchainwire.ToMessages` maps them to `llms.MessageContent` (system → `ChatMessageTypeSystem`,
  user → `Human`, assistant → `AI`). Base64 images become `llms.BinaryContent`, URL images
  `llms.ImageURLContent`.
- **`toolHandlers`** — only `aiserver.ToolFunc` values are callable. Each requested tool runs in
  order under an `execute_tool <name>` span; the assistant turn is replayed with
  `resp.AssistantMessage()` and each result goes back as one `ChatMessageTypeTool` message with a
  `llms.ToolCallResponse`. Tool errors propagate (wrapped, `errors.Is` holds). A request for an
  unregistered tool returns `ErrNoToolHandler` (its `execute_tool` span ends ERROR); more than 10 tool rounds returns `ErrMaxSteps`.
- **`variables`** — template substitution in `ComposePrompt`; the `__ld` / `ldContext` entries
  are written to the root span by `aiserver.StartHandlerRun`.
- **Spans** — root `langchain.invoke` (blocking) / `langchain.stream` (streaming), `chat <model>`
  per turn and `execute_tool <name>` per call as siblings under the root (`aiserver.HandlerRun`).
- **Usage** — read from `ContentChoice.GenerationInfo`, first choice that has it:
  Anthropic `InputTokens`/`OutputTokens`/`CacheReadInputTokens`/`CacheCreationInputTokens`
  (input is uncached; cache is folded in), OpenAI `PromptTokens`/`CompletionTokens`/
  `PromptCachedTokens` (input already includes cache), Bedrock `input_tokens`/`output_tokens`.
  The handler returns `{input_tokens, output_tokens}` with cache-inclusive input.
- **Streaming** — `llms.WithStreamingFunc`; each text chunk is one `chunk` event. Breaking out
  of the iterator makes the callback return an error, which stops the provider stream, and the
  spans end as abandoned. `done.output` is every turn's text joined.

## OTel setup

The tracer provider is registered by `aiserver.InitClient` / `aiserver.InitWithClient` in the
core client. This package only emits spans through the global provider; it never configures
OpenTelemetry. The OTel SDK is already a dependency of `aiserver`, so there is no extra install
(`go get github.com/launchdarkly/go-ai-sdk/aiserver`). Endpoint options are in
[`aiserver`](../aiserver) (`InitOptions.OTLPEndpoint`).

## `InitClient()` — when to call it

Not required: `aiserver.Config(...).Invoke` / `Stream` initialize the client lazily from
`LD_SDK_KEY`. Call `aiserver.InitClient` yourself to pass custom options (base URIs, OTLP
endpoint), `aiserver.InitWithClient` to bring your own LD client (tests, edge runtimes), or
either one at startup to pre-warm before the first request.

## Dependencies

| Dependency | Why |
|---|---|
| `github.com/launchdarkly/go-ai-sdk/aiserver` | Handler contract, prompt composition, span helpers, usage reporting. |
| `github.com/launchdarkly/go-ai-sdk/llms` | `llms.Model` interface, message and tool types, and the built-in models routed by `provider.name`. |
| `github.com/launchdarkly/go-sdk-common/v3/ldcontext` | `LangChainMessages` context argument. |

## Common pitfalls

- **An injected `Options.Model` is used for every provider.** It ignores `config.model.name`
  unless your model reads it; use `ModelFactory` when the config should pick the model.
- **Bedrock Converse takes inline images only.** An `https://` image URL fails on the built-in
  Bedrock model; `data:` URLs and base64 images work. OpenAI and Anthropic accept both.
- **OpenAI JSON mode needs the word "JSON" in the prompt.** The handler adds the schema
  instruction for this; a custom model that sets JSON mode on its own must do the same.
- **`StreamingFunc` receives text only.** The built-in models merge streamed tool calls into
  the final response; a custom model must not send tool-call deltas through it, or they are
  streamed to the caller as text.
- **Anthropic returns one `ContentChoice` per content block, each carrying the full message
  usage.** Summing usage across choices would multiply it; the handler reads one.
