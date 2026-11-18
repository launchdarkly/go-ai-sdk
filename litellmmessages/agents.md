# litellmmessages: agent guide

## Role and tier

Tier 1 handler package. `providesFor` is `{Provider: "*", Mode: "messages"}`. It sends every
call to a LiteLLM proxy over the OpenAI-compatible Chat Completions API
(`client.Chat.Completions.New` / `NewStreaming`) from `github.com/openai/openai-go/v3`.

## File map

| File | Responsibility |
| --- | --- |
| `handler.go` | Factory, proxy client, convenience wrapper, model id, parameter allowlist, the tool loop, streaming, and span calls. |
| `handler_test.go` | Spec tests against an `httptest` LiteLLM proxy and an in-memory span exporter. |
| `fake_test.go` | The fake proxy (canned blocking and SSE answers, a 500, a held stream), fixtures, and an in-memory LaunchDarkly client. |
| `README.md` | User-facing install, examples, environment, and API reference. |
| `agents.md` | This file. |

Shared Chat Completions mapping (messages, function tools, parameter picking, `response_format`,
usage, tool-call turns, one streamed turn) lives in `internal/chatwire`. The Bedrock
inference-profile prefix rule is `handlerkit.BedrockModelID`.

## Public exports

```go
const PackageName, PackageVersion, SpanName
var ErrNoBaseURL, ErrMaxSteps error
type ClientFactory func(aiserver.AiConfig) (*openai.Client, error)
type Options struct{ BaseURL, APIKey string; Client *openai.Client; ClientFactory ClientFactory; MaxRetries *int; CaptureContent aiserver.ContentCapture }
func NewHandler(Options) (*aiserver.ProviderHandler, error)
type InvokeOptions struct{ ToolHandlers; Registry; SkipJudges; Variables; Handler Options }
func LiteLLMMessages(ctx, configKey, userInput string, userCtx ldcontext.Context, *InvokeOptions) (aiserver.ProviderResponse, error)
```

Client choice per call: `ClientFactory` (called with the evaluated config), else `Client`, else
the client `NewHandler` built from `BaseURL` / `LITELLM_BASE_URL` and `APIKey` /
`LITELLM_API_KEY` / `"not-needed"`. No URL and no injected client is `ErrNoBaseURL`.

`Options.MaxRetries` is the retry count on the built client (`option.WithMaxRetries`); nil keeps
the SDK default, 0 means one attempt. A negative count, or a count with `Client` or
`ClientFactory`, is refused with an error naming `MaxRetries`. `max_retries` / `num_retries` in
`model.parameters` are never forwarded.

`Options.CaptureContent` gates prompt, completion, tool and judge-explanation content on spans:
`aiserver.CaptureDefault` (the zero value) follows `aiserver.Configure`, else on.

`init()` calls `handlerkit.RegisterAiSdkPackage(PackageName, PackageVersion)`. `NewHandler`
reports `litellm-messages.createLiteLLMMessagesHandler`; `LiteLLMMessages` reports only
`litellm-messages.litellmMessages`.

## Implementation details

The handler receives `aiserver.HandlerRequest{Config, UserInput, ToolHandlers, Variables, History}`.

- **`Config`**: `handlerkit.ComposePrompt(req, ModeMessages)` builds the system prompt and turns.
  `requestModel` gives the model sent: `Model.Name`, with `Model.Region` applied by
  `handlerkit.BedrockModelID` to a `bedrock/` name's last segment (any provider, no ARN) or to a
  bare name under the `Bedrock` provider (case-insensitive). The same id names the `chat <model>`
  spans and `gen_ai.request.model`, and is `gen_ai.response.model` when the proxy reports no
  model. `Tools` is filtered by `handlerkit.CallableTools`. `OutputFormat` becomes
  `response_format` on every blocking turn. Allowlisted `Model.Parameters` go out as extra JSON
  fields, unchanged.
- **`ToolHandlers`**: only `ToolFunc` values are offered and executed. A model call for a tool
  that was not offered fails the run, naming the tool; the raw map is never consulted for it. A
  `NativeTool` is not offered.
- **`Variables`** / **`History`**: as in every messages handler (`ComposePrompt`,
  `StartHandlerRun`). Image blocks become `image_url` parts.

Tool loop: each proxy turn gets a `chat <model>` span, each tool call an `execute_tool <name>`
span, all direct children of the `invoke_agent` root. Each follow-up re-sends the whole
conversation: the previous messages, the assistant `tool_calls` message, and one `role: tool`
message per call. Blocking output is the final turn's text; streaming `done.output` is the
concatenation of every streamed delta.

## OTel setup

Handler packages emit spans through the global tracer provider and do not configure OTel. The
provider is installed by `aiserver.InitClient` / `aiserver.InitWithClient`. See
[aiserver/agents.md](../aiserver/agents.md) for endpoint configuration.

## Dependencies

| Dependency | Why |
| --- | --- |
| `github.com/launchdarkly/go-ai-sdk/aiserver` | Handler contract, prompt composition, span kit, usage reporting. |
| `github.com/launchdarkly/go-ai-sdk/internal/chatwire` | Chat Completions mapping. |
| `github.com/openai/openai-go/v3` | Chat Completions client and types, pointed at the proxy. |
| `github.com/launchdarkly/go-sdk-common/v3` | `ldcontext.Context` for the convenience wrapper. |

## Common pitfalls

- **`openai-go` reads `OPENAI_*` by default.** `proxyClient` overrides the base URL and key and
  deletes the organization, project and `OPENAI_CUSTOM_HEADERS` headers. Keep that when changing
  how the client is built; `TestBuiltClientTargetsOnlyTheProxy` checks it.
- **Two unscoped `*` messages handlers replace each other** in a Registry. Next to
  `langchainmessages`, scope one with `WithProviders`.
- **Cached tokens are already in `input_tokens`.** Do not add `cached_tokens` again.
- **Streaming ignores `outputFormat`.** A streamed structured-output config returns free text.
- **Tool results are strings.** Non-string results are JSON-encoded (`aiserver.ToolResultString`).
