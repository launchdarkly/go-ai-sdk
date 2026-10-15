# openaimessages: agent guide

## Role and tier

Tier 1 handler package. `providesFor` is `{Provider: "OpenAI", Mode: "messages"}`. It wraps the
OpenAI Responses API (`client.Responses.New` / `NewStreaming`) from `github.com/openai/openai-go/v3`.

## File map

| File | Responsibility |
| --- | --- |
| `handler.go` | Factory, convenience wrapper, prompt/tool mapping, the tool loop, streaming, and span calls. |
| `handler_test.go` | Spec tests against an `httptest` OpenAI server and an in-memory span exporter. |
| `fake_ld_test.go` | In-memory LaunchDarkly client that serves one variation and records track calls. |
| `README.md` | User-facing install, examples, environment, and API reference. |
| `agents.md` | This file. |

Shared Responses API mapping (input items, tool params, usage, finish reasons, streamed turn)
lives in `internal/openaiwire`, also used by `openaiagents`.

## Public exports

```go
const PackageName, PackageVersion, SpanName
var ProvidesFor aiserver.ProvidesFor
var ErrMaxSteps error
type Options struct{ Client *openai.Client; APIKey string; CaptureContent bool }
func NewHandler(Options) *aiserver.ProviderHandler
type InvokeOptions struct{ ToolHandlers; Registry; SkipJudges; Variables; Handler Options }
func OpenAIMessages(ctx, configKey, userInput string, userCtx ldcontext.Context, *InvokeOptions) (aiserver.ProviderResponse, error)
```

`init()` calls `aiserver.RegisterAiSdkPackage(PackageName, PackageVersion)`.

## Implementation details

The handler receives `aiserver.HandlerRequest{Config, UserInput, ToolHandlers, Variables, History}`.

- **`Config`**: `aiserver.ComposePrompt(req, ModeMessages)` builds the system prompt and turns
  (messages win over instructions; prompt and history rules live there). `Model.Name` is the request
  model. `Tools` is filtered by `aiserver.CallableTools`. `OutputFormat` becomes `text.format` on
  the first blocking call. Allowlisted `Model.Parameters` go out as extra JSON fields.
- **`UserInput`**: appended by `ComposePrompt`; never absent, `""` when empty.
- **`ToolHandlers`**: only `ToolFunc` values are offered and executed. A model call for a tool that
  was not offered fails the run; the raw map is never consulted for it.
- **`Variables`**: template values for `ParseTemplate`, plus the `__ld` / `ldContext` entries that
  `aiserver.StartHandlerRun` writes onto the root span.
- **`History`**: spliced by `ComposePrompt`; `toInput` maps image blocks to `input_image`.

Tool loop: each provider turn gets a `chat <model>` span, each tool call an `execute_tool <name>`
span, all direct children of the `openai.messages` root. Follow-ups send only the
`function_call_output` items with `previous_response_id`. Blocking output is the final response's
`output_text`; streaming `done.output` is the concatenation of every streamed delta.

## OTel setup

Handler packages emit spans through the global tracer provider and do not configure OTel. The
provider is installed by `aiserver.InitClient` / `aiserver.InitWithClient` (OTLP over HTTP via
`InitOptions.OTLPEndpoint`). The OTel SDK is already a dependency of `aiserver`; there is no extra
to install. See the core client's [aiserver/agents.md](../aiserver/agents.md) for endpoint configuration.

## `InitClient()`: when to call it

Lazy initialization is the default: `aiserver.Config(...).Invoke` initializes the client from
`LD_SDK_KEY` on first use. Call `aiserver.InitClient` explicitly for custom options (base URIs,
OTLP endpoint, service name), call `aiserver.InitWithClient` to bring your own LaunchDarkly client
(BYOC, including edge runtimes), or call it at startup to pre-warm the connection.

## Dependencies

| Dependency | Why |
| --- | --- |
| `github.com/launchdarkly/go-ai-sdk/aiserver` | Handler contract, prompt composition, span kit, usage reporting. |
| `github.com/launchdarkly/go-ai-sdk/internal/openaiwire` | Responses API mapping shared with `openaiagents`. |
| `github.com/openai/openai-go/v3` | Responses API client and types. |
| `github.com/launchdarkly/go-sdk-common/v3` | `ldcontext.Context` for the convenience wrapper. |

## Common pitfalls

- **Cached tokens are already in `input_tokens`.** Do not add `cached_tokens` to the input count
  or return it as `cache_read_input_tokens`; `aiserver.ParseUsage` would fold it in a second time.
- **Follow-up turns rely on `previous_response_id`** for the conversation; `tools` and
  `text.format` are re-sent each turn because the Responses API does not carry them over.
  A request with `store: false` in `model.parameters` breaks tool loops, because OpenAI cannot
  chain from a response it did not store.
- **Streaming ignores `outputFormat`.** A streamed structured-output config returns free text.
- **Tool results are strings.** Non-string results are JSON-encoded (`aiserver.ToolResultString`);
  a tool returning a struct without JSON tags sends Go field names to the model.
