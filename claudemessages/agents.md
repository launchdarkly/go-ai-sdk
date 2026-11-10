# claudemessages: agent guide

## Role and tier

Tier 1 handler package. `providesFor` is `{Provider: "Anthropic", Mode: "messages"}`. It maps an
AI Config to one Anthropic Messages API request (`client.Messages.New` / `NewStreaming` from
`github.com/anthropics/anthropic-sdk-go`) and runs a local tool loop over config function tools.

## File map

| File | Responsibility |
| --- | --- |
| `handler.go` | Factory, convenience wrapper, model-parameter allowlist, the tool loop, streaming, span calls. |
| `fake_ld_test.go` | Fake LaunchDarkly client that records track calls, and the `withLD` test setup. |
| `handler_test.go` | `httptest` Anthropic server and the handler tests (prompt, tools, loop, telemetry, streaming, wrapper, usage events). |
| `README.md` | User-facing install, examples, environment, and API reference. |
| `agents.md` | This file. |

Shared Anthropic wire mapping (turns, tools, max_tokens, finish reasons, usage) lives in
`internal/anthropicwire`, used by this package and `claudeagents`.

## Public exports

```go
const PackageName, PackageVersion, SpanName
var ErrToolNotFound error
type Options struct{ Client *anthropic.Client; APIKey string; MaxRetries *int; CaptureContent aiserver.ContentCapture }
func NewHandler(Options) (*aiserver.ProviderHandler, error)
type InvokeOptions struct{ ToolHandlers; Registry; SkipJudges; Variables; Handler Options }
func ClaudeMessages(ctx, configKey, userInput string, userCtx ldcontext.Context, *InvokeOptions) (aiserver.ProviderResponse, error)
```

`Options.MaxRetries` is the retry count on the client `NewHandler` builds (`option.WithMaxRetries`); nil keeps the SDK default, 0 means one attempt. `NewHandler` and the convenience function refuse a negative count, or `MaxRetries` with `Client`, with an error naming `MaxRetries`.

`Options.CaptureContent` gates prompt, completion, tool and judge-explanation content on spans:
`aiserver.CaptureDefault` (the zero value) follows `aiserver.Configure`, else on;
`aiserver.CaptureOn` and `aiserver.CaptureOff` force it on or off. Model, usage, status and
`gen_ai.tool.name` are written either way.

`init()` calls `handlerkit.RegisterAiSdkPackage(PackageName, PackageVersion)`. `NewHandler` reports
`claude-messages.createClaudeMessagesHandler` and `ClaudeMessages` reports
`claude-messages.claudeMessages`; the wrapper builds its handler through the
non-reporting internal path, so one call emits only its own helper.

## Implementation details

The handler receives `aiserver.HandlerRequest{Config, UserInput, ToolHandlers, Variables, History}`.

- **`Config`**: `handlerkit.ComposePrompt(req, ModeMessages)` builds the system prompt and turns
  (`messages` win over `instructions`). On the blocking path `OutputFormat` is appended with
  `kitbase.WithOutputFormat`. `Model.Name` is the request model. `max_tokens` is read from
  `Model.Parameters`; the allowlisted parameters in `forwardedParameters` and the rebuilt
  `typedShapes` objects are sent as extra request fields, with `output_config` reduced to `effort`
  and a top-level `effort` moved into it.
- **`UserInput`**: appended by `handlerkit.ComposePrompt`; never absent, `""` when empty.
- **`ToolHandlers`**: `anthropicwire.Tools` offers each `Config.Tools` entry whose key has a
  `ToolFunc`, named by its key, with the JSON Schema rebuilt field for field. A
  `tool_use` block for a name that was not offered fails the run with `ErrToolNotFound`, naming
  the tool; its `execute_tool <name>` span ends ERROR with the root. A tool error fails the run. Tool results go back as `tool_result` blocks.
- **`Variables`**: template values for `ParseTemplate`, plus the `__ld` / `ldContext` entries that
  `handlerkit.StartHandlerRun` writes onto the root span.
- **`History`**: spliced by `handlerkit.ComposePrompt` as structured turns; images map to native Anthropic
  image blocks.

Loop: at most 10 provider turns (the JS and Python bound). One `chat <model>` span per turn and
one `execute_tool <name>` span per tool call, siblings under the `invoke_agent` root
(`gen_ai.operation.name = invoke_agent`). The blocking output is the last response's text; the stream's
`done` output and the streamed root's `gen_ai.completion.0.content` are every turn's text
concatenated. A stream consumer that stops iterating marks the root
`launchdarkly.stream.abandoned` and leaves its status unset.

## OTel setup

Handler packages emit spans through the global tracer provider and do not configure OTel. The
provider is installed by `aiserver.InitClient` / `aiserver.InitWithClient` (OTLP over HTTP via
`InitOptions.OTLPEndpoint`), with `aiserver.ConversationIDSpanProcessor` registered. The OTel SDK
is already a dependency of `aiserver`; there is no extra to install. See the core client's
[`agents.md`](../aiserver/agents.md) for endpoint configuration.

## `InitClient()`: when to call it

Lazy initialization is the default: `aiserver.Config(...).Invoke` / `Stream` and `ClaudeMessages`
initialize the client from `LD_SDK_KEY` on first use. Call `aiserver.InitClient` explicitly for
custom options (base URIs, OTLP endpoint, service name), call `aiserver.InitWithClient` to bring
your own LaunchDarkly client (BYOC, including edge runtimes), or call it at startup to pre-warm
the connection.

## Dependencies

| Dependency | Why |
| --- | --- |
| `github.com/launchdarkly/go-ai-sdk/aiserver` | Handler contract, prompt composition, span kit, usage reporting. |
| `github.com/launchdarkly/go-ai-sdk/internal/anthropicwire` | Anthropic wire mapping shared with `claudeagents`. |
| `github.com/anthropics/anthropic-sdk-go` | Messages API client, streaming accumulator. |
| `github.com/launchdarkly/go-sdk-common/v3` | `ldcontext.Context` for the convenience wrapper. |

## Common pitfalls

- **A tool needs both a config entry and a `ToolFunc`.** A `ToolHandlers` entry whose key is not
  in `config.tools` is never offered, and a config tool without a `ToolFunc` is silently left out
  of the request. If the model still names either one, the run fails with `ErrToolNotFound`.
- **Blocking and streaming outputs differ after a tool turn.** Blocking returns the last
  response's text; the stream's `done` output concatenates every turn, including text the model
  wrote before calling a tool.
- **`outputFormat` is a prompt instruction, not enforcement.** It is appended to the system
  prompt on the blocking path only; streaming ignores it, and the model can still return
  non-JSON.
- **Unlisted `model.parameters` are dropped.** Only the allowlisted keys reach the API, so a
  parameter such as `frequency_penalty` has no effect. `stream` and `model` are owned by the
  handler and cannot be overridden.
- **The tool loop stops after 10 turns.** If the model still requests tools on the tenth turn,
  the run ends with an empty output rather than an error.
