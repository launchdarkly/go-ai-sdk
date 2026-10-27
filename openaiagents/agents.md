# openaiagents: agent guide

## Role and tier

Tier 1 handler package. `providesFor` is `{Provider: "OpenAI", Mode: "agent"}`. It wraps the
OpenAI Responses API (`client.Responses.New` / `NewStreaming`) from `github.com/openai/openai-go/v3`
with a package-local agent loop that stands in for the OpenAI Agents SDK, which has no official
Go version.

## File map

| File | Responsibility |
| --- | --- |
| `handler.go` | Package doc, constants, sentinel errors, `Options`, `NewHandler`, `OpenAIAgents`, and the handler's blocking and streaming entry points. |
| `runner.go` | The agent / runner model: building agents from configs, request construction, the turn loop, tool and handoff execution, run hooks. |
| `outputtype.go` | Unexported `outputType` and `buildOutputType` (structured-output `text.format`). |
| `graph.go` | `OpenAIGraph` and the native adapter `ToOpenAIAgents` / `NativeGraph`. |
| `fake_test.go` | `httptest` OpenAI server, per-key fake LaunchDarkly client, span exporter helpers. |
| `handler_test.go` | Handler, wrapper, structured output and usage-report tests. |
| `graph_test.go` | Graph wrapper and native adapter tests. |
| `README.md` | User-facing install, examples, environment, and API reference. |
| `agents.md` | This file. |

Shared Responses API mapping (input items, tool params, usage, finish reasons, streamed turn)
lives in `internal/openaiwire`, also used by `openaimessages`.

## Public exports

```go
const PackageName, PackageVersion, SpanName, StreamSpanName
var ProvidesFor aiserver.ProvidesFor
var ErrMaxTurns, ErrToolNotOffered, ErrNoRoot, ErrNilGraph error
type Options struct{ Client *openai.Client; APIKey string; CaptureContent bool; MaxTurns int }
func NewHandler(Options) *aiserver.ProviderHandler
type InvokeOptions struct{ ToolHandlers; Registry; SkipJudges; Variables; Handler Options }
func OpenAIAgents(ctx, configKey, userInput string, userCtx ldcontext.Context, *InvokeOptions) (aiserver.ProviderResponse, error)
func OpenAIGraph(key string, options aiserver.GraphOptions, handler Options) *aiserver.AgentGraph
type NativeGraphOptions struct{ ToolHandlers; Context ldcontext.Context; Handler Options }
type NativeGraph struct{ /* unexported */ }
func ToOpenAIAgents(*aiserver.GraphDefinition, *NativeGraphOptions) *NativeGraph
func (*NativeGraph) Invoke(ctx, input string, opts ...aiserver.CallOption) (aiserver.GraphResponse, error)
```

`init()` calls `aiserver.RegisterAiSdkPackage(PackageName, PackageVersion)`. Usage helpers:
`openai-agents.createOpenAIAgentHandler` (`NewHandler`), `openai-agents.openaiAgents`,
`openai-agents.openaiGraph`, `openai-agents.toOpenAIAgents`. Internal paths (`newHandler`,
`internal/sdkapi.Config`, `internal/sdkapi.Graph`) never report.

## Implementation details

The handler receives `aiserver.HandlerRequest{Config, UserInput, ToolHandlers, Variables, History}`.

- **`Config`**: `aiserver.ComposePrompt(req, ModeAgent)` gives the system prompt (instructions win
  over `messages` system entries) and the turns. The system prompt is the Responses
  `instructions`. `Tools` are filtered by `aiserver.CallableTools`; tools whose handler is a
  `NativeTool` become hosted tools. `OutputFormat` becomes `buildOutputType` on the blocking path.
  Allowlisted `Model.Parameters` go out as extra JSON fields (camelCase keys are read as their
  snake_case names; snake_case wins); `max_turns` caps the run.
- **`UserInput`**: appended by `ComposePrompt`; `""` when empty, never absent.
- **`ToolHandlers`**: only offered `ToolFunc` values run; a call for anything else fails with
  `ErrToolNotOffered`. A tool error fails the run (it is not returned to the model).
- **`Variables`**: template values, plus the `__ld` / `ldContext` entries that
  `aiserver.StartHandlerRun` writes onto the root span.
- **`History`**: spliced by `ComposePrompt` into input items before `userInput`.

Run loop (`runner.run`): each turn sends the active agent's instructions, tools, handoff tools and
`text.format`; follow-up turns chain with `previous_response_id` and send only the
`function_call_output` items. A turn with no function calls is final. A `transfer_to_<agent>`
call switches the active agent (the first one per turn wins; the output is
`{"assistant":"<name>"}`), firing `handoff` then `agentStart`. `agentEnd` fires once, for the
final agent.

Handler spans: root `openai.agent.run` (or `openai.agent.run.stream`), `gen_ai.operation.name =
invoke_agent`, one `chat <model>` child per turn and one `execute_tool <name>` child per tool call.

Native graph (`NativeGraph.Invoke`): one agent per reachable node (`Traverse`), then a handoff per
outgoing edge (cycles are fine). History and input seed the root only. Hooks map to LD events
when `NativeGraphOptions.Context` is set (`nativeRun`). Each stretch one agent is active, from
`agentStart` to its `handoff` or `agentEnd`, is a node run: an `openai.agent.run` span under the
graph span (node association + context identity, `chat` / `execute_tool` children) and per-node
`$ld:ai:duration:total`, `$ld:ai:generation:success` (or `:error`) and `$ld:ai:tokens:*` (> 0).
`agentStart` → `$ld:ai:graph:node` (once per node, `nodeKey` / `index`). A handoff is tracked when
its target's node run ends: `$ld:ai:graph:handoff_success`, or `:handoff_failure` if it failed
(`sourceKey` / `targetKey`). `$ld:ai:graph:duration:total` on every path; success adds
`total_tokens` (> 0) and `invocation_success`, failure `invocation_failure`. `$ld:ai:graph:*`
events carry the graph's `TrackData` (as `aiserver.Graph`). Function tools are wrapped with
`aiserver.WrapToolHandlers` for `$ld:ai:tool_call`. The `ld.ai.graph` span has `ld.ai.graph.key`,
`launchdarkly.graph.path` and usage; it is ended on every path.

## OTel setup

Handler packages emit spans through the global tracer provider and do not configure OTel. The
provider is installed by `aiserver.InitClient` / `aiserver.InitWithClient` (OTLP over HTTP via
`InitOptions.OTLPEndpoint`). The OTel SDK is already a dependency of `aiserver`; there is no extra
to install. See the core client's [aiserver/agents.md](../aiserver/agents.md) for endpoint configuration.

## `InitClient()`: when to call it

Lazy initialization is the default: `aiserver.Config(...).Invoke` and `aiserver.ResolveGraph`
initialize the client from `LD_SDK_KEY` on first use. Call `aiserver.InitClient` explicitly for
custom options (base URIs, OTLP endpoint, service name), call `aiserver.InitWithClient` to bring
your own LaunchDarkly client (BYOC, including edge runtimes), or call it at startup to pre-warm
the connection.

## Dependencies

| Dependency | Why |
| --- | --- |
| `github.com/launchdarkly/go-ai-sdk/aiserver` | Handler contract, prompt composition, span kit, graph definition, tracking, usage reporting. |
| `github.com/launchdarkly/go-ai-sdk/internal/openaiwire` | Responses API mapping shared with `openaimessages`. |
| `github.com/openai/openai-go/v3` | Responses API client and types. |
| `github.com/launchdarkly/go-sdk-common/v3` | `ldcontext.Context` for the wrapper and the native adapter. |
| `github.com/google/uuid` | Per-run ids for native graph events. |
| `go.opentelemetry.io/otel` | The native adapter's `ld.ai.graph` span. |

## Common pitfalls

- **Follow-up turns rely on `previous_response_id`.** `store: false` in `model.parameters` breaks
  tool loops and handoffs, because OpenAI cannot chain from a response it did not store.
- **A `NativeTool`'s `ToolName` is the Responses API tool type** (`web_search`, `file_search`,
  ...), not the LD tool key. Hosted-tool settings such as `vector_store_ids` go in the config
  tool's `customParameters`.
- **`NativeGraphOptions.Context` gates all tracking.** Leaving it unset silently disables every
  LD event from the native adapter.
- **Streaming ignores `outputFormat`.** The streamed `done.output` is the concatenation of every
  text delta across turns, not a parsed object.
- **Cached tokens are already in `input_tokens`.** Do not add them again.
