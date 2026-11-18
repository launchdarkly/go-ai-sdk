# litellmagents: agent guide

## Role and tier

Tier 1 handler package. `providesFor` is `{Provider: "*", Mode: "agent"}`. It sends every call to
a LiteLLM proxy over Chat Completions (`client.Chat.Completions.New` / `NewStreaming`) from
`github.com/openai/openai-go/v3`, with the shared agent loop in `internal/agentloop` standing in
for the OpenAI Agents SDK; this package supplies the agent type and the Chat Completions model.

## File map

| File | Responsibility |
| --- | --- |
| `handler.go` | Package doc, constants, sentinel errors, `Options` and the proxy connection (`connect`), `NewHandler`, `LiteLLMAgents`, the handler's blocking and streaming entry points. |
| `agent.go` | The agent (tools, handoff tools, request construction, parameter allowlist, output schema), the Bedrock model id (`requestModel`), and `chatModel`, the Chat Completions wire for `internal/agentloop`. |
| `graph.go` | `LiteLLMGraph` and the native adapter `ToLiteLLMAgents` / `NativeGraph`. |
| `fake_test.go` | `httptest` Chat Completions proxy (JSON, SSE, hanging stream), OpenAI tripwire, fake LaunchDarkly client, span helpers. |
| `handler_test.go` | Handler, connection, Bedrock id, tools, streaming, telemetry and usage-report tests. |
| `parameters_test.go` | Allowlist and never-forwarded parameter tests. |
| `graph_test.go` | Graph wrapper and native adapter tests. |
| `README.md` | User-facing install, examples, environment, and API reference. |
| `agents.md` | This file. |

Shared Chat Completions mapping (messages, function tools, response format, usage, tool-call
replay, streamed turn) lives in `internal/chatwire`.

## Public exports

```go
const PackageName, PackageVersion, SpanName
var ErrNoBaseURL, ErrMaxTurns, ErrToolNotOffered, ErrNoRoot, ErrNilGraph error
type Options struct{ BaseURL, APIKey string; Client *openai.Client; ClientFactory func(aiserver.AiConfig) (*openai.Client, error); CaptureContent aiserver.ContentCapture; MaxTurns int }
func NewHandler(Options) (*aiserver.ProviderHandler, error)
type InvokeOptions struct{ ToolHandlers; Registry; SkipJudges; Variables; Handler Options }
func LiteLLMAgents(ctx, configKey, userInput string, userCtx ldcontext.Context, *InvokeOptions) (aiserver.ProviderResponse, error)
func LiteLLMGraph(key string, options aiserver.GraphOptions, handler Options) (*aiserver.AgentGraph, error)
type NativeGraphOptions struct{ ToolHandlers; Context ldcontext.Context; Handler Options }
type NativeGraph struct{ /* unexported */ }
func ToLiteLLMAgents(*aiserver.GraphDefinition, *NativeGraphOptions) (*NativeGraph, error)
func (*NativeGraph) Invoke(ctx, input string, opts ...aiserver.CallOption) (aiserver.GraphResponse, error)
```

`init()` calls `handlerkit.RegisterAiSdkPackage(PackageName, PackageVersion)`. Usage helpers:
`litellm-agents.createLiteLLMAgentsHandler` (`NewHandler`), `litellm-agents.litellmAgents`,
`litellm-agents.litellmGraph`, `litellm-agents.toLiteLLMAgents`. Each reports only itself:
`LiteLLMAgents` and `LiteLLMGraph` build their handler with the unexported `newHandler`.

## Implementation details

- **Connection** (`Options.connect`): `ClientFactory` (called per call with the evaluated config;
  for the native graph, the root's) wins over `Client`, which wins over a client built from
  `BaseURL` / `LITELLM_BASE_URL` and `APIKey` / `LITELLM_API_KEY`. Without any of them,
  `NewHandler`, `LiteLLMGraph` and `LiteLLMAgents` return `ErrNoBaseURL`; `ToLiteLLMAgents`
  returns it from `Invoke`. The built client overrides the `OPENAI_*` defaults openai-go applies
  (base URL, API key, admin key, organization and project headers).
- **Config**: `handlerkit.ComposePrompt(req, ModeAgent)` gives the system prompt and turns. The
  system prompt is the first message of every turn, so a handoff target's instructions replace the
  source's. `requestModel` applies the Bedrock region prefix (route-aware, as the JavaScript
  package does). Allowlisted parameters go out as extra JSON fields after
  `agentloop.SnakeParameters`.
- **Tools**: `handlerkit.CallableTools`; a `NativeTool` is skipped and warned about once per tool
  key per process (`warnNativeSkipped`). A call for a tool that was not offered fails with
  `ErrToolNotOffered`; a tool error fails the run.
- **Run loop**: `agentloop.Runner` with `chatModel`, which keeps the conversation (turns, assistant
  tool-call messages, `role: tool` results) and resends it every turn, across handoffs.

Handler spans: root `invoke_agent` (blocking and streaming), `gen_ai.system = litellm`,
`gen_ai.provider.name` = the lowercased provider or `litellm`, one `chat <sent model id>` child
per turn and one `execute_tool <name>` child per tool call.

Native graph: the same structure and events as `openaiagents.ToOpenAIAgents` (see its
agents.md): one agent per reachable node, a handoff per outgoing edge, node runs as children of
the `launchdarkly.graph` span, `$ld:ai:graph:*` events only with `NativeGraphOptions.Context`. One
proxy client serves every node; each node's model id is its own. A missing proxy, a failing
`ClientFactory`, a rootless definition and a run error share one error path: the graph span is
ERROR and `$ld:ai:graph:invocation_failure` is tracked. A nil or disabled definition returns
before the span opens.

## OTel setup

Handler packages emit spans through the global tracer provider and do not configure OTel. See
[aiserver/agents.md](../aiserver/agents.md).

## Dependencies

| Dependency | Why |
| --- | --- |
| `github.com/launchdarkly/go-ai-sdk/aiserver` | Handler contract, prompt composition, span kit, graph definition, tracking, usage reporting. |
| `github.com/launchdarkly/go-ai-sdk/internal/chatwire` | Chat Completions mapping. |
| `github.com/launchdarkly/go-ai-sdk/internal/agentloop` | The agent run loop: turns, tool and handoff execution, run hooks. |
| `github.com/openai/openai-go/v3` | Chat Completions client pointed at the proxy. |
| `github.com/launchdarkly/go-sdk-common/v3` | `ldcontext.Context`. |
| `github.com/google/uuid` | Per-run ids for native graph events. |
| `go.opentelemetry.io/otel` | The native adapter's `launchdarkly.graph` span. |

## Common pitfalls

- **An injected `Client` must point at the proxy.** The handler sends the model name unchanged;
  `OPENAI_BASE_URL` is not a fallback.
- **The native graph's `ClientFactory` sees the root's config only.** One client serves the whole
  run.
- **`NativeTool` is never offered.** Use `openaiagents` or `claudeagents` for provider built-ins.
- **Streaming ignores `outputFormat`.**
- **`OPENAI_CUSTOM_HEADERS` is still applied by openai-go** to the client this package builds.
  Inject a `Client` if that environment variable is set for another purpose.
