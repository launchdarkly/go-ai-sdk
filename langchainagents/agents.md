# langchainagents — agent notes

## Role and tier

Tier 1 handler package. The handler's `ProvidesFor` field is `{"*", "agent"}`: the wildcard fallback for any
agent-mode variation. Built on the SDK's own `llms` package; graphs run on the internal `internal/stategraph` runner. Also
exports the `LangChainGraph` wrapper and the `ToLangGraph` native adapter.

## File map

| File | Responsibility |
|---|---|
| `handler.go` | Factory, options, convenience wrapper, the handler run (prompt, spans, model, tools) for blocking and streaming. |
| `react.go` | `createReactAgent`: the create_react_agent equivalent on a `stategraph.Graph` (agent node, tools node, tools condition), the `messages` append schema, the `stateGraph` test seam. |
| `graph.go` | `LangChainGraph`, `ToLangGraph` / `LangGraph.Invoke`: one outer `stategraph.Graph` node per `GraphNode`, handoff tools returning `stategraph.Command`, per-node and graph tracking, the `launchdarkly.graph` span. |
| `handler_test.go` | Handler tests: factory, prompt, tools, tool loop, telemetry, errors, streaming, structured output, history, model routing, usage reporting. |
| `graph_test.go` | `LangChainGraph` and `ToLangGraph` tests, including one against the real `stategraph` runner. |
| `fake_test.go` | Fake `llms.Model` (scripted turns, records calls), recording fake LD client, tracer setup. |
| `README.md` | User docs. |
| `agents.md` | This file. |

Shared `llms` plumbing (model routing, message/tool conversion, usage keys, one model turn,
one tool call) lives in
[`internal/langchainwire`](../internal/langchainwire), also used by `langchainmessages`.

## Public exports

```go
const PackageName, PackageVersion, SpanName
var ErrRecursionLimit, ErrNoToolHandler, ErrEmptyResponse, ErrNoRoot, ErrNilGraph, ErrNodeName
type ModelFactory func(cfg aiserver.AiConfig) (llms.Model, error)
type Options struct{ Model llms.Model; ModelFactory ModelFactory; CaptureContent aiserver.ContentCapture }
func NewHandler(opts Options) (*aiserver.ProviderHandler, error)
type InvokeOptions struct{ ToolHandlers; Registry; SkipJudges; Variables; Handler Options }
func LangChainAgents(ctx, configKey, userInput string, userCtx ldcontext.Context, opts *InvokeOptions) (aiserver.ProviderResponse, error)
func LangChainGraph(key string, options aiserver.GraphOptions, handler Options) (*aiserver.AgentGraph, error)
type NativeGraphOptions struct{ ToolHandlers; ModelFactory func(*aiserver.GraphNode) (llms.Model, error); Context ldcontext.Context }
type LangGraph struct{ ... }
func ToLangGraph(def *aiserver.GraphDefinition, opts *NativeGraphOptions) (*LangGraph, error)
func (g *LangGraph) Invoke(ctx, input string, opts ...aiserver.CallOption) (aiserver.GraphResponse, error)
```

`Options.CaptureContent` gates prompt, completion, tool and judge-explanation content on spans:
`aiserver.CaptureDefault` (the zero value) follows `aiserver.Configure`, else on;
`aiserver.CaptureOn` and `aiserver.CaptureOff` force it on or off. Model, usage, status and
`gen_ai.tool.name` are written either way.

`init()` registers the package. Usage helpers: `langchain-agents.createLangChainAgentsHandler`,
`langchain-agents.langchainAgents`, `langchain-agents.langchainGraph`,
`langchain-agents.toLangGraph`; wrappers then use the non-reporting `newHandler`,
`internal/sdkapi.Config` / `internal/sdkapi.Graph`.

## Implementation details

- **`config`** — `ModelConfig` applies the Bedrock `model.region` prefix; lowercased
  `provider.name` is `gen_ai.provider.name` (`aws.bedrock` for `bedrock`; `gen_ai.system` is always `langchain`) and picks the built-in model. `handlerkit.ComposePrompt(req,
  ModeAgent)` gives the system prompt (instructions win over system messages) and the turns.
  `outputFormat` (blocking only) appends `kitbase.OutputFormatInstruction` to the system prompt
  (best effort). `model.parameters` → `llms.CallOption`s; `model.custom` is never read.
- **`userInput` / `history`** — the turns become the agent's initial `messages` state
  (structured `llms.MessageContent`, images as `BinaryContent` / `ImageURLContent`). The
  system prompt is prepended at each model call, never stored in state, never containing history.
- **`toolHandlers`** — `handlerkit.CallableTools` filters config tools to `ToolFunc`s; the
  tools node runs each requested call in order under an `execute_tool` span. Unoffered tool →
  `ErrNoToolHandler`, with an ERROR `execute_tool` span; tool errors propagate.
- **`variables`** — template substitution in `handlerkit.ComposePrompt`; `__ld` / `ldContext` reach the root
  span through `handlerkit.StartHandlerRun`.
- **Agent graph** — `createReactAgent` compiles `agent` (model call) → tools condition →
  `tools` → `agent` …, ending at `END` when the model answers without tool calls. State is
  `map[string]any{"messages": []llms.MessageContent}` carried as `any`; `messages` is appended to, other keys
  are overwritten. 25 node executions max (`ErrRecursionLimit`); `stategraph` has no limit of
  its own.
- **ToLangGraph** — the outer `stategraph.Graph` has one node per `GraphNode` (`Traverse` order),
  entry point = root (`SetEntryPoint` is LangGraph's `START`), an edge to
  `END` for terminal and multi-child nodes, an edge to the child for single-child nodes. Each node
  function runs that node's ReAct agent (its own inner `stategraph.Graph`, which holds the ToolNode
  equivalent) over the graph's messages. A `transfer_to_<target>` tool returns
  `&stategraph.Command{Goto: target}`; the inner graph ends, and the outer node returns a `Command`
  with the new messages, which the runner follows instead of the static edge. History is read by
  the root node only and never enters the shared state. Node names are sanitized keys; a name
  equal to `END` or shared by two nodes fails with `ErrNodeName` (`AddNode` would silently replace).
- **Tracking (ToLangGraph, only with `NativeGraphOptions.Context`)** — per first node visit
  `$ld:ai:graph:node {nodeKey, index}`; per node run `$ld:ai:duration:total`,
  `$ld:ai:generation:success` (or `:error`), `$ld:ai:tokens:{total,input,output}` when > 0;
  `$ld:ai:tool_call` through `handlerkit.WrapToolHandlers` (NativeTools via `handlerkit.TrackNativeTool`);
  for every hop into a node (a handoff or the single static edge), after that node has run,
  `$ld:ai:graph:handoff_success {sourceKey, targetKey}` or `:handoff_failure` when it failed;
  `$ld:ai:graph:duration:total` on every path; on success `$ld:ai:graph:total_tokens` (when > 0)
  and `$ld:ai:graph:invocation_success`; on error `$ld:ai:graph:invocation_failure`.
  `$ld:ai:graph:*` events carry the graph's `TrackData` (as `aiserver.Graph`), per-node events the
  node's. `$ld:ai:graph:path` is never tracked.
- **Spans** — handler: root `invoke_agent` (`gen_ai.operation.name = invoke_agent`),
  `chat <model>` / `execute_tool <name>` children. ToLangGraph: `launchdarkly.graph` with
  `launchdarkly.graph.key`, LD association attributes and the context identity, usage,
  `launchdarkly.graph.path`; ended on every path. A node run opens no root span: its `chat` /
  `execute_tool` spans (handoff and NativeTool calls included) are children of the graph span.
- **Streaming** — `llms.WithStreamingFunc` deltas are `chunk` events; `done.output` joins every
  turn's text. Node functions run on the caller's goroutine, one at a time, so
  `yield` is never called concurrently.

## OTel setup

The tracer provider is registered by `aiserver.InitClient` / `aiserver.InitWithClient` in the
core client. This package only emits spans through the global provider; it never configures
OpenTelemetry. No extra install (`go get github.com/launchdarkly/go-ai-sdk/aiserver`). Endpoint
options: [`aiserver`](../aiserver) (`InitOptions.OTLPEndpoint`).

## `InitClient()` — when to call it

Not required: `aiserver.Config(...).Invoke` / `Stream`, `Graph` and `ResolveGraph` initialize the
client lazily from `LD_SDK_KEY`. Call `aiserver.InitClient` for custom options,
`aiserver.InitWithClient` to bring your own client (tests, edge runtimes), or either at startup
to pre-warm.

## Dependencies

| Dependency | Why |
|---|---|
| `github.com/launchdarkly/go-ai-sdk/aiserver` | Handler contract, prompt composition, span helpers, graph definition, tracking. |
| `github.com/launchdarkly/go-ai-sdk/internal/langchainwire` | `llms` plumbing shared with `langchainmessages`. |
| `github.com/launchdarkly/go-ai-sdk/llms` | `llms.Model`, message and tool types, built-in models (routed via langchainwire). Replaces langchaingo. |
| `github.com/launchdarkly/go-ai-sdk/internal/stategraph` | `Graph`, `Command`, `END`: the state-graph runner (replaces langgraphgo). |
| `github.com/google/uuid` | Per-run id for graph tracking. |
| `go.opentelemetry.io/otel` | `launchdarkly.graph` span. |
| `github.com/launchdarkly/go-sdk-common/v3/ldcontext` | Context arguments. |

## Common pitfalls

- **No tracking without `NativeGraphOptions.Context`.** The zero context means "do not track",
  matching the JS/Python `context` option; spans are still emitted.
- **A handed-off node sees the source node's `transfer_to_*` tool call** in its messages but not
  the tool definition. OpenAI accepts this; Anthropic may reject tool blocks for undefined tools.
- **An injected `Options.Model` ignores `config.model.name`.** Use `ModelFactory` when the
  variation should pick the model.
