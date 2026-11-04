# langchainagents

LangChain agent handler for the LaunchDarkly AI SDK for Go, plus the `LangChainGraph`
wrapper and the `ToLangGraph` native graph adapter. Models, messages and tool calling come
from the SDK's own [`llms`](../llms) package, and graphs run on the SDK's own state-graph
runner. LangChain publishes no official Go SDK, so this package does not depend on a
community port.

```bash
go get github.com/launchdarkly/go-ai-sdk/langchainagents
```

## providesFor

`ProvidesFor = {Provider: "*", Mode: "agent"}`. The `*` wildcard makes this the fallback for
any variation whose `_ldMeta.mode` is `agent`: the router picks it when no handler names the
variation's `provider.name` exactly. List exact-provider agent handlers (`claudeagents`,
`openaiagents`, `bedrockagents`) first if you want them to win.

## Examples

Single handler, through the convenience wrapper:

```go
package main

import (
	"context"
	"fmt"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/langchainagents"
)

func main() {
	ctx := context.Background()
	defer aiserver.Shutdown(ctx)
	resp, err := langchainagents.LangChainAgents(ctx, "my-agent", "Find my last order",
		ldcontext.New("user-1"), &langchainagents.InvokeOptions{
			Variables: map[string]any{"tone": "brief"},
			ToolHandlers: map[string]aiserver.ToolHandler{
				"lookup_order": aiserver.ToolFunc(func(ctx context.Context, args map[string]any) (any, error) {
					return map[string]any{"id": 42, "status": "shipped"}, nil
				}),
			},
		})
	if err != nil {
		panic(err)
	}
	fmt.Println(resp.Text())
}
```

Single handler with `aiserver.Config` and an injected model:

```go
import "github.com/launchdarkly/go-ai-sdk/llms"

llm, err := llms.NewOpenAI(llms.OpenAIOptions{Model: "gpt-4o"}) // reads OPENAI_API_KEY
if err != nil {
	return err
}
h := langchainagents.NewHandler(langchainagents.Options{Model: llm})
resp, err := aiserver.Config(aiserver.ConfigArgs{Key: "my-agent", Handlers: []*aiserver.ProviderHandler{h}}).
	Invoke(ctx, "Hello", ldcontext.New("user-1"))
```

Multi-handler: exact-provider agents first, LangChain as the `*` fallback:

```go
import "github.com/launchdarkly/go-ai-sdk/claudeagents"

cfg := aiserver.Config(aiserver.ConfigArgs{Key: "my-agent", Handlers: []*aiserver.ProviderHandler{
	claudeagents.NewHandler(claudeagents.Options{}),       // Anthropic/agent
	langchainagents.NewHandler(langchainagents.Options{}), // */agent: OpenAI, Bedrock, ...
}})
for ev, err := range cfg.Stream(ctx, "Hello", ldcontext.New("user-1")) {
	if err != nil {
		return err
	}
	fmt.Print(ev.Text)
}
```

Graph convenience wrapper (every node runs on this handler; the `Options` value supplies
the model):

```go
out, err := langchainagents.LangChainGraph("support-graph",
	aiserver.GraphOptions{ToolHandlers: tools}, langchainagents.Options{Model: llm}).
	Invoke(ctx, "I was double charged", ldcontext.New("user-1"))
```

Native LangGraph runner: the resolved graph becomes one state graph on the SDK's own runner.

```go
user := ldcontext.New("user-1")
def := aiserver.ResolveGraph(ctx, "support-graph", user, aiserver.GraphOptions{})
out, err := langchainagents.ToLangGraph(def, &langchainagents.NativeGraphOptions{
	ToolHandlers: tools,
	Context:      user, // without it, no LaunchDarkly events are tracked
}).Invoke(ctx, "I was double charged", aiserver.WithHistory(history))
```

## API

```go
func NewHandler(opts Options) *aiserver.ProviderHandler
type Options struct {
	Model          llms.Model   // used for every call when set
	ModelFactory   ModelFactory // func(cfg aiserver.AiConfig) (llms.Model, error), per call
	CaptureContent bool         // default false
}
func LangChainAgents(ctx, configKey, userInput string, userCtx ldcontext.Context, opts *InvokeOptions) (aiserver.ProviderResponse, error)
func LangChainGraph(key string, options aiserver.GraphOptions, handler Options) *aiserver.AgentGraph
func ToLangGraph(def *aiserver.GraphDefinition, opts *NativeGraphOptions) *LangGraph
func (g *LangGraph) Invoke(ctx context.Context, input string, opts ...aiserver.CallOption) (aiserver.GraphResponse, error)
```

Errors for `errors.Is`: `ErrRecursionLimit` (more than 25 node executions in one run, LangGraph's
default), `ErrNoToolHandler`, `ErrEmptyResponse`, `ErrNoRoot`, `ErrNilGraph`,
`ErrNodeName` (a node key that, with characters outside `[a-zA-Z0-9_-]` replaced by `_`, is
`END` or another node's name), and `aiserver.ErrGraphDisabled` from `ToLangGraph` on a disabled graph.

## Model selection

| Source | Model used |
|---|---|
| `Options.Model` / `NativeGraphOptions.ModelFactory` | the model you supply |
| `Options.ModelFactory` | `ModelFactory(config)` per call |
| neither | built from lowercased `provider.name`: `anthropic` → `llms.NewAnthropic` (Messages API), `bedrock` → `llms.NewBedrock` (Converse API; model id prefixed with `model.region`), anything else → `llms.NewOpenAI` (Chat Completions, so any OpenAI-compatible endpoint works through `OPENAI_BASE_URL`) |

`config.model.parameters` maps to `llms` call options (`max_tokens`, `temperature`,
`top_p`, `top_k`, `seed`, `frequency_penalty`, `presence_penalty`, `stop`). An explicit `0`
is sent; an unset value is left to the provider's default. `config.model.custom` is never read.

## Telemetry

The handler's root span is `langchain.agent` (blocking and streaming), with `chat <model>` and
`execute_tool <name>` children. `gen_ai.system` is the lowercased `provider.name`, not
`langchain`. `ToLangGraph` opens one `ld.ai.graph` span carrying `ld.ai.graph.key`, with one
`langchain.agent` span per node run under it.

## Environment variables

| Variable | Read by | Required |
|---|---|---|
| `LD_SDK_KEY` | `aiserver` lazy init | Required unless you call `aiserver.InitWithClient` |
| `OPENAI_API_KEY` | built-in OpenAI model | Required for OpenAI / fallback providers without an injected model |
| `ANTHROPIC_API_KEY` | built-in Anthropic model | Required for `provider.name = Anthropic` without an injected model |
| `OPENAI_BASE_URL` (or `OPENAI_API_BASE`) / `OPENAI_ORGANIZATION` / `OPENAI_MODEL` | built-in OpenAI model | Optional |
| AWS credential chain (`AWS_REGION`, `AWS_PROFILE`, `AWS_BEARER_TOKEN_BEDROCK`, ...) | built-in Bedrock model | Required for `provider.name = Bedrock` without an injected model |

The handler package reads no environment variables itself; the built-in `llms` models and the
core client do.
