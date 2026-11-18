# litellmagents

LaunchDarkly AI SDK agent handler for a [LiteLLM](https://docs.litellm.ai/) proxy. Every call goes
to the proxy over the OpenAI-compatible Chat Completions API (with
[`openai-go`](https://github.com/openai/openai-go)), and the proxy routes the model name to its
upstream, so one handler serves any provider the proxy is configured for:
`anthropic/claude-sonnet-4`, `gemini/gemini-2.5-pro`, `bedrock/...`, or a proxy alias.

Go has no OpenAI Agents SDK, so this package runs the same agent loop as `openaiagents`: an agent
has instructions, tools, handoffs and an optional output schema; a run executes tools, follows
handoffs, and ends at the first turn with no tool calls.

```sh
go get github.com/launchdarkly/go-ai-sdk/litellmagents
```

## providesFor

`{Provider: "*", Mode: "agent"}`. The handler runs for agent-mode variations of any provider.

## Examples

Single handler with `config()`:

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/litellmagents"
)

func main() {
	ctx := context.Background()
	if _, err := aiserver.InitClient(ctx, aiserver.InitOptions{}); err != nil { // reads LD_SDK_KEY
		log.Fatal(err)
	}
	defer aiserver.Shutdown(ctx)

	h, err := litellmagents.NewHandler(litellmagents.Options{}) // reads LITELLM_BASE_URL and LITELLM_API_KEY
	if err != nil {
		log.Fatal(err) // ErrNoBaseURL without a proxy URL
	}
	tools := map[string]aiserver.ToolHandler{
		"get_weather": aiserver.ToolFunc(func(ctx context.Context, args map[string]any) (any, error) {
			return map[string]any{"city": args["city"], "forecast": "sunny"}, nil
		}),
	}
	res, err := aiserver.Config(aiserver.ConfigArgs{Key: "my-agent", Handlers: []*aiserver.ProviderHandler{h}, ToolHandlers: tools}).
		Invoke(ctx, "What is the weather in Paris?", ldcontext.New("user-123"))
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(res.Response)
}
```

Explicit proxy, or a client per call:

```go
h, err := litellmagents.NewHandler(litellmagents.Options{BaseURL: "http://localhost:4000", APIKey: "sk-proxy"})

h, err = litellmagents.NewHandler(litellmagents.Options{
	ClientFactory: func(cfg aiserver.AiConfig) (*openai.Client, error) {
		c := openai.NewClient(option.WithBaseURL(proxyFor(cfg.Model.Name)), option.WithAPIKey(key))
		return &c, nil
	},
})
```

Convenience wrapper:

```go
res, err := litellmagents.LiteLLMAgents(ctx, "my-agent", "Hello", ldcontext.New("user-123"),
	&litellmagents.InvokeOptions{Handler: litellmagents.Options{BaseURL: "http://localhost:4000"}})
```

Graph convenience wrapper (`aiserver.Graph` with this handler pre-bound):

```go
g, err := litellmagents.LiteLLMGraph("support-graph", aiserver.GraphOptions{ToolHandlers: tools}, litellmagents.Options{})
if err != nil {
	log.Fatal(err)
}
out, err := g.Invoke(ctx, "I was double charged", ldcontext.New("user-123"))
```

Native graph adapter (one multi-agent run with handoffs, every node on the same proxy):

```go
userCtx := ldcontext.New("user-123")
def, err := aiserver.ResolveGraph(ctx, "support-graph", userCtx, aiserver.GraphOptions{})
if err != nil {
	log.Fatal(err)
}
g, err := litellmagents.ToLiteLLMAgents(def, &litellmagents.NativeGraphOptions{
	ToolHandlers: tools,
	Context:      userCtx, // omit to track no events
})
if err != nil {
	log.Fatal(err)
}
out, err := g.Invoke(ctx, "I was double charged") // ErrNoBaseURL here without a proxy URL
```

## Environment variables

| Variable | Required | Description |
| --- | --- | --- |
| `LITELLM_BASE_URL` | Yes, unless `Options.BaseURL`, `Options.Client` or `Options.ClientFactory` is set | LiteLLM proxy URL. |
| `LITELLM_API_KEY` | Optional | Proxy key, unless `Options.APIKey` is set. A proxy without authentication needs none. |
| `LD_SDK_KEY` | Yes, unless `InitOptions.SDKKey` or `aiserver.InitWithClient` is used | LaunchDarkly server SDK key, read by `aiserver`. |

`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_ORG_ID` and `OPENAI_PROJECT_ID` are never sent to
the proxy, and there is no fallback to the OpenAI API.

## API

```go
const PackageName = "github.com/launchdarkly/go-ai-sdk/litellmagents"
const PackageVersion string // this release's version
const SpanName = "invoke_agent" // root span, blocking and streaming
var ErrNoBaseURL, ErrMaxTurns, ErrToolNotOffered, ErrNoRoot, ErrNilGraph error

type Options struct {
	BaseURL        string // else LITELLM_BASE_URL
	APIKey         string // else LITELLM_API_KEY
	Client         *openai.Client // used for every call when set; must point at the proxy
	ClientFactory  func(aiserver.AiConfig) (*openai.Client, error) // per call; wins over Client
	CaptureContent aiserver.ContentCapture // CaptureDefault follows aiserver.Configure, else on
	MaxTurns       int // 0 means 10; a variation's max_turns wins
}
func NewHandler(opts Options) (*aiserver.ProviderHandler, error)

type InvokeOptions struct {
	ToolHandlers map[string]aiserver.ToolHandler
	Registry     *aiserver.Registry
	SkipJudges   bool
	Variables    map[string]any
	Handler      Options
}
func LiteLLMAgents(ctx context.Context, configKey, userInput string, userCtx ldcontext.Context, opts *InvokeOptions) (aiserver.ProviderResponse, error)

func LiteLLMGraph(key string, options aiserver.GraphOptions, handler Options) (*aiserver.AgentGraph, error)

type NativeGraphOptions struct {
	ToolHandlers map[string]aiserver.ToolHandler
	Context      ldcontext.Context
	Handler      Options
}
func ToLiteLLMAgents(def *aiserver.GraphDefinition, opts *NativeGraphOptions) (*NativeGraph, error)
func (g *NativeGraph) Invoke(ctx context.Context, input string, opts ...aiserver.CallOption) (aiserver.GraphResponse, error)
```

Behavior:

- The model name is sent as configured. For Bedrock, `model.region` (`us`, `eu`, `global`, ...)
  is prepended once to the model id: after any `bedrock/<route>/` prefix, whatever the provider,
  and to a bare name when the provider is Bedrock. An id that already has a cross-region prefix,
  and an ARN, is unchanged. Spans report the id that was sent.
- The system prompt (instructions win over `config.messages` system entries) is the first
  message; config turns, history and `userInput` follow. History images become `image_url` parts.
- Config tools are sent as function tools only when `ToolHandlers` holds a `ToolFunc` for them.
  Chat Completions has no hosted tools, so a `NativeTool` is not sent and is logged once per tool
  key. A run stops after `max_turns` model turns (default 10) with `ErrMaxTurns`.
- `outputFormat` is sent as `response_format` (`{"type": "json_schema", "json_schema": {"name":
  "output", "strict": false, "schema": ...}}`, the schema root typed `object`) on every blocking
  turn, and the parsed JSON value is returned. Streaming ignores `outputFormat`.
- Forwarded `model.parameters`: `temperature`, `top_p`, `max_tokens`, `frequency_penalty`,
  `presence_penalty`, `parallel_tool_calls`, `tool_choice` (a string only), `reasoning.effort`
  (a string only, as `reasoning_effort`), and `verbosity` (or `text.verbosity`, which wins).
  `max_turns` caps the run. camelCase spellings (`topP`, `maxTokens`) are read; the snake_case key
  wins when both are set. Every other key, including `store` and LiteLLM's own routing,
  credential and mock keys, is dropped.
- Every turn resends the whole conversation (Chat Completions keeps no server-side state). After a
  handoff, the target's instructions replace the system message.
- Usage is `input_tokens` / `output_tokens` summed over every turn (input already includes cached
  tokens). Streaming requests usage in the final chunk.
- `gen_ai.provider.name` is the configured provider, lowercased, or `litellm` when unset.

## Differences from the JavaScript package

- Go has no OpenAI Agents SDK: the loop is this module's own `internal/agentloop`.
