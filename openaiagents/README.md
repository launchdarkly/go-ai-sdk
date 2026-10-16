# openaiagents

LaunchDarkly AI SDK agent handler for OpenAI, built on the
[`openai-go`](https://github.com/openai/openai-go) Responses API. Go has no official OpenAI
Agents SDK, so this package runs its own agent loop with the same observable behavior: an agent
has instructions, tools, handoffs and an optional output type; a run executes tools, follows
handoffs, and ends at the first turn with no tool calls.

```sh
go get github.com/launchdarkly/go-ai-sdk/openaiagents
```

## providesFor

`{Provider: "OpenAI", Mode: "agent"}`. The handler runs for AI Config variations whose
`provider.name` is `OpenAI` and whose mode is `agent`.

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
	"github.com/launchdarkly/go-ai-sdk/openaiagents"
)

func main() {
	ctx := context.Background()
	if _, err := aiserver.InitClient(ctx, aiserver.InitOptions{}); err != nil { // reads LD_SDK_KEY
		log.Fatal(err)
	}
	defer aiserver.Shutdown(ctx)

	h := openaiagents.NewHandler(openaiagents.Options{}) // reads OPENAI_API_KEY
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

Multiple handlers: the variation's provider and mode pick one.

```go
handlers := []*aiserver.ProviderHandler{
	openaiagents.NewHandler(openaiagents.Options{}),
	openaimessages.NewHandler(openaimessages.Options{}), // github.com/launchdarkly/go-ai-sdk/openaimessages
}
res, err := aiserver.Config(aiserver.ConfigArgs{Key: "my-ai-config", Handlers: handlers}).
	Invoke(ctx, "Hello", ldcontext.New("user-123"))
```

Convenience wrapper:

```go
res, err := openaiagents.OpenAIAgents(ctx, "my-agent", "Hello", ldcontext.New("user-123"),
	&openaiagents.InvokeOptions{Variables: map[string]any{"user_input": "Hello"}})
```

Graph convenience wrapper (`aiserver.Graph` with this handler pre-bound):

```go
g := openaiagents.OpenAIGraph("support-graph", aiserver.GraphOptions{ToolHandlers: tools}, openaiagents.Options{})
out, err := g.Invoke(ctx, "I was double charged", ldcontext.New("user-123"))
fmt.Println(out.Response, out.Usage.Total)
```

Native graph adapter (one multi-agent run with handoffs):

```go
userCtx := ldcontext.New("user-123")
def, err := aiserver.ResolveGraph(ctx, "support-graph", userCtx, aiserver.GraphOptions{})
if err != nil {
	log.Fatal(err)
}
out, err := openaiagents.ToOpenAIAgents(def, &openaiagents.NativeGraphOptions{
	ToolHandlers: tools,
	Context:      userCtx, // omit to track no events
}).Invoke(ctx, "I was double charged")
```

Hosted (built-in) tools: register a `NativeTool` under the config tool's key. Its `ToolName` is
the Responses API tool type, and the config tool's `customParameters` are sent with it.

```go
tools := map[string]aiserver.ToolHandler{"search": &aiserver.NativeTool{ToolName: "web_search"}}
```

## Environment variables

| Variable | Required | Description |
| --- | --- | --- |
| `OPENAI_API_KEY` | Yes, unless `Options.Client` or `Options.APIKey` is set | OpenAI API key, read by `openai-go`. |
| `OPENAI_BASE_URL` | Optional | Overrides the API base URL (read by `openai-go`). |
| `OPENAI_ORG_ID`, `OPENAI_PROJECT_ID` | Optional | Organization and project headers (read by `openai-go`). |
| `LD_SDK_KEY` | Yes, unless `InitOptions.SDKKey` or `aiserver.InitWithClient` is used | LaunchDarkly server SDK key, read by `aiserver`. |

## API

```go
const PackageName = "github.com/launchdarkly/go-ai-sdk/openaiagents"
const PackageVersion = "0.1.0"
const SpanName = "openai.agent.run"              // blocking root span
const StreamSpanName = "openai.agent.run.stream" // streaming root span
var ProvidesFor = aiserver.ProvidesFor{Provider: "OpenAI", Mode: aiserver.ModeAgent}
var ErrMaxTurns, ErrToolNotOffered, ErrNoRoot, ErrNilGraph error

type Options struct {
	Client         *openai.Client // used for every call when set
	APIKey         string         // used when Client is nil; else OPENAI_API_KEY
	CaptureContent bool           // default false
	MaxTurns       int            // 0 means 10; a variation's max_turns wins
}
func NewHandler(opts Options) *aiserver.ProviderHandler

type InvokeOptions struct {
	ToolHandlers map[string]aiserver.ToolHandler
	Registry     *aiserver.Registry
	SkipJudges   bool
	Variables    map[string]any
	Handler      Options
}
func OpenAIAgents(ctx context.Context, configKey, userInput string, userCtx ldcontext.Context, opts *InvokeOptions) (aiserver.ProviderResponse, error)

func OpenAIGraph(key string, options aiserver.GraphOptions, handler Options) *aiserver.AgentGraph

type OutputType struct{ Type, Name string; Strict bool; Schema map[string]any }
func BuildOutputType(outputFormat map[string]any) *OutputType

type NativeGraphOptions struct {
	ToolHandlers map[string]aiserver.ToolHandler
	Context      ldcontext.Context
	Handler      Options
}
func ToOpenAIAgents(def *aiserver.GraphDefinition, opts *NativeGraphOptions) *NativeGraph
func (g *NativeGraph) Invoke(ctx context.Context, input string, opts ...aiserver.CallOption) (aiserver.GraphResponse, error)
```

Behavior:

- The system prompt (instructions win over `config.messages` system entries) is sent as the
  Responses API `instructions`. Config turns, history and `userInput` are input items; history
  images become `input_image` parts.
- Config tools are sent as function tools only when `ToolHandlers` holds a `ToolFunc` for them; a
  `NativeTool` becomes a hosted tool. A run stops after `max_turns` model turns (default 10) with
  `ErrMaxTurns`.
- `outputFormat` becomes `BuildOutputType(outputFormat)` sent as `text.format`, and the parsed JSON
  value is returned. Unparseable output is returned as the raw string. Streaming ignores
  `outputFormat`.
- Forwarded `model.parameters`: `temperature`, `top_p`, `max_output_tokens` (also from
  `max_tokens` / `max_completion_tokens`), `parallel_tool_calls`, `tool_choice`, `reasoning`,
  `store`, `truncation`, `prompt_cache_retention`, `context_management`, and `verbosity` (or
  `text.verbosity`, which wins) as `text.verbosity`. `max_turns` caps the run. camelCase spellings
  (`topP`, `maxTokens`) are accepted; the snake_case key wins when both are set.
- Usage is `input_tokens` / `output_tokens` summed over every turn (OpenAI input already includes
  cached tokens).
