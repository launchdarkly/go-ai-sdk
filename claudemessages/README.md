# claudemessages

LaunchDarkly AI SDK messages handler for Anthropic Claude, built on the Anthropic Messages API
([`anthropic-sdk-go`](https://github.com/anthropics/anthropic-sdk-go)). It turns an AI Config
variation into a Messages API request, runs function tools in a local tool loop, and reports
spans and token usage. It is the Go counterpart of the JavaScript and Python `claude-messages`
packages.

```sh
go get github.com/launchdarkly/go-ai-sdk/claudemessages
```

## providesFor

`{Provider: "Anthropic", Mode: "messages"}`. The handler runs for AI Config variations whose
`provider.name` is `Anthropic` and whose `_ldMeta.mode` is not `agent`: `messages`, or absent,
since the SDK maps every other mode to `messages`. Agent-mode Anthropic
variations go to [`claudeagents`](../claudeagents).

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
	"github.com/launchdarkly/go-ai-sdk/claudemessages"
)

func main() {
	ctx := context.Background()
	if _, err := aiserver.InitClient(ctx, aiserver.InitOptions{}); err != nil { // reads LD_SDK_KEY
		log.Fatal(err)
	}
	defer aiserver.Shutdown(ctx)

	h, err := claudemessages.NewHandler(claudemessages.Options{}) // reads ANTHROPIC_API_KEY
	if err != nil {
		log.Fatal(err)
	}
	res, err := aiserver.Config(aiserver.ConfigArgs{
		Key:      "my-ai-config",
		Handlers: []*aiserver.ProviderHandler{h},
		ToolHandlers: map[string]aiserver.ToolHandler{
			"get-weather": aiserver.ToolFunc(func(ctx context.Context, args map[string]any) (any, error) {
				return fmt.Sprintf("sunny in %v", args["city"]), nil
			}),
		},
	}).Invoke(ctx, "What is the weather in Paris?", ldcontext.New("user-123"),
		aiserver.WithVariables(map[string]any{"tone": "friendly"}))
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(res.Response)
}
```

Multiple handlers: the variation's provider and mode pick one.

```go
import (
	"github.com/launchdarkly/go-ai-sdk/claudeagents"
	"github.com/launchdarkly/go-ai-sdk/claudemessages"
)

messages, err := claudemessages.NewHandler(claudemessages.Options{})
if err != nil {
	log.Fatal(err)
}
agents, err := claudeagents.NewHandler(claudeagents.Options{})
if err != nil {
	log.Fatal(err)
}
handlers := []*aiserver.ProviderHandler{messages, agents}
res, err := aiserver.Config(aiserver.ConfigArgs{Key: "my-ai-config", Handlers: handlers}).
	Invoke(ctx, "Hello", ldcontext.New("user-123"))
```

Streaming:

```go
for ev, err := range aiserver.Config(aiserver.ConfigArgs{Key: "my-ai-config", Handlers: handlers}).
	Stream(ctx, "Tell me a story", ldcontext.New("user-123")) {
	if err != nil {
		log.Fatal(err)
	}
	if ev.Type == aiserver.EventChunk {
		fmt.Print(ev.Text)
	}
}
```

Convenience wrapper (`config()` with a fresh handler; `opts` may be `nil`):

```go
res, err := claudemessages.ClaudeMessages(ctx, "my-ai-config", "Hello", ldcontext.New("user-123"),
	&claudemessages.InvokeOptions{Variables: map[string]any{"user_input": "Hello"}})
```

## Environment variables

| Variable | Required | Description |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | Yes, unless `Options.Client` or `Options.APIKey` is set | Anthropic API key, read by `anthropic-sdk-go`. |
| `ANTHROPIC_BASE_URL` | Optional | Overrides the API base URL (read by `anthropic-sdk-go`). |
| `LD_SDK_KEY` | Yes, unless `InitOptions.SDKKey` or `aiserver.InitWithClient` is used | LaunchDarkly server SDK key, read by `aiserver`. |

## API

```go
const PackageName = "github.com/launchdarkly/go-ai-sdk/claudemessages"
const PackageVersion = "0.1.0"
const SpanName = "invoke_agent"
var ErrToolNotFound error

type Options struct {
	Client         *anthropic.Client // used when set
	APIKey         string            // used when Client is nil; else ANTHROPIC_API_KEY
	MaxRetries     *int              // retries on the client NewHandler builds; nil = SDK default; refused with Client or < 0
	CaptureContent aiserver.ContentCapture // CaptureDefault follows aiserver.Configure, else on
}
// NewHandler is createClaudeMessagesHandler. The handler has Invoke and Stream; the error reports an invalid option (see MaxRetries).
func NewHandler(Options) (*aiserver.ProviderHandler, error)

type InvokeOptions struct {
	ToolHandlers map[string]aiserver.ToolHandler
	Registry     *aiserver.Registry
	SkipJudges   bool
	Variables    map[string]any // forwarded to Invoke, not to the config
	Handler      Options
}
func ClaudeMessages(ctx context.Context, configKey, userInput string, userCtx ldcontext.Context,
	opts *InvokeOptions) (aiserver.ProviderResponse, error)
```

Behaviour summary:

- `config.messages` wins over `config.instructions`; system messages become the `system` field.
- Only config tools that have a `ToolFunc` in `ToolHandlers` are sent. A `tool_use` for any other
  name fails the run with `ErrToolNotFound` (its `execute_tool` span ends ERROR). A tool error fails the run.
- Blocking output is the last model response's text. A streamed run's `done` output is every
  turn's text concatenated.
- `config.outputFormat` is appended to the system prompt as a JSON-schema instruction on the
  blocking path only.
- These `model.parameters` keys are forwarded: `cache_control`, `output_config`, `service_tier`,
  `stop_sequences`, `temperature`, `thinking`, `tool_choice`, `top_k`, `top_p`. Any other key is
  dropped. `output_config` keeps only `effort`, and a top-level `effort` moves into
  `output_config.effort` (an explicit `output_config.effort` wins). `thinking`, `tool_choice` and
  `cache_control` keep only the sub-keys their `type` takes; a value with a missing or unknown
  `type`, or without a required sub-key, is dropped. `max_tokens` sets `max_tokens`
  (default 1024). Keys are read in snake_case only: a camelCase spelling such as `maxTokens` is
  dropped.
- `inference_geo` is not forwarded, so a config that set it gets the workspace's default inference
  region. An application that needs a fixed region can pass a `Client` built with
  `option.WithJSONSet("inference_geo", "<region>")`. It is applied after the handler serializes the
  request, so the handler cannot overwrite it, and it applies to every request the client sends.
