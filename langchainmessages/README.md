# langchainmessages

LangChain messages handler for the LaunchDarkly AI SDK for Go, built on
[langchaingo](https://github.com/tmc/langchaingo) (`github.com/tmc/langchaingo` v0.1.15).
One handler serves any provider: it picks a langchaingo model from the AI Config's
`provider.name`, or uses the model you inject.

```bash
go get github.com/launchdarkly/go-ai-sdk/langchainmessages
```

## providesFor

`ProvidesFor = {Provider: "*", Mode: "messages"}`. The `*` wildcard makes this the fallback
handler for any variation whose `_ldMeta.mode` is `messages` (or absent): the router picks it
when no handler names the variation's `provider.name` exactly. Put it last in a handler list
so exact-provider handlers win.

## Examples

Single handler, through the convenience wrapper:

```go
package main

import (
	"context"
	"fmt"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/langchainmessages"
)

func main() {
	ctx := context.Background()
	defer aiserver.Shutdown(ctx)
	resp, err := langchainmessages.LangChainMessages(ctx, "my-config", "What's the weather?",
		ldcontext.New("user-1"), &langchainmessages.InvokeOptions{
			Variables: map[string]any{"city": "Paris"},
			ToolHandlers: map[string]aiserver.ToolHandler{
				"get_weather": aiserver.ToolFunc(func(ctx context.Context, args map[string]any) (any, error) {
					return "sunny in " + fmt.Sprint(args["city"]), nil
				}),
			},
		})
	if err != nil {
		panic(err)
	}
	fmt.Println(resp.Response)
}
```

Single handler with `aiserver.Config`, an injected model, and streaming:

```go
import "github.com/tmc/langchaingo/llms/openai"

llm, err := openai.New() // reads OPENAI_API_KEY
if err != nil {
	return err
}
h := langchainmessages.NewHandler(langchainmessages.Options{Model: llm})
cfg := aiserver.Config(aiserver.ConfigArgs{Key: "my-config", Handlers: []*aiserver.ProviderHandler{h}})
for ev, err := range cfg.Stream(ctx, "Hello", ldcontext.New("user-1")) {
	if err != nil {
		return err
	}
	fmt.Print(ev.Text)
}
```

Multi-handler: exact-provider handlers first, LangChain as the `*` fallback:

```go
import "github.com/launchdarkly/go-ai-sdk/claudemessages"

cfg := aiserver.Config(aiserver.ConfigArgs{Key: "my-config", Handlers: []*aiserver.ProviderHandler{
	claudemessages.NewHandler(claudemessages.Options{}), // Anthropic/messages
	langchainmessages.NewHandler(langchainmessages.Options{}), // */messages: OpenAI, Bedrock, ...
}})
resp, err := cfg.Invoke(ctx, "Hello", ldcontext.New("user-1"))
```

Build the model per request from the evaluated config:

```go
h := langchainmessages.NewHandler(langchainmessages.Options{
	ModelFactory: func(c aiserver.AiConfig) (llms.Model, error) {
		return anthropic.New(anthropic.WithModel(c.Model.Name), anthropic.WithToken(key))
	},
})
```

## Model selection

| `Options` | Model used |
|---|---|
| `Model` set | that `llms.Model` for every call |
| `ModelFactory` set | `ModelFactory(config)` per call |
| neither | built from lowercased `provider.name`: `anthropic` → `llms/anthropic`, `bedrock` → `llms/bedrock` (model id prefixed with `model.region`), anything else → `llms/openai` |

`config.model.parameters` maps to langchaingo call options: `max_tokens`/`maxTokens`,
`temperature`, `top_p`/`topP`, `top_k`/`topK`, `seed`, `frequency_penalty`,
`presence_penalty`, `stop`/`stop_sequences`. `config.model.custom` is never read.

## Environment variables

| Variable | Read by | Required |
|---|---|---|
| `LD_SDK_KEY` | `aiserver` lazy init | Required unless you call `aiserver.InitWithClient` |
| `ANTHROPIC_API_KEY` | built-in Anthropic model | Required for `provider.name = Anthropic` without an injected model |
| `OPENAI_API_KEY` | built-in OpenAI model | Required for OpenAI / fallback providers without an injected model |
| `OPENAI_BASE_URL` / `OPENAI_API_BASE` | built-in OpenAI model | Optional: OpenAI-compatible endpoint |
| `OPENAI_ORGANIZATION` | built-in OpenAI model | Optional |
| AWS credential chain (`AWS_REGION`, `AWS_PROFILE`, `AWS_ACCESS_KEY_ID`, ...) | built-in Bedrock model | Required for `provider.name = Bedrock` without an injected model |

The package itself reads no environment variables; langchaingo and the core client do.

## API reference

```go
const PackageName = "github.com/launchdarkly/go-ai-sdk/langchainmessages"
const PackageVersion = "0.1.0"
const SpanName = "langchain.invoke"        // blocking root span
const StreamSpanName = "langchain.stream"  // streaming root span
var ProvidesFor = aiserver.ProvidesFor{Provider: "*", Mode: aiserver.ModeMessages}
var ErrMaxSteps, ErrNoToolHandler, ErrEmptyResponse error

type ModelFactory func(cfg aiserver.AiConfig) (llms.Model, error)
type Options struct {
	Model          llms.Model
	ModelFactory   ModelFactory
	CaptureContent aiserver.ContentCapture // CaptureDefault follows aiserver.Configure, else on
}
func NewHandler(opts Options) *aiserver.ProviderHandler

type InvokeOptions struct {
	ToolHandlers map[string]aiserver.ToolHandler
	Registry     *aiserver.Registry
	SkipJudges   bool
	Variables    map[string]any
	Handler      Options
}
func LangChainMessages(ctx context.Context, configKey, userInput string, userCtx ldcontext.Context, opts *InvokeOptions) (aiserver.ProviderResponse, error)
```

`config.outputFormat` (blocking only): with no callable tools the model runs in JSON mode with
the schema in the system prompt (built-in OpenAI models also bind a `json_schema`
`response_format`) and `Invoke` returns the decoded JSON value. With tools, OpenAI providers
keep JSON mode on every turn and return the string for the client to parse; other providers
run the tool loop without enforcement. Streaming ignores `outputFormat`.
