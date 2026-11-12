# openaimessages

LaunchDarkly AI SDK handler for the OpenAI Responses API, built on
[`openai-go`](https://github.com/openai/openai-go).

```sh
go get github.com/launchdarkly/go-ai-sdk/openaimessages
```

## providesFor

`{Provider: "OpenAI", Mode: "messages"}`. The handler runs for AI Config variations whose
`provider.name` is `OpenAI` and whose mode is `completion` (normalized to `messages`).

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
	"github.com/launchdarkly/go-ai-sdk/openaimessages"
)

func main() {
	ctx := context.Background()
	if _, err := aiserver.InitClient(ctx, aiserver.InitOptions{}); err != nil { // reads LD_SDK_KEY
		log.Fatal(err)
	}
	defer aiserver.Shutdown(ctx)

	h, err := openaimessages.NewHandler(openaimessages.Options{}) // reads OPENAI_API_KEY
	if err != nil {
		log.Fatal(err)
	}
	res, err := aiserver.Config(aiserver.ConfigArgs{Key: "my-ai-config", Handlers: []*aiserver.ProviderHandler{h}}).
		Invoke(ctx, "What is LaunchDarkly?", ldcontext.New("user-123"))
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(res.Response)
}
```

Multiple handlers: the variation's provider picks one.

```go
openai, err := openaimessages.NewHandler(openaimessages.Options{})
if err != nil {
	log.Fatal(err)
}
claude, err := claudemessages.NewHandler(claudemessages.Options{}) // github.com/launchdarkly/go-ai-sdk/claudemessages
if err != nil {
	log.Fatal(err)
}
handlers := []*aiserver.ProviderHandler{openai, claude}
res, err := aiserver.Config(aiserver.ConfigArgs{Key: "my-ai-config", Handlers: handlers}).
	Invoke(ctx, "Hello", ldcontext.New("user-123"))
```

Streaming:

```go
for ev, err := range aiserver.Config(aiserver.ConfigArgs{Key: "my-ai-config", Handlers: handlers}).
	Stream(ctx, "Hello", ldcontext.New("user-123")) {
	if err != nil {
		log.Fatal(err)
	}
	if ev.Type == "chunk" {
		fmt.Print(ev.Text)
	}
}
```

Convenience wrapper (one call, handler built for you):

```go
res, err := openaimessages.OpenAIMessages(ctx, "my-ai-config", "Hello", ldcontext.New("user-123"),
	&openaimessages.InvokeOptions{Variables: map[string]any{"user_input": "Hello"}})
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
const PackageName = "github.com/launchdarkly/go-ai-sdk/openaimessages"
const PackageVersion = "0.1.0"
const SpanName = "invoke_agent"
var ErrMaxSteps error

type Options struct {
	Client         *openai.Client // used for every call when set
	APIKey         string         // used when Client is nil; else OPENAI_API_KEY
	CaptureContent aiserver.ContentCapture // CaptureDefault follows aiserver.Configure, else on
}

func NewHandler(opts Options) (*aiserver.ProviderHandler, error)

type InvokeOptions struct {
	ToolHandlers map[string]aiserver.ToolHandler
	Registry     *aiserver.Registry
	SkipJudges   bool
	Variables    map[string]any
	Handler      Options
}

func OpenAIMessages(ctx context.Context, configKey, userInput string, userCtx ldcontext.Context, opts *InvokeOptions) (aiserver.ProviderResponse, error)
```

`NewHandler` returns a `*aiserver.ProviderHandler` (with both `Invoke` and `Stream` set) and an `error` for an invalid option; the error is nil for now.

Behavior:

- The system prompt is sent as a leading `system` input message. Conversation turns and history
  map to Responses input messages; history images become `input_image` parts with a data URL or
  the image URL.
- Config tools are sent only when `ToolHandlers` holds a `ToolFunc` for them. The tool loop runs at
  most 10 tool turns and then returns `ErrMaxSteps`.
- `outputFormat` is sent as `text.format` (`type: json_schema`, `name: output`, `strict: false`) on
  every blocking call. Follow-up tool turns chain with `previous_response_id` and re-send
  `tools` and `text`, which the Responses API does not carry over. Streaming ignores `outputFormat`.
- These `model.parameters` keys are forwarded: `max_output_tokens`, `max_tool_calls`,
  `parallel_tool_calls`, `prompt_cache_key`, `reasoning`, `service_tier`, `temperature`,
  `tool_choice`, `top_logprobs`, `top_p`. Any other key is dropped. `max_tokens` and
  `max_completion_tokens` are renamed to `max_output_tokens`. `reasoning` keeps only `effort` and
  `summary`, and `tool_choice` must be a string or an object with a string `type`.
- Some keys a config may already set are not forwarded, so OpenAI's defaults apply:
  - `store`: OpenAI stores responses when `store` is omitted, so a config's `store: false` no
    longer stops that.
  - `truncation` and `context_management`: truncation defaults to `disabled`, so a long
    `previous_response_id` chain that relied on `truncation: "auto"` or compaction can fail with a
    context-length error.
  - `include`: `top_logprobs` is forwarded, but without `include: ["message.output_text.logprobs"]`
    no log probabilities come back, so it has no visible effect.

  To send one of these anyway, pass a `Client` built with `option.WithJSONSet`, for example
  `option.WithJSONSet("store", false)`. It is applied after the handler serializes the request, so
  the handler cannot overwrite it. It applies to every request the client sends, including the
  follow-up tool turns that chain with `previous_response_id`, so `store: false` there breaks the
  tool loop.
- Usage is returned as `input_tokens` / `output_tokens` summed over all turns. OpenAI's input count
  already includes cached tokens.
