# litellmmessages

LaunchDarkly AI SDK handler for a [LiteLLM](https://docs.litellm.ai/) proxy. It talks to the
proxy's OpenAI-compatible Chat Completions API with
[`openai-go`](https://github.com/openai/openai-go), so one handler serves every model the proxy
routes: provider-qualified names such as `anthropic/claude-sonnet-4`, `bedrock/...`,
`gemini/...`, and proxy aliases.

```sh
go get github.com/launchdarkly/go-ai-sdk/litellmmessages
```

## providesFor

`{Provider: "*", Mode: "messages"}`. The handler runs for messages-mode AI Config variations
(mode `completion`, normalized to `messages`) of any provider that no provider-specific handler
claims. To serve only some providers, scope it with `h.WithProviders("Anthropic", "Gemini")`.

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
	"github.com/launchdarkly/go-ai-sdk/litellmmessages"
)

func main() {
	ctx := context.Background()
	if _, err := aiserver.InitClient(ctx, aiserver.InitOptions{}); err != nil { // reads LD_SDK_KEY
		log.Fatal(err)
	}
	defer aiserver.Shutdown(ctx)

	h, err := litellmmessages.NewHandler(litellmmessages.Options{}) // reads LITELLM_BASE_URL, LITELLM_API_KEY
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

Next to provider handlers: an exact provider match wins, so LiteLLM serves only the rest.

```go
proxy, err := litellmmessages.NewHandler(litellmmessages.Options{BaseURL: "http://localhost:4000"})
if err != nil {
	log.Fatal(err)
}
openai, err := openaimessages.NewHandler(openaimessages.Options{}) // github.com/launchdarkly/go-ai-sdk/openaimessages
if err != nil {
	log.Fatal(err)
}
handlers := []*aiserver.ProviderHandler{openai, proxy}
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

A client per call, chosen from the evaluated config:

```go
h, err := litellmmessages.NewHandler(litellmmessages.Options{
	ClientFactory: func(cfg aiserver.AiConfig) (*openai.Client, error) {
		c := openai.NewClient(option.WithBaseURL(proxyFor(cfg.Model.Name)), option.WithAPIKey(key))
		return &c, nil
	},
})
```

Convenience wrapper (one call, handler built for you):

```go
res, err := litellmmessages.LiteLLMMessages(ctx, "my-ai-config", "Hello", ldcontext.New("user-123"),
	&litellmmessages.InvokeOptions{
		Handler: litellmmessages.Options{BaseURL: "http://localhost:4000"},
		ToolHandlers: map[string]aiserver.ToolHandler{
			"search-ld-documentation": aiserver.ToolFunc(searchDocs),
		},
		Variables: map[string]any{"user_input": "Hello"},
	})
```

## Environment variables

| Variable | Required | Description |
| --- | --- | --- |
| `LITELLM_BASE_URL` | Yes, unless `Options.BaseURL`, `Options.Client` or `Options.ClientFactory` is set | LiteLLM proxy base URL. |
| `LITELLM_API_KEY` | Optional | Proxy key, unless `Options.APIKey` is set. Without one, `Bearer not-needed` is sent, which an unauthenticated proxy accepts. |
| `LD_SDK_KEY` | Yes, unless `InitOptions.SDKKey` or `aiserver.InitWithClient` is used | LaunchDarkly server SDK key, read by `aiserver`. |

The `OPENAI_*` variables `openai-go` reads by default (`OPENAI_BASE_URL`, `OPENAI_API_KEY`,
`OPENAI_ORG_ID`, `OPENAI_PROJECT_ID`, `OPENAI_CUSTOM_HEADERS`) never reach a client this handler
builds. A client you inject is used as you built it.

## API

```go
const PackageName = "github.com/launchdarkly/go-ai-sdk/litellmmessages"
const PackageVersion // this package's release version
const SpanName = "invoke_agent"
var ErrNoBaseURL, ErrMaxSteps error

type ClientFactory func(cfg aiserver.AiConfig) (*openai.Client, error)

type Options struct {
	BaseURL        string         // proxy URL; else LITELLM_BASE_URL
	APIKey         string         // proxy key; else LITELLM_API_KEY; else "not-needed"
	Client         *openai.Client // used for every call when set; must point at a LiteLLM proxy
	ClientFactory  ClientFactory  // called per call with the evaluated config; wins over Client
	MaxRetries     *int           // retries on the client NewHandler builds; nil = SDK default; refused with Client, ClientFactory or < 0
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

func LiteLLMMessages(ctx context.Context, configKey, userInput string, userCtx ldcontext.Context, opts *InvokeOptions) (aiserver.ProviderResponse, error)
```

`NewHandler` returns a `*aiserver.ProviderHandler` (with both `Invoke` and `Stream` set), or an
error: `ErrNoBaseURL` with no proxy URL and no injected client, or a refusal naming `MaxRetries`
for a negative count or a count given with `Client` / `ClientFactory`. `LiteLLMMessages` returns
the same error as its call's error.

Behavior:

- Every request goes to `<base URL>/chat/completions` on the proxy. There is no fallback to a
  provider endpoint: a failing proxy fails the call.
- `model.name` is sent unchanged, with one exception for Bedrock. When `model.region` holds a
  cross-region inference-profile prefix (`us`, `eu`, `global`, ...), it is prepended once to a
  `bedrock/...` name's last segment (`bedrock/converse/us.anthropic.claude-...`), and to a bare
  name under the `Bedrock` provider. An id that already has a known prefix, or holds an ARN, is
  sent as configured. Telemetry reports the id sent.
- The system prompt is a leading `system` message. Conversation turns and history keep their
  roles; history images become `image_url` parts (the image URL, or a `data:` URI).
- Config tools are sent as function tools only when `ToolHandlers` holds a `ToolFunc` for them.
  Each tool turn replays the assistant `tool_calls` message and one `role: tool` result per call.
  The loop runs at most 10 tool turns and then returns `ErrMaxSteps`.
- `outputFormat` is sent as `response_format` (`type: json_schema`, `name: output`,
  `strict: false`) on every blocking call; a `json_schema`, `json` or missing schema `type`
  becomes `object`. The answer is returned as the raw JSON string. Streaming ignores
  `outputFormat`.
- These `model.parameters` keys are forwarded unchanged: `frequency_penalty`,
  `max_completion_tokens`, `max_tokens`, `parallel_tool_calls`, `presence_penalty`,
  `reasoning_effort`, `seed`, `stop`, `temperature`, `tool_choice`, `top_p`. Every other key is
  dropped, including LiteLLM routing, credential and retry keys (`api_base`, `api_key`,
  `custom_llm_provider`, `num_retries`, ...) and keys the handler owns (`model`, `messages`,
  `tools`, `stream`, `response_format`).
- Streaming sends `stream: true` with `stream_options.include_usage`, joins tool-call fragments
  by index, and closes the response body when the consumer stops.
- Usage is returned as `input_tokens` / `output_tokens` summed over all turns. `prompt_tokens`
  already includes cached tokens.
- Spans: `gen_ai.system` is `litellm`; `gen_ai.provider.name` is the config's provider in lower
  case (`aws.bedrock` for Bedrock), else `litellm`.
