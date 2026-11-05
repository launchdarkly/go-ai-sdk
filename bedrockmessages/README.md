# bedrockmessages

LaunchDarkly AI SDK handler for the Amazon Bedrock Runtime **Converse** API
(`Converse` for blocking calls, `ConverseStream` for streaming), built on
`github.com/aws/aws-sdk-go-v2/service/bedrockruntime`.

```sh
go get github.com/launchdarkly/go-ai-sdk/bedrockmessages
```

## providesFor

`ProvidesFor = {Provider: "Bedrock", Mode: "messages"}`.
The handler serves AI Config variations whose `provider.name` is `Bedrock` and whose
`_ldMeta.mode` is `messages` (or absent, which normalises to `messages`).

Model fields the handler reads:

| Variation field | Use |
| --- | --- |
| `model.name` | Bedrock model or inference-profile id, e.g. `anthropic.claude-sonnet-4-5` |
| `model.region` | Short inference-profile prefix (`us`, `eu`, `global`), prepended once: `us` + `anthropic.x` → `us.anthropic.x`; `us` + `us.anthropic.x` stays `us.anthropic.x`. Not an AWS region. |
| `model.parameters` | `maxTokens`/`max_tokens`, `temperature`, `topP`/`top_p`, `stopSequences`/`stop_sequences` → `inferenceConfig`. Other keys are ignored. |
| `model.custom` | **Never read.** Forward values explicitly with `ConverseOptions` if you need them. |

## Examples

Single handler:

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/bedrockmessages"
)

func main() {
	ctx := context.Background()
	defer aiserver.Shutdown(ctx)

	h, err := bedrockmessages.NewHandler(bedrockmessages.Options{Region: "us-east-1"})
	if err != nil {
		log.Fatal(err)
	}
	resp, err := aiserver.Config(aiserver.ConfigArgs{
		Key:      "support-assistant",
		Handlers: []*aiserver.ProviderHandler{h},
	}).Invoke(ctx, "How do I reset my password?", ldcontext.New("user-123"))
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp.Response)
}
```

Multiple handlers (the variation's provider picks one):

```go
bedrock, err := bedrockmessages.NewHandler(bedrockmessages.Options{})
if err != nil {
	log.Fatal(err)
}
resp, err := aiserver.Config(aiserver.ConfigArgs{
	Key:      "support-assistant",
	Handlers: []*aiserver.ProviderHandler{bedrock, claudemessages.NewHandler(claudemessages.Options{})},
}).Invoke(ctx, userInput, ldctx)
```

Convenience wrapper, tools, and a per-turn Converse escape hatch:

```go
resp, err := bedrockmessages.BedrockMessages(ctx, "support-assistant", "Weather in Paris?", ldctx,
	&bedrockmessages.InvokeOptions{
		ToolHandlers: map[string]aiserver.ToolHandler{
			"get_weather": aiserver.ToolFunc(func(ctx context.Context, args map[string]any) (any, error) {
				return "sunny", nil
			}),
		},
		Handler: bedrockmessages.Options{
			ConverseOptions: func(cfg aiserver.AiConfig) *bedrockruntime.ConverseInput {
				return &bedrockruntime.ConverseInput{
					GuardrailConfig: &types.GuardrailConfiguration{
						GuardrailIdentifier: aws.String("gr-123"), GuardrailVersion: aws.String("1"),
					},
				}
			},
		},
	})
```

Streaming uses the same handler through `aiserver.Config(...).Stream(...)`.

## Environment variables

All are read by the AWS SDK when the handler builds its own client; none is read when you pass `Options.Client`.

| Variable | Required | Description |
| --- | --- | --- |
| `AWS_REGION` / `AWS_DEFAULT_REGION` | Yes, unless `Options.Region` or shared config sets a region | Bedrock Runtime endpoint and SigV4 region. Never part of `modelId`. |
| `AWS_BEARER_TOKEN_BEDROCK` | Optional | Bedrock API key sent as a bearer token. `Options.APIKey` overrides it. |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, `AWS_PROFILE`, … | Optional | Default AWS credential chain, used when no bearer token is set. |
| `AWS_ENDPOINT_URL_BEDROCK_RUNTIME` / `AWS_ENDPOINT_URL` | Optional | Endpoint override (VPC endpoints, testing). |
| `LD_SDK_KEY` and OTel variables | See the core `aiserver` package | Read by the core client, not this package. |

## API reference

```go
const PackageName = "github.com/launchdarkly/go-ai-sdk/bedrockmessages"
const PackageVersion = "0.1.0"
const SpanName = "invoke_agent"
const GenAISystem = "aws.bedrock"
var ProvidesFor = aiserver.ProvidesFor{Provider: "Bedrock", Mode: aiserver.ModeMessages}
var ErrUnsupportedContent error
var ErrToolNotFound error // the model requested a tool that was not offered
var ErrMaxSteps error // the model still asked for tools after 10 turns

type Options struct {
	Client          *bedrockruntime.Client // used as-is; never reconfigured
	APIKey          string                 // bearer token; beats AWS_BEARER_TOKEN_BEDROCK
	Region          string                 // endpoint/SigV4 region for the constructed client
	ConverseOptions func(cfg aiserver.AiConfig) *bedrockruntime.ConverseInput
	CaptureContent  aiserver.ContentCapture // CaptureDefault follows aiserver.Configure, else on
}

func NewHandler(opts Options) (*aiserver.ProviderHandler, error)

type InvokeOptions struct {
	ToolHandlers map[string]aiserver.ToolHandler
	Registry     *aiserver.Registry
	SkipJudges   bool
	Variables    map[string]any
	Handler      Options
}

func BedrockMessages(ctx context.Context, configKey, userInput string, userCtx ldcontext.Context, opts *InvokeOptions) (aiserver.ProviderResponse, error)
```

- `NewHandler` returns an error only when it must build a client and the AWS configuration cannot be loaded.
  With `Options.Client` it never fails.
- `ConverseOptions` runs once per provider turn (tool follow-ups and streaming turns included). Non-zero fields
  of the returned input override the handler's standard fields; `ModelId`, `Messages` and `System` cannot be
  overridden, and when the handler sends tools only the returned `ToolConfig.ToolChoice` applies.
- History images must be base64 (`png`, `jpeg`, `gif`, `webp`). URL images are not fetched; they fail with
  `ErrUnsupportedContent`.
- `outputFormat` is best-effort: the JSON-schema instruction is appended to the system prompt on blocking calls.
