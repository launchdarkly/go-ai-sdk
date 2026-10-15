# bedrockagents

LaunchDarkly AI SDK **agent** handler for Amazon Bedrock. Go has no Strands Agents SDK, so this package runs
the agent loop itself on the Bedrock Runtime **Converse** API (`Converse` for blocking runs, `ConverseStream`
for streaming), built on `github.com/aws/aws-sdk-go-v2/service/bedrockruntime`. Local tools run between model
turns until the model stops asking for them.

```sh
go get github.com/launchdarkly/go-ai-sdk/bedrockagents
```

## providesFor

`ProvidesFor = {Provider: "Bedrock", Mode: "agent"}`. The handler serves AI
Config variations whose `provider.name` is `Bedrock` and whose `_ldMeta.mode` is `agent`. Variations in
`messages` mode go to `bedrockmessages`.

Variation fields the handler reads:

| Variation field | Use |
| --- | --- |
| `instructions` | System prompt. Takes priority over system-role `messages` (agent mode). |
| `messages` | Conversation turns before history; system entries are used only when `instructions` is empty. |
| `model.name` | Bedrock model or inference-profile id, e.g. `anthropic.claude-sonnet-4-5`. |
| `model.region` | Short inference-profile prefix (`us`, `eu`, `global`), prepended once: `us` + `anthropic.x` → `us.anthropic.x`; `us.anthropic.x` stays as is. Not an AWS region. |
| `model.parameters` | `maxTokens`/`max_tokens`, `temperature`, `topP`/`top_p`, `stopSequences`/`stop_sequences` → `inferenceConfig`. Other keys are ignored. |
| `model.custom` | **Never read.** Return values from `ModelOptions` if you need them. |
| `tools` | Offered to the model only when `ToolHandlers` has an `aiserver.ToolFunc` under the same key. |
| `outputFormat` | Blocking runs send it as Converse native structured output (`outputConfig.textFormat`, `json_schema`). Streaming ignores it. |

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
	"github.com/launchdarkly/go-ai-sdk/bedrockagents"
)

func main() {
	ctx := context.Background()
	defer aiserver.Shutdown(ctx)

	h, err := bedrockagents.NewHandler(bedrockagents.Options{Region: "us-east-1"})
	if err != nil {
		log.Fatal(err)
	}
	resp, err := aiserver.Config(aiserver.ConfigArgs{
		Key:      "travel-agent",
		Handlers: []*aiserver.ProviderHandler{h},
		ToolHandlers: map[string]aiserver.ToolHandler{
			"get_weather": aiserver.ToolFunc(func(ctx context.Context, args map[string]any) (any, error) {
				return "sunny in " + fmt.Sprint(args["city"]), nil
			}),
		},
	}).Invoke(ctx, "Should I pack an umbrella for Paris?", ldcontext.New("user-123"))
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp.Response)
}
```

Multiple handlers (the variation's provider and mode pick one):

```go
bedrock, err := bedrockagents.NewHandler(bedrockagents.Options{})
if err != nil {
	log.Fatal(err)
}
messages, err := bedrockmessages.NewHandler(bedrockmessages.Options{})
if err != nil {
	log.Fatal(err)
}
resp, err := aiserver.Config(aiserver.ConfigArgs{
	Key:      "travel-agent",
	Handlers: []*aiserver.ProviderHandler{bedrock, messages},
}).Invoke(ctx, userInput, ldctx)
```

Convenience wrapper with a guardrail from the model-options callback:

```go
resp, err := bedrockagents.BedrockAgents(ctx, "travel-agent", "Weather in Paris?", ldctx,
	&bedrockagents.InvokeOptions{
		ToolHandlers: tools,
		Handler: bedrockagents.Options{
			ModelOptions: func(cfg aiserver.AiConfig) *bedrockruntime.ConverseInput {
				return &bedrockruntime.ConverseInput{
					GuardrailConfig: &types.GuardrailConfiguration{
						GuardrailIdentifier: aws.String("gr-123"), GuardrailVersion: aws.String("1"),
					},
				}
			},
		},
	})
```

Multi-agent graph with the pre-wired Bedrock agent handler:

```go
g, err := bedrockagents.BedrockGraph("travel-graph",
	aiserver.GraphOptions{ToolHandlers: tools},
	bedrockagents.Options{Region: "us-east-1"})
if err != nil {
	log.Fatal(err)
}
out, err := g.Invoke(ctx, "Plan a weekend in Lisbon", ldctx)
```

Streaming uses the same handler through `aiserver.Config(...).Stream(...)` or `g.Stream(...)`.

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
const PackageName = "github.com/launchdarkly/go-ai-sdk/bedrockagents"
const PackageVersion = "0.1.0"
const SpanName = "invoke_agent"
const GenAISystem = "aws.bedrock"
const DefaultMaxSteps = 10
var ProvidesFor = aiserver.ProvidesFor{Provider: "Bedrock", Mode: aiserver.ModeAgent}
var ErrUnsupportedContent error // same value as bedrockmessages.ErrUnsupportedContent
var ErrToolNotFound error       // same value as bedrockmessages.ErrToolNotFound

type Options struct {
	Client         *bedrockruntime.Client // used as-is; never reconfigured
	APIKey         string                 // bearer token; beats AWS_BEARER_TOKEN_BEDROCK
	Region         string                 // endpoint/SigV4 region for the constructed client
	ModelOptions   func(cfg aiserver.AiConfig) *bedrockruntime.ConverseInput
	CaptureContent bool                   // default false
	MaxSteps       int                    // model turns per call; 0 = DefaultMaxSteps
}

func NewHandler(opts Options) (*aiserver.ProviderHandler, error)

type InvokeOptions struct {
	ToolHandlers map[string]aiserver.ToolHandler
	Registry     *aiserver.Registry
	SkipJudges   bool
	Variables    map[string]any
	Handler      Options
}

func BedrockAgents(ctx context.Context, configKey, userInput string, userCtx ldcontext.Context, opts *InvokeOptions) (aiserver.ProviderResponse, error)
func BedrockGraph(key string, opts aiserver.GraphOptions, handlerOpts Options) (*aiserver.AgentGraph, error)
```

- `NewHandler`, `BedrockAgents` and `BedrockGraph` return an error from client construction only when they must
  build a client and the AWS configuration cannot be loaded. With `Options.Client` that never happens.
- `ModelOptions` runs once per handler call (the analogue of building a Strands `BedrockModel`); its non-zero
  fields override the standard fields on every model turn of that call. `ModelId`, `Messages` and `System`
  cannot be overridden, and when the handler sends tools only the returned `ToolConfig.ToolChoice` applies. An
  `OutputConfig` without `TextFormat` (for example only `Effort`) keeps the structured-output format.
- `BedrockGraph` ignores `opts.Handlers`: the graph always has exactly one handler, built from `handlerOpts`.
  Every other `GraphOptions` field is forwarded.
- The blocking result is the agent's final message. A streaming `done` event carries the text of every turn.
- When the model still asks for tools at `MaxSteps`, the call fails with `ErrMaxSteps`.
- History images must be base64 (`png`, `jpeg`, `gif`, `webp`). URL images are not fetched; they fail with
  `ErrUnsupportedContent`.
