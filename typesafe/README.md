# typesafe

LaunchDarkly AI SDK judge handler for TypeSafe Jev. It calls the TypeSafe System One REST API
(`POST /v1/systemone`) directly; TypeSafe publishes no Go SDK.

```sh
go get github.com/launchdarkly/go-ai-sdk/typesafe
```

## providesFor

`{Provider: "TypeSafe", Mode: "messages"}`. Judge mode normalizes to `messages`, so a judge
variation whose `provider.name` is `TypeSafe` selects this handler. It is judge-only: no
streaming, no tools, no `outputFormat`.

## Example

Register the handler next to the handler that runs your AI Config. Judges attached to the config
run through it automatically.

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/openaimessages"
	"github.com/launchdarkly/go-ai-sdk/typesafe"
)

func main() {
	ctx := context.Background()
	if _, err := aiserver.InitClient(ctx, aiserver.InitOptions{}); err != nil { // reads LD_SDK_KEY
		log.Fatal(err)
	}
	defer aiserver.Shutdown(ctx)

	openai, err := openaimessages.NewHandler(openaimessages.Options{})
	if err != nil {
		log.Fatal(err)
	}
	jev, err := typesafe.NewHandler(typesafe.Options{}) // reads TYPESAFE_API_KEY when a judge runs
	if err != nil {
		log.Fatal(err)
	}
	res, err := aiserver.Config(aiserver.ConfigArgs{Key: "my-ai-config", Handlers: []*aiserver.ProviderHandler{openai, jev}}).
		Invoke(ctx, "What is LaunchDarkly?", ldcontext.New("user-123"))
	if err != nil {
		log.Fatal(err)
	}
	for key, r := range res.JudgeResults { // one entry per label: "<judge key>.<label key>"
		fmt.Println(key, r.Score, r.Response)
	}
}
```

## Environment variables

| Variable | Required | Description |
| --- | --- | --- |
| `TYPESAFE_API_KEY` | Yes | TypeSafe API key, sent as a Bearer token. Without it the call fails before any request. |
| `TYPESAFE_BASE_URL` | Optional | API base URL; default `https://api.typesafe.ai`. |
| `TYPESAFE_DEFAULT_MODEL` | Optional | Model sent when the config has no `model.name`; default `jev-latest`. |
| `LD_SDK_KEY` | Yes, unless `InitOptions.SDKKey` or `aiserver.InitWithClient` is used | LaunchDarkly server SDK key, read by `aiserver`. |

## API

```go
const PackageName = "github.com/launchdarkly/go-ai-sdk/typesafe"
const PackageVersion = "1.0.0"
const SpanName = "invoke_agent"

type Options struct {
	CaptureContent aiserver.ContentCapture // CaptureDefault follows aiserver.Configure, else on
}

func NewHandler(opts Options) (*aiserver.ProviderHandler, error)

type QuestionType string // Noul, Choice, Score
type Question struct {
	Key, EventKey, Instructions string
	Type                        QuestionType
	Criteria                    any // decoded JSON: a map (noul, choice), a list (score), or nil
}
var ErrInvalidClassifiers error

func ExtractQuestions(cfg aiserver.AiConfig) ([]Question, error)
```

Behavior:

- **Questions.** `ExtractQuestions` reads the variation's top-level `classifiers` list, one
  question per entry, in order. Each entry needs a non-empty `key`, `eventKey` and `instructions`,
  and a `type` of `noul`, `choice` or `score`. `choice` needs a non-empty criteria object and
  `score` a list of at least two levels. Any problem returns an error wrapping
  `ErrInvalidClassifiers`, before any request.
- **State.** Jev receives the judge variables that are present and non-empty: `input`,
  `response_to_evaluate` (as `output`), `message_history` without the SDK's `{score, reasoning}`
  formatting block, `trajectory`, `expected_output`, `ground_truth_context`, plus `history` and a
  `user_input` that is not already the output.
- **One call.** Every question goes in one request with `config.model.name` (else
  `TYPESAFE_DEFAULT_MODEL`, else `jev-latest`). The client times out after 10 s, retries 408, 429
  and 5xx at most twice (honouring `Retry-After`), and stops when `ctx` is cancelled. Model
  parameters and tools are never sent.
- **Output.** `{"kind":"typesafe","results":[{"key","eventKey","score","reason"}, ...]}` in label
  order. A noul scores its yes-probability; a choice the selected choice's probability, then
  `confidence`, then 1; a score the rubric index divided by `levels - 1`, clamped to 0-1. The
  reason is the selected label's criteria text. A missing answer fails the call. The core expands
  the output into one judge result per label, each tracked on its own `eventKey`.
- **Usage.** Returned as `input_tokens` / `output_tokens` from the response (`inputTokens` /
  `outputTokens` are read too).
- **Telemetry.** `invoke_agent` → one `chat <model.name>` span (`chat jev` when the name is empty)
  with `gen_ai.system` = `typesafe`, the response's model as `gen_ai.response.model`, and token
  usage when the response reports it. With content capture on, the chat span carries the state as
  a single user prompt and the output as the completion. The root carries the LaunchDarkly
  attributes, context identity and the `feature_flag` event.
