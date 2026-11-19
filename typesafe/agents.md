# typesafe: agent guide

## Role and tier

Tier 1 judge handler package with no provider SDK: an unexported `net/http` client calls the
TypeSafe System One API. `providesFor` is `{Provider: "TypeSafe", Mode: "messages"}`.

## File map

| File | Responsibility |
| --- | --- |
| `handler.go` | Factory, usage report, span tree and content, output and usage mapping. |
| `questions.go` | `ExtractQuestions`, the state builder, and score and reason mapping (port of the JavaScript `questions.ts`). |
| `client.go` | System One request: env vars, Bearer auth, 10 s timeout, retries with `Retry-After`. |
| `fake_test.go` | `httptest` System One server, network guard (`TestMain`), fake LaunchDarkly client. |
| `*_test.go` | Spec tests; no test reaches the network. |

## Public exports

```go
const PackageName, PackageVersion, SpanName
type Options struct{ CaptureContent aiserver.ContentCapture }
func NewHandler(Options) (*aiserver.ProviderHandler, error)
type QuestionType string; const Noul, Choice, Score
type Question struct{ Key, EventKey string; Type QuestionType; Instructions string; Criteria any }
var ErrInvalidClassifiers error
func ExtractQuestions(aiserver.AiConfig) ([]Question, error)
```

`NewHandler` never fails today; the error keeps the shape of the other handlers. It reports
`typesafe.createTypesafeHandler` once per process. `init()` calls
`handlerkit.RegisterAiSdkPackage(PackageName, PackageVersion)`.

## Implementation details

- **Questions** come only from `AiConfig.Classifiers`. Every extraction error wraps
  `ErrInvalidClassifiers` and names `classifiers`; it is returned before any span or request.
- **State** drops missing, null and empty-string values. `message_history` is cut at
  `internal/judging.FormattingInstructions`, the block the core appends.
- **Spans.** `handlerkit.StartHandlerRun` opens the root on a context with capture forced off, so
  the root never carries content. The chat span gets the state (`telemetry.SetChatContent`) before
  the call and the completion after it, both gated by the handler's own context. The span and
  `gen_ai.request.model` use `model.name`, or `jev` when it is empty; the wire default
  (`jev-latest`) never appears in a span name.
- **Usage.** No `usage` object in the response marks the turn unreported: the chat span writes
  zeros and the root writes none.
- **Errors.** A non-2xx response returns an error with the status, the `x-typesafe-request-id`
  header and the body, never the key. A missing answer value returns an error after the call.
