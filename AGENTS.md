# LaunchDarkly AI SDK for Go — agents.md

Architectural contract for contributors and coding agents. Behaviour follows
the LaunchDarkly AI SDK contract shared with the JavaScript and Python SDKs;
where it leaves a public detail open, follow the JavaScript SDK, then Python.
Internals follow Go idiom (see [Language idioms](#language-idioms)).

## Package hierarchy

```
Tier 0  aiserver                 core: no provider SDK imports
        experimental/shadowrequests  experimental: shadow requests
        internal/{bridge,trajectory,judging}   shared internals
Tier 1  claudemessages openaimessages bedrockmessages langchainmessages
        claudeagents   openaiagents   bedrockagents
        internal/{anthropicwire,openaiwire,bedrockwire}  provider wire helpers shared by a pair
        skills         Agent Skills (core; imports aiserver and golang.org/x/sys, no provider SDK)
        evaluations    Evaluations from Code (core; imports aiserver and golang.org/x/text, no provider SDK)
```

Tier 1 packages import `aiserver` and their provider SDK only (`skills` and `evaluations` have none). `aiserver`
never imports a Tier 1 package, and core never names an experimental type:
the two meet only through `internal/bridge`.

## The ProviderHandler contract

```go
type ProviderHandler struct {
	ProvidesFor    ProvidesFor // {Provider, Mode}; Provider "*" = any provider
	CaptureContent ContentCapture // CaptureOn / CaptureOff; CaptureDefault follows Configure, else on
	Invoke         InvokeFunc  // func(ctx, HandlerRequest) (HandlerResult, error)
	Stream         StreamFunc  // optional: func(ctx, HandlerRequest) iter.Seq2[HandlerStreamEvent, error]
}

type HandlerRequest struct {
	Config       AiConfig               // the resolved variation
	UserInput    string                 // "" when absent
	ToolHandlers map[string]ToolHandler // already wrapped for tracking and trajectory
	Variables    map[string]any         // caller variables + ldContext + __ld (TrackData)
	History      []Message              // runtime conversation; may hold image blocks
}
```

`ProvidesFor` is required for routing: the core matches the variation's
`provider.name` and normalised mode (`agent`, else `messages`). Build handlers
with `CreateHandler` (applications). SDK packages call `internal/sdkapi`,
which does not report usage; they report their own helper usage instead.

A handler must: compose its prompt with `aiserver.ComposePrompt`; offer only
config tools that have a `ToolFunc` (`CallableTools`); run the tool loop and
propagate tool errors; open its span tree with `StartHandlerRun`; return
provider usage in a shape `ParseUsage` reads (cache buckets unfolded for
Anthropic/Bedrock, cache-inclusive input for OpenAI/LangChain); and on
streaming, yield `chunk` events then exactly one `done`, stopping cleanly when
the consumer stops.

## NativeTool sentinel

A `*aiserver.NativeTool{ToolName}` value in `ToolHandlers` marks a provider
built-in instead of a local function. After the core wraps it, the handler
detects it with a type switch (`case *aiserver.NativeTool`), wires the
provider's built-in named `ToolName`, and calls `Track()` when the model uses
it so `$ld:ai:tool_call` is sent with the LaunchDarkly tool key. This replaces
the JS `NATIVE_TOOL_KEY` symbol.

Built-ins per provider (only the agent handlers wire `NativeTool`; every
`*messages` handler offers only `ToolFunc` tools via `CallableTools`, so a
`NativeTool` there is not sent to the model):

| Provider | Package | Sentinel → provider built-in | Unsupported |
|---|---|---|---|
| Anthropic | `claudeagents` (`builtins.go`, `handler.go` `serverTool`, `hosttools.go`) | `ClaudeWebSearch` (`WebSearch`) → `web_search_20250305`; `ClaudeWebFetch` (`WebFetch`) → `web_fetch_20250910` server tools. `ClaudeBash`, `ClaudeRead`, `ClaudeEdit`, `ClaudeWrite`, `ClaudeGlob`, `ClaudeGrep`, `ClaudeTodoWrite`, `ClaudeNotebookEdit` → function tools under the Claude Code name and schema, executed on the host; file tools confined to `Options.WorkDir` (`os.Root`), Bash not sandboxed. `Track()` fires per call. | Any other `ToolName`: the config fails with `ErrUnsupportedNativeTool` |
| OpenAI | `openaiagents` (`runner.go` `newAgent`, `openaiwire.HostedTool`) | No exported sentinels: `&aiserver.NativeTool{ToolName: "web_search"}` (or any hosted Responses tool type, e.g. `file_search`, `code_interpreter`) becomes a hosted tool `{"type": ToolName, ...customParameters}`. `Track()` fires on the matching `<type>_call` output item. | — (the type is passed through; the API rejects unknown types) |
| Bedrock | `bedrockagents` | none | `NativeTool` is not wired |
| LangChain | `langchainmessages` | none | `NativeTool` is not wired |

## Lifecycle

`InitClient` (lazy on first `Invoke`) → `Config` / `Graph` → `Shutdown`.
`InitWithClient` installs an existing `aiserver.LDClient` (BYOC) and still sets
up telemetry. A failed init is never cached.

## Agent graphs

A graph flag holds a `GraphTopology` (`root` + `edges`). `ResolveGraph`
evaluates every referenced node config; any disabled node disables the graph.
`GraphDefinition` exposes `Root`, `GetNode`, `GetChildNodes`,
`GetParentNodes`, `TerminalNodes`, `EdgesFrom`, `Traverse` (root first),
`ReverseTraverse` (leaves first, root last), `RunNode` and `Route`. Agent
packages build `XxxGraph` wrappers on `internal/sdkapi.Graph` and native runners on
`ResolveGraph`, emitting graph events with `TrackGraphEvent`.

## OTel setup

`aiserver.InitClient` / `InitWithClient` register the global tracer provider
(OTLP/HTTP + gzip, `launchdarkly.project_id` / `service.name` /
`deployment.environment` resource, W3C propagators,
`ConversationIDSpanProcessor`). Handler packages only start spans through
`StartHandlerRun`; they never configure OTel. The OTel SDK modules are ordinary
`go.mod` dependencies, so there is nothing extra to install. Endpoint and
resource configuration (`OTEL_EXPORTER_OTLP_ENDPOINT`, `LD_SERVICE_NAME`,
`LD_ENVIRONMENT`): see [aiserver/agents.md](aiserver/agents.md#otel-setup) and
[aiserver/README.md](aiserver/README.md#configuration).

## Dependencies

One module (`go.mod`); each package imports only what it needs.

| Module | Used by | Why |
|---|---|---|
| `github.com/launchdarkly/go-server-sdk/v7` | `aiserver` | Default LaunchDarkly client |
| `github.com/launchdarkly/go-sdk-common/v3` | all | `ldcontext`, `ldvalue` |
| `go.opentelemetry.io/otel` (+ `sdk`, `trace`, `otlptracehttp`) | `aiserver`, handlers | Tracer provider, spans, OTLP export |
| `github.com/google/uuid` | `aiserver` | Run ids |
| `github.com/anthropics/anthropic-sdk-go` | `claudemessages`, `claudeagents`, `llms` | Anthropic Messages API |
| `github.com/openai/openai-go/v3` | `openaimessages`, `openaiagents`, `llms` | OpenAI Responses API; Chat Completions in `llms` |
| `github.com/aws/aws-sdk-go-v2` (+ `config`, `credentials`, `service/bedrockruntime`, `eventstream`), `github.com/aws/smithy-go` | `bedrockmessages`, `bedrockagents`, `llms` | Bedrock Converse / ConverseStream |
| `golang.org/x/sys` | `skills` | `*at()` filesystem calls for skill materialization |
| `golang.org/x/text` | `evaluations` | Unicode case folding for the tool key collision check |

## Language idioms

The public contract is shared with the JavaScript and Python SDKs. Everything
else is written the way a Go developer expects: see [Effective Go](https://go.dev/doc/effective_go)
and [Go Code Review Comments](https://go.dev/wiki/CodeReviewComments). Do not port
another SDK's shape into Go. If the contract and a Go idiom conflict, the contract
wins: raise the conflict instead of bending either one.

### What never varies

Public behaviour, defaults, telemetry (span names, attributes, events), wire
payloads, usage IDs and error *conditions* (what fails, and before or after which
I/O). A name is the contract's name in Go casing. A Go-only exported name is fine
when Go needs a named type for a field, parameter or result.

### Errors

- Return `error` last. SDK code does not panic. A panic in caller code
  (handlers, tools, scorers) is recorded and re-raised when it ran on the
  caller's goroutine, and recovered and returned as an error when the SDK ran
  it on its own goroutine.
- Export `ErrXxx` sentinels for conditions callers branch on, and a struct type
  (`evaluations.APIError`) when they need fields. Wrap with
  `fmt.Errorf("…: %w", err)` so `errors.Is` / `errors.As` work. Callers never
  match error strings.
- Where the other SDKs raise one error class, Go exports sentinels instead.
- Return an error where the other SDKs return a safe fallback only when the
  contract says so. Then the result returned with the error is still safe to
  use: `ResolveGraph` returns a non-nil, disabled `*GraphDefinition`.

### Layout

One module. A core feature may be its own package next to `aiserver` (for
example `skills`, `evaluations`) that imports `aiserver`. Experimental names live
only in an `experimental/<feature>` package (today `experimental/shadowrequests`).
Shared code goes in `internal/`. The tier rules are in Package hierarchy above.

### Naming

MixedCaps with Go initialisms (`AIConfig`, `URL`, `ID`); `AiConfig`,
`ParseAiConfig` and `RegisterAiSdkPackage` are existing exceptions, not a
pattern to copy. Handler constructors are `NewHandler(Options)`; an entry point
the contract names keeps that name (`Graph`, `ClaudeGraph`). An exported name may
repeat its package name when that keeps it aligned with the other SDKs. A name
that would collide with a core type gets a suffix (`AIConfigRef`), never a
different meaning.

### Concurrency

A call that makes or waits on a network request takes `ctx context.Context`
first and honours cancellation. The exception is `skills.LaunchDarklySkillStore`,
which streams in the background from `Start` until `Close`; `WaitForSkills`
takes a timeout. Calls block until done; callers add goroutines. A goroutine
that outlives its call has an owner the caller can stop: `Close`
(`skills.SkillWatcher`, `LaunchDarklySkillStore`) or `aiserver.Shutdown` (shadow
requests). Streams are `iter.Seq2[T, error]` and stop cleanly when the consumer
breaks.

### Options

An `XxxOptions` struct, with zero values meaning the default; a `*XxxOptions`
parameter may be nil for all defaults. Per-call extras on `Config` and `Graph`
runs are `...CallOption` (`WithVariables`, `WithHistory`). Durations are
`time.Duration`. Use a pointer field (`*float64`) only when zero is a valid
explicit value. A closed set is a named type with constants (`GenerationMode`).
A one-of-several argument is a sealed interface (`Dataset`), not `any`.

### Testing

Standard `testing`, table tests, hand-written fakes (`fake_test.go`); no network
(`httptest` for HTTP). Core packages pin their exported names in a surface test
(`surface_test.go`; `shadow_requests_surface_test.go` for shadow requests). CI runs
`go test -race -shuffle=on ./...`, so a test of a once-per-process effect (usage
reporting, the global tracer) must not depend on test order: reset the global
first (`aiserver.Shutdown` clears the usage records and the SDK's tracer provider).

### Docs

Every new exported name gets a doc comment that starts with the name. Each
package has a package comment (in `doc.go` or its main file). A feature or
handler package also has a `README.md` and an `agents.md`. Reuse the other SDKs'
wording for the same concept.

### Dependencies

Standard library first, then `golang.org/x`. A new module needs a row in the
Dependencies table above saying why. A provider SDK is imported only by that
provider's Tier 1 packages, their `internal/*wire` helper and `llms`.

### References

`make lint` (gofmt check and `go vet`), `make test` (`go test -race ./...`),
`make build`, `make vulncheck`.

## Common pitfalls

- **A `NativeTool` on a messages handler or Bedrock/LangChain is dropped
  silently.** Only `claudeagents` and `openaiagents` wire built-ins; elsewhere
  the tool is just not offered.
- **Tool keys must match the variation's `tools` map exactly.** A handler
  under a differently-cased key is never offered to the model.
- **Prompt, completion and tool content is on spans by default**. Set the handler's `CaptureContent: aiserver.CaptureOff`, or `aiserver.Configure` before the first call, to keep it off spans. Model, usage and status are written either way.
- **Pass the request `ctx`.** Spans, the conversation id and cancellation follow
  it; `context.Background()` detaches the call from the caller's trace.
- **Breaking out of a `Config(...).Stream` loop abandons the run**: no success or error is
  tracked. Drain to the `done` event if you need the metrics.
- **One disabled node disables the whole graph** (`ErrGraphDisabled`).

## Commit and PR titles

PRs are squash-merged and the title becomes the commit, which release-please turns into the public changelog.

- Use conventional commits: `feat`, `fix`, `perf`, `revert` show in the changelog; `docs`, `chore`, `refactor`, `test`, `build`, `ci`, `style` are hidden. Add `!` for breaking changes.
- The scope names the package or area (`graph`, `lifecycle`, `evaluations`). Never put a ticket key in the PR title, commit scope or PR body. The branch name carries it (`user/TICKET/description`).
- `shadow-requests` routes `feat`/`fix`/`perf` to **Experimental**. It must be exactly that, lowercase, and the only scope.
- Wrong title after merge? Add a `BEGIN_COMMIT_OVERRIDE` / `END_COMMIT_OVERRIDE` block to the PR body before the release PR merges.
