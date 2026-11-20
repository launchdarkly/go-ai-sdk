# opencodeagents

LaunchDarkly AI SDK agent handler for [OpenCode](https://opencode.ai). Each call starts its own
`opencode serve` child on loopback, runs one OpenCode session over its HTTP API, and stops the
child when the call ends. OpenCode runs the model and its own built-in tools; the handler decides
which tools the session may use, serves your `ToolFunc` tools to it, and reports the run like
every other handler (returns, tracking, spans).

Verified against OpenCode **1.19.5** (a live bridged-tool turn runs in CI against that release).

No OpenCode client library is used: the HTTP, event-stream and tool-bridge code is plain
`net/http`, so the package adds no dependency.

```sh
go get github.com/launchdarkly/go-ai-sdk/opencodeagents
npm install -g opencode-ai   # the opencode CLI must be on PATH (or set Options.Command)
```

> **Security: OpenCode's built-ins act on your host.** `OpenCodeBash`, `OpenCodeEdit` and the
> other sentinels let the *model* run commands and change files in `WorkDir`, as the child's
> user. A session may use only the tools this call offers (every other tool is denied), but an
> offered tool is not sandboxed. Offer them only for configs and inputs you trust, and run in a
> container or VM with a dedicated working directory.

## providesFor

`{Provider: "*", Mode: "agent"}`: any provider's agent-mode variation. A judge routed here runs
as its own OpenCode call, with every tool denied.

`langchainagents` also serves `{*, agent}`. Pass one of the two in a handler list, not both.

## Examples

```go
h, err := opencodeagents.NewHandler(opencodeagents.Options{
	WorkDir:    "/srv/agent-work",
	ConfigPath: "/srv/agent-work/opencode.json",
	Env:        []string{"PATH=" + os.Getenv("PATH"), "HOME=/srv/agent-home", "OPENAI_API_KEY=" + key},
})
if err != nil {
	return err
}
resp, err := aiserver.Config(aiserver.ConfigArgs{
	Key: "my-agent", Handlers: []*aiserver.ProviderHandler{h},
	ToolHandlers: map[string]aiserver.ToolHandler{
		"lookup-order": aiserver.ToolFunc(lookupOrder), // served to OpenCode over a per-call MCP bridge
		"shell":        opencodeagents.OpenCodeBash,     // OpenCode's own bash tool
	},
}).Invoke(ctx, "Where is order 42?", userCtx)
```

Convenience, one call:

```go
resp, err := opencodeagents.OpenCodeAgents(ctx, "my-agent", "Where is order 42?", userCtx,
	&opencodeagents.InvokeOptions{Handler: opts, ToolHandlers: tools})
```

`Config(...).Stream` yields the assistant's text as it arrives, then one `done`.

## The child process

- **Environment.** `Options.Env` is the child's *whole* environment. It is required: a nil `Env`
  is refused, because Go's `exec.Cmd` would otherwise pass the parent's environment (and its
  secrets, such as `LD_SDK_KEY`) to the model's tools. Pass only what OpenCode needs: `PATH`,
  `HOME`, and the provider credentials. An empty, non-nil slice is an empty environment. The
  handler sets `OPENCODE_CONFIG`, `OPENCODE_SERVER_USERNAME` and `OPENCODE_SERVER_PASSWORD` (a
  new random password per call) itself.
- **Paths.** `WorkDir` and `ConfigPath` must be non-empty when the handler is built, and must
  exist when a call runs; a missing one fails that call before a child starts.
- **Lifecycle.** One child per call, on a free `127.0.0.1` port, ready once `GET /global/health`
  reports healthy (45 s budget). A child that exits during startup fails the call with the last
  2 KiB of its stderr. The call stops the child when it returns, including when the context is
  cancelled or the stream consumer stops early: SIGTERM to its process group, SIGKILL after 5 s.
  **On Windows there is no process group**: only the `opencode` process itself is stopped.
- **Session.** The handler opens `GET /event`, creates a session whose `permission` ruleset
  denies every tool and then allows the offered ones, and posts the prompt. The turn ends on the
  session's `session.idle`, or 5 s after the message request returned with no further event. A
  call that ends any other way aborts the session.

## Tools

The offered set is the intersection of the config's tools and the tool handlers. A config tool
with no handler, and a handler the config does not list, are not offered. With nothing offered,
every OpenCode tool is denied.

| Sentinel | OpenCode tool | | Sentinel | OpenCode tool |
| --- | --- | --- | --- | --- |
| `OpenCodeBash` | `bash` | | `OpenCodeTask` | `task` |
| `OpenCodeRead` | `read` | | `OpenCodeSkill` | `skill` |
| `OpenCodeEdit` | `edit` | | `OpenCodeTodoWrite` | `todowrite` |
| `OpenCodeWrite` | `write` | | `OpenCodeWebFetch` | `webfetch` |
| `OpenCodeGlob` | `glob` | | `OpenCodeWebSearch` | `websearch` |
| `OpenCodeGrep` | `grep` | | `OpenCodeApplyPatch` | `apply_patch` |

- **`edit`, `write` and `apply_patch` are one grant.** OpenCode checks all three against the
  `edit` permission, so offering any one of them allows all three.
- **`skill` is denied** unless the config lists it and you pass `OpenCodeSkill`.
- **A `ToolFunc`** is served from a loopback MCP server the call starts (its own port and bearer
  token), attached to the child with `POST /mcp`, and allowed as `ld_<run>_<key>`. A tool error
  goes back to the model as a tool error result.
- **Tracking.** A sentinel sends `$ld:ai:tool_call` once per tool call, under its config key; a
  `ToolFunc` is tracked by the core wrapper when it runs.

## Model

`Options.Model` maps a config to OpenCode's `providerID` and `modelID`. The default splits a
`provider/model` name (`bedrock/` becomes `amazon-bedrock`); otherwise `Bedrock` (any case)
maps to `amazon-bedrock`, `Gemini` to `google`, and any other provider name is lowercased. An
`amazon-bedrock` model id gets `model.region` as its inference-profile prefix (`us.<id>`), once:
the region is trimmed (`" us "` and `"us."` mean `us`), and an id that already starts with a
cross-region prefix (`global.`, `us.`, `eu.`, ...) or is an ARN is sent unchanged. Telemetry
reports the id that was sent.

No `model.parameters` key is forwarded: OpenCode's message request has no field for sampling,
length or tool choice, and takes them from its own configuration.

## Telemetry

One `invoke_agent` root span per call, one `chat <model>` span per assistant message (with that
message's tokens), and one `execute_tool <key>` span per tool call, all children of the root.
A failed tool call fails its span without recording OpenCode's error text, and a session error
(`*SessionError`) carries the error's name, never the provider's error body. Content capture
follows `Options.CaptureContent`.

## Differences from the Python package

- No shared process: Python's `async with handler` lets concurrent calls share one child. Go
  always starts one per call (Python's default).
- `Options.Env` is a `[]string` and nil is refused at `NewHandler`; Python raises when `env` is
  missing.
- The child runs in its own process group and is stopped as a group; Python stops only the
  `opencode` process.

## API

```go
const PackageName = "github.com/launchdarkly/go-ai-sdk/opencodeagents"
const SpanName = "invoke_agent"

type Options struct {
	WorkDir, ConfigPath string   // required
	Env                 []string // required, the child's whole environment
	Command             []string // default {"opencode"}
	Model               func(cfg *aiserver.AiConfig) (providerID, modelID string, err error)
	Agent               string   // OpenCode agent name, sent with the message when set
	CaptureContent      aiserver.ContentCapture
}
func NewHandler(Options) (*aiserver.ProviderHandler, error)

type InvokeOptions struct {
	ToolHandlers map[string]aiserver.ToolHandler
	Registry     *aiserver.Registry
	SkipJudges   bool
	Variables    map[string]any
	Handler      Options
}
func OpenCodeAgents(ctx context.Context, configKey, userInput string, userCtx ldcontext.Context, opts *InvokeOptions) (aiserver.ProviderResponse, error)

type SessionError struct{ Name string }

var OpenCodeBash, OpenCodeRead, OpenCodeEdit, OpenCodeWrite, OpenCodeGlob, OpenCodeGrep,
	OpenCodeTask, OpenCodeSkill, OpenCodeTodoWrite, OpenCodeWebFetch, OpenCodeWebSearch,
	OpenCodeApplyPatch *aiserver.NativeTool
```

`PackageVersion` is the module's release version.
