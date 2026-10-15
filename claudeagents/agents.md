# claudeagents: agent guide

## Role and tier

Tier 1 handler package. `providesFor` is `{Provider: "Anthropic", Mode: "agent"}`. It runs an
agent loop on the Anthropic Messages API (`client.Messages.New` / `NewStreaming`) from
`github.com/anthropics/anthropic-sdk-go`, because Go has no Claude Agent SDK. It also exports the
`claudeGraph` wrapper, the `toClaudeAgents` native adapter and the Claude Code
built-in `NativeTool` sentinels.

## File map

| File | Responsibility |
| --- | --- |
| `handler.go` | Factory, convenience wrapper, tool partitioning, the agent loop, streaming, span calls. |
| `graph.go` | `ClaudeGraph`: `aiserver.GraphInternal` with exactly this package's handler. |
| `native.go` | `ToClaudeAgents`: sub-agent tools built in `ReverseTraverse`, graph span and graph events. |
| `params.go` | `model.parameters` → Messages API (`thinking`, `output_config.effort` / `.format`, `anthropic-beta`, `fallbackModel`); the JS `FORWARDED_QUERY_OPTION_KEYS` allowlist. |
| `builtins.go` | The ten Claude Code built-in `NativeTool` sentinels. |
| `hosttools.go` | The eight host-executed built-ins: schemas, `workspace` (WorkDir confinement via `os.Root`, todo list), executors. |
| `proc_unix.go`, `proc_other.go` | Bash process group (killed as a group on timeout) on Unix; no-op elsewhere. |
| `hosttools_test.go` | Per-tool tests, confinement abuse matrix, Bash timeout/cap, handler and `ToClaudeAgents` wiring. |
| `fake_test.go` | `httptest` Anthropic server, fake LaunchDarkly client, span exporter setup. |
| `handler_test.go` | Handler tests (prompt, tools, loop, telemetry, streaming, history). |
| `native_test.go` | `ClaudeGraph`, `ToClaudeAgents` and conversation-id graph tests. |
| `misc_test.go` | Built-in sentinels, convenience wrapper, usage events. |
| `README.md` | User-facing install, examples, environment, and API reference. |
| `agents.md` | This file. |

Shared Anthropic wire mapping (turns, tools, max_tokens, finish reasons, usage) lives in
`internal/anthropicwire`, used by this package and `claudemessages`.

## Public exports

```go
const PackageName, PackageVersion, SpanName, DefaultMaxSteps, SubAgentToolPrefix
var ProvidesFor aiserver.ProvidesFor
var ErrToolNotFound, ErrUnsupportedNativeTool, ErrMaxSteps error
type Options struct{ Client *anthropic.Client; APIKey string; CaptureContent bool; MaxSteps int; WorkDir string }
func NewHandler(Options) *aiserver.ProviderHandler
type InvokeOptions struct{ ToolHandlers; Registry; SkipJudges; Variables; Handler Options }
func ClaudeAgents(ctx, configKey, userInput string, userCtx ldcontext.Context, *InvokeOptions) (aiserver.ProviderResponse, error)
type GraphOptions struct{ ToolHandlers; GraphJudge; Registry }
func ClaudeGraph(key string, GraphOptions, Options) *aiserver.AgentGraph
type AdapterOptions struct{ ToolHandlers; Handler Options }
type GraphRunner struct{ /* unexported */ }
func ToClaudeAgents(*aiserver.GraphDefinition, *AdapterOptions) *GraphRunner
func (*GraphRunner) Invoke(ctx, input string, opts ...aiserver.CallOption) (aiserver.GraphResponse, error)
var ClaudeBash, ClaudeRead, ClaudeEdit, ClaudeWrite, ClaudeGlob, ClaudeGrep, ClaudeWebFetch, ClaudeWebSearch, ClaudeTodoWrite, ClaudeNotebookEdit *aiserver.NativeTool
```

`init()` calls `aiserver.RegisterAiSdkPackage(PackageName, PackageVersion)`.

## Implementation details

The handler receives `aiserver.HandlerRequest{Config, UserInput, ToolHandlers, Variables, History}`.

- **`Config`**: `aiserver.ComposePrompt(req, ModeAgent)` builds the system prompt and turns
  (instructions win over system messages). On the blocking path `OutputFormat` is appended with
  `aiserver.WithOutputFormat`. `Model.Name` is the request model; `max_tokens` and `max_turns`
  are read from `Model.Parameters`.
- **`UserInput`**: appended by `ComposePrompt`; never absent, `""` when empty.
- **`ToolHandlers`**: for each `Config.Tools` key, a `ToolFunc` becomes a custom tool and a
  `*aiserver.NativeTool` becomes either its Anthropic server tool (`WebSearch` → `web_search`,
  `WebFetch` → `web_fetch`) or, for the eight names in `hostTools`, a plain function tool named
  `ToolName` with Claude Code's input schema, executed by `toolset.runHost`. Any other
  `ToolName` fails the run with `ErrUnsupportedNativeTool` before the first call. A
  `server_tool_use` block, or a `tool_use` for a host built-in, calls the wrapped NativeTool's
  `Track()`, so `$ld:ai:tool_call` carries the LaunchDarkly map key. A `tool_use` for a name
  with no executor fails with `ErrToolNotFound`, naming the tool.

Host built-ins: `buildTools` opens a `workspace` per run (`os.OpenRoot(WorkDir)`, closed when the
run ends) only when one is enabled. `workspace.rel` rejects `..` components and absolute paths
outside `WorkDir` (or its symlink-resolved form); all file I/O then goes through the `os.Root`,
which refuses symlinks leaving it. Directory walks use `root.FS()` with `fs.WalkDir` (no
symlinked-directory traversal). Executor errors become `is_error` tool results and mark the
`execute_tool` span ERROR; they never fail the run (unlike `ToolFunc` errors), so the
model can recover as it does in Claude Code. The tool span's message is a short status (e.g.
`bash: exit code 3`), not the command output.

Why plain function tools and not Anthropic's client tool types: `anthropic-sdk-go` exposes
`bash_20250124` and `text_editor_20250728`, but their names are fixed (`bash`,
`str_replace_based_edit_tool`) and their inputs differ (`{command, restart}`; one tool with
`view`/`create`/`str_replace`/`insert` commands). Using them would change the tool names the
model sees and the trajectory records, and leave Read/Write/Glob/Grep/TodoWrite/NotebookEdit
without a type anyway. Claude Code itself sends its built-ins as named custom tools, so this
package does the same.
- **`Variables`**: template values for `ParseTemplate`, plus the `__ld` / `ldContext` entries that
  `aiserver.StartHandlerRun` writes onto the root span.
- **`History`**: spliced by `ComposePrompt` as structured turns; images map to native Anthropic
  image blocks.

Loop: one `chat <model>` span per provider turn and one `execute_tool <name>` span per tool
call, siblings under the `claude.query` root (`gen_ai.operation.name = invoke_agent`). A tool
executes with its `execute_tool` span as the active span. `pause_turn` resends the assistant
turn. Running out of steps while the model still wants tools returns `ErrMaxSteps`.

`ToClaudeAgents`: `ReverseTraverse` builds, leaf-first, one `mcp__subagents__<key>` tool per
edge into a non-root node (unsafe characters become `_`; a name over 64 characters, or one a
sibling already uses, is cut and suffixed with a hash of the key) (description: `edge.handoff.description`, else the first 120 runes of
the child's instructions, else its key). Running the root starts the run; calling a sub-agent
tool tracks `$ld:ai:graph:node` once per node with a 0-based `index`, runs the child with its
own sub-agent tools, and then tracks `$ld:ai:graph:handoff_success` (`sourceKey`/`targetKey`), or
`:handoff_failure` when the child failed. `$ld:ai:graph:*` events carry the graph's `TrackData`
(as `aiserver.Graph`); a definition resolved without a context tracks nothing. Each Anthropic
node tracks its own generation, duration and token events under a node `TrackData` that shares
the graph's run id. Each node's `claude.query` root span gets the LaunchDarkly association
(including `launchdarkly.graph.key`) and the user's context identity through `spanIdentity`,
because the core only reads the typed context from variables it builds itself. A non-Anthropic node runs through `GraphDefinition.RunNode` and gets no
sub-agent tools. The `ld.ai.graph` span carries `ld.ai.graph.key`, `launchdarkly.graph.path` and
the summed usage and is ended on every path; the result is the root's final text.

## OTel setup

Handler packages emit spans through the global tracer provider and do not configure OTel. The
provider is installed by `aiserver.InitClient` / `aiserver.InitWithClient` (OTLP over HTTP via
`InitOptions.OTLPEndpoint`), with `aiserver.ConversationIDSpanProcessor` registered. The OTel SDK
is already a dependency of `aiserver`; there is no extra to install. See the core client's
`agents.md` for endpoint configuration.

## `InitClient()`: when to call it

Lazy initialization is the default: `aiserver.Config(...).Invoke`, `ClaudeGraph(...).Invoke` and
`aiserver.ResolveGraph` initialize the client from `LD_SDK_KEY` on first use. Call
`aiserver.InitClient` explicitly for custom options (base URIs, OTLP endpoint, service name),
call `aiserver.InitWithClient` to bring your own LaunchDarkly client (BYOC, including edge
runtimes), or call it at startup to pre-warm the connection.

## Dependencies

| Dependency | Why |
| --- | --- |
| `github.com/launchdarkly/go-ai-sdk/aiserver` | Handler contract, prompt composition, span kit, graph definition, tracking, usage reporting. |
| `github.com/launchdarkly/go-ai-sdk/internal/anthropicwire` | Anthropic wire mapping shared with `claudemessages`. |
| `github.com/anthropics/anthropic-sdk-go` | Messages API client, server-tool params, streaming accumulator. |
| `github.com/launchdarkly/go-sdk-common/v3` | `ldcontext.Context` for the wrapper and graph tracking. |
| `go.opentelemetry.io/otel` | The `ld.ai.graph` span in `ToClaudeAgents`. |
| `github.com/google/uuid` | Per-run graph run id; NotebookEdit cell ids. |

## Common pitfalls

- **Host built-ins run model-chosen actions on this machine.** File tools are confined to
  `Options.WorkDir`; `ClaudeBash` is not (it only starts there) and inherits the environment.
  Never weaken `workspace.rel` or route file I/O around `w.root`; the abuse matrix in
  `hosttools_test.go` must keep passing.
- **`WorkDir` defaults to the process working directory at handler construction**, not at call
  time; a later `os.Chdir` does not move it.
- **Not Claude Code parity:** no sandbox or permission prompts, no `.gitignore` handling in
  Glob/Grep (only `.git` is skipped), Grep uses RE2 not ripgrep and supports only `pattern`,
  `path`, `glob`, `output_mode`, `-i`, `-n`, `-A`, `-B`, `-C`, `head_limit`, and Read does not
  render images, PDFs or notebooks. Edit and Write (over an existing file) follow Claude Code's
  read-before-edit rule through `workspace.seen` (modification time at the last Read or write).
- **A NativeTool needs a config tool entry.** Like function tools, a sentinel is only
  enabled when its map key is also a key in `config.tools`; JS enables any NativeTool present in
  `toolHandlers`.
- **`graph()` streams.** `ClaudeGraph` nodes run through the handler's `Stream`, so the
  provider sees streaming requests and `outputFormat` is not applied inside a graph.
- **Blocking and streaming outputs differ after a tool turn.** Blocking returns the last
  response's text; the stream's `done` output concatenates every turn.
- **Non-Anthropic nodes in `ToClaudeAgents` need handlers on the definition.** They run through
  `RunNode`, which uses the handlers passed to `aiserver.ResolveGraph`, and cannot call their own
  children.
