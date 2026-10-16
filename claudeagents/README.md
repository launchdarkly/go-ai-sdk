# claudeagents

LaunchDarkly AI SDK agent handler for Anthropic Claude, built on the Anthropic Messages API
([`anthropic-sdk-go`](https://github.com/anthropics/anthropic-sdk-go)) with an in-process agent
loop. Go has no Claude Agent SDK, so this package reproduces the observable contract of the
JavaScript and Python `claude-agents` packages (returns, tracking, spans, graph events), and runs
the Claude Code built-in tools (`Bash`, `Read`, `Edit`, ...) itself.

> **Security: the built-ins act on your host.** `ClaudeBash`, `ClaudeRead`, `ClaudeEdit`,
> `ClaudeWrite`, `ClaudeGlob`, `ClaudeGrep` and `ClaudeNotebookEdit` execute the actions the
> *model* chooses, in this process, with this process's user and permissions. File tools are
> confined to `Options.WorkDir`. **`ClaudeBash` is not sandboxed**: it runs any command the
> model writes with `/bin/bash -c`, starting in `WorkDir`, with the full process environment
> (including secrets such as `ANTHROPIC_API_KEY` and `LD_SDK_KEY`), and can read, change or send
> anything the process user can. Only enable these sentinels for configs and inputs you trust, and
> run in a container or VM with a dedicated working directory and least-privilege credentials.

```sh
go get github.com/launchdarkly/go-ai-sdk/claudeagents
```

## providesFor

`{Provider: "Anthropic", Mode: "agent"}`. The handler runs for AI Config variations whose
`provider.name` is `Anthropic` and whose `_ldMeta.mode` is `agent`.

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
	"github.com/launchdarkly/go-ai-sdk/claudeagents"
)

func main() {
	ctx := context.Background()
	if _, err := aiserver.InitClient(ctx, aiserver.InitOptions{}); err != nil { // reads LD_SDK_KEY
		log.Fatal(err)
	}
	defer aiserver.Shutdown(ctx)

	h := claudeagents.NewHandler(claudeagents.Options{}) // reads ANTHROPIC_API_KEY
	res, err := aiserver.Config(aiserver.ConfigArgs{
		Key:      "my-agent-config",
		Handlers: []*aiserver.ProviderHandler{h},
		ToolHandlers: map[string]aiserver.ToolHandler{
			"web-search": claudeagents.ClaudeWebSearch, // config tool key → Anthropic web_search
			"lookup": aiserver.ToolFunc(func(ctx context.Context, args map[string]any) (any, error) {
				return "order 42 shipped", nil
			}),
		},
	}).Invoke(ctx, "Where is my order?", ldcontext.New("user-123"))
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(res.Response)
}
```

Multiple handlers: the variation's provider and mode pick one.

```go
handlers := []*aiserver.ProviderHandler{
	claudeagents.NewHandler(claudeagents.Options{}),
	claudemessages.NewHandler(claudemessages.Options{}), // github.com/launchdarkly/go-ai-sdk/claudemessages
}
res, err := aiserver.Config(aiserver.ConfigArgs{Key: "my-ai-config", Handlers: handlers}).
	Invoke(ctx, "Hello", ldcontext.New("user-123"))
```

Convenience wrapper:

```go
res, err := claudeagents.ClaudeAgents(ctx, "my-agent-config", "Hello", ldcontext.New("user-123"),
	&claudeagents.InvokeOptions{Variables: map[string]any{"user_input": "Hello"}})
```

Graph convenience wrapper (`graph()` with this handler pre-bound):

```go
out, err := claudeagents.ClaudeGraph("support-graph", claudeagents.GraphOptions{ToolHandlers: tools},
	claudeagents.Options{}).Invoke(ctx, "I was double charged", ldcontext.New("user-123"))
```

Native adapter: every child node becomes a sub-agent tool of its parent, and the root drives
the run.

```go
def, err := aiserver.ResolveGraph(ctx, "support-graph", ldcontext.New("user-123"), aiserver.GraphOptions{})
if err != nil {
	log.Fatal(err)
}
out, err := claudeagents.ToClaudeAgents(def, &claudeagents.AdapterOptions{ToolHandlers: tools}).
	Invoke(ctx, "I was double charged")
```

## Built-in tools

Put a sentinel in `ToolHandlers` under a config tool key. Each model call of the tool sends
`$ld:ai:tool_call` with that key and gets an `execute_tool <name>` span.

| Sentinel | Offered as | Runs where | Behaviour |
| --- | --- | --- | --- |
| `ClaudeWebSearch` | `web_search_20250305` server tool | Anthropic | |
| `ClaudeWebFetch` | `web_fetch_20250910` server tool | Anthropic | |
| `ClaudeBash` | `Bash{command, timeout?, description?}` | host | `/bin/bash -c` in `WorkDir`; timeout in ms, default 2 min, max 10 min (the process group is killed); stdout+stderr combined, capped at 30000 bytes; a non-zero exit returns the output plus `Exit code N` as an error result. |
| `ClaudeRead` | `Read{file_path, offset?, limit?}` | host | `cat -n` style numbering; `offset` is a 1-based line; default 2000 lines; lines over 2000 chars cut; binary files rejected; only the first 10 MiB is scanned. |
| `ClaudeEdit` | `Edit{file_path, old_string, new_string, replace_all?}` | host | Exact match. Error if not found, or found more than once without `replace_all`. Keeps the file mode. The file must have been read with `Read` in this run and not changed since. |
| `ClaudeWrite` | `Write{file_path, content}` | host | Creates or overwrites; creates parent directories. An existing file must have been read with `Read` in this run and not changed since. |
| `ClaudeGlob` | `Glob{pattern, path?}` | host | `**` and `{a,b}` supported; `*.go` matches only the top of `path`; newest first, at most 100; `.git` skipped; symlinked directories not followed. |
| `ClaudeGrep` | `Grep{pattern, path?, glob?, output_mode?, -i?, -n?, -A?, -B?, -C?, head_limit?}` | host | Go RE2 syntax, line by line; `output_mode` `files_with_matches` (default, newest first), `content` (`path:line:text`; context lines from `-A`/`-B`/`-C` as `path-line-text`, groups split by `--`), `count`; binary and >10 MiB files skipped; output capped at 30000 bytes. |
| `ClaudeTodoWrite` | `TodoWrite{todos: [{content, status, activeForm?}]}` | memory | Replaces the run's list (status `pending`, `in_progress` or `completed`) and confirms. The list lives for one run. |
| `ClaudeNotebookEdit` | `NotebookEdit{notebook_path, cell_id?, new_source, cell_type?, edit_mode?}` | host | `.ipynb` only; `replace` (default; clears a code cell's outputs), `insert` (after `cell_id`, or first; needs `cell_type`), `delete`; `cell_id` is a cell id or `cell-N`. |

Confinement: a file path may be relative to `WorkDir` or absolute inside it; `..` is rejected,
and every file operation goes through `os.Root`, so a symlink that points outside `WorkDir` (file
or parent directory) is refused. Any tool failure, including a refused path, goes back to the
model as an `is_error` tool result and marks the span ERROR; it does not fail the run. Config
errors fail the run before the first provider call: a `NativeTool` with any other `ToolName`
(`ErrUnsupportedNativeTool`), a `WorkDir` that cannot be opened, or a function tool whose key
equals a built-in's name.

## Environment variables

| Variable | Required | Description |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | Yes, unless `Options.Client` or `Options.APIKey` is set | Anthropic API key, read by `anthropic-sdk-go`. |
| `ANTHROPIC_BASE_URL` | Optional | Overrides the API base URL (read by `anthropic-sdk-go`). |
| `LD_SDK_KEY` | Yes, unless `InitOptions.SDKKey` or `aiserver.InitWithClient` is used | LaunchDarkly server SDK key, read by `aiserver`. |

`ClaudeBash` commands inherit the whole process environment unchanged, as in Claude Code. To
keep a variable from the model's commands, unset it in the process or run the handler in a
process without it.

## API

```go
const PackageName = "github.com/launchdarkly/go-ai-sdk/claudeagents"
const PackageVersion = "0.1.0"
const SpanName = "claude.query"
const DefaultMaxSteps = 10
const SubAgentToolPrefix = "mcp__subagents__"
var ProvidesFor = aiserver.ProvidesFor{Provider: "Anthropic", Mode: aiserver.ModeAgent}
var ErrToolNotFound, ErrUnsupportedNativeTool, ErrMaxSteps error

type Options struct {
	Client         *anthropic.Client // used when set
	APIKey         string            // used when Client is nil; else ANTHROPIC_API_KEY
	CaptureContent bool              // default false
	MaxSteps       int               // 0 → DefaultMaxSteps; config max_turns wins
	WorkDir        string            // host built-ins run in and are confined to it; "" → process working directory at construction
}
func NewHandler(Options) *aiserver.ProviderHandler

type InvokeOptions struct {
	ToolHandlers map[string]aiserver.ToolHandler
	Registry     *aiserver.Registry
	SkipJudges   bool
	Variables    map[string]any
	Handler      Options
}
func ClaudeAgents(ctx, configKey, userInput string, userCtx ldcontext.Context, opts *InvokeOptions) (aiserver.ProviderResponse, error)

type GraphOptions struct { ToolHandlers; GraphJudge string; Registry } // no Handlers
func ClaudeGraph(key string, opts GraphOptions, handler Options) *aiserver.AgentGraph

type AdapterOptions struct { ToolHandlers map[string]aiserver.ToolHandler; Handler Options }
func ToClaudeAgents(def *aiserver.GraphDefinition, opts *AdapterOptions) *GraphRunner
func (*GraphRunner) Invoke(ctx, input string, opts ...aiserver.CallOption) (aiserver.GraphResponse, error)

var ClaudeBash, ClaudeRead, ClaudeEdit, ClaudeWrite, ClaudeGlob, ClaudeGrep,
	ClaudeWebFetch, ClaudeWebSearch, ClaudeTodoWrite, ClaudeNotebookEdit *aiserver.NativeTool
```

Blocking output is the last model response's text (the Agent SDK `result`). A streamed run's
`done` output is every turn's text concatenated. `config.outputFormat` is appended to the
system prompt as a JSON-schema instruction on the blocking path only.

`config.model.parameters` keys the JavaScript handler forwards to the Agent SDK (camelCase or
snake_case) map onto the Messages API: `maxTurns` → step bound; `thinking` / `maxThinkingTokens`
→ `thinking`; `effort` → `output_config.effort`; `outputFormat` (`{type: "json_schema", schema}`)
→ `output_config.format` (structured outputs, both paths); `betas` → `anthropic-beta` header;
`fallbackModel` → retried once on a 5xx before any output, then used for the rest of the run.
`maxBudgetUsd` is not supported.
