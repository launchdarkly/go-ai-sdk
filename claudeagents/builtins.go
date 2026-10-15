package claudeagents

import "github.com/launchdarkly/go-ai-sdk/aiserver"

// Claude Code built-in tool sentinels. Put one as a value in a
// toolHandlers map, under the config's tool key, to enable that capability.
//
// ClaudeWebSearch and ClaudeWebFetch map to Anthropic server tools, which
// Anthropic executes. The other eight are offered to the model under their
// Claude Code name and input schema and executed by this package on the
// host, inside Options.WorkDir. ClaudeBash runs arbitrary shell commands
// with the process environment: enable it only for trusted configs.
var (
	// ClaudeBash runs a command with /bin/bash -c in Options.WorkDir.
	ClaudeBash = &aiserver.NativeTool{ToolName: "Bash"}
	// ClaudeRead reads a file inside Options.WorkDir, line-numbered.
	ClaudeRead = &aiserver.NativeTool{ToolName: "Read"}
	// ClaudeEdit replaces an exact string in a file inside Options.WorkDir.
	ClaudeEdit = &aiserver.NativeTool{ToolName: "Edit"}
	// ClaudeWrite creates or overwrites a file inside Options.WorkDir.
	ClaudeWrite = &aiserver.NativeTool{ToolName: "Write"}
	// ClaudeGlob finds files by glob pattern inside Options.WorkDir.
	ClaudeGlob = &aiserver.NativeTool{ToolName: "Glob"}
	// ClaudeGrep searches file contents inside Options.WorkDir.
	ClaudeGrep = &aiserver.NativeTool{ToolName: "Grep"}
	// ClaudeWebFetch maps to the Anthropic web_fetch server tool.
	ClaudeWebFetch = &aiserver.NativeTool{ToolName: "WebFetch"}
	// ClaudeWebSearch maps to the Anthropic web_search server tool.
	ClaudeWebSearch = &aiserver.NativeTool{ToolName: "WebSearch"}
	// ClaudeTodoWrite keeps a task list in memory for the current run.
	ClaudeTodoWrite = &aiserver.NativeTool{ToolName: "TodoWrite"}
	// ClaudeNotebookEdit edits a Jupyter notebook cell inside Options.WorkDir.
	ClaudeNotebookEdit = &aiserver.NativeTool{ToolName: "NotebookEdit"}
)
