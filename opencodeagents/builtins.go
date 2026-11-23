package opencodeagents

import "github.com/launchdarkly/go-ai-sdk/aiserver"

// The OpenCode built-in tools, as NativeTool sentinels: put one in
// ToolHandlers under a config tool key to allow that built-in for the call.
// OpenCode has no list tool, so there is no OpenCodeList. OpenCodeEdit,
// OpenCodeWrite and OpenCodeApplyPatch are one grant: OpenCode checks all
// three against the edit permission.
var (
	OpenCodeBash       = &aiserver.NativeTool{ToolName: "bash"}
	OpenCodeRead       = &aiserver.NativeTool{ToolName: "read"}
	OpenCodeEdit       = &aiserver.NativeTool{ToolName: "edit"}
	OpenCodeWrite      = &aiserver.NativeTool{ToolName: "write"}
	OpenCodeGlob       = &aiserver.NativeTool{ToolName: "glob"}
	OpenCodeGrep       = &aiserver.NativeTool{ToolName: "grep"}
	OpenCodeTask       = &aiserver.NativeTool{ToolName: "task"}
	OpenCodeSkill      = &aiserver.NativeTool{ToolName: "skill"}
	OpenCodeTodoWrite  = &aiserver.NativeTool{ToolName: "todowrite"}
	OpenCodeWebFetch   = &aiserver.NativeTool{ToolName: "webfetch"}
	OpenCodeWebSearch  = &aiserver.NativeTool{ToolName: "websearch"}
	OpenCodeApplyPatch = &aiserver.NativeTool{ToolName: "apply_patch"}
)
