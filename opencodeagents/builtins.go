package opencodeagents

import "github.com/launchdarkly/go-ai-sdk/aiserver"

// The OpenCode built-in tools, as NativeTool sentinels. OpenCode has no list
// tool, so there is no OpenCodeList.
var (
	OpenCodeBash       *aiserver.NativeTool
	OpenCodeRead       *aiserver.NativeTool
	OpenCodeEdit       *aiserver.NativeTool
	OpenCodeWrite      *aiserver.NativeTool
	OpenCodeGlob       *aiserver.NativeTool
	OpenCodeGrep       *aiserver.NativeTool
	OpenCodeTask       *aiserver.NativeTool
	OpenCodeSkill      *aiserver.NativeTool
	OpenCodeTodoWrite  *aiserver.NativeTool
	OpenCodeWebFetch   *aiserver.NativeTool
	OpenCodeWebSearch  *aiserver.NativeTool
	OpenCodeApplyPatch *aiserver.NativeTool
)
