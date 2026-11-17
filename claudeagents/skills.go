package claudeagents

import "github.com/launchdarkly/go-ai-sdk/aiserver"

// SkillToolName is the name of the tool the skills handler offers the model.
// It stands in for the Claude Code Skill built-in.
const SkillToolName = "Skill"

// SkillsHandlerOptions configure NewSkillsHandler.
type SkillsHandlerOptions struct {
	Options
	// SkillsTempDir is the parent of each invoke's private skills directory.
	// Empty means os.TempDir().
	SkillsTempDir string
}

// NewSkillsHandler returns the Anthropic agent handler that gives the model
// a variation's skills (createClaudeAgentsSkillsHandler).
//
// On each invoke it writes the variation's pinned skills from the store set
// with skills.SetSkillStore into a private directory under SkillsTempDir,
// offers the model a Skill tool that returns a listed skill's SKILL.md, and
// removes the directory when the run ends. If any skill cannot be written the
// invoke fails before the model is called, and the error names the skills.
// A variation with no skills runs exactly as NewHandler's would.
//
// Materialization is supported on unix only; elsewhere a variation with
// skills fails closed.
func NewSkillsHandler(opts SkillsHandlerOptions) (*aiserver.ProviderHandler, error) {
	return newHandler(opts.Options), nil
}
