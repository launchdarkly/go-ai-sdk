package claudeagents

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"os"
	"path/filepath"
	"strings"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/anthropicwire"
	"github.com/launchdarkly/go-ai-sdk/internal/bridge"
	"github.com/launchdarkly/go-ai-sdk/internal/handlerkit"
	"github.com/launchdarkly/go-ai-sdk/internal/sdkapi"
	"github.com/launchdarkly/go-ai-sdk/skills" // also installs the bridge's skills writer
)

// skillToolName is the name of the tool the skills handler offers the model.
// It stands in for the Claude Code Skill built-in.
const skillToolName = "Skill"

// skillNamespace qualifies a skill key in the Skill tool, as the Claude Code
// plugin namespace does.
const skillNamespace = "launchdarkly:"

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
// skills fails closed with an error wrapping skills.ErrUnsupportedPlatform.
func NewSkillsHandler(opts SkillsHandlerOptions) (*aiserver.ProviderHandler, error) {
	handlerkit.ReportUsage("claude-agents.createClaudeAgentsSkillsHandler", PackageName, PackageVersion)
	s := &skillsAgent{agent: newAgent(opts.Options), parent: opts.SkillsTempDir}
	return sdkapi.CreateHandler(providesFor, s.invoke, s.stream, opts.CaptureContent), nil
}

type skillsAgent struct {
	*agent
	parent string // "" means os.TempDir() at invoke time
}

func (s *skillsAgent) invoke(ctx context.Context, req aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
	if !hasSkills(req.Config) {
		return s.agent.invoke(ctx, req)
	}
	inv, err := s.materialize(req.Config)
	if err != nil {
		return aiserver.HandlerResult{}, err
	}
	defer inv.remove()
	return s.invokeWith(ctx, req, []localTool{inv.tool()})
}

func (s *skillsAgent) stream(ctx context.Context, req aiserver.HandlerRequest) iter.Seq2[aiserver.HandlerStreamEvent, error] {
	if !hasSkills(req.Config) {
		return s.agent.stream(ctx, req)
	}
	return func(yield func(aiserver.HandlerStreamEvent, error) bool) {
		inv, err := s.materialize(req.Config)
		if err != nil {
			yield(aiserver.HandlerStreamEvent{}, err)
			return
		}
		defer inv.remove()
		s.streamWith(ctx, req, []localTool{inv.tool()}, yield)
	}
}

// hasSkills is false when the config has no "skills" field or an empty one:
// that run is the core handler's. Anything else is validated by the writer.
func hasSkills(cfg aiserver.AiConfig) bool {
	v, ok := cfg.Raw["skills"]
	if !ok {
		return false
	}
	list, isList := v.([]any)
	return !isList || len(list) > 0
}

// invokeSkills is one invoke's private skills directory.
type invokeSkills struct {
	dir     string
	root    *os.Root          // <dir>/launchdarkly/skills
	keys    map[string]string // qualified name → key
	listing string            // the Skill tool's skill list
}

// materialize creates the invoke's directory and writes the variation's
// skills into <dir>/launchdarkly/skills. On any failure nothing is left.
func (s *skillsAgent) materialize(cfg aiserver.AiConfig) (*invokeSkills, error) {
	parent := s.parent
	if parent == "" {
		parent = os.TempDir()
	}
	// Resolved once, so a parent symlink retargeted mid-run changes nothing.
	real, err := filepath.EvalSymlinks(parent)
	if err != nil {
		return nil, fmt.Errorf("claudeagents: the skills temp directory is unusable: %w", err)
	}
	dir, err := os.MkdirTemp(real, "ld-skills-")
	if err != nil {
		return nil, fmt.Errorf("claudeagents: create the skills directory: %w", err)
	}
	inv := &invokeSkills{dir: dir, keys: map[string]string{}}
	if err := inv.write(cfg); err != nil {
		inv.remove()
		return nil, err
	}
	return inv, nil
}

func (inv *invokeSkills) write(cfg aiserver.AiConfig) error {
	ns := filepath.Join(inv.dir, "launchdarkly")
	if err := os.Mkdir(ns, 0o700); err != nil {
		return fmt.Errorf("claudeagents: create the skills directory: %w", err)
	}
	skillsDir := filepath.Join(ns, "skills")
	written, failures, err := bridge.WriteConfigSkills(&cfg, skillsDir)
	if err != nil {
		return fmt.Errorf("claudeagents: the variation's skills could not be written: %w", err)
	}
	if len(failures) > 0 {
		var keys, msgs []string
		for _, f := range failures {
			if f.Key != "" {
				keys = append(keys, f.Key)
			}
			msgs = append(msgs, f.Message)
		}
		return fmt.Errorf("claudeagents: skills [%s] could not be written, so the model was not called: %s",
			strings.Join(keys, ", "), strings.Join(msgs, "; "))
	}
	root, err := os.OpenRoot(skillsDir)
	if err != nil {
		return fmt.Errorf("claudeagents: open the skills directory: %w", err)
	}
	inv.root = root
	var b strings.Builder
	for _, w := range written {
		inv.keys[skillNamespace+w.Key] = w.Key
		fmt.Fprintf(&b, "\n- %s%s: %s", skillNamespace, w.Key, w.Description)
	}
	inv.listing = b.String()
	return nil
}

// remove deletes the directory. A failure is logged and never replaces the
// run's outcome.
func (inv *invokeSkills) remove() {
	if inv.root != nil {
		_ = inv.root.Close()
	}
	if err := os.RemoveAll(inv.dir); err != nil {
		bridge.Logger().Warn("claudeagents: the invoke's skills directory could not be removed", "dir", inv.dir, "error", err)
	}
}

// tool is the Go stand-in for the Claude Code Skill built-in. It serves only
// a listed skill's SKILL.md, read through the os.Root on the skills
// directory; anything else is a tool error the model sees.
func (inv *invokeSkills) tool() localTool {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"skill": map[string]any{"type": "string", "description": `The skill's name, e.g. "launchdarkly:<key>".`},
		},
		"required": []any{"skill"},
	}
	desc := "Load a skill's instructions (its SKILL.md). Call it when a task matches a skill below. Available skills:" + inv.listing
	return localTool{
		name:  skillToolName,
		param: anthropicwire.FunctionTool(skillToolName, desc, schema),
		soft:  true,
		fn: func(_ context.Context, args map[string]any) (any, error) {
			name, _ := args["skill"].(string)
			key, ok := inv.keys[name]
			if !ok {
				return nil, fmt.Errorf("unknown skill %q; available skills:%s", name, inv.listing)
			}
			b, err := inv.root.ReadFile(key + "/" + skills.SkillFilename)
			if err != nil {
				return nil, errors.New("skill " + name + " could not be read")
			}
			return string(b), nil
		},
	}
}
