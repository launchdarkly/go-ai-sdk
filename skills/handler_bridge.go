package skills

import (
	"fmt"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/bridge"
)

func init() { bridge.WriteConfigSkills = writeConfigSkills }

// writeConfigSkills is SkillRefs then WriteSkills with pruning off, reporting
// no usage: the Claude Agents skills handler's entry point. Each reference is
// resolved once, so the description returned is the written version's. A
// resolution failure stops before anything is written.
func writeConfigSkills(config any, root string) ([]bridge.WrittenSkill, []bridge.SkillFailure, error) {
	c, _ := config.(*aiserver.AiConfig)
	refs, err := skillRefs(c)
	if err != nil {
		return nil, nil, err
	}
	if !supportsDirFD {
		return nil, nil, ErrUnsupportedPlatform
	}
	opts := &WriteSkillsOptions{NoPrune: true}
	cfg, err := parseWriteOptions(opts)
	if err != nil {
		return nil, nil, err
	}
	items := make([]SkillInput, len(refs))
	for i, r := range refs {
		items[i] = r
	}
	pending, _, err := resolveRequests(selection{items: items}, cfg)
	if err != nil {
		return nil, nil, err
	}
	var (
		written  []bridge.WrittenSkill
		failures []bridge.SkillFailure
		resolved []SkillInput
	)
	for _, p := range pending {
		if p.skill == nil {
			msg := p.err
			if msg == "" { // writeAll's wording for the same case
				msg = fmt.Sprintf("skill '%s' could not be resolved", safeKey(p.key))
			}
			failures = append(failures, bridge.SkillFailure{Key: p.key, Message: msg})
			continue
		}
		resolved = append(resolved, *p.skill)
		written = append(written, bridge.WrittenSkill{Key: p.skill.key, Description: p.skill.description})
	}
	if len(failures) > 0 {
		return nil, failures, nil
	}
	rep, err := writeSkills(selection{items: resolved}, root, opts)
	if err != nil {
		return nil, nil, err
	}
	for _, a := range rep.Errors() {
		failures = append(failures, bridge.SkillFailure{Key: a.key, Message: a.err})
	}
	if len(failures) > 0 {
		return nil, failures, nil
	}
	return written, nil, nil
}
