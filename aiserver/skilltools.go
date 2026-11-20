package aiserver

import (
	"fmt"
	"maps"
	"reflect"
	"regexp"
	"sync"

	"github.com/launchdarkly/go-ai-sdk/internal/bridge"
)

// withSkillTools adds the tools of the skills cfg attaches to cfg.Tools, the
// tools a handler receives. It runs once per resolution, for Config runs and
// graph nodes only: judges, InspectConfig, shadow runs and evaluations never
// merge. Skills are read in the variation's skills order
// and an exact duplicate is skipped. A key the variation already has keeps
// the variation's definition, and a key two skills define differently keeps
// the first skill's, each with a warning. A skill that cannot be read, or no
// skills package or store, adds nothing.
//
// The config it returns is marked as merged, and a marked config is returned
// unchanged, reading no skill and logging nothing: merging again would find
// each skill tool already in cfg.Tools and warn as for a variation conflict.
func withSkillTools(configKey string, cfg AiConfig) AiConfig {
	if cfg.skillToolsMerged {
		return cfg
	}
	cfg.skillToolsMerged = true
	refs, _ := cfg.Raw["skills"].([]any)
	resolve, _ := bridge.SkillTools.(func(string, int) []Tool)
	if len(refs) == 0 || resolve == nil {
		return cfg
	}
	tools := maps.Clone(cfg.Tools)
	from := map[string]string{} // tool key → "skill-key@version" that added it
	for _, r := range refs {
		// kitParseAiConfig has validated every entry.
		ref, _ := r.(map[string]any)
		skillKey, _ := ref["key"].(string)
		v, _ := ref["version"].(float64)
		version := int(v)
		skill := fmt.Sprintf("%s@%d", skillKey, version)
		for _, t := range resolve(skillKey, version) {
			if _, own := cfg.Tools[t.Name]; own {
				warnSkillToolOnce(configKey, skill, t.Name, fmt.Sprintf(
					"skill '%s' version %d pins tool %s, which AI Config '%s' also defines; the variation's definition is offered",
					skillKey, version, logToolKey(t.Name), configKey))
				continue
			}
			if held, dup := tools[t.Name]; dup {
				if !reflect.DeepEqual(held, t) {
					warnSkillToolOnce(configKey, skill, t.Name, fmt.Sprintf(
						"skills '%s' and '%s' on AI Config '%s' define tool %s differently; the first skill's definition is offered",
						from[t.Name], skill, configKey, logToolKey(t.Name)))
				}
				continue
			}
			if tools == nil {
				tools = map[string]Tool{}
			}
			tools[t.Name] = t
			from[t.Name] = skill
		}
	}
	cfg.Tools = tools
	return cfg
}

// The skill envelope for a key and version never changes, so each conflict
// warns once per process (until Shutdown).
var (
	skillToolWarnMu sync.Mutex
	skillToolWarned = map[[3]string]bool{}
)

func warnSkillToolOnce(configKey, skill, tool, msg string) {
	skillToolWarnMu.Lock()
	id := [3]string{configKey, skill, tool}
	seen := skillToolWarned[id]
	skillToolWarned[id] = true
	skillToolWarnMu.Unlock()
	if !seen {
		sdkLogger().Warn(msg, "config_key", configKey, "skill", skill)
	}
}

// logSafeToolKey is the shape a skill tool key must have to be quoted in a
// log line; keys come from the wire, so anything else could inject into it.
var logSafeToolKey = regexp.MustCompile(`^[A-Za-z0-9._-]{1,256}$`)

func logToolKey(key string) string {
	if logSafeToolKey.MatchString(key) {
		return "'" + key + "'"
	}
	return "(a key that is not log-safe)"
}

func resetSkillToolWarnings() {
	skillToolWarnMu.Lock()
	defer skillToolWarnMu.Unlock()
	clear(skillToolWarned)
}
