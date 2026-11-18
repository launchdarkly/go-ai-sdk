package skills

import (
	"fmt"
	"regexp"
	"sync"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/bridge"
)

func init() {
	// The core adds an attached skill's tools to the tools a handler receives.
	bridge.SkillTools = skillTools
}

// skillTools returns the tools of one attached skill, read through the core
// retrieval path so no usage is reported. With no store it returns nil
// silently: a variation's skills may be in use for WriteSkills alone.
func skillTools(key string, version int) []aiserver.Tool {
	s := currentStore()
	if s == nil {
		return nil
	}
	res := resolveFromStore(s, key, version)
	if res.skill == nil {
		msg := fmt.Sprintf("skill '%s' version %d adds no tools (%s): %s", safeKey(key), version, res.reason, res.err)
		if !storeIsInitialized(s) {
			// A miss before the store's first payload does not use up the warning.
			currentLogger().Warn(msg, "skill_key", safeKey(key), "skill_version", version, "reason", string(res.reason))
			return nil
		}
		warnOnce(toolWarning{key, version, "", warnUnreadable}, msg, "reason", string(res.reason))
		return nil
	}
	return res.skill.tools
}

// decodeSkillTools reads an envelope's optional tools array with the
// per-entry rule kitParseAiConfig applies to config.tools. A malformed entry
// is dropped with a warning; the skill is still returned and no integrity
// signal is recorded, because contentHash does not cover tools.
func decodeSkillTools(key string, version int, v any, present bool) []aiserver.Tool {
	if !present {
		return nil
	}
	entries, ok := v.([]any)
	if !ok {
		warnOnce(toolWarning{key, version, "", warnNotArray}, fmt.Sprintf("skill '%s' version %d has a tools value that is not an array; no tools are offered", safeKey(key), version))
		return nil
	}
	var out []aiserver.Tool
	seen := map[string]bool{}
	for i, e := range entries {
		obj, _ := e.(map[string]any)
		name, nameOK := obj["name"].(string)
		tool := toolLabel(i, name)
		params, paramsOK := obj["parameters"].(map[string]any)
		var problem string
		switch {
		case !nameOK:
			problem = "has no string name"
		case !paramsOK:
			problem = "has no object parameters"
		case obj["type"] != "function":
			problem = `has a type that is not "function"`
		case seen[name]:
			problem = "repeats an earlier tool's name"
		}
		if problem != "" {
			warnOnce(toolWarning{key, version, tool, warnDropped}, fmt.Sprintf("skill '%s' version %d: tool %s %s and is dropped", safeKey(key), version, tool, problem))
			continue
		}
		seen[name] = true
		t := aiserver.Tool{Name: name, Type: "function", Parameters: deepCloneMap(params)}
		// A wrongly typed optional field is dropped, keeping the tool.
		if d, present := obj["description"]; present && d != nil {
			if t.Description, ok = d.(string); !ok {
				warnOnce(toolWarning{key, version, tool, warnBadDescription}, fmt.Sprintf("skill '%s' version %d: tool %s has a description that is not a string; it is offered without one", safeKey(key), version, tool))
			}
		}
		if cp, present := obj["customParameters"]; present && cp != nil {
			m, isMap := cp.(map[string]any)
			if !isMap {
				warnOnce(toolWarning{key, version, tool, warnBadCustomParameters}, fmt.Sprintf("skill '%s' version %d: tool %s has customParameters that is not an object; it is offered without them", safeKey(key), version, tool))
			}
			t.CustomParameters = deepCloneMap(m)
		}
		out = append(out, t)
	}
	return out
}

// The envelope for a key and version never changes and skills are read on
// every run, so each drop warns once per process (until aiserver.Shutdown).
var (
	toolWarnMu sync.Mutex
	toolWarned = map[toolWarning]bool{}
)

type toolWarning struct {
	key     string
	version int
	tool    string
	kind    int
}

const (
	warnUnreadable = iota
	warnNotArray
	warnDropped
	warnBadDescription
	warnBadCustomParameters
)

func warnOnce(id toolWarning, msg string, attrs ...any) {
	toolWarnMu.Lock()
	seen := toolWarned[id]
	toolWarned[id] = true
	toolWarnMu.Unlock()
	if seen {
		return
	}
	attrs = append([]any{"skill_key", safeKey(id.key), "skill_version", id.version}, attrs...)
	if id.kind >= warnDropped {
		attrs = append(attrs, "tool", id.tool)
	}
	currentLogger().Warn(msg, attrs...)
}

func clearToolWarnings() {
	toolWarnMu.Lock()
	defer toolWarnMu.Unlock()
	clear(toolWarned)
}

// logSafeToolName is the shape a tool name must have to be quoted in a log
// line, so a name from the wire cannot inject into it.
var logSafeToolName = regexp.MustCompile(`^[A-Za-z0-9._-]{1,256}$`)

// toolLabel names a skill's tool by position, with its name only when the
// name is log-safe: tools[1] ('get-weather').
func toolLabel(i int, name string) string {
	if logSafeToolName.MatchString(name) {
		return fmt.Sprintf("tools[%d] ('%s')", i, name)
	}
	return fmt.Sprintf("tools[%d]", i)
}

// cloneTools deep-copies tools so no caller shares a schema map with a Skill.
// It never returns nil.
func cloneTools(tools []aiserver.Tool) []aiserver.Tool {
	out := make([]aiserver.Tool, len(tools))
	for i, t := range tools {
		t.Parameters = deepCloneMap(t.Parameters)
		t.CustomParameters = deepCloneMap(t.CustomParameters)
		out[i] = t
	}
	return out
}

func deepCloneMap(m map[string]any) map[string]any {
	if m == nil {
		return nil
	}
	out := make(map[string]any, len(m))
	for k, v := range m {
		out[k] = deepCloneValue(v)
	}
	return out
}

func deepCloneValue(v any) any {
	switch x := v.(type) {
	case map[string]any:
		return deepCloneMap(x)
	case []any:
		out := make([]any, len(x))
		for i, e := range x {
			out[i] = deepCloneValue(e)
		}
		return out
	}
	return v
}
