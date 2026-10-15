package aiserver

import (
	"encoding/json"
	"math"
	"regexp"
)

var skillKeyRE = regexp.MustCompile(`^[a-z0-9][a-z0-9-]*$`)

func isObject(v any) (map[string]any, bool) {
	m, ok := v.(map[string]any)
	return m, ok
}

// ParseAiConfig validates a decoded flag variation. raw is the
// encoding/json decoding of the variation (map[string]any). The JS and Python
// `{success, data} | {success:false, error}` maps to Go's (value, error).
func ParseAiConfig(raw any) (AiConfig, error) {
	obj, ok := isObject(raw)
	if !ok {
		return AiConfig{}, errorf("config must be an object")
	}
	model, ok := isObject(obj["model"])
	if _, isStr := model["name"].(string); !ok || !isStr {
		return AiConfig{}, errorf("model.name is required and must be a string")
	}
	provider, ok := isObject(obj["provider"])
	if _, isStr := provider["name"].(string); !ok || !isStr {
		return AiConfig{}, errorf("provider.name is required and must be a string")
	}
	_, hasInstructions := obj["instructions"].(string)
	msgs, isArr := obj["messages"].([]any)
	if !hasInstructions && len(msgs) == 0 {
		return AiConfig{}, errorf("AiConfig must have either instructions or a non-empty messages array")
	}
	if isArr {
		for _, m := range msgs {
			mo, _ := isObject(m)
			switch mo["role"] {
			case "user", "assistant", "system":
			default:
				return AiConfig{}, errorf("invalid message role: %v", mo["role"])
			}
		}
	}
	if t, present := obj["tools"]; present {
		tools, ok := isObject(t)
		if !ok {
			return AiConfig{}, errorf("tools must be an object")
		}
		for k, v := range tools {
			if err := validateTool(k, v); err != nil {
				return AiConfig{}, err
			}
		}
	}
	if of, present := obj["outputFormat"]; present {
		if _, ok := isObject(of); !ok {
			return AiConfig{}, errorf("outputFormat must be an object (JSON Schema)")
		}
	}
	if s, present := obj["skills"]; present {
		if err := validateSkills(s); err != nil {
			return AiConfig{}, err
		}
	}

	b, err := json.Marshal(obj)
	if err != nil {
		return AiConfig{}, errorf("config is not JSON-encodable: %v", err)
	}
	var cfg AiConfig
	if err := json.Unmarshal(b, &cfg); err != nil {
		return AiConfig{}, errorf("config has a wrongly typed field: %v", err)
	}
	cfg.Raw = obj
	return cfg, nil
}

func validateTool(key string, v any) error {
	t, ok := isObject(v)
	if !ok {
		return errorf("tools.%s must be an object", key)
	}
	if _, ok := t["name"].(string); !ok {
		return errorf("tools.%s.name must be a string", key)
	}
	if t["type"] != "function" {
		return errorf(`tools.%s.type must be "function"`, key)
	}
	if _, ok := isObject(t["parameters"]); !ok {
		return errorf("tools.%s.parameters must be an object", key)
	}
	return nil
}

// validateSkills fails closed on any malformed `skills` entry, including null.
func validateSkills(v any) error {
	arr, ok := v.([]any)
	if !ok {
		return errorf("skills must be an array")
	}
	for i, e := range arr {
		entry, ok := isObject(e)
		if !ok {
			return errorf("skills[%d] must be an object", i)
		}
		key, ok := entry["key"].(string)
		if !ok || len(key) > 256 || !skillKeyRE.MatchString(key) {
			return errorf("skills[%d].key is invalid", i)
		}
		// encoding/json decodes numbers to float64, so a bool or string fails
		// the type assertion and 2.5 fails the integer check.
		ver, ok := entry["version"].(float64)
		if !ok || ver < 1 || ver != math.Trunc(ver) {
			return errorf("skills[%d].version must be an integer >= 1", i)
		}
	}
	return nil
}
