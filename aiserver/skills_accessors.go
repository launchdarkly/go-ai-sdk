package aiserver

import "fmt"

// SkillRefs projects a resolved AI Config's "skills" array into references.
// It is pure: no network, no client, no store, no telemetry.
//
// A nil config, which is what a failed InspectConfig returns, is an
// error rather than an empty list: an empty list handed to WriteSkills would
// prune every managed skill. A config without "skills" returns an empty list.
// A "skills" value that is not an array, or any malformed entry (reachable
// only by a config that bypassed ParseAiConfig), rejects the whole
// field, as the JavaScript and Python SDKs do: a partial list handed to
// WriteSkills with pruning would delete the dropped skill's files.
func SkillRefs(config *AiConfig) ([]SkillReference, error) {
	if config == nil {
		return nil, invalidArg("SkillRefs was given a nil AI Config. A failed InspectConfig returns a nil " +
			"Config; handle that before deriving skill references, because an empty list passed to " +
			"WriteSkills would prune every skill it manages")
	}
	refs := []SkillReference{}
	raw, present := config.Raw["skills"]
	if !present {
		return refs, nil
	}
	entries, ok := raw.([]any)
	if !ok {
		return nil, invalidArg("the AI Config's skills field is not an array of {key, version} objects")
	}
	for i, e := range entries {
		entry, _ := e.(map[string]any)
		key, _ := entry["key"].(string)
		version, versionOK := versionOf(entry["version"])
		if entry == nil || !isValidSkillKey(entry["key"]) || !versionOK {
			return nil, invalidArg(fmt.Sprintf("the AI Config's skills entry at position %d is not a {key, version} "+
				"object with a valid key and an integer version >= 1", i))
		}
		refs = append(refs, SkillReference{key: key, version: version})
	}
	return refs, nil
}

// GetSkill returns one verified skill, or nil when it is missing, at another
// version, or fails verification (GetSkillResult says which). version 0 means
// the newest version the store holds. The only error is ErrNoSkillStore.
func GetSkill(key string, version int) (*Skill, error) {
	s, err := requireStore()
	if err != nil {
		return nil, err
	}
	return resolveFromStore(s, key, version).skill, nil
}

// GetSkillResult is GetSkill reporting why there is no skill. It runs the
// same retrieval, verification and telemetry exactly once.
func GetSkillResult(key string, version int) (SkillOutcome, error) {
	s, err := requireStore()
	if err != nil {
		return SkillOutcome{}, err
	}
	r := resolveFromStore(s, key, version)
	return SkillOutcome{skill: r.skill, reason: r.reason, detail: r.err}, nil
}

// GetSkills returns the verified skills for refs, in input order. Entries that
// are missing, at another version, or fail verification are omitted.
func GetSkills[R SkillRequest](refs []R) ([]Skill, error) {
	s, err := requireStore()
	if err != nil {
		return nil, err
	}
	out := []Skill{}
	served := 0
	for _, ref := range refs {
		key, version := ref.target()
		r := resolveFromStore(s, key, version)
		if r.skill != nil {
			out = append(out, *r.skill)
		}
		if r.reason == ReasonOK || r.reason == ReasonIntegrityFailure {
			served++
		}
	}
	logWithholdingSummary("requested skills the store served", served, len(out))
	return out, nil
}

// AllSkills returns every verified skill the store holds, the newest version
// per key, omitting any that fail verification. A store error yields an empty
// list (logged); WriteAllSkills does not use this path for that reason.
func AllSkills() ([]Skill, error) {
	s, err := requireStore()
	if err != nil {
		return nil, err
	}
	objects, err := allObjects(s)
	if err != nil {
		currentLogger().Error("the skill store failed while listing skills", "error", err)
		return []Skill{}, nil
	}
	candidates := newestByKey(objects)
	out := []Skill{}
	for _, c := range candidates {
		if sk := verifyRawSkill(c.raw); sk != nil {
			out = append(out, *sk)
		}
	}
	logWithholdingSummary("skills held by the store", len(candidates), len(out))
	return out, nil
}
