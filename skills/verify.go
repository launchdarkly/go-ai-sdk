package skills

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"maps"
	"slices"
	"unicode/utf8"
)

// verifiedContent is content that passed verification: the verbatim bytes
// and the locally computed hash, never the store's.
type verifiedContent struct {
	bytes []byte
	hash  string
}

// verifyBytes checks size, then encoding, then hash, in that fixed order so
// input failing several checks always reports the same code. wireText
// is true for a wire object's text, which must be valid UTF-8; an already
// verified Skill.Content is opaque bytes and is hashed directly. It returns
// the failure reason, or "" on success, after recording the failure.
func verifyBytes(key string, content []byte, wireText bool, expectedHash string, version int) (verifiedContent, string) {
	if len(content) > maxSkillContentBytes {
		reason := fmt.Sprintf("content is %d bytes, over the %d byte cap", len(content), maxSkillContentBytes)
		recordIntegrityFailure(integrityFailure{key: key, reason: reason, code: codeOverSizeCap, version: version, expectedHash: expectedHash})
		return verifiedContent{}, reason
	}
	if wireText && !utf8.Valid(content) {
		reason := "content is not encodable as UTF-8"
		recordIntegrityFailure(integrityFailure{key: key, reason: reason, code: codeNotUTF8, version: version, expectedHash: expectedHash})
		return verifiedContent{}, reason
	}
	sum := sha256.Sum256(content)
	observed := hex.EncodeToString(sum[:])
	// Exact comparison: a digest differing only in case or padding is rejected,
	// never normalized.
	if observed != expectedHash {
		reason := "content hash mismatch"
		recordIntegrityFailure(integrityFailure{
			key: key, reason: reason, code: codeHashMismatch, version: version,
			expectedHash: expectedHash, observedHash: observed,
		})
		return verifiedContent{}, reason
	}
	return verifiedContent{bytes: content, hash: observed}, ""
}

// verifyRawSkill turns one untrusted raw store object into a Skill, or
// returns nil after recording why. Unverified content is never returned.
func verifyRawSkill(raw any) *Skill {
	obj, ok := raw.(map[string]any)
	if !ok {
		recordIntegrityFailure(integrityFailure{key: nil, reason: "raw skill object is not an object", code: codeNotAnObject})
		return nil
	}
	key, _ := obj["key"].(string)
	if !isValidSkillKey(obj["key"]) {
		recordIntegrityFailure(integrityFailure{key: key, reason: "key is not a valid skill key", code: codeInvalidKey})
		return nil
	}
	version, ok := versionOf(obj["version"])
	if !ok {
		recordIntegrityFailure(integrityFailure{key: key, reason: "version is not an integer >= 1", code: codeInvalidVersion})
		return nil
	}
	content, ok := obj["content"].(string)
	if !ok {
		recordIntegrityFailure(integrityFailure{key: key, reason: "content is missing or not a string", code: codeMissingContent, version: version})
		return nil
	}
	expected, ok := obj["contentHash"].(string)
	if !ok {
		recordIntegrityFailure(integrityFailure{key: key, reason: "contentHash is missing or not a string", code: codeMissingContentHash, version: version})
		return nil
	}
	// []byte(content) copies, so the Skill never shares the store's memory.
	v, failed := verifyBytes(key, []byte(content), true, expected, version)
	if failed != "" {
		return nil
	}
	name, _ := obj["name"].(string)
	description, _ := obj["description"].(string)
	return &Skill{key: key, version: version, content: v.bytes, contentHash: v.hash, name: name, description: description}
}

// resolution is one key resolved against a store: the skill, or why not.
type resolution struct {
	reason SkillOutcomeReason
	skill  *Skill
	err    string
	// unavailable is set exactly when the store could not answer. It
	// suppresses pruning, so an outage never deletes managed files.
	unavailable bool
}

// resolveFromStore is the one retrieval path shared by GetSkill,
// GetSkillResult, GetSkills and WriteSkills.
func resolveFromStore(s SkillStore, key string, wanted int) resolution {
	raw, err := getObject(s, key, wanted)
	if err != nil {
		currentLogger().Error("the skill store failed while retrieving a skill", "skill_key", safeKey(key), "error", err)
		// The store's error stays in the log: its text is not ours to vouch
		// for and can carry a filesystem path (an *fs.PathError), which the
		// outcome detail must never contain.
		return resolution{
			reason:      ReasonStoreUnavailable,
			err:         fmt.Sprintf("skill '%s' could not be retrieved: the skill store returned an error (logged)", safeKey(key)),
			unavailable: true,
		}
	}
	if _, ok := raw.(map[string]any); !ok {
		// A non-object answer to a pinned query says nothing about the key:
		// absent, with no integrity signal.
		return resolution{reason: ReasonAbsent, err: fmt.Sprintf("skill '%s' is not available from the configured skill store", safeKey(key))}
	}
	skill := verifyRawSkill(raw)
	if skill == nil {
		return resolution{reason: ReasonIntegrityFailure, err: fmt.Sprintf("skill '%s' failed integrity verification and was withheld", safeKey(key))}
	}
	// The key check runs before the version check: an answer that is not the
	// requested skill makes its version moot.
	if skill.key != key {
		recordKeyMismatch(key, skill.key)
		return resolution{reason: ReasonIntegrityFailure, err: fmt.Sprintf("skill '%s' is not available: the store answered under key '%s'", safeKey(key), skill.key)}
	}
	if wanted != 0 && skill.version != wanted {
		recordVersionMismatch(key, wanted, skill.version)
		return resolution{reason: ReasonWrongVersion, err: fmt.Sprintf("skill '%s' version %d is not available (the store holds version %d)", safeKey(key), wanted, skill.version)}
	}
	return resolution{reason: ReasonOK, skill: skill}
}

// candidate is one raw object chosen from a listing, with the store's map key
// kept to attribute a failure when the object's own key is unusable.
type candidate struct {
	objectKey string
	raw       any
}

// newestByKey keeps the highest version of each key. Objects without a usable
// key and version are kept so verification reports them, unless their key
// resolved from another, well-formed version. Output is sorted for
// deterministic reports.
func newestByKey(objects map[string]any) []candidate {
	type held struct {
		c       candidate
		version int
	}
	best := map[string]held{}
	var unusable []candidate
	for _, objectKey := range slices.Sorted(maps.Keys(objects)) {
		raw := objects[objectKey]
		obj, _ := raw.(map[string]any)
		key, keyOK := obj["key"].(string)
		version, versionOK := versionOf(obj["version"])
		if obj == nil || !keyOK || !isValidSkillKey(key) || !versionOK {
			unusable = append(unusable, candidate{objectKey, raw})
			continue
		}
		if h, ok := best[key]; !ok || version > h.version {
			best[key] = held{candidate{objectKey, raw}, version}
		}
	}
	out := make([]candidate, 0, len(best)+len(unusable))
	for _, key := range slices.Sorted(maps.Keys(best)) {
		out = append(out, best[key].c)
	}
	for _, c := range unusable {
		obj, _ := c.raw.(map[string]any)
		if key, ok := obj["key"].(string); ok && isValidSkillKey(key) {
			if _, resolved := best[key]; resolved {
				continue
			}
		}
		out = append(out, c)
	}
	return out
}

func mapsKeys[V any](m map[string]V) func(func(string) bool) {
	return func(yield func(string) bool) {
		for k := range m {
			if !yield(k) {
				return
			}
		}
	}
}

// logWithholdingSummary logs one warning per batch in which content was
// withheld, so an empty result is not mistaken for "no skills".
func logWithholdingSummary(subject string, requested, resolved int) {
	withheld := requested - resolved
	if withheld <= 0 {
		return
	}
	if resolved == 0 {
		currentLogger().Warn(fmt.Sprintf("all %d %s were withheld and no skill content is available; every object failed verification", requested, subject))
		return
	}
	currentLogger().Warn(fmt.Sprintf("%d of %d %s were withheld; see the preceding errors for the per-skill reason", withheld, requested, subject))
}
