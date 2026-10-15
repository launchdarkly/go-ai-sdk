package skills

import (
	"maps"
)

// This file is the single place skills telemetry signals and integrity log
// records are built. The emitter is never called from anywhere else.

const language = "go"

const (
	signalIntegrityFailure = "AgentControl Skill Integrity Failure"
	signalMaterialized     = "AgentControl Skill Materialized"
	signalRevoked          = "AgentControl Skill Revoked Received"
)

// integrityFailureEvent is the stable identity of the local integrity-failure
// log record. Customers' SIEM rules match on it: a compatibility surface.
const integrityFailureEvent = "ld.skills.integrity_failure"

const actionWithheld = "withheld"

const (
	redactedKey     = "<invalid-key>"
	redactedHash    = "<not-a-sha256-digest>"
	redactedVersion = "<invalid-version>"
)

// integrityReasonCode is the closed ten-token reason_code vocabulary.
type integrityReasonCode string

const (
	codeNotAnObject        integrityReasonCode = "not_an_object"
	codeInvalidKey         integrityReasonCode = "invalid_key"
	codeInvalidVersion     integrityReasonCode = "invalid_version"
	codeMissingContent     integrityReasonCode = "missing_content"
	codeMissingContentHash integrityReasonCode = "missing_content_hash"
	codeNotUTF8            integrityReasonCode = "not_utf8"
	codeOverSizeCap        integrityReasonCode = "over_size_cap"
	codeHashMismatch       integrityReasonCode = "hash_mismatch"
	codeKeyMismatch        integrityReasonCode = "key_mismatch"
	codeVersionMismatch    integrityReasonCode = "version_mismatch"
)

var integrityReasonCodes = []integrityReasonCode{
	codeNotAnObject, codeInvalidKey, codeInvalidVersion, codeMissingContent, codeMissingContentHash,
	codeNotUTF8, codeOverSizeCap, codeHashMismatch, codeKeyMismatch, codeVersionMismatch,
}

func emit(signal string, properties map[string]any) {
	stateMu.Lock()
	e := emitter
	stateMu.Unlock()
	defer func() {
		if r := recover(); r != nil {
			currentLogger().Warn("skills telemetry emitter panicked; ignoring", "panic", r)
		}
	}()
	e.Record(signal, properties)
}

func safeKey(v any) string {
	if k, ok := v.(string); ok && isValidSkillKey(k) {
		return k
	}
	return redactedKey
}

// writeRecord logs the record twice over: the event name
// and sorted-key compact JSON in the message text, and the same mapping as a
// structured attribute.
func writeRecord(record map[string]any) {
	line, err := marshalSortedASCII(record, false)
	if err != nil {
		line = []byte("{}")
	}
	currentLogger().Error(integrityFailureEvent+" "+string(line), "ld_skills", record)
}

// integrityFailure holds what one verification failure knows. version,
// expectedHash and observedHash are optional; absent fields are omitted.
type integrityFailure struct {
	key          any
	reason       string
	code         integrityReasonCode
	version      any
	expectedHash any
	observedHash string
}

// recordIntegrityFailure writes the log record and emits the signal. The
// store-supplied key and expected hash are shape-checked and redacted first,
// so a hostile store cannot publish the skill body through either.
func recordIntegrityFailure(f integrityFailure) {
	properties := map[string]any{"skill_key": safeKey(f.key), "language": language}
	if v, ok := versionOf(f.version); ok {
		properties["version"] = v
	}
	if h, ok := f.expectedHash.(string); ok {
		if sha256HexRE.MatchString(h) {
			properties["expected_hash"] = h
		} else {
			properties["expected_hash"] = redactedHash
		}
	}
	if f.observedHash != "" {
		properties["observed_hash"] = f.observedHash
	}
	record := maps.Clone(properties)
	record["event"] = integrityFailureEvent
	record["action"] = actionWithheld
	record["reason"] = f.reason
	record["reason_code"] = string(f.code)
	writeRecord(record)
	emit(signalIntegrityFailure, properties)
}

// recordKeyMismatch is log record only, no signal: the usual cause is a broken
// store adapter, not tampering.
func recordKeyMismatch(requested, served string) {
	writeRecord(map[string]any{
		"action":      actionWithheld,
		"event":       integrityFailureEvent,
		"language":    language,
		"reason":      "the skill store answered under a different key than the one requested",
		"reason_code": string(codeKeyMismatch),
		"served_key":  safeKey(served),
		"skill_key":   safeKey(requested),
	})
}

// recordVersionMismatch is log record only, no signal. requested is the
// caller's own pin and is logged as given; served is shape-checked so it stays
// an integer in the JSON.
func recordVersionMismatch(key string, requested, served int) {
	var servedValue any = served
	if served < 1 {
		servedValue = redactedVersion
	}
	writeRecord(map[string]any{
		"action":         actionWithheld,
		"event":          integrityFailureEvent,
		"language":       language,
		"reason":         "the skill store answered with a different version than the one requested",
		"reason_code":    string(codeVersionMismatch),
		"served_version": servedValue,
		"skill_key":      safeKey(key),
		"version":        requested,
	})
}

// recordMaterialized carries no filesystem path: paths belong in the report.
func recordMaterialized(key string, contentBytes int, contentHash string, action ReconcileActionKind) {
	emit(signalMaterialized, map[string]any{
		"skill_key":        safeKey(key),
		"content_bytes":    contentBytes,
		"content_hash":     contentHash,
		"reconcile_action": string(action),
		"language":         language,
	})
}

// recordRevoked takes both fields from the manifest, which is untrusted.
func recordRevoked(key any, version any) {
	properties := map[string]any{"skill_key": safeKey(key), "removed_from_disk": true, "language": language}
	if v, ok := versionOf(version); ok {
		properties["version"] = v
	}
	emit(signalRevoked, properties)
}
