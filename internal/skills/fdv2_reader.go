package skills

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

// The FDv2 wire contract. Three related constants, none an
// alias of another: the payload kind and object kind are LaunchDarkly's wire
// values, skillObjectKind is the SDK's own seam string.
const (
	// fdv2PayloadKind is declared as ?kinds= on every request. Without it the
	// connection is served the flag payload and no skills.
	fdv2PayloadKind = "agent-skill"
	// fdv2ObjectKind is the kind on each skill object inside the payload.
	fdv2ObjectKind = "skill"
	// fdv2KeyDelimiter separates key from version in an object's wire key.
	fdv2KeyDelimiter = ":"
)

const (
	eventServerIntent       = "server-intent"
	eventPutObject          = "put-object"
	eventDeleteObject       = "delete-object"
	eventPayloadTransferred = "payload-transferred"
	eventHeartbeat          = "heart-beat"
	eventGoodbye            = "goodbye"
	eventError              = "error"
)

// knownEvents are the six events whose data the reader acts on. Only these
// are parsed, so a heart-beat or an unknown event can never fail a connection.
var knownEvents = map[string]bool{
	eventServerIntent: true, eventPutObject: true, eventDeleteObject: true,
	eventPayloadTransferred: true, eventGoodbye: true, eventError: true,
}

const (
	intentTransferFull    = "xfer-full"
	intentTransferChanges = "xfer-changes"
	intentNone            = "none"
)

// envelopeFields are copied verbatim, never coerced or defaulted: filling in
// a missing field would forge what verification checks.
var envelopeFields = []string{"contentType", "content", "contentHash", "name", "description"}

var payloadSelectorRE = regexp.MustCompile(`\(p:([^:()]+):\d+\)`)

// StoreDiagnostics are counters describing what the FDv2 transport has seen.
// They tell "this environment has no skills" apart from "every skill was
// withheld". The field list is exhaustive.
type StoreDiagnostics struct {
	// PayloadsTransferred counts payload-transferred events.
	PayloadsTransferred int
	// SkillObjectsReceived counts put-object events identified as skills.
	SkillObjectsReceived int
	// ObjectsIgnored counts objects skipped because they were not skills.
	ObjectsIgnored int
	// ObjectsRevoked counts skill keys that left the held set entirely.
	ObjectsRevoked int
	// PayloadsIgnored counts transfers declined because they completed a
	// payload other than the skill payload. Normally zero.
	PayloadsIgnored int
	// HashlessObjects counts skill objects with no contentHash. Nonzero means
	// skills are being withheld with missing_content_hash.
	HashlessObjects int
	// ConnectionFailures counts consecutive recoverable failures; reset by a
	// committed payload or a none intent.
	ConnectionFailures int
	// LastError is the most recent transport error, human-readable, or "".
	LastError string
}

// tombstone is a delete-object narrowed to what it revokes. version 0
// revokes every version of the key, including a version-less entry: erring
// that way never serves withdrawn content.
type tombstone struct {
	key     string
	version int
}

// transferOutcome is what one event did. Nothing in the reader does I/O.
type transferOutcome struct {
	committed  bool
	changes    []any
	basis      string
	disconnect string
	// upToDate is a none intent: a completed exchange that commits nothing.
	upToDate bool
	// recycled marks a goodbye: routine once the connection completed an
	// exchange.
	recycled bool
}

// fdv2Reader applies FDv2 events to the committed object set. It is pure:
// no sockets, goroutines or clock. Changes commit at payload-transferred,
// never half-applied; only the first payload intent is read, and the payload
// skills arrive on is learnt so a transfer of any other is declined.
type fdv2Reader struct {
	committed *objectSet
	intent    string
	pending   *objectSet
	changes   []any
	diag      StoreDiagnostics

	warnedHashless  map[string]bool
	intentPayloadID string
	skillPayloadID  string
	skillsInPayload int
	warnedMultiple  bool
	warnedForeign   bool
}

func newFDv2Reader(committed *objectSet) *fdv2Reader {
	return &fdv2Reader{committed: committed, warnedHashless: map[string]bool{}}
}

// handle routes one event. data is the event's JSON (already checked to be
// JSON for a known event). Unknown event names are ignored by contract.
func (r *fdv2Reader) handle(name string, data []byte) transferOutcome {
	var obj map[string]json.RawMessage
	if knownEvents[name] {
		_ = json.Unmarshal(data, &obj) // valid JSON that is not an object leaves obj nil
	}
	switch name {
	case eventServerIntent:
		return r.serverIntent(obj)
	case eventPutObject:
		return r.putObject(obj)
	case eventDeleteObject:
		return r.deleteObject(obj)
	case eventPayloadTransferred:
		return r.payloadTransferred(obj)
	case eventError:
		r.abandon()
		return transferOutcome{disconnect: fmt.Sprintf("server sent error: %s", jsonString(obj["reason"]))}
	case eventGoodbye:
		return r.goodbye(obj)
	case eventHeartbeat:
		return transferOutcome{}
	default:
		currentLogger().Debug("ignoring an unknown FDv2 event", "event", name)
		return transferOutcome{}
	}
}

func jsonString(raw json.RawMessage) string {
	var s string
	_ = json.Unmarshal(raw, &s)
	return s
}

func jsonBool(raw json.RawMessage) bool {
	var b bool
	_ = json.Unmarshal(raw, &b)
	return b
}

func (r *fdv2Reader) serverIntent(obj map[string]json.RawMessage) transferOutcome {
	var payloads []map[string]json.RawMessage
	_ = json.Unmarshal(obj["payloads"], &payloads)
	if len(payloads) == 0 {
		return transferOutcome{disconnect: "server-intent carried no payload description"}
	}
	if len(payloads) > 1 && !r.warnedMultiple {
		r.warnedMultiple = true
		currentLogger().Warn("an FDv2 server-intent described more than one payload; only the first is read, as the "+
			"protocol requires, and it is taken to be the payload skills arrive on", "payloads", len(payloads))
	}
	first := payloads[0]
	intent := jsonString(first["intentCode"])
	r.intent = intent
	r.intentPayloadID = jsonString(first["id"])
	r.changes = nil
	r.skillsInPayload = 0
	switch intent {
	case intentTransferFull:
		r.pending = newObjectSet() // built beside the live set, so an interrupted transfer keeps it
	case intentTransferChanges:
		r.pending = r.committed.clone()
	case intentNone:
		// Current, but not finished: later edits arrive on this connection
		// with no second intent, so expect changes.
		r.intent = intentTransferChanges
		r.pending = nil
		return transferOutcome{upToDate: true}
	default:
		currentLogger().Debug("ignoring an FDv2 server-intent with an unrecognised intentCode", "intent_code", intent)
		r.pending = nil
	}
	return transferOutcome{}
}

// target is the pending set a skill event applies to, or nil when the event
// is not a skill or the current intent carries no objects. An object with no
// preceding intent is treated as a delta.
func (r *fdv2Reader) target(obj map[string]json.RawMessage) *objectSet {
	if obj == nil || jsonString(obj["kind"]) != fdv2ObjectKind {
		r.diag.ObjectsIgnored++
		return nil
	}
	if r.pending == nil {
		if r.intent == "" {
			r.intent = intentTransferChanges
		}
		if r.intent != intentTransferFull && r.intent != intentTransferChanges {
			return nil
		}
		r.pending = r.committed.clone()
	}
	return r.pending
}

// wireIdentity splits <key>:<version>. Malformed versions are kept so
// verification reports them under a recognisable key: no delimiter means no
// version; non-digit text is carried as the version; an empty key is dropped.
// Leading zeros are accepted.
func wireIdentity(raw json.RawMessage) (key string, version any, hasVersion, ok bool) {
	wire := jsonString(raw)
	key, text, found := strings.Cut(wire, fdv2KeyDelimiter)
	if key == "" {
		return "", nil, false, false
	}
	if !found {
		return key, nil, false, true
	}
	if text != "" && strings.Trim(text, "0123456789") == "" {
		if n, err := strconv.Atoi(text); err == nil {
			return key, n, true, true
		}
	}
	return key, text, true, true
}

func (r *fdv2Reader) putObject(obj map[string]json.RawMessage) transferOutcome {
	t := r.target(obj)
	if t == nil {
		return transferOutcome{}
	}
	key, version, hasVersion, ok := wireIdentity(obj["key"])
	if !ok {
		currentLogger().Warn("an FDv2 skill put-object carried no usable key and was dropped")
		return transferOutcome{}
	}
	// The event's own version is the payload's version, never the skill's.
	raw := map[string]any{"key": key}
	if hasVersion {
		raw["version"] = version
	}
	var envelope map[string]json.RawMessage
	_ = json.Unmarshal(obj["object"], &envelope)
	for _, field := range envelopeFields {
		value, present := envelope[field]
		if !present {
			continue
		}
		if field == "content" {
			if s, ok := decodeJSONStringVerbatim(value); ok {
				raw[field] = s
				continue
			}
		}
		var v any
		_ = json.Unmarshal(value, &v)
		raw[field] = v
	}
	t.put(key, raw)
	r.changes = append(r.changes, raw)
	r.diag.SkillObjectsReceived++
	r.skillsInPayload++
	if _, ok := raw["contentHash"].(string); !ok {
		r.diag.HashlessObjects++
		id := fmt.Sprintf("%s:%v", key, raw["version"])
		if !r.warnedHashless[id] {
			r.warnedHashless[id] = true
			currentLogger().Error("a delivered skill arrived without a contentHash and will be withheld with "+
				"missing_content_hash; the SDK cannot work around this. Contact LaunchDarkly support if skills in "+
				"your environment arrive without one", "skill_key", safeKey(key))
		}
	}
	return transferOutcome{}
}

func (r *fdv2Reader) deleteObject(obj map[string]json.RawMessage) transferOutcome {
	t := r.target(obj)
	if t == nil {
		return transferOutcome{}
	}
	key, version, _, ok := wireIdentity(obj["key"])
	if !ok {
		currentLogger().Warn("an FDv2 skill delete-object carried no usable key and was ignored")
		return transferOutcome{}
	}
	ts := tombstone{key: key}
	if v, ok := versionOf(version); ok {
		ts.version = v
	}
	held := t.get(key, 0) != nil
	t.remove(ts)
	// Counted per key: only a key that is now gone was revoked.
	if held && t.get(key, 0) == nil {
		r.diag.ObjectsRevoked++
	}
	r.skillsInPayload++
	r.changes = append(r.changes, tombstoneChange(ts))
	return transferOutcome{}
}

func tombstoneChange(ts tombstone) map[string]any {
	var v any
	if ts.version != 0 {
		v = ts.version
	}
	return map[string]any{"key": ts.key, "version": v}
}

func (r *fdv2Reader) payloadTransferred(obj map[string]json.RawMessage) transferOutcome {
	state := jsonString(obj["state"])
	payloadID := r.intentPayloadID
	if payloadID == "" {
		if m := payloadSelectorRE.FindStringSubmatch(state); m != nil {
			payloadID = m[1]
		}
	}
	foreign := r.skillPayloadID != "" && payloadID != "" && payloadID != r.skillPayloadID
	applied := !foreign && r.pending != nil
	switch {
	case foreign:
		if !r.warnedForeign {
			r.warnedForeign = true
			currentLogger().Warn("an FDv2 transfer of another payload was not applied to the skills held; applying it "+
				"would have replaced them with whatever that payload carried", "payload", payloadID, "skill_payload", r.skillPayloadID)
		}
		r.diag.PayloadsIgnored++
		r.changes = nil
	case r.pending != nil:
		if r.intent == intentTransferFull {
			// A full transfer revokes by omission: diff before the swap.
			revoked, keysGone := revocationsBetween(r.committed, r.pending)
			r.changes = append(r.changes, revoked...)
			r.diag.ObjectsRevoked += keysGone
		}
		*r.committed = *r.pending
		if r.skillsInPayload > 0 && payloadID != "" {
			r.skillPayloadID = payloadID
		}
	}
	changes := r.changes
	r.abandon()
	r.diag.PayloadsTransferred++
	if !applied {
		// Not a commit, not up to date, and no selector adopted: none of
		// these shapes applied anything.
		return transferOutcome{}
	}
	return transferOutcome{committed: true, changes: changes, basis: state}
}

// revocationsBetween is a tombstone for every (key, version) the incoming set
// no longer holds, plus how many keys left entirely (not version moves).
func revocationsBetween(current, incoming *objectSet) ([]any, int) {
	var out []any
	gone := map[string]bool{}
	for key, held := range current.versions {
		for v := range held {
			if incoming.versions[key][v] == nil {
				out = append(out, tombstoneChange(tombstone{key: key, version: v}))
				if incoming.get(key, 0) == nil {
					gone[key] = true
				}
			}
		}
	}
	for key := range current.loose {
		if _, ok := incoming.loose[key]; !ok {
			out = append(out, map[string]any{"key": key, "version": nil})
			if incoming.get(key, 0) == nil {
				gone[key] = true
			}
		}
	}
	return out, len(gone)
}

func (r *fdv2Reader) goodbye(obj map[string]json.RawMessage) transferOutcome {
	reason := jsonString(obj["reason"])
	r.abandon()
	if jsonBool(obj["catastrophe"]) {
		// Recoverable and counted, logged at ERROR even when silent: the one
		// goodbye that reports a server-side failure.
		currentLogger().Error("the FDv2 server reported a catastrophic failure; reconnecting with backoff", "reason", reason)
		return transferOutcome{disconnect: "server sent a catastrophic goodbye: " + reason}
	}
	if !jsonBool(obj["silent"]) {
		currentLogger().Debug("FDv2 connection closing", "reason", reason)
	}
	return transferOutcome{disconnect: "server said goodbye: " + reason, recycled: true}
}

// abandon drops the in-flight payload and keeps what is committed.
func (r *fdv2Reader) abandon() {
	r.pending = nil
	r.intent = ""
	r.intentPayloadID = ""
	r.skillsInPayload = 0
	r.changes = nil
}

// remove drops what ts revokes.
func (s *objectSet) remove(ts tombstone) {
	if ts.version == 0 {
		delete(s.versions, ts.key)
		delete(s.loose, ts.key)
		return
	}
	if held := s.versions[ts.key]; held != nil {
		delete(held, ts.version)
		if len(held) == 0 {
			delete(s.versions, ts.key)
		}
	}
}
