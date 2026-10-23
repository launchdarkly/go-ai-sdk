//go:build unix

package aiserver

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func tamperedRaw(key string) map[string]any {
	raw := rawSkill(key, 1, "the real body")
	raw["content"] = "the tampered body"
	return raw
}

func TestIntegritySignalAndRecordShape(t *testing.T) {
	rec, logs := setup(t)
	SetSkillStore(NewInMemorySkillStore(map[string]any{"t": tamperedRaw("t")}))
	if got, _ := GetSkill("t", 0); got != nil {
		t.Fatal("tampered skill returned")
	}
	sigs := rec.named(signalIntegrityFailure)
	if len(sigs) != 1 {
		t.Fatalf("signals %v", sigs)
	}
	wantSignal := []string{"expected_hash", "language", "observed_hash", "skill_key", "version"}
	if keys := sortedKeys(sigs[0].props); !slices.Equal(keys, wantSignal) {
		t.Fatalf("signal keys %v, want %v", keys, wantSignal)
	}
	if sigs[0].props["language"] != "go" {
		t.Fatal("language must be go")
	}
	entries := logs.all()
	var entry logEntry
	for _, e := range entries {
		if strings.HasPrefix(e.msg, "ld.skills.integrity_failure ") {
			entry = e
		}
	}
	if entry.msg == "" {
		t.Fatal("no integrity record in the message text")
	}
	text := strings.TrimPrefix(entry.msg, "ld.skills.integrity_failure ")
	var fromText map[string]any
	if err := json.Unmarshal([]byte(text), &fromText); err != nil {
		t.Fatalf("message text is not JSON: %v", err)
	}
	structured := entry.attrs["ld_skills"].(map[string]any)
	if len(fromText) != len(structured) {
		t.Fatalf("text %v vs structured %v", fromText, structured)
	}
	for k, v := range structured {
		if fmt.Sprint(fromText[k]) != fmt.Sprint(v) {
			t.Fatalf("text and structured disagree on %s", k)
		}
	}
	// Sorted keys, compact.
	var order []string
	dec := json.NewDecoder(strings.NewReader(text))
	_, _ = dec.Token()
	for dec.More() {
		k, _ := dec.Token()
		order = append(order, k.(string))
		var skip any
		_ = dec.Decode(&skip)
	}
	if !slices.IsSorted(order) || strings.Contains(text, ": ") {
		t.Fatalf("record JSON not sorted/compact: %s", text)
	}
	// The record is the signal plus exactly four more fields, values shared.
	for k, v := range sigs[0].props {
		if fmt.Sprint(structured[k]) != fmt.Sprint(v) {
			t.Fatalf("record and signal disagree on %s", k)
		}
	}
	extra := map[string]bool{}
	for k := range structured {
		if _, in := sigs[0].props[k]; !in {
			extra[k] = true
		}
	}
	if !maps.Equal(extra, map[string]bool{"event": true, "action": true, "reason": true, "reason_code": true}) {
		t.Fatalf("record extras %v", extra)
	}
	for _, k := range []string{"event", "action", "reason", "reason_code"} {
		if _, leaked := sigs[0].props[k]; leaked {
			t.Fatalf("record-only field %s leaked into the signal", k)
		}
	}
	if structured["event"] != "ld.skills.integrity_failure" || structured["action"] != "withheld" {
		t.Fatalf("record %v", structured)
	}
	for k, v := range structured {
		if v == nil {
			t.Fatalf("absent field %s recorded as null", k)
		}
	}
}

func TestRecordEscapesNonASCIIForCrossSDKParity(t *testing.T) {
	line, err := marshalSortedASCII(map[string]any{"b": "é 🧪 <&>", "a": 1}, false)
	if err != nil {
		t.Fatal(err)
	}
	if string(line) != `{"a":1,"b":"\u00e9 \ud83e\uddea <&>"}` {
		t.Fatalf("got %s", line)
	}
}

// Redaction is exercised, not bypassed: the hash and the key each carry the body.
func TestStoreSuppliedFieldsAreRedacted(t *testing.T) {
	body := "SECRET SKILL BODY " + strings.Repeat("x", 80)
	hashIsBody := map[string]any{"key": "k", "version": 1, "content": body, "contentHash": body}
	keyIsBody := map[string]any{"key": body, "version": 1, "content": body, "contentHash": sha(body)}
	for name, raw := range map[string]map[string]any{"hash": hashIsBody, "key": keyIsBody} {
		rec, logs := setup(t)
		SetSkillStore(&funcStore{
			get: func(string, string, int) (any, error) { return raw, nil },
			all: func(string) (map[string]any, error) { return map[string]any{"x": raw}, nil },
		})
		_, _ = GetSkill("k", 0)
		_, _ = AllSkills()
		if len(rec.all()) == 0 {
			t.Fatalf("%s: no signal recorded, so redaction was not exercised", name)
		}
		for _, s := range rec.all() {
			for k, v := range s.props {
				if strings.Contains(fmt.Sprint(v), "SECRET") {
					t.Fatalf("%s: signal property %s carries the body", name, k)
				}
			}
		}
		for _, r := range logs.records() {
			for k, v := range r {
				if strings.Contains(fmt.Sprint(v), "SECRET") {
					t.Fatalf("%s: record field %s carries the body", name, k)
				}
			}
		}
	}
}

// One defect caught at the accessor and before a write records the same keys.
func TestSignalShapeIndependentOfLayer(t *testing.T) {
	rec, _ := setup(t)
	over := strings.Repeat("o", maxSkillContentBytes+1)
	SetSkillStore(&funcStore{get: func(string, string, int) (any, error) { return rawSkill("big", 1, over), nil }})
	_, _ = GetSkill("big", 0)
	_, err := WriteSkills([]Skill{skillOf("big", 1, over)}, t.TempDir(), nil)
	if err != nil {
		t.Fatal(err)
	}
	sigs := rec.named(signalIntegrityFailure)
	if len(sigs) != 2 {
		t.Fatalf("signals %d", len(sigs))
	}
	a, b := sortedKeys(sigs[0].props), sortedKeys(sigs[1].props)
	if !slices.Equal(a, b) || !slices.Contains(a, "expected_hash") {
		t.Fatalf("accessor keys %v vs write keys %v", a, b)
	}
}

func TestMaterializedRevokedAndAllowlist(t *testing.T) {
	rec, logs := setup(t)
	fake := &fakeSkillsLD{}
	if _, err := InitWithClient(context.Background(), fake, InitOptions{}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = Shutdown(context.Background()) })
	fake.tracks = nil
	root := t.TempDir()
	store := NewInMemorySkillStore(map[string]any{"a": rawSkill("a", 1, "a1"), "b": rawSkill("b", 1, "b1"), "t": tamperedRaw("t")})
	SetSkillStore(store)
	_, _ = GetSkill("a", 0)
	_, _ = GetSkill("t", 0)
	_, _ = GetSkills([]SkillKey{"a", "b"})
	_, _ = AllSkills()
	_, _ = SkillRefs(&AiConfig{Raw: map[string]any{"skills": []any{}}})
	if _, err := WriteSkills([]SkillKey{"a", "b"}, root, nil); err != nil {
		t.Fatal(err)
	}
	_ = store.Put(rawSkill("a", 2, "a2"))
	if _, err := WriteSkills([]SkillKey{"a", "b"}, root, nil); err != nil { // a updated, b skipped_current
		t.Fatal(err)
	}
	report, err := WriteSkills([]SkillReference{NewSkillReference("a", 2)}, root, nil)
	if err != nil {
		t.Fatal(err)
	}
	kinds := map[ReconcileActionKind]int{}
	for _, a := range report.Actions() {
		kinds[a.Action()]++
	}
	if kinds[ActionSkippedCurrent] != 1 || kinds[ActionRemoved] != 1 {
		t.Fatalf("third run %v", report.Actions())
	}
	mats := rec.named(signalMaterialized)
	var actions []string
	for _, m := range mats {
		actions = append(actions, m.props["reconcile_action"].(string))
		want := []string{"content_bytes", "content_hash", "language", "reconcile_action", "skill_key"}
		if keys := sortedKeys(m.props); !slices.Equal(keys, want) {
			t.Fatalf("materialized keys %v", keys)
		}
		for _, v := range m.props {
			if s, ok := v.(string); ok && (strings.Contains(s, "/") || strings.Contains(s, root)) {
				t.Fatalf("a path reached telemetry: %v", m.props)
			}
		}
	}
	if !slices.Equal(actions, []string{"written", "written", "updated", "skipped_current", "skipped_current"}) {
		t.Fatalf("materialized actions %v", actions)
	}
	revoked := rec.named(signalRevoked)
	if len(revoked) != 1 || revoked[0].props["removed_from_disk"] != true || revoked[0].props["skill_key"] != "b" || revoked[0].props["version"] != 1 {
		t.Fatalf("revoked %v", revoked)
	}
	allowed := []string{signalIntegrityFailure, signalMaterialized, signalRevoked}
	for _, s := range rec.all() {
		if !slices.Contains(allowed, s.name) {
			t.Fatalf("signal outside the allowlist: %s", s.name)
		}
		if s.name == "AgentControl Skill SDK Reference Returned" || s.name == "AgentControl Skill Content Retrieved" {
			t.Fatalf("removed signal emitted: %s", s.name)
		}
		for _, v := range s.props {
			for _, body := range []string{"a1", "a2", "b1", "the real body", "the tampered body"} {
				if v == body {
					t.Fatalf("skill body in telemetry: %v", s.props)
				}
			}
		}
	}
	if len(fake.tracks) != 0 {
		t.Fatalf("skills operations called client.track: %v", fake.tracks)
	}
	_ = logs
}

func TestRevokedVersionAbsentForMalformedManifestEntry(t *testing.T) {
	rec, _ := setup(t)
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "old"), 0o755); err != nil {
		t.Fatal(err)
	}
	_ = os.WriteFile(filepath.Join(root, "old", "SKILL.md"), []byte("x"), 0o644)
	writeManifestFile(t, root, map[string]any{"old/SKILL.md": map[string]any{"key": "old", "version": "nope", "sha256": sha("x")}})
	SetSkillStore(NewInMemorySkillStore(nil))
	report, err := WriteSkills([]SkillKey{}, root, nil)
	if err != nil || !report.OK() {
		t.Fatalf("%v %v", report.Actions(), err)
	}
	revoked := rec.named(signalRevoked)
	if len(revoked) != 1 {
		t.Fatalf("revoked %v", revoked)
	}
	if _, has := revoked[0].props["version"]; has {
		t.Fatal("an unusable manifest version must be omitted, not recorded")
	}
}

type panickingEmitter struct{}

func (panickingEmitter) Record(string, map[string]any) { panic("emitter exploded") }

func TestThrowingAndDefaultEmitterNeverBreakOperations(t *testing.T) {
	for name, e := range map[string]telemetryEmitter{"panicking": panickingEmitter{}, "default": nil} {
		_, _ = setup(t)
		setSkillEmitterForTesting(e)
		SetSkillStore(NewInMemorySkillStore(map[string]any{"t": tamperedRaw("t"), "a": rawSkill("a", 1, "a")}))
		if got, _ := GetSkill("t", 0); got != nil {
			t.Fatalf("%s: tampered skill returned", name)
		}
		report, err := WriteSkills([]SkillKey{"a", "t"}, t.TempDir(), nil)
		if err != nil || len(report.Actions()) != 2 || report.Actions()[0].Action() != ActionWritten || report.Actions()[1].Action() != ActionError {
			t.Fatalf("%s: report %v err %v", name, report.Actions(), err)
		}
	}
}

// the panic is caught and logged at WARN, never propagated or silent.
func TestPanickingEmitterIsLoggedAtWarn(t *testing.T) {
	_, logs := setup(t)
	setSkillEmitterForTesting(panickingEmitter{})
	SetSkillStore(NewInMemorySkillStore(map[string]any{"t": tamperedRaw("t")}))
	if got, _ := GetSkill("t", 0); got != nil {
		t.Fatal("tampered skill returned")
	}
	if !logs.contains(slog.LevelWarn, "emitter panicked") {
		t.Fatalf("no WARN naming the emitter panic: %v", logs.all())
	}
}

// the revoked signal's skill_key comes from the manifest, which anyone
// with write access to the root can author, so it is shape-checked like the
// wire object's.
func TestRevokedSignalRedactsAManifestKeyThatIsTheBody(t *testing.T) {
	body := "SECRET SKILL BODY\n---\nname: x\n"
	rec, _ := setup(t)
	root := t.TempDir()
	writeManifestFile(t, root, map[string]any{body + "/SKILL.md": map[string]any{"key": body, "version": 1, "sha256": sha("x")}})
	SetSkillStore(NewInMemorySkillStore(nil))
	if _, err := WriteSkills([]SkillKey{}, root, nil); err != nil {
		t.Fatal(err)
	}
	// The emitter boundary itself: a revocation recorded under that key.
	recordRevoked(body, 1)
	revoked := rec.named(signalRevoked)
	if len(revoked) == 0 {
		t.Fatal("no revoked signal recorded, so redaction was not exercised")
	}
	for _, s := range rec.all() {
		for k, v := range s.props {
			if strings.Contains(fmt.Sprint(v), "SECRET") {
				t.Fatalf("%s property %s carries the body", s.name, k)
			}
		}
	}
	if got := revoked[len(revoked)-1].props["skill_key"]; got != redactedKey {
		t.Fatalf("skill_key %v, want the fixed placeholder", got)
	}
}

func writeManifestFile(t *testing.T, root string, entries map[string]any) {
	t.Helper()
	b, _ := json.Marshal(map[string]any{"manifestVersion": 1, "entries": entries})
	if err := os.WriteFile(filepath.Join(root, ".launchdarkly-skills.json"), b, 0o644); err != nil {
		t.Fatal(err)
	}
}
