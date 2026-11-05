package skills

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"strings"
	"testing"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

func TestInMemoryStoreRoundTrip(t *testing.T) {
	s := NewInMemorySkillStore(map[string]any{"x": rawSkill("a", 1, "one")})
	if err := s.Put(rawSkill("b", 1, "two")); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"a", "b"} {
		got, _ := s.GetObject("skill", key, 0)
		if got.(map[string]any)["key"] != key {
			t.Fatalf("%s did not round-trip: %v", key, got)
		}
	}
	if got, _ := s.GetObject("skill", "nope", 0); got != nil {
		t.Fatalf("unknown key returned %v", got)
	}
	if err := s.Put(map[string]any{"version": 1}); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("Put without a key: %v", err)
	}
}

// Satisfied by construction (Go maps have no prototype chain); regression
// protection only.
func TestInMemoryStoreInheritedNames(t *testing.T) {
	empty := NewInMemorySkillStore(nil)
	if got, _ := empty.GetObject("skill", "constructor", 0); got != nil {
		t.Fatalf("empty store answered constructor with %v", got)
	}
	s := NewInMemorySkillStore(map[string]any{"a": rawSkill("a", 1, "x")})
	_ = s.Put(rawSkill("__proto__", 1, "y"))
	if got, _ := s.GetObject("skill", "a", 0); got.(map[string]any)["content"] != "x" {
		t.Fatal("__proto__ put changed another key")
	}
	if got, _ := s.GetObject("skill", "other", 0); got != nil {
		t.Fatal("__proto__ put made an absent key resolve")
	}
}

func TestInMemoryStoreVersions(t *testing.T) {
	s := NewInMemorySkillStore(nil)
	_ = s.Put(rawSkill("a", 1, "v1"))
	_ = s.Put(rawSkill("a", 3, "v3"))
	newest, _ := s.GetObject("skill", "a", 0)
	pinned, _ := s.GetObject("skill", "a", 1)
	other, _ := s.GetObject("skill", "a", 2)
	wrongKind, _ := s.GetObject("flag", "a", 0)
	if newest.(map[string]any)["version"] != 3 || pinned.(map[string]any)["version"] != 1 || other != nil || wrongKind != nil {
		t.Fatalf("newest=%v pinned=%v other=%v wrongKind=%v", newest, pinned, other, wrongKind)
	}
	all, _ := s.AllObjects("skill")
	if len(all) != 2 {
		t.Fatalf("AllObjects held %d entries, want one per (key, version)", len(all))
	}
	if all, _ := s.AllObjects("flag"); len(all) != 0 {
		t.Fatal("AllObjects for another kind returned objects")
	}
}

// pinFallthroughCases runs the two pin rules against a store factory.
func pinFallthroughCases(t *testing.T, build func(objs ...map[string]any) SkillStore) {
	t.Helper()
	malformed := map[string]any{"key": "m", "version": "latest", "content": "x", "contentHash": sha("x")}
	onlyMalformed := build(malformed)
	if got, _ := onlyMalformed.GetObject("skill", "m", 4); got == nil {
		t.Error("a key holding only a malformed object must answer a pin with it")
	}
	both := build(rawSkill("k", 2, "good"), map[string]any{"key": "k", "version": "junk", "content": "x", "contentHash": sha("x")})
	if got, _ := both.GetObject("skill", "k", 7); got != nil {
		t.Errorf("a pin missing well-formed versions must be a plain miss, got %v", got)
	}
}

func TestPinFallthroughInMemory(t *testing.T) {
	pinFallthroughCases(t, func(objs ...map[string]any) SkillStore {
		s := NewInMemorySkillStore(nil)
		for _, o := range objs {
			_ = s.Put(o)
		}
		return s
	})
}

func TestInMemoryStoreListeners(t *testing.T) {
	s := NewInMemorySkillStore(nil)
	var got []any
	l := NewListener(func(raw any) { got = append(got, raw) })
	if err := s.AddListener("skill", l); err != nil {
		t.Fatal(err)
	}
	if err := s.AddListener("flag", l); !errors.Is(err, ErrUnsupportedKind) {
		t.Fatalf("listener on another kind: %v", err)
	}
	raw := map[string]any{"key": "a", "content": 42} // unverified, malformed: passed verbatim
	_ = s.Put(raw)
	if len(got) != 1 || got[0].(map[string]any)["content"] != 42 {
		t.Fatalf("listener saw %v", got)
	}
	_ = s.AddListener("skill", l) // registered twice
	s.RemoveListener("skill", l)
	_ = s.Put(rawSkill("b", 1, "x"))
	if len(got) != 2 {
		t.Fatalf("after one removal of two registrations, want one call, have %d total", len(got))
	}
	s.RemoveListener("skill", l)
	s.RemoveListener("skill", l)                         // never-registered: no-op
	s.RemoveListener("skill", NewListener(func(any) {})) // never registered at all
	_ = s.Put(rawSkill("c", 1, "x"))
	if len(got) != 2 {
		t.Fatal("removed listener still fired")
	}
}

func TestStoreProbe(t *testing.T) {
	_, logs := setup(t)
	inner := NewInMemorySkillStore(nil)
	if !storeIsInitialized(inner) {
		t.Error("a store without the probe must be treated as initialized")
	}
	if storeIsInitialized(&probeStore{SkillStore: inner, err: errStoreDown}) {
		t.Error("a failing probe must count as not initialized")
	}
	if storeIsInitialized(&probeStore{SkillStore: inner, panic: true}) {
		t.Error("a panicking probe must count as not initialized")
	}
	if !logs.contains(slog.LevelWarn, "IsInitialized") {
		t.Error("a failing probe must log a warning")
	}
}

func TestSetSkillStoreLifecycle(t *testing.T) {
	_, _ = setup(t)
	fake := &fakeLD{}
	if _, err := aiserver.InitWithClient(context.Background(), fake, aiserver.InitOptions{}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = aiserver.Shutdown(context.Background()) })
	first := NewInMemorySkillStore(map[string]any{"a": rawSkill("a", 1, "first")})
	second := NewInMemorySkillStore(map[string]any{"a": rawSkill("a", 1, "second")})
	SetSkillStore(first)
	SetSkillStore(second)
	got, _ := GetSkill("a", 0)
	if got == nil || string(got.Content()) != "second" {
		t.Fatalf("accessors did not read the new store: %v", got)
	}
	if c, err := aiserver.GetClient(); err != nil || c != fake {
		t.Fatal("setting a store disturbed the client singleton")
	}
	SetSkillStore(nil)
	var typedNil *InMemorySkillStore
	SetSkillStore(typedNil)
	if got, _ := GetSkill("a", 0); got == nil {
		t.Fatal("a nil store cleared the configured one")
	}
	if err := aiserver.Shutdown(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err := GetSkill("a", 0); !errors.Is(err, ErrNoSkillStore) {
		t.Fatalf("Shutdown did not clear the store: %v", err)
	}
}

func TestNoStoreConfigured(t *testing.T) {
	_, _ = setup(t)
	_, e1 := GetSkill("a", 0)
	_, e2 := GetSkillResult("a", 0)
	_, e3 := GetSkills([]SkillKey{"a"})
	_, e4 := AllSkills()
	for i, err := range []error{e1, e2, e3, e4} {
		if !errors.Is(err, ErrNoSkillStore) {
			t.Fatalf("accessor %d: %v", i, err)
		}
		msg := err.Error()
		if !strings.Contains(msg, "no skill store is configured") || !strings.Contains(msg, "SetSkillStore") || !strings.Contains(msg, "InMemorySkillStore") {
			t.Fatalf("message does not name the remedy: %q", msg)
		}
	}
}

func TestClearStateClearsStoreAndEmitter(t *testing.T) {
	rec, _ := setup(t)
	SetSkillStore(NewInMemorySkillStore(nil))
	ClearState()
	if _, err := GetSkill("a", 0); !errors.Is(err, ErrNoSkillStore) {
		t.Fatal("ClearState left the store")
	}
	recordRevoked("a", 1)
	if len(rec.all()) != 0 {
		t.Fatal("ClearState left the emitter")
	}
}

func TestGetSkillAccessors(t *testing.T) {
	_, _ = setup(t)
	s := NewInMemorySkillStore(nil)
	_ = s.Put(rawSkill("a", 1, "v1"))
	_ = s.Put(rawSkill("a", 2, "v2 ✨"))
	_ = s.Put(rawSkill("b", 1, "bee"))
	SetSkillStore(s)

	got, _ := GetSkill("a", 0)
	if got == nil || got.Version() != 2 || string(got.Content()) != "v2 ✨" || got.ContentHash() != sha("v2 ✨") || got.Name() != "N a" || got.Description() != "D" {
		t.Fatalf("newest = %+v", got)
	}
	if got, _ := GetSkill("a", 1); got == nil || string(got.Content()) != "v1" {
		t.Fatalf("pinned v1 = %+v", got)
	}
	if got, _ := GetSkill("a", 5); got != nil {
		t.Fatal("a version the store does not hold returned a skill")
	}
	if got, _ := GetSkill("zzz", 0); got != nil {
		t.Fatal("missing key returned a skill")
	}
	batch, _ := GetSkills([]SkillRequest{SkillKey("b"), NewSkillReference("missing", 1), NewSkillReference("a", 1), SkillKey("a")})
	var keys []string
	for _, sk := range batch {
		keys = append(keys, fmt.Sprintf("%s@%d", sk.Key(), sk.Version()))
	}
	if !slices.Equal(keys, []string{"b@1", "a@1", "a@2"}) {
		t.Fatalf("batch = %v", keys)
	}
}

func TestGetSkillsOmitsIntegrityFailures(t *testing.T) {
	rec, _ := setup(t)
	bad := rawSkill("b", 1, "bee")
	bad["content"] = "tampered"
	SetSkillStore(NewInMemorySkillStore(map[string]any{"a": rawSkill("a", 1, "a"), "b": bad, "c": rawSkill("c", 1, "c")}))
	got, _ := GetSkills([]SkillKey{"a", "b", "c"})
	if len(got) != 2 || got[0].Key() != "a" || got[1].Key() != "c" {
		t.Fatalf("got %v", got)
	}
	if n := len(rec.named(signalIntegrityFailure)); n != 1 {
		t.Fatalf("integrity signals = %d, want 1", n)
	}
}

func TestAllSkills(t *testing.T) {
	rec, logs := setup(t)
	s := NewInMemorySkillStore(map[string]any{
		"a:1":   rawSkill("a", 1, "one"),
		"a:2":   rawSkill("a", 2, "two"),
		"other": rawSkill("b", 1, "b"), // listed under a disagreeing map key
		"junk":  map[string]any{"key": "Bad!", "version": 1, "content": "x", "contentHash": sha("x")},
	})
	SetSkillStore(s)
	got, _ := AllSkills()
	var ids []string
	for _, sk := range got {
		ids = append(ids, fmt.Sprintf("%s@%d", sk.Key(), sk.Version()))
	}
	if !slices.Equal(ids, []string{"a@2", "b@1"}) {
		t.Fatalf("AllSkills = %v", ids)
	}
	sigs := rec.named(signalIntegrityFailure)
	if len(sigs) != 1 || sigs[0].props["skill_key"] != redactedKey {
		t.Fatalf("want exactly one signal, for the unusable object: %v", sigs)
	}
	for _, r := range logs.records() {
		if r["reason_code"] == "key_mismatch" || r["reason_code"] == "version_mismatch" {
			t.Fatalf("a listing has no request to mismatch: %v", r)
		}
	}
}

func TestAllSkillsStoreErrorIsEmpty(t *testing.T) {
	_, _ = setup(t)
	SetSkillStore(&funcStore{all: func(string) (map[string]any, error) { return nil, errStoreDown }})
	got, err := AllSkills()
	if err != nil || len(got) != 0 {
		t.Fatalf("got %v %v", got, err)
	}
}

func TestGetSkillResultReasons(t *testing.T) {
	rec, logs := setup(t)
	tampered := rawSkill("t", 1, "x")
	tampered["content"] = "y"
	SetSkillStore(NewInMemorySkillStore(map[string]any{"ok": rawSkill("ok", 1, "fine"), "t": tampered}))

	ok, _ := GetSkillResult("ok", 0)
	direct, _ := GetSkill("ok", 0)
	if ok.Reason() != ReasonOK || string(ok.Skill().Content()) != string(direct.Content()) || ok.Skill().ContentHash() != direct.ContentHash() {
		t.Fatalf("ok outcome %+v", ok)
	}
	absent, _ := GetSkillResult("nope", 0)
	integrity, _ := GetSkillResult("t", 0)
	if absent.Reason() != ReasonAbsent || integrity.Reason() != ReasonIntegrityFailure || absent.Reason() == integrity.Reason() {
		t.Fatalf("absent=%v integrity=%v", absent.Reason(), integrity.Reason())
	}
	// Telemetry not doubled: one GetSkillResult on the tampered skill, one signal.
	if n := len(rec.named(signalIntegrityFailure)); n != 1 {
		t.Fatalf("signals = %d", n)
	}
	_, _ = GetSkill("t", 0)
	if n := len(rec.named(signalIntegrityFailure)); n != 2 {
		t.Fatalf("GetSkill and GetSkillResult must record the same count; total %d", n)
	}

	down := &funcStore{get: func(string, string, int) (any, error) { return nil, errStoreDown }}
	SetSkillStore(down)
	unavailable, _ := GetSkillResult("ok", 0)
	if unavailable.Reason() != ReasonStoreUnavailable {
		t.Fatalf("raising store: %v", unavailable.Reason())
	}
	panicking := &funcStore{get: func(string, string, int) (any, error) { panic("boom") }}
	SetSkillStore(panicking)
	if o, _ := GetSkillResult("ok", 0); o.Reason() != ReasonStoreUnavailable {
		t.Fatalf("panicking store: %v", o.Reason())
	}

	// wrong_version needs a hand-built store that ignores the pin.
	before := len(rec.all())
	liar := &funcStore{get: func(_, key string, _ int) (any, error) { return rawSkill(key, 3, "v3"), nil }}
	SetSkillStore(liar)
	wv, _ := GetSkillResult("ok", 2)
	if wv.Reason() != ReasonWrongVersion {
		t.Fatalf("lying store: %v", wv.Reason())
	}
	if got, _ := GetSkill("ok", 2); got != nil {
		t.Fatal("GetSkill returned a wrong-version skill")
	}
	if len(rec.all()) != before {
		t.Fatal("version_mismatch must not emit the product signal")
	}
	var vm []map[string]any
	for _, r := range logs.records() {
		if r["reason_code"] == "version_mismatch" {
			vm = append(vm, r)
		}
	}
	if len(vm) != 2 { // once through GetSkillResult, once through GetSkill
		t.Fatalf("version_mismatch records = %d", len(vm))
	}
	wantKeys := []string{"action", "event", "language", "reason", "reason_code", "served_version", "skill_key", "version"}
	if keys := sortedKeys(vm[0]); !slices.Equal(keys, wantKeys) {
		t.Fatalf("record keys %v, want %v", keys, wantKeys)
	}
	if vm[0]["version"] != 2 || vm[0]["served_version"] != 3 || vm[0]["action"] != "withheld" {
		t.Fatalf("record %v", vm[0])
	}
	if _, isInt := vm[0]["served_version"].(int); !isInt {
		t.Fatal("served_version must be an integer")
	}
}

func sortedKeys(m map[string]any) []string {
	var out []string
	for k := range m {
		out = append(out, k)
	}
	slices.Sort(out)
	return out
}

func TestKeyMismatch(t *testing.T) {
	rec, logs := setup(t)
	sub := &funcStore{get: func(string, string, int) (any, error) { return rawSkill("other", 1, "x"), nil }}
	SetSkillStore(sub)
	o, _ := GetSkillResult("wanted", 0)
	if o.Reason() != ReasonIntegrityFailure {
		t.Fatalf("substituting store: %v", o.Reason())
	}
	if len(rec.all()) != 0 {
		t.Fatal("key_mismatch must not emit the product signal")
	}
	recs := logs.records()
	if len(recs) != 1 || recs[0]["reason_code"] != "key_mismatch" || recs[0]["skill_key"] != "wanted" || recs[0]["served_key"] != "other" {
		t.Fatalf("records %v", recs)
	}
	want := []string{"action", "event", "language", "reason", "reason_code", "served_key", "skill_key"}
	if keys := sortedKeys(recs[0]); !slices.Equal(keys, want) {
		t.Fatalf("keys %v, want %v", keys, want)
	}
	// GetSkill is the path the record exists for: it has no outcome to say
	// why it returned nil, so the record is all an operator sees.
	if got, err := GetSkill("wanted", 0); got != nil || err != nil {
		t.Fatalf("GetSkill on a substituting store: %v %v", got, err)
	}
	if len(rec.all()) != 0 {
		t.Fatal("key_mismatch through GetSkill must not emit the product signal")
	}
	recs = logs.records()
	if len(recs) != 2 || recs[1]["reason_code"] != "key_mismatch" || recs[1]["skill_key"] != "wanted" || recs[1]["served_key"] != "other" {
		t.Fatalf("GetSkill must write exactly one key_mismatch record: %v", recs)
	}
	// Key and version both disagree: the key check wins.
	both := &funcStore{get: func(string, string, int) (any, error) { return rawSkill("other", 9, "x"), nil }}
	SetSkillStore(both)
	o, _ = GetSkillResult("wanted", 2)
	if o.Reason() != ReasonIntegrityFailure {
		t.Fatalf("double mismatch: %v", o.Reason())
	}
	if last := logs.records()[len(logs.records())-1]; last["reason_code"] != "key_mismatch" {
		t.Fatalf("double mismatch code %v", last["reason_code"])
	}
}

func TestPinnedNonObjectIsAbsentListedIsNotAnObject(t *testing.T) {
	for _, bad := range []any{"a string", []any{1, 2}, nil} {
		rec, logs := setup(t)
		SetSkillStore(&funcStore{
			get: func(string, string, int) (any, error) { return bad, nil },
			all: func(string) (map[string]any, error) { return map[string]any{"x": bad}, nil },
		})
		o, _ := GetSkillResult("x", 0)
		if o.Reason() != ReasonAbsent || len(rec.all()) != 0 || len(logs.records()) != 0 {
			t.Fatalf("pinned %v: reason=%v signals=%v", bad, o.Reason(), rec.all())
		}
		_, _ = AllSkills()
		sigs := rec.named(signalIntegrityFailure)
		recs := logs.records()
		if len(sigs) != 1 || len(recs) != 1 || recs[0]["reason_code"] != "not_an_object" {
			t.Fatalf("listed %v: signals=%v records=%v", bad, sigs, recs)
		}
	}
}

// One failure per reason code, through an accessor, asserting the code.
func TestEachFailureReportsItsOwnCode(t *testing.T) {
	big := strings.Repeat("a", maxSkillContentBytes+1)
	cases := []struct {
		code string
		raw  map[string]any
	}{
		{"invalid_key", map[string]any{"key": "Bad", "version": 1, "content": "x", "contentHash": sha("x")}},
		{"invalid_version", map[string]any{"key": "k", "version": 0, "content": "x", "contentHash": sha("x")}},
		{"missing_content", map[string]any{"key": "k", "version": 1, "contentHash": sha("x")}},
		{"missing_content_hash", map[string]any{"key": "k", "version": 1, "content": "x"}},
		{"not_utf8", map[string]any{"key": "k", "version": 1, "content": "\xed\xa0\x80", "contentHash": sha("\xed\xa0\x80")}},
		{"over_size_cap", map[string]any{"key": "k", "version": 1, "content": big, "contentHash": sha(big)}},
		{"hash_mismatch", map[string]any{"key": "k", "version": 1, "content": "x", "contentHash": sha("y")}},
		// Size precedes encoding: over-cap content that is also not UTF-8.
		{"over_size_cap", map[string]any{"key": "k", "version": 1, "content": big + "\xed\xa0\x80", "contentHash": sha("z")}},
	}
	for _, c := range cases {
		_, logs := setup(t)
		SetSkillStore(&funcStore{get: func(string, string, int) (any, error) { return c.raw, nil }})
		if o, _ := GetSkillResult("k", 0); o.Reason() != ReasonIntegrityFailure {
			t.Fatalf("%s: reason %v", c.code, o.Reason())
		}
		recs := logs.records()
		if len(recs) != 1 || recs[0]["reason_code"] != c.code {
			t.Fatalf("%s: records %v", c.code, recs)
		}
	}
	// not_an_object is reachable only on the listing path.
	_, logs := setup(t)
	SetSkillStore(&funcStore{all: func(string) (map[string]any, error) { return map[string]any{"x": 7}, nil }})
	_, _ = AllSkills()
	if recs := logs.records(); len(recs) != 1 || recs[0]["reason_code"] != "not_an_object" {
		t.Fatalf("not_an_object: %v", recs)
	}
}

func TestInvalidKeysAndVersions(t *testing.T) {
	_, _ = setup(t)
	keys := map[string]bool{
		strings.Repeat("a", 256): true, strings.Repeat("a", 257): false, "-lead": false, "Up": false,
		"a/b": false, "a\n": false, "": false, "ok-key-1": true,
	}
	for key, valid := range keys {
		SetSkillStore(&funcStore{get: func(string, string, int) (any, error) { return rawSkill(key, 1, "x"), nil }})
		got, _ := GetSkill(key, 0)
		if (got != nil) != valid {
			t.Errorf("key %q (len %d): got=%v want valid=%v", key, len(key), got != nil, valid)
		}
	}
	for _, v := range []any{0, -1, 2.5, nil, true, "3", float64(0)} {
		raw := rawSkill("k", 1, "x")
		if v == nil {
			delete(raw, "version")
		} else {
			raw["version"] = v
		}
		SetSkillStore(&funcStore{get: func(string, string, int) (any, error) { return raw, nil }})
		if got, _ := GetSkill("k", 0); got != nil {
			t.Errorf("version %#v accepted", v)
		}
	}
	for _, v := range []any{1, int64(4), float64(5)} {
		raw := rawSkill("k", 1, "x")
		raw["version"] = v
		SetSkillStore(&funcStore{get: func(string, string, int) (any, error) { return raw, nil }})
		if got, _ := GetSkill("k", 0); got == nil {
			t.Errorf("version %#v rejected", v)
		}
	}
}

func TestSizeCapBoundary(t *testing.T) {
	rec, logs := setup(t)
	atCap := strings.Repeat("b", maxSkillContentBytes)
	SetSkillStore(&funcStore{get: func(string, string, int) (any, error) { return rawSkill("k", 1, atCap), nil }})
	if got, _ := GetSkill("k", 0); got == nil {
		t.Fatal("content at exactly the cap was withheld")
	}
	over := atCap + "b"
	SetSkillStore(&funcStore{get: func(string, string, int) (any, error) { return rawSkill("k", 1, over), nil }})
	if got, _ := GetSkill("k", 0); got != nil {
		t.Fatal("content one byte over the cap was returned")
	}
	if len(rec.named(signalIntegrityFailure)) != 1 {
		t.Fatal("over-cap must record the integrity signal")
	}
	reason := logs.records()[0]["reason"].(string)
	if !strings.Contains(reason, fmt.Sprint(maxSkillContentBytes)) || !strings.Contains(reason, fmt.Sprint(maxSkillContentBytes+1)) {
		t.Fatalf("reason must name the cap and the observed size: %q", reason)
	}
}

func TestVerificationIsByteExact(t *testing.T) {
	_, _ = setup(t)
	emoji := "skill 🧪 ünïcödé"
	off := rawSkill("k", 1, emoji)
	off["content"] = emoji + "!"
	for raw, want := range map[*map[string]any]bool{ptr(rawSkill("k", 1, emoji)): true, &off: false} {
		r := *raw
		SetSkillStore(&funcStore{get: func(string, string, int) (any, error) { return r, nil }})
		if got, _ := GetSkill("k", 0); (got != nil) != want {
			t.Fatalf("byte-exact: got=%v want=%v", got != nil, want)
		}
	}
}

func ptr[T any](v T) *T { return &v }

// Go strings are bytes, so there is no encoder to be lossy. The equivalent of
// the lossy-encoder construction: hash exactly the bytes held (and, apart,
// the U+FFFD a laundering decoder would substitute) and the content must
// still be withheld, so the hash comparison is not what rejects it.
func TestNotUTF8IsWithheldEvenWithMatchingHash(t *testing.T) {
	for _, hash := range []string{sha("\xed\xa0\x80"), sha("�")} {
		_, logs := setup(t)
		raw := map[string]any{"key": "k", "version": 1, "content": "\xed\xa0\x80", "contentHash": hash}
		SetSkillStore(&funcStore{get: func(string, string, int) (any, error) { return raw, nil }})
		if got, _ := GetSkill("k", 0); got != nil {
			t.Fatal("content with no UTF-8 encoding was returned")
		}
		if logs.records()[0]["reason_code"] != "not_utf8" {
			t.Fatalf("code %v", logs.records()[0]["reason_code"])
		}
	}
}

func TestLocallyComputedHashReachesSkill(t *testing.T) {
	_, logs := setup(t)
	SetSkillStore(&funcStore{get: func(string, string, int) (any, error) { return rawSkill("k", 1, "body"), nil }})
	got, _ := GetSkill("k", 0)
	sum := sha256.Sum256([]byte("body"))
	if got.ContentHash() != hex.EncodeToString(sum[:]) || len(got.ContentHash()) != 64 || strings.ToLower(got.ContentHash()) != got.ContentHash() {
		t.Fatalf("hash %q", got.ContentHash())
	}
	for _, variant := range []string{strings.ToUpper(sha("body")), " " + sha("body"), sha("body") + "\n"} {
		raw := rawSkill("k", 1, "body")
		raw["contentHash"] = variant
		SetSkillStore(&funcStore{get: func(string, string, int) (any, error) { return raw, nil }})
		if got, _ := GetSkill("k", 0); got != nil {
			t.Fatalf("non-canonical hash %q was normalized and accepted", variant)
		}
	}
	for _, r := range logs.records() {
		if r["reason_code"] != "hash_mismatch" {
			t.Fatalf("code %v", r["reason_code"])
		}
	}
}

// the delivery transport answers pins exactly as the in-memory store.
func TestPinFallthroughFDv2(t *testing.T) {
	pinFallthroughCases(t, func(objs ...map[string]any) SkillStore {
		s, _ := NewLaunchDarklySkillStore("sdk-test", nil)
		r := s.reader
		r.handle("server-intent", []byte(`{"payloads":[{"id":"p","intentCode":"xfer-full"}]}`))
		for _, o := range objs {
			wireKey := fmt.Sprintf("%s:%v", o["key"], o["version"])
			body := fmt.Sprintf(`{"kind":"skill","key":%q,"version":1,"object":{"content":%q,"contentHash":%q}}`, wireKey, o["content"], o["contentHash"])
			r.handle("put-object", []byte(body))
		}
		r.handle("payload-transferred", []byte(`{"state":"(p:p:1)","version":1}`))
		return s
	})
}
