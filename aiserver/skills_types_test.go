package aiserver

import (
	"context"
	"encoding/json"
	"errors"
	"io/fs"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/launchdarkly/go-sdk-common/v3/ldvalue"
)

// immutability: Go has no runtime "assignment raises"; the equivalent
// guarantee is that no field is exported, so assignment does not compile.
func TestValueTypesAreImmutable(t *testing.T) {
	for _, v := range []any{SkillReference{}, Skill{}, SkillOutcome{}, ReconcileAction{}, ReconcileReport{}} {
		rt := reflect.TypeOf(v)
		for i := range rt.NumField() {
			if rt.Field(i).IsExported() {
				t.Errorf("%s.%s is exported, so the value is mutable", rt.Name(), rt.Field(i).Name)
			}
		}
	}
}

// Content is bytes, and neither the caller's buffer nor the
// returned slice can change the Skill's bytes.
func TestSkillContentIsAnOpaqueSnapshottedByteBuffer(t *testing.T) {
	if reflect.TypeOf(Skill{}.Content()) != reflect.TypeOf([]byte(nil)) {
		t.Fatal("Content is not []byte")
	}
	buf := []byte("original")
	s := NewSkill(SkillFields{Key: "a", Version: 1, Content: buf, ContentHash: sha("original")})
	buf[0] = 'X'
	got := s.Content()
	got[1] = 'Y'
	if string(s.Content()) != "original" {
		t.Fatalf("content changed through a shared buffer: %q", s.Content())
	}
}

// no reader over content exists. Skill's method set is
// exactly the documented accessors, so a frontmatter or decode method fails.
func TestSkillHasNoContentReader(t *testing.T) {
	var names []string
	rt := reflect.TypeOf(Skill{})
	for i := range rt.NumMethod() {
		names = append(names, rt.Method(i).Name)
	}
	want := []string{"Content", "ContentHash", "Description", "Key", "Name", "Version"}
	if !slices.Equal(names, want) {
		t.Fatalf("Skill methods = %v, want exactly %v", names, want)
	}
}

func TestReconcileReportOKAndErrors(t *testing.T) {
	clean := newReconcileReport([]ReconcileAction{
		{key: "a", action: ActionWritten}, {key: "b", action: ActionUpdated},
		{key: "c", action: ActionSkippedCurrent}, {key: "d", action: ActionRemoved},
	})
	mixed := newReconcileReport([]ReconcileAction{
		{key: "a", action: ActionWritten}, {key: "e1", action: ActionError}, {key: "b", action: ActionUpdated},
		{key: "c", action: ActionSkippedCurrent}, {key: "e2", action: ActionError}, {key: "d", action: ActionRemoved},
		{key: "f", action: ActionWritten},
	})
	if !clean.OK() || len(clean.Errors()) != 0 {
		t.Fatalf("clean report: ok=%v errors=%v", clean.OK(), clean.Errors())
	}
	if mixed.OK() {
		t.Fatal("report with errors reads ok")
	}
	errs := mixed.Errors()
	if len(errs) != 2 || errs[0].Key() != "e1" || errs[1].Key() != "e2" {
		t.Fatalf("errors = %v, want e1, e2 in order", errs)
	}
	for _, r := range []ReconcileReport{clean, mixed} {
		if r.OK() != (len(r.Errors()) == 0) {
			t.Fatal("OK and Errors disagree")
		}
	}
}

// the closed sets admit exactly their documented tokens.
func TestClosedSets(t *testing.T) {
	str := func(v any) []string {
		rv := reflect.ValueOf(v)
		out := []string{}
		for i := range rv.Len() {
			out = append(out, rv.Index(i).String())
		}
		return out
	}
	cases := []struct {
		got  []string
		want []string
	}{
		{str(skillOutcomeReasons), []string{"absent", "integrity_failure", "ok", "store_unavailable", "wrong_version"}},
		{str(reconcileActionKinds), []string{"written", "updated", "skipped_current", "removed", "error"}},
		{str(onUnavailableValues), []string{"keep", "raise"}},
		{str(integrityReasonCodes), []string{"not_an_object", "invalid_key", "invalid_version", "missing_content",
			"missing_content_hash", "not_utf8", "over_size_cap", "hash_mismatch", "key_mismatch", "version_mismatch"}},
	}
	for _, c := range cases {
		if !slices.Equal(c.got, c.want) {
			t.Errorf("closed set = %v, want %v", c.got, c.want)
		}
	}
	if !slices.IsSorted(str(skillOutcomeReasons)) {
		t.Error("SkillOutcomeReason tokens are not alphabetical")
	}
}

func TestSkillOutcomeInvariants(t *testing.T) {
	_, _ = setup(t)
	mem := NewInMemorySkillStore(map[string]any{"good": rawSkill("good", 1, "body")})
	bad := rawSkill("bad", 1, "body")
	bad["contentHash"] = sha("other")
	_ = mem.Put(bad)
	// A store error that names a filesystem path, as an os-backed store's would.
	pathErr := &fs.PathError{Op: "open", Path: "/srv/secret/down/SKILL.md", Err: fs.ErrPermission}
	SetSkillStore(&funcStore{get: func(kind, key string, version int) (any, error) {
		switch key {
		case "down":
			return nil, pathErr
		case "pinned": // ignores the pin: wrong_version
			return rawSkill("pinned", 3, "body"), nil
		}
		return mem.GetObject(kind, key, version)
	}})
	cases := []struct {
		key     string
		version int
		want    SkillOutcomeReason
	}{
		{"good", 0, ReasonOK},
		{"bad", 0, ReasonIntegrityFailure},
		{"missing", 0, ReasonAbsent},
		{"down", 0, ReasonStoreUnavailable},
		{"pinned", 2, ReasonWrongVersion},
	}
	for _, c := range cases {
		o, err := GetSkillResult(c.key, c.version)
		if err != nil || o.Reason() != c.want {
			t.Fatalf("%s: reason=%v err=%v", c.key, o.Reason(), err)
		}
		if (o.Skill() != nil) != (o.Reason() == ReasonOK) {
			t.Errorf("%s: skill populated iff ok violated", c.key)
		}
		if (o.Detail() != "") == (o.Reason() == ReasonOK) {
			t.Errorf("%s: detail set iff not ok violated: %q", c.key, o.Detail())
		}
		if o.Reason() != ReasonOK && (!strings.Contains(o.Detail(), c.key) || strings.Contains(o.Detail(), "body") ||
			strings.Contains(o.Detail(), "/srv") || strings.Contains(o.Detail(), "SKILL.md")) {
			t.Errorf("%s: detail must name the key and carry no content or path: %q", c.key, o.Detail())
		}
	}
}

// skillRefs.
func TestSkillRefs(t *testing.T) {
	rec, _ := setup(t)
	cfg := func(raw map[string]any) *AiConfig { return &AiConfig{Raw: raw} }

	if _, err := SkillRefs(nil); err == nil || !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("nil config: err=%v, want ErrInvalidArgument", err)
	}
	if _, err := SkillRefs(cfg(map[string]any{"skills": "nope"})); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("non-array skills: err=%v", err)
	}
	for name, raw := range map[string]map[string]any{"absent": {}, "empty": {"skills": []any{}}, "nil raw": nil} {
		refs, err := SkillRefs(cfg(raw))
		if err != nil || refs == nil || len(refs) != 0 {
			t.Fatalf("%s: refs=%v err=%v, want empty non-nil", name, refs, err)
		}
	}
	refs, err := SkillRefs(cfg(map[string]any{"skills": []any{
		map[string]any{"key": "a", "version": float64(1)},
		map[string]any{"key": "b", "version": float64(3)},
	}}))
	if err != nil || len(refs) != 2 || refs[0].Key() != "a" || refs[0].Version() != 1 || refs[1].Key() != "b" || refs[1].Version() != 3 {
		t.Fatalf("refs=%v err=%v", refs, err)
	}
	// One malformed entry rejects the whole field (as JS and Python): a
	// partial list passed to WriteSkills with pruning would delete files.
	for _, bad := range []any{"junk", map[string]any{"key": "Bad", "version": float64(1)}, map[string]any{"key": "c", "version": 2.5}} {
		refs, err = SkillRefs(cfg(map[string]any{"skills": []any{map[string]any{"key": "a", "version": float64(1)}, bad}}))
		if refs != nil || !errors.Is(err, ErrInvalidArgument) || !strings.Contains(err.Error(), "position 1") {
			t.Fatalf("%v: refs=%v err=%v, want the whole field rejected", bad, refs, err)
		}
	}
	if len(rec.all()) != 0 {
		t.Fatalf("SkillRefs recorded telemetry: %v", rec.all())
	}
}

// a config carrying skills rides through InspectConfig untouched, and
// SkillRefs needs no client or store.
func TestSkillRefsThroughInspectConfig(t *testing.T) {
	_, _ = setup(t)
	fake := &fakeSkillsLD{variations: map[string]any{"cfg": map[string]any{
		"_ldMeta":      map[string]any{"enabled": true, "variationKey": "v1"},
		"model":        map[string]any{"name": "m"},
		"provider":     map[string]any{"name": "p"},
		"instructions": "hi",
		"skills":       []any{map[string]any{"key": "pdf", "version": 2}},
	}}}
	if _, err := InitWithClient(context.Background(), fake, InitOptions{}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = Shutdown(context.Background()) })
	res := InspectConfig(context.Background(), "cfg", ldcontext.New("u"))
	if res.Config == nil || res.Config.Raw["skills"] == nil {
		t.Fatalf("skills did not ride through InspectConfig: %+v", res)
	}
	refs, err := SkillRefs(res.Config)
	if err != nil || len(refs) != 1 || refs[0].Key() != "pdf" || refs[0].Version() != 2 {
		t.Fatalf("refs=%v err=%v", refs, err)
	}
	failed := InspectConfig(context.Background(), "missing", ldcontext.New("u"))
	if _, err := SkillRefs(failed.Config); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("a failed InspectConfig must not yield an empty list: %v", err)
	}
}

// fakeSkillsLD implements LDClient and records tracks.
type fakeSkillsLD struct {
	variations map[string]any
	tracks     []string
}

func (f *fakeSkillsLD) JSONVariation(key string, _ ldcontext.Context, def ldvalue.Value) (ldvalue.Value, error) {
	v, ok := f.variations[key]
	if !ok {
		return def, nil
	}
	b, _ := json.Marshal(v)
	return ldvalue.Parse(b), nil
}
func (f *fakeSkillsLD) TrackMetric(event string, _ ldcontext.Context, _ float64, _ ldvalue.Value) error {
	f.tracks = append(f.tracks, event)
	return nil
}
func (f *fakeSkillsLD) Flush()       {}
func (f *fakeSkillsLD) Close() error { return nil }
