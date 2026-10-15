package skills

import (
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"log/slog"
	"slices"
	"strings"
	"testing"
)

// wire helpers build FDv2 event data.
func intent(code, id string) []byte {
	return fmt.Appendf(nil, `{"payloads":[{"id":%q,"target":1,"intentCode":%q}]}`, id, code)
}

func put(wireKey, content string) []byte {
	return putWith(wireKey, map[string]any{"content": content, "contentHash": sha(content), "name": "n"})
}

func putWith(wireKey string, envelope map[string]any) []byte {
	b, _ := json.Marshal(map[string]any{"kind": "skill", "key": wireKey, "version": 99, "object": envelope})
	return b
}

func del(wireKey string) []byte {
	return fmt.Appendf(nil, `{"kind":"skill","key":%q,"version":99}`, wireKey)
}

func transferred(selector string) []byte {
	return fmt.Appendf(nil, `{"state":%q,"version":1}`, selector)
}

type readerHarness struct {
	t   *testing.T
	set *objectSet
	r   *fdv2Reader
}

func newHarness(t *testing.T) *readerHarness {
	set := newObjectSet()
	return &readerHarness{t: t, set: set, r: newFDv2Reader(set)}
}

func (h *readerHarness) feed(name string, data []byte) transferOutcome { return h.r.handle(name, data) }

func (h *readerHarness) commitFull(id string, puts ...[]byte) transferOutcome {
	h.feed("server-intent", intent("xfer-full", id))
	for _, p := range puts {
		h.feed("put-object", p)
	}
	return h.feed("payload-transferred", transferred("(p:"+id+":1)"))
}

func (h *readerHarness) keys() []string {
	var out []string
	for k := range h.set.snapshot() {
		out = append(out, k)
	}
	slices.Sort(out)
	return out
}

func TestReaderCommitsOnlyAtPayloadTransferred(t *testing.T) {
	_, _ = setup(t)
	h := newHarness(t)
	h.commitFull("p", put("a:1", "a"))
	h.feed("server-intent", intent("xfer-full", "p"))
	h.feed("put-object", put("b:1", "b"))
	if !slices.Equal(h.keys(), []string{"a:1"}) {
		t.Fatalf("mid-transfer the store must serve the previous contents: %v", h.keys())
	}
	out := h.feed("payload-transferred", transferred("(p:p:2)"))
	if !out.committed || out.basis != "(p:p:2)" || !slices.Equal(h.keys(), []string{"b:1"}) {
		t.Fatalf("after commit: %v %+v", h.keys(), out)
	}
	// The change list: the put, plus the tombstone for what the full transfer dropped.
	if len(out.changes) != 2 {
		t.Fatalf("changes %v", out.changes)
	}
	tomb := out.changes[1].(map[string]any)
	if tomb["key"] != "a" || tomb["version"] != 1 || tomb["content"] != nil {
		t.Fatalf("tombstone %v", tomb)
	}
}

func TestReaderIntents(t *testing.T) {
	_, _ = setup(t)
	h := newHarness(t)
	h.commitFull("p", put("a:1", "a"), put("b:1", "b"))
	h.feed("server-intent", intent("xfer-changes", "p"))
	h.feed("put-object", put("c:1", "c"))
	h.feed("payload-transferred", transferred("(p:p:2)"))
	if !slices.Equal(h.keys(), []string{"a:1", "b:1", "c:1"}) {
		t.Fatalf("xfer-changes must amend: %v", h.keys())
	}
	out := h.feed("server-intent", intent("none", "p"))
	if !out.upToDate || !slices.Equal(h.keys(), []string{"a:1", "b:1", "c:1"}) {
		t.Fatalf("none must leave the set alone: %v %+v", h.keys(), out)
	}
	// After none the stream carries changes, and they apply.
	h.feed("put-object", put("d:1", "d"))
	if out := h.feed("payload-transferred", transferred("(p:p:3)")); !out.committed {
		t.Fatal("a put after none was not committed")
	}
	h.feed("server-intent", intent("none", "p"))
	h.feed("delete-object", del("a:1"))
	out = h.feed("payload-transferred", transferred("(p:p:4)"))
	if !out.committed || slices.Contains(h.keys(), "a:1") || h.r.diag.ObjectsRevoked != 1 || h.r.diag.ObjectsIgnored != 0 {
		t.Fatalf("a delete after none: %v %+v", h.keys(), h.r.diag)
	}
	if len(out.changes) != 1 || out.changes[0].(map[string]any)["key"] != "a" {
		t.Fatalf("tombstone not published: %v", out.changes)
	}
}

func TestReaderFullTransferRevocations(t *testing.T) {
	_, _ = setup(t)
	h := newHarness(t)
	h.commitFull("p", put("a:1", "a"), put("b:1", "b"))
	out := h.commitFull("p", put("a:1", "a"))
	if len(out.changes) != 2 || out.changes[1].(map[string]any)["key"] != "b" || h.r.diag.ObjectsRevoked != 1 {
		t.Fatalf("omitted skill: %v %+v", out.changes, h.r.diag)
	}
	out = h.commitFull("p")
	if len(out.changes) != 1 || out.changes[0].(map[string]any)["key"] != "a" || len(h.keys()) != 0 {
		t.Fatalf("empty full transfer: %v", out.changes)
	}
	// A version move: a put and a tombstone, and no revocation counted.
	h.commitFull("p", put("v:1", "one"))
	before := h.r.diag.ObjectsRevoked
	out = h.commitFull("p", put("v:2", "two"))
	if len(out.changes) != 2 || h.r.diag.ObjectsRevoked != before {
		t.Fatalf("version move: %v revoked=%d", out.changes, h.r.diag.ObjectsRevoked-before)
	}
	// xfer-changes mentioning nothing revokes nothing.
	h.feed("server-intent", intent("xfer-changes", "p"))
	out = h.feed("payload-transferred", transferred("(p:p:9)"))
	if len(out.changes) != 0 || !slices.Equal(h.keys(), []string{"v:2"}) {
		t.Fatalf("empty delta: %v", h.keys())
	}
	// A delete for a key never held: tombstone published, not counted.
	h.feed("server-intent", intent("xfer-changes", "p"))
	h.feed("delete-object", del("never:1"))
	out = h.feed("payload-transferred", transferred("(p:p:10)"))
	if len(out.changes) != 1 || h.r.diag.ObjectsRevoked != before {
		t.Fatalf("delete of an unheld key: %v %+v", out.changes, h.r.diag)
	}
}

func TestReaderForeignPayloadIsDeclined(t *testing.T) {
	_, logs := setup(t)
	h := newHarness(t)
	h.commitFull("skills", put("a:1", "a"))
	for range 3 {
		out := h.commitFull("flags")
		if out.committed || out.basis != "" {
			t.Fatalf("a foreign transfer committed: %+v", out)
		}
	}
	if !slices.Equal(h.keys(), []string{"a:1"}) || h.r.diag.PayloadsIgnored != 3 {
		t.Fatalf("foreign transfer altered the set: %v %+v", h.keys(), h.r.diag)
	}
	warned := 0
	for _, e := range logs.all() {
		if e.level == slog.LevelWarn && strings.Contains(e.msg, "another payload") {
			warned++
		}
	}
	if warned != 1 {
		t.Fatalf("warned %d times, want once", warned)
	}
}

func TestReaderTransfersThatApplyNothing(t *testing.T) {
	_, _ = setup(t)
	shapes := map[string]func(h *readerHarness){
		"none":          func(h *readerHarness) { h.feed("server-intent", intent("none", "skills")) },
		"unknown code":  func(h *readerHarness) { h.feed("server-intent", intent("xfer-sideways", "skills")) },
		"lone transfer": func(h *readerHarness) {},
		"foreign":       func(h *readerHarness) { h.feed("server-intent", intent("xfer-full", "flags")) },
	}
	for name, shape := range shapes {
		h := newHarness(t)
		h.commitFull("skills", put("a:1", "a"))
		before := h.r.diag.PayloadsTransferred
		shape(h)
		out := h.feed("payload-transferred", transferred("(p:other:7)"))
		if out.committed || out.upToDate || out.basis != "" || !slices.Equal(h.keys(), []string{"a:1"}) || h.r.diag.PayloadsTransferred != before+1 {
			t.Fatalf("%s: %+v keys=%v", name, out, h.keys())
		}
	}
}

func TestReaderErrorAndGoodbyeAbandonInFlight(t *testing.T) {
	_, _ = setup(t)
	for _, ev := range []struct{ name, data string }{{"error", `{"reason":"x"}`}, {"goodbye", `{"reason":"bye"}`}} {
		h := newHarness(t)
		h.commitFull("p", put("a:1", "a"))
		h.feed("server-intent", intent("xfer-full", "p"))
		h.feed("put-object", put("b:1", "b"))
		out := h.feed(ev.name, []byte(ev.data))
		if out.disconnect == "" {
			t.Fatalf("%s did not disconnect", ev.name)
		}
		if out := h.feed("payload-transferred", transferred("(p:p:2)")); out.committed {
			t.Fatalf("%s: the abandoned payload committed later", ev.name)
		}
		if !slices.Equal(h.keys(), []string{"a:1"}) {
			t.Fatalf("%s: committed contents lost: %v", ev.name, h.keys())
		}
	}
}

func TestReaderWireIdentity(t *testing.T) {
	_, _ = setup(t)
	h := newHarness(t)
	h.commitFull("p",
		putWith("pdf", map[string]any{"content": "x", "contentHash": sha("x")}),
		putWith("txt:latest", map[string]any{"content": "x", "contentHash": sha("x")}),
		putWith("emp:", map[string]any{"content": "x", "contentHash": sha("x")}),
		putWith("a:1:2", map[string]any{"content": "x", "contentHash": sha("x")}),
		putWith(":3", map[string]any{"content": "x", "contentHash": sha("x")}),
		putWith("lead:03", map[string]any{"content": "x", "contentHash": sha("x"), "version": 7}),
	)
	all := h.set.snapshot()
	if _, ok := all["pdf"]; !ok {
		t.Error("no delimiter: must be held version-less under its key")
	}
	if all["txt"].(map[string]any)["version"] != "latest" || all["emp"].(map[string]any)["version"] != "" || all["a"].(map[string]any)["version"] != "1:2" {
		t.Errorf("non-numeric versions must be carried through as text: %v", all)
	}
	if _, ok := all[""]; ok {
		t.Error("an empty key must be dropped")
	}
	if got := all["lead:3"]; got == nil || got.(map[string]any)["version"] != 3 {
		t.Errorf("leading zeros, and the version comes off the wire key, not a version field: %v", all)
	}
	// Verification reports the malformed ones under a recognisable key.
	_, logs := setup(t)
	SetSkillStore(&funcStore{get: func(_, key string, _ int) (any, error) { return h.set.get(key, 0), nil }})
	_, _ = GetSkill("pdf", 0)
	if recs := logs.records(); len(recs) != 1 || recs[0]["reason_code"] != "invalid_version" || recs[0]["skill_key"] != "pdf" {
		t.Fatalf("records %v", recs)
	}
}

// the envelope fields read are exactly contentType, content,
// contentHash, name and description; key and version come off the wire key.
// Anything else in the envelope is dropped, so a store cannot smuggle a field
// through to whatever reads the held object.
func TestReaderReadsExactlyTheEnvelopeFields(t *testing.T) {
	_, _ = setup(t)
	h := newHarness(t)
	h.commitFull("p", putWith("a:1", map[string]any{
		"contentType": "text/markdown", "content": "x", "contentHash": sha("x"), "name": "N", "description": "D",
		"extra": "smuggled", "key": "b", "version": 5,
	}))
	raw, _ := h.set.get("a", 1).(map[string]any)
	want := []string{"content", "contentHash", "contentType", "description", "key", "name", "version"}
	if got := sortedKeys(raw); !slices.Equal(got, want) {
		t.Fatalf("held object fields %v, want %v", got, want)
	}
	if raw["key"] != "a" || raw["version"] != 1 || raw["contentType"] != "text/markdown" || raw["name"] != "N" || raw["description"] != "D" {
		t.Fatalf("held object %v", raw)
	}
}

func TestReaderDoesNotVerifyAndCountsHashless(t *testing.T) {
	_, logs := setup(t)
	h := newHarness(t)
	h.commitFull("p",
		putWith("tampered:1", map[string]any{"content": "real", "contentHash": sha("other")}),
		putWith("hashless:1", map[string]any{"content": "real"}),
	)
	if len(h.keys()) != 2 || h.r.diag.HashlessObjects != 1 {
		t.Fatalf("held %v diag %+v", h.keys(), h.r.diag)
	}
	SetSkillStore(&funcStore{get: func(_, key string, v int) (any, error) { return h.set.get(key, v), nil }})
	for key, code := range map[string]string{"tampered": "hash_mismatch", "hashless": "missing_content_hash"} {
		if o, _ := GetSkillResult(key, 0); o.Reason() != ReasonIntegrityFailure {
			t.Fatalf("%s: %v", key, o.Reason())
		}
		found := false
		for _, r := range logs.records() {
			found = found || (r["skill_key"] == key && r["reason_code"] == code)
		}
		if !found {
			t.Fatalf("%s: no %s record", key, code)
		}
	}
}

func TestReaderLoneSurrogateIsNotLaundered(t *testing.T) {
	_, logs := setup(t)
	h := newHarness(t)
	// \ud800 with the hash a laundering decoder would produce (U+FFFD).
	body := `{"kind":"skill","key":"s:1","version":1,"object":{"content":"\` + `ud800","contentHash":"` + sha("\xef\xbf\xbd") + `"}}`
	h.feed("server-intent", intent("xfer-full", "p"))
	h.feed("put-object", []byte(body))
	h.feed("payload-transferred", transferred("(p:p:1)"))
	SetSkillStore(&funcStore{get: func(_, key string, v int) (any, error) { return h.set.get(key, v), nil }})
	if got, _ := GetSkill("s", 0); got != nil {
		t.Fatal("a lone surrogate was laundered into U+FFFD and verified")
	}
	if logs.records()[0]["reason_code"] != "not_utf8" {
		t.Fatalf("code %v", logs.records()[0]["reason_code"])
	}
	// A proper pair still decodes.
	if s, ok := decodeJSONStringVerbatim([]byte(`"\` + `ud83e\` + `uddea \` + `u00e9\n"`)); !ok || s != "🧪 é\n" {
		t.Fatalf("pair decode %q %v", s, ok)
	}
}

func TestReaderNonSkillObjectsAreSkippedAndCounted(t *testing.T) {
	_, _ = setup(t)
	h := newHarness(t)
	h.feed("server-intent", intent("xfer-full", "p"))
	h.feed("put-object", []byte(`{"kind":"flag","key":"f:1","version":1,"object":{}}`))
	h.feed("put-object", []byte(`{"kind":"Skill","key":"g:1","version":1,"object":{}}`))
	h.feed("put-object", []byte(`{"key":"nokind:1","version":1,"object":{}}`))
	h.feed("put-object", put("a:1", "a"))
	h.feed("payload-transferred", transferred("(p:p:1)"))
	if h.r.diag.ObjectsIgnored != 3 || !slices.Equal(h.keys(), []string{"a:1"}) || h.r.diag.SkillObjectsReceived != 1 {
		t.Fatalf("%v %+v", h.keys(), h.r.diag)
	}
}

// Experimental-stage rule, asserted both ways so changing it is deliberate.
func TestReaderKeylessObjectsAreIgnoredAndTheTransferCommits(t *testing.T) {
	_, logs := setup(t)
	h := newHarness(t)
	h.commitFull("p", put("a:1", "a"))
	h.feed("server-intent", intent("xfer-changes", "p"))
	h.feed("delete-object", []byte(`{"kind":"skill","version":1}`))
	if out := h.feed("payload-transferred", transferred("(p:p:2)")); !out.committed || !slices.Equal(h.keys(), []string{"a:1"}) {
		t.Fatalf("keyless delete: %v", h.keys())
	}
	h.feed("server-intent", intent("xfer-full", "p"))
	h.feed("put-object", []byte(`{"kind":"skill","version":1,"object":{"content":"a"}}`))
	if out := h.feed("payload-transferred", transferred("(p:p:3)")); !out.committed || len(h.keys()) != 0 {
		t.Fatalf("keyless put in a full transfer: %v", h.keys())
	}
	// Ignored, but not silently.
	if !logs.contains(slog.LevelWarn, "delete-object carried no usable key") || !logs.contains(slog.LevelWarn, "put-object carried no usable key") {
		t.Fatal("a keyless put or delete must be warned about")
	}
}

// Two payloads on one connection (a "flagging,agent-skill" declaration):
// only the first is read, and the reader warns once, not per intent.
func TestReaderWarnsOnceAboutMultiplePayloads(t *testing.T) {
	_, logs := setup(t)
	h := newHarness(t)
	two := []byte(`{"payloads":[{"id":"p","target":1,"intentCode":"xfer-full"},{"id":"q","target":1,"intentCode":"xfer-full"}]}`)
	for range 2 {
		h.feed("server-intent", two)
		h.feed("put-object", put("a:1", "a"))
		h.feed("payload-transferred", transferred("(p:p:1)"))
	}
	warned := 0
	for _, e := range logs.all() {
		if e.level == slog.LevelWarn && strings.Contains(e.msg, "more than one payload") {
			warned++
		}
	}
	if warned != 1 || !slices.Equal(h.keys(), []string{"a:1"}) {
		t.Fatalf("warned %d times (want 1); keys %v", warned, h.keys())
	}
}

func TestReaderHeartbeatAndUnknownEvents(t *testing.T) {
	_, _ = setup(t)
	h := newHarness(t)
	h.commitFull("p", put("a:1", "a"))
	for _, name := range []string{"heart-beat", "something-new"} {
		if out := h.feed(name, []byte("not json {")); out.disconnect != "" || out.committed {
			t.Fatalf("%s disturbed state: %+v", name, out)
		}
	}
	if !slices.Equal(h.keys(), []string{"a:1"}) {
		t.Fatal("state disturbed")
	}
}

// The three kind constants are separate literals, none derived from another.
func TestWireKindConstants(t *testing.T) {
	if fdv2PayloadKind != "agent-skill" || fdv2ObjectKind != "skill" || skillObjectKind != "skill" {
		t.Fatal("kind values")
	}
	fset := token.NewFileSet()
	for _, file := range []string{"fdv2_reader.go", "types.go"} {
		f, err := parser.ParseFile(fset, file, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		ast.Inspect(f, func(n ast.Node) bool {
			vs, ok := n.(*ast.ValueSpec)
			if !ok {
				return true
			}
			for i, name := range vs.Names {
				switch name.Name {
				case "fdv2PayloadKind", "fdv2ObjectKind", "skillObjectKind":
					if _, lit := vs.Values[i].(*ast.BasicLit); !lit {
						t.Errorf("%s is not its own literal", name.Name)
					}
				}
			}
			return true
		})
	}
}
