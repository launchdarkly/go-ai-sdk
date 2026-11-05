//go:build unix

package skills

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"golang.org/x/sys/unix"
)

func readFile(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func readManifest(t *testing.T, root string) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal([]byte(readFile(t, filepath.Join(root, ".launchdarkly-skills.json"))), &m); err != nil {
		t.Fatalf("manifest is not valid JSON: %v", err)
	}
	return m
}

func manifestEntries(t *testing.T, root string) map[string]any {
	t.Helper()
	return readManifest(t, root)["entries"].(map[string]any)
}

func actionKinds(r ReconcileReport) []string {
	var out []string
	for _, a := range r.Actions() {
		out = append(out, a.Key()+"="+string(a.Action()))
	}
	return out
}

func mustWrite[I SkillInput](t *testing.T, in []I, root string, opts *WriteSkillsOptions) ReconcileReport {
	t.Helper()
	r, err := WriteSkills(in, root, opts)
	if err != nil {
		t.Fatal(err)
	}
	return r
}

// renameCall is one intercepted rename, with the directory identity resolved
// inside the call (the descriptor is closed as soon as the write returns).
type renameCall struct {
	src, dst         string
	srcFd, dstFd     int
	dev, ino         uint64
	sameDirectoryFds bool
}

// hookRename installs a rename spy for SKILL.md destinations; before runs
// first and may return an error to inject.
func hookRename(t *testing.T, before func(c renameCall) error) *[]renameCall {
	t.Helper()
	var (
		mu    sync.Mutex
		calls []renameCall
	)
	real := fsOps.rename
	fsOps.rename = func(oldFd int, oldName string, newFd int, newName string) error {
		if !strings.HasSuffix(newName, "SKILL.md") {
			return real(oldFd, oldName, newFd, newName)
		}
		var st unix.Stat_t
		_ = unix.Fstat(newFd, &st)
		c := renameCall{src: oldName, dst: newName, srcFd: oldFd, dstFd: newFd, dev: uint64(st.Dev), ino: uint64(st.Ino), sameDirectoryFds: oldFd == newFd}
		mu.Lock()
		calls = append(calls, c)
		mu.Unlock()
		if before != nil {
			if err := before(c); err != nil {
				return err
			}
		}
		return real(oldFd, oldName, newFd, newName)
	}
	t.Cleanup(func() { fsOps.rename = real })
	return &calls
}

func inode(t *testing.T, path string) (uint64, uint64) {
	t.Helper()
	var st unix.Stat_t
	if err := unix.Lstat(path, &st); err != nil {
		t.Fatal(err)
	}
	return uint64(st.Dev), uint64(st.Ino)
}

func TestWriteNewSkill(t *testing.T) {
	_, _ = setup(t)
	root := t.TempDir()
	r := mustWrite(t, []Skill{skillOf("pdf", 3, "# PDF\nbody ✨")}, root, nil)
	a := r.Actions()
	real, _ := filepath.EvalSymlinks(root)
	if len(a) != 1 || a[0].Action() != ActionWritten || a[0].Key() != "pdf" || a[0].Version() != 3 || a[0].Path() != filepath.Join(real, "pdf", "SKILL.md") || !r.OK() {
		t.Fatalf("actions %+v", a)
	}
	if readFile(t, filepath.Join(root, "pdf", "SKILL.md")) != "# PDF\nbody ✨" {
		t.Fatal("content not verbatim")
	}
}

func TestReferencesResolveThroughStore(t *testing.T) {
	_, _ = setup(t)
	root := t.TempDir()
	s := NewInMemorySkillStore(map[string]any{"a": rawSkill("a", 1, "aa"), "b": rawSkill("b", 2, "bb")})
	SetSkillStore(s)
	r := mustWrite(t, []SkillInput{SkillKey("a"), NewSkillReference("b", 2)}, root, nil)
	if !slices.Equal(actionKinds(r), []string{"a=written", "b=written"}) || readFile(t, filepath.Join(root, "b", "SKILL.md")) != "bb" {
		t.Fatalf("%v", actionKinds(r))
	}
	// Round trip: a Skill from the accessors verifies identically on write.
	got, _ := GetSkill("a", 0)
	other := t.TempDir()
	if r := mustWrite(t, []*Skill{got}, other, nil); !r.OK() || r.Actions()[0].Action() != ActionWritten {
		t.Fatalf("round-tripped skill: %v", actionKinds(r))
	}
}

func TestWriteAllCollapsesToNewestVersion(t *testing.T) {
	_, _ = setup(t)
	root := t.TempDir()
	s := NewInMemorySkillStore(nil)
	_ = s.Put(rawSkill("a", 1, "old"))
	_ = s.Put(rawSkill("a", 2, "new"))
	_ = s.Put(rawSkill("b", 1, "b"))
	_ = s.Put(rawSkill("c", 1, "c"))
	SetSkillStore(s)
	r, err := WriteAllSkills(root, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(actionKinds(r), []string{"a=written", "b=written", "c=written"}) || r.Actions()[0].Version() != 2 {
		t.Fatalf("%v", actionKinds(r))
	}
	if readFile(t, filepath.Join(root, "a", "SKILL.md")) != "new" {
		t.Fatal("the newest version must win")
	}
}

func TestManifestFormatAndPreservation(t *testing.T) {
	_, _ = setup(t)
	root := t.TempDir()
	mustWrite(t, []Skill{skillOf("a", 1, "a")}, root, nil)
	m := readManifest(t, root)
	e := m["entries"].(map[string]any)["a/SKILL.md"].(map[string]any)
	if m["manifestVersion"] != float64(1) || e["key"] != "a" || e["version"] != float64(1) || e["sha256"] != sha("a") {
		t.Fatalf("manifest %v", m)
	}
	if _, err := time.Parse("2006-01-02T15:04:05Z", e["writtenAt"].(string)); err != nil {
		t.Fatalf("writtenAt %v", e["writtenAt"])
	}

	// Unknown fields survive, and the bytes are exactly the cross-SDK form.
	root2 := t.TempDir()
	_ = os.MkdirAll(filepath.Join(root2, "a"), 0o755)
	_ = os.WriteFile(filepath.Join(root2, "a", "SKILL.md"), []byte("a"), 0o644)
	seed := `{"manifestVersion":1,"zeta":"é 🧪 <&>","entries":{"a/SKILL.md":{"key":"a","version":1,"sha256":"` + sha("a") + `","writtenAt":"2020-01-01T00:00:00Z","futureField":1.50}}}`
	_ = os.WriteFile(filepath.Join(root2, ".launchdarkly-skills.json"), []byte(seed), 0o644)
	r := mustWrite(t, []Skill{skillOf("a", 1, "a")}, root2, nil)
	if !slices.Equal(actionKinds(r), []string{"a=skipped_current"}) {
		t.Fatalf("%v", actionKinds(r))
	}
	want := "{\n  \"entries\": {\n    \"a/SKILL.md\": {\n      \"futureField\": 1.50,\n      \"key\": \"a\",\n      \"sha256\": \"" + sha("a") +
		"\",\n      \"version\": 1,\n      \"writtenAt\": \"2020-01-01T00:00:00Z\"\n    }\n  },\n  \"manifestVersion\": 1,\n  \"zeta\": \"" +
		`\u00e9 \ud83e\uddea <&>` + "\"\n}"
	if got := readFile(t, filepath.Join(root2, ".launchdarkly-skills.json")); got != want {
		t.Fatalf("manifest bytes:\n%s\nwant:\n%s", got, want)
	}
}

func TestReconcileSemantics(t *testing.T) {
	_, _ = setup(t)
	root := t.TempDir()
	target := filepath.Join(root, "a", "SKILL.md")
	mustWrite(t, []Skill{skillOf("a", 1, "v1")}, root, nil)
	before, _ := os.Stat(target)
	writtenAt := manifestEntries(t, root)["a/SKILL.md"].(map[string]any)["writtenAt"]
	calls := hookRename(t, nil)
	time.Sleep(10 * time.Millisecond)
	r := mustWrite(t, []Skill{skillOf("a", 1, "v1")}, root, nil)
	after, _ := os.Stat(target)
	if !slices.Equal(actionKinds(r), []string{"a=skipped_current"}) || len(*calls) != 0 || !after.ModTime().Equal(before.ModTime()) {
		t.Fatalf("unchanged: %v renames=%d", actionKinds(r), len(*calls))
	}
	if manifestEntries(t, root)["a/SKILL.md"].(map[string]any)["writtenAt"] != writtenAt {
		t.Fatal("skipped_current must not re-stamp writtenAt")
	}
	if r := mustWrite(t, []Skill{skillOf("a", 2, "v2")}, root, nil); !slices.Equal(actionKinds(r), []string{"a=updated"}) || readFile(t, target) != "v2" {
		t.Fatalf("new version: %v", actionKinds(r))
	}
	_ = os.WriteFile(target, []byte("local edit"), 0o644)
	if r := mustWrite(t, []Skill{skillOf("a", 2, "v2")}, root, nil); !slices.Equal(actionKinds(r), []string{"a=updated"}) || readFile(t, target) != "v2" {
		t.Fatalf("tampering: %v", actionKinds(r))
	}
}

func TestAdoption(t *testing.T) {
	_, _ = setup(t)
	root := t.TempDir()
	_ = os.MkdirAll(filepath.Join(root, "a"), 0o755)
	_ = os.WriteFile(filepath.Join(root, "a", "SKILL.md"), []byte("content"), 0o644)
	r := mustWrite(t, []Skill{skillOf("a", 1, "content")}, root, nil)
	if !slices.Equal(actionKinds(r), []string{"a=skipped_current"}) {
		t.Fatalf("identical bytes: %v", actionKinds(r))
	}
	e := manifestEntries(t, root)["a/SKILL.md"].(map[string]any)
	if e["key"] != "a" || e["writtenAt"] == nil {
		t.Fatalf("adopted entry %v", e)
	}
	// Adopted is prunable afterwards.
	if r := mustWrite(t, []Skill{}, root, nil); !slices.Equal(actionKinds(r), []string{"a=removed"}) {
		t.Fatalf("prune of adopted: %v", actionKinds(r))
	}

	// One byte different: refused, untouched.
	root2 := t.TempDir()
	_ = os.MkdirAll(filepath.Join(root2, "a"), 0o755)
	_ = os.WriteFile(filepath.Join(root2, "a", "SKILL.md"), []byte("contenT"), 0o644)
	r = mustWrite(t, []Skill{skillOf("a", 1, "content")}, root2, nil)
	if !slices.Equal(actionKinds(r), []string{"a=error"}) || readFile(t, filepath.Join(root2, "a", "SKILL.md")) != "contenT" {
		t.Fatalf("differing bytes: %v", actionKinds(r))
	}
	if _, listed := manifestEntries(t, root2)["a/SKILL.md"]; listed {
		t.Fatal("a refused file was claimed")
	}
	// Content plus trailing bytes is not current (the len+1 bound).
	root3 := t.TempDir()
	_ = os.MkdirAll(filepath.Join(root3, "a"), 0o755)
	_ = os.WriteFile(filepath.Join(root3, "a", "SKILL.md"), []byte("content"+strings.Repeat("!", 1<<20)), 0o644)
	if r := mustWrite(t, []Skill{skillOf("a", 1, "content")}, root3, nil); !slices.Equal(actionKinds(r), []string{"a=error"}) {
		t.Fatalf("over-large planted file: %v", actionKinds(r))
	}
}

func TestUnreadableUnmanagedFileIsNeverOverwritten(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("root reads a mode-000 file")
	}
	_, _ = setup(t)
	root := t.TempDir()
	_ = os.MkdirAll(filepath.Join(root, "a"), 0o755)
	p := filepath.Join(root, "a", "SKILL.md")
	_ = os.WriteFile(p, []byte("mine"), 0o000)
	t.Cleanup(func() { _ = os.Chmod(p, 0o644) })
	r := mustWrite(t, []Skill{skillOf("a", 1, "content")}, root, nil)
	if !slices.Equal(actionKinds(r), []string{"a=error"}) || !strings.Contains(r.Errors()[0].ErrorMessage(), "could not be read") {
		t.Fatalf("%v %v", actionKinds(r), r.Errors())
	}
	_ = os.Chmod(p, 0o644)
	if readFile(t, p) != "mine" {
		t.Fatal("overwritten")
	}
}

func TestPruneSemantics(t *testing.T) {
	_, _ = setup(t)
	root := t.TempDir()
	mustWrite(t, []Skill{skillOf("a", 1, "a"), skillOf("b", 4, "b")}, root, nil)
	_ = os.WriteFile(filepath.Join(root, "notes.txt"), []byte("user file"), 0o644)
	_ = os.WriteFile(filepath.Join(root, "b", "extra.txt"), []byte("user file"), 0o644)

	keep := mustWrite(t, []Skill{skillOf("a", 1, "a")}, root, &WriteSkillsOptions{NoPrune: true})
	if slices.Contains(actionKinds(keep), "b=removed") {
		t.Fatal("NoPrune removed a skill")
	}
	r := mustWrite(t, []Skill{skillOf("a", 1, "a")}, root, nil)
	if !slices.Equal(actionKinds(r), []string{"a=skipped_current", "b=removed"}) || r.Actions()[1].Version() != 4 {
		t.Fatalf("%v", actionKinds(r))
	}
	if _, err := os.Stat(filepath.Join(root, "b", "SKILL.md")); !os.IsNotExist(err) {
		t.Fatal("pruned file still on disk")
	}
	if readFile(t, filepath.Join(root, "notes.txt")) != "user file" || readFile(t, filepath.Join(root, "b", "extra.txt")) != "user file" {
		t.Fatal("prune touched an unmanaged file")
	}
	if _, listed := manifestEntries(t, root)["b/SKILL.md"]; listed {
		t.Fatal("pruned entry kept")
	}
	// The emptied directory is removed too.
	mustWrite(t, []Skill{skillOf("c", 1, "c")}, root, nil)
	mustWrite(t, []Skill{}, root, nil)
	if _, err := os.Stat(filepath.Join(root, "c")); !os.IsNotExist(err) {
		t.Fatal("empty skill directory left behind")
	}
}

func TestPruneRefusalReportsManifestVersion(t *testing.T) {
	_, _ = setup(t)
	root := t.TempDir()
	outside := t.TempDir()
	_ = os.Symlink(outside, filepath.Join(root, "a"))
	writeManifestFile(t, root, map[string]any{"a/SKILL.md": map[string]any{"key": "a", "version": 7, "sha256": sha("x")}})
	r := mustWrite(t, []Skill{}, root, nil)
	if !slices.Equal(actionKinds(r), []string{"a=error"}) || r.Actions()[0].Version() != 7 {
		t.Fatalf("%v version=%d", actionKinds(r), r.Actions()[0].Version())
	}
}

func TestWithheldSkillIsProtectedByItsKey(t *testing.T) {
	_, _ = setup(t)
	root := t.TempDir()
	mustWrite(t, []Skill{skillOf("good", 1, "g"), skillOf("bad", 1, "b"), skillOf("gone", 1, "x")}, root, nil)
	tampered := rawSkill("bad", 2, "b2")
	tampered["content"] = "evil"
	SetSkillStore(NewInMemorySkillStore(map[string]any{
		"good:1":   rawSkill("good", 1, "g"),
		"bad:2 !!": tampered, // the map key is not a valid skill key; the object's own key is
	}))
	r, _ := WriteAllSkills(root, nil)
	if !slices.Equal(actionKinds(r), []string{"bad=error", "good=skipped_current", "gone=removed"}) {
		t.Fatalf("%v", actionKinds(r))
	}
	if readFile(t, filepath.Join(root, "bad", "SKILL.md")) != "b" {
		t.Fatal("withheld skill's file was not kept")
	}
	if _, listed := manifestEntries(t, root)["bad/SKILL.md"]; !listed {
		t.Fatal("withheld skill's entry was dropped")
	}
}

func TestUnattributableWithholdingSuppressesPrune(t *testing.T) {
	_, _ = setup(t)
	root := t.TempDir()
	mustWrite(t, []Skill{skillOf("kept", 1, "k")}, root, nil)
	SetSkillStore(NewInMemorySkillStore(map[string]any{"x:1 !!": map[string]any{"version": 1, "content": "c", "contentHash": sha("c")}}))
	r, _ := WriteAllSkills(root, nil)
	if !slices.Equal(actionKinds(r), []string{"=error"}) {
		t.Fatalf("%v", actionKinds(r))
	}
	if readFile(t, filepath.Join(root, "kept", "SKILL.md")) != "k" {
		t.Fatal("file pruned")
	}
	if _, listed := manifestEntries(t, root)["kept/SKILL.md"]; !listed {
		t.Fatal("entry dropped")
	}
}

func TestIncompleteRetrievalSuppressesPrune(t *testing.T) {
	_, _ = setup(t)
	root := t.TempDir()
	mustWrite(t, []Skill{skillOf("stale", 1, "s")}, root, nil)
	inner := NewInMemorySkillStore(map[string]any{"ok": rawSkill("ok", 1, "o")})
	SetSkillStore(&funcStore{get: func(kind, key string, v int) (any, error) {
		if key == "down" {
			return nil, errStoreDown
		}
		return inner.GetObject(kind, key, v)
	}})
	r := mustWrite(t, []SkillKey{"ok", "down"}, root, nil)
	if !slices.Equal(actionKinds(r), []string{"ok=written", "down=error"}) {
		t.Fatalf("%v", actionKinds(r))
	}
	if readFile(t, filepath.Join(root, "stale", "SKILL.md")) != "s" {
		t.Fatal("pruned under an incomplete retrieval")
	}
	if _, listed := manifestEntries(t, root)["stale/SKILL.md"]; !listed {
		t.Fatal("entry dropped")
	}
}

func TestUninitializedStoreDeletesNothing(t *testing.T) {
	_, _ = setup(t)
	root := t.TempDir()
	mustWrite(t, []Skill{skillOf("a", 1, "a")}, root, nil)
	probe := &probeStore{SkillStore: NewInMemorySkillStore(nil)}
	SetSkillStore(probe)
	r, _ := WriteAllSkills(root, nil)
	if r.OK() || !slices.Equal(actionKinds(r), []string{"=error"}) || !strings.Contains(r.Errors()[0].ErrorMessage(), "WaitForSkills") {
		t.Fatalf("%v %v", actionKinds(r), r.Errors())
	}
	if readFile(t, filepath.Join(root, "a", "SKILL.md")) != "a" {
		t.Fatal("deleted against an uninitialized store")
	}
	probe.ready = true
	r, _ = WriteAllSkills(root, nil)
	if !slices.Equal(actionKinds(r), []string{"a=removed"}) {
		t.Fatalf("after initialization: %v", actionKinds(r))
	}
}

func TestWrongVersionOnTheWritePath(t *testing.T) {
	_, logs := setup(t)
	root := t.TempDir()
	mustWrite(t, []Skill{skillOf("pinned", 1, "p"), skillOf("revoked", 1, "r")}, root, nil)
	SetSkillStore(&funcStore{get: func(_, key string, _ int) (any, error) { return rawSkill(key, 9, "nine"), nil }})
	r := mustWrite(t, []SkillReference{NewSkillReference("pinned", 1)}, root, nil)
	if !slices.Equal(actionKinds(r), []string{"pinned=error", "revoked=removed"}) {
		t.Fatalf("%v", actionKinds(r))
	}
	if readFile(t, filepath.Join(root, "pinned", "SKILL.md")) != "p" {
		t.Fatal("wrong_version lost the file on disk")
	}
	found := false
	for _, rec := range logs.records() {
		found = found || rec["reason_code"] == "version_mismatch"
	}
	if !found {
		t.Fatal("version_mismatch record not written on the write path")
	}
	// key_mismatch reaches the write path through the same resolution.
	SetSkillStore(&funcStore{get: func(string, string, int) (any, error) { return rawSkill("impostor", 1, "i"), nil }})
	r = mustWrite(t, []SkillReference{NewSkillReference("pinned", 1)}, root, nil)
	if !slices.Equal(actionKinds(r), []string{"pinned=error"}) || readFile(t, filepath.Join(root, "pinned", "SKILL.md")) != "p" {
		t.Fatalf("key mismatch on write: %v", actionKinds(r))
	}
	if last := logs.records()[len(logs.records())-1]; last["reason_code"] != "key_mismatch" {
		t.Fatalf("record %v", last)
	}
}

func TestRootHandling(t *testing.T) {
	_, _ = setup(t)
	base := t.TempDir()
	leaf := filepath.Join(base, "new")
	if r := mustWrite(t, []Skill{skillOf("a", 1, "a")}, leaf, nil); !r.OK() {
		t.Fatalf("%v", actionKinds(r))
	}
	if _, err := WriteSkills([]Skill{skillOf("a", 1, "a")}, filepath.Join(base, "x", "y", "z"), nil); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("missing ancestors: %v", err)
	}
	if _, err := os.Stat(filepath.Join(base, "x")); !os.IsNotExist(err) {
		t.Fatal("ancestors were created")
	}
	file := filepath.Join(base, "file")
	_ = os.WriteFile(file, nil, 0o644)
	if _, err := WriteSkills([]Skill{skillOf("a", 1, "a")}, file, nil); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("root is a file: %v", err)
	}
	if _, err := WriteSkills([]Skill{}, "", nil); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("empty root: %v", err)
	}
	if _, err := WriteSkills([]*Skill{nil}, base, nil); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("nil skill: %v", err)
	}
}

func TestAtomicRenameFailure(t *testing.T) {
	_, _ = setup(t)
	for _, existing := range []bool{true, false} {
		root := t.TempDir()
		if existing {
			mustWrite(t, []Skill{skillOf("a", 1, "prior")}, root, nil)
		}
		injected := errors.New("injected rename failure")
		calls := hookRename(t, func(renameCall) error { return injected })
		r := mustWrite(t, []Skill{skillOf("a", 2, "next")}, root, nil)
		dev, ino := inode(t, filepath.Join(root, "a"))
		if len(*calls) != 1 {
			t.Fatalf("rename attempted %d times, want exactly 1", len(*calls))
		}
		c := (*calls)[0]
		if !c.sameDirectoryFds || c.dst != "SKILL.md" || c.dev != dev || c.ino != ino || !strings.HasPrefix(c.src, ".SKILL.md.") {
			t.Fatalf("temp file not in the target's own directory: %+v", c)
		}
		if !slices.Equal(actionKinds(r), []string{"a=error"}) || !strings.Contains(r.Errors()[0].ErrorMessage(), "injected rename failure") {
			t.Fatalf("%v %v", actionKinds(r), r.Errors())
		}
		names, _ := os.ReadDir(filepath.Join(root, "a"))
		var listed []string
		for _, n := range names {
			listed = append(listed, n.Name())
		}
		want := []string{}
		if existing {
			want = []string{"SKILL.md"}
			if readFile(t, filepath.Join(root, "a", "SKILL.md")) != "prior" {
				t.Fatal("target changed")
			}
		}
		if !slices.Equal(listed, want) && !(len(listed) == 0 && len(want) == 0) {
			t.Fatalf("leftovers %v", listed)
		}
		_ = readManifest(t, root) // still valid JSON after a per-skill error
		fsOps.rename = unix.Renameat
	}
}

func TestOrphanTempFilesAreSwept(t *testing.T) {
	_, _ = setup(t)
	root := t.TempDir()
	mustWrite(t, []Skill{skillOf("a", 1, "a")}, root, nil)
	orphan := filepath.Join(root, "a", ".SKILL.md.0123456789abcdef.tmp")
	lookalike := filepath.Join(root, "a", ".SKILL.md.notatemp.tmp")
	_ = os.WriteFile(orphan, []byte("partial"), 0o600)
	_ = os.WriteFile(lookalike, []byte("user"), 0o600)
	mustWrite(t, []Skill{skillOf("a", 2, "b")}, root, nil)
	if _, err := os.Stat(orphan); !os.IsNotExist(err) {
		t.Fatal("orphan temp file not swept")
	}
	if _, err := os.Stat(lookalike); err != nil {
		t.Fatal("a non-matching name was swept")
	}
}

func TestPermissions(t *testing.T) {
	_, _ = setup(t)
	old := syscall.Umask(0o077)
	defer syscall.Umask(old)
	root := t.TempDir()
	_ = os.Chmod(root, 0o750)
	_ = os.Mkdir(filepath.Join(root, "existing"), 0o700)
	mustWrite(t, []Skill{skillOf("a", 1, "a"), skillOf("existing", 1, "e")}, root, nil)
	mode := func(p string) os.FileMode {
		st, err := os.Stat(p)
		if err != nil {
			t.Fatal(err)
		}
		return st.Mode()
	}
	if m := mode(filepath.Join(root, "a", "SKILL.md")).Perm(); m != 0o644 {
		t.Fatalf("SKILL.md mode %o", m)
	}
	if m := mode(filepath.Join(root, "a")).Perm(); m != 0o755 {
		t.Fatalf("new skill directory mode %o under umask 077", m)
	}
	if m := mode(filepath.Join(root, "existing")).Perm(); m != 0o700 {
		t.Fatalf("existing directory mode changed to %o", m)
	}
	if m := mode(root).Perm(); m != 0o750 {
		t.Fatalf("root mode changed to %o", m)
	}
}

func TestSetgidInheritedBitIsKept(t *testing.T) {
	_, _ = setup(t)
	root := t.TempDir()
	if err := os.Chmod(root, 0o755|os.ModeSetgid); err != nil {
		t.Skip(err)
	}
	probe := filepath.Join(root, "probe")
	_ = os.Mkdir(probe, 0o755)
	st, _ := os.Stat(probe)
	_ = os.Remove(probe)
	if st.Mode()&os.ModeSetgid == 0 {
		t.Skip("this platform does not propagate setgid to new directories")
	}
	mustWrite(t, []Skill{skillOf("a", 1, "a")}, root, nil)
	st, _ = os.Stat(filepath.Join(root, "a"))
	if st.Mode()&os.ModeSetgid == 0 || st.Mode().Perm() != 0o755 {
		t.Fatalf("mode %v", st.Mode())
	}
}

func TestResilienceKeepAndRaise(t *testing.T) {
	_, _ = setup(t)
	root := t.TempDir()
	mustWrite(t, []Skill{skillOf("a", 1, "a")}, root, nil)
	before := readFile(t, filepath.Join(root, ".launchdarkly-skills.json"))
	r := mustWrite(t, []SkillKey{"a", "b"}, root, nil) // no store configured
	if !slices.Equal(actionKinds(r), []string{"a=error", "b=error"}) || readFile(t, filepath.Join(root, "a", "SKILL.md")) != "a" {
		t.Fatalf("keep: %v", actionKinds(r))
	}
	if !strings.Contains(r.Errors()[0].ErrorMessage(), "skill retrieval unavailable") {
		t.Fatalf("keep message %q", r.Errors()[0].ErrorMessage())
	}
	_, err := WriteSkills([]SkillKey{"a"}, root, &WriteSkillsOptions{OnUnavailable: OnUnavailableRaise})
	if !errors.Is(err, ErrSkillsUnavailable) || !strings.Contains(err.Error(), "unavailable") {
		t.Fatalf("raise: %v", err)
	}
	SetSkillStore(&funcStore{get: func(string, string, int) (any, error) { return nil, errStoreDown }})
	if _, err := WriteSkills([]SkillKey{"a"}, root, &WriteSkillsOptions{OnUnavailable: OnUnavailableRaise}); !errors.Is(err, ErrSkillsUnavailable) {
		t.Fatalf("raising store with raise: %v", err)
	}
	if readFile(t, filepath.Join(root, ".launchdarkly-skills.json")) != before {
		t.Fatal("a failed run changed the manifest")
	}
	if _, err := WriteSkills([]SkillKey{"a"}, root, &WriteSkillsOptions{OnUnavailable: "sometimes"}); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("unknown OnUnavailable: %v", err)
	}
}

func TestTimeout(t *testing.T) {
	_, _ = setup(t)
	root := t.TempDir()
	store := &funcStore{get: func(_, key string, _ int) (any, error) { return rawSkill(key, 1, key), nil }}
	SetSkillStore(store)
	zero := time.Duration(0)
	r := mustWrite(t, []SkillKey{"a"}, root, &WriteSkillsOptions{Timeout: &zero})
	if store.calls != 0 || !slices.Equal(actionKinds(r), []string{"a=error"}) || !strings.Contains(r.Errors()[0].ErrorMessage(), "timeout") {
		t.Fatalf("timeout 0: calls=%d %v", store.calls, r.Errors())
	}
	if _, err := WriteSkills([]SkillKey{"a"}, root, &WriteSkillsOptions{Timeout: &zero, OnUnavailable: OnUnavailableRaise}); !errors.Is(err, ErrSkillsUnavailable) || !strings.Contains(err.Error(), "timeout") {
		t.Fatalf("timeout 0 raise: %v", err)
	}
	negative := -time.Second
	if _, err := WriteSkills([]SkillKey{"a"}, root, &WriteSkillsOptions{Timeout: &negative}); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("negative timeout: %v", err)
	}
}

func TestDeadlineIsCheckedInsideThePruneLoop(t *testing.T) {
	_, _ = setup(t)
	root := t.TempDir()
	var many []Skill
	for i := range 20 {
		many = append(many, skillOf(fmt.Sprintf("s%02d", i), 1, "x"))
	}
	mustWrite(t, many, root, nil)
	real := fsOps.unlink
	fsOps.unlink = func(fd int, name string) error {
		time.Sleep(20 * time.Millisecond)
		return real(fd, name)
	}
	t.Cleanup(func() { fsOps.unlink = real })
	budget := 150 * time.Millisecond
	r := mustWrite(t, []Skill{}, root, &WriteSkillsOptions{Timeout: &budget})
	removed, timedOut := 0, 0
	for _, a := range r.Actions() {
		switch {
		case a.Action() == ActionRemoved:
			removed++
		case a.Action() == ActionError && strings.Contains(a.ErrorMessage(), "timeout was exhausted"):
			timedOut++
		}
	}
	if removed == 0 || timedOut == 0 || removed+timedOut != 20 {
		t.Fatalf("removed=%d timedOut=%d", removed, timedOut)
	}
	if n := len(manifestEntries(t, root)); n != timedOut {
		t.Fatalf("entries not reached must stay listed: %d listed, %d timed out", n, timedOut)
	}
}

func TestNoRetry(t *testing.T) {
	_, _ = setup(t)
	store := &funcStore{get: func(string, string, int) (any, error) { return nil, errStoreDown }}
	SetSkillStore(store)
	mustWrite(t, []SkillKey{"a", "b", "c"}, t.TempDir(), nil)
	if store.calls != 3 {
		t.Fatalf("store called %d times for 3 skills", store.calls)
	}
}

func TestVerifyThenWrite(t *testing.T) {
	rec, _ := setup(t)
	root := t.TempDir()
	bad := NewSkill(SkillFields{Key: "a", Version: 1, Content: []byte("body"), ContentHash: sha("other")})
	r := mustWrite(t, []Skill{bad}, root, nil)
	if !slices.Equal(actionKinds(r), []string{"a=error"}) || len(rec.named(signalIntegrityFailure)) != 1 {
		t.Fatalf("%v", actionKinds(r))
	}
	if _, err := os.Stat(filepath.Join(root, "a")); !os.IsNotExist(err) {
		t.Fatal("something was written for a skill that failed verification")
	}
}

// bytes hashed are the bytes written, whatever the caller does to the
// buffers between the hash and the rename.
func TestBytesHashedAreBytesWritten(t *testing.T) {
	_, _ = setup(t)
	root := t.TempDir()
	buf := []byte("original bytes")
	s := NewSkill(SkillFields{Key: "a", Version: 1, Content: buf, ContentHash: sha("original bytes")})
	view := s.Content()
	fired := false
	hookRename(t, func(renameCall) error {
		fired = true
		copy(buf, "MUTATED!")
		copy(view, "MUTATED!")
		return nil
	})
	r := mustWrite(t, []Skill{s}, root, nil)
	if !fired || readFile(t, filepath.Join(root, "a", "SKILL.md")) != "original bytes" {
		t.Fatalf("fired=%v file=%q", fired, readFile(t, filepath.Join(root, "a", "SKILL.md")))
	}
	if manifestEntries(t, root)["a/SKILL.md"].(map[string]any)["sha256"] != sha("original bytes") || !r.OK() {
		t.Fatal("reported hash does not match the bytes on disk")
	}
	_ = bytes.MinRead
}
