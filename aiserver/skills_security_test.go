//go:build unix

package aiserver

import (
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"golang.org/x/sys/unix"
)

func listDir(t *testing.T, dir string) []string {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	var out []string
	for _, e := range entries {
		out = append(out, e.Name())
	}
	return out
}

func dirsIn(t *testing.T, dir string) []string {
	t.Helper()
	var out []string
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if e.IsDir() || e.Type()&os.ModeSymlink != 0 {
			out = append(out, e.Name())
		}
	}
	return out
}

// each hostile key is refused with no filesystem operation attempted,
// proven by interception and paired with a positive control in the same class.
func TestPathTraversalMatrix(t *testing.T) {
	if !supportsDirFD {
		t.Skip("no *at() family")
	}
	hostile := []string{
		"../evil", "..", ".", "", "/etc/cron.d/x", `..\evil`, "c:evil", "skill:ads", "sk\x00ill",
		"-skill", "Evil", "a/b", strings.Repeat("a", 257), strings.Repeat("a", 256), "con",
	}
	for _, key := range hostile {
		_, _ = setup(t)
		parent := t.TempDir()
		root := filepath.Join(parent, "root")
		_ = os.Mkdir(root, 0o755)
		parentBefore := listDir(t, parent)
		calls := hookRename(t, nil)
		r := mustWrite(t, []Skill{skillOf(key, 1, "payload")}, root, nil)
		if len(r.Actions()) != 1 || r.Actions()[0].Action() != ActionError || r.Actions()[0].Key() != key {
			t.Fatalf("key %q: %v", key, actionKinds(r))
		}
		if len(*calls) != 0 {
			t.Fatalf("key %q: a rename was attempted", key)
		}
		if d := dirsIn(t, root); len(d) != 0 {
			t.Fatalf("key %q: created %v under the root", key, d)
		}
		if !slices.Equal(listDir(t, parent), parentBefore) {
			t.Fatalf("key %q: the root's parent changed", key)
		}
		// Positive control: a valid key under the same interception renames once.
		_ = mustWrite(t, []Skill{skillOf("valid", 1, "payload")}, root, nil)
		if len(*calls) != 1 {
			t.Fatalf("positive control renamed %d times", len(*calls))
		}
	}
	// The longest filesystem-legal key writes.
	_, _ = setup(t)
	long := strings.Repeat("a", 255)
	if r := mustWrite(t, []Skill{skillOf(long, 1, "x")}, t.TempDir(), nil); !r.OK() {
		t.Fatalf("255-character key: %v", r.Errors())
	}
}

func TestSymlinkedRootRaises(t *testing.T) {
	_, _ = setup(t)
	real := t.TempDir()
	link := filepath.Join(t.TempDir(), "link")
	_ = os.Symlink(real, link)
	if _, err := WriteSkills([]Skill{skillOf("a", 1, "a")}, link, nil); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("symlinked root: %v", err)
	}
	if len(listDir(t, real)) != 0 {
		t.Fatal("written through a symlinked root")
	}
}

func TestSymlinkedSkillDirectory(t *testing.T) {
	_, _ = setup(t)
	root, outside := t.TempDir(), t.TempDir()
	_ = os.Symlink(outside, filepath.Join(root, "a"))
	r := mustWrite(t, []Skill{skillOf("a", 1, "a")}, root, nil)
	if !slices.Equal(actionKinds(r), []string{"a=error"}) || len(listDir(t, outside)) != 0 {
		t.Fatalf("%v outside=%v", actionKinds(r), listDir(t, outside))
	}
}

func symlinkedTarget(t *testing.T, outsideContent string, managed bool) (root, outsideFile string) {
	t.Helper()
	root = t.TempDir()
	outsideFile = filepath.Join(t.TempDir(), "victim")
	_ = os.WriteFile(outsideFile, []byte(outsideContent), 0o644)
	_ = os.Mkdir(filepath.Join(root, "a"), 0o755)
	_ = os.Symlink(outsideFile, filepath.Join(root, "a", "SKILL.md"))
	if managed {
		writeManifestFile(t, root, map[string]any{"a/SKILL.md": map[string]any{"key": "a", "version": 1, "sha256": sha(outsideContent)}})
	}
	return root, outsideFile
}

func TestSymlinkedTargetFile(t *testing.T) {
	_, _ = setup(t)
	root, victim := symlinkedTarget(t, "victim", true)
	r := mustWrite(t, []Skill{skillOf("a", 2, "new")}, root, nil)
	if !slices.Equal(actionKinds(r), []string{"a=error"}) || readFile(t, victim) != "victim" {
		t.Fatalf("write path: %v", actionKinds(r))
	}

	// Prune path: the refusal is the contract, not the victim's survival.
	_, _ = setup(t)
	root, _ = symlinkedTarget(t, "victim", true)
	r = mustWrite(t, []Skill{}, root, nil)
	if !slices.Equal(actionKinds(r), []string{"a=error"}) || r.Actions()[0].Version() != 1 {
		t.Fatalf("prune path: %v", actionKinds(r))
	}
	if _, err := os.Lstat(filepath.Join(root, "a", "SKILL.md")); err != nil {
		t.Fatal("the symlink itself was removed")
	}
	if _, listed := manifestEntries(t, root)["a/SKILL.md"]; !listed {
		t.Fatal("manifest entry dropped")
	}

	// Unmanaged, and the link's target bytes are the resolved content: must
	// not be hashed through and adopted.
	_, _ = setup(t)
	root, victim = symlinkedTarget(t, "resolved", false)
	r = mustWrite(t, []Skill{skillOf("a", 1, "resolved")}, root, nil)
	if !slices.Equal(actionKinds(r), []string{"a=error"}) || readFile(t, victim) != "resolved" {
		t.Fatalf("adoption read: %v", actionKinds(r))
	}
	if _, listed := manifestEntries(t, root)["a/SKILL.md"]; listed {
		t.Fatal("a symlink was adopted into the manifest")
	}
}

func withDeadline(t *testing.T, d time.Duration, fn func()) {
	t.Helper()
	done := make(chan struct{})
	go func() { defer close(done); fn() }()
	select {
	case <-done:
	case <-time.After(d):
		t.Fatal("the operation blocked: a FIFO was opened blocking")
	}
}

func TestFIFOAtManagedPath(t *testing.T) {
	for _, managed := range []bool{false, true} {
		_, _ = setup(t)
		root := t.TempDir()
		_ = os.Mkdir(filepath.Join(root, "a"), 0o755)
		fifo := filepath.Join(root, "a", "SKILL.md")
		if err := unix.Mkfifo(fifo, 0o644); err != nil {
			t.Skip("no FIFOs:", err)
		}
		if managed {
			writeManifestFile(t, root, map[string]any{"a/SKILL.md": map[string]any{"key": "a", "version": 1}})
		}
		var r ReconcileReport
		withDeadline(t, 5*time.Second, func() { r = mustWrite(t, []Skill{skillOf("a", 1, "x")}, root, nil) })
		if !slices.Equal(actionKinds(r), []string{"a=error"}) {
			t.Fatalf("managed=%v: %v", managed, actionKinds(r))
		}
		if st, err := os.Lstat(fifo); err != nil || st.Mode()&os.ModeNamedPipe == 0 {
			t.Fatal("the FIFO was replaced")
		}
	}
}

func TestManifestFIFOAndSymlink(t *testing.T) {
	_, _ = setup(t)
	root := t.TempDir()
	if err := unix.Mkfifo(filepath.Join(root, ".launchdarkly-skills.json"), 0o644); err != nil {
		t.Skip("no FIFOs:", err)
	}
	var r ReconcileReport
	withDeadline(t, 5*time.Second, func() { r = mustWrite(t, []Skill{}, root, nil) })
	if r.OK() || !strings.Contains(r.Errors()[0].ErrorMessage(), ".launchdarkly-skills.json") || r.Errors()[0].Key() != "" {
		t.Fatalf("FIFO manifest: %v", r.Errors())
	}

	// A symlinked manifest that, if followed, would authorize a prune.
	_, _ = setup(t)
	root = t.TempDir()
	_ = os.Mkdir(filepath.Join(root, "a"), 0o755)
	_ = os.WriteFile(filepath.Join(root, "a", "SKILL.md"), []byte("a"), 0o644)
	elsewhere := t.TempDir()
	writeManifestFile(t, elsewhere, map[string]any{"a/SKILL.md": map[string]any{"key": "a", "version": 1}})
	_ = os.Symlink(filepath.Join(elsewhere, ".launchdarkly-skills.json"), filepath.Join(root, ".launchdarkly-skills.json"))
	r = mustWrite(t, []Skill{}, root, nil)
	if r.OK() || slices.Contains(actionKinds(r), "a=removed") || readFile(t, filepath.Join(root, "a", "SKILL.md")) != "a" {
		t.Fatalf("symlinked manifest was followed: %v", actionKinds(r))
	}
}

// per-key TOCTOU: the swap fires from the rename hook.
func TestDirectorySwappedAtTheInstantOfTheWrite(t *testing.T) {
	_, _ = setup(t)
	root, outside := t.TempDir(), t.TempDir()
	aside := filepath.Join(root, "aside")
	fired := false
	hookRename(t, func(renameCall) error {
		if err := os.Rename(filepath.Join(root, "a"), aside); err != nil {
			return err
		}
		fired = os.Symlink(outside, filepath.Join(root, "a")) == nil
		return nil
	})
	r := mustWrite(t, []Skill{skillOf("a", 1, "payload")}, root, nil)
	if !fired {
		t.Fatal("the race never fired")
	}
	if len(listDir(t, outside)) != 0 {
		t.Fatalf("the write was redirected outside: %v", listDir(t, outside))
	}
	if readFile(t, filepath.Join(aside, "SKILL.md")) != "payload" {
		t.Fatal("the write did not land in the validated directory")
	}
	if !slices.Equal(actionKinds(r), []string{"a=written"}) {
		t.Fatalf("%v", actionKinds(r))
	}
}

func TestDirectorySwappedAtTheInstantOfThePrune(t *testing.T) {
	_, _ = setup(t)
	root, outside := t.TempDir(), t.TempDir()
	mustWrite(t, []Skill{skillOf("a", 1, "managed")}, root, nil)
	_ = os.WriteFile(filepath.Join(outside, "SKILL.md"), []byte("victim"), 0o644)
	aside := filepath.Join(root, "aside")
	real := fsOps.unlink
	fired := false
	fsOps.unlink = func(fd int, name string) error {
		if strings.HasSuffix(name, "SKILL.md") && !fired {
			_ = os.Rename(filepath.Join(root, "a"), aside)
			fired = os.Symlink(outside, filepath.Join(root, "a")) == nil
		}
		return real(fd, name)
	}
	t.Cleanup(func() { fsOps.unlink = real })
	mustWrite(t, []Skill{}, root, nil)
	if !fired {
		t.Fatal("the race never fired")
	}
	if readFile(t, filepath.Join(outside, "SKILL.md")) != "victim" {
		t.Fatal("the prune deleted the attacker-chosen file")
	}
	if _, err := os.Stat(filepath.Join(aside, "SKILL.md")); !os.IsNotExist(err) {
		t.Fatal("the managed file was not the one removed")
	}
}

// root level: the swap fires from the root-open hook, right after the
// pin, so the manifest read and every operation must use the pinned root.
func TestRootSwappedAfterThePin(t *testing.T) {
	for _, prunePath := range []bool{false, true} {
		_, _ = setup(t)
		parent := t.TempDir()
		root := filepath.Join(parent, "root")
		_ = os.Mkdir(root, 0o755)
		mustWrite(t, []Skill{skillOf("a", 1, "v1")}, root, nil)
		outside := t.TempDir()
		writeManifestFile(t, outside, map[string]any{})
		outsideManifest := readFile(t, filepath.Join(outside, ".launchdarkly-skills.json"))
		aside := filepath.Join(parent, "aside")
		real := fsOps.openRoot
		fired := false
		fsOps.openRoot = func(p string) (int, error) {
			fd, err := real(p)
			if err == nil {
				_ = os.Rename(root, aside)
				fired = os.Symlink(outside, root) == nil
			}
			return fd, err
		}
		var in []Skill
		if !prunePath {
			in = []Skill{skillOf("a", 2, "v2")}
		}
		r := mustWrite(t, in, root, nil)
		fsOps.openRoot = real
		if !fired {
			t.Fatal("the race never fired")
		}
		if prunePath {
			if !slices.Equal(actionKinds(r), []string{"a=removed"}) {
				t.Fatalf("prune: %v (the pinned manifest was not the one read)", actionKinds(r))
			}
			if _, err := os.Stat(filepath.Join(aside, "a", "SKILL.md")); !os.IsNotExist(err) {
				t.Fatal("prune did not land under the validated root")
			}
		} else {
			if !slices.Equal(actionKinds(r), []string{"a=updated"}) || readFile(t, filepath.Join(aside, "a", "SKILL.md")) != "v2" {
				t.Fatalf("write: %v", actionKinds(r))
			}
		}
		if !slices.Equal(listDir(t, outside), []string{".launchdarkly-skills.json"}) || readFile(t, filepath.Join(outside, ".launchdarkly-skills.json")) != outsideManifest {
			t.Fatalf("the outside directory was touched: %v", listDir(t, outside))
		}
	}
}

// clobber protection.
func TestClobberProtection(t *testing.T) {
	_, _ = setup(t)
	root := t.TempDir()
	_ = os.Mkdir(filepath.Join(root, "a"), 0o755)
	_ = os.WriteFile(filepath.Join(root, "a", "SKILL.md"), []byte("user authored"), 0o644)
	r := mustWrite(t, []Skill{skillOf("a", 1, "ld content")}, root, nil)
	if !slices.Equal(actionKinds(r), []string{"a=error"}) || readFile(t, filepath.Join(root, "a", "SKILL.md")) != "user authored" {
		t.Fatalf("unmanaged: %v", actionKinds(r))
	}
	if r := mustWrite(t, []Skill{}, root, nil); slices.Contains(actionKinds(r), "a=removed") {
		t.Fatal("pruned an unmanaged file")
	}

	// A manifest entry under another key authorizes nothing destructive.
	_, _ = setup(t)
	root = t.TempDir()
	_ = os.Mkdir(filepath.Join(root, "a"), 0o755)
	_ = os.WriteFile(filepath.Join(root, "a", "SKILL.md"), []byte("user authored"), 0o644)
	writeManifestFile(t, root, map[string]any{"a/SKILL.md": map[string]any{"key": "b", "version": 1}})
	r = mustWrite(t, []Skill{skillOf("a", 1, "ld content")}, root, nil)
	if !slices.Contains(actionKinds(r), "a=error") || readFile(t, filepath.Join(root, "a", "SKILL.md")) != "user authored" {
		t.Fatalf("mismatched key, differing bytes: %v", actionKinds(r))
	}

	// Byte-identical under a mismatched-key entry: adopted and re-keyed.
	_, _ = setup(t)
	root = t.TempDir()
	_ = os.Mkdir(filepath.Join(root, "a"), 0o755)
	_ = os.WriteFile(filepath.Join(root, "a", "SKILL.md"), []byte("ld content"), 0o644)
	writeManifestFile(t, root, map[string]any{"a/SKILL.md": map[string]any{"key": "zzz", "version": 1}})
	calls := hookRename(t, nil)
	r = mustWrite(t, []Skill{skillOf("a", 1, "ld content")}, root, nil)
	if !slices.Contains(actionKinds(r), "a=skipped_current") || len(*calls) != 0 {
		t.Fatalf("mismatched key, identical bytes: %v renames=%d", actionKinds(r), len(*calls))
	}
	if manifestEntries(t, root)["a/SKILL.md"].(map[string]any)["key"] != "a" {
		t.Fatal("entry not re-keyed")
	}
}

// corrupt manifest fails closed.
func TestCorruptManifestFailsClosed(t *testing.T) {
	populated := `"entries":{"a/SKILL.md":{"key":"a","version":1,"sha256":"x"}}`
	cases := map[string]string{
		"garbage":          "{not json",
		"entries array":    `{"manifestVersion":1,"entries":[]}`,
		"entries absent":   `{"manifestVersion":1}`,
		"not an object":    `[1,2]`,
		"version too high": `{"manifestVersion":2,` + populated + `}`,
		"version zero":     `{"manifestVersion":0,` + populated + `}`,
		"version negative": `{"manifestVersion":-1,` + populated + `}`,
		"version boolean":  `{"manifestVersion":true,` + populated + `}`,
		"version string":   `{"manifestVersion":"1",` + populated + `}`,
		"version float":    `{"manifestVersion":1.0,` + populated + `}`,
		"trailing garbage": `{"manifestVersion":1,` + populated + `} x`,
		"invalid utf-8":    "{\"manifestVersion\":1,\"x\":\"\xff\"," + populated + "}",
	}
	for name, body := range cases {
		_, _ = setup(t)
		root := t.TempDir()
		_ = os.Mkdir(filepath.Join(root, "a"), 0o755)
		_ = os.WriteFile(filepath.Join(root, "a", "SKILL.md"), []byte("diverging"), 0o644)
		manifest := filepath.Join(root, ".launchdarkly-skills.json")
		_ = os.WriteFile(manifest, []byte(body), 0o644)
		r := mustWrite(t, []Skill{skillOf("a", 2, "resolved"), skillOf("fresh", 1, "new")}, root, nil)
		if readFile(t, filepath.Join(root, "a", "SKILL.md")) != "diverging" {
			t.Fatalf("%s: a managed file was overwritten", name)
		}
		if readFile(t, filepath.Join(root, "fresh", "SKILL.md")) != "new" {
			t.Fatalf("%s: a brand-new file was not written", name)
		}
		var manifestErr *ReconcileAction
		for _, a := range r.Errors() {
			if a.Key() == "" && strings.Contains(a.ErrorMessage(), ".launchdarkly-skills.json") {
				manifestErr = &a
			}
		}
		if manifestErr == nil {
			t.Fatalf("%s: no run-level error naming the manifest: %v", name, r.Errors())
		}
		r = mustWrite(t, []Skill{}, root, nil)
		if slices.Contains(actionKinds(r), "a=removed") {
			t.Fatalf("%s: pruned under a corrupt manifest", name)
		}
		if _, err := os.Stat(filepath.Join(root, "a", "SKILL.md")); err != nil {
			t.Fatalf("%s: file removed", name)
		}
		if readFile(t, manifest) != body {
			t.Fatalf("%s: the corrupt manifest was destroyed", name)
		}
	}
}

func TestManifestPathNotOwnableIsRefused(t *testing.T) {
	_, _ = setup(t)
	root := t.TempDir()
	_ = os.Mkdir(filepath.Join(root, "b"), 0o755)
	_ = os.WriteFile(filepath.Join(root, "b", "SKILL.md"), []byte("b"), 0o644)
	writeManifestFile(t, root, map[string]any{"b/SKILL.md": map[string]any{"key": "a", "version": 3}})
	r := mustWrite(t, []Skill{}, root, nil)
	if !slices.Equal(actionKinds(r), []string{"a=error"}) || r.Actions()[0].Version() != 3 || readFile(t, filepath.Join(root, "b", "SKILL.md")) != "b" {
		t.Fatalf("%v", actionKinds(r))
	}
}
