package layout

import (
	"encoding/json"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// root is the module root, from this package's directory.
const root = "../.."

// Every file with a release-please version marker is in extra-files, so
// release-please bumps it; a missing file keeps reporting its first version.
func TestReleaseMarkersInExtraFiles(t *testing.T) {
	var cfg struct {
		Packages map[string]struct {
			ExtraFiles []string `json:"extra-files"`
		} `json:"packages"`
	}
	readJSON(t, "release-please-config.json", &cfg)
	extra := cfg.Packages["."].ExtraFiles
	for _, f := range markedFiles(t) {
		if !slices.Contains(extra, f) {
			t.Errorf("%s has a release-please marker but is not in release-please-config.json extra-files", f)
		}
	}
}

// markedFiles lists the module's non-test Go files that carry the
// x-release-please-version marker, as slash paths from the module root.
func markedFiles(t *testing.T) []string {
	t.Helper()
	var out []string
	err := filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			// Skips .git and .claude, whose worktrees hold other checkouts.
			if p != root && strings.HasPrefix(d.Name(), ".") {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(p, ".go") || strings.HasSuffix(p, "_test.go") {
			return nil
		}
		b, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		if strings.Contains(string(b), "x-release-please-version") {
			rel, _ := filepath.Rel(root, p)
			out = append(out, filepath.ToSlash(rel))
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walk: %v", err)
	}
	if len(out) < 10 {
		t.Fatalf("marker scan scope wrong: got %d files, want >= 10", len(out))
	}
	return out
}

func readJSON(t *testing.T, name string, v any) {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(root, name))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, v); err != nil {
		t.Fatalf("%s: %v", name, err)
	}
}
