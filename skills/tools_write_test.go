//go:build unix

package skills

import (
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"testing"
)

// §3.21a "Tools stay in memory only" and §3.22: WriteSkills writes only
// SKILL.md, and the manifest is the same as for the skill without tools.
func TestWriteSkillsWritesNoTool(t *testing.T) {
	_, _ = setup(t)
	withTools, without := t.TempDir(), t.TempDir()
	storeOf(rawSkillWithTools("pdf", 1, "# PDF", []any{toolEntry("get-weather")}))
	mustWrite(t, []SkillInput{SkillKey("pdf")}, withTools, nil)
	storeOf(rawSkill("pdf", 1, "# PDF"))
	mustWrite(t, []SkillInput{SkillKey("pdf")}, without, nil)

	entries, err := os.ReadDir(filepath.Join(withTools, "pdf"))
	if err != nil || len(entries) != 1 || entries[0].Name() != SkillFilename {
		t.Fatalf("skill directory = %v, %v", entries, err)
	}
	if readFile(t, filepath.Join(withTools, "pdf", SkillFilename)) != "# PDF" {
		t.Fatal("SKILL.md is not the content verbatim")
	}
	// No clock seam: writtenAt (second resolution) is the one field two writes
	// may differ in, so it is normalized and every other byte must match.
	writtenAt := regexp.MustCompile(`"writtenAt": "[^"]*"`)
	read := func(root string) string {
		return writtenAt.ReplaceAllString(readFile(t, filepath.Join(root, ManifestFilename)), `"writtenAt": "T"`)
	}
	if a, b := read(withTools), read(without); a != b {
		t.Fatalf("manifest with tools:\n%s\ndiffers from without:\n%s", a, b)
	}
}

func TestToolsDoNotChangeAReconcile(t *testing.T) {
	_, _ = setup(t)
	root := t.TempDir()
	storeOf(rawSkill("pdf", 1, "# PDF"))
	mustWrite(t, []SkillInput{SkillKey("pdf")}, root, nil)
	storeOf(rawSkillWithTools("pdf", 1, "# PDF", []any{toolEntry("get-weather")}))
	if got := actionKinds(mustWrite(t, []SkillInput{SkillKey("pdf")}, root, nil)); !slices.Equal(got, []string{"pdf=skipped_current"}) {
		t.Fatalf("actions = %v", got)
	}
}
