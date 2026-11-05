package skills

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// skillsSurface pins the exported names of package skills.
// A new or removed name fails here, so it shows in review.
var skillsSurface = []string{
	"ActionError", "ActionRemoved", "ActionSkippedCurrent", "ActionUpdated", "ActionWritten", "AllSkills",
	"ErrInvalidArgument", "ErrNoSkillStore", "ErrSkillsUnavailable", "ErrStoreClosed", "ErrStoreNotWatchable",
	"ErrUnsupportedKind", "ErrUnsupportedPlatform", "GetSkill", "GetSkillResult", "GetSkills",
	"InMemorySkillStore", "InitializationProber", "LaunchDarklySkillStore", "LaunchDarklySkillStoreMode",
	"LaunchDarklySkillStoreModePoll", "LaunchDarklySkillStoreModeStream", "LaunchDarklySkillStoreOptions",
	"Listener", "ListenerRemover", "ListenerStore", "ManifestFilename", "ManifestVersion",
	"NewInMemorySkillStore", "NewLaunchDarklySkillStore", "NewListener", "NewSkill", "NewSkillOutcome",
	"NewSkillReference", "OnUnavailable", "OnUnavailableKeep", "OnUnavailableRaise", "ReasonAbsent",
	"ReasonIntegrityFailure", "ReasonOK", "ReasonStoreUnavailable", "ReasonWrongVersion", "ReconcileAction",
	"ReconcileActionKind", "ReconcileReport", "SetSkillStore", "Skill", "SkillFields", "SkillFilename",
	"SkillInput", "SkillKey", "SkillOutcome", "SkillOutcomeReason", "SkillReference", "SkillRefs",
	"SkillRequest", "SkillStore", "SkillWatcher", "StoreDiagnostics", "WatchAllSkills", "WatchSkills",
	"WatchSkillsOptions", "WriteAllSkills", "WriteSkills", "WriteSkillsOptions",
}

func TestSkillsSurfaceIsPinned(t *testing.T) {
	files, _ := filepath.Glob("*.go")
	fset := token.NewFileSet()
	var got []string
	for _, file := range files {
		if strings.HasSuffix(file, "_test.go") {
			continue
		}
		f, err := parser.ParseFile(fset, file, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		for name := range f.Scope.Objects {
			if ast.IsExported(name) {
				got = append(got, name)
			}
		}
	}
	slices.Sort(got)
	want := slices.Sorted(slices.Values(skillsSurface))
	if !slices.Equal(got, want) {
		var added, removed []string
		for _, n := range got {
			if !slices.Contains(want, n) {
				added = append(added, n)
			}
		}
		for _, n := range want {
			if !slices.Contains(got, n) {
				removed = append(removed, n)
			}
		}
		t.Errorf("skills surface changed: added %v, removed %v", added, removed)
	}
}
