package experimental_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"go/ast"
	"go/parser"
	"go/token"
	"log/slog"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/launchdarkly/go-sdk-common/v3/ldvalue"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/aiserver/experimental"
)

// skillsSurface is the documented Agent Skills surface of the experimental
// entry point. A new exported name in skills.go fails this test, which is
// what keeps a reader over Skill content from being added.
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

func exportedDecls(t *testing.T, files ...string) []string {
	t.Helper()
	fset := token.NewFileSet()
	var out []string
	for _, file := range files {
		f, err := parser.ParseFile(fset, file, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		for name, obj := range f.Scope.Objects {
			if ast.IsExported(name) && obj.Kind != ast.Bad {
				out = append(out, name)
			}
		}
	}
	slices.Sort(out)
	return out
}

func coreFiles(t *testing.T) []string {
	t.Helper()
	all, _ := filepath.Glob("../*.go")
	var out []string
	for _, f := range all {
		if !strings.HasSuffix(f, "_test.go") {
			out = append(out, f)
		}
	}
	if len(out) == 0 {
		t.Fatal("no core files found")
	}
	return out
}

func TestSkillsSurfaceIsExactlyTheDocumentedSet(t *testing.T) {
	got := exportedDecls(t, "skills.go")
	want := slices.Sorted(slices.Values(skillsSurface))
	if !slices.Equal(got, want) {
		t.Fatalf("experimental skills surface:\n got %v\nwant %v", got, want)
	}
	root := exportedDecls(t, coreFiles(t)...)
	for _, name := range skillsSurface {
		if slices.Contains(root, name) {
			t.Errorf("%s leaked onto the aiserver package root", name)
		}
	}
	unexported := []string{"SkillObjectKind", "MaxSkillContentBytes", "FDv2ObjectKind", "FDv2PayloadKind",
		"DefaultBaseURI", "DefaultStreamURI", "DefaultDebounce", "SkillsDefaultBaseURI", "SkillsDefaultStreamURI", "SkillsDefaultDebounceMs"}
	for _, name := range unexported {
		if slices.Contains(got, name) || slices.Contains(root, name) {
			t.Errorf("%s must not be exported", name)
		}
	}
}

// Core never names experimental code.
func TestCoreDoesNotImportExperimentalCode(t *testing.T) {
	fset := token.NewFileSet()
	for _, file := range coreFiles(t) {
		f, err := parser.ParseFile(fset, file, nil, parser.ImportsOnly)
		if err != nil {
			t.Fatal(err)
		}
		for _, imp := range f.Imports {
			path, _ := strconv.Unquote(imp.Path.Value)
			if strings.Contains(path, "internal/skills") || strings.Contains(path, "experimental") {
				t.Fatalf("%s imports %s", file, path)
			}
		}
	}
}

func TestConstantsAndClosedSets(t *testing.T) {
	if experimental.SkillFilename != "SKILL.md" || experimental.ManifestFilename != ".launchdarkly-skills.json" || experimental.ManifestVersion != 1 {
		t.Fatal("constant values")
	}
	reasons := []experimental.SkillOutcomeReason{experimental.ReasonAbsent, experimental.ReasonIntegrityFailure,
		experimental.ReasonOK, experimental.ReasonStoreUnavailable, experimental.ReasonWrongVersion}
	kinds := []experimental.ReconcileActionKind{experimental.ActionWritten, experimental.ActionUpdated,
		experimental.ActionSkippedCurrent, experimental.ActionRemoved, experimental.ActionError}
	modes := []experimental.OnUnavailable{experimental.OnUnavailableKeep, experimental.OnUnavailableRaise}
	if strings.Join(toStrings(reasons), ",") != "absent,integrity_failure,ok,store_unavailable,wrong_version" ||
		strings.Join(toStrings(kinds), ",") != "written,updated,skipped_current,removed,error" ||
		strings.Join(toStrings(modes), ",") != "keep,raise" {
		t.Fatal("closed-set tokens")
	}
}

func toStrings[S ~string](in []S) []string {
	out := make([]string, len(in))
	for i, s := range in {
		out[i] = string(s)
	}
	return out
}

type track struct {
	event string
	data  map[string]any
}

type fakeLD struct {
	mu     sync.Mutex
	tracks []track
}

func (f *fakeLD) JSONVariation(string, ldcontext.Context, ldvalue.Value) (ldvalue.Value, error) {
	return ldvalue.Null(), nil
}
func (f *fakeLD) TrackMetric(event string, _ ldcontext.Context, _ float64, data ldvalue.Value) error {
	var d map[string]any
	_ = json.Unmarshal([]byte(data.JSONString()), &d)
	f.mu.Lock()
	defer f.mu.Unlock()
	f.tracks = append(f.tracks, track{event, d})
	return nil
}
func (f *fakeLD) Flush()       {}
func (f *fakeLD) Close() error { return nil }

func (f *fakeLD) usage() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	var out []string
	for _, tr := range f.tracks {
		if tr.event == "$ld:ai:sdk:usage" {
			out = append(out, tr.data["helper"].(string))
			if tr.data["helperPackageName"] != aiserver.PackageName || tr.data["aiSdkName"] != aiserver.PackageName || tr.data["aiSdkLanguage"] != "go" {
				panic("usage event identity")
			}
		}
	}
	return out
}

// The skills helpers have no usage ID, and skills code never tracks, so none of them sends
// a usage event; JavaScript and Python send none either.
func TestSkillsHelpersReportNoUsage(t *testing.T) {
	ctx := context.Background()
	fake := &fakeLD{}
	if _, err := aiserver.InitWithClient(ctx, fake, aiserver.InitOptions{}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = aiserver.Shutdown(ctx) })
	store := experimental.NewInMemorySkillStore(nil)
	experimental.SetSkillStore(store)
	root := t.TempDir()
	calls := []struct {
		helper string
		call   func()
	}{
		{"WatchAllSkills", func() {
			_, w, err := experimental.WatchAllSkills(ctx, root, nil)
			if err != nil {
				t.Fatal(err)
			}
			w.Close()
		}},
		{"GetSkill", func() { _, _ = experimental.GetSkill("a", 0) }},
		{"GetSkillResult", func() { _, _ = experimental.GetSkillResult("a", 0) }},
		{"GetSkills", func() { _, _ = experimental.GetSkills([]experimental.SkillKey{"a"}) }},
		{"AllSkills", func() { _, _ = experimental.AllSkills() }},
		{"WriteAllSkills", func() { _, _ = experimental.WriteAllSkills(root, nil) }},
	}
	for _, c := range calls {
		before := len(fake.usage())
		c.call()
		c.call()
		if got := fake.usage()[before:]; len(got) != 0 {
			t.Fatalf("%s: usage events %v", c.helper, got)
		}
	}
	// SkillRefs is pure: no usage event, no track at all.
	before := len(fake.tracks)
	_, _ = experimental.SkillRefs(&aiserver.AiConfig{})
	if len(fake.tracks) != before {
		t.Fatal("SkillRefs called track")
	}
}

func TestShutdownClearsTheStore(t *testing.T) {
	// A client exists, so the usage event is sent now rather than held for
	// the next test's client.
	if _, err := aiserver.InitWithClient(context.Background(), &fakeLD{}, aiserver.InitOptions{}); err != nil {
		t.Fatal(err)
	}
	experimental.SetSkillStore(experimental.NewInMemorySkillStore(nil))
	if _, err := experimental.GetSkill("a", 0); err != nil {
		t.Fatal(err)
	}
	if err := aiserver.Shutdown(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err := aiserver.InitWithClient(context.Background(), &fakeLD{}, aiserver.InitOptions{}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = aiserver.Shutdown(context.Background()) })
	if _, err := experimental.GetSkill("a", 0); !errors.Is(err, experimental.ErrNoSkillStore) {
		t.Fatalf("after Shutdown: %v", err)
	}
}

// aiserver.SetLogger routes the experimental features' logs too.
func TestSetLoggerRoutesSkillsLogs(t *testing.T) {
	var buf bytes.Buffer
	aiserver.SetLogger(slog.New(slog.NewTextHandler(&buf, nil)))
	t.Cleanup(func() { aiserver.SetLogger(nil) })
	if _, err := aiserver.InitWithClient(context.Background(), &fakeLD{}, aiserver.InitOptions{}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = aiserver.Shutdown(context.Background()) })
	experimental.SetSkillStore(experimental.NewInMemorySkillStore(map[string]any{
		"a": map[string]any{"key": "a", "version": 1, "content": "hi"}, // no contentHash
	}))
	res, err := experimental.GetSkillResult("a", 0)
	if err != nil || res.Reason() != experimental.ReasonIntegrityFailure {
		t.Fatalf("GetSkillResult: %v %v", res, err)
	}
	if !strings.Contains(buf.String(), "ld.skills.integrity_failure") {
		t.Fatalf("integrity record not on the SDK logger: %q", buf.String())
	}
}
