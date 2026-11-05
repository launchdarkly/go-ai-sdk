package skills_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"strings"
	"sync"
	"testing"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/launchdarkly/go-sdk-common/v3/ldvalue"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/skills"
)

func TestConstantsAndClosedSets(t *testing.T) {
	if skills.SkillFilename != "SKILL.md" || skills.ManifestFilename != ".launchdarkly-skills.json" || skills.ManifestVersion != 1 {
		t.Fatal("constant values")
	}
	reasons := []skills.SkillOutcomeReason{skills.ReasonAbsent, skills.ReasonIntegrityFailure,
		skills.ReasonOK, skills.ReasonStoreUnavailable, skills.ReasonWrongVersion}
	kinds := []skills.ReconcileActionKind{skills.ActionWritten, skills.ActionUpdated,
		skills.ActionSkippedCurrent, skills.ActionRemoved, skills.ActionError}
	modes := []skills.OnUnavailable{skills.OnUnavailableKeep, skills.OnUnavailableRaise}
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

type usageFakeLD struct {
	mu     sync.Mutex
	tracks []track
}

func (f *usageFakeLD) JSONVariation(string, ldcontext.Context, ldvalue.Value) (ldvalue.Value, error) {
	return ldvalue.Null(), nil
}
func (f *usageFakeLD) TrackMetric(event string, _ ldcontext.Context, _ float64, data ldvalue.Value) error {
	var d map[string]any
	_ = json.Unmarshal([]byte(data.JSONString()), &d)
	f.mu.Lock()
	defer f.mu.Unlock()
	f.tracks = append(f.tracks, track{event, d})
	return nil
}
func (f *usageFakeLD) Flush()       {}
func (f *usageFakeLD) Close() error { return nil }

func (f *usageFakeLD) usage() []string {
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

func TestShutdownClearsTheStore(t *testing.T) {
	// A client exists, so the usage event is sent now rather than held for
	// the next test's client.
	if _, err := aiserver.InitWithClient(context.Background(), &usageFakeLD{}, aiserver.InitOptions{}); err != nil {
		t.Fatal(err)
	}
	skills.SetSkillStore(skills.NewInMemorySkillStore(nil))
	if _, err := skills.GetSkill("a", 0); err != nil {
		t.Fatal(err)
	}
	if err := aiserver.Shutdown(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err := aiserver.InitWithClient(context.Background(), &usageFakeLD{}, aiserver.InitOptions{}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = aiserver.Shutdown(context.Background()) })
	if _, err := skills.GetSkill("a", 0); !errors.Is(err, skills.ErrNoSkillStore) {
		t.Fatalf("after Shutdown: %v", err)
	}
}

// SetLogger routes the skills logs too.
func TestSetLoggerRoutesSkillsLogs(t *testing.T) {
	var buf bytes.Buffer
	aiserver.SetLogger(slog.New(slog.NewTextHandler(&buf, nil)))
	t.Cleanup(func() { aiserver.SetLogger(nil) })
	if _, err := aiserver.InitWithClient(context.Background(), &usageFakeLD{}, aiserver.InitOptions{}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = aiserver.Shutdown(context.Background()) })
	skills.SetSkillStore(skills.NewInMemorySkillStore(map[string]any{
		"a": map[string]any{"key": "a", "version": 1, "content": "hi"}, // no contentHash
	}))
	res, err := skills.GetSkillResult("a", 0)
	if err != nil || res.Reason() != skills.ReasonIntegrityFailure {
		t.Fatalf("GetSkillResult: %v %v", res, err)
	}
	if !strings.Contains(buf.String(), "ld.skills.integrity_failure") {
		t.Fatalf("integrity record not on the SDK logger: %q", buf.String())
	}
}
