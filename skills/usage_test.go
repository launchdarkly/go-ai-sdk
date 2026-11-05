package skills_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"slices"
	"testing"
	"time"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/skills"
)

func validRaw(key string, version int, content string) map[string]any {
	sum := sha256.Sum256([]byte(content))
	return map[string]any{"key": key, "version": version, "content": content, "contentHash": hex.EncodeToString(sum[:])}
}

// freshClient drops any usage held by an earlier test, then installs a
// recording client and a store holding one valid skill.
func freshClient(t *testing.T) (*usageFakeLD, *skills.InMemorySkillStore) {
	t.Helper()
	ctx := context.Background()
	_ = aiserver.Shutdown(ctx)
	fake := &usageFakeLD{}
	if _, err := aiserver.InitWithClient(ctx, fake, aiserver.InitOptions{}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = aiserver.Shutdown(ctx) })
	store := skills.NewInMemorySkillStore(map[string]any{"a": validRaw("a", 1, "a1")})
	skills.SetSkillStore(store)
	return fake, store
}

// Each public helper reports its client.* ID once, from the core package.
// WriteAllSkills and WatchAllSkills are the "*" forms and report the same ID.
func TestSkillsHelperUsagePerFunction(t *testing.T) {
	ctx := context.Background()
	rows := []struct {
		helper string
		call   func(t *testing.T)
	}{
		{"client.allSkills", func(*testing.T) { _, _ = skills.AllSkills() }},
		{"client.getSkill", func(*testing.T) { _, _ = skills.GetSkill("a", 0) }},
		{"client.getSkillResult", func(*testing.T) { _, _ = skills.GetSkillResult("a", 0) }},
		{"client.getSkills", func(*testing.T) { _, _ = skills.GetSkills([]skills.SkillKey{"a"}) }},
		{"client.skillRefs", func(*testing.T) {
			_, _ = skills.SkillRefs(&aiserver.AiConfig{Raw: map[string]any{"skills": []any{}}})
		}},
		{"client.writeSkills", func(t *testing.T) { _, _ = skills.WriteSkills([]skills.SkillKey{"a"}, t.TempDir(), nil) }},
		{"client.writeSkills", func(t *testing.T) { _, _ = skills.WriteAllSkills(t.TempDir(), nil) }},
		{"client.watchSkills", func(t *testing.T) {
			_, w, err := skills.WatchSkills(ctx, []skills.SkillKey{"a"}, t.TempDir(), nil)
			if err != nil {
				t.Fatal(err)
			}
			w.Close()
		}},
		{"client.watchSkills", func(t *testing.T) {
			_, w, err := skills.WatchAllSkills(ctx, t.TempDir(), nil)
			if err != nil {
				t.Fatal(err)
			}
			w.Close()
		}},
	}
	for _, r := range rows {
		fake, _ := freshClient(t)
		r.call(t)
		r.call(t)
		if got := fake.usage(); !slices.Equal(got, []string{r.helper}) {
			t.Errorf("%s: usage events %v", r.helper, got)
		}
		for _, tr := range fake.tracks {
			if tr.event == "$ld:ai:sdk:usage" && (tr.data["helperPackageVersion"] != aiserver.PackageVersion || tr.data["aiSdkVersion"] != aiserver.PackageVersion) {
				t.Errorf("%s: package version in %v", r.helper, tr.data)
			}
		}
	}
}

func TestSkillsUsageOncePerClient(t *testing.T) {
	first, _ := freshClient(t)
	_, _ = skills.GetSkill("a", 0)
	_, _ = skills.GetSkill("a", 0)
	second, _ := freshClient(t) // shutdown clears the store; freshClient sets it again
	_, _ = skills.GetSkill("a", 0)
	if !slices.Equal(first.usage(), []string{"client.getSkill"}) || !slices.Equal(second.usage(), []string{"client.getSkill"}) {
		t.Fatalf("usage across shutdown: first=%v second=%v", first.usage(), second.usage())
	}
}

func TestSkillRefsUsageHeldUntilAClientExists(t *testing.T) {
	ctx := context.Background()
	_ = aiserver.Shutdown(ctx)
	cfg := &aiserver.AiConfig{Raw: map[string]any{"skills": []any{map[string]any{"key": "a", "version": 1}}}}
	if refs, err := skills.SkillRefs(cfg); err != nil || len(refs) != 1 {
		t.Fatalf("SkillRefs without a client or store: %v %v", refs, err)
	}
	fake := &usageFakeLD{}
	if _, err := aiserver.InitWithClient(ctx, fake, aiserver.InitOptions{}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = aiserver.Shutdown(ctx) })
	if got := fake.usage(); !slices.Equal(got, []string{"client.skillRefs"}) {
		t.Fatalf("held usage after init: %v", got)
	}
}

func TestSkillsUsageReportedWhenTheCallFails(t *testing.T) {
	_, _ = freshClient(t)
	_ = aiserver.Shutdown(context.Background())
	fake := &usageFakeLD{}
	if _, err := aiserver.InitWithClient(context.Background(), fake, aiserver.InitOptions{}); err != nil {
		t.Fatal(err)
	}
	// Shutdown cleared the store, so GetSkill fails.
	if _, err := skills.GetSkill("a", 0); !errors.Is(err, skills.ErrNoSkillStore) {
		t.Fatalf("GetSkill with no store: %v", err)
	}
	if got := fake.usage(); !slices.Equal(got, []string{"client.getSkill"}) {
		t.Fatalf("usage events %v", got)
	}
}

func TestSkillsInternalCallsDoNotReport(t *testing.T) {
	ctx := context.Background()
	t.Run("watch reports only itself, and a re-reconcile reports nothing", func(t *testing.T) {
		fake, store := freshClient(t)
		_, w, err := skills.WatchAllSkills(ctx, t.TempDir(), &skills.WatchSkillsOptions{Debounce: 10 * time.Millisecond})
		if err != nil {
			t.Fatal(err)
		}
		defer w.Close()
		if err := store.Put(validRaw("b", 1, "b1")); err != nil {
			t.Fatal(err)
		}
		deadline := time.Now().Add(5 * time.Second)
		for w.Reconciles() < 1 && time.Now().Before(deadline) {
			time.Sleep(5 * time.Millisecond)
		}
		if w.Reconciles() < 1 {
			t.Fatal("no delivery-triggered reconcile")
		}
		if got := fake.usage(); !slices.Equal(got, []string{"client.watchSkills"}) {
			t.Fatalf("usage events %v", got)
		}
	})
	t.Run("writeSkills resolves without the accessors", func(t *testing.T) {
		fake, _ := freshClient(t)
		root := t.TempDir()
		_, _ = skills.WriteSkills([]skills.SkillReference{skills.NewSkillReference("a", 1)}, root, nil)
		_, _ = skills.WriteSkills([]skills.SkillKey{"a"}, root, nil)
		_, _ = skills.WriteAllSkills(root, nil)
		if got := fake.usage(); !slices.Equal(got, []string{"client.writeSkills"}) {
			t.Fatalf("usage events %v", got)
		}
	})
	t.Run("accessors do not call each other", func(t *testing.T) {
		for _, c := range []struct {
			helper string
			call   func()
		}{
			{"client.getSkills", func() { _, _ = skills.GetSkills([]skills.SkillKey{"a"}) }},
			{"client.getSkillResult", func() { _, _ = skills.GetSkillResult("a", 0) }},
			{"client.getSkill", func() { _, _ = skills.GetSkill("a", 0) }},
		} {
			fake, _ := freshClient(t)
			c.call()
			if got := fake.usage(); !slices.Equal(got, []string{c.helper}) {
				t.Errorf("%s: usage events %v", c.helper, got)
			}
		}
	})
	t.Run("configuration and constructors do not report", func(t *testing.T) {
		fake, _ := freshClient(t)
		skills.SetSkillStore(skills.NewInMemorySkillStore(nil))
		store, err := skills.NewLaunchDarklySkillStore("sdk-test", nil) // never started: no network
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		if got := fake.usage(); len(got) != 0 {
			t.Fatalf("usage events %v", got)
		}
	})
}
