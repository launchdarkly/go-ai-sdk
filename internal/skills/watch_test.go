//go:build unix

package skills

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"
)

// countingStore records listener removals.
type countingStore struct {
	*InMemorySkillStore
	mu      sync.Mutex
	removed int
}

func (c *countingStore) RemoveListener(kind string, l *Listener) {
	c.mu.Lock()
	c.removed++
	c.mu.Unlock()
	c.InMemorySkillStore.RemoveListener(kind, l)
}

// addOnlyStore has AddListener but no RemoveListener.
type addOnlyStore struct {
	SkillStore
	add func(string, *Listener) error
}

func (a addOnlyStore) AddListener(kind string, l *Listener) error { return a.add(kind, l) }

func waitDone(t *testing.T, w *SkillWatcher) {
	t.Helper()
	select {
	case <-w.done:
	case <-time.After(5 * time.Second):
		t.Fatal("the watcher goroutine leaked")
	}
}

func TestWatchRequiresAWatchableStore(t *testing.T) {
	_, _ = setup(t)
	root := t.TempDir()
	if _, _, err := WatchAllSkills(context.Background(), root, nil); !errors.Is(err, ErrNoSkillStore) {
		t.Fatalf("no store: %v", err)
	}
	SetSkillStore(&funcStore{})
	_, _, err := WatchAllSkills(context.Background(), root, nil)
	if !errors.Is(err, ErrStoreNotWatchable) || !strings.Contains(err.Error(), "WriteSkills") || !strings.Contains(err.Error(), "FDv2SkillStore") {
		t.Fatalf("store without AddListener: %v", err)
	}
	SetSkillStore(NewInMemorySkillStore(nil))
	if _, _, err := WatchAllSkills(context.Background(), root, &WatchSkillsOptions{Debounce: -time.Millisecond}); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("negative debounce: %v", err)
	}
}

func TestWatchInitialReconcileRunsOnTheCaller(t *testing.T) {
	_, _ = setup(t)
	store := &countingStore{InMemorySkillStore: NewInMemorySkillStore(nil)}
	SetSkillStore(store)
	_, _, err := WatchAllSkills(context.Background(), filepath.Join(t.TempDir(), "x", "y"), nil)
	if !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("a bad root must fail WatchSkills itself: %v", err)
	}
	if store.removed != 1 {
		t.Fatal("a failed start must detach its listener")
	}
}

// Every WriteSkills argument means the same thing: one case driven
// through both gives the same report.
func TestWatchPassesArgumentsThrough(t *testing.T) {
	_, _ = setup(t)
	SetSkillStore(NewInMemorySkillStore(map[string]any{"a": rawSkill("a", 1, "a")}))
	opts := WriteSkillsOptions{NoPrune: true}
	direct, watched := t.TempDir(), t.TempDir()
	for _, root := range []string{direct, watched} {
		mustWrite(t, []Skill{skillOf("stale", 1, "s")}, root, nil)
	}
	want := mustWrite(t, []SkillKey{"a"}, direct, &opts)
	got, w, err := WatchSkills(context.Background(), []SkillKey{"a"}, watched, &WatchSkillsOptions{WriteSkillsOptions: opts})
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	if !slices.Equal(actionKinds(got), actionKinds(want)) {
		t.Fatalf("watched %v, direct %v", actionKinds(got), actionKinds(want))
	}
}

func TestWatchCoalescesABurstIntoOneReconcile(t *testing.T) {
	_, _ = setup(t)
	store := NewInMemorySkillStore(map[string]any{"a": rawSkill("a", 1, "a")})
	SetSkillStore(store)
	var mu sync.Mutex
	var reports []ReconcileReport
	report, w, err := WatchAllSkills(context.Background(), t.TempDir(), &WatchSkillsOptions{
		Debounce:    200 * time.Millisecond,
		OnReconcile: func(r ReconcileReport) { mu.Lock(); reports = append(reports, r); mu.Unlock() },
	})
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	if !slices.Equal(actionKinds(report), []string{"a=written"}) || w.Reconciles() != 0 {
		t.Fatalf("initial: %v reconciles=%d", actionKinds(report), w.Reconciles())
	}
	// Three commits spread over time, after registration, each far enough
	// apart that only the debounce window can merge them.
	for i := range 3 {
		_ = store.Put(rawSkill("b", i+1, "b"))
		time.Sleep(40 * time.Millisecond)
	}
	eventually(t, "a reconcile", func() bool { return w.Reconciles() >= 1 })
	time.Sleep(400 * time.Millisecond)
	if n := w.Reconciles(); n != 1 {
		t.Fatalf("a burst of three produced %d reconciles, want exactly 1", n)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(reports) != 1 || !slices.Contains(actionKinds(reports[0]), "b=written") {
		t.Fatalf("OnReconcile saw %d reports (initial excluded): %v", len(reports), reports)
	}
}

func TestWatchSurvivesAFailingReconcile(t *testing.T) {
	_, _ = setup(t)
	store := NewInMemorySkillStore(nil)
	SetSkillStore(store)
	parent := t.TempDir()
	root := filepath.Join(parent, "root")
	_, w, err := WatchAllSkills(context.Background(), root, &WatchSkillsOptions{
		Debounce:    time.Millisecond,
		OnReconcile: func(ReconcileReport) { panic("callback exploded") },
	})
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	_ = os.RemoveAll(parent) // the next reconcile fails: the root's parent is gone
	_ = store.Put(rawSkill("a", 1, "a"))
	time.Sleep(50 * time.Millisecond)
	if w.Reconciles() != 0 {
		t.Fatal("a failed reconcile was counted")
	}
	_ = os.MkdirAll(root, 0o755)
	_ = store.Put(rawSkill("a", 2, "a"))
	eventually(t, "the next reconcile", func() bool { return w.Reconciles() == 1 })
	_ = store.Put(rawSkill("a", 3, "a"))
	eventually(t, "a reconcile after a panicking callback", func() bool { return w.Reconciles() == 2 })
}

func TestWatchCloseDetachesAndStops(t *testing.T) {
	_, _ = setup(t)
	store := &countingStore{InMemorySkillStore: NewInMemorySkillStore(nil)}
	SetSkillStore(store)
	_, w, err := WatchAllSkills(context.Background(), t.TempDir(), &WatchSkillsOptions{Debounce: time.Millisecond})
	if err != nil {
		t.Fatal(err)
	}
	w.Close()
	w.Close()
	waitDone(t, w)
	if store.removed != 1 {
		t.Fatalf("RemoveListener called %d times", store.removed)
	}
	_ = store.Put(rawSkill("a", 1, "a"))
	time.Sleep(30 * time.Millisecond)
	if w.Reconciles() != 0 {
		t.Fatal("reconciled after Close")
	}
	// Without RemoveListener, Close still works.
	inner := NewInMemorySkillStore(nil)
	SetSkillStore(addOnlyStore{SkillStore: inner, add: inner.AddListener})
	_, w2, err := WatchAllSkills(context.Background(), t.TempDir(), nil)
	if err != nil {
		t.Fatal(err)
	}
	w2.Close()
	waitDone(t, w2)
}

func TestWatchStopsWhenTheContextIsDone(t *testing.T) {
	_, _ = setup(t)
	store := &countingStore{InMemorySkillStore: NewInMemorySkillStore(nil)}
	SetSkillStore(store)
	ctx, cancel := context.WithCancel(context.Background())
	_, w, err := WatchAllSkills(ctx, t.TempDir(), &WatchSkillsOptions{Debounce: time.Hour})
	if err != nil {
		t.Fatal(err)
	}
	_ = store.Put(rawSkill("a", 1, "a")) // parks the worker in the debounce
	time.Sleep(10 * time.Millisecond)
	cancel()
	waitDone(t, w)
	if store.removed != 1 {
		t.Fatal("cancellation did not detach")
	}
}

// End to end: a revocation over a live stream prunes within the debounce.
func TestWatchPrunesARevocationDeliveredOverTheStream(t *testing.T) {
	_, _ = setup(t)
	ft := &fakeTransport{}
	gate := make(chan struct{})
	ft.push(fakeResp{body: sse(commitEvents("p", "(p:p:1)", put("a:1", "a"))...), gate: gate,
		then: sse(commitEvents("p", "(p:p:2)")...), hold: true}) // then the last skill is revoked
	s := newTestStore(t, FDv2ModeStream, ft, nil)
	start(t, s)
	if !s.WaitForSkills(5 * time.Second) {
		t.Fatal("no first payload")
	}
	SetSkillStore(s)
	root := t.TempDir()
	report, w, err := WatchAllSkills(context.Background(), root, &WatchSkillsOptions{Debounce: 20 * time.Millisecond})
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	close(gate)
	if !slices.Equal(actionKinds(report), []string{"a=written"}) {
		t.Fatalf("initial %v", actionKinds(report))
	}
	eventually(t, "the revocation to prune", func() bool {
		_, err := os.Stat(filepath.Join(root, "a", "SKILL.md"))
		return os.IsNotExist(err)
	})
	if s.Diagnostics().ObjectsRevoked != 1 {
		t.Fatalf("diag %+v", s.Diagnostics())
	}
}
