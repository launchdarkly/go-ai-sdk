//go:build unix

package skills

import (
	"context"
	"errors"
	"log/slog"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/launchdarkly/go-sdk-common/v3/ldvalue"
	ld "github.com/launchdarkly/go-server-sdk/v7"
	"github.com/launchdarkly/go-server-sdk/v7/interfaces"
	"github.com/launchdarkly/go-server-sdk/v7/ldcomponents"
	"github.com/launchdarkly/go-server-sdk/v7/testhelpers/ldtestdata"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

// plainLD is a client brought to InitWithClient that implements only
// aiserver.LDClient: it has no change-listener API.
type plainLD struct {
	mu     sync.Mutex
	value  func(key string, def ldvalue.Value) (ldvalue.Value, error)
	calls  []variationCall
	tracks []string
}

type variationCall struct {
	key string
	ctx ldcontext.Context
	def ldvalue.Value
}

func (f *plainLD) JSONVariation(key string, ctx ldcontext.Context, def ldvalue.Value) (ldvalue.Value, error) {
	f.mu.Lock()
	f.calls = append(f.calls, variationCall{key, ctx, def})
	value := f.value
	f.mu.Unlock()
	if value == nil {
		return def, nil
	}
	return value(key, def)
}

func (f *plainLD) TrackMetric(event string, _ ldcontext.Context, _ float64, data ldvalue.Value) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if event == "$ld:ai:sdk:usage" {
		event += " " + data.GetByKey("helper").StringValue()
	}
	f.tracks = append(f.tracks, event)
	return nil
}
func (f *plainLD) Flush()       {}
func (f *plainLD) Close() error { return nil }

func (f *plainLD) setValue(v func(key string, def ldvalue.Value) (ldvalue.Value, error)) {
	f.mu.Lock()
	f.value = v
	f.mu.Unlock()
}

func (f *plainLD) variationCalls() []variationCall {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.calls)
}

func (f *plainLD) tracked() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.tracks)
}

// trackedLD adds the base SDK's change signal and offline probe.
type trackedLD struct {
	plainLD
	tracker *fakeTracker
	offline bool
}

func (f *trackedLD) GetFlagTracker() interfaces.FlagTracker { return f.tracker }
func (f *trackedLD) IsOffline() bool                        { return f.offline }

// fakeTracker hands out unbuffered channels, as the base SDK's tracker does,
// so a send returns only once the watcher has taken the change.
type fakeTracker struct {
	mu        sync.Mutex
	listeners []chan interfaces.FlagChangeEvent
	added     int
	removed   int
}

func (f *fakeTracker) AddFlagChangeListener() <-chan interfaces.FlagChangeEvent {
	f.mu.Lock()
	defer f.mu.Unlock()
	ch := make(chan interfaces.FlagChangeEvent)
	f.listeners = append(f.listeners, ch)
	f.added++
	return ch
}

func (f *fakeTracker) RemoveFlagChangeListener(listener <-chan interfaces.FlagChangeEvent) {
	f.mu.Lock()
	defer f.mu.Unlock()
	for i, ch := range f.listeners {
		if (<-chan interfaces.FlagChangeEvent)(ch) == listener {
			f.listeners = slices.Delete(f.listeners, i, i+1)
			f.removed++
			close(ch)
			return
		}
	}
}

func (f *fakeTracker) AddFlagValueChangeListener(string, ldcontext.Context, ldvalue.Value) <-chan interfaces.FlagValueChangeEvent {
	panic("the watcher must not use the value-change listener")
}

func (f *fakeTracker) RemoveFlagValueChangeListener(<-chan interfaces.FlagValueChangeEvent) {}

// fire delivers one change to every listener and returns how many took it.
// A listener that does not take it within two seconds fails the test: it
// would be blocking the base SDK's delivery goroutine.
func (f *fakeTracker) fire(t *testing.T, key string) int {
	t.Helper()
	f.mu.Lock()
	ls := slices.Clone(f.listeners)
	f.mu.Unlock()
	for _, ch := range ls {
		select {
		case ch <- interfaces.FlagChangeEvent{Key: key}:
		case <-time.After(2 * time.Second):
			t.Fatalf("a config-change listener did not take the change to %q", key)
		}
	}
	return len(ls)
}

func (f *fakeTracker) counts() (added, removed int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.added, f.removed
}

// installClient installs c as the AI client. Call it before SetSkillStore:
// aiserver.Shutdown clears the store.
func installClient(t *testing.T, c aiserver.LDClient) {
	t.Helper()
	ctx := context.Background()
	_ = aiserver.Shutdown(ctx)
	if _, err := aiserver.InitWithClient(ctx, c, aiserver.InitOptions{}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = aiserver.Shutdown(ctx) })
}

func noClient(t *testing.T) {
	t.Helper()
	_ = aiserver.Shutdown(context.Background())
}

// configValue is an AI Config variation; skills nil leaves the field out.
func configValue(enabled any, skills any) ldvalue.Value {
	m := map[string]any{"_ldMeta": map[string]any{"enabled": enabled, "variationKey": "v1"}, "model": map[string]any{"name": "m"}}
	if skills != nil {
		m["skills"] = skills
	}
	return ldvalue.CopyArbitraryValue(m)
}

func skillsList(pairs ...any) []any {
	var out []any
	for i := 0; i < len(pairs); i += 2 {
		out = append(out, map[string]any{"key": pairs[i], "version": pairs[i+1]})
	}
	return out
}

// countingSource counts reads and answers through answer(n), n from 1.
type countingSource struct {
	mu     sync.Mutex
	n      int
	answer func(ctx context.Context, n int) ([]SkillReference, error)
}

func (c *countingSource) refs(ctx context.Context) ([]SkillReference, error) {
	c.mu.Lock()
	c.n++
	n, answer := c.n, c.answer
	c.mu.Unlock()
	return answer(ctx, n)
}

func (c *countingSource) reads() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.n
}

func answering(refs ...SkillReference) func(context.Context, int) ([]SkillReference, error) {
	return func(context.Context, int) ([]SkillReference, error) { return append([]SkillReference{}, refs...), nil }
}

// fixed is a refs function with a fixed, non-nil answer.
func fixed(refs ...SkillReference) func(context.Context) ([]SkillReference, error) {
	a := answering(refs...)
	return func(ctx context.Context) ([]SkillReference, error) { return a(ctx, 0) }
}

func newSource(t *testing.T, refs func(context.Context) ([]SkillReference, error), configKeys ...string) SkillSource {
	t.Helper()
	src, err := NewSkillSource(refs, configKeys...)
	if err != nil {
		t.Fatalf("NewSkillSource: %v", err)
	}
	return src
}

// readSource reads a source the way a reconcile does.
func readSource(t *testing.T, src SkillSource) ([]SkillReference, error) {
	t.Helper()
	if src.refs == nil {
		t.Fatal("the source has no refs function")
	}
	return src.refs(context.Background())
}

func refPairs(refs []SkillReference) []string {
	var out []string
	for _, r := range refs {
		out = append(out, r.Key()+"@"+strconv.Itoa(r.Version()))
	}
	return out
}

func watchStore(objects map[string]any) *countingStore {
	s := &countingStore{InMemorySkillStore: NewInMemorySkillStore(objects)}
	SetSkillStore(s)
	return s
}

// revoke drops one version from an in-memory store and notifies, as a
// delivered revocation does.
func revoke(s *InMemorySkillStore, key string, version int) {
	ts := tombstone{key: key, version: version}
	s.mu.Lock()
	s.objects.remove(ts)
	s.mu.Unlock()
	s.listeners.notify([]any{tombstoneChange(ts)})
}

func waitReconciles(t *testing.T, w *SkillWatcher, n int) {
	t.Helper()
	eventually(t, "reconcile", func() bool { return w.Reconciles() >= n })
}

func watchFrom(t *testing.T, src SkillSource, root string, opts *WatchSkillsOptions) (ReconcileReport, *SkillWatcher) {
	t.Helper()
	report, w, err := WatchSkillsFrom(context.Background(), src, root, opts)
	if err != nil {
		t.Fatalf("WatchSkillsFrom: %v", err)
	}
	t.Cleanup(w.Close)
	return report, w
}

// reportLog collects OnReconcile reports.
type reportLog struct {
	mu      sync.Mutex
	reports []ReconcileReport
}

func (r *reportLog) add(rep ReconcileReport) {
	r.mu.Lock()
	r.reports = append(r.reports, rep)
	r.mu.Unlock()
}

func (r *reportLog) all() []ReconcileReport {
	r.mu.Lock()
	defer r.mu.Unlock()
	return slices.Clone(r.reports)
}

func (r *reportLog) last(t *testing.T) ReconcileReport {
	t.Helper()
	all := r.all()
	if len(all) == 0 {
		t.Fatal("no delivery-triggered report")
	}
	return all[len(all)-1]
}

func actionFor(r ReconcileReport, key string) (ReconcileAction, bool) {
	for _, a := range r.Actions() {
		if a.Key() == key {
			return a, true
		}
	}
	return ReconcileAction{}, false
}

// ---- Construction ----

func TestSkillSourceConstruction(t *testing.T) {
	_, _ = setup(t)
	refs := func(context.Context) ([]SkillReference, error) { return nil, nil }
	bad := map[string]func() error{
		"nil refs":            func() error { _, err := NewSkillSource(nil); return err },
		"empty config key":    func() error { _, err := NewSkillSource(refs, ""); return err },
		"one empty key":       func() error { _, err := NewSkillSource(refs, "agent", ""); return err },
		"whitespace-only key": func() error { _, err := NewSkillSource(refs, "  "); return err },
	}
	for name, call := range bad {
		if err := call(); !errors.Is(err, ErrInvalidSkillSource) || !errors.Is(err, ErrInvalidArgument) {
			t.Errorf("%s: %v", name, err)
		}
	}
	for _, keys := range [][]string{nil, {"agent"}, {"agent", "planner"}} {
		if _, err := NewSkillSource(refs, keys...); err != nil {
			t.Errorf("%v refused: %v", keys, err)
		}
	}
	root := t.TempDir()
	for name, src := range map[string]SkillSource{
		"zero SkillSource":   {},
		`ConfigSkills("")`:   ConfigSkills("", ldcontext.New("u")),
		`ConfigSkills("  ")`: ConfigSkills("  ", ldcontext.New("u")),
	} {
		if _, err := WriteSkillsFrom(context.Background(), src, root, nil); !errors.Is(err, ErrInvalidSkillSource) {
			t.Errorf("WriteSkillsFrom(%s): %v", name, err)
		}
		SetSkillStore(NewInMemorySkillStore(nil))
		if _, w, err := WatchSkillsFrom(context.Background(), src, root, nil); !errors.Is(err, ErrInvalidSkillSource) || w != nil {
			t.Errorf("WatchSkillsFrom(%s): %v", name, err)
		}
	}
}

// The keys are copied: changing the caller's slice afterwards changes nothing.
func TestSkillSourceCopiesItsConfigKeys(t *testing.T) {
	_, _ = setup(t)
	tr := &fakeTracker{}
	installClient(t, &trackedLD{tracker: tr})
	watchStore(map[string]any{"a": rawSkill("a", 1, "a")})
	cs := &countingSource{answer: answering(NewSkillReference("a", 1))}
	keys := []string{"agent"}
	src := newSource(t, cs.refs, keys...)
	keys[0] = "other"
	_, w := watchFrom(t, src, t.TempDir(), &WatchSkillsOptions{Debounce: 10 * time.Millisecond})
	tr.fire(t, "agent")
	waitReconciles(t, w, 1)
}

// ---- WriteSkillsFrom ----

func TestWriteSkillsFromMatchesTheList(t *testing.T) {
	_, _ = setup(t)
	SetSkillStore(NewInMemorySkillStore(map[string]any{"a": rawSkill("a", 1, "a"), "b": rawSkill("b", 2, "b")}))
	list, fromSource := t.TempDir(), t.TempDir()
	for _, refs := range [][]SkillReference{
		{NewSkillReference("a", 1), NewSkillReference("b", 2)},
		{NewSkillReference("a", 1)}, // drops b: both forms prune it
		{NewSkillReference("a", 1), NewSkillReference("missing", 1)},
	} {
		want := mustWrite(t, refs, list, nil)
		got, err := WriteSkillsFrom(context.Background(), newSource(t, fixed(refs...)), fromSource, nil)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(actionKinds(got), actionKinds(want)) || got.OK() != want.OK() {
			t.Fatalf("source %v, list %v", actionKinds(got), actionKinds(want))
		}
		for i, a := range want.Actions() {
			if got.Actions()[i].Version() != a.Version() {
				t.Fatalf("%s: version %d, list %d", a.Key(), got.Actions()[i].Version(), a.Version())
			}
		}
	}
}

func TestWriteSkillsFromReadsOnce(t *testing.T) {
	for _, mode := range []OnUnavailable{OnUnavailableKeep, OnUnavailableRaise} {
		_, _ = setup(t)
		SetSkillStore(NewInMemorySkillStore(nil))
		cs := &countingSource{answer: func(context.Context, int) ([]SkillReference, error) { return nil, errStoreDown }}
		_, _ = WriteSkillsFrom(context.Background(), newSource(t, cs.refs), t.TempDir(), &WriteSkillsOptions{OnUnavailable: mode})
		if cs.reads() != 1 {
			t.Fatalf("%s: refs read %d times", mode, cs.reads())
		}
	}
}

// A source that cannot answer is an unavailable retrieval: nothing is written
// or removed, and the manifest keeps its entries.
func TestWriteSkillsFromFailedReadIsUnavailable(t *testing.T) {
	failures := map[string]func(context.Context, int) ([]SkillReference, error){
		"refs returns an error": func(context.Context, int) ([]SkillReference, error) { return nil, errStoreDown },
		// A nil slice is Go's "no answer"; an empty set is a non-nil empty slice.
		"nil answer":                     func(context.Context, int) ([]SkillReference, error) { return nil, nil },
		"an entry with no key":           answering(NewSkillReference("", 1)),
		"one valid, one malformed entry": answering(NewSkillReference("a", 1), NewSkillReference("", 1)),
		"an answer and an error together": func(context.Context, int) ([]SkillReference, error) {
			return []SkillReference{NewSkillReference("a", 1)}, errStoreDown
		},
	}
	for name, answer := range failures {
		for _, mode := range []OnUnavailable{OnUnavailableKeep, OnUnavailableRaise} {
			_, _ = setup(t)
			SetSkillStore(NewInMemorySkillStore(map[string]any{"a": rawSkill("a", 1, "a"), "b": rawSkill("b", 1, "b")}))
			root := t.TempDir()
			mustWrite(t, []SkillKey{"a", "b"}, root, nil)
			r, err := WriteSkillsFrom(context.Background(), newSource(t, (&countingSource{answer: answer}).refs), root, &WriteSkillsOptions{OnUnavailable: mode})
			if mode == OnUnavailableRaise {
				if !errors.Is(err, ErrSkillsUnavailable) || !strings.Contains(err.Error(), "skill source") {
					t.Fatalf("%s, raise: %v", name, err)
				}
			} else if err != nil || r.OK() || !slices.Equal(actionKinds(r), []string{"=error"}) || !strings.Contains(r.Errors()[0].ErrorMessage(), "skill source") {
				t.Fatalf("%s, keep: %v %v %v", name, err, actionKinds(r), r.Errors())
			}
			for _, k := range []string{"a", "b"} {
				if readFile(t, filepath.Join(root, k, "SKILL.md")) != k {
					t.Fatalf("%s, %s: %s changed on disk", name, mode, k)
				}
				if _, ok := manifestEntries(t, root)[k+"/SKILL.md"]; !ok {
					t.Fatalf("%s, %s: manifest entry for %s dropped", name, mode, k)
				}
			}
		}
	}
}

func TestWriteSkillsFromEmptyAnswerPrunes(t *testing.T) {
	_, _ = setup(t)
	SetSkillStore(NewInMemorySkillStore(map[string]any{"a": rawSkill("a", 1, "a")}))
	root := t.TempDir()
	mustWrite(t, []SkillKey{"a"}, root, nil)
	r, err := WriteSkillsFrom(context.Background(), newSource(t, fixed()), root, nil)
	if err != nil || !r.OK() || !slices.Equal(actionKinds(r), []string{"a=removed"}) {
		t.Fatalf("an empty answer: %v %v", err, actionKinds(r))
	}
}

func TestWriteSkillsFromTimeoutBoundsTheSource(t *testing.T) {
	_, _ = setup(t)
	SetSkillStore(NewInMemorySkillStore(map[string]any{"a": rawSkill("a", 1, "a")}))
	zero := time.Duration(0)
	cs := &countingSource{answer: answering(NewSkillReference("a", 1))}
	r, err := WriteSkillsFrom(context.Background(), newSource(t, cs.refs), t.TempDir(), &WriteSkillsOptions{Timeout: &zero})
	if err != nil || cs.reads() != 0 || r.OK() {
		t.Fatalf("timeout 0: err=%v reads=%d %v", err, cs.reads(), actionKinds(r))
	}
	// A source that answers only after the deadline is a failed read: its late
	// answer is not written.
	short := 50 * time.Millisecond
	var hadDeadline bool
	late := &countingSource{answer: func(ctx context.Context, _ int) ([]SkillReference, error) {
		_, hadDeadline = ctx.Deadline()
		<-ctx.Done()
		return []SkillReference{NewSkillReference("a", 1)}, nil
	}}
	root := t.TempDir()
	r, err = WriteSkillsFrom(context.Background(), newSource(t, late.refs), root, &WriteSkillsOptions{Timeout: &short})
	if err != nil || r.OK() || !slices.Equal(actionKinds(r), []string{"=error"}) || !hadDeadline {
		t.Fatalf("late source: err=%v %v deadline=%v", err, actionKinds(r), hadDeadline)
	}
	if _, ok := actionFor(r, "a"); ok {
		t.Fatal("the late answer was written")
	}
}

// ---- ConfigSkills ----

func TestConfigSkillsClassification(t *testing.T) {
	const key = "agent-config"
	value := func(v ldvalue.Value, err error) func(string, ldvalue.Value) (ldvalue.Value, error) {
		return func(string, ldvalue.Value) (ldvalue.Value, error) { return v, err }
	}
	// Failed reads first: these are the rows that must never read as [].
	failed := []struct {
		name  string
		value func(string, ldvalue.Value) (ldvalue.Value, error) // nil: no client
	}{
		{"no initialized AI client", nil},
		{"variation returns an error", value(ldvalue.Null(), errors.New("evaluation failed"))},
		{"the default (null)", func(_ string, def ldvalue.Value) (ldvalue.Value, error) { return def, nil }},
		{"a string", value(ldvalue.String("x"), nil)},
		{"an array", value(ldvalue.ArrayOf(), nil)},
		{"a bool", value(ldvalue.Bool(false), nil)},
		{"no _ldMeta", value(ldvalue.CopyArbitraryValue(map[string]any{"model": map[string]any{}}), nil)},
		{"_ldMeta not an object", value(ldvalue.CopyArbitraryValue(map[string]any{"_ldMeta": true}), nil)},
		{"enabled not a boolean", value(configValue("yes", nil), nil)},
		{"enabled missing", value(ldvalue.CopyArbitraryValue(map[string]any{"_ldMeta": map[string]any{}}), nil)},
		{"skills not a list", value(configValue(true, "nope"), nil)},
		{"a skills entry without a string key", value(configValue(true, []any{map[string]any{"key": 7, "version": 1}}), nil)},
	}
	// The message names the cause as well as the key.
	causes := map[string]string{"variation returns an error": "evaluation failed"}
	for _, row := range failed {
		_, _ = setup(t)
		if row.value == nil {
			noClient(t)
		} else {
			installClient(t, &plainLD{value: row.value})
		}
		refs, err := readSource(t, ConfigSkills(key, ldcontext.New("u")))
		if err == nil || !strings.Contains(err.Error(), key) {
			t.Errorf("%s: want a failed read naming %q, got %v %v", row.name, key, refPairs(refs), err)
		} else if cause := causes[row.name]; !strings.Contains(err.Error(), cause) {
			t.Errorf("%s: the failed read does not name its cause %q: %v", row.name, cause, err)
		}
	}
	answers := []struct {
		name  string
		value ldvalue.Value
		want  []string
	}{
		{"disabled", configValue(false, skillsList("a", 1)), nil},
		{"enabled, no skills", configValue(true, nil), nil},
		{"enabled, skills", configValue(true, skillsList("a", 2, "b", 1)), []string{"a@2", "b@1"}},
	}
	for _, row := range answers {
		_, _ = setup(t)
		installClient(t, &plainLD{value: value(row.value, nil)})
		refs, err := readSource(t, ConfigSkills(key, ldcontext.New("u")))
		if err != nil || refs == nil || !slices.Equal(refPairs(refs), row.want) {
			t.Errorf("%s: want %v (non-nil), got %v (nil=%v) %v", row.name, row.want, refPairs(refs), refs == nil, err)
		}
	}
}

func TestConfigSkillsEvaluatesWithANullDefaultAndNeverInitializes(t *testing.T) {
	_, _ = setup(t)
	noClient(t)
	src := ConfigSkills("agent-config", ldcontext.New("u"))
	if _, err := readSource(t, src); err == nil {
		t.Fatal("no client must be a failed read")
	}
	if _, err := aiserver.GetClient(); !errors.Is(err, aiserver.ErrNotInitialized) {
		t.Fatalf("a read initialized the client: %v", err)
	}
	fake := &plainLD{value: func(string, ldvalue.Value) (ldvalue.Value, error) { return configValue(true, nil), nil }}
	installClient(t, fake)
	userCtx := ldcontext.New("user-7")
	if _, err := readSource(t, ConfigSkills("agent-config", userCtx)); err != nil {
		t.Fatal(err)
	}
	calls := fake.variationCalls()
	if len(calls) != 1 || calls[0].key != "agent-config" || !calls[0].def.IsNull() || calls[0].ctx.Key() != "user-7" {
		t.Fatalf("variation calls %+v", calls)
	}
}

func TestConfigSkillsEvaluatesAfresh(t *testing.T) {
	_, _ = setup(t)
	fake := &plainLD{}
	installClient(t, fake)
	src := ConfigSkills("agent-config", ldcontext.New("u"))
	fake.setValue(func(string, ldvalue.Value) (ldvalue.Value, error) { return configValue(true, skillsList("a", 1)), nil })
	first, err1 := readSource(t, src)
	fake.setValue(func(string, ldvalue.Value) (ldvalue.Value, error) { return configValue(true, skillsList("a", 2)), nil })
	second, err2 := readSource(t, src)
	if err1 != nil || err2 != nil || !slices.Equal(refPairs(first), []string{"a@1"}) || !slices.Equal(refPairs(second), []string{"a@2"}) {
		t.Fatalf("reads %v %v (%v %v)", refPairs(first), refPairs(second), err1, err2)
	}
}

// A read sends no AI telemetry and no skills signal; the one usage event is
// sent when ConfigSkills is called.
func TestConfigSkillsReadSendsNoTelemetry(t *testing.T) {
	rec, _ := setup(t)
	fake := &plainLD{value: func(string, ldvalue.Value) (ldvalue.Value, error) {
		return configValue(true, skillsList("a", 1)), nil
	}}
	installClient(t, fake)
	before := len(fake.tracked()) // InitWithClient's own events
	since := func() []string { return fake.tracked()[before:] }
	src := ConfigSkills("agent-config", ldcontext.New("u"))
	if got := since(); !slices.Equal(got, []string{"$ld:ai:sdk:usage client.configSkills"}) {
		t.Fatalf("at call time: %v", got)
	}
	for range 3 {
		if _, err := readSource(t, src); err != nil {
			t.Fatal(err)
		}
	}
	if got := since(); !slices.Equal(got, []string{"$ld:ai:sdk:usage client.configSkills"}) {
		t.Fatalf("after three reads: %v", got)
	}
	if s := rec.all(); len(s) != 0 {
		t.Fatalf("a read emitted skills signals: %v", s)
	}
}

// ---- WatchSkillsFrom ----

func TestWatchSkillsFromReadsOnEveryTrigger(t *testing.T) {
	_, _ = setup(t)
	tr := &fakeTracker{}
	installClient(t, &trackedLD{tracker: tr})
	store := watchStore(map[string]any{"a": rawSkill("a", 1, "a")})
	entered, release := make(chan struct{}), make(chan struct{})
	var releaseOnce sync.Once
	cs := &countingSource{answer: func(_ context.Context, n int) ([]SkillReference, error) {
		if n == 4 {
			close(entered)
			<-release
		}
		return []SkillReference{NewSkillReference("a", 1)}, nil
	}}
	_, w := watchFrom(t, newSource(t, cs.refs, "agent"), t.TempDir(), &WatchSkillsOptions{Debounce: 10 * time.Millisecond})
	t.Cleanup(func() { releaseOnce.Do(func() { close(release) }) }) // runs before w.Close
	if cs.reads() != 1 {
		t.Fatalf("initial reads %d", cs.reads())
	}
	if err := store.Put(rawSkill("b", 1, "b")); err != nil {
		t.Fatal(err)
	}
	waitReconciles(t, w, 1)
	if cs.reads() != 2 {
		t.Fatalf("a store change: reads %d", cs.reads())
	}
	if n := tr.fire(t, "agent"); n != 1 {
		t.Fatalf("%d config listeners", n)
	}
	waitReconciles(t, w, 2)
	if cs.reads() != 3 {
		t.Fatalf("a config change: reads %d", cs.reads())
	}
	// An unrelated key does not wake the watcher. This is checked on the wake
	// signal itself, while a read holds the watcher's goroutine: nothing can
	// take the signal, and the debounce cannot merge it into a later one.
	tr.fire(t, "agent")
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("the config change never reached the source")
	}
	// The listener takes the second send only after it has handled the first.
	tr.fire(t, "unrelated")
	tr.fire(t, "unrelated")
	if len(w.wake) != 0 {
		t.Fatal("an unrelated key woke the watcher")
	}
	// The probe sees a real wake: a followed key raises the same signal.
	tr.fire(t, "agent")
	tr.fire(t, "agent")
	if len(w.wake) != 1 {
		t.Fatal("a followed key changed during a read did not wake the watcher")
	}
	releaseOnce.Do(func() { close(release) })
	waitReconciles(t, w, 4)
	if cs.reads() != 5 {
		t.Fatalf("one wake after the held read: reads %d", cs.reads())
	}
}

// The config listener only signals: it takes every change while a read is
// still running on the watcher's goroutine.
func TestWatchSkillsFromConfigListenerOnlySignals(t *testing.T) {
	_, _ = setup(t)
	tr := &fakeTracker{}
	installClient(t, &trackedLD{tracker: tr})
	watchStore(map[string]any{"a": rawSkill("a", 1, "a")})
	entered, release := make(chan struct{}), make(chan struct{})
	cs := &countingSource{answer: func(_ context.Context, n int) ([]SkillReference, error) {
		if n == 2 {
			close(entered)
			<-release
		}
		return []SkillReference{NewSkillReference("a", 1)}, nil
	}}
	_, w := watchFrom(t, newSource(t, cs.refs, "agent"), t.TempDir(), &WatchSkillsOptions{Debounce: 10 * time.Millisecond})
	tr.fire(t, "agent")
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("the config change never reached the source")
	}
	for range 3 {
		tr.fire(t, "agent") // fails the test if the listener is stuck in refs
	}
	close(release)
	waitReconciles(t, w, 2)
	time.Sleep(100 * time.Millisecond)
	if w.Reconciles() != 2 || cs.reads() != 3 {
		t.Fatalf("changes during a reconcile schedule exactly one more: reconciles %d, reads %d", w.Reconciles(), cs.reads())
	}
}

func TestWatchSkillsFromRegistersTheConfigListenerFirst(t *testing.T) {
	_, _ = setup(t)
	tr := &fakeTracker{}
	installClient(t, &trackedLD{tracker: tr})
	watchStore(map[string]any{"a": rawSkill("a", 1, "a")})
	listening := -1
	cs := &countingSource{answer: func(_ context.Context, n int) ([]SkillReference, error) {
		if n == 1 {
			listening = tr.fire(t, "agent") // a change landing during the initial reconcile
		}
		return []SkillReference{NewSkillReference("a", 1)}, nil
	}}
	_, w := watchFrom(t, newSource(t, cs.refs, "agent"), t.TempDir(), &WatchSkillsOptions{Debounce: 10 * time.Millisecond})
	if listening != 1 {
		t.Fatalf("%d config listeners during the initial reconcile", listening)
	}
	waitReconciles(t, w, 1)
}

func TestWatchSkillsFromNeedsAChangeSignal(t *testing.T) {
	cases := map[string]func(t *testing.T) *fakeTracker{
		"no AI client":                func(t *testing.T) *fakeTracker { noClient(t); return nil },
		"a client with no change API": func(t *testing.T) *fakeTracker { installClient(t, &plainLD{}); return nil },
		"a client that is offline": func(t *testing.T) *fakeTracker {
			tr := &fakeTracker{}
			installClient(t, &trackedLD{tracker: tr, offline: true})
			return tr
		},
	}
	for name, arrange := range cases {
		_, _ = setup(t)
		tr := arrange(t)
		store := watchStore(nil)
		_, w, err := WatchSkillsFrom(context.Background(), newSource(t, fixed(), "agent"), t.TempDir(), nil)
		if err == nil || w != nil || !strings.Contains(err.Error(), "Notify") || !strings.Contains(err.Error(), "LaunchDarkly client") ||
			!strings.Contains(err.Error(), "configKeys") {
			t.Errorf("%s: %v", name, err)
		}
		if store.removed != 1 {
			t.Errorf("%s: the store listener was not detached", name)
		}
		if tr != nil {
			if added, removed := tr.counts(); added != removed {
				t.Errorf("%s: config listener left registered (%d added, %d removed)", name, added, removed)
			}
		}
	}
	// No config keys probes nothing.
	_, _ = setup(t)
	noClient(t)
	watchStore(nil)
	watchFrom(t, newSource(t, fixed()), t.TempDir(), nil)
}

func TestFixedListAndStarNeverSubscribeToConfigChanges(t *testing.T) {
	watchBoth := func(t *testing.T, client string) {
		t.Helper()
		_, w1, err1 := WatchSkills(context.Background(), []SkillKey{"a"}, t.TempDir(), nil)
		_, w2, err2 := WatchAllSkills(context.Background(), t.TempDir(), nil)
		if err1 != nil || err2 != nil {
			t.Fatal(client, err1, err2)
		}
		w1.Close()
		w2.Close()
	}
	_, _ = setup(t)
	tr := &fakeTracker{}
	installClient(t, &trackedLD{tracker: tr})
	watchStore(map[string]any{"a": rawSkill("a", 1, "a")})
	watchBoth(t, "a client with a change API")
	if added, _ := tr.counts(); added != 0 {
		t.Fatalf("%d config listeners registered", added)
	}
	// Neither form needs a change signal, so a client without one is fine.
	_, _ = setup(t)
	installClient(t, &plainLD{})
	watchStore(map[string]any{"a": rawSkill("a", 1, "a")})
	watchBoth(t, "a client with no change API")
}

// Notify goes through the same trailing debounce: three calls spread over
// less than one window make one reconcile.
func TestNotifyGoesThroughTheDebounce(t *testing.T) {
	_, _ = setup(t)
	noClient(t)
	watchStore(map[string]any{"a": rawSkill("a", 1, "a")})
	cs := &countingSource{answer: answering(NewSkillReference("a", 1))}
	_, w := watchFrom(t, newSource(t, cs.refs), t.TempDir(), &WatchSkillsOptions{Debounce: 150 * time.Millisecond})
	for i := range 3 {
		if i > 0 {
			time.Sleep(40 * time.Millisecond)
		}
		w.Notify()
	}
	if w.Reconciles() != 0 {
		t.Fatal("reconciled inside the debounce window")
	}
	waitReconciles(t, w, 1)
	time.Sleep(300 * time.Millisecond)
	if w.Reconciles() != 1 || cs.reads() != 2 {
		t.Fatalf("reconciles %d, reads %d", w.Reconciles(), cs.reads())
	}
	w.Close()
	w.Notify() // after Close: a no-op, not a panic
}

func TestTheDebounceSpansEveryTriggerKind(t *testing.T) {
	_, _ = setup(t)
	tr := &fakeTracker{}
	installClient(t, &trackedLD{tracker: tr})
	store := watchStore(map[string]any{"a": rawSkill("a", 1, "a")})
	cs := &countingSource{answer: answering(NewSkillReference("a", 1))}
	_, w := watchFrom(t, newSource(t, cs.refs, "agent"), t.TempDir(), &WatchSkillsOptions{Debounce: 150 * time.Millisecond})
	if err := store.Put(rawSkill("b", 1, "b")); err != nil {
		t.Fatal(err)
	}
	time.Sleep(40 * time.Millisecond)
	tr.fire(t, "agent")
	time.Sleep(40 * time.Millisecond)
	w.Notify()
	waitReconciles(t, w, 1)
	time.Sleep(300 * time.Millisecond)
	if w.Reconciles() != 1 || cs.reads() != 2 {
		t.Fatalf("reconciles %d, reads %d", w.Reconciles(), cs.reads())
	}
}

func TestWatchSkillsFromCloseDetachesTheConfigListener(t *testing.T) {
	_, _ = setup(t)
	tr := &fakeTracker{}
	installClient(t, &trackedLD{tracker: tr})
	store := watchStore(map[string]any{"a": rawSkill("a", 1, "a")})
	cs := &countingSource{answer: answering(NewSkillReference("a", 1))}
	_, w := watchFrom(t, newSource(t, cs.refs, "agent"), t.TempDir(), &WatchSkillsOptions{Debounce: 10 * time.Millisecond})
	w.Close()
	w.Close()
	if added, removed := tr.counts(); added != 1 || removed != 1 {
		t.Fatalf("config listeners added %d, removed %d", added, removed)
	}
	if store.removed != 1 {
		t.Fatalf("store RemoveListener called %d times", store.removed)
	}
	if n := tr.fire(t, "agent"); n != 0 {
		t.Fatalf("%d config listeners still take changes", n)
	}
	if cs.reads() != 1 || w.Reconciles() != 0 {
		t.Fatalf("after Close: reads %d, reconciles %d", cs.reads(), w.Reconciles())
	}
}

// A version bump converges in either arrival order, by overwriting: no
// reconcile reports removed.
func TestWatchSkillsFromVersionBumpConverges(t *testing.T) {
	type endState struct{ content, sha string }
	run := func(t *testing.T, configFirst bool) endState {
		_, _ = setup(t)
		tr := &fakeTracker{}
		installClient(t, &trackedLD{tracker: tr})
		store := watchStore(map[string]any{"v2": rawSkill("pdf-extraction", 2, "v2 body")})
		var mu sync.Mutex
		version := 2
		cs := &countingSource{answer: func(context.Context, int) ([]SkillReference, error) {
			mu.Lock()
			defer mu.Unlock()
			return []SkillReference{NewSkillReference("pdf-extraction", version)}, nil
		}}
		setVersion := func(v int) { mu.Lock(); version = v; mu.Unlock() }
		log := &reportLog{}
		root := t.TempDir()
		initial, w := watchFrom(t, newSource(t, cs.refs, "agent"), root, &WatchSkillsOptions{Debounce: 20 * time.Millisecond, OnReconcile: log.add})
		if !slices.Equal(actionKinds(initial), []string{"pdf-extraction=written"}) {
			t.Fatalf("initial %v", actionKinds(initial))
		}
		if n := len(log.all()); n != 0 {
			t.Fatalf("the initial report went to OnReconcile (%d reports)", n)
		}
		file := filepath.Join(root, "pdf-extraction", "SKILL.md")
		settle := func(want int) {
			waitReconciles(t, w, want)
			time.Sleep(80 * time.Millisecond)
		}
		if configFirst {
			setVersion(3)
			tr.fire(t, "agent")
		} else {
			if err := store.Put(rawSkill("pdf-extraction", 3, "v3 body")); err != nil {
				t.Fatal(err)
			}
			revoke(store.InMemorySkillStore, "pdf-extraction", 2)
		}
		settle(1)
		mid := log.last(t)
		if a, _ := actionFor(mid, "pdf-extraction"); a.Action() != ActionError {
			t.Fatalf("first change: %v", actionKinds(mid))
		}
		if readFile(t, file) != "v2 body" || manifestEntries(t, root)["pdf-extraction/SKILL.md"].(map[string]any)["version"] != float64(2) {
			t.Fatal("v2 left disk before v3 could replace it")
		}
		before := w.Reconciles()
		if configFirst {
			if err := store.Put(rawSkill("pdf-extraction", 3, "v3 body")); err != nil {
				t.Fatal(err)
			}
		} else {
			setVersion(3)
			tr.fire(t, "agent")
		}
		settle(before + 1)
		end := log.last(t)
		if a, _ := actionFor(end, "pdf-extraction"); a.Action() != ActionUpdated || a.Version() != 3 || !end.OK() {
			t.Fatalf("second change: %v", actionKinds(end))
		}
		for _, r := range log.all() {
			if a, _ := actionFor(r, "pdf-extraction"); a.Action() == ActionRemoved {
				t.Fatal("the bump pruned before it wrote")
			}
		}
		entry := manifestEntries(t, root)["pdf-extraction/SKILL.md"].(map[string]any)
		if entry["version"] != float64(3) {
			t.Fatalf("manifest %v", entry)
		}
		return endState{readFile(t, file), entry["sha256"].(string)}
	}
	configFirst := run(t, true)
	storeFirst := run(t, false)
	if configFirst != storeFirst || configFirst.content != "v3 body" {
		t.Fatalf("end states differ: %+v vs %+v", configFirst, storeFirst)
	}
}

// Dropping a reference prunes on the next run whose retrieval is available.
func TestDroppedReferencePrunesWhenRetrievalIsAvailable(t *testing.T) {
	_, _ = setup(t)
	mem := NewInMemorySkillStore(map[string]any{"a": rawSkill("a", 1, "a"), "b": rawSkill("b", 1, "b"), "c": rawSkill("c", 1, "c")})
	probe := &probeStore{SkillStore: mem, ready: true}
	SetSkillStore(probe)
	var mu sync.Mutex
	// c is written first, then leaves the store: it is on disk and in the
	// manifest, but absent when it is requested again.
	answer := []SkillReference{NewSkillReference("a", 1), NewSkillReference("b", 1), NewSkillReference("c", 1)}
	src := newSource(t, func(context.Context) ([]SkillReference, error) {
		mu.Lock()
		defer mu.Unlock()
		return slices.Clone(answer), nil
	})
	set := func(refs ...SkillReference) { mu.Lock(); answer = refs; mu.Unlock() }
	write := func(root string) ReconcileReport {
		r, err := WriteSkillsFrom(context.Background(), src, root, nil)
		if err != nil {
			t.Fatal(err)
		}
		return r
	}
	clean, withAbsent, unavailable := t.TempDir(), t.TempDir(), t.TempDir()
	write(clean)
	write(withAbsent)
	write(unavailable)
	revoke(mem, "c", 1)
	set(NewSkillReference("a", 1))
	if r := write(clean); !slices.Contains(actionKinds(r), "b=removed") {
		t.Fatalf("clean: %v", actionKinds(r))
	}
	set(NewSkillReference("a", 1), NewSkillReference("c", 1))
	if r := write(withAbsent); !slices.Contains(actionKinds(r), "b=removed") || !slices.Contains(actionKinds(r), "c=error") {
		t.Fatalf("with an absent reference: %v", actionKinds(r))
	}
	if readFile(t, filepath.Join(withAbsent, "c", "SKILL.md")) != "c" {
		t.Fatal("the absent skill's file was not kept")
	}
	if _, ok := manifestEntries(t, withAbsent)["c/SKILL.md"]; !ok {
		t.Fatal("the absent skill's manifest entry was dropped")
	}
	set(NewSkillReference("a", 1))
	probe.ready = false
	if r := write(unavailable); slices.Contains(actionKinds(r), "b=removed") || readFile(t, filepath.Join(unavailable, "b", "SKILL.md")) != "b" {
		t.Fatalf("pruned while unavailable: %v", actionKinds(r))
	}
	probe.ready = true
	if r := write(unavailable); !slices.Contains(actionKinds(r), "b=removed") {
		t.Fatalf("next available run: %v", actionKinds(r))
	}
}

// A source that answers, then fails, then answers leaves the first run's
// files alone through the failure, and the watcher carries on.
func TestWatchSkillsFromSurvivesAFailedRead(t *testing.T) {
	_, _ = setup(t)
	noClient(t)
	watchStore(map[string]any{"a": rawSkill("a", 1, "a"), "b": rawSkill("b", 1, "b")})
	cs := &countingSource{answer: func(_ context.Context, n int) ([]SkillReference, error) {
		switch n {
		case 1:
			return []SkillReference{NewSkillReference("a", 1), NewSkillReference("b", 1)}, nil
		case 2:
			return nil, errStoreDown
		}
		return []SkillReference{NewSkillReference("a", 1)}, nil
	}}
	log := &reportLog{}
	root := t.TempDir()
	_, w := watchFrom(t, newSource(t, cs.refs), root, &WatchSkillsOptions{Debounce: 10 * time.Millisecond, OnReconcile: log.add})
	w.Notify()
	waitReconciles(t, w, 1)
	if r := log.last(t); r.OK() || !slices.Equal(actionKinds(r), []string{"=error"}) {
		t.Fatalf("failed read: %v", actionKinds(r))
	}
	for _, k := range []string{"a", "b"} {
		if readFile(t, filepath.Join(root, k, "SKILL.md")) != k {
			t.Fatalf("%s changed during the failure", k)
		}
	}
	w.Notify()
	waitReconciles(t, w, 2)
	if r := log.last(t); !slices.Contains(actionKinds(r), "b=removed") {
		t.Fatalf("after recovery: %v", actionKinds(r))
	}
}

// Under OnUnavailableRaise a later failed read is logged, not returned to
// anyone, and the next trigger reads the source again.
func TestWatchSkillsFromLogsALaterRaise(t *testing.T) {
	_, logs := setup(t)
	noClient(t)
	watchStore(map[string]any{"a": rawSkill("a", 1, "a"), "b": rawSkill("b", 1, "b")})
	cs := &countingSource{answer: func(_ context.Context, n int) ([]SkillReference, error) {
		switch n {
		case 1:
			return []SkillReference{NewSkillReference("a", 1), NewSkillReference("b", 1)}, nil
		case 2:
			return nil, errStoreDown
		}
		return []SkillReference{NewSkillReference("a", 1)}, nil
	}}
	log := &reportLog{}
	root := t.TempDir()
	_, w := watchFrom(t, newSource(t, cs.refs), root, &WatchSkillsOptions{Debounce: 10 * time.Millisecond, OnReconcile: log.add,
		WriteSkillsOptions: WriteSkillsOptions{OnUnavailable: OnUnavailableRaise}})
	errorLogged := func() bool {
		for _, e := range logs.all() {
			if e.level == slog.LevelError {
				return true
			}
		}
		return false
	}
	w.Notify()
	eventually(t, "the raised failure to be logged", errorLogged)
	if cs.reads() != 2 {
		t.Fatalf("reads %d", cs.reads())
	}
	for _, k := range []string{"a", "b"} {
		if readFile(t, filepath.Join(root, k, "SKILL.md")) != k {
			t.Fatalf("%s changed during the failure", k)
		}
	}
	w.Notify()
	waitReconciles(t, w, 1)
	if cs.reads() != 3 {
		t.Fatalf("the next trigger: reads %d", cs.reads())
	}
	if r := log.last(t); !slices.Contains(actionKinds(r), "b=removed") {
		t.Fatalf("after recovery: %v", actionKinds(r))
	}
}

func TestWatchSkillsFromInitialFailedRead(t *testing.T) {
	_, _ = setup(t)
	tr := &fakeTracker{}
	installClient(t, &trackedLD{tracker: tr})
	store := watchStore(nil)
	failing := func(context.Context) ([]SkillReference, error) { return nil, errStoreDown }
	_, w, err := WatchSkillsFrom(context.Background(), newSource(t, failing, "agent"), t.TempDir(),
		&WatchSkillsOptions{WriteSkillsOptions: WriteSkillsOptions{OnUnavailable: OnUnavailableRaise}})
	if !errors.Is(err, ErrSkillsUnavailable) || w != nil {
		t.Fatalf("raise: %v", err)
	}
	if added, removed := tr.counts(); store.removed != 1 || added != 1 || removed != 1 {
		t.Fatalf("listeners not detached: store %d, config %d/%d", store.removed, added, removed)
	}
	report, _ := watchFrom(t, newSource(t, failing, "agent"), t.TempDir(), nil)
	if report.OK() || !slices.Equal(actionKinds(report), []string{"=error"}) {
		t.Fatalf("keep: %v", actionKinds(report))
	}
}

// One version bump end to end through the base SDK's own test data source,
// so the change signal is the real one, not a fake's idea of it.
func TestConfigSkillsFollowsTheRealBaseSDK(t *testing.T) {
	_, _ = setup(t)
	td := ldtestdata.DataSource()
	td.Update(td.Flag("agent-config").ValueForAll(configValue(true, skillsList("pdf-extraction", 2))))
	client, err := ld.MakeCustomClient("sdk-test", ld.Config{DataSource: td, Events: ldcomponents.NoEvents(), DiagnosticOptOut: true,
		Logging: ldcomponents.NoLogging()}, 5*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = client.Close() })
	installClient(t, client)
	watchStore(map[string]any{"v2": rawSkill("pdf-extraction", 2, "v2 body"), "v3": rawSkill("pdf-extraction", 3, "v3 body")})
	root := t.TempDir()
	log := &reportLog{}
	_, w := watchFrom(t, ConfigSkills("agent-config", ldcontext.New("u")), root, &WatchSkillsOptions{Debounce: 20 * time.Millisecond, OnReconcile: log.add})
	if readFile(t, filepath.Join(root, "pdf-extraction", "SKILL.md")) != "v2 body" {
		t.Fatal("initial reconcile did not write v2")
	}
	td.Update(td.Flag("agent-config").ValueForAll(configValue(true, skillsList("pdf-extraction", 3))))
	waitReconciles(t, w, 1)
	if a, _ := actionFor(log.last(t), "pdf-extraction"); a.Action() != ActionUpdated || a.Version() != 3 {
		t.Fatalf("after the config change: %v", actionKinds(log.last(t)))
	}
	if readFile(t, filepath.Join(root, "pdf-extraction", "SKILL.md")) != "v3 body" {
		t.Fatal("v3 not on disk")
	}
}
