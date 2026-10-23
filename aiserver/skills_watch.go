package aiserver

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// defaultDebounce is how long the watcher waits after the last change before
// reconciling. Unexported, as in Python; TypeScript publishes its own default.
const defaultDebounce = 500 * time.Millisecond

// watcherCloseTimeout bounds how long Close waits for an in-flight reconcile.
const watcherCloseTimeout = 15 * time.Second

// WatchSkillsOptions are WriteSkills' options plus the watcher's own.
type WatchSkillsOptions struct {
	// WriteSkillsOptions are passed to every reconcile unchanged.
	WriteSkillsOptions
	// Debounce is how long to wait after the last delivery change before
	// reconciling, so a burst of changes triggers one reconcile. 0 means the
	// default, 500ms; negative is an ErrInvalidArgument.
	Debounce time.Duration
	// OnReconcile receives each delivery-triggered reconcile's report, not
	// the initial one (WatchSkills returns that). A panic in it is logged.
	OnReconcile func(ReconcileReport)
}

// SkillWatcher is a running watch, returned by WatchSkills. Stop it with
// Close, or by cancelling the context WatchSkills was given.
//
// One watcher per root: interleaved reconciles of one root can lose manifest
// entries. The watcher serializes its own reconciles only.
type SkillWatcher struct {
	sel         selection
	root        string
	opts        WriteSkillsOptions
	debounce    time.Duration
	onReconcile func(ReconcileReport)

	store    SkillStore
	listener *Listener

	wake       chan struct{}
	stop       chan struct{}
	stopOnce   sync.Once
	done       chan struct{}
	started    atomic.Bool
	reconciles atomic.Int64

	mu         sync.Mutex
	registered bool
}

// WatchSkills reconciles now, then again whenever delivery changes, so a
// revocation delivered over a live stream prunes the skill's files within the
// debounce window. The initial reconcile runs on the calling goroutine and its
// report is returned, so a bad root fails here exactly as with WriteSkills.
//
// It needs a store implementing ListenerStore: with no store it returns
// ErrNoSkillStore, and with a store that cannot report changes it returns
// ErrStoreNotWatchable rather than degrading to a one-shot reconcile.
//
// The watcher runs on its own goroutine until Close or until ctx is done.
func WatchSkills[I SkillInput](ctx context.Context, skills []I, root string, opts *WatchSkillsOptions) (ReconcileReport, *SkillWatcher, error) {
	items := make([]SkillInput, len(skills))
	for i, s := range skills {
		items[i] = s
	}
	return watchSkills(ctx, selection{items: items}, root, opts)
}

// WatchAllSkills is WatchSkills over every skill the store holds (the "*"
// selection); with it, a skill revoked in LaunchDarkly is removed from disk.
func WatchAllSkills(ctx context.Context, root string, opts *WatchSkillsOptions) (ReconcileReport, *SkillWatcher, error) {
	return watchSkills(ctx, selection{all: true}, root, opts)
}

func watchSkills(ctx context.Context, sel selection, root string, opts *WatchSkillsOptions) (ReconcileReport, *SkillWatcher, error) {
	if opts == nil {
		opts = &WatchSkillsOptions{}
	}
	s := currentStore()
	if s == nil {
		return ReconcileReport{}, nil, fmt.Errorf("WatchSkills needs a configured skill store: %w", ErrNoSkillStore)
	}
	ls, ok := s.(ListenerStore)
	if !ok {
		return ReconcileReport{}, nil, fmt.Errorf("%w: WatchSkills needs a store implementing AddListener, and %T does not, "+
			"so delivery changes cannot be observed. Use WriteSkills for a one-shot reconcile, or configure a "+
			"delivery-backed store (FDv2SkillStore)", ErrStoreNotWatchable, s)
	}
	if opts.Debounce < 0 {
		return ReconcileReport{}, nil, invalidArg("debounce must be a non-negative duration, got %v", opts.Debounce)
	}
	if _, err := parseWriteOptions(&opts.WriteSkillsOptions); err != nil {
		return ReconcileReport{}, nil, err
	}
	debounce := opts.Debounce
	if debounce == 0 {
		debounce = defaultDebounce
	}
	w := &SkillWatcher{
		sel: sel, root: root, opts: opts.WriteSkillsOptions, debounce: debounce, onReconcile: opts.OnReconcile,
		store: s, wake: make(chan struct{}, 1), stop: make(chan struct{}), done: make(chan struct{}),
	}
	w.listener = NewListener(func(any) { w.notify() })
	// Registered before the initial reconcile, so a change delivered after its
	// store snapshot is still seen (nothing re-reconciles on a timer).
	if err := ls.AddListener(skillObjectKind, w.listener); err != nil {
		return ReconcileReport{}, nil, fmt.Errorf("%w: %w", ErrStoreNotWatchable, err)
	}
	w.registered = true
	report, err := writeSkills(sel, root, &w.opts)
	if err != nil {
		w.detach()
		close(w.done)
		return ReconcileReport{}, nil, err
	}
	w.started.Store(true)
	go w.run(ctx)
	return report, w, nil
}

// notify records that something changed; it runs nothing, since it is called
// on the store's delivery goroutine.
func (w *SkillWatcher) notify() {
	select {
	case w.wake <- struct{}{}:
	default:
	}
}

func (w *SkillWatcher) run(ctx context.Context) {
	defer close(w.done)
	defer w.detach()
	for {
		select {
		case <-ctx.Done():
			return
		case <-w.stop:
			return
		case <-w.wake:
		}
		// Trailing debounce: every change inside the window restarts it.
		timer := time.NewTimer(w.debounce)
		for waiting := true; waiting; {
			select {
			case <-ctx.Done():
				timer.Stop()
				return
			case <-w.stop:
				timer.Stop()
				return
			case <-w.wake:
				timer.Reset(w.debounce)
			case <-timer.C:
				waiting = false
			}
		}
		w.reconcileOnce()
	}
}

// reconcileOnce never lets a failure stop the watcher: one that died on a bad
// run would silently stop pruning.
func (w *SkillWatcher) reconcileOnce() {
	defer func() {
		if r := recover(); r != nil {
			currentLogger().Error("a skill re-reconcile panicked; the watcher continues", "panic", r)
		}
	}()
	report, err := writeSkills(w.sel, w.root, &w.opts)
	if err != nil {
		currentLogger().Error("a skill re-reconcile failed; the watcher continues", "error", err)
		return
	}
	w.reconciles.Add(1)
	if w.onReconcile != nil {
		w.onReconcile(report)
	}
}

// Reconciles is the number of delivery-triggered reconciles completed,
// excluding the initial one.
func (w *SkillWatcher) Reconciles() int { return int(w.reconciles.Load()) }

// Close detaches from the store (through RemoveListener when the store has
// it) and waits for an in-flight reconcile to finish rather than interrupting
// it mid-write. Idempotent; leaves files on disk as they are.
func (w *SkillWatcher) Close() {
	w.detach()
	w.stopOnce.Do(func() { close(w.stop) })
	if !w.started.Load() {
		return
	}
	select {
	case <-w.done:
	case <-time.After(watcherCloseTimeout):
		currentLogger().Warn("a skill watcher did not stop within its close timeout")
	}
}

func (w *SkillWatcher) detach() {
	w.mu.Lock()
	if !w.registered {
		w.mu.Unlock()
		return
	}
	w.registered = false
	w.mu.Unlock()
	if r, ok := w.store.(ListenerRemover); ok {
		r.RemoveListener(skillObjectKind, w.listener)
	}
}
