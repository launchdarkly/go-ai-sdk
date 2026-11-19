package skills

import (
	"context"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/launchdarkly/go-sdk-common/v3/ldvalue"
	"github.com/launchdarkly/go-server-sdk/v7/interfaces"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/handlerkit"
)

// ErrInvalidSkillSource is returned for a SkillSource that cannot be used: a
// nil refs function, an empty config key, or the zero SkillSource. It wraps
// ErrInvalidArgument.
var ErrInvalidSkillSource = fmt.Errorf("%w: invalid skill source", ErrInvalidArgument)

// SkillSource is a requested set of skills that is read when it is needed
// rather than fixed when the call is made. WriteSkillsFrom reads it once;
// WatchSkillsFrom reads it on every trigger. Build one with NewSkillSource, or
// with ConfigSkills for the skills one AI Config pins for one context.
//
// The zero SkillSource is invalid: every call refuses it.
type SkillSource struct {
	refs       func(ctx context.Context) ([]SkillReference, error)
	configKeys []string
	invalid    string // why ConfigSkills could not build a source
}

// NewSkillSource builds a SkillSource from refs, which returns the current
// references, and the AI Config keys whose changes should make a watcher read
// it again.
//
// refs is given a context carrying the reconcile's deadline and must honour
// it. A nil slice or an error from refs is a failed read, which changes
// nothing on disk; the empty set is []SkillReference{}, which prunes every
// managed skill. An entry with an invalid key or a version below 1 fails the
// whole read. A nil refs or a blank config key returns ErrInvalidSkillSource.
func NewSkillSource(refs func(ctx context.Context) ([]SkillReference, error), configKeys ...string) (SkillSource, error) {
	if refs == nil {
		return SkillSource{}, fmt.Errorf("%w: NewSkillSource was given a nil refs function", ErrInvalidSkillSource)
	}
	for i, k := range configKeys {
		if strings.TrimSpace(k) == "" {
			return SkillSource{}, fmt.Errorf("%w: config key %d is blank", ErrInvalidSkillSource, i)
		}
	}
	return SkillSource{refs: refs, configKeys: slices.Clone(configKeys)}, nil
}

// ConfigSkills is the SkillSource for the skills the AI Config key pins for
// ldctx, evaluated on every read with the configured AI client.
//
// Each read is a real evaluation, with a null default, through the client
// aiserver.InitClient or InitWithClient installed; it never initializes one.
// No client, an evaluation error, the default, a value that is not an AI
// Config, or a malformed skills list is a failed read, never an empty set. A
// disabled variation answers no skills, so its skills leave disk. A blank key
// gives a source every call refuses with ErrInvalidSkillSource.
//
// Under WatchSkillsFrom the source follows changes to key, so the client must
// have a change signal (the LaunchDarkly client does, unless it is offline).
func ConfigSkills(key string, ldctx ldcontext.Context) SkillSource {
	handlerkit.ReportUsage("client.configSkills", "", "")
	if strings.TrimSpace(key) == "" {
		return SkillSource{invalid: "ConfigSkills was given a blank AI Config key"}
	}
	return SkillSource{
		refs:       func(context.Context) ([]SkillReference, error) { return configRefs(key, ldctx) },
		configKeys: []string{key},
	}
}

func configRefs(key string, ldctx ldcontext.Context) ([]SkillReference, error) {
	failed := func(format string, args ...any) error {
		return fmt.Errorf("the AI Config %q gave no skill list: "+format, append([]any{key}, args...)...)
	}
	c, err := aiserver.GetClient()
	if err != nil {
		return nil, failed("%w", err)
	}
	// Null, not a disabled config: a client that is not ready answers the
	// default, and a disabled config would read as "no skills" and prune.
	v, err := c.JSONVariation(key, ldctx, ldvalue.Null())
	if err != nil {
		return nil, failed("the evaluation failed: %w", err)
	}
	if v.Type() != ldvalue.ObjectType {
		return nil, failed("it evaluated to %s, not an AI Config (the flag is missing, the client is not ready, or the evaluation failed)", v.Type())
	}
	meta := v.GetByKey("_ldMeta")
	enabled := meta.GetByKey("enabled")
	if meta.Type() != ldvalue.ObjectType || enabled.Type() != ldvalue.BoolType {
		return nil, failed("it has no _ldMeta object with a boolean enabled")
	}
	if !enabled.BoolValue() {
		return []SkillReference{}, nil
	}
	// Only the skills field is read: a config used just to pin skills needs no
	// model or provider, which the full AI Config parser would demand.
	raw, _ := v.AsArbitraryValue().(map[string]any)
	refs, err := skillRefs(&aiserver.AiConfig{Raw: raw})
	if err != nil {
		return nil, failed("%w", err)
	}
	return refs, nil
}

func (s SkillSource) validate() error {
	if s.refs == nil {
		reason := s.invalid
		if reason == "" {
			reason = "the zero SkillSource has no refs function; build one with NewSkillSource or ConfigSkills"
		}
		return fmt.Errorf("%w: %s", ErrInvalidSkillSource, reason)
	}
	return nil
}

// WriteSkillsFrom is WriteSkills for the set src answers, read once within
// the reconcile's timeout. A failed read is unavailable retrieval: nothing on
// disk changes, and with OnUnavailableRaise it returns an error wrapping
// ErrSkillsUnavailable.
func WriteSkillsFrom(ctx context.Context, src SkillSource, root string, opts *WriteSkillsOptions) (ReconcileReport, error) {
	handlerkit.ReportUsage("client.writeSkills", "", "")
	if err := src.validate(); err != nil {
		return ReconcileReport{}, err
	}
	return writeSkills(selection{src: &src, ctx: ctx}, root, opts)
}

// WatchSkillsFrom is WatchSkills for the set src answers, read again on every
// trigger: a store change, a change to one of its config keys, or Notify.
//
// A source with config keys needs the AI client's change signal. With no
// client, a client without GetFlagTracker, or an offline client it returns an
// error wrapping ErrStoreNotWatchable; a source built with no config keys
// needs no client and is re-read on store changes and Notify only.
func WatchSkillsFrom(ctx context.Context, src SkillSource, root string, opts *WatchSkillsOptions) (ReconcileReport, *SkillWatcher, error) {
	handlerkit.ReportUsage("client.watchSkills", "", "")
	if err := src.validate(); err != nil {
		return ReconcileReport{}, nil, err
	}
	return watchSkills(ctx, selection{src: &src, ctx: ctx}, root, opts)
}

// Notify schedules a reconcile through the same debounce as a store or config
// change, for a source whose answer changed for a reason the watcher cannot
// see. After Close it does nothing.
func (w *SkillWatcher) Notify() { w.notify() }

// changeTracker is what a client needs for WatchSkillsFrom to follow config
// keys; *ld.LDClient has both methods.
type changeTracker interface {
	GetFlagTracker() interfaces.FlagTracker
	IsOffline() bool
}

// followConfig registers one flag-change listener that only signals the
// watcher for the source's keys.
func (w *SkillWatcher) followConfig(keys []string) error {
	var tracker interfaces.FlagTracker
	if c, err := aiserver.GetClient(); err == nil {
		if ct, ok := c.(changeTracker); ok && !ct.IsOffline() {
			tracker = ct.GetFlagTracker()
		}
	}
	if tracker == nil {
		return fmt.Errorf("%w: WatchSkillsFrom follows the config keys %q through the LaunchDarkly client's "+
			"flag-change signal, and the AI client has none (not initialized, offline, or a client without "+
			"a flag tracker). Initialize an online LaunchDarkly client, or build the source with empty configKeys "+
			"and call SkillWatcher.Notify when its answer changes", ErrStoreNotWatchable, keys)
	}
	ch := tracker.AddFlagChangeListener()
	w.mu.Lock()
	w.tracker, w.flagCh = tracker, ch
	w.mu.Unlock()
	// Drain until RemoveFlagChangeListener closes ch: the base SDK sends while
	// holding its lock, so once the small buffer fills, stopping early would block it.
	go func() {
		for ev := range ch {
			if slices.Contains(keys, ev.Key) {
				w.notify()
			}
		}
	}()
	return nil
}

// readSourceRefs reads src within cfg's budget. A failed read returns the reason
// for an unavailable run.
func readSourceRefs(ctx context.Context, src *SkillSource, cfg writeConfig) ([]SkillReference, string) {
	if cfg.expired() {
		return nil, "the timeout was exhausted before the skill source could be read"
	}
	if ctx == nil {
		ctx = context.Background()
	}
	rctx, cancel := context.WithTimeout(ctx, cfg.timeout-time.Since(cfg.start))
	defer cancel()
	refs, err := src.refs(rctx)
	switch {
	case err != nil:
		return nil, fmt.Sprintf("the skill source failed: %v", err)
	case rctx.Err() != nil:
		return nil, fmt.Sprintf("the skill source did not answer in time: %v", rctx.Err())
	case refs == nil:
		return nil, "the skill source answered nil; an empty set is []SkillReference{}"
	}
	for i, r := range refs {
		if !isValidSkillKey(r.key) || r.version < 1 {
			return nil, fmt.Sprintf("the skill source answered an invalid entry at position %d "+
				"(a valid key and a version >= 1 are required)", i)
		}
	}
	return refs, ""
}
