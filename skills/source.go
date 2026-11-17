package skills

import (
	"context"
	"errors"
	"fmt"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
)

// errNotImplemented marks the API stubs that the tests are written against.
var errNotImplemented = errors.New("skills: not implemented")

// ErrInvalidSkillSource is returned for a SkillSource that cannot be used: a
// nil refs function, an empty config key, or the zero SkillSource. It wraps
// ErrInvalidArgument.
var ErrInvalidSkillSource = fmt.Errorf("%w: invalid skill source", ErrInvalidArgument)

// SkillSource is a requested set of skills that is read when it is needed
// rather than fixed when the call is made. WriteSkillsFrom reads it once;
// WatchSkillsFrom reads it on every trigger. Build one with NewSkillSource, or
// with ConfigSkills for the skills one AI Config pins for one context.
type SkillSource struct {
	refs       func(ctx context.Context) ([]SkillReference, error)
	configKeys []string
}

// NewSkillSource builds a SkillSource from refs, which returns the current
// references, and the AI Config keys whose changes should make a watcher read
// it again.
func NewSkillSource(refs func(ctx context.Context) ([]SkillReference, error), configKeys ...string) (SkillSource, error) {
	return SkillSource{}, errNotImplemented
}

// ConfigSkills is the SkillSource for the skills the AI Config key pins for
// ldctx, evaluated on every read with the configured AI client.
func ConfigSkills(key string, ldctx ldcontext.Context) SkillSource {
	return SkillSource{}
}

// WriteSkillsFrom is WriteSkills for the set src answers, read once.
func WriteSkillsFrom(ctx context.Context, src SkillSource, root string, opts *WriteSkillsOptions) (ReconcileReport, error) {
	return ReconcileReport{}, errNotImplemented
}

// WatchSkillsFrom is WatchSkills for the set src answers, read again on every
// trigger: a store change, a change to one of its config keys, or Notify.
func WatchSkillsFrom(ctx context.Context, src SkillSource, root string, opts *WatchSkillsOptions) (ReconcileReport, *SkillWatcher, error) {
	return ReconcileReport{}, nil, errNotImplemented
}

// Notify schedules a reconcile through the same debounce as a store or config
// change.
func (w *SkillWatcher) Notify() {}
