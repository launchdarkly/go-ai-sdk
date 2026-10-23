package aiserver

import (
	"fmt"
	"slices"
	"time"
)

// defaultWriteTimeout is WriteSkills' default time budget.
const defaultWriteTimeout = 10 * time.Second

// WriteSkillsOptions are WriteSkills' options. The zero value (and nil) is the
// default: prune on, a 10 second timeout, OnUnavailableKeep.
type WriteSkillsOptions struct {
	// NoPrune keeps formerly managed skills that are no longer requested.
	// Pruning is how revocation takes effect, so it is on by default; the
	// field is inverted so the zero value keeps the default.
	NoPrune bool
	// Timeout bounds the whole call, including retrieval and every prune
	// step. nil means 10 seconds. A pointer, because zero is a meaningful
	// value: an already exhausted budget, in which nothing is retrieved.
	// Negative is an ErrInvalidArgument.
	Timeout *time.Duration
	// OnUnavailable is OnUnavailableKeep ("" means keep) or OnUnavailableRaise.
	OnUnavailable OnUnavailable
}

// writeConfig is a validated WriteSkillsOptions with the clock started.
type writeConfig struct {
	prune         bool
	timeout       time.Duration
	onUnavailable OnUnavailable
	start         time.Time
}

// expired compares elapsed monotonic time against the budget rather than
// computing a deadline, so no budget can overflow.
func (c writeConfig) expired() bool { return time.Since(c.start) >= c.timeout }

func parseWriteOptions(opts *WriteSkillsOptions) (writeConfig, error) {
	cfg := writeConfig{prune: true, timeout: defaultWriteTimeout, onUnavailable: OnUnavailableKeep, start: time.Now()}
	if opts == nil {
		return cfg, nil
	}
	cfg.prune = !opts.NoPrune
	if opts.Timeout != nil {
		if *opts.Timeout < 0 {
			return writeConfig{}, invalidArg("timeout must be a non-negative duration, got %v", *opts.Timeout)
		}
		cfg.timeout = *opts.Timeout
	}
	if opts.OnUnavailable != "" {
		if !slices.Contains(onUnavailableValues, opts.OnUnavailable) {
			return writeConfig{}, invalidArg("OnUnavailable must be %q or %q, got %q", OnUnavailableKeep, OnUnavailableRaise, opts.OnUnavailable)
		}
		cfg.onUnavailable = opts.OnUnavailable
	}
	return cfg, nil
}

// selection is what a reconcile writes: an explicit list, or every skill the
// store holds.
type selection struct {
	all   bool
	items []SkillInput
}

// WriteSkills materializes skills under a managed root at
// <root>/<key>/SKILL.md, driven by the manifest <root>/.launchdarkly-skills.json:
// it overwrites or deletes only files the manifest records as the SDK's.
// Skill values are written as given (after re-verification); references and
// keys resolve through the configured store.
//
// Caller errors (an unusable root, an invalid option, a nil *Skill) return an
// error wrapping ErrInvalidArgument. With OnUnavailableRaise, content that
// could not be retrieved returns an error wrapping ErrSkillsUnavailable.
// Every other outcome, per-skill failures included, is in the report.
//
// Run at most one reconcile per root at a time.
func WriteSkills[I SkillInput](skills []I, root string, opts *WriteSkillsOptions) (ReconcileReport, error) {
	items := make([]SkillInput, len(skills))
	for i, s := range skills {
		items[i] = s
	}
	return writeSkills(selection{items: items}, root, opts)
}

// WriteAllSkills is WriteSkills for every skill the store holds, the newest
// version per key (the "*" selection). It enumerates raw store objects itself
// rather than going through AllSkills, so a skill that fails verification
// keeps its files on disk instead of reading as revoked.
func WriteAllSkills(root string, opts *WriteSkillsOptions) (ReconcileReport, error) {
	return writeSkills(selection{all: true}, root, opts)
}

// pendingWrite is one requested skill: resolved content, or why there is none.
type pendingWrite struct {
	key   string
	skill *Skill
	err   string
}

const unavailablePrefix = "skill retrieval unavailable: "

// availableStore is the single gate deciding whether retrieval may be
// attempted: an exhausted budget, no store, or a store that
// has not received its initial data each block it, which marks the run
// incomplete and suppresses pruning.
func availableStore(cfg writeConfig, subject string) (SkillStore, string) {
	if cfg.expired() {
		return nil, fmt.Sprintf("the timeout was exhausted before %s could be retrieved", subject)
	}
	s := currentStore()
	if s == nil {
		return nil, noStoreMessage
	}
	if !storeIsInitialized(s) {
		return nil, fmt.Sprintf("the skill store has not received its initial data, so %s could not be "+
			"retrieved and nothing on disk was changed. Wait for delivery before reconciling: "+
			"FDv2SkillStore.WaitForSkills(timeout) returns true once the first payload has arrived", subject)
	}
	return s, ""
}

func unavailableError(reason string) error {
	return fmt.Errorf("%w: %s", ErrSkillsUnavailable, reason)
}

// resolveRequests turns the input into one pending write per skill, and
// reports whether retrieval was incomplete (which suppresses pruning). An
// absent or withheld reference keeps its key in the set, carrying an error,
// so prune leaves its files alone.
func resolveRequests(sel selection, cfg writeConfig) ([]pendingWrite, bool, error) {
	if sel.all {
		return resolveAll(cfg)
	}
	var (
		out        []pendingWrite
		incomplete bool
	)
	for _, item := range sel.items {
		switch v := item.(type) {
		case Skill:
			out = append(out, pendingWrite{key: v.key, skill: &v})
			continue
		case *Skill:
			if v == nil {
				return nil, false, invalidArg("WriteSkills was given a nil *Skill")
			}
			c := *v
			out = append(out, pendingWrite{key: c.key, skill: &c})
			continue
		}
		req, ok := item.(SkillRequest)
		if !ok {
			return nil, false, invalidArg("WriteSkills was given an unsupported input %T", item)
		}
		key, version := req.target()
		r := resolveReference(cfg, key, version)
		if r.unavailable {
			incomplete = true
			if cfg.onUnavailable == OnUnavailableRaise {
				return nil, false, unavailableError(r.err)
			}
			r.err = unavailablePrefix + r.err
		}
		out = append(out, pendingWrite{key: key, skill: r.skill, err: r.err})
	}
	return out, incomplete, nil
}

func resolveReference(cfg writeConfig, key string, version int) resolution {
	s, blocked := availableStore(cfg, fmt.Sprintf("'%s'", safeKey(key)))
	if blocked != "" {
		return resolution{reason: ReasonStoreUnavailable, err: blocked, unavailable: true}
	}
	return resolveFromStore(s, key, version)
}

func resolveAll(cfg writeConfig) ([]pendingWrite, bool, error) {
	unavailableRun := func(reason string) ([]pendingWrite, bool, error) {
		if cfg.onUnavailable == OnUnavailableRaise {
			return nil, false, unavailableError(reason)
		}
		return []pendingWrite{{key: "", err: unavailablePrefix + reason}}, true, nil
	}
	s, blocked := availableStore(cfg, "the skill set")
	if blocked != "" {
		return unavailableRun(blocked)
	}
	objects, err := allObjects(s)
	if err != nil {
		currentLogger().Error("the skill store failed while listing skills", "error", err)
		return unavailableRun(fmt.Sprintf("the skill store returned an error while listing skills: %v", err))
	}
	candidates := newestByKey(objects)
	out := make([]pendingWrite, 0, len(candidates))
	resolved, unattributed := 0, false
	for _, c := range candidates {
		p := pendingForRaw(c)
		if p.skill != nil {
			resolved++
		} else if p.key == "" {
			unattributed = true
		}
		out = append(out, p)
	}
	logWithholdingSummary("skills held by the store", len(candidates), resolved)
	// A failure that cannot be attributed to any key cannot protect its copy
	// on disk, so it suppresses pruning for the whole run.
	return out, unattributed, nil
}

// pendingForRaw verifies one listed object. A failure keeps its key in the
// requested set: the object's own key first (what an earlier run wrote to
// disk), the store's map key second, the run level last.
func pendingForRaw(c candidate) pendingWrite {
	if sk := verifyRawSkill(c.raw); sk != nil {
		return pendingWrite{key: sk.key, skill: sk}
	}
	obj, _ := c.raw.(map[string]any)
	key, _ := obj["key"].(string)
	if !isValidSkillKey(key) {
		key = c.objectKey
	}
	if !isValidSkillKey(key) {
		return pendingWrite{key: "", err: "the skill store served an object under an invalid key; it was withheld"}
	}
	return pendingWrite{key: key, err: fmt.Sprintf("skill '%s' failed integrity verification and was withheld; the copy already on disk was left alone", key)}
}

func runError(msg string) ReconcileAction {
	return ReconcileAction{key: "", action: ActionError, err: msg}
}
