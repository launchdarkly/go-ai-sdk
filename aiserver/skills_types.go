// Agent Skills: value types, the SkillStore seam, integrity verification,
// filesystem materialization, the FDv2 delivery transport and the watcher.

package aiserver

import "slices"

// SkillFilename is the single file each skill materializes to, under
// <root>/<key>/.
const SkillFilename = "SKILL.md"

// ManifestFilename is the SDK's record of what it has written under a
// managed root. Add it to .gitignore when the root is inside a repository.
const ManifestFilename = ".launchdarkly-skills.json"

// ManifestVersion is the manifest schema version this release writes, and
// the highest it can read.
const ManifestVersion = 1

// skillObjectKind is the kind the accessors pass to SkillStore.GetObject and
// AllObjects. An SDK seam string, not a wire contract, so it is not exported.
const skillObjectKind = "skill"

// maxSkillContentBytes is the backstop cap on skill content. It sits far
// above the platform's own limit and is deliberately unexported.
const maxSkillContentBytes = 10 * 1024 * 1024

// SkillReference is a version-pinned pointer to a skill, as attached to an AI
// Config variation. It is immutable: the fields are unexported and only read
// through methods.
type SkillReference struct {
	key     string
	version int
}

// NewSkillReference builds a reference. Like the other SDKs it does not
// validate its fields; retrieval and materialization do.
func NewSkillReference(key string, version int) SkillReference {
	return SkillReference{key: key, version: version}
}

// Key is the skill key.
func (r SkillReference) Key() string { return r.key }

// Version is the pinned skill version.
func (r SkillReference) Version() int { return r.version }

func (r SkillReference) target() (string, int) { return r.key, r.version }
func (SkillReference) isSkillInput()           {}

// SkillKey is a bare skill key: a request for the newest version the store
// holds.
type SkillKey string

func (k SkillKey) target() (string, int) { return string(k), 0 }
func (SkillKey) isSkillInput()           {}

// SkillRequest is a SkillReference or a SkillKey: something a store can
// resolve. The interface is sealed.
type SkillRequest interface {
	target() (key string, version int)
}

// SkillInput is a Skill, a *Skill, a SkillReference or a SkillKey: anything
// WriteSkills accepts. The interface is sealed.
type SkillInput interface {
	isSkillInput()
}

// SkillFields are the inputs to NewSkill.
type SkillFields struct {
	Key         string
	Version     int
	Content     []byte
	ContentHash string
	Name        string
	Description string
}

// Skill is one verified SKILL.md document. It is immutable: Content returns a
// copy, and NewSkill copies its input, so bytes that were hashed cannot change
// before they are written.
type Skill struct {
	key         string
	version     int
	content     []byte
	contentHash string
	name        string
	description string
}

// NewSkill builds a Skill from caller-supplied content (bring your own
// content). It does not verify anything; WriteSkills re-verifies the hash
// immediately before writing.
func NewSkill(f SkillFields) Skill {
	return Skill{
		key:         f.Key,
		version:     f.Version,
		content:     slices.Clone(f.Content),
		contentHash: f.ContentHash,
		name:        f.Name,
		description: f.Description,
	}
}

// Key is the skill key.
func (s Skill) Key() string { return s.key }

// Version is the skill version.
func (s Skill) Version() int { return s.version }

// Content returns a copy of the verbatim bytes LaunchDarkly delivered. The SDK
// never decodes or parses them.
func (s Skill) Content() []byte { return slices.Clone(s.content) }

// ContentHash is the sha256, lowercase hex, computed locally over Content.
func (s Skill) ContentHash() string { return s.contentHash }

// Name is the display name from LaunchDarkly metadata, or "".
func (s Skill) Name() string { return s.name }

// Description is the description from LaunchDarkly metadata, or "".
func (s Skill) Description() string { return s.description }

func (Skill) isSkillInput() {}

// SkillOutcomeReason is the closed set of retrieval outcomes.
type SkillOutcomeReason string

// The five outcome tokens, in alphabetical order.
const (
	// ReasonAbsent means the store does not hold the key (or the pinned version).
	ReasonAbsent SkillOutcomeReason = "absent"
	// ReasonIntegrityFailure means content was delivered and failed
	// verification, or arrived under another key. Fail closed on it.
	ReasonIntegrityFailure SkillOutcomeReason = "integrity_failure"
	// ReasonOK means a verified skill was returned.
	ReasonOK SkillOutcomeReason = "ok"
	// ReasonStoreUnavailable means the store returned an error. An outage,
	// not a deletion.
	ReasonStoreUnavailable SkillOutcomeReason = "store_unavailable"
	// ReasonWrongVersion means the store answered a pin with another version.
	ReasonWrongVersion SkillOutcomeReason = "wrong_version"
)

var skillOutcomeReasons = []SkillOutcomeReason{
	ReasonAbsent, ReasonIntegrityFailure, ReasonOK, ReasonStoreUnavailable, ReasonWrongVersion,
}

// SkillOutcome is the reported form of one retrieval. Immutable.
type SkillOutcome struct {
	skill  *Skill
	reason SkillOutcomeReason
	detail string
}

// NewSkillOutcome builds an outcome, for callers wrapping their own retrieval
// in the same shape.
func NewSkillOutcome(skill *Skill, reason SkillOutcomeReason, detail string) SkillOutcome {
	if skill != nil {
		c := *skill
		skill = &c
	}
	return SkillOutcome{skill: skill, reason: reason, detail: detail}
}

// Skill is the verified skill; non-nil only when Reason is ReasonOK.
func (o SkillOutcome) Skill() *Skill {
	if o.skill == nil {
		return nil
	}
	c := *o.skill
	return &c
}

// Reason is which outcome happened.
func (o SkillOutcome) Reason() SkillOutcomeReason { return o.reason }

// Detail is a human-readable message, set for every reason except ReasonOK.
// It names the skill key and the failure mode, never content or a path.
// Branch on Reason, not on Detail.
func (o SkillOutcome) Detail() string { return o.detail }

// ReconcileActionKind is the closed set of per-skill reconcile outcomes.
type ReconcileActionKind string

// The five reconcile action kinds.
const (
	// ActionWritten means the file did not exist and now holds the content.
	ActionWritten ReconcileActionKind = "written"
	// ActionUpdated means a managed file held other bytes and was overwritten.
	ActionUpdated ReconcileActionKind = "updated"
	// ActionSkippedCurrent means the bytes on disk already are the content.
	ActionSkippedCurrent ReconcileActionKind = "skipped_current"
	// ActionRemoved means the skill is no longer managed and is not on disk.
	ActionRemoved ReconcileActionKind = "removed"
	// ActionError means the outcome was refused or failed; see ErrorMessage.
	ActionError ReconcileActionKind = "error"
)

var reconcileActionKinds = []ReconcileActionKind{
	ActionWritten, ActionUpdated, ActionSkippedCurrent, ActionRemoved, ActionError,
}

// OnUnavailable is how WriteSkills reacts to content it could not retrieve.
type OnUnavailable string

// The two OnUnavailable values. The zero value "" means OnUnavailableKeep.
const (
	// OnUnavailableKeep reports the failure and leaves files alone (default).
	OnUnavailableKeep OnUnavailable = "keep"
	// OnUnavailableRaise returns ErrSkillsUnavailable instead.
	OnUnavailableRaise OnUnavailable = "raise"
)

var onUnavailableValues = []OnUnavailable{OnUnavailableKeep, OnUnavailableRaise}

// ReconcileAction is what WriteSkills did, or refused to do, for one skill.
// Immutable.
type ReconcileAction struct {
	key     string
	action  ReconcileActionKind
	version int
	path    string
	err     string
}

// Key is the skill key, or "" for a run-level failure that belongs to no
// single skill (a corrupt manifest, an unwritable manifest, a retrieval that
// failed before any key was known). Expect "" when grouping by key.
func (a ReconcileAction) Key() string { return a.key }

// Action is the outcome kind.
func (a ReconcileAction) Action() ReconcileActionKind { return a.action }

// Version is the skill version, or 0 when it is not known.
func (a ReconcileAction) Version() int { return a.version }

// Path is the canonical resolved path of SKILL.md, or "" when none was
// determined.
func (a ReconcileAction) Path() string { return a.path }

// ErrorMessage is the failure detail, set only when Action is ActionError.
func (a ReconcileAction) ErrorMessage() string { return a.err }

// ReconcileReport is the result of one WriteSkills run. Immutable.
type ReconcileReport struct {
	actions []ReconcileAction
}

func newReconcileReport(actions []ReconcileAction) ReconcileReport {
	return ReconcileReport{actions: slices.Clone(actions)}
}

// Actions returns every outcome, in order.
func (r ReconcileReport) Actions() []ReconcileAction { return slices.Clone(r.actions) }

// OK reports whether no action is an error. It is true iff Errors is empty.
func (r ReconcileReport) OK() bool { return len(r.Errors()) == 0 }

// Errors returns the error actions, in Actions order.
func (r ReconcileReport) Errors() []ReconcileAction {
	out := []ReconcileAction{}
	for _, a := range r.actions {
		if a.action == ActionError {
			out = append(out, a)
		}
	}
	return out
}
