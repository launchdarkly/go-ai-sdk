package experimental

import (
	"context"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/skills"
)

// Agent Skills. Experimental: every name in this
// file may change in a minor release.

// SkillReference is a version-pinned pointer to a skill on an AI Config.
// Experimental.
type SkillReference = skills.SkillReference

// SkillKey is a bare skill key, meaning the newest version. Experimental.
type SkillKey = skills.SkillKey

// SkillRequest is a SkillReference or a SkillKey. Experimental.
type SkillRequest = skills.SkillRequest

// SkillInput is a Skill, *Skill, SkillReference or SkillKey. Experimental.
type SkillInput = skills.SkillInput

// Skill is one verified SKILL.md document; Content is opaque bytes the SDK
// never parses. Experimental.
type Skill = skills.Skill

// SkillFields are the inputs to NewSkill. Experimental.
type SkillFields = skills.SkillFields

// SkillOutcome is the reported form of one retrieval. Experimental.
type SkillOutcome = skills.SkillOutcome

// SkillOutcomeReason is the closed set of retrieval outcomes. Experimental.
type SkillOutcomeReason = skills.SkillOutcomeReason

// ReconcileAction is one WriteSkills outcome. Experimental.
type ReconcileAction = skills.ReconcileAction

// ReconcileActionKind is the closed set of reconcile outcomes. Experimental.
type ReconcileActionKind = skills.ReconcileActionKind

// ReconcileReport is the result of one WriteSkills run. Experimental.
type ReconcileReport = skills.ReconcileReport

// OnUnavailable selects how WriteSkills reacts to unretrievable content.
// Experimental.
type OnUnavailable = skills.OnUnavailable

// SkillStore is the seam skill content is read through. Experimental.
type SkillStore = skills.SkillStore

// ListenerStore is the optional change-notification half of the seam.
// Experimental.
type ListenerStore = skills.ListenerStore

// ListenerRemover is the optional unregistration half of the seam.
// Experimental.
type ListenerRemover = skills.ListenerRemover

// InitializationProber is the optional readiness probe of the seam.
// Experimental.
type InitializationProber = skills.InitializationProber

// Listener is a change callback registered with a ListenerStore.
// Experimental.
type Listener = skills.Listener

// InMemorySkillStore is a SkillStore for local development and tests.
// Experimental.
type InMemorySkillStore = skills.InMemorySkillStore

// LaunchDarklySkillStore is the SkillStore fed by LaunchDarkly delivery. Experimental.
type LaunchDarklySkillStore = skills.LaunchDarklySkillStore

// LaunchDarklySkillStoreOptions configure NewLaunchDarklySkillStore. Experimental.
type LaunchDarklySkillStoreOptions = skills.LaunchDarklySkillStoreOptions

// LaunchDarklySkillStoreMode selects streaming or polling delivery. Experimental.
type LaunchDarklySkillStoreMode = skills.LaunchDarklySkillStoreMode

// StoreDiagnostics are the FDv2 transport counters. Experimental.
type StoreDiagnostics = skills.StoreDiagnostics

// WriteSkillsOptions are WriteSkills' options. Experimental.
type WriteSkillsOptions = skills.WriteSkillsOptions

// WatchSkillsOptions are WatchSkills' options. Experimental.
type WatchSkillsOptions = skills.WatchSkillsOptions

// SkillWatcher is a running watch. Experimental.
type SkillWatcher = skills.SkillWatcher

// On-disk layout constants. Experimental.
const (
	// SkillFilename is the file each skill materializes to: "SKILL.md".
	SkillFilename = skills.SkillFilename
	// ManifestFilename is the manifest under a managed root:
	// ".launchdarkly-skills.json". Add it to .gitignore.
	ManifestFilename = skills.ManifestFilename
	// ManifestVersion is the manifest schema version this release writes: 1.
	ManifestVersion = skills.ManifestVersion
)

// SkillOutcomeReason values. Experimental.
const (
	// ReasonAbsent: the store does not hold the key.
	ReasonAbsent = skills.ReasonAbsent
	// ReasonIntegrityFailure: content failed verification. Fail closed.
	ReasonIntegrityFailure = skills.ReasonIntegrityFailure
	// ReasonOK: a verified skill was returned.
	ReasonOK = skills.ReasonOK
	// ReasonStoreUnavailable: the store returned an error.
	ReasonStoreUnavailable = skills.ReasonStoreUnavailable
	// ReasonWrongVersion: the store answered a pin with another version.
	ReasonWrongVersion = skills.ReasonWrongVersion
)

// ReconcileActionKind values. Experimental.
const (
	// ActionWritten: a new file was written.
	ActionWritten = skills.ActionWritten
	// ActionUpdated: a managed file was overwritten.
	ActionUpdated = skills.ActionUpdated
	// ActionSkippedCurrent: the bytes on disk already are the content.
	ActionSkippedCurrent = skills.ActionSkippedCurrent
	// ActionRemoved: a formerly managed skill is no longer on disk.
	ActionRemoved = skills.ActionRemoved
	// ActionError: the outcome was refused or failed.
	ActionError = skills.ActionError
)

// OnUnavailable and LaunchDarklySkillStoreMode values. Experimental.
const (
	// OnUnavailableKeep reports unretrievable content and keeps files (default).
	OnUnavailableKeep = skills.OnUnavailableKeep
	// OnUnavailableRaise returns ErrSkillsUnavailable instead.
	OnUnavailableRaise = skills.OnUnavailableRaise
	// LaunchDarklySkillStoreModeStream holds a long-lived connection (default).
	LaunchDarklySkillStoreModeStream = skills.LaunchDarklySkillStoreModeStream
	// LaunchDarklySkillStoreModePoll polls every PollInterval.
	LaunchDarklySkillStoreModePoll = skills.LaunchDarklySkillStoreModePoll
)

// Sentinel errors; match with errors.Is. Experimental.
var (
	// ErrNoSkillStore: no skill store is configured.
	ErrNoSkillStore = skills.ErrNoSkillStore
	// ErrInvalidArgument: a caller error (root, option, credential, URI).
	ErrInvalidArgument = skills.ErrInvalidArgument
	// ErrSkillsUnavailable: content could not be retrieved, with OnUnavailableRaise.
	ErrSkillsUnavailable = skills.ErrSkillsUnavailable
	// ErrStoreNotWatchable: the configured store cannot report changes.
	ErrStoreNotWatchable = skills.ErrStoreNotWatchable
	// ErrUnsupportedKind: AddListener for a kind other than "skill".
	ErrUnsupportedKind = skills.ErrUnsupportedKind
	// ErrStoreClosed: LaunchDarklySkillStore.Start after Close.
	ErrStoreClosed = skills.ErrStoreClosed
	// ErrUnsupportedPlatform: materialization needs a POSIX platform.
	ErrUnsupportedPlatform = skills.ErrUnsupportedPlatform
)

// NewSkillReference builds a reference without validating it. Experimental.
func NewSkillReference(key string, version int) SkillReference {
	return skills.NewSkillReference(key, version)
}

// NewSkill builds a Skill from your own content; WriteSkills re-verifies its
// hash before writing. Experimental.
func NewSkill(f SkillFields) Skill { return skills.NewSkill(f) }

// NewSkillOutcome builds an outcome, for wrapping your own retrieval in the
// same shape. Experimental.
func NewSkillOutcome(skill *Skill, reason SkillOutcomeReason, detail string) SkillOutcome {
	return skills.NewSkillOutcome(skill, reason, detail)
}

// NewListener wraps a change callback. Experimental.
func NewListener(fn func(raw any)) *Listener { return skills.NewListener(fn) }

// NewInMemorySkillStore builds an in-memory store from raw objects.
// Experimental.
func NewInMemorySkillStore(objects map[string]any) *InMemorySkillStore {
	return skills.NewInMemorySkillStore(objects)
}

// NewLaunchDarklySkillStore builds a delivery-backed store for a server-side SDK key.
// Call Start, then WaitForSkills, then SetSkillStore. Experimental.
func NewLaunchDarklySkillStore(sdkKey string, opts *LaunchDarklySkillStoreOptions) (*LaunchDarklySkillStore, error) {
	return skills.NewLaunchDarklySkillStore(sdkKey, opts)
}

// SetSkillStore configures the store the accessors and WriteSkills read
// from. A nil store is ignored; aiserver.Shutdown clears it. Replacing a store
// does not close the old one, and a running watcher keeps the store it
// started with. Experimental.
func SetSkillStore(store SkillStore) { skills.SetSkillStore(store) }

// SkillRefs projects a resolved AI Config's skills array into references. A
// nil config (a failed InspectConfig) is an error, never an empty list.
// Experimental.
func SkillRefs(config *aiserver.AiConfig) ([]SkillReference, error) { return skills.SkillRefs(config) }

// GetSkill returns one verified skill, or nil. version 0 means newest.
// Experimental.
func GetSkill(key string, version int) (*Skill, error) {
	return skills.GetSkill(key, version)
}

// GetSkillResult is GetSkill reporting why there is no skill. Experimental.
func GetSkillResult(key string, version int) (SkillOutcome, error) {
	return skills.GetSkillResult(key, version)
}

// GetSkills returns the verified skills for refs, in order, omitting misses.
// Experimental.
func GetSkills[R SkillRequest](refs []R) ([]Skill, error) {
	return skills.GetSkills(refs)
}

// AllSkills returns every verified skill the store holds, newest per key.
// Experimental.
func AllSkills() ([]Skill, error) {
	return skills.AllSkills()
}

// WriteSkills materializes skills under root. Experimental.
func WriteSkills[I SkillInput](skillsIn []I, root string, opts *WriteSkillsOptions) (ReconcileReport, error) {
	return skills.WriteSkills(skillsIn, root, opts)
}

// WriteAllSkills materializes every skill the store holds (the "*" key).
// Experimental.
func WriteAllSkills(root string, opts *WriteSkillsOptions) (ReconcileReport, error) {
	return skills.WriteAllSkills(root, opts)
}

// WatchSkills reconciles now, then on every delivery change, until Close or
// ctx is done. Experimental.
func WatchSkills[I SkillInput](ctx context.Context, skillsIn []I, root string, opts *WatchSkillsOptions) (ReconcileReport, *SkillWatcher, error) {
	return skills.WatchSkills(ctx, skillsIn, root, opts)
}

// WatchAllSkills is WatchSkills over every skill the store holds.
// Experimental.
func WatchAllSkills(ctx context.Context, root string, opts *WatchSkillsOptions) (ReconcileReport, *SkillWatcher, error) {
	return skills.WatchAllSkills(ctx, root, opts)
}
