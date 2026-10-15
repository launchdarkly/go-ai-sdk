package skills

import (
	"errors"
	"fmt"
)

// Sentinel errors. Every error the package returns wraps exactly one of these,
// so callers branch with errors.Is.
var (
	// ErrNoSkillStore means no skill store is configured.
	ErrNoSkillStore = errors.New(noStoreMessage)

	// ErrInvalidArgument is a caller error: an unusable root, a negative
	// timeout or debounce, an unknown OnUnavailable, a bad credential or URI.
	ErrInvalidArgument = errors.New("launchdarkly ai skills: invalid argument")

	// ErrSkillsUnavailable is returned by WriteSkills with OnUnavailableRaise
	// when skill content could not be retrieved.
	ErrSkillsUnavailable = errors.New("launchdarkly ai skills: skill retrieval unavailable")

	// ErrStoreNotWatchable means the configured store cannot report changes.
	ErrStoreNotWatchable = errors.New("launchdarkly ai skills: the configured skill store cannot be watched")

	// ErrUnsupportedKind is returned by AddListener for a kind other than "skill".
	ErrUnsupportedKind = errors.New("launchdarkly ai skills: unsupported object kind")

	// ErrStoreClosed is returned by FDv2SkillStore.Start after Close.
	ErrStoreClosed = errors.New("launchdarkly ai skills: the FDv2SkillStore is closed")

	// ErrUnsupportedPlatform is returned by WriteSkills where the platform has
	// no descriptor-relative (*at) filesystem calls.
	ErrUnsupportedPlatform = errors.New("launchdarkly ai skills: materialization needs a POSIX platform")
)

// noStoreMessage names the setter and both shipped stores. Callers
// match on "skill store"; keep that phrase.
const noStoreMessage = "launchdarkly ai skills: no skill store is configured, so skill content " +
	"cannot be retrieved. Configure one with experimental.SetSkillStore(store): " +
	"FDv2SkillStore (experimental.NewFDv2SkillStore) receives content from LaunchDarkly, " +
	"and InMemorySkillStore (experimental.NewInMemorySkillStore) is for local development and testing"

func invalidArg(format string, args ...any) error {
	return fmt.Errorf("%w: "+format, append([]any{ErrInvalidArgument}, args...)...)
}
