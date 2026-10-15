package aiserver

import (
	"errors"
	"fmt"
)

// Errors returned by the core, for use with errors.Is. Each is wrapped with
// the flag key or provider it concerns.
var (
	// ErrNotInitialized means GetClient ran before InitClient or InitWithClient.
	ErrNotInitialized = errors.New("launchdarkly ai: client not initialized")
	// ErrConfigNotFound means the flag returned no AI Config: it does not
	// exist, is not an AI Config, or could not be evaluated.
	ErrConfigNotFound = errors.New("launchdarkly ai: AI Config not found")
	// ErrConfigDisabled means the AI Config variation is turned off.
	ErrConfigDisabled = errors.New("launchdarkly ai: AI Config disabled")
	// ErrNoHandler means no registered handler serves the variation's
	// provider and mode.
	ErrNoHandler = errors.New("launchdarkly ai: no handler for provider")
	// ErrGraphDisabled means a graph flag, or a node it references, does not
	// resolve to an enabled configuration.
	ErrGraphDisabled = errors.New("launchdarkly ai: agent graph is disabled")
)

func errorf(format string, args ...any) error { return fmt.Errorf("launchdarkly ai: "+format, args...) }
