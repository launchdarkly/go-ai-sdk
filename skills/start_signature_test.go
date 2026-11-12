package skills

// Start reports only an error: Go has no chaining idiom, and the store is the
// receiver the caller already holds.
var _ func(*LaunchDarklySkillStore) error = (*LaunchDarklySkillStore).Start
