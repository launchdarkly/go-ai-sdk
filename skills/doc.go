// Package skills is Agent Skills for the LaunchDarkly AI SDK for Go:
// version-pinned skill references on AI Configs, verified skill content from a
// SkillStore, materialization to disk, the LaunchDarkly delivery transport,
// and a watcher that keeps disk in sync.
//
// It is core, not experimental, and imports aiserver the way the provider
// packages do. Configure a store with SetSkillStore; aiserver.Shutdown clears it.
package skills
