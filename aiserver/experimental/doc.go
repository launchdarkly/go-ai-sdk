// Package experimental is the experimental entry point of the LaunchDarkly AI
// SDK for Go. Every name here is published but may change
// in a minor release, with a changelog entry under Experimental. None of these
// names is exported from package aiserver, and aiserver never names them.
//
// Current experimental features:
//
//   - Agent Skills (skills.go): version-pinned skill references on AI Configs,
//     verified skill content from a SkillStore, materialization to disk, the
//     FDv2 delivery transport, and a watcher that keeps disk in sync.
//   - Evaluations from Code (evaluations.go).
//   - Shadow requests (shadow_requests.go): ShadowConfig runs a candidate
//     variation on the served call's input and tracks it without serving it.
//
// When a feature is promoted to core, its names here stay as deprecated
// aliases of the aiserver names until the next major release.
package experimental
