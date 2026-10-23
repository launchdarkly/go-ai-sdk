// Package experimental is the experimental entry point of the LaunchDarkly AI
// SDK for Go. Every name here is published but may change
// in a minor release, with a changelog entry under Experimental. None of these
// names is exported from package aiserver, and aiserver never names them.
//
// Current experimental features:
//
//   - Evaluations from Code (evaluations.go).
//
// When a feature is promoted to core, its names move to package aiserver and
// leave this package. Agent Skills has been promoted: use the aiserver names.
package experimental
