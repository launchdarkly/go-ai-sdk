//go:build !unix

package opencodeagents

import "testing"

// assertGone is not checked where the tests cannot signal a process id.
func assertGone(*testing.T, fakeState) {}
