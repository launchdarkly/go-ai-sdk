//go:build !unix

package claudeagents

import "os/exec"

// killProcessGroup is a no-op where process groups are unavailable; on
// timeout only the shell itself is killed.
func killProcessGroup(*exec.Cmd) {}
