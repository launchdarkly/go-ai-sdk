//go:build !unix

// Package procgroup runs a child process in its own process group, so that
// cancelling the command also stops the processes the child started.
package procgroup

import "os/exec"

// KillOnCancel is a no-op where process groups are unavailable; when the
// command's context is done only the command itself is killed.
func KillOnCancel(*exec.Cmd) {}
