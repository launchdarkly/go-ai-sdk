//go:build !unix

// Package procgroup runs a child process in its own process group, so that
// cancelling the command also stops the processes the child started.
package procgroup

import "os/exec"

// KillOnCancel is a no-op where process groups are unavailable; when the
// command's context is done only the command itself is killed. Call it
// before Start or Run, as on Unix, where it overwrites cmd.SysProcAttr and
// cmd.Cancel.
func KillOnCancel(*exec.Cmd) {}

// Isolate is a no-op where process groups are unavailable.
func Isolate(*exec.Cmd) {}

// Terminate kills the command itself: there is no group, and no SIGTERM.
func Terminate(cmd *exec.Cmd) error { return cmd.Process.Kill() }

// Kill kills the command itself.
func Kill(cmd *exec.Cmd) error { return cmd.Process.Kill() }
