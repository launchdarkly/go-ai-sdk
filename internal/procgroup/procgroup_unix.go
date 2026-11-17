//go:build unix

// Package procgroup runs a child process in its own process group, so that
// cancelling the command also stops the processes the child started.
package procgroup

import (
	"os/exec"
	"syscall"
)

// KillOnCancel runs the command in its own process group and kills the whole
// group when the command's context is done, so children started by the
// command do not outlive it.
func KillOnCancel(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error { return syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL) }
}
