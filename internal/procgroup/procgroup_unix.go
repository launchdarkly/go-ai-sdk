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
// command do not outlive it. Call it before Start or Run. It overwrites
// cmd.SysProcAttr and cmd.Cancel.
func KillOnCancel(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error { return syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL) }
}

// Isolate runs the command in its own process group. Call it before Start;
// it overwrites cmd.SysProcAttr. Stop the group with Terminate and Kill.
func Isolate(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}

// Terminate sends SIGTERM to the started command's process group, or to the
// command itself when it has no group of its own.
func Terminate(cmd *exec.Cmd) error { return signalGroup(cmd, syscall.SIGTERM) }

// Kill sends SIGKILL to the started command's process group, or to the
// command itself when it has no group of its own.
func Kill(cmd *exec.Cmd) error { return signalGroup(cmd, syscall.SIGKILL) }

func signalGroup(cmd *exec.Cmd, sig syscall.Signal) error {
	if err := syscall.Kill(-cmd.Process.Pid, sig); err == nil {
		return nil
	}
	return cmd.Process.Signal(sig)
}
