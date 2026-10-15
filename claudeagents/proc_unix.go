//go:build unix

package claudeagents

import (
	"os/exec"
	"syscall"
)

// killProcessGroup runs the command in its own process group and kills the
// whole group on timeout, so children started by the shell do not outlive it.
func killProcessGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error { return syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL) }
}
