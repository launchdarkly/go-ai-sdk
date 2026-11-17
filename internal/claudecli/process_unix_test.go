//go:build unix

package claudecli

import (
	"context"
	"errors"
	"syscall"
	"testing"
	"time"
)

// waitGone fails the test unless pid exits within a few seconds.
func waitGone(t *testing.T, pid int) {
	t.Helper()
	if pid == 0 {
		t.Error("no pid recorded")
		return
	}
	for deadline := time.Now().Add(3 * time.Second); time.Now().Before(deadline); time.Sleep(20 * time.Millisecond) {
		if errors.Is(syscall.Kill(pid, 0), syscall.ESRCH) {
			return
		}
	}
	t.Errorf("process %d still running", pid)
}

// The kill reaches the CLI's whole process group, so a child the CLI
// started does not outlive the run.
func TestGroupKill(t *testing.T) {
	opts, read := newFake(t, fakeScript{Steps: []fakeStep{{Child: true}, {Send: msg(assistantLine)}, {Hang: true}}})
	for m, err := range Run(context.Background(), opts, "hello") {
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
		if m.Type == "assistant" {
			break
		}
	}
	r := read()
	waitGone(t, r.PID)
	waitGone(t, r.ChildPID)
}
