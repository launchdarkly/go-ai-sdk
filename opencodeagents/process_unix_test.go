//go:build unix

package opencodeagents

import (
	"errors"
	"syscall"
	"testing"
	"time"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

// assertGone checks that the call left no fake process, or process the fake
// started, running.
func assertGone(t *testing.T, s fakeState) {
	t.Helper()
	for _, pid := range []int{s.PID, s.GrandchildPID} {
		if pid != 0 && alive(pid) {
			t.Errorf("process %d is still running", pid)
		}
	}
}

func alive(pid int) bool {
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if err := syscall.Kill(pid, 0); errors.Is(err, syscall.ESRCH) {
			return false
		}
		time.Sleep(20 * time.Millisecond)
	}
	return true
}

// The child is asked to stop with SIGTERM, and its whole process group goes
// with it.
func TestStopTerminatesTheProcessGroup(t *testing.T) {
	setup(t, nil)
	f := newFake(t, fakeSpec{Grandchild: true, Events: reply("ok", 1, 1)})
	if _, err := invoke(t, f.handler(t), aiserver.HandlerRequest{Config: cfg(nil), UserInput: "q"}); err != nil {
		t.Fatal(err)
	}
	s := f.state(t)
	if s.GrandchildPID == 0 {
		t.Fatal("the fake started no grandchild")
	}
	if !s.Terminated {
		t.Error("the child was not sent SIGTERM")
	}
	assertGone(t, s)
}

// A child that ignores SIGTERM is killed after the grace period.
func TestStopKillsAfterTheGracePeriod(t *testing.T) {
	setup(t, nil)
	short(t, &shutdownGrace, 300*time.Millisecond)
	f := newFake(t, fakeSpec{IgnoreTerm: true, Grandchild: true, Events: reply("ok", 1, 1)})
	start := time.Now()
	if _, err := invoke(t, f.handler(t), aiserver.HandlerRequest{Config: cfg(nil), UserInput: "q"}); err != nil {
		t.Fatal(err)
	}
	if d := time.Since(start); d > 5*time.Second {
		t.Errorf("stop took %s", d)
	}
	s := f.state(t)
	if !s.Terminated {
		t.Error("SIGTERM was not tried first")
	}
	assertGone(t, s)
}

// Early stream stop and cancellation leave no grandchild either.
func TestNoProcessOutlivesAnEarlyExit(t *testing.T) {
	setup(t, nil)
	f := newFake(t, fakeSpec{HoldMessage: true, Grandchild: true, Events: []map[string]any{assistant("m1"), delta("m1", "p1", "x")}})
	for ev, err := range f.handler(t).Stream(bg, aiserver.HandlerRequest{Config: cfg(nil), UserInput: "q"}) {
		if err != nil || ev.Type == aiserver.EventChunk {
			break
		}
	}
	assertGone(t, f.state(t))
}
