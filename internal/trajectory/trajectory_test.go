package trajectory

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"
)

var bg = context.Background()

func TestRenderShapeAndOrder(t *testing.T) {
	r := NewRecorder()
	search := r.Observe("search", func(_ context.Context, a map[string]any) (any, error) { return map[string]any{"hits": 2.0}, nil })
	fail := r.Observe("refund", func(context.Context, map[string]any) (any, error) { return nil, errors.New("denied") })
	_ = r.Observe("unused", func(context.Context, map[string]any) (any, error) { return nil, nil })
	_, _ = search(bg, map[string]any{"q": "café", "b": 1.0})
	if _, err := fail(bg, nil); err == nil || err.Error() != "denied" {
		t.Fatalf("recorder changed the error: %v", err)
	}
	want := strings.Join([]string{
		"Tools available: search, refund, unused",
		"Tool calls made while producing the response, in order:",
		"1. search",
		`   arguments: {"b":1,"q":"café"}`,
		`   result: {"hits":2}`,
		"2. refund",
		"   arguments: ",
		"   error: denied",
	}, "\n")
	if got := r.Render(); got != want {
		t.Errorf("render:\n%s\nwant:\n%s", got, want)
	}
}

func TestNoEscapesAndNoNull(t *testing.T) {
	r := NewRecorder()
	f := r.Observe("t", func(context.Context, map[string]any) (any, error) {
		return map[string]any{"city": "東京", "x": "<a&b>"}, nil
	})
	g := r.Observe("void", func(context.Context, map[string]any) (any, error) { return nil, nil })
	_, _ = f(bg, nil)
	_, _ = g(bg, nil)
	out := r.Render()
	if strings.Contains(out, `\u`) || !strings.Contains(out, "東京") || !strings.Contains(out, "<a&b>") {
		t.Errorf("escaped output: %s", out)
	}
	if strings.Contains(out, "null") {
		t.Errorf("absent value rendered as null: %s", out)
	}
}

func TestEmptyCases(t *testing.T) {
	if got := NewRecorder().Render(); got != "" {
		t.Errorf("no tools should render nothing, got %q", got)
	}
	r := NewRecorder()
	r.Observe("t", func(context.Context, map[string]any) (any, error) { return nil, nil })
	if got := r.Render(); !strings.Contains(got, "No tool calls were made") {
		t.Errorf("offered-but-unused: %q", got)
	}
}

func TestCapsStillExecute(t *testing.T) {
	r := NewRecorder()
	ran := 0
	f := r.Observe("t", func(context.Context, map[string]any) (any, error) {
		ran++
		return strings.Repeat("é", MaxValueChars+10), nil
	})
	for range MaxRecordedCalls + 3 {
		_, _ = f(bg, nil)
	}
	if ran != MaxRecordedCalls+3 || len(r.Invocations()) != MaxRecordedCalls || r.Omitted() != 3 {
		t.Fatalf("ran=%d recorded=%d omitted=%d", ran, len(r.Invocations()), r.Omitted())
	}
	res := r.Invocations()[0].Result
	if !strings.HasSuffix(res, truncationSuffix) || len([]rune(res)) != MaxValueChars {
		t.Errorf("truncation: %d runes", len([]rune(res)))
	}
	if !strings.Contains(r.Render(), "(3 further tool call(s) were made but not recorded.)") {
		t.Error("omitted count not rendered")
	}
}

func TestStartOrderAndIsolation(t *testing.T) {
	r := NewRecorder()
	slowStarted := make(chan struct{})
	slow := r.Observe("slow", func(context.Context, map[string]any) (any, error) {
		close(slowStarted)
		time.Sleep(20 * time.Millisecond)
		return "s", nil
	})
	fast := r.Observe("fast", func(context.Context, map[string]any) (any, error) { return "f", nil })
	var wg sync.WaitGroup
	wg.Go(func() { _, _ = slow(bg, nil) })
	<-slowStarted
	_, _ = fast(bg, nil)
	wg.Wait()
	if inv := r.Invocations(); inv[0].Name != "slow" || inv[1].Name != "fast" || inv[0].Result != "s" {
		t.Errorf("order = %+v", inv)
	}

	// Two units sharing one tool implementation keep separate trajectories.
	shared := func(_ context.Context, a map[string]any) (any, error) { return a["unit"], nil }
	a, b := NewRecorder(), NewRecorder()
	fa, fb := a.Observe("t", shared), b.Observe("t", shared)
	var wg2 sync.WaitGroup
	for i := range 20 {
		wg2.Go(func() {
			if i%2 == 0 {
				_, _ = fa(bg, map[string]any{"unit": "a"})
			} else {
				_, _ = fb(bg, map[string]any{"unit": "b"})
			}
		})
	}
	wg2.Wait()
	for _, inv := range a.Invocations() {
		if inv.Result != "a" {
			t.Fatalf("unit a recorded %q", inv.Result)
		}
	}
	if len(a.Invocations()) != 10 || len(b.Invocations()) != 10 {
		t.Errorf("a=%d b=%d", len(a.Invocations()), len(b.Invocations()))
	}
}

func TestShouldObserve(t *testing.T) {
	if ShouldObserve("__handoff_x", nil) || !ShouldObserve("t", nil) || ShouldObserve("t", map[string]bool{"u": true}) {
		t.Error("ShouldObserve wrong")
	}
}
