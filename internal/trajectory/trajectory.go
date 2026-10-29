// Package trajectory records the tool calls one invocation or dataset row
// makes and renders them for a judge. It is shared by the
// online judge path and the offline evaluations runner.
package trajectory

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
)

const (
	// MaxRecordedCalls caps recorded calls per unit; later calls still run
	// and are counted (a cross-language contract).
	MaxRecordedCalls = 50
	// MaxValueChars caps the rendered text of one argument bag or result.
	MaxValueChars = 2000
	// HandoffPrefix marks synthetic graph-routing tools, never recorded.
	HandoffPrefix = "__handoff_"

	truncationSuffix = "… (truncated)"
)

// Func is a recordable tool: the same shape as aiserver.ToolFunc, declared
// here so this package does not import the core.
type Func = func(ctx context.Context, args map[string]any) (any, error)

// Invocation is one recorded call. Values are rendered when recorded so a
// tool that later mutates its arguments cannot rewrite what a judge reads.
type Invocation struct {
	Name      string
	Arguments string
	Result    string
	Error     string
	Failed    bool
	// Started and Ended bracket the call; Done is false while it is still
	// open (or if it panicked), and then Ended is zero. Render ignores them.
	Started time.Time
	Ended   time.Time
	Done    bool
}

// Recorder records one unit's tool calls in call-start order. Safe for
// concurrent use; never share one across units.
type Recorder struct {
	mu          sync.Mutex
	limit       int
	invocations []Invocation
	omitted     int
	observable  []string
}

// NewRecorder returns a Recorder with the contract limit.
func NewRecorder() *Recorder { return &Recorder{limit: MaxRecordedCalls} }

// Observe marks name as a tool the model was offered, for the "available"
// line, and returns a wrapped fn that records each call. The wrapper returns
// exactly what fn returns.
func (r *Recorder) Observe(name string, fn Func) Func {
	r.mu.Lock()
	r.observable = append(r.observable, name)
	r.mu.Unlock()
	return func(ctx context.Context, args map[string]any) (any, error) {
		slot := r.reserve(name, args)
		res, err := fn(ctx, args)
		r.complete(slot, res, err)
		return res, err
	}
}

// ShouldObserve reports whether a tool key is recordable: not a synthetic
// handoff tool, and (when exposed is non-nil) offered by the config.
func ShouldObserve(name string, exposed map[string]bool) bool {
	if strings.HasPrefix(name, HandoffPrefix) {
		return false
	}
	return exposed == nil || exposed[name]
}

func (r *Recorder) reserve(name string, args map[string]any) int {
	var rendered string
	if args != nil {
		rendered = renderValue(args)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.invocations) >= r.limit {
		r.omitted++
		return -1
	}
	r.invocations = append(r.invocations, Invocation{Name: name, Arguments: rendered, Started: time.Now()})
	return len(r.invocations) - 1
}

func (r *Recorder) complete(slot int, res any, err error) {
	if slot < 0 {
		return
	}
	ended := time.Now()
	var result, msg string
	if err != nil {
		msg = truncate(err.Error())
	} else {
		result = renderValue(res)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	inv := &r.invocations[slot]
	inv.Result, inv.Error, inv.Failed = result, msg, err != nil
	inv.Ended, inv.Done = ended, true
}

// Invocations returns the recorded calls, oldest first.
func (r *Recorder) Invocations() []Invocation {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]Invocation(nil), r.invocations...)
}

// Omitted is how many calls ran past the recording limit.
func (r *Recorder) Omitted() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.omitted
}

// Render returns the text a judge reads, or "" when nothing was observable.
func (r *Recorder) Render() string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return Render(r.invocations, r.observable, r.omitted)
}

// Render formats a trajectory: the available tools, then one numbered entry
// per call with its arguments and result or error. A unit that had tools and
// called none says so, because that is what a tool-selection judge grades.
func Render(invocations []Invocation, observable []string, omitted int) string {
	if len(observable) == 0 && len(invocations) == 0 {
		return ""
	}
	var lines []string
	if len(observable) > 0 {
		lines = append(lines, "Tools available: "+strings.Join(observable, ", "))
	}
	if len(invocations) == 0 {
		lines = append(lines, "No tool calls were made while producing the response.")
		return strings.Join(lines, "\n")
	}
	lines = append(lines, "Tool calls made while producing the response, in order:")
	for i, inv := range invocations {
		lines = append(lines, fmt.Sprintf("%d. %s", i+1, inv.Name), "   arguments: "+inv.Arguments)
		if inv.Failed {
			lines = append(lines, "   error: "+inv.Error)
		} else {
			lines = append(lines, "   result: "+inv.Result)
		}
	}
	if omitted > 0 {
		lines = append(lines, fmt.Sprintf("(%d further tool call(s) were made but not recorded.)", omitted))
	}
	return strings.Join(lines, "\n")
}

// renderValue renders a value for a judge: strings verbatim, nil as "",
// everything else compact JSON with sorted keys and literal non-ASCII.
func renderValue(v any) string {
	switch t := v.(type) {
	case nil:
		return ""
	case string:
		return truncate(t)
	}
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false) // a model reads this; keep <, >, & literal
	if err := enc.Encode(v); err != nil {
		return truncate(fmt.Sprint(v))
	}
	return truncate(strings.TrimSuffix(buf.String(), "\n"))
}

// truncate bounds text to MaxValueChars characters (runes, so a multi-byte
// character is never split), marking the cut.
func truncate(s string) string {
	if utf8.RuneCountInString(s) <= MaxValueChars {
		return s
	}
	keep := MaxValueChars - utf8.RuneCountInString(truncationSuffix)
	runes := []rune(s)
	return string(runes[:keep]) + truncationSuffix
}
