// Package telemetry holds the SDK's span-content gate and span helpers shared
// by aiserver, the handler packages and the experimental runners. It is
// internal so these primitives can change without breaking the public API;
// aiserver.Configure, SDKOptions and ContentCapture are the public face.
package telemetry

import (
	"context"
	"sync"
	"sync/atomic"
)

// Capture values, matching aiserver.ContentCapture.
const (
	CaptureDefault uint8 = iota
	CaptureOn
	CaptureOff
)

var (
	mu sync.Mutex
	// defaultOff is the resolved process default (Configure); false is the
	// built-in default, on.
	defaultOff bool
	// locked latches on the first invocation; Reset clears it.
	locked bool
)

// Configure sets the process default from c (CaptureDefault means the
// built-in default, on). Before the first invocation it always succeeds;
// after it, it succeeds only when the resolved value is unchanged.
func Configure(c uint8) bool {
	mu.Lock()
	defer mu.Unlock()
	off := c == CaptureOff
	if locked && off != defaultOff {
		return false
	}
	defaultOff = off
	return true
}

// Reset restores the built-in default and clears the lock.
func Reset() {
	mu.Lock()
	defer mu.Unlock()
	defaultOff, locked = false, false
}

// Resolve is c's effective value under the current process default, without
// latching Configure.
func Resolve(c uint8) bool {
	switch c {
	case CaptureOn:
		return true
	case CaptureOff:
		return false
	}
	mu.Lock()
	defer mu.Unlock()
	return !defaultOff
}

type captureKey struct{}

// binding is one handler call's resolved capture setting. pending marks a
// binding the core made for a handler whose own wrapper (Enter) is about to
// run: that wrapper keeps the core's binding instead of re-binding its own.
type binding struct {
	on      bool
	pending atomic.Bool
}

// StartInvocation marks the start of one handler call: it latches Configure
// and binds c, resolved against the process default, on ctx. A CaptureDefault
// binding resolves to the process default, so a nested call with no setting
// of its own does not inherit an enclosing handler's explicit value.
func StartInvocation(ctx context.Context, c uint8) context.Context {
	return context.WithValue(ctx, captureKey{}, newBinding(c))
}

func newBinding(c uint8) *binding {
	mu.Lock()
	locked = true
	on := !defaultOff
	mu.Unlock()
	switch c {
	case CaptureOn:
		on = true
	case CaptureOff:
		on = false
	}
	return &binding{on: on}
}

// BindCall is StartInvocation for the core call path (Config, judges,
// evaluations), which knows the handler value it routed to. The binding is
// pending, so the handler's CreateHandler wrapper, which closes over the
// handler it was built from, keeps this binding: a copied handler with a
// different CaptureContent is honoured.
func BindCall(ctx context.Context, c uint8) context.Context {
	b := newBinding(c)
	b.pending.Store(true)
	return context.WithValue(ctx, captureKey{}, b)
}

// Enter is the CreateHandler wrapper's binding: it consumes a pending core
// binding on ctx, or else starts a new invocation with the handler's own c.
// Either way, nested calls under the returned context bind afresh.
func Enter(ctx context.Context, c uint8) context.Context {
	if b, _ := ctx.Value(captureKey{}).(*binding); b != nil && b.pending.CompareAndSwap(true, false) {
		return ctx
	}
	return StartInvocation(ctx, c)
}

// Settle clears a pending binding on ctx, for a boundary such as a tool call
// that a handler without an Enter wrapper (a struct literal) reaches: a
// handler invoked from there binds its own setting.
func Settle(ctx context.Context) {
	if b, _ := ctx.Value(captureKey{}).(*binding); b != nil {
		b.pending.Store(false)
	}
}

// Enabled reports whether content may be written on spans for the handler
// call ctx belongs to; outside any handler call it is the process default.
func Enabled(ctx context.Context) bool {
	if b, _ := ctx.Value(captureKey{}).(*binding); b != nil {
		return b.on
	}
	return Resolve(CaptureDefault)
}
