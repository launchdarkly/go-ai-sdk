package aiserver

import (
	"context"
	"errors"
	"iter"
	"sync"
	"sync/atomic"
)

// ContentCapture says whether prompt, completion and tool content is written
// on spans. The zero value, CaptureDefault, is unset: it defers to the
// configured default (Configure), then to the built-in default, on.
type ContentCapture uint8

const (
	CaptureDefault ContentCapture = iota
	CaptureOn
	CaptureOff
)

// SDKOptions are process-wide behaviour defaults, set with Configure. They
// are not init options: InitClient and InitWithClient never change them.
type SDKOptions struct {
	// CaptureContent is the default for a handler whose own setting is
	// CaptureDefault. CaptureDefault here means the built-in default, on.
	CaptureContent ContentCapture
}

// ErrConfigureLocked is returned by Configure when different options are
// passed after the first AI invocation.
var ErrConfigureLocked = errors.New("aiserver: Configure cannot change options after the first AI invocation; call Shutdown first")

var (
	optionsMu sync.Mutex
	options   SDKOptions
	// optionsLocked latches on the first ProviderHandler call or stream
	// start; Shutdown clears it.
	optionsLocked atomic.Bool
)

// Configure sets the process-wide defaults. It may replace them until the
// first AI invocation (a ProviderHandler call or the start of its stream);
// after that the same options are accepted and different ones return
// ErrConfigureLocked. Shutdown restores the built-in defaults and the lock.
// Handlers read the defaults when they run, so a handler built before
// Configure uses the configured value.
func Configure(opts SDKOptions) error {
	optionsMu.Lock()
	defer optionsMu.Unlock()
	if optionsLocked.Load() && opts != options {
		return ErrConfigureLocked
	}
	options = opts
	return nil
}

func resetOptions() {
	optionsMu.Lock()
	defer optionsMu.Unlock()
	options = SDKOptions{}
	optionsLocked.Store(false)
}

// enabled resolves the setting: an explicit CaptureOn or CaptureOff wins,
// then the configured default, then on.
func (c ContentCapture) enabled() bool {
	switch c {
	case CaptureOn:
		return true
	case CaptureOff:
		return false
	}
	optionsMu.Lock()
	d := options.CaptureContent
	optionsMu.Unlock()
	return d != CaptureOff
}

type captureKey struct{}

// startInvocation marks the start of one handler call: it latches Configure
// and binds the handler's capture setting on ctx. The binding is set even for
// CaptureDefault, so a nested call with no setting of its own follows the
// process default rather than the enclosing handler's explicit value.
func startInvocation(ctx context.Context, c ContentCapture) context.Context {
	optionsLocked.Store(true)
	return context.WithValue(ctx, captureKey{}, c)
}

// CaptureContentEnabled reports whether content may be written on spans for
// the handler call ctx belongs to: that handler's explicit setting, else the
// configured default, else on. Outside any handler call it is the process
// default.
func CaptureContentEnabled(ctx context.Context) bool {
	c, _ := ctx.Value(captureKey{}).(ContentCapture)
	return c.enabled()
}

// bindInvocation wraps a handler's functions so a direct call (not through
// Config or a judge) still starts an invocation with the handler's setting.
// It reads h.CaptureContent at call time, so a later change to the field is
// honoured.
func bindInvocation(h *ProviderHandler) *ProviderHandler {
	invoke, stream := h.Invoke, h.Stream
	if invoke != nil {
		h.Invoke = func(ctx context.Context, req HandlerRequest) (HandlerResult, error) {
			return invoke(startInvocation(ctx, h.CaptureContent), req)
		}
	}
	if stream != nil {
		h.Stream = func(ctx context.Context, req HandlerRequest) iter.Seq2[HandlerStreamEvent, error] {
			return stream(startInvocation(ctx, h.CaptureContent), req)
		}
	}
	return h
}
