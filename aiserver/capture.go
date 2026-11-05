package aiserver

import (
	"context"
	"errors"
	"fmt"
	"iter"

	"github.com/launchdarkly/go-ai-sdk/internal/telemetry"
)

// ContentCapture says whether prompt, completion and tool content is written
// on spans. The zero value, CaptureDefault, is unset: it defers to the
// configured default (Configure), then to the built-in default, on.
type ContentCapture uint8

const (
	CaptureDefault ContentCapture = ContentCapture(telemetry.CaptureDefault)
	CaptureOn      ContentCapture = ContentCapture(telemetry.CaptureOn)
	CaptureOff     ContentCapture = ContentCapture(telemetry.CaptureOff)
)

// SDKOptions are process-wide behaviour defaults, set with Configure. They
// are not init options: InitClient and InitWithClient never change them.
type SDKOptions struct {
	// CaptureContent is the default for a handler whose own setting is
	// CaptureDefault. CaptureDefault here means the built-in default, on.
	CaptureContent ContentCapture
}

// ErrConfigureLocked is returned by Configure when options with a different
// effect are passed after the first AI invocation.
var ErrConfigureLocked = errors.New("aiserver: Configure cannot change options after the first AI invocation; call Shutdown first")

// Configure sets the process-wide defaults. It may replace them until the
// first AI invocation (a ProviderHandler call or the start of its stream);
// after that, options with the same effect are accepted (CaptureDefault and
// CaptureOn both mean on) and others return ErrConfigureLocked. A
// CaptureContent value other than the three constants is an error. Shutdown
// restores the built-in defaults and clears the lock. Handlers read the
// defaults when they run, so a handler built before Configure uses the
// configured value.
func Configure(opts SDKOptions) error {
	if opts.CaptureContent > CaptureOff {
		return fmt.Errorf("aiserver: Configure: invalid ContentCapture %d", opts.CaptureContent)
	}
	if !telemetry.Configure(uint8(opts.CaptureContent)) {
		return ErrConfigureLocked
	}
	return nil
}

func resetOptions() { telemetry.Reset() }

// enabled resolves the setting: an explicit CaptureOn or CaptureOff wins,
// then the configured default, then on.
func (c ContentCapture) enabled() bool { return telemetry.Resolve(uint8(c)) }

// startInvocation latches Configure and binds c on ctx for one handler call.
func startInvocation(ctx context.Context, c ContentCapture) context.Context {
	return telemetry.StartInvocation(ctx, uint8(c))
}

// bindCall is startInvocation for the core call path: the binding is pending,
// so the handler's own CreateHandler wrapper keeps it rather than re-binding
// the value of the handler it was built from.
func bindCall(ctx context.Context, c ContentCapture) context.Context {
	return telemetry.BindCall(ctx, uint8(c))
}

// bindInvocation wraps a handler's functions so a direct call (not through
// Config or a judge) still starts an invocation with the handler's setting.
// It reads h.CaptureContent at call time, so a later change to the field is
// honoured. Through the core, the core's binding (from the handler value it
// routed to, which may be a copy of h) is kept.
func bindInvocation(h *ProviderHandler) *ProviderHandler {
	invoke, stream := h.Invoke, h.Stream
	if invoke != nil {
		h.Invoke = func(ctx context.Context, req HandlerRequest) (HandlerResult, error) {
			return invoke(telemetry.Enter(ctx, uint8(h.CaptureContent)), req)
		}
	}
	if stream != nil {
		h.Stream = func(ctx context.Context, req HandlerRequest) iter.Seq2[HandlerStreamEvent, error] {
			return stream(telemetry.Enter(ctx, uint8(h.CaptureContent)), req)
		}
	}
	return h
}
