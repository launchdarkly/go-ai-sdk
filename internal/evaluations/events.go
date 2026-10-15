package evaluations

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log/slog"
	"reflect"
	"strings"
	"time"
	"unicode/utf16"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/launchdarkly/go-sdk-common/v3/ldvalue"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

// Event names.
const (
	GenerationEventName = "$ld:ai:offline-evals:generation"
	CriterionEventName  = "$ld:ai:offline-evals:criterion"
)

// defaultFlushBound is the most events queued between flushes. The
// Go server SDK's default event capacity is 10 000, the same as Python's.
const defaultFlushBound = 500

// flushWait bounds a blocking flush. The Go server SDK's Flush only starts
// delivery; FlushAndWait is the Go form of awaiting the flush.
const flushWait = 30 * time.Second

// flushWaiter is implemented by *ld.LDClient.
type flushWaiter interface {
	FlushAndWait(timeout time.Duration) bool
}

// canonicalJSON is the eventId serialization: keys sorted,
// compact, and byte-identical to Python's json.dumps(sort_keys=True,
// separators=(",", ":")). Python's default ensure_ascii escapes every
// non-ASCII character as \uXXXX (surrogate pairs above U+FFFF) and leaves
// <, > and & literal. encoding/json escapes <>& unless SetEscapeHTML(false),
// already escapes U+2028/U+2029 the same way, and writes other non-ASCII
// literally, so those are escaped here.
func canonicalJSON(v any) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	var out strings.Builder
	for _, r := range strings.TrimSuffix(buf.String(), "\n") {
		switch {
		case r < 0x80:
			out.WriteRune(r)
		case r > 0xFFFF:
			hi, lo := utf16.EncodeRune(r)
			fmt.Fprintf(&out, `\u%04x\u%04x`, hi, lo)
		default:
			fmt.Fprintf(&out, `\u%04x`, r)
		}
	}
	return []byte(out.String()), nil
}

// eventID is the hex SHA-256 of the canonical identity set.
func eventID(identity map[string]any) (string, error) {
	b, err := canonicalJSON(identity)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:]), nil
}

func timestamp(t time.Time) string { return t.UTC().Format(time.RFC3339Nano) }

func usageWire(u map[string]any) map[string]any {
	n := aiserver.ParseUsage(u)
	return map[string]any{"inputTokens": n.Input, "outputTokens": n.Output}
}

// emitter tracks result events on the SDK client and flushes every bound
// events, because SDK buffer overflow is silent.
type emitter struct {
	client  aiserver.LDClient
	ctx     ldcontext.Context
	bound   int
	pending int
	logger  *slog.Logger
	// identity is the run identity set shared by every event.
	identity map[string]any
	// details are the row-independent detail fields (evaluationKey, ...).
	details map[string]any
}

func newEmitter(client aiserver.LDClient, bound int, logger *slog.Logger, projectKey, evaluationID, runID, evaluationKey string, evaluationVersion int, datasetID, datasetKey string) *emitter {
	identity := map[string]any{"projectKey": projectKey, "evaluationId": evaluationID, "evaluationRunId": runID, "runId": runID}
	details := map[string]any{"evaluationKey": evaluationKey, "evaluationVersion": evaluationVersion}
	if datasetID != "" { // an inline run has no dataset: omit, never null
		identity["datasetId"] = datasetID
		details["datasetKey"] = datasetKey
	}
	ctx := ldcontext.NewBuilder(runID).Kind("evaluation").
		SetString("projectKey", projectKey).SetString("evaluationId", evaluationID).Build()
	return &emitter{client: client, ctx: ctx, bound: bound, logger: logger, identity: identity, details: details}
}

// payload builds an event: identity (+ extra identity), eventId, emittedAt,
// shared details, then the result's own fields. Nil values, including typed
// nils such as a nil map a handler returned as Output, are omitted rather
// than serialized as null.
func (e *emitter) payload(rowIndex int, extraIdentity map[string]any, fields map[string]any) (map[string]any, error) {
	identity := map[string]any{"rowIndex": rowIndex}
	for k, v := range e.identity {
		identity[k] = v
	}
	for k, v := range extraIdentity {
		identity[k] = v
	}
	id, err := eventID(identity)
	if err != nil {
		return nil, err
	}
	out := map[string]any{"eventId": id, "emittedAt": timestamp(time.Now())}
	for _, m := range []map[string]any{identity, e.details, fields} {
		for k, v := range m {
			if !isNil(v) {
				out[k] = v
			}
		}
	}
	return out, nil
}

// track queues one event. A failure building or tracking it is logged and
// skipped so it cannot cost the events around it.
func (e *emitter) track(name string, rowIndex int, build func() (map[string]any, error)) {
	defer func() {
		if r := recover(); r != nil {
			e.logger.Error("failed to emit evaluation event", "event", name, "rowIndex", rowIndex, "panic", r)
		}
	}()
	e.pending++
	if e.pending >= e.bound {
		defer e.flush()
	}
	p, err := build()
	if err == nil {
		var b []byte
		if b, err = json.Marshal(p); err == nil {
			err = e.client.TrackMetric(name, e.ctx, 1, ldvalue.Parse(b))
		}
	}
	if err != nil {
		e.logger.Error("failed to emit evaluation event", "event", name, "rowIndex", rowIndex, "error", err)
	}
}

func isNil(v any) bool {
	if v == nil {
		return true
	}
	switch rv := reflect.ValueOf(v); rv.Kind() {
	case reflect.Map, reflect.Slice, reflect.Pointer, reflect.Interface, reflect.Func, reflect.Chan:
		return rv.IsNil()
	}
	return false
}

func (e *emitter) flush() {
	e.pending = 0
	defer func() {
		if r := recover(); r != nil {
			e.logger.Error("LaunchDarkly client flush failed", "panic", r)
		}
	}()
	if fw, ok := e.client.(flushWaiter); ok {
		if !fw.FlushAndWait(flushWait) {
			e.logger.Warn("LaunchDarkly event flush did not finish in time; events may still be in flight", "timeout", flushWait)
		}
		return
	}
	e.client.Flush()
}
