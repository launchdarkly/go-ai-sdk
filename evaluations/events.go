package evaluations

import (
	"bytes"
	"cmp"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"time"
	"unicode/utf16"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/launchdarkly/go-sdk-common/v3/ldvalue"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

// Event names.
const (
	generationEventName = "$ld:ai:offline-evals:generation"
	criterionEventName  = "$ld:ai:offline-evals:criterion"
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

// canonicalJSON is RFC 8785 canonical JSON (JCS), the form eventId and
// contentHash hash in every SDK: no whitespace, object keys sorted by UTF-16
// code unit, strings and numbers spelled as JavaScript's JSON.stringify spells
// them. Values are first normalized by jcsValue.
func canonicalJSON(v any) ([]byte, error) {
	t, err := jcsValue(v)
	if err != nil {
		return nil, err
	}
	var b strings.Builder
	if err := writeJCS(&b, t); err != nil {
		return nil, err
	}
	return []byte(b.String()), nil
}

// jcsValue returns v as the JSON tree canonicalJSON writes, built only from
// nil, bool, string, float64, map[string]any and []any. Every number becomes
// a float64 (so an integer above 2^53 becomes the nearest double), -0 becomes
// 0, and NaN or ±Inf becomes nil (null), as JSON.stringify writes them. The
// RFC would reject non-finite numbers instead; the SDK contract writes null.
// The generation event carries this tree too, so it holds what was hashed.
func jcsValue(v any) (any, error) {
	if isNil(v) {
		return nil, nil
	}
	switch x := v.(type) {
	case bool, string:
		return x, nil
	case json.Number:
		f, _ := strconv.ParseFloat(string(x), 64) // out of range parses to ±Inf, written as null
		return jcsNumber(f), nil
	case map[string]any:
		out := make(map[string]any, len(x))
		for k, e := range x {
			n, err := jcsValue(e)
			if err != nil {
				return nil, err
			}
			out[k] = n
		}
		return out, nil
	case []any:
		out := make([]any, len(x))
		for i, e := range x {
			n, err := jcsValue(e)
			if err != nil {
				return nil, err
			}
			out[i] = n
		}
		return out, nil
	}
	switch rv := reflect.ValueOf(v); rv.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return jcsNumber(float64(rv.Int())), nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return jcsNumber(float64(rv.Uint())), nil
	case reflect.Float32, reflect.Float64:
		return jcsNumber(rv.Float()), nil
	}
	// Any other value (a struct, a typed map or slice) is hashed as the JSON
	// it marshals to.
	// ponytail: a struct holding NaN or ±Inf fails json.Marshal here and its
	// event is logged and dropped; walk structs by reflection if handlers
	// ever return such values.
	b, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.UseNumber()
	var decoded any
	if err := dec.Decode(&decoded); err != nil {
		return nil, err
	}
	return jcsValue(decoded)
}

func jcsNumber(f float64) any {
	if math.IsNaN(f) || math.IsInf(f, 0) {
		return nil
	}
	return f + 0 // -0 + 0 is +0
}

// writeJCS writes a jcsValue tree. Any other type is an error, never
// silently skipped, so a gap in jcsValue cannot change a hash unnoticed.
func writeJCS(b *strings.Builder, v any) error {
	switch x := v.(type) {
	case nil:
		b.WriteString("null")
	case bool:
		b.WriteString(strconv.FormatBool(x))
	case float64:
		// encoding/json spells a finite float64 as ECMAScript does.
		n, err := json.Marshal(x)
		if err != nil {
			return err
		}
		b.Write(n)
	case string:
		writeJCSString(b, x)
	case []any:
		b.WriteByte('[')
		for i, e := range x {
			if i > 0 {
				b.WriteByte(',')
			}
			if err := writeJCS(b, e); err != nil {
				return err
			}
		}
		b.WriteByte(']')
	case map[string]any:
		keys := make([]string, 0, len(x))
		for k := range x {
			keys = append(keys, k)
		}
		slices.SortFunc(keys, func(a, c string) int {
			// Distinct keys can share UTF-16 units (invalid UTF-8 becomes
			// U+FFFD), so break the tie on the bytes for a stable order.
			return cmp.Or(slices.Compare(utf16.Encode([]rune(a)), utf16.Encode([]rune(c))), cmp.Compare(a, c))
		})
		b.WriteByte('{')
		for i, k := range keys {
			if i > 0 {
				b.WriteByte(',')
			}
			writeJCSString(b, k)
			b.WriteByte(':')
			if err := writeJCS(b, x[k]); err != nil {
				return err
			}
		}
		b.WriteByte('}')
	default:
		return fmt.Errorf("evaluations: canonical JSON: unsupported type %T", v)
	}
	return nil
}

// writeJCSString escapes only what JSON.stringify escapes: the quote, the
// backslash and control characters. Everything else, U+2028 and U+2029
// included, is written as is. Invalid UTF-8 becomes U+FFFD.
func writeJCSString(b *strings.Builder, s string) {
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			b.WriteString(`\"`)
		case '\\':
			b.WriteString(`\\`)
		case '\b':
			b.WriteString(`\b`)
		case '\f':
			b.WriteString(`\f`)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		default:
			if r < 0x20 {
				fmt.Fprintf(b, `\u%04x`, r)
			} else {
				b.WriteRune(r)
			}
		}
	}
	b.WriteByte('"')
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

// contentHash is the hex SHA-256 of the canonical JSON of what a row
// generated (status, output, error, and errorMessage and usage when set), so
// the same content hashes the same in every SDK whatever the row or run.
func contentHash(content map[string]any) (string, error) {
	b, err := canonicalJSON(content)
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
