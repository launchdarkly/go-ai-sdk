package aiserver

import (
	"context"
	"sync"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/launchdarkly/go-ai-sdk/internal/telemetry"
)

// judgeScope holds a judge handler's root span open after the handler
// finishes, so the score can be written onto it. Only handlers built
// on HandlerRun participate; a custom handler's span ends normally.
type judgeScope struct {
	key     string
	mu      sync.Mutex
	span    trace.Span
	endedAt time.Time
}

type judgeScopeKey struct{}

func judgeScopeFrom(ctx context.Context) *judgeScope {
	s, _ := ctx.Value(judgeScopeKey{}).(*judgeScope)
	return s
}

// hold claims the first root span ended inside the scope. It returns false
// when the span should be ended normally.
func (s *judgeScope) hold(span trace.Span) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.span != nil {
		return false
	}
	s.span, s.endedAt = span, time.Now()
	return true
}

// record writes the evaluation event and mirrored attributes. explanation is
// written only when non-empty (the caller applies the captureContent gate).
func (s *judgeScope) record(score float64, explanation string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.span == nil {
		return
	}
	attrs := []attribute.KeyValue{
		attribute.String("gen_ai.evaluation.name", s.key),
		attribute.Float64("gen_ai.evaluation.score.value", score),
	}
	if explanation != "" {
		attrs = append(attrs, attribute.String("gen_ai.evaluation.explanation", explanation))
	}
	s.span.AddEvent("gen_ai.evaluation.result", trace.WithAttributes(attrs...))
	s.span.SetAttributes(attrs...)
}

// release ends a held span at the handler's own end time, so tracking and
// parsing after the handler returned are not counted in its duration.
func (s *judgeScope) release() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.span != nil {
		s.span.End(trace.WithTimestamp(s.endedAt))
	}
}

// withJudgeEvaluation runs fn with a judge scope on ctx. fn calls record
// after it has a finite score. target is the graded run's track data, the
// one the judge's metric event carries: every SDK span the judge starts gets
// its non-empty run id, config key and variation key, so a judge can be
// joined to the run it graded.
func withJudgeEvaluation(ctx context.Context, judgeKey string, target TrackData, fn func(ctx context.Context, record func(score float64, explanation string)) error) error {
	s := &judgeScope{key: judgeKey}
	defer s.release()
	ctx = telemetry.WithSpanAttributes(ctx, judgeTargetAttributes(target)...)
	return fn(context.WithValue(ctx, judgeScopeKey{}, s), s.record)
}

// judgeTargetAttributes are the launchdarkly.judge.target.* attributes for
// the graded run, each only when non-empty.
func judgeTargetAttributes(td TrackData) []attribute.KeyValue {
	var attrs []attribute.KeyValue
	for _, a := range [...]struct{ key, value string }{
		{"launchdarkly.judge.target.run.id", td.RunID},
		{"launchdarkly.judge.target.config.key", td.ConfigKey},
		{"launchdarkly.judge.target.variation.key", td.VariationKey},
	} {
		if a.value != "" {
			attrs = append(attrs, attribute.String(a.key, a.value))
		}
	}
	return attrs
}
