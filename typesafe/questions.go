package typesafe

import (
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/judging"
)

// QuestionType is a Jev question type.
type QuestionType string

const (
	Noul   QuestionType = "noul"
	Choice QuestionType = "choice"
	Score  QuestionType = "score"
)

// Question is one Jev question read from a classifiers entry. Criteria is the
// entry's decoded JSON criteria: a map for noul and choice, a list for score,
// nil for a noul without criteria.
type Question struct {
	Key          string
	EventKey     string
	Type         QuestionType
	Instructions string
	Criteria     any
}

// ErrInvalidClassifiers is wrapped by every ExtractQuestions error.
var ErrInvalidClassifiers = errors.New("typesafe: invalid classifiers")

func invalid(format string, args ...any) error {
	return fmt.Errorf("%w: "+format, append([]any{ErrInvalidClassifiers}, args...)...)
}

// ExtractQuestions reads the Jev questions from the judge variation's
// classifiers list, in order (extractTypesafeQuestions).
func ExtractQuestions(cfg aiserver.AiConfig) ([]Question, error) {
	if len(cfg.Classifiers) == 0 {
		return nil, invalid("classifiers must be a non-empty list")
	}
	keys, events := map[string]bool{}, map[string]bool{}
	out := make([]Question, 0, len(cfg.Classifiers))
	for i, label := range cfg.Classifiers {
		if label == nil {
			return nil, invalid("classifiers[%d] must be an object", i)
		}
		key, _ := label["key"].(string)
		if strings.TrimSpace(key) == "" {
			return nil, invalid("classifiers[%d] is missing a key", i)
		}
		if keys[key] {
			return nil, invalid("classifiers has a duplicate key %q", key)
		}
		keys[key] = true
		eventKey, _ := label["eventKey"].(string)
		if strings.TrimSpace(eventKey) == "" {
			return nil, invalid("classifiers label %q is missing an eventKey", key)
		}
		if events[eventKey] {
			return nil, invalid("classifiers has a duplicate eventKey %q", eventKey)
		}
		events[eventKey] = true
		typ, _ := label["type"].(string)
		qt := QuestionType(typ)
		if qt != Noul && qt != Choice && qt != Score {
			return nil, invalid("classifiers label %q has an unsupported type", key)
		}
		instructions, _ := label["instructions"].(string)
		if strings.TrimSpace(instructions) == "" {
			return nil, invalid("classifiers label %q is missing instructions", key)
		}
		raw, has := label["criteria"]
		criteria, err := criteriaFor(key, qt, raw, has)
		if err != nil {
			return nil, err
		}
		out = append(out, Question{Key: key, EventKey: eventKey, Type: qt, Instructions: instructions, Criteria: criteria})
	}
	return out, nil
}

// stringOrNull reports whether v is a JSON string or null.
func stringOrNull(v any) bool {
	_, isString := v.(string)
	return v == nil || isString
}

func criteriaFor(key string, qt QuestionType, raw any, has bool) (any, error) {
	switch qt {
	case Noul:
		if !has {
			return nil, nil
		}
		m, isMap := raw.(map[string]any)
		if !isMap {
			return nil, invalid("classifiers label %q has invalid noul criteria", key)
		}
		criteria := map[string]any{}
		for _, side := range []string{"true", "false"} {
			d, has := m[side]
			if !has {
				continue
			}
			if !stringOrNull(d) {
				return nil, invalid("classifiers label %q has a non-string noul description", key)
			}
			criteria[side] = d
		}
		if len(criteria) == 0 {
			return nil, nil
		}
		return criteria, nil
	case Choice:
		m, isMap := raw.(map[string]any)
		if !isMap || len(m) == 0 {
			return nil, invalid("classifiers label %q requires choice criteria", key)
		}
		criteria := make(map[string]any, len(m))
		for name, d := range m {
			if strings.TrimSpace(name) == "" {
				return nil, invalid("classifiers label %q has an empty choice", key)
			}
			if !stringOrNull(d) {
				return nil, invalid("classifiers label %q has a non-string choice description", key)
			}
			criteria[name] = d
		}
		return criteria, nil
	}
	levels, isList := raw.([]any)
	if !isList || len(levels) < 2 {
		return nil, invalid("classifiers label %q requires at least two score levels", key)
	}
	for _, l := range levels {
		if !stringOrNull(l) {
			return nil, invalid("classifiers label %q has a non-string score level", key)
		}
	}
	return append([]any(nil), levels...), nil
}

// buildState is everything the judge was given except the shared
// {score, reasoning} formatting block. Missing, null and empty-string values
// are left out.
func buildState(req aiserver.HandlerRequest) map[string]any {
	state := map[string]any{}
	assign := func(key string, v any) {
		if v == nil || v == "" {
			return
		}
		state[key] = v
	}
	vars := req.Variables
	assign("input", vars["input"])
	var output any = req.UserInput
	if v, has := vars["response_to_evaluate"]; has && v != nil {
		output = v
	}
	assign("output", output)
	if mh, isText := vars["message_history"].(string); isText {
		if i := strings.Index(mh, judging.FormattingInstructions); i >= 0 {
			mh = mh[:i]
		}
		assign("message_history", strings.TrimSpace(mh))
	}
	assign("trajectory", vars["trajectory"])
	assign("expected_output", vars["expected_output"])
	assign("ground_truth_context", vars["ground_truth_context"])
	if req.UserInput != "" && output != any(req.UserInput) {
		state["user_input"] = req.UserInput
	}
	if len(req.History) > 0 {
		state["history"] = req.History
	}
	return state
}

func finite(v any) (float64, bool) {
	f, isNum := v.(float64)
	return f, isNum && !math.IsNaN(f) && !math.IsInf(f, 0)
}

func clamp(v float64) float64 { return min(1, max(0, v)) }

// field reads one field of a decoded answer object; nil when absent.
func field(answer any, name string) any {
	m, _ := answer.(map[string]any)
	return m[name]
}

// scoreFromAnswer maps one Jev answer onto the 0-1 judge score. A missing
// value is an error.
func scoreFromAnswer(q Question, answer any) (float64, error) {
	switch q.Type {
	case Noul:
		p, isNum := finite(field(answer, "noul"))
		if !isNum {
			return 0, fmt.Errorf("typesafe: answer %q is missing a noul probability", q.Key)
		}
		return clamp(p), nil
	case Choice:
		selected, _ := field(answer, "choice").(string)
		if probs, isMap := field(answer, "probabilities").(map[string]any); isMap && selected != "" {
			if p, isNum := finite(probs[selected]); isNum {
				return clamp(p), nil
			}
		}
		if c, isNum := finite(field(answer, "confidence")); isNum {
			return clamp(c), nil
		}
		if selected != "" {
			return 1, nil
		}
		return 0, fmt.Errorf("typesafe: answer %q is missing a choice", q.Key)
	}
	s, isNum := finite(field(answer, "score"))
	if !isNum {
		return 0, fmt.Errorf("typesafe: answer %q is missing a score", q.Key)
	}
	levels := 0
	if legend, isMap := field(answer, "legend").(map[string]any); isMap {
		levels = len(legend)
	} else if c, isList := q.Criteria.([]any); isList {
		levels = len(c)
	}
	// Jev's score is an expected rubric index; two levels divide by 1.
	if levels > 1 {
		return clamp(s / float64(levels-1)), nil
	}
	return clamp(s), nil
}

func criterionText(v any) string {
	s, _ := v.(string)
	if strings.TrimSpace(s) == "" {
		return ""
	}
	return s
}

// nearest is the key closest to score; ties go to the lower key.
func nearest(score float64, keys []float64) float64 {
	best := keys[0]
	for _, k := range keys[1:] {
		if d, bd := math.Abs(k-score), math.Abs(best-score); d < bd || (d == bd && k < best) {
			best = k
		}
	}
	return best
}

// reasonFromAnswer is the selected label value, stored as the result reason.
func reasonFromAnswer(q Question, answer any) string {
	switch q.Type {
	case Noul:
		p, _ := finite(field(answer, "noul"))
		side := "false"
		if p >= 0.5 {
			side = "true"
		}
		if m, isMap := q.Criteria.(map[string]any); isMap {
			if d := criterionText(m[side]); d != "" {
				return d
			}
		}
		return side
	case Choice:
		name, _ := field(answer, "choice").(string)
		if m, isMap := q.Criteria.(map[string]any); isMap && name != "" {
			if d := criterionText(m[name]); d != "" {
				return d
			}
		}
		return name
	}
	s, isNum := finite(field(answer, "score"))
	if !isNum {
		return ""
	}
	if legend, isMap := field(answer, "legend").(map[string]any); isMap {
		var keys []float64
		for k := range legend {
			if f, err := strconv.ParseFloat(k, 64); err == nil && !math.IsInf(f, 0) && !math.IsNaN(f) {
				keys = append(keys, f)
			}
		}
		if len(keys) > 0 {
			if d := criterionText(legend[strconv.FormatFloat(nearest(s, keys), 'f', -1, 64)]); d != "" {
				return d
			}
		}
	}
	if c, isList := q.Criteria.([]any); isList && len(c) > 0 {
		idx := make([]float64, len(c))
		for i := range c {
			idx[i] = float64(i)
		}
		if d := criterionText(c[int(nearest(s, idx))]); d != "" {
			return d
		}
	}
	return ""
}
