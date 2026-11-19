// Package judging is the contract every judge path shares: the
// message_history a judge is shown and how its verdict is parsed.
// The online inline path, the online deferred path, and the offline
// evaluations runner all call it, so they cannot drift.
package judging

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"reflect"
	"regexp"
	"strings"
)

// FormattingInstructions tells the judge model the verdict shape. It is
// appended by BuildMessageHistory, not by callers: default judge templates
// read it only through {{message_history}}.
const FormattingInstructions = "Your response MUST be in valid JSON format with the following structure:\n" +
	`{ "score": <number, 0-1>, "reasoning": <string> }` + "\n" +
	"The output must be valid, parseable JSON. Do not include additional tags, comments, formatting, or newlines.\n" +
	"It should be returned in a format that is immediately parseable by a JSON parsing function. Do not include ```json tags."

// ErrInvalidVerdict means the judge did not return a non-empty JSON object.
var ErrInvalidVerdict = errors.New("invalid JSON from judge")

// BuildMessageHistory joins input, trajectory, output and the formatting
// instructions in that order, skipping empty parts.
func BuildMessageHistory(userInput, trajectory, output string) string {
	var parts []string
	for _, p := range []string{userInput, trajectory, output, FormattingInstructions} {
		if p != "" {
			parts = append(parts, p)
		}
	}
	return strings.Join(parts, "\n\n")
}

var (
	openFence  = regexp.MustCompile("^```(?:json)?[ \t]*\r?\n?")
	closeFence = regexp.MustCompile("\r?\n?```$")
)

// ParseJSONWithPossibleFences parses JSON optionally wrapped in markdown
// fences. ok is false when the text is not JSON.
func ParseJSONWithPossibleFences(text string) (value any, ok bool) {
	s := strings.TrimSpace(text)
	s = strings.TrimSpace(closeFence.ReplaceAllString(openFence.ReplaceAllString(s, ""), ""))
	if err := json.Unmarshal([]byte(s), &value); err != nil {
		return nil, false
	}
	return value, true
}

// ParseVerdict parses a judge response (a JSON string, possibly fenced, or a
// decoded object) into its raw score and reasoning. The score is returned
// as given; apply NumericScore for metric use.
func ParseVerdict(raw any) (score any, reasoning string, err error) {
	var obj map[string]any
	switch t := raw.(type) {
	case map[string]any:
		obj = t
	case string:
		v, _ := ParseJSONWithPossibleFences(t)
		obj, _ = v.(map[string]any)
	}
	if len(obj) == 0 {
		return nil, "", ErrInvalidVerdict
	}
	r := reasonText(obj["reasoning"])
	if r == "" {
		r = reasonText(obj["reason"])
	}
	return obj["score"], r, nil
}

// NumericScore returns score as a float only when it is a finite JSON
// number. A bool, string, nil, NaN or infinity is not a score.
func NumericScore(score any) (float64, bool) {
	f, ok := score.(float64)
	if !ok || math.IsNaN(f) || math.IsInf(f, 0) {
		return 0, false
	}
	return f, true
}

// reasonText stringifies a reasoning value: "empty" is judged after
// stringifying, so a number or object still counts as reasoning, but an
// empty list or object does not.
func reasonText(v any) string {
	switch t := v.(type) {
	case nil:
		return ""
	case string:
		return t
	default:
		b, err := json.Marshal(t)
		if err != nil {
			return fmt.Sprint(t)
		}
		if s := string(b); s != "[]" && s != "{}" {
			return s
		}
		return ""
	}
}

// TypesafeEntry is one label of a TypeSafe judge call.
type TypesafeEntry struct {
	Key      string
	EventKey string
	Score    float64
	Reason   string
}

// TypesafeEntries reads a TypeSafe judge payload, `{"kind": "typesafe",
// "results": [...]}` (a JSON string, possibly fenced, or a decoded object).
// It returns nil, nil for any other payload, so the {score, reasoning}
// parser runs, and an error when the payload is TypeSafe but malformed.
func TypesafeEntries(raw any) ([]TypesafeEntry, error) {
	if s, ok := raw.(string); ok {
		raw, _ = ParseJSONWithPossibleFences(s)
	}
	obj, _ := raw.(map[string]any)
	if obj["kind"] != "typesafe" {
		return nil, nil
	}
	results, _ := obj["results"].([]any)
	if len(results) == 0 {
		return nil, errors.New("TypeSafe judge output is missing results")
	}
	entries := make([]TypesafeEntry, len(results))
	for i, r := range results {
		item, _ := r.(map[string]any)
		key, _ := item["key"].(string)
		if key == "" {
			return nil, errors.New("TypeSafe judge result is missing a key")
		}
		score, ok := labelScore(item["score"])
		if !ok {
			return nil, fmt.Errorf("TypeSafe judge result '%s' is missing a finite score", key)
		}
		eventKey, _ := item["eventKey"].(string)
		if eventKey == "" {
			return nil, fmt.Errorf("TypeSafe judge result '%s' is missing an eventKey", key)
		}
		reason, _ := item["reason"].(string)
		entries[i] = TypesafeEntry{Key: key, EventKey: eventKey, Score: score, Reason: reason}
	}
	return entries, nil
}

// labelScore is NumericScore that also takes a Go integer, from a handler
// that returned a literal map rather than decoded JSON. A bool is no score.
func labelScore(v any) (float64, bool) {
	if f, ok := NumericScore(v); ok {
		return f, true
	}
	switch rv := reflect.ValueOf(v); rv.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return float64(rv.Int()), true
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return float64(rv.Uint()), true
	}
	return 0, false
}

// HasOutputFormat reports whether a judge config carries its own
// outputFormat, either parsed or only in the raw variation.
func HasOutputFormat(outputFormat, raw map[string]any) bool {
	_, inRaw := raw["outputFormat"]
	return outputFormat != nil || inRaw
}
