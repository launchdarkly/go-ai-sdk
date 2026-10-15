package aiserver

import (
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"

	"github.com/launchdarkly/go-ai-sdk/internal/judging"
)

var placeholderRE = regexp.MustCompile(`\{\{([\w.]+)\}\}`)

// ParseTemplate replaces {{path}} placeholders with values from variables.
// Unknown paths are left as-is.
func ParseTemplate(template string, variables map[string]any) string {
	return placeholderRE.ReplaceAllStringFunc(template, func(match string) string {
		var cur any = variables
		for _, part := range strings.Split(match[2:len(match)-2], ".") {
			m, ok := cur.(map[string]any)
			if !ok {
				return match
			}
			if cur, ok = m[part]; !ok {
				return match
			}
		}
		if cur == nil {
			return match
		}
		return stringify(cur)
	})
}

func stringify(v any) string {
	switch t := v.(type) {
	case string:
		return t
	case float64:
		return strconv.FormatFloat(t, 'f', -1, 64)
	case bool, int, int64:
		return fmt.Sprint(t)
	default:
		// Known limitation: other SDKs disagree on how objects/arrays coerce (TS gives
		// "[object Object]", Python a dict repr); JSON is the only sane choice.
		b, err := json.Marshal(t)
		if err != nil {
			return fmt.Sprint(t)
		}
		return string(b)
	}
}

// ParseJSONWithPossibleFences parses JSON optionally wrapped in markdown
// fences. ok is false when the text is not JSON; it never logs.
func ParseJSONWithPossibleFences(text string) (value any, ok bool) {
	return judging.ParseJSONWithPossibleFences(text)
}

var usageKeyPairs = []struct {
	in, out                 string
	cacheRead, cacheCreated []string
}{
	{"input_tokens", "output_tokens", []string{"cache_read_input_tokens"}, []string{"cache_creation_input_tokens"}},
	{"inputTokens", "outputTokens", []string{"cacheReadInputTokens"}, []string{"cacheCreationInputTokens", "cacheWriteInputTokens"}},
	{"input", "output", nil, nil},
}

func numberOrZero(v any) float64 {
	var f float64
	switch t := v.(type) {
	case float64:
		f = t
	case int:
		f = float64(t)
	case int64:
		f = float64(t)
	case json.Number:
		f, _ = t.Float64()
	}
	if math.IsNaN(f) || math.IsInf(f, 0) {
		return 0
	}
	return f
}

// ParseUsage normalises provider token counts. The first key pair with
// both keys present wins; total is always computed. Cache fields are folded
// into input as the TypeScript implementation does.
func ParseUsage(usage map[string]any) TokenUsage {
	for _, p := range usageKeyPairs {
		inV, hasIn := usage[p.in]
		outV, hasOut := usage[p.out]
		if !hasIn || !hasOut {
			continue
		}
		var cacheRead, cacheCreated float64
		hasDetails := false
		for _, k := range p.cacheRead {
			_, present := usage[k]
			hasDetails = hasDetails || present
			cacheRead += numberOrZero(usage[k])
		}
		for _, k := range p.cacheCreated {
			_, present := usage[k]
			hasDetails = hasDetails || present
			cacheCreated += numberOrZero(usage[k])
		}
		uncached := numberOrZero(inV)
		input := uncached + cacheRead + cacheCreated
		output := numberOrZero(outV)
		u := TokenUsage{Input: input, Output: output, Total: input + output}
		if hasDetails {
			u.InputDetails = &InputDetails{Uncached: uncached, CacheRead: cacheRead, CacheCreation: cacheCreated}
		}
		return u
	}
	return TokenUsage{}
}

// NormalizeMode maps a variation mode to a handler mode.
func NormalizeMode(mode string) Mode {
	if mode == "agent" {
		return ModeAgent
	}
	return ModeMessages
}
