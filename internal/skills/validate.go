package skills

import (
	"encoding/json"
	"math"
	"regexp"
	"strconv"
)

// skillKeyGrammar is quoted in rejection messages. RE2's $ matches only at the
// end of the text, so "key\n" is rejected.
const skillKeyGrammar = `^[a-z0-9][a-z0-9-]*$`

var skillKeyRE = regexp.MustCompile(skillKeyGrammar)

// skillKeyMaxLength is the data model's bound. WriteSkills
// applies the tighter filesystem bound maxPathComponentBytes.
const skillKeyMaxLength = 256

var sha256HexRE = regexp.MustCompile(`^[0-9a-f]{64}$`)

// skillKeyRejection says why v is not a valid skill key, or "" when it is.
func skillKeyRejection(v any) string {
	key, ok := v.(string)
	if !ok {
		return "must be a string"
	}
	if len(key) > skillKeyMaxLength {
		return "must be at most 256 characters"
	}
	if !skillKeyRE.MatchString(key) {
		return "must match " + skillKeyGrammar
	}
	return ""
}

func isValidSkillKey(v any) bool { return skillKeyRejection(v) == "" }

// versionOf reads a skill version: an integer >= 1. It accepts the integer
// kinds Go code builds raw objects with, a float64 holding an integer (what
// encoding/json produces) and an integral json.Number. A bool never counts.
func versionOf(v any) (int, bool) {
	var n int64
	switch x := v.(type) {
	case int:
		n = int64(x)
	case int8:
		n = int64(x)
	case int16:
		n = int64(x)
	case int32:
		n = int64(x)
	case int64:
		n = x
	case uint8:
		n = int64(x)
	case uint16:
		n = int64(x)
	case uint32:
		n = int64(x)
	case uint:
		if uint64(x) > math.MaxInt64 {
			return 0, false
		}
		n = int64(x)
	case uint64:
		if x > math.MaxInt64 {
			return 0, false
		}
		n = int64(x)
	case float64:
		if x != math.Trunc(x) || x < 1 || x > 1<<53 {
			return 0, false
		}
		n = int64(x)
	case float32:
		return versionOf(float64(x))
	case json.Number:
		i, err := strconv.ParseInt(string(x), 10, 64)
		if err != nil {
			return 0, false
		}
		n = i
	default:
		return 0, false
	}
	if n < 1 || n > math.MaxInt {
		return 0, false
	}
	return int(n), true
}

func isValidSkillVersion(v any) bool {
	_, ok := versionOf(v)
	return ok
}
