package typesafe

import (
	"errors"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
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

// ExtractQuestions reads the Jev questions from the judge variation's
// classifiers list, in order (extractTypesafeQuestions).
func ExtractQuestions(cfg aiserver.AiConfig) ([]Question, error) {
	return nil, errNotImplemented
}
