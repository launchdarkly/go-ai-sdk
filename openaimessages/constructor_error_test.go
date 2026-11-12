package openaimessages

import (
	"testing"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

// The constructors return an error so an invalid option can be refused at
// construction without a later breaking change. No option is refused yet.
var _ func(Options) (*aiserver.ProviderHandler, error) = NewHandler

func TestConstructorsReturnNilErrorForValidOptions(t *testing.T) {
	h, err := NewHandler(Options{})
	if err != nil || h == nil {
		t.Fatalf("NewHandler(Options{}) = %v, %v; want a handler and nil", h, err)
	}
}
