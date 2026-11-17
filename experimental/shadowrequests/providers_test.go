package shadowrequests

import (
	"slices"
	"testing"
	"time"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

// The deadline wrapper is what a shadow run routes with, so it must keep the
// handler's provider list: dropping it would let a scoped wildcard run a
// provider outside its list.
func TestWithDeadlineKeepsProviders(t *testing.T) {
	h := &aiserver.ProviderHandler{ProvidesFor: aiserver.ProvidesFor{Provider: "*", Mode: aiserver.ModeMessages}, Providers: []string{"Bedrock"}}
	if got := withDeadline(h, time.Second); !slices.Equal(got.Providers, []string{"Bedrock"}) {
		t.Errorf("Providers = %v, want [Bedrock]", got.Providers)
	}
}
