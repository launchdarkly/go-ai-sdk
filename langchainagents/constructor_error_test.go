package langchainagents

import (
	"testing"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

// The constructors return an error so an invalid option (an invalid
// Options.Providers) is refused at construction.
var _ func(Options) (*aiserver.ProviderHandler, error) = NewHandler
var _ func(string, aiserver.GraphOptions, Options) (*aiserver.AgentGraph, error) = LangChainGraph

func TestConstructorsReturnNilErrorForValidOptions(t *testing.T) {
	h, err := NewHandler(Options{})
	if err != nil || h == nil {
		t.Fatalf("NewHandler(Options{}) = %v, %v; want a handler and nil", h, err)
	}
	g, err := LangChainGraph("g", aiserver.GraphOptions{}, Options{})
	if err != nil || g == nil {
		t.Fatalf("LangChainGraph = %v, %v; want a graph and nil", g, err)
	}
}
