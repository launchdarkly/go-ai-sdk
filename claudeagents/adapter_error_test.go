package claudeagents

import "github.com/launchdarkly/go-ai-sdk/aiserver"

// The native adapter returns an error so an invalid option can be refused
// when it is built. No option is refused yet.
var _ func(*aiserver.GraphDefinition, *AdapterOptions) (*GraphRunner, error) = ToClaudeAgents

// must panics on err; the adapters return a nil error for every option today.
func must[T any](v T, err error) T {
	if err != nil {
		panic(err)
	}
	return v
}
