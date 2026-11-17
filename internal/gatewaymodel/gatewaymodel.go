// Package gatewaymodel builds the Vercel AI Gateway "creator/model" id for an
// AI Config. It is the Go port of the TypeScript and Python Vercel handlers'
// gateway model id mapping, and must stay in sync with both.
package gatewaymodel

import (
	"errors"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

// The conditions ID fails on. Each error ID returns wraps one of these and
// names the provider and model.
var (
	// ErrCreatorNotServed means the provider's own models are not in the Gateway catalog.
	ErrCreatorNotServed = errors.New("Vercel AI Gateway currently exposes no models created by this provider")
	// ErrMultipleCreators means the provider hosts several creators and the model names none.
	ErrMultipleCreators = errors.New("provider hosts models from multiple creators")
	// ErrUnknownProvider means the provider has no Gateway creator.
	ErrUnknownProvider = errors.New("cannot map provider to a Vercel AI Gateway creator")
)

// ID returns the AI Gateway model id for cfg: an explicit "creator/model" name
// unchanged, otherwise the creator read from the model name, then from the
// provider. It returns an error when no single creator can be determined.
func ID(cfg aiserver.AiConfig) (string, error) {
	return "", errors.New("gatewaymodel: not implemented")
}
