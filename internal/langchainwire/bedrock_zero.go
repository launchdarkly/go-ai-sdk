package langchainwire

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/smithy-go/middleware"
)

// langchaingo's Bedrock request structs tag temperature `omitempty`, so an
// explicit temperature of 0 never reaches the model and it runs at its
// default (1.0). The handler marks such calls on the context, and this
// client middleware puts the 0 back into the request body.

type zeroTemperatureKey struct{}

func withExplicitZeroTemperature(ctx context.Context) context.Context {
	return context.WithValue(ctx, zeroTemperatureKey{}, true)
}

// newBedrockClient builds a Bedrock Runtime client from the default AWS
// configuration, with the zero-temperature middleware installed.
func newBedrockClient(ctx context.Context) (*bedrockruntime.Client, error) {
	cfg, err := config.LoadDefaultConfig(ctx)
	if err != nil {
		return nil, fmt.Errorf("langchain: load AWS config: %w", err)
	}
	return bedrockruntime.NewFromConfig(cfg, func(o *bedrockruntime.Options) {
		o.APIOptions = append(o.APIOptions, func(s *middleware.Stack) error {
			return s.Initialize.Add(zeroTemperatureMiddleware, middleware.After)
		})
	}), nil
}

var zeroTemperatureMiddleware = middleware.InitializeMiddlewareFunc("LDExplicitZeroTemperature",
	func(ctx context.Context, in middleware.InitializeInput, next middleware.InitializeHandler) (middleware.InitializeOutput, middleware.Metadata, error) {
		if ctx.Value(zeroTemperatureKey{}) != nil {
			switch p := in.Parameters.(type) {
			case *bedrockruntime.InvokeModelInput:
				p.Body = withZeroTemperature(p.Body)
			case *bedrockruntime.InvokeModelWithResponseStreamInput:
				p.Body = withZeroTemperature(p.Body)
			}
		}
		return next.HandleInitialize(ctx, in)
	})

// withZeroTemperature adds "temperature": 0 to an Anthropic-on-Bedrock body
// that lacks it. Other providers' bodies, and anything unparseable, pass
// through unchanged.
// Known limitation: only the Anthropic body shape is patched; other Bedrock model
// families keep langchaingo's behaviour until a config needs them at 0.
func withZeroTemperature(body []byte) []byte {
	var m map[string]any
	if err := json.Unmarshal(body, &m); err != nil {
		return body
	}
	if _, anthropic := m["anthropic_version"]; !anthropic {
		return body
	}
	if _, set := m["temperature"]; set {
		return body
	}
	m["temperature"] = 0
	out, err := json.Marshal(m)
	if err != nil {
		return body
	}
	return out
}
