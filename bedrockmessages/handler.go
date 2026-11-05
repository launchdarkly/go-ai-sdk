// Package bedrockmessages is the Amazon Bedrock Runtime Converse handler for
// aiserver (providesFor Bedrock/messages). It drives Converse for blocking
// calls and ConverseStream for streaming, runs the local tool loop, and emits
// the provider-neutral invoke_agent → chat / execute_tool span tree.
package bedrockmessages

import (
	"context"
	"fmt"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/bedrockwire"
	"github.com/launchdarkly/go-ai-sdk/internal/sdkapi"
)

const (
	// PackageName is this package's identity for $ld:ai:sdk:info and
	// $ld:ai:sdk:usage.
	PackageName = "github.com/launchdarkly/go-ai-sdk/bedrockmessages"
	// PackageVersion is this package's release version.
	PackageVersion = "0.1.0" // x-release-please-version
	// SpanName is the root span name for Bedrock handlers. Its gen_ai.operation.name is also "invoke_agent".
	SpanName = bedrockwire.SpanName
	// GenAISystem is the gen_ai.system and gen_ai.provider.name value.
	GenAISystem = bedrockwire.GenAISystem
)

// ProvidesFor is the variation provider/mode this handler serves.
var ProvidesFor = aiserver.ProvidesFor{Provider: "Bedrock", Mode: aiserver.ModeMessages}

// ErrUnsupportedContent reports a history content block Bedrock Converse
// cannot carry without a policy this package does not apply, such as a URL
// image (the handler never downloads images) or an unknown image type.
var ErrUnsupportedContent = bedrockwire.ErrUnsupportedContent

// ErrToolNotFound is returned when the model requests a tool that was not
// offered (no config tool with a ToolFunc under that name). The error names
// the tool.
var ErrToolNotFound = bedrockwire.ErrToolNotFound

// ErrMaxSteps is returned when the model still asks for tools after the
// turn bound.
var ErrMaxSteps = bedrockwire.ErrMaxSteps

func init() { aiserver.RegisterAiSdkPackage(PackageName, PackageVersion) }

// Options configure NewHandler.
type Options struct {
	// Client, when set, is used as-is for every turn and stream. APIKey,
	// Region and environment configuration never modify it.
	Client *bedrockruntime.Client
	// APIKey is a Bedrock API key sent as a bearer token. It takes precedence
	// over AWS_BEARER_TOKEN_BEDROCK; with neither, the default AWS credential
	// chain signs requests with SigV4. Ignored when Client is set.
	APIKey string
	// Region is the AWS endpoint / SigV4 region (e.g. "us-east-1"). It only
	// configures the constructed client and never enters the modelId. Empty
	// means AWS_REGION / AWS_DEFAULT_REGION / shared config. Ignored when
	// Client is set.
	Region string
	// ConverseOptions is called once per provider turn (tool follow-ups and
	// streaming turns included). Every non-zero field of the returned input is
	// merged over the handler's standard fields, so it can set or override any
	// Converse argument, including fields added by future SDK versions.
	// ModelId, Messages and System stay handler-generated. When the handler
	// sends tools, the returned ToolConfig contributes only its ToolChoice.
	// For ConverseStream, GuardrailConfig is converted to the stream form.
	// Return nil to change nothing.
	ConverseOptions func(cfg aiserver.AiConfig) *bedrockruntime.ConverseInput
	// CaptureContent gates prompt, completion, tool and judge-explanation
	// content on spans. CaptureDefault follows aiserver.Configure, else on.
	CaptureContent aiserver.ContentCapture
}

// NewHandler returns the Bedrock messages handler
// (createBedrockMessagesHandler). Without Options.Client it builds a new
// Bedrock Runtime client from the default AWS configuration, which fails only
// when that configuration cannot be loaded (for example a malformed shared
// config file); the error is returned instead of deferred to the first call.
func NewHandler(opts Options) (*aiserver.ProviderHandler, error) {
	aiserver.ReportUsage("bedrock-messages.createBedrockMessagesHandler", PackageName, PackageVersion)
	h, err := build(opts)
	if err != nil {
		return nil, err
	}
	return h.provider(), nil
}

type handler struct {
	*bedrockwire.Engine
	captureContent aiserver.ContentCapture
}

// build makes the handler without reporting usage. The tool loop has
// no fixed limit; the engine's default of 10 turns matches the other
// messages handlers.
func build(opts Options) (*handler, error) {
	c := opts.Client
	if c == nil {
		var err error
		if c, err = bedrockwire.NewClient(opts.APIKey, opts.Region); err != nil {
			return nil, fmt.Errorf("bedrockmessages: %w", err)
		}
	}
	return &handler{Engine: &bedrockwire.Engine{
		Client: c, ConverseOptions: opts.ConverseOptions, Mode: aiserver.ModeMessages,
		Tracer: PackageName, Prefix: "bedrockmessages",
	}, captureContent: opts.CaptureContent}, nil
}

func (h *handler) provider() *aiserver.ProviderHandler {
	return sdkapi.CreateHandler(ProvidesFor, h.Invoke, h.Stream, h.captureContent)
}

// InvokeOptions are the optional arguments to BedrockMessages.
type InvokeOptions struct {
	ToolHandlers map[string]aiserver.ToolHandler
	Registry     *aiserver.Registry
	SkipJudges   bool
	// Variables are passed as the third argument to Invoke, nil when unset.
	Variables map[string]any
	// Handler configures the handler built for this call.
	Handler Options
}

// BedrockMessages invokes configKey through a fresh Bedrock messages handler.
// opts may be nil.
func BedrockMessages(ctx context.Context, configKey, userInput string, userCtx ldcontext.Context, opts *InvokeOptions) (aiserver.ProviderResponse, error) {
	aiserver.ReportUsage("bedrock-messages.bedrockMessages", PackageName, PackageVersion)
	if opts == nil {
		opts = &InvokeOptions{}
	}
	h, err := build(opts.Handler)
	if err != nil {
		return aiserver.ProviderResponse{}, err
	}
	return sdkapi.Config(aiserver.ConfigArgs{
		Key: configKey, Handlers: []*aiserver.ProviderHandler{h.provider()},
		ToolHandlers: opts.ToolHandlers, Registry: opts.Registry, SkipJudges: opts.SkipJudges,
	}).Invoke(ctx, userInput, userCtx, aiserver.WithVariables(opts.Variables))
}
