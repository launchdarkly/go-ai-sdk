// Package bedrockagents is the Amazon Bedrock agent handler for aiserver
// (providesFor Bedrock/agent). Go has no Strands Agents SDK, so the agent
// loop runs on the Bedrock Runtime Converse API: Converse for blocking runs,
// ConverseStream for streaming, local tools executed between model turns,
// and one invoke_agent root span with a chat child per model turn and an
// execute_tool sibling per tool call.
package bedrockagents

import (
	"context"
	"fmt"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/bedrockwire"
)

const (
	// PackageName is this package's identity for $ld:ai:sdk:info and
	// $ld:ai:sdk:usage.
	PackageName = "github.com/launchdarkly/go-ai-sdk/bedrockagents"
	// PackageVersion is this package's release version.
	PackageVersion = "0.1.0" // x-release-please-version
	// SpanName is the root span name and root gen_ai.operation.name.
	SpanName = bedrockwire.SpanName
	// GenAISystem is the gen_ai.system and gen_ai.provider.name value.
	GenAISystem = bedrockwire.GenAISystem
	// DefaultMaxSteps is the model-turn bound used when Options.MaxSteps is
	// zero.
	DefaultMaxSteps = bedrockwire.DefaultMaxSteps
)

// ProvidesFor is the variation provider/mode this handler serves.
var ProvidesFor = aiserver.ProvidesFor{Provider: "Bedrock", Mode: aiserver.ModeAgent}

// ErrUnsupportedContent reports a history content block Bedrock Converse
// cannot carry without a policy this package does not apply, such as a URL
// image (images are never downloaded) or an unknown image type. It is
// the same value as bedrockmessages.ErrUnsupportedContent.
var ErrUnsupportedContent = bedrockwire.ErrUnsupportedContent

// ErrToolNotFound is returned when the model requests a tool that was not
// offered: one missing from config.tools, or without a registered function,
// even if ToolHandlers has a function under that name. The error names the
// tool, and the call's execute_tool span ends ERROR. It is the same value as
// bedrockmessages.ErrToolNotFound.
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
	// configures the constructed client and never comes from or enters the
	// modelId. Empty means AWS_REGION / AWS_DEFAULT_REGION / shared config.
	// Ignored when Client is set.
	Region string
	// ModelOptions is the Go form of the Strands modelOptions(config)
	// callback. It is called once per handler call, and every
	// non-zero field of the returned input is merged over the handler's
	// standard fields on each model turn of that call, so it can set or
	// override any Converse argument (InferenceConfig, GuardrailConfig,
	// AdditionalModelRequestFields, PerformanceConfig, OutputConfig.Effort,
	// and fields added by future SDK versions). ModelId, Messages and System
	// stay handler-generated. When the handler sends tools, the returned
	// ToolConfig contributes only its ToolChoice. A returned OutputConfig
	// without a TextFormat keeps the handler's structured-output format. For
	// ConverseStream, GuardrailConfig is converted to the stream form. Return
	// nil to change nothing.
	ModelOptions func(cfg aiserver.AiConfig) *bedrockruntime.ConverseInput
	// CaptureContent is attached to the handler so the client core can gate
	// judge explanations on it. Default false.
	CaptureContent bool
	// MaxSteps bounds the model turns in one call. When the model still asks
	// for tools at the bound, the call fails with ErrMaxSteps. Zero means
	// DefaultMaxSteps.
	MaxSteps int
}

// NewHandler returns the Bedrock agent handler (createBedrockAgentsHandler).
// Without Options.Client it builds a new Bedrock Runtime client from the
// default AWS configuration, which fails only when that configuration cannot
// be loaded (for example a malformed shared config file); the error is
// returned instead of deferred to the first call.
func NewHandler(opts Options) (*aiserver.ProviderHandler, error) {
	aiserver.ReportUsage("bedrock-agents.createBedrockAgentsHandler", PackageName, PackageVersion)
	h, err := build(opts)
	if err != nil {
		return nil, err
	}
	return h.provider(), nil
}

type handler struct {
	*bedrockwire.Engine
	captureContent bool
}

// build makes the handler without reporting usage.
func build(opts Options) (*handler, error) {
	c := opts.Client
	if c == nil {
		var err error
		if c, err = bedrockwire.NewClient(opts.APIKey, opts.Region); err != nil {
			return nil, fmt.Errorf("bedrockagents: %w", err)
		}
	}
	return &handler{Engine: &bedrockwire.Engine{
		Client: c, ConverseOptions: opts.ModelOptions, Mode: aiserver.ModeAgent,
		Tracer: PackageName, Prefix: "bedrockagents", MaxSteps: opts.MaxSteps,
	}, captureContent: opts.CaptureContent}, nil
}

func (h *handler) provider() *aiserver.ProviderHandler {
	return aiserver.CreateHandlerInternal(ProvidesFor, h.Invoke, h.Stream, h.captureContent)
}

// InvokeOptions are the optional arguments to BedrockAgents.
type InvokeOptions struct {
	// ToolHandlers are the local tools the agent may call.
	ToolHandlers map[string]aiserver.ToolHandler
	// Registry supplies shared handlers and tools.
	Registry *aiserver.Registry
	// SkipJudges defers judges to ProviderResponse.JudgeTasks.
	SkipJudges bool
	// Variables are passed as the third argument to Invoke, nil when unset.
	Variables map[string]any
	// Handler configures the handler built for this call.
	Handler Options
}

// BedrockAgents invokes configKey through a fresh Bedrock agent handler.
// opts may be nil.
func BedrockAgents(ctx context.Context, configKey, userInput string, userCtx ldcontext.Context, opts *InvokeOptions) (aiserver.ProviderResponse, error) {
	aiserver.ReportUsage("bedrock-agents.bedrockAgents", PackageName, PackageVersion)
	if opts == nil {
		opts = &InvokeOptions{}
	}
	h, err := build(opts.Handler)
	if err != nil {
		return aiserver.ProviderResponse{}, err
	}
	return aiserver.ConfigInternal(aiserver.ConfigArgs{
		Key: configKey, Handlers: []*aiserver.ProviderHandler{h.provider()},
		ToolHandlers: opts.ToolHandlers, Registry: opts.Registry, SkipJudges: opts.SkipJudges,
	}).Invoke(ctx, userInput, userCtx, aiserver.WithVariables(opts.Variables))
}

// graphInternal is the non-reporting core graph entry; tests replace it to
// observe the options BedrockGraph forwards.
var graphInternal = aiserver.GraphInternal

// BedrockGraph returns graph(key) with exactly one Bedrock agent handler,
// built from handlerOpts (bedrockGraph). Every other field of
// opts is forwarded unchanged; opts.Handlers is ignored, so a caller cannot
// bind a different or additional handler. The error is NewHandler's.
func BedrockGraph(key string, opts aiserver.GraphOptions, handlerOpts Options) (*aiserver.AgentGraph, error) {
	aiserver.ReportUsage("bedrock-agents.bedrockGraph", PackageName, PackageVersion)
	h, err := build(handlerOpts)
	if err != nil {
		return nil, err
	}
	opts.Handlers = []*aiserver.ProviderHandler{h.provider()}
	return graphInternal(key, opts), nil
}
