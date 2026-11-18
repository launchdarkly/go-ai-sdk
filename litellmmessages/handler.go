// Package litellmmessages is the LiteLLM messages handler for aiserver
// (providesFor */messages). It sends every call to a LiteLLM proxy over the
// OpenAI-compatible Chat Completions API, so one handler serves any model the
// proxy routes: provider-qualified names such as anthropic/claude-sonnet-4,
// bedrock/..., gemini/..., and proxy aliases.
package litellmmessages

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"maps"
	"os"
	"strings"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/chatwire"
	"github.com/launchdarkly/go-ai-sdk/internal/handlerkit"
	"github.com/launchdarkly/go-ai-sdk/internal/kitbase"
	"github.com/launchdarkly/go-ai-sdk/internal/openaiwire"
	"github.com/launchdarkly/go-ai-sdk/internal/sdkapi"
)

const (
	// PackageName is the Go import path, registered for `$ld:ai:sdk:info`.
	PackageName = "github.com/launchdarkly/go-ai-sdk/litellmmessages"
	// PackageVersion is this package's release version.
	PackageVersion = "1.0.0" // x-release-please-version
	// SpanName is the root span name for both the blocking and streaming paths.
	SpanName = "invoke_agent"
	// maxSteps bounds the tool loop, matching the JavaScript handler.
	maxSteps = 10
	// system is gen_ai.system on every span.
	system = "litellm"
)

var (
	// ErrNoBaseURL is returned by NewHandler when no proxy base URL is set
	// (Options.BaseURL or LITELLM_BASE_URL) and no client is injected.
	ErrNoBaseURL = errors.New("litellmmessages: LiteLLM proxy base URL is required (set Options.BaseURL or LITELLM_BASE_URL)")
	// ErrMaxSteps is returned when the model keeps requesting tools past the
	// tool-loop bound.
	ErrMaxSteps = errors.New("litellmmessages: tool loop exceeded the maximum number of steps")
)

// providesFor is the variation provider/mode this handler serves: any
// provider, since the proxy routes the model.
var providesFor = aiserver.ProvidesFor{Provider: "*", Mode: aiserver.ModeMessages}

func init() { handlerkit.RegisterAiSdkPackage(PackageName, PackageVersion) }

// ClientFactory returns the client for one call from the evaluated AI
// Config.
type ClientFactory func(cfg aiserver.AiConfig) (*openai.Client, error)

// Options configure NewHandler.
type Options struct {
	// BaseURL is the LiteLLM proxy base URL. "" falls back to LITELLM_BASE_URL.
	// The OPENAI_* environment variables are never read.
	BaseURL string
	// APIKey is the proxy key. "" falls back to LITELLM_API_KEY; an
	// unauthenticated proxy needs neither.
	APIKey string
	// Client is used for every call when set, instead of a client built from
	// BaseURL and APIKey. It must point at a LiteLLM proxy.
	Client *openai.Client
	// ClientFactory is called per call, after the AI Config is evaluated,
	// when set. It wins over Client.
	ClientFactory ClientFactory
	// MaxRetries is the retry count on the client NewHandler builds. nil keeps
	// the provider SDK's default; 0 means a single attempt. A negative count, or
	// MaxRetries together with Client or ClientFactory, is refused by NewHandler.
	MaxRetries *int
	// CaptureContent gates prompt, completion, tool and judge-explanation
	// content on spans. CaptureDefault follows aiserver.Configure, else on.
	CaptureContent aiserver.ContentCapture
}

// NewHandler returns the LiteLLM messages handler
// (createLiteLLMMessagesHandler). It returns ErrNoBaseURL without a proxy
// base URL or an injected client, and an error for an invalid
// Options.MaxRetries.
func NewHandler(opts Options) (*aiserver.ProviderHandler, error) {
	handlerkit.ReportUsage("litellm-messages.createLiteLLMMessagesHandler", PackageName, PackageVersion)
	return newHandler(opts)
}

// newHandler builds the handler without reporting usage. It checks the options
// itself, so no path builds a client from an unchecked count.
func newHandler(opts Options) (*aiserver.ProviderHandler, error) {
	supplied := ""
	switch {
	case opts.ClientFactory != nil:
		supplied = "ClientFactory"
	case opts.Client != nil:
		supplied = "Client"
	}
	if err := handlerkit.CheckMaxRetries("litellmmessages", opts.MaxRetries, supplied); err != nil {
		return nil, err
	}
	h := &handler{client: opts.Client, factory: opts.ClientFactory}
	if supplied == "" {
		c, err := proxyClient(opts)
		if err != nil {
			return nil, err
		}
		h.client = c
	}
	return sdkapi.CreateHandler(providesFor, h.invoke, h.stream, opts.CaptureContent), nil
}

// proxyClient builds a client for the LiteLLM proxy. openai-go applies the
// OPENAI_* environment first; the base URL and key are overridden here and the
// headers it adds are deleted, so nothing from that environment is sent.
func proxyClient(opts Options) (*openai.Client, error) {
	base := opts.BaseURL
	if base == "" {
		base = os.Getenv("LITELLM_BASE_URL")
	}
	if base == "" {
		return nil, ErrNoBaseURL
	}
	key := opts.APIKey
	if key == "" {
		key = os.Getenv("LITELLM_API_KEY")
	}
	if key == "" {
		key = "not-needed" // openai-go requires a key; an open proxy ignores it
	}
	reqOpts := []option.RequestOption{
		option.WithBaseURL(base), option.WithAPIKey(key),
		option.WithHeaderDel("OpenAI-Organization"), option.WithHeaderDel("OpenAI-Project"),
	}
	for _, line := range strings.Split(os.Getenv("OPENAI_CUSTOM_HEADERS"), "\n") {
		if name, _, ok := strings.Cut(line, ":"); ok {
			reqOpts = append(reqOpts, option.WithHeaderDel(strings.TrimSpace(name)))
		}
	}
	if opts.MaxRetries != nil {
		reqOpts = append(reqOpts, option.WithMaxRetries(*opts.MaxRetries))
	}
	c := openai.NewClient(reqOpts...)
	return &c, nil
}

// InvokeOptions are the optional arguments to LiteLLMMessages.
type InvokeOptions struct {
	ToolHandlers map[string]aiserver.ToolHandler
	Registry     *aiserver.Registry
	SkipJudges   bool
	// Variables are passed to Invoke, not to the config.
	Variables map[string]any
	// Handler configures the handler built for this call.
	Handler Options
}

// LiteLLMMessages invokes configKey through a fresh LiteLLM messages handler
// (litellmMessages). A NewHandler error is returned as this call's error.
// opts may be nil.
func LiteLLMMessages(ctx context.Context, configKey, userInput string, userCtx ldcontext.Context, opts *InvokeOptions) (aiserver.ProviderResponse, error) {
	handlerkit.ReportUsage("litellm-messages.litellmMessages", PackageName, PackageVersion)
	if opts == nil {
		opts = &InvokeOptions{}
	}
	h, err := newHandler(opts.Handler)
	if err != nil {
		return aiserver.ProviderResponse{}, err
	}
	return sdkapi.Config(aiserver.ConfigArgs{
		Key: configKey, Handlers: []*aiserver.ProviderHandler{h},
		ToolHandlers: opts.ToolHandlers, Registry: opts.Registry, SkipJudges: opts.SkipJudges,
	}).Invoke(ctx, userInput, userCtx, aiserver.WithVariables(opts.Variables))
}

type handler struct {
	client  *openai.Client
	factory ClientFactory
}

// requestModel is the LiteLLM model for cfg. A bedrock/ name gets the Bedrock
// inference-profile prefix on its last segment whatever the provider, so every
// bedrock/<route>/ form works; a bare name gets it only under the Bedrock
// provider. Any other name, and a bedrock/ name holding an ARN, is unchanged.
func requestModel(cfg aiserver.AiConfig) string {
	name, region := cfg.Model.Name, cfg.Model.Region
	if strings.HasPrefix(name, "bedrock/") {
		if strings.Contains(name, "arn:") {
			return name
		}
		cut := strings.LastIndex(name, "/") + 1
		return name[:cut] + handlerkit.BedrockModelID(name[cut:], region)
	}
	if strings.EqualFold(cfg.Provider.Name, "bedrock") && !strings.Contains(name, "/") {
		return handlerkit.BedrockModelID(name, region)
	}
	return name
}

// providerName is gen_ai.provider.name: the config provider in lower case,
// else litellm.
func providerName(cfg aiserver.AiConfig) string {
	if cfg.Provider.Name == "" {
		return system
	}
	return strings.ToLower(cfg.Provider.Name)
}

// forwardedParameters are the config.model.parameters keys forwarded: settings
// that shape what the model generates. Handler-owned fields (model, messages,
// tools, stream, stream_options, response_format), LiteLLM routing,
// credential and retry keys, and identity or retention keys are absent.
var forwardedParameters = []string{
	"frequency_penalty", "max_completion_tokens", "max_tokens", "parallel_tool_calls", "presence_penalty",
	"reasoning_effort", "seed", "stop", "temperature", "tool_choice", "top_p",
}

// schema is outputFormat as a JSON Schema object: a json_schema, json or
// missing type becomes object.
func schema(outputFormat map[string]any) map[string]any {
	s := maps.Clone(outputFormat)
	switch s["type"] {
	case "json_schema", "json", nil:
		s["type"] = "object"
	}
	return s
}

// runUsage is the handler-result usage summed over every turn. It omits cache
// keys: prompt_tokens already includes them.
type runUsage struct{ input, output float64 }

func (u *runUsage) add(s kitbase.SpanUsage) { u.input += s.Input; u.output += s.Output }

func (u runUsage) raw() map[string]any {
	return map[string]any{"input_tokens": u.input, "output_tokens": u.output}
}

// runTools executes the requested calls and returns one role: tool message
// per call. A call for a tool that was not offered is an error, never a
// lookup in the raw handler map.
func runTools(r kitbase.Run, calls []chatwire.ToolCall, callable map[string]aiserver.Tool, handlers map[string]aiserver.ToolHandler) ([]openai.ChatCompletionMessageParamUnion, error) {
	var out []openai.ChatCompletionMessageParamUnion
	for _, call := range calls {
		span := r.StartTool(call.Name, call.ID)
		r.ToolArguments(span, call.Arguments)
		res, err := callTool(r.Context(), call, callable, handlers)
		if err != nil {
			return nil, err // Fail ends the open tool span
		}
		r.ToolResult(span, res)
		r.EndTool(span, nil)
		out = append(out, chatwire.ToolMessage(call.ID, aiserver.ToolResultString(res)))
	}
	return out, nil
}

func callTool(ctx context.Context, call chatwire.ToolCall, callable map[string]aiserver.Tool, handlers map[string]aiserver.ToolHandler) (any, error) {
	fn, _ := handlers[call.Name].(aiserver.ToolFunc)
	if _, offered := callable[call.Name]; !offered || fn == nil {
		return nil, fmt.Errorf("litellmmessages: no handler registered for tool %q", call.Name)
	}
	args, err := openaiwire.ParseArguments(call.Arguments)
	if err != nil {
		return nil, fmt.Errorf("litellmmessages: tool %q arguments: %w", call.Name, err)
	}
	res, err := fn(ctx, args)
	if err != nil {
		return nil, fmt.Errorf("litellmmessages: tool %q: %w", call.Name, err) // propagate
	}
	return res, nil
}

// clientFor is the client for one call: the factory's, else the handler's.
func (h *handler) clientFor(cfg aiserver.AiConfig) (*openai.Client, error) {
	if h.factory == nil {
		return h.client, nil
	}
	c, err := h.factory(cfg)
	if err != nil {
		return nil, fmt.Errorf("litellmmessages: client factory: %w", err)
	}
	if c == nil {
		return nil, errors.New("litellmmessages: client factory returned no client")
	}
	return c, nil
}

// run is the tool loop shared by both paths; onText is nil when blocking.
// abandoned reports that the stream consumer stopped iterating.
func (h *handler) run(ctx context.Context, req aiserver.HandlerRequest, onText func(string) bool) (out string, usage runUsage, abandoned bool, err error) {
	model := requestModel(req.Config)
	p := handlerkit.ComposePrompt(req, aiserver.ModeMessages)
	r := handlerkit.StartHandlerRun(ctx, kitbase.HandlerSpanOptions{
		Tracer: PackageName, Name: SpanName, System: system, RequestModel: model, ProviderName: providerName(req.Config),
	}, req, p)
	defer r.EndOnPanic()

	client, err := h.clientFor(req.Config)
	if err != nil {
		r.Fail(err)
		return "", usage, false, err
	}
	callable := handlerkit.CallableTools(req.Config, req.ToolHandlers)
	params := openai.ChatCompletionNewParams{
		Model:    model,
		Messages: chatwire.Messages(p.System, p.Turns),
		Tools:    chatwire.FunctionTools(callable),
	}
	if extra := chatwire.ModelParameters(req.Config.Model.Parameters, forwardedParameters); len(extra) > 0 {
		params.SetExtraFields(extra)
	}
	if onText == nil && req.Config.OutputFormat != nil { // streaming ignores outputFormat
		params.ResponseFormat = chatwire.ResponseFormat(schema(req.Config.OutputFormat))
	}

	var streamed strings.Builder
	lastModel := model
	for step := 0; ; step++ {
		chat := r.StartChat(model)
		turn, stopped, terr := turnOf(r.Context(), client, params, onText, &streamed)
		if stopped {
			r.Abandon()
			return "", usage, true, nil
		}
		if terr != nil {
			r.Fail(terr)
			return "", usage, false, terr
		}
		usage.add(turn.Usage)
		lastModel = model
		if turn.Model != "" {
			lastModel = turn.Model
		}
		finish := turn.Finish
		if len(turn.ToolCalls) > 0 {
			finish = "tool_calls"
		}
		r.EndChat(chat, lastModel, turn.Usage, finish)

		if len(turn.ToolCalls) == 0 {
			out = turn.Text
			break
		}
		if step >= maxSteps {
			terr = fmt.Errorf("%w (%d)", ErrMaxSteps, maxSteps)
			r.Fail(terr)
			return "", usage, false, terr
		}
		results, terr := runTools(r, turn.ToolCalls, callable, req.ToolHandlers)
		if terr != nil {
			r.Fail(terr)
			return "", usage, false, terr
		}
		params.Messages = append(params.Messages, chatwire.AssistantMessage(turn))
		params.Messages = append(params.Messages, results...)
	}
	if onText != nil {
		out = streamed.String() // done.output spans every streamed turn
	}
	r.Finish(lastModel, out)
	return out, usage, false, nil
}

// turnOf performs one proxy call. When onText is set it streams, appending
// each text delta to streamed and passing it to onText; stopped reports that
// onText returned false.
func turnOf(ctx context.Context, client *openai.Client, params openai.ChatCompletionNewParams, onText func(string) bool, streamed *strings.Builder) (turn chatwire.Turn, stopped bool, err error) {
	if onText == nil {
		resp, err := client.Chat.Completions.New(ctx, params)
		if err != nil {
			return chatwire.Turn{}, false, fmt.Errorf("litellmmessages: chat.completions.create: %w", err)
		}
		if turn, err = chatwire.TurnOf(resp); err != nil {
			return chatwire.Turn{}, false, fmt.Errorf("litellmmessages: %w", err)
		}
		return turn, false, nil
	}
	turn, stopped, err = chatwire.StreamTurn(ctx, client, params, func(t string) bool {
		streamed.WriteString(t)
		return onText(t)
	})
	if err != nil {
		return chatwire.Turn{}, false, fmt.Errorf("litellmmessages: chat.completions.stream: %w", err)
	}
	return turn, stopped, nil
}

func (h *handler) invoke(ctx context.Context, req aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
	out, usage, _, err := h.run(ctx, req, nil)
	if err != nil {
		return aiserver.HandlerResult{}, err
	}
	return aiserver.HandlerResult{Output: out, Usage: usage.raw()}, nil
}

func (h *handler) stream(ctx context.Context, req aiserver.HandlerRequest) iter.Seq2[aiserver.HandlerStreamEvent, error] {
	return func(yield func(aiserver.HandlerStreamEvent, error) bool) {
		out, usage, abandoned, err := h.run(ctx, req, func(t string) bool {
			return yield(aiserver.HandlerStreamEvent{Type: aiserver.EventChunk, Text: t}, nil)
		})
		if abandoned {
			return
		}
		if err != nil {
			yield(aiserver.HandlerStreamEvent{}, err)
			return
		}
		yield(aiserver.HandlerStreamEvent{Type: aiserver.EventDone, Output: &out, Usage: usage.raw()}, nil)
	}
}
