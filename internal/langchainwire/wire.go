// Package langchainwire is the llms plumbing shared by the LangChain
// handlers (langchainmessages, langchainagents): model routing by provider,
// prompt and tool conversion, call options, usage extraction, one model turn
// (blocking or streamed) and one tool execution.
package langchainwire

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"sort"
	"strings"

	anthropicoption "github.com/anthropics/anthropic-sdk-go/option"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	openaioption "github.com/openai/openai-go/v3/option"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/bedrockwire"
	"github.com/launchdarkly/go-ai-sdk/internal/handlerkit"
	"github.com/launchdarkly/go-ai-sdk/internal/kitbase"
	"github.com/launchdarkly/go-ai-sdk/internal/telemetry"
	"github.com/launchdarkly/go-ai-sdk/llms"
)

var (
	// ErrNoToolHandler is returned when the model requests a tool that has no
	// registered ToolFunc.
	ErrNoToolHandler = errors.New("langchain: model requested a tool with no registered handler")
	// ErrEmptyResponse is returned when the model returns no choices.
	ErrEmptyResponse = errors.New("langchain: model returned no choices")

	errAbandoned = errors.New("langchain: stream abandoned by consumer")
)

// GenAISystem is gen_ai.system on every LangChain span. The provider that
// serves the request goes in gen_ai.provider.name (SpanProviderName).
const GenAISystem = "langchain"

// ProviderName is the lowercased config provider, which picks the model. An
// empty name falls back to "openai", as in JavaScript.
func ProviderName(cfg aiserver.AiConfig) string {
	if cfg.Provider.Name == "" {
		return "openai"
	}
	return strings.ToLower(cfg.Provider.Name)
}

// SpanProviderName is gen_ai.provider.name: ProviderName, except the Bedrock
// route, which is "aws.bedrock" as on the Bedrock handlers' spans.
func SpanProviderName(cfg aiserver.AiConfig) string {
	return telemetry.ProviderName(ProviderName(cfg))
}

// ModelConfig applies the Bedrock inference-profile prefix so the
// model call and the spans use the resolved model id.
func ModelConfig(cfg aiserver.AiConfig) aiserver.AiConfig {
	prefix := cfg.Model.Region
	if ProviderName(cfg) != "bedrock" || prefix == "" || strings.HasPrefix(cfg.Model.Name, prefix+".") {
		return cfg
	}
	cfg.Model.Name = prefix + "." + cfg.Model.Name
	return cfg
}

// ResolveModel returns model when set, else factory(cfg) when set, else the
// built-in route for cfg's provider (DefaultModel). A non-nil schema binds
// OpenAI's json_schema response format on the built-in route, and a non-nil
// maxRetries sets the built-in model's retry count.
func ResolveModel(ctx context.Context, model llms.Model, factory func(aiserver.AiConfig) (llms.Model, error), cfg aiserver.AiConfig, schema map[string]any, maxRetries *int) (llms.Model, error) {
	if model != nil {
		return model, nil
	}
	if factory != nil {
		m, err := factory(cfg)
		if err != nil {
			return nil, fmt.Errorf("langchain: model factory: %w", err)
		}
		if m == nil {
			return nil, errors.New("langchain: model factory returned a nil model")
		}
		return m, nil
	}
	return DefaultModel(ctx, cfg, schema, maxRetries)
}

// DefaultModel builds a model from config.provider.name: "anthropic",
// "bedrock", or OpenAI Chat Completions for anything else (also any
// OpenAI-compatible endpoint, through OPENAI_BASE_URL). A non-nil schema binds
// OpenAI's json_schema response format, sent verbatim. A non-nil maxRetries
// is the retry count on the client the model builds: option.WithMaxRetries for
// OpenAI and Anthropic, and maxRetries+1 attempts on the AWS client for Bedrock
// (never 0 attempts, which the AWS SDK reads as its default).
func DefaultModel(_ context.Context, cfg aiserver.AiConfig, schema map[string]any, maxRetries *int) (llms.Model, error) {
	var (
		m   llms.Model
		err error
	)
	switch ProviderName(cfg) {
	case "anthropic":
		o := llms.AnthropicOptions{Model: cfg.Model.Name}
		if maxRetries != nil {
			o.RequestOptions = []anthropicoption.RequestOption{anthropicoption.WithMaxRetries(*maxRetries)}
		}
		m, err = llms.NewAnthropic(o)
	case "bedrock":
		o := llms.BedrockOptions{Model: cfg.Model.Name}
		if maxRetries != nil {
			attempts := *maxRetries + 1
			if o.Client, err = bedrockwire.NewClient("", "", func(c *bedrockruntime.Options) { c.RetryMaxAttempts = attempts }); err != nil {
				break
			}
		}
		m, err = llms.NewBedrock(o)
	default:
		o := llms.OpenAIOptions{Model: cfg.Model.Name, JSONSchema: schema}
		if maxRetries != nil {
			o.RequestOptions = []openaioption.RequestOption{openaioption.WithMaxRetries(*maxRetries)}
		}
		m, err = llms.NewOpenAI(o)
	}
	if err != nil {
		return nil, fmt.Errorf("langchain: build %s model: %w", ProviderName(cfg), err)
	}
	return m, nil
}

// ToParts maps one message's content to llms parts.
func ToParts(m aiserver.Message) []llms.ContentPart {
	if m.Blocks == nil {
		return []llms.ContentPart{llms.TextContent{Text: m.Text}}
	}
	parts := make([]llms.ContentPart, 0, len(m.Blocks))
	for _, b := range m.Blocks {
		switch {
		case b.Type == "text":
			parts = append(parts, llms.TextContent{Text: b.Text})
		case b.Type == "image" && b.Source != nil && b.Source.Type == "base64":
			// BinaryContent is the image form every built-in model accepts;
			// OpenAI sends it as a data: image_url.
			data, err := base64.StdEncoding.DecodeString(b.Source.Data)
			if err != nil {
				parts = append(parts, llms.ImageURLContent{URL: handlerkit.ImageURL(*b.Source)})
				continue
			}
			parts = append(parts, llms.BinaryContent{MIMEType: b.Source.MediaType, Data: data})
		case b.Type == "image" && b.Source != nil:
			parts = append(parts, llms.ImageURLContent{URL: handlerkit.ImageURL(*b.Source)})
		}
	}
	return parts
}

// ToMessages maps a composed prompt to llms messages: a system
// message when System is set, then one message per turn.
func ToMessages(p handlerkit.Prompt) []llms.MessageContent {
	out := make([]llms.MessageContent, 0, len(p.Turns)+1)
	if p.System != "" {
		out = append(out, TextParts(llms.ChatMessageTypeSystem, p.System))
	}
	for _, t := range p.Turns {
		role := llms.ChatMessageTypeHuman
		if t.Role == "assistant" {
			role = llms.ChatMessageTypeAI
		}
		out = append(out, llms.MessageContent{Role: role, Parts: ToParts(t)})
	}
	return out
}

// FunctionTool is an llms function tool definition.
func FunctionTool(name, description string, parameters map[string]any) llms.Tool {
	return llms.Tool{Type: "function", Function: &llms.FunctionDefinition{Name: name, Description: description, Parameters: parameters}}
}

// BuildTools forwards config tools that have a local ToolFunc, sorted
// by name so the request is deterministic.
func BuildTools(cfg aiserver.AiConfig, handlers map[string]aiserver.ToolHandler) ([]llms.Tool, map[string]aiserver.ToolFunc) {
	callable := handlerkit.CallableTools(cfg, handlers)
	keys := make([]string, 0, len(callable))
	for k := range callable {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	tools := make([]llms.Tool, 0, len(keys))
	fns := make(map[string]aiserver.ToolFunc, len(keys))
	for _, k := range keys {
		t := callable[k]
		fns[k] = handlers[k].(aiserver.ToolFunc)
		tools = append(tools, FunctionTool(k, t.Description, t.Parameters))
	}
	return tools, fns
}

// Number reads a JSON or Go numeric value.
func Number(v any) (float64, bool) {
	switch n := v.(type) {
	case int:
		return float64(n), true
	case int32:
		return float64(n), true
	case int64:
		return float64(n), true
	case float64:
		return n, true
	case float32:
		return float64(n), true
	}
	return 0, false
}

// CallOptions maps the documented common config.model.parameters to
// llms call options. Keys are read in snake_case only: a camelCase spelling
// such as maxTokens is not a table key, so it is dropped.
// config.model.custom is never read.
func CallOptions(params map[string]any) []llms.CallOption {
	var out []llms.CallOption
	num := func(key string) (float64, bool) { return Number(params[key]) }
	if v, ok := num("max_tokens"); ok {
		out = append(out, llms.WithMaxTokens(int(v)))
	}
	if v, ok := num("temperature"); ok {
		out = append(out, llms.WithTemperature(v))
	}
	if v, ok := num("top_p"); ok {
		out = append(out, llms.WithTopP(v))
	}
	if v, ok := num("top_k"); ok {
		out = append(out, llms.WithTopK(int(v)))
	}
	if v, ok := num("seed"); ok {
		out = append(out, llms.WithSeed(int(v)))
	}
	if v, ok := num("frequency_penalty"); ok {
		out = append(out, llms.WithFrequencyPenalty(v))
	}
	if v, ok := num("presence_penalty"); ok {
		out = append(out, llms.WithPresencePenalty(v))
	}
	for _, k := range []string{"stop_sequences", "stop"} { // stop_sequences wins when both are set
		raw, _ := params[k].([]any)
		var words []string
		for _, w := range raw {
			if s, ok := w.(string); ok {
				words = append(words, s)
			}
		}
		if len(words) > 0 {
			out = append(out, llms.WithStopWords(words))
			break
		}
	}
	return out
}

// choiceUsage reads token counts from a choice's GenerationInfo using the key
// families the built-in llms models write (langchaingo's key names): Anthropic (InputTokens, uncached),
// OpenAI (PromptTokens, cache-inclusive) and Bedrock (input_tokens).
func choiceUsage(info map[string]any) (kitbase.SpanUsage, bool) {
	get := func(k string) float64 { v, _ := Number(info[k]); return v }
	has := func(k string) bool { _, ok := info[k]; return ok }
	switch {
	case has("InputTokens") || has("OutputTokens"):
		read, created := get("CacheReadInputTokens"), get("CacheCreationInputTokens")
		return kitbase.SpanUsage{Input: get("InputTokens") + read + created, Output: get("OutputTokens"), CacheRead: read, CacheCreation: created}, true
	case has("PromptTokens") || has("CompletionTokens"):
		return kitbase.SpanUsage{Input: get("PromptTokens"), Output: get("CompletionTokens"), CacheRead: get("PromptCachedTokens")}, true
	case has("input_tokens") || has("output_tokens"):
		return kitbase.SpanUsage{Input: get("input_tokens"), Output: get("output_tokens")}, true
	}
	return kitbase.SpanUsage{}, false
}

// ResponseUsage takes usage from the first choice that reports it: the
// Anthropic provider repeats the message usage on every content-block
// choice. A response without usage yields Unreported zeros.
func ResponseUsage(resp *llms.ContentResponse) kitbase.SpanUsage {
	if resp == nil {
		return kitbase.SpanUsage{Unreported: true}
	}
	for _, c := range resp.Choices {
		if c == nil {
			continue
		}
		if u, ok := choiceUsage(c.GenerationInfo); ok {
			return u
		}
	}
	return kitbase.SpanUsage{Unreported: true}
}

// ResponseText concatenates every choice's content.
func ResponseText(resp *llms.ContentResponse) string {
	var b strings.Builder
	for _, c := range resp.Choices {
		if c != nil {
			b.WriteString(c.Content)
		}
	}
	return b.String()
}

// ResponseToolCalls collects every choice's tool calls, in order.
func ResponseToolCalls(resp *llms.ContentResponse) []llms.ToolCall {
	var out []llms.ToolCall
	for _, c := range resp.Choices {
		if c != nil {
			out = append(out, c.ToolCalls...)
		}
	}
	return out
}

var finishReasons = map[string]string{
	"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length", "tool_use": "tool_calls",
	"refusal": "content_filter", "guardrail_intervened": "content_filter", "content_filtered": "content_filter",
}

// FinishReason is the first choice's stop reason in OpenAI vocabulary.
func FinishReason(resp *llms.ContentResponse) string {
	for _, c := range resp.Choices {
		if c != nil && c.StopReason != "" {
			if m, ok := finishReasons[strings.ToLower(c.StopReason)]; ok {
				return m
			}
			return c.StopReason
		}
	}
	return ""
}

// IsEmptyResponse reports a model call that returned no content: this
// package's ErrEmptyResponse or a model's llms.ErrEmptyResponse.
func IsEmptyResponse(err error) bool {
	return errors.Is(err, ErrEmptyResponse) || errors.Is(err, llms.ErrEmptyResponse)
}

// Turn performs one model call. When onText is set it streams text chunks;
// stopped reports that the consumer stopped iterating. text is the turn's
// output text: the streamed text when streaming, else the response content.
func Turn(ctx context.Context, model llms.Model, msgs []llms.MessageContent, opts []llms.CallOption, onText func(string) bool) (resp *llms.ContentResponse, text string, stopped bool, err error) {
	var streamed strings.Builder
	if onText != nil {
		opts = append(slices.Clone(opts), llms.WithStreamingFunc(func(_ context.Context, chunk []byte) error {
			if len(chunk) == 0 {
				return nil
			}
			streamed.Write(chunk)
			if !onText(string(chunk)) {
				stopped = true
				return errAbandoned
			}
			return nil
		}))
	}
	resp, err = model.GenerateContent(ctx, msgs, opts...)
	if stopped {
		return nil, "", true, nil
	}
	if err != nil {
		return nil, "", false, fmt.Errorf("langchain: model call: %w", err)
	}
	if resp == nil || len(resp.Choices) == 0 {
		return nil, "", false, ErrEmptyResponse
	}
	text = ResponseText(resp)
	if onText == nil {
		return resp, text, false, nil
	}
	if streamed.Len() == 0 && text != "" {
		// A model that ignores StreamingFunc still yields its text once.
		if !onText(text) {
			return nil, "", true, nil
		}
		return resp, text, false, nil
	}
	return resp, streamed.String(), false, nil
}

// ToolResponse is the tool-role message answering call tc with content.
func ToolResponse(tc llms.ToolCall, content string) llms.MessageContent {
	name := ""
	if tc.FunctionCall != nil {
		name = tc.FunctionCall.Name
	}
	return llms.MessageContent{Role: llms.ChatMessageTypeTool, Parts: []llms.ContentPart{llms.ToolCallResponse{
		ToolCallID: tc.ID, Name: name, Content: content,
	}}}
}

// RunTool executes one requested tool. When r is set the call runs under an
// execute_tool span; on error the span is left open for r.Fail. Tool errors
// propagate.
func RunTool(ctx context.Context, r kitbase.Run, fns map[string]aiserver.ToolFunc, tc llms.ToolCall) (llms.MessageContent, error) {
	if tc.FunctionCall == nil {
		return llms.MessageContent{}, fmt.Errorf("%w: tool call %q has no function", ErrNoToolHandler, tc.ID)
	}
	name := tc.FunctionCall.Name
	var end func(out any)
	if r != nil { // opened first so a missing tool shows as a failed call, as in JS
		s := r.StartTool(name, tc.ID)
		r.ToolArguments(s, tc.FunctionCall.Arguments)
		end = func(out any) {
			r.ToolResult(s, out)
			r.EndTool(s, nil)
		}
	}
	fn := fns[name]
	if fn == nil {
		return llms.MessageContent{}, fmt.Errorf("%w: %q", ErrNoToolHandler, name)
	}
	args, err := DecodeArgs(tc)
	if err != nil {
		return llms.MessageContent{}, err
	}
	out, err := fn(ctx, args)
	if err != nil {
		return llms.MessageContent{}, fmt.Errorf("langchain: tool %q: %w", name, err)
	}
	if end != nil {
		end(out)
	}
	return ToolResponse(tc, aiserver.ToolResultString(out)), nil
}

// DecodeArgs decodes a tool call's JSON arguments; empty arguments are nil.
func DecodeArgs(tc llms.ToolCall) (map[string]any, error) {
	var args map[string]any
	if tc.FunctionCall == nil {
		return nil, nil
	}
	if a := strings.TrimSpace(tc.FunctionCall.Arguments); a != "" {
		if err := json.Unmarshal([]byte(a), &args); err != nil {
			return nil, fmt.Errorf("langchain: decode arguments for tool %q: %w", tc.FunctionCall.Name, err)
		}
	}
	return args, nil
}

// TextParts is a message of role with one text part per string.
func TextParts(role llms.ChatMessageType, parts ...string) llms.MessageContent {
	m := llms.MessageContent{Role: role, Parts: make([]llms.ContentPart, 0, len(parts))}
	for _, p := range parts {
		m.Parts = append(m.Parts, llms.TextContent{Text: p})
	}
	return m
}

// AssistantMessage is the reply as an AI message for the conversation:
// every choice's Parts in order, or else the first choice's text and tool
// calls.
func AssistantMessage(r *llms.ContentResponse) llms.MessageContent {
	msg := llms.MessageContent{Role: llms.ChatMessageTypeAI}
	if r == nil {
		return msg
	}
	for _, c := range r.Choices {
		if c != nil {
			msg.Parts = append(msg.Parts, c.Parts...)
		}
	}
	if len(msg.Parts) > 0 || len(r.Choices) == 0 || r.Choices[0] == nil {
		return msg
	}
	c := r.Choices[0]
	if c.Content != "" {
		msg.Parts = append(msg.Parts, llms.TextContent{Text: c.Content})
	}
	for _, tc := range c.ToolCalls {
		msg.Parts = append(msg.Parts, tc)
	}
	return msg
}
