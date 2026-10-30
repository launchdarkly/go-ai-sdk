// Package langchainwire is the langchaingo plumbing shared by the LangChain
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

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/anthropic"
	"github.com/tmc/langchaingo/llms/bedrock"
	"github.com/tmc/langchaingo/llms/openai"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

var (
	// ErrNoToolHandler is returned when the model requests a tool that has no
	// registered ToolFunc.
	ErrNoToolHandler = errors.New("langchain: model requested a tool with no registered handler")
	// ErrEmptyResponse is returned when the model returns no choices.
	ErrEmptyResponse = errors.New("langchain: model returned no choices")

	errAbandoned = errors.New("langchain: stream abandoned by consumer")
)

// ProviderName is the lowercased config provider for gen_ai.system. An
// empty name falls back to "openai", as in JavaScript.
func ProviderName(cfg aiserver.AiConfig) string {
	if cfg.Provider.Name == "" {
		return "openai"
	}
	return strings.ToLower(cfg.Provider.Name)
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

// MarkExplicitZeroTemperature marks ctx when params sets temperature to 0,
// so the Bedrock client middleware can restore the value langchaingo drops.
// Call it before deriving span or model contexts from ctx.
func MarkExplicitZeroTemperature(ctx context.Context, params map[string]any) context.Context {
	if t, ok := Number(params["temperature"]); ok && t == 0 {
		return withExplicitZeroTemperature(ctx)
	}
	return ctx
}

// ResolveModel returns model when set, else factory(cfg) when set, else the
// built-in route for cfg's provider (DefaultModel). A non-nil schema binds
// OpenAI's json_schema response format on the built-in route.
func ResolveModel(ctx context.Context, model llms.Model, factory func(aiserver.AiConfig) (llms.Model, error), cfg aiserver.AiConfig, schema map[string]any) (llms.Model, error) {
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
	return DefaultModel(ctx, cfg, schema)
}

// DefaultModel builds a langchaingo model from config.provider.name:
// "anthropic", "bedrock", or OpenAI for anything else. A non-nil schema binds
// OpenAI's json_schema response format.
func DefaultModel(ctx context.Context, cfg aiserver.AiConfig, schema map[string]any) (llms.Model, error) {
	var (
		m   llms.Model
		err error
	)
	switch p := ProviderName(cfg); p {
	case "anthropic":
		m, err = anthropic.New(anthropic.WithModel(cfg.Model.Name))
	case "bedrock":
		var client *bedrockruntime.Client
		if client, err = newBedrockClient(ctx); err == nil {
			m, err = bedrock.NewWithContext(ctx, bedrock.WithModel(cfg.Model.Name), bedrock.WithClient(client))
		}
	default:
		opts := []openai.Option{openai.WithModel(cfg.Model.Name)}
		if schema != nil {
			rf, rerr := responseFormat(schema)
			if rerr != nil {
				return nil, rerr
			}
			opts = append(opts, openai.WithResponseFormat(rf))
		}
		m, err = openai.New(opts...)
	}
	if err != nil {
		return nil, fmt.Errorf("langchain: build %s model: %w", ProviderName(cfg), err)
	}
	return m, nil
}

// responseFormat converts a raw JSON Schema into langchaingo's typed OpenAI
// json_schema response format.
// Known limitation: the typed struct drops JSON Schema keywords it has no field for
// (anyOf, format, ...); pass a ModelFactory to send a schema verbatim.
func responseFormat(schema map[string]any) (*openai.ResponseFormat, error) {
	b, err := json.Marshal(schema)
	if err != nil {
		return nil, fmt.Errorf("langchain: encode outputFormat: %w", err)
	}
	var prop openai.ResponseFormatJSONSchemaProperty
	if err := json.Unmarshal(b, &prop); err != nil {
		return nil, fmt.Errorf("langchain: outputFormat is not a supported JSON Schema: %w", err)
	}
	return &openai.ResponseFormat{Type: "json_schema", JSONSchema: &openai.ResponseFormatJSONSchema{Name: "output", Schema: &prop}}, nil
}

// ToParts maps one message's content to langchaingo parts.
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
			// BinaryContent is the image form every langchaingo provider
			// accepts; OpenAI renders it as a data: image_url.
			data, err := base64.StdEncoding.DecodeString(b.Source.Data)
			if err != nil {
				parts = append(parts, llms.ImageURLContent{URL: aiserver.ImageURL(*b.Source)})
				continue
			}
			parts = append(parts, llms.BinaryContent{MIMEType: b.Source.MediaType, Data: data})
		case b.Type == "image" && b.Source != nil:
			parts = append(parts, llms.ImageURLContent{URL: aiserver.ImageURL(*b.Source)})
		}
	}
	return parts
}

// ToMessages maps a composed prompt to langchaingo messages: a system
// message when System is set, then one message per turn.
func ToMessages(p aiserver.Prompt) []llms.MessageContent {
	out := make([]llms.MessageContent, 0, len(p.Turns)+1)
	if p.System != "" {
		out = append(out, llms.TextParts(llms.ChatMessageTypeSystem, p.System))
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

// FunctionTool is a langchaingo function tool definition.
func FunctionTool(name, description string, parameters map[string]any) llms.Tool {
	return llms.Tool{Type: "function", Function: &llms.FunctionDefinition{Name: name, Description: description, Parameters: parameters}}
}

// BuildTools forwards config tools that have a local ToolFunc, sorted
// by name so the request is deterministic.
func BuildTools(cfg aiserver.AiConfig, handlers map[string]aiserver.ToolHandler) ([]llms.Tool, map[string]aiserver.ToolFunc) {
	callable := aiserver.CallableTools(cfg, handlers)
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
// langchaingo call options. config.model.custom is never read.
func CallOptions(params map[string]any) []llms.CallOption {
	var out []llms.CallOption
	num := func(keys ...string) (float64, bool) {
		for _, k := range keys {
			if v, ok := Number(params[k]); ok {
				return v, true
			}
		}
		return 0, false
	}
	if v, ok := num("max_tokens", "maxTokens"); ok {
		out = append(out, llms.WithMaxTokens(int(v)))
	}
	if v, ok := num("temperature"); ok {
		out = append(out, llms.WithTemperature(v))
	}
	if v, ok := num("top_p", "topP"); ok {
		out = append(out, llms.WithTopP(v))
	}
	if v, ok := num("top_k", "topK"); ok {
		out = append(out, llms.WithTopK(int(v)))
	}
	if v, ok := num("seed"); ok {
		out = append(out, llms.WithSeed(int(v)))
	}
	if v, ok := num("frequency_penalty", "frequencyPenalty"); ok {
		out = append(out, llms.WithFrequencyPenalty(v))
	}
	if v, ok := num("presence_penalty", "presencePenalty"); ok {
		out = append(out, llms.WithPresencePenalty(v))
	}
	for _, k := range []string{"stop", "stop_sequences", "stopSequences"} {
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
// families langchaingo v0.1.15 writes: Anthropic (InputTokens, uncached),
// OpenAI (PromptTokens, cache-inclusive) and Bedrock (input_tokens).
func choiceUsage(info map[string]any) (aiserver.SpanUsage, bool) {
	get := func(k string) float64 { v, _ := Number(info[k]); return v }
	has := func(k string) bool { _, ok := info[k]; return ok }
	switch {
	case has("InputTokens") || has("OutputTokens"):
		read, created := get("CacheReadInputTokens"), get("CacheCreationInputTokens")
		return aiserver.SpanUsage{Input: get("InputTokens") + read + created, Output: get("OutputTokens"), CacheRead: read, CacheCreation: created}, true
	case has("PromptTokens") || has("CompletionTokens"):
		return aiserver.SpanUsage{Input: get("PromptTokens"), Output: get("CompletionTokens"), CacheRead: get("PromptCachedTokens")}, true
	case has("input_tokens") || has("output_tokens"):
		return aiserver.SpanUsage{Input: get("input_tokens"), Output: get("output_tokens")}, true
	}
	return aiserver.SpanUsage{}, false
}

// ResponseUsage takes usage from the first choice that reports it: the
// Anthropic provider repeats the message usage on every content-block
// choice. A response without usage yields zeros.
func ResponseUsage(resp *llms.ContentResponse) aiserver.SpanUsage {
	if resp == nil {
		return aiserver.SpanUsage{}
	}
	for _, c := range resp.Choices {
		if c == nil {
			continue
		}
		if u, ok := choiceUsage(c.GenerationInfo); ok {
			return u
		}
	}
	return aiserver.SpanUsage{}
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

// isToolCallChunk reports a streaming chunk that carries tool-call deltas
// rather than text: langchaingo's OpenAI client passes the JSON-encoded
// deltas to the same StreamingFunc as text.
// Known limitation: a text delta that is itself a JSON array of {"function":...}
// objects would be dropped; upgrade if langchaingo splits the callbacks.
func isToolCallChunk(b []byte) bool {
	if len(b) == 0 || b[0] != '[' {
		return false
	}
	var calls []map[string]json.RawMessage
	if json.Unmarshal(b, &calls) != nil || len(calls) == 0 {
		return false
	}
	for _, c := range calls {
		if _, ok := c["function"]; !ok {
			return false
		}
	}
	return true
}

// IsEmptyResponse reports a model call that returned no content: this
// package's ErrEmptyResponse or a langchaingo provider's.
func IsEmptyResponse(err error) bool {
	return errors.Is(err, ErrEmptyResponse) || errors.Is(err, anthropic.ErrEmptyResponse) || errors.Is(err, openai.ErrEmptyResponse)
}

// Turn performs one model call. When onText is set it streams text chunks;
// stopped reports that the consumer stopped iterating. text is the turn's
// output text: the streamed text when streaming, else the response content.
func Turn(ctx context.Context, model llms.Model, msgs []llms.MessageContent, opts []llms.CallOption, onText func(string) bool) (resp *llms.ContentResponse, text string, stopped bool, err error) {
	var streamed strings.Builder
	if onText != nil {
		opts = append(slices.Clone(opts), llms.WithStreamingFunc(func(_ context.Context, chunk []byte) error {
			if len(chunk) == 0 || isToolCallChunk(chunk) {
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
func RunTool(ctx context.Context, r *aiserver.HandlerRun, fns map[string]aiserver.ToolFunc, tc llms.ToolCall) (llms.MessageContent, error) {
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
