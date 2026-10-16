// Package llms is the chat-model interface behind the LangChain handlers
// (langchainmessages, langchainagents): messages, tools, call options and
// the Model a handler calls. Its names follow the subset of
// github.com/tmc/langchaingo/llms those handlers used, so a langchaingo
// model ports with a small adapter. The built-in models for OpenAI,
// Anthropic and Bedrock are in this package too (see New).
package llms

import (
	"context"
	"errors"
)

// ErrEmptyResponse is returned when a model's reply has no content.
var ErrEmptyResponse = errors.New("llms: model returned no content")

// Model is a chat model. GenerateContent sends messages and returns the
// model's reply; a set CallOptions.StreamingFunc receives text deltas as
// they arrive, and the returned response is still the complete reply.
type Model interface {
	GenerateContent(ctx context.Context, messages []MessageContent, options ...CallOption) (*ContentResponse, error)
}

// ChatMessageType is a message's role.
type ChatMessageType string

const (
	ChatMessageTypeSystem ChatMessageType = "system"
	ChatMessageTypeHuman  ChatMessageType = "human"
	ChatMessageTypeAI     ChatMessageType = "ai"
	ChatMessageTypeTool   ChatMessageType = "tool"
)

// MessageContent is one message: a role and its content parts.
type MessageContent struct {
	Role  ChatMessageType
	Parts []ContentPart
}

// TextParts is a message of role with one text part per string.
func TextParts(role ChatMessageType, parts ...string) MessageContent {
	m := MessageContent{Role: role, Parts: make([]ContentPart, 0, len(parts))}
	for _, p := range parts {
		m.Parts = append(m.Parts, TextContent{Text: p})
	}
	return m
}

// ContentPart is one piece of a message: TextContent, ImageURLContent,
// BinaryContent, ToolCall or ToolCallResponse.
type ContentPart interface{ isPart() }

// TextContent is text.
type TextContent struct{ Text string }

// ImageURLContent is an image by URL (including data: URLs).
type ImageURLContent struct {
	URL    string
	Detail string
}

// BinaryContent is inline data, used for images.
type BinaryContent struct {
	MIMEType string
	Data     []byte
}

// ToolCall is a tool call the model made, on an AI message.
type ToolCall struct {
	ID           string
	Type         string
	FunctionCall *FunctionCall
}

// FunctionCall names the function and carries its JSON arguments.
type FunctionCall struct {
	Name      string
	Arguments string
}

// ToolCallResponse answers a ToolCall, on a tool message.
type ToolCallResponse struct {
	ToolCallID string
	Name       string
	Content    string
}

func (TextContent) isPart()      {}
func (ImageURLContent) isPart()  {}
func (BinaryContent) isPart()    {}
func (ToolCall) isPart()         {}
func (ToolCallResponse) isPart() {}

// Tool is a tool the model may call. Type is "function".
type Tool struct {
	Type     string
	Function *FunctionDefinition
}

// FunctionDefinition describes a function tool; Parameters is a JSON Schema.
type FunctionDefinition struct {
	Name        string
	Description string
	Parameters  any
}

// ContentResponse is a model's reply.
type ContentResponse struct {
	Choices []*ContentChoice
}

// ContentChoice is one choice of a reply. GenerationInfo carries token
// usage under the provider's key names (see the built-in models).
type ContentChoice struct {
	Content        string
	StopReason     string
	GenerationInfo map[string]any
	ToolCalls      []ToolCall
	// Parts, when set, is the choice's content in order (text and tool
	// calls); AssistantMessage prefers it over Content and ToolCalls.
	Parts []ContentPart
}

// AssistantMessage is the reply as an AI message for the conversation:
// every choice's Parts in order, or else the first choice's text and tool
// calls.
func (r *ContentResponse) AssistantMessage() MessageContent {
	msg := MessageContent{Role: ChatMessageTypeAI}
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
		msg.Parts = append(msg.Parts, TextContent{Text: c.Content})
	}
	for _, tc := range c.ToolCalls {
		msg.Parts = append(msg.Parts, tc)
	}
	return msg
}

// CallOptions are the per-call settings. A nil pointer is unset; a set
// pointer is sent even when it is zero.
type CallOptions struct {
	Model            string
	MaxTokens        int
	Temperature      *float64
	TopP             *float64
	TopK             *int
	Seed             *int
	FrequencyPenalty *float64
	PresencePenalty  *float64
	StopWords        []string
	// JSONMode asks for a JSON object reply where the provider supports it.
	JSONMode bool
	Tools    []Tool
	// StreamingFunc receives each text delta; returning an error stops the
	// call, which then returns that error.
	StreamingFunc func(ctx context.Context, chunk []byte) error
}

// CallOption sets a CallOptions field.
type CallOption func(*CallOptions)

// Apply returns the options with every opt applied.
func Apply(opts ...CallOption) CallOptions {
	var o CallOptions
	for _, opt := range opts {
		opt(&o)
	}
	return o
}

func WithModel(model string) CallOption { return func(o *CallOptions) { o.Model = model } }
func WithMaxTokens(n int) CallOption    { return func(o *CallOptions) { o.MaxTokens = n } }
func WithTemperature(t float64) CallOption {
	return func(o *CallOptions) { o.Temperature = &t }
}
func WithTopP(p float64) CallOption { return func(o *CallOptions) { o.TopP = &p } }
func WithTopK(k int) CallOption     { return func(o *CallOptions) { o.TopK = &k } }
func WithSeed(s int) CallOption     { return func(o *CallOptions) { o.Seed = &s } }
func WithFrequencyPenalty(p float64) CallOption {
	return func(o *CallOptions) { o.FrequencyPenalty = &p }
}
func WithPresencePenalty(p float64) CallOption {
	return func(o *CallOptions) { o.PresencePenalty = &p }
}
func WithStopWords(words []string) CallOption { return func(o *CallOptions) { o.StopWords = words } }
func WithJSONMode() CallOption                { return func(o *CallOptions) { o.JSONMode = true } }
func WithTools(tools []Tool) CallOption       { return func(o *CallOptions) { o.Tools = tools } }
func WithStreamingFunc(f func(ctx context.Context, chunk []byte) error) CallOption {
	return func(o *CallOptions) { o.StreamingFunc = f }
}
