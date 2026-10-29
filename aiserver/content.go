package aiserver

import (
	"context"
	"encoding/json"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

// MessagePart is one part of a gen_ai.input.messages or
// gen_ai.output.messages entry, in the OpenTelemetry GenAI shape: a "text"
// part carries Content; a "tool_call" part carries ID, Name and Arguments.
type MessagePart struct {
	Type      string `json:"type"`
	Content   string `json:"content,omitempty"`
	ID        string `json:"id,omitempty"`
	Name      string `json:"name,omitempty"`
	Arguments any    `json:"arguments,omitempty"`
}

// TextPart is a "text" message part.
func TextPart(text string) MessagePart { return MessagePart{Type: "text", Content: text} }

// ToolCallPart is a "tool_call" message part.
func ToolCallPart(id, name string, arguments any) MessagePart {
	return MessagePart{Type: "tool_call", ID: id, Name: name, Arguments: arguments}
}

// ChatMessage is one gen_ai.input.messages or gen_ai.output.messages entry.
type ChatMessage struct {
	Role         string        `json:"role"`
	Parts        []MessagePart `json:"parts"`
	FinishReason string        `json:"finish_reason,omitempty"`
}

// ChatContent is the content of one chat call. Empty fields are not written.
type ChatContent struct {
	SystemInstructions string
	InputMessages      []ChatMessage
	ToolDefinitions    []Tool
	OutputMessages     []ChatMessage
}

// SetChatContent writes gen_ai.system_instructions, gen_ai.input.messages,
// gen_ai.tool.definitions and gen_ai.output.messages on span as JSON, only
// when CaptureContentEnabled(ctx).
func SetChatContent(ctx context.Context, span trace.Span, c ChatContent) {
	if !CaptureContentEnabled(ctx) {
		return
	}
	var attrs []attribute.KeyValue
	add := func(key string, v any) {
		if b, err := json.Marshal(v); err == nil {
			attrs = append(attrs, attribute.String(key, string(b)))
		}
	}
	if c.SystemInstructions != "" {
		add("gen_ai.system_instructions", []MessagePart{TextPart(c.SystemInstructions)})
	}
	if len(c.InputMessages) > 0 {
		add("gen_ai.input.messages", c.InputMessages)
	}
	if len(c.ToolDefinitions) > 0 {
		defs := make([]map[string]any, len(c.ToolDefinitions))
		for i, t := range c.ToolDefinitions {
			typ := t.Type
			if typ == "" {
				typ = "function"
			}
			defs[i] = map[string]any{"type": typ, "name": t.Name, "description": t.Description, "parameters": t.Parameters}
		}
		add("gen_ai.tool.definitions", defs)
	}
	if len(c.OutputMessages) > 0 {
		add("gen_ai.output.messages", c.OutputMessages)
	}
	span.SetAttributes(attrs...)
}

// SetToolCallArguments writes gen_ai.tool.call.arguments (a string as is,
// anything else as JSON), only when CaptureContentEnabled(ctx).
func SetToolCallArguments(ctx context.Context, span trace.Span, arguments any) {
	if CaptureContentEnabled(ctx) {
		span.SetAttributes(attribute.String("gen_ai.tool.call.arguments", ToolResultString(arguments)))
	}
}

// SetToolCallResult writes gen_ai.tool.call.result (a string as is,
// anything else as JSON), only when CaptureContentEnabled(ctx).
func SetToolCallResult(ctx context.Context, span trace.Span, result any) {
	if CaptureContentEnabled(ctx) {
		span.SetAttributes(attribute.String("gen_ai.tool.call.result", ToolResultString(result)))
	}
}
