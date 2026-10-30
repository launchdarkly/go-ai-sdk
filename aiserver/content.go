package aiserver

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

// MessagePart is one part of a gen_ai.input.messages or
// gen_ai.output.messages entry, in the OpenTelemetry GenAI shape: a "text"
// part carries Content; a "tool_call" part carries Name, and optionally ID
// and Arguments.
type MessagePart struct {
	Type      string
	Content   string
	ID        string
	Name      string
	Arguments any
}

// MarshalJSON writes only the members the part's type uses, in the Python
// SDK's key order: a text part is {type, content}, a tool call
// {type, id?, name, arguments?}.
func (p MessagePart) MarshalJSON() ([]byte, error) {
	type textPart struct {
		Type    string `json:"type"`
		Content string `json:"content"`
	}
	type toolCallPart struct {
		Type      string `json:"type"`
		ID        string `json:"id,omitempty"`
		Name      string `json:"name"`
		Arguments any    `json:"arguments,omitempty"`
	}
	if p.Type == "tool_call" {
		return json.Marshal(toolCallPart{p.Type, p.ID, p.Name, p.Arguments})
	}
	return json.Marshal(textPart{p.Type, p.Content})
}

// text flattens the part for the OpenLLMetry carrier: text as is, a tool call
// as {"name", "arguments"} JSON.
func (p MessagePart) text() string {
	if p.Type != "tool_call" {
		return p.Content
	}
	b, _ := json.Marshal(struct {
		Name      string `json:"name"`
		Arguments any    `json:"arguments,omitempty"`
	}{p.Name, p.Arguments})
	return string(b)
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

// text joins the message's non-empty parts with newlines.
func (m ChatMessage) text() string {
	var lines []string
	for _, p := range m.Parts {
		if t := p.text(); t != "" {
			lines = append(lines, t)
		}
	}
	return strings.Join(lines, "\n")
}

// ChatContent is the content of one chat call. Empty fields are not written.
type ChatContent struct {
	SystemInstructions string
	InputMessages      []ChatMessage
	ToolDefinitions    []Tool
	OutputMessages     []ChatMessage
}

// SetChatContent writes a chat call's content on span, only when
// CaptureContentEnabled(ctx), in three carriers that cannot disagree: the
// GenAI JSON attributes (gen_ai.system_instructions, gen_ai.input.messages,
// gen_ai.tool.definitions, gen_ai.output.messages); the OpenLLMetry numbered
// attributes (gen_ai.prompt.N.*, with the system instructions as message 0,
// and gen_ai.completion.N.*), which LaunchDarkly's trace view reads; and the
// gen_ai.content.prompt / gen_ai.content.completion events. The prompt event
// is written whenever capture is on; the output side writes nothing without
// output messages.
func SetChatContent(ctx context.Context, span trace.Span, c ChatContent) {
	if !CaptureContentEnabled(ctx) {
		return
	}
	var attrs []attribute.KeyValue
	addJSON := func(key string, v any) {
		if b, err := json.Marshal(v); err == nil {
			attrs = append(attrs, attribute.String(key, string(b)))
		}
	}
	if c.SystemInstructions != "" {
		addJSON("gen_ai.system_instructions", []MessagePart{TextPart(c.SystemInstructions)})
	}
	if len(c.InputMessages) > 0 {
		addJSON("gen_ai.input.messages", c.InputMessages)
	}
	prompt := c.InputMessages
	if c.SystemInstructions != "" {
		prompt = append([]ChatMessage{{Role: "system", Parts: []MessagePart{TextPart(c.SystemInstructions)}}}, prompt...)
	}
	attrs = append(attrs, openLLMetry("gen_ai.prompt", prompt)...)
	if len(c.ToolDefinitions) > 0 {
		defs := make([]map[string]any, len(c.ToolDefinitions))
		for i, t := range c.ToolDefinitions {
			d := map[string]any{"type": "function", "name": t.Name}
			if t.Description != "" {
				d["description"] = t.Description
			}
			if t.Parameters != nil {
				d["parameters"] = t.Parameters
			}
			defs[i] = d
		}
		addJSON("gen_ai.tool.definitions", defs)
	}
	if len(c.OutputMessages) > 0 {
		addJSON("gen_ai.output.messages", c.OutputMessages)
		attrs = append(attrs, openLLMetry("gen_ai.completion", c.OutputMessages)...)
	}
	span.SetAttributes(attrs...)

	lines := make([]string, len(prompt))
	for i, m := range prompt {
		lines[i] = m.Role + ": " + m.text()
	}
	span.AddEvent("gen_ai.content.prompt", trace.WithAttributes(attribute.String("gen_ai.prompt", strings.Join(lines, "\n"))))
	if len(c.OutputMessages) > 0 {
		out := make([]string, len(c.OutputMessages))
		for i, m := range c.OutputMessages {
			out[i] = m.text()
		}
		span.AddEvent("gen_ai.content.completion", trace.WithAttributes(attribute.String("gen_ai.completion", strings.Join(out, "\n"))))
	}
}

func openLLMetry(prefix string, msgs []ChatMessage) []attribute.KeyValue {
	var attrs []attribute.KeyValue
	for i, m := range msgs {
		attrs = append(attrs,
			attribute.String(fmt.Sprintf("%s.%d.role", prefix, i), m.Role),
			attribute.String(fmt.Sprintf("%s.%d.content", prefix, i), m.text()))
	}
	return attrs
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
