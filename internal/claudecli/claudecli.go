// Package claudecli runs one Claude Code CLI session over its stream-JSON
// control protocol, the way the Claude Agent SDKs drive the CLI: one
// process per run, an initialize handshake, the prompt as a user message,
// and the CLI's hook and SDK MCP requests answered through callbacks.
//
// It is the transport for the claudeagents CLI runtime. It knows nothing
// about AI Configs, spans or tools beyond the callbacks it is given.
package claudecli

import (
	"context"
	"encoding/json"
	"errors"
	"iter"
	"time"
)

// MinVersion is the oldest claude CLI Run accepts.
const MinVersion = "2.0.0"

// ErrVersion is returned when the CLI reports a version older than
// MinVersion. The error message names both versions.
var ErrVersion = errors.New("claudecli: claude CLI is older than the supported minimum")

// killGrace is how long an interrupted CLI gets to exit before its process
// group is killed.
var killGrace = 5 * time.Second

// Hook events Run registers when Options.OnHook is set.
const (
	HookPreToolUse         = "PreToolUse"
	HookPostToolUse        = "PostToolUse"
	HookPostToolUseFailure = "PostToolUseFailure"
)

// Options configure one Run.
type Options struct {
	// Path is the CLI executable. Empty means "claude" on PATH.
	Path string
	// Dir is the CLI's working directory. Empty means the current process's.
	Dir string
	// Env holds KEY=VALUE entries applied last, over the inherited
	// environment and CLAUDE_CODE_ENTRYPOINT.
	Env []string

	// SystemPrompt is always sent as --system-prompt, so empty replaces the
	// CLI's default system prompt with none, as the Agent SDKs do.
	SystemPrompt string
	// Tools is the built-in tool set, always sent: empty sends --tools "".
	Tools []string
	// AllowedTools is sent as --allowedTools, followed by one Skill(name)
	// entry per Skills name.
	AllowedTools []string
	// Model is sent as --model when non-empty.
	Model string
	// MaxTurns is sent as --max-turns when positive.
	MaxTurns int
	// SettingSources is always sent as --setting-sources=<csv>, so nil
	// loads no host settings.
	SettingSources []string
	// MCPServers names the SDK MCP servers announced in --mcp-config. Their
	// requests reach OnMCP.
	MCPServers []string
	// PluginDirs are sent as one --plugin-dir each.
	PluginDirs []string
	// PartialMessages sends --include-partial-messages (stream_event
	// messages).
	PartialMessages bool

	// Agents is the initialize request's agents map, omitted when empty.
	Agents map[string]any
	// Skills is the initialize request's skills list, omitted when empty.
	Skills []string

	// OnHook, when set, registers the tool hooks in initialize and is
	// called for each hook callback. A returned error is sent back to the
	// CLI as the request's error.
	OnHook func(ctx context.Context, in HookInput) error
	// OnMCP answers one JSON-RPC message for an SDK MCP server. A nil reply
	// (a notification) is acknowledged with an empty result.
	OnMCP func(ctx context.Context, server string, msg json.RawMessage) (json.RawMessage, error)
}

// HookInput is the input of a hook_callback request.
type HookInput struct {
	Event        string          `json:"hook_event_name"`
	ToolName     string          `json:"tool_name"`
	ToolUseID    string          `json:"tool_use_id"`
	ToolInput    json.RawMessage `json:"tool_input,omitempty"`
	ToolResponse json.RawMessage `json:"tool_response,omitempty"`
	Error        string          `json:"error,omitempty"`
}

// Message is one message the CLI writes to stdout. Control messages are
// handled inside Run and never yielded.
type Message struct {
	// Type is "system", "assistant", "user", "stream_event" or "result".
	Type    string
	Subtype string
	// Assistant is set for Type "assistant", User for "user" and Result for
	// "result".
	Assistant *Assistant
	User      *User
	Result    *Result
	// Raw is the whole line as the CLI wrote it.
	Raw json.RawMessage
}

// Assistant is an assistant message.
type Assistant struct {
	Model           string
	ParentToolUseID string
	Content         []Block
	Usage           Usage
}

// User is a user message, which carries tool results back to the model.
type User struct {
	ParentToolUseID string
	Content         []Block
}

// Block is one content block.
type Block struct {
	Type      string          `json:"type"` // text, thinking, tool_use, tool_result, ...
	Text      string          `json:"text,omitempty"`
	Thinking  string          `json:"thinking,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   json.RawMessage `json:"content,omitempty"`
	IsError   bool            `json:"is_error,omitempty"`
}

// Usage is the Anthropic token usage the CLI reports.
type Usage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
}

// Result is the run's final message.
type Result struct {
	Subtype    string `json:"subtype"`
	IsError    bool   `json:"is_error"`
	Result     string `json:"result"`
	NumTurns   int    `json:"num_turns"`
	StopReason string `json:"stop_reason"`
	Usage      Usage  `json:"usage"`
}

// Run checks the CLI version, starts the CLI, completes the initialize
// handshake, sends prompt and yields the CLI's messages up to and including
// the result. The run ends at the result: stdin is closed and the process
// is waited for.
//
// Stopping the iteration early, or cancelling ctx, sends an interrupt,
// closes stdin and, after a grace period, kills the CLI's process group.
// A cancelled run yields ctx's error last. Errors from a CLI that exits
// early include the tail of its stderr.
func Run(ctx context.Context, opts Options, prompt string) iter.Seq2[Message, error] {
	return func(yield func(Message, error) bool) {
		yield(Message{}, errors.New("claudecli: not implemented"))
	}
}
