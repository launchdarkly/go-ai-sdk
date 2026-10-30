// Package aiserver is the core LaunchDarkly AI SDK client for Go: AI Config
// resolution, provider-handler routing, tracking, and judges.
package aiserver

import (
	"context"
	"encoding/json"
	"iter"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
)

// Mode is the handler mode a ProviderHandler serves.
type Mode string

const (
	ModeMessages Mode = "messages"
	ModeAgent    Mode = "agent"
)

// WildcardProvider matches any provider name.
const WildcardProvider = "*"

// ProvidesFor is the [provider, mode] tuple. The zero value means
// "no providesFor": such a handler is never deduplicated by a Registry.
type ProvidesFor struct {
	Provider string
	Mode     Mode
}

func (p ProvidesFor) IsZero() bool { return p == ProvidesFor{} }

// ContentBlock is the LaunchDarkly-canonical multimodal block.
type ContentBlock struct {
	Type   string       `json:"type"`
	Text   string       `json:"text,omitempty"`
	Source *ImageSource `json:"source,omitempty"`
}

type ImageSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
	URL       string `json:"url,omitempty"`
}

// Message is one conversation turn. Content is either Text or Blocks; the JSON
// form is `content: string | ContentBlock[]`.
type Message struct {
	Role   string
	Text   string
	Blocks []ContentBlock
}

func (m Message) MarshalJSON() ([]byte, error) {
	var content any = m.Text
	if m.Blocks != nil {
		content = m.Blocks
	}
	return json.Marshal(struct {
		Role    string `json:"role"`
		Content any    `json:"content"`
	}{m.Role, content})
}

func (m *Message) UnmarshalJSON(b []byte) error {
	var raw struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	}
	if err := json.Unmarshal(b, &raw); err != nil {
		return err
	}
	m.Role = raw.Role
	if len(raw.Content) > 0 && raw.Content[0] == '[' {
		return json.Unmarshal(raw.Content, &m.Blocks)
	}
	if len(raw.Content) > 0 && string(raw.Content) != "null" {
		return json.Unmarshal(raw.Content, &m.Text)
	}
	return nil
}

type Model struct {
	Name       string         `json:"name"`
	Region     string         `json:"region,omitempty"`
	Parameters map[string]any `json:"parameters,omitempty"`
	Custom     map[string]any `json:"custom,omitempty"`
}

type Tool struct {
	Name             string         `json:"name"`
	Type             string         `json:"type"`
	Description      string         `json:"description,omitempty"`
	Parameters       map[string]any `json:"parameters"`
	CustomParameters map[string]any `json:"customParameters,omitempty"`
}

type JudgeRef struct {
	Key          string  `json:"key"`
	SamplingRate float64 `json:"samplingRate"`
}

type JudgeConfiguration struct {
	Judges []JudgeRef `json:"judges,omitempty"`
}

// AiConfig is the AiConfigRep shape the JavaScript and Python SDKs share. Raw keeps the full variation so fields
// this struct does not model (e.g. experimental `skills`) survive a round
// trip, matching TypeScript's `raw as AiConfigRep` passthrough.
type AiConfig struct {
	Instructions string    `json:"instructions,omitempty"`
	Messages     []Message `json:"messages,omitempty"`
	Model        Model     `json:"model"`
	Provider     struct {
		Name string `json:"name"`
	} `json:"provider"`
	Tools               map[string]Tool     `json:"tools,omitempty"`
	JudgeConfiguration  *JudgeConfiguration `json:"judgeConfiguration,omitempty"`
	EvaluationMetricKey string              `json:"evaluationMetricKey,omitempty"`
	OutputFormat        map[string]any      `json:"outputFormat,omitempty"`
	Raw                 map[string]any      `json:"-"`
}

// VariationMeta is the variation's `_ldMeta`.
type VariationMeta struct {
	Enabled      bool   `json:"enabled"`
	VariationKey string `json:"variationKey,omitempty"`
	Version      int    `json:"version,omitempty"`
	Mode         string `json:"mode,omitempty"`
	// ModelKey / ModelVersion identify a pinned model config; stamped onto
	// TrackData when present.
	ModelKey     string `json:"modelKey,omitempty"`
	ModelVersion *int   `json:"modelVersion,omitempty"`
}

type InputDetails struct {
	Uncached      float64 `json:"uncached"`
	CacheRead     float64 `json:"cacheRead"`
	CacheCreation float64 `json:"cacheCreation"`
}

type TokenUsage struct {
	Input        float64       `json:"input"`
	Output       float64       `json:"output"`
	Total        float64       `json:"total"`
	InputDetails *InputDetails `json:"inputDetails,omitempty"`
}

// TrackData is the payload on every `$ld:ai:*` event.
type TrackData struct {
	RunID          string `json:"runId"`
	ConfigKey      string `json:"configKey"`
	VariationKey   string `json:"variationKey"`
	Version        int    `json:"version"`
	ModelName      string `json:"modelName"`
	ProviderName   string `json:"providerName"`
	GraphKey       string `json:"graphKey,omitempty"`
	ToolKey        string `json:"toolKey,omitempty"`
	JudgeConfigKey string `json:"judgeConfigKey,omitempty"`
	EnvironmentID  string `json:"environmentId,omitempty"`
	ModelKey       string `json:"modelKey,omitempty"`
	ModelVersion   *int   `json:"modelVersion,omitempty"`
	extra          *extraTrackData
}

// extraTrackData holds fields merged into every event's data. A pointer
// keeps TrackData comparable.
type extraTrackData struct{ fields map[string]any }

// ToolHandler is a value in a toolHandlers map: a ToolFunc or a NativeTool.
type ToolHandler interface{ isToolHandler() }

// ToolFunc is a locally executed tool. The calling convention is
// open in other SDKs (TS passes arbitrary positional args); Go fixes it to the decoded JSON
// argument object the model sent.
type ToolFunc func(ctx context.Context, args map[string]any) (any, error)

func (ToolFunc) isToolHandler() {}

// NativeTool marks a provider built-in tool. After WrapToolHandlers,
// Track emits `$ld:ai:tool_call`; this replaces TypeScript's zero-arg stub
// carrying the instance under NATIVE_TOOL_KEY.
type NativeTool struct {
	ToolName string
	track    func()
}

func (*NativeTool) isToolHandler() {}

// Track records a native tool call. It is a no-op on an unwrapped NativeTool.
func (n *NativeTool) Track() {
	if n.track != nil {
		n.track()
	}
}

// HandlerRequest carries the positional handler arguments of the JS and Python SDKs
// (config, userInput, toolHandlers, variables, history).
type HandlerRequest struct {
	Config       AiConfig
	UserInput    string
	ToolHandlers map[string]ToolHandler
	Variables    map[string]any
	History      []Message
}

// HandlerResult is `{ output?, usage? }`. Output is a string, or a decoded
// JSON value for first-class structured output.
type HandlerResult struct {
	Output any
	Usage  map[string]any
}

// HandlerStreamEvent is a handler's stream item: Type "chunk" or "done".
type HandlerStreamEvent struct {
	Type   EventType
	Text   string
	Output *string
	Usage  map[string]any
}

type (
	InvokeFunc func(ctx context.Context, req HandlerRequest) (HandlerResult, error)
	StreamFunc func(ctx context.Context, req HandlerRequest) iter.Seq2[HandlerStreamEvent, error]
)

// ProviderHandler is the Go shape of a JS-style "callable with providesFor,
// captureContent and optional stream attached". Go functions cannot carry
// properties, so the handler is a struct holding the functions.
type ProviderHandler struct {
	ProvidesFor    ProvidesFor
	CaptureContent bool
	Invoke         InvokeFunc
	Stream         StreamFunc // optional; nil falls back to Invoke
}

// CreateHandler attaches providesFor and captureContent to a handler.
func CreateHandler(pf ProvidesFor, invoke InvokeFunc, stream StreamFunc, captureContent bool) *ProviderHandler {
	reportUsage("client.createHandler", "", "")
	return createHandler(pf, invoke, stream, captureContent)
}

// createHandler is CreateHandler without the usage report. Handler packages
// call internal/sdkapi.CreateHandler, which builds the same struct.
func createHandler(pf ProvidesFor, invoke InvokeFunc, stream StreamFunc, captureContent bool) *ProviderHandler {
	return &ProviderHandler{ProvidesFor: pf, CaptureContent: captureContent, Invoke: invoke, Stream: stream}
}

type JudgeResult struct {
	Score    float64    `json:"score"`
	Response string     `json:"response"`
	Usage    TokenUsage `json:"usage"`
}

// ProviderResponse is the result of Config(...).Invoke.
type ProviderResponse struct {
	Response     any                    `json:"response"`
	Usage        TokenUsage             `json:"usage"`
	TrackData    TrackData              `json:"trackData"`
	JudgeResults map[string]JudgeResult `json:"judgeResults,omitempty"` // nil when judges were skipped
	JudgeTasks   []JudgeTask            `json:"judgeTasks,omitempty"`   // non-nil only when SkipJudges
}

// JudgeTask is a serialisable deferred judge.
type JudgeTask struct {
	ConfigKey           string            `json:"configKey"`
	JudgeConfig         AiConfig          `json:"judgeConfig"`
	JudgeMeta           VariationMeta     `json:"judgeMeta"`
	ActualOutput        string            `json:"actualOutput"`
	UserInput           string            `json:"userInput"`
	Trajectory          string            `json:"trajectory"`
	UserContext         ldcontext.Context `json:"userContext"`
	JudgeProvider       string            `json:"judgeProvider"`
	JudgeMode           Mode              `json:"judgeMode"`
	CollapseMessages    bool              `json:"collapseMessages"`
	EvaluationMetricKey string            `json:"evaluationMetricKey,omitempty"`
	ParentTrackData     TrackData         `json:"parentTrackData"`
}

// StreamEvent is a public stream item from Config(...).Stream.
type StreamEvent struct {
	Type         EventType              `json:"type"` // "chunk" | "done"
	Text         string                 `json:"text,omitempty"`
	Response     string                 `json:"response,omitempty"`
	Usage        TokenUsage             `json:"usage"`
	JudgeResults map[string]JudgeResult `json:"judgeResults,omitempty"`
}

// EventType names a stream event. String constants still compare with it, so
// `ev.Type == EventChunk` keeps working.
type EventType string

// Stream event types for Config(...).Stream, handler streams and Graph(...).Stream.
const (
	EventChunk     EventType = "chunk"
	EventDone      EventType = "done"
	EventNodeStart EventType = "node_start"
	EventNodeDone  EventType = "node_done"
	EventHandoff   EventType = "handoff"
)

// CallOptions are the optional per-call inputs to Invoke and Stream.
type CallOptions struct {
	// Variables fill {{placeholders}} in the config's prompts.
	Variables map[string]any
	// History is prior conversation, placed before the input. In a
	// graph it reaches the root node only.
	History []Message
}

// CallOption sets one field of CallOptions.
type CallOption func(*CallOptions)

// WithVariables passes template variables to the call.
func WithVariables(v map[string]any) CallOption { return func(o *CallOptions) { o.Variables = v } }

// WithHistory passes prior conversation turns to the call.
func WithHistory(h []Message) CallOption { return func(o *CallOptions) { o.History = h } }

// NewCallOptions applies opts in order. For handler and adapter packages that
// accept CallOption themselves.
func NewCallOptions(opts ...CallOption) CallOptions {
	var o CallOptions
	for _, opt := range opts {
		opt(&o)
	}
	return o
}

// Text returns the response as text: a string as-is, a parsed structured
// output (outputFormat) as JSON.
func (r ProviderResponse) Text() string { return outputString(r.Response) }
