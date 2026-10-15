// Package bedrockwire is the Amazon Bedrock Runtime Converse engine shared by
// the bedrockmessages and bedrockagents handlers: modelId construction,
// LaunchDarkly turns to Converse messages, toolSpecs, inferenceConfig, the caller's Converse options merge, ConverseStream
// accumulation, client construction, and the local tool loop with its
// invoke_agent → chat / execute_tool span tree.
package bedrockwire

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/aws/smithy-go/auth/bearer"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

const (
	// SpanName is the root span name and root gen_ai.operation.name for
	// Bedrock handlers.
	SpanName = "invoke_agent"
	// GenAISystem is the gen_ai.system and gen_ai.provider.name value.
	GenAISystem = "aws.bedrock"
	// DefaultMaxSteps bounds the tool loop when Engine.MaxSteps is zero.
	DefaultMaxSteps = 10
)

// ErrUnsupportedContent reports a history content block Bedrock Converse
// cannot carry without a policy this package does not apply, such as a URL
// image (the handler never downloads images) or an unknown image type.
var ErrUnsupportedContent = errors.New("bedrock: unsupported content block")

// ErrToolNotFound reports a toolUse block naming a tool that was not offered
// (no config tool with a ToolFunc under that name), so this is a model
// or provider fault; failing keeps toolResults one-to-one with toolUses.
var ErrToolNotFound = errors.New("bedrock: no handler registered for tool")

// ErrMaxSteps reports that the model still asked for tools when the turn
// bound was reached.
var ErrMaxSteps = errors.New("bedrock: tool loop exceeded the maximum number of steps")

// NewClient builds a Bedrock Runtime client from the default AWS
// configuration. region, when set, is the endpoint / SigV4 region. The SDK
// applies AWS_BEARER_TOKEN_BEDROCK itself; the option function runs after
// that, so a non-empty apiKey wins.
func NewClient(apiKey, region string) (*bedrockruntime.Client, error) {
	var load []func(*awsconfig.LoadOptions) error
	if region != "" {
		load = append(load, awsconfig.WithRegion(region))
	}
	cfg, err := awsconfig.LoadDefaultConfig(context.Background(), load...)
	if err != nil {
		return nil, fmt.Errorf("load AWS configuration: %w", err)
	}
	return bedrockruntime.NewFromConfig(cfg, func(o *bedrockruntime.Options) {
		if apiKey != "" {
			o.BearerAuthTokenProvider = bearer.StaticTokenProvider{Token: bearer.Token{Value: apiKey}}
			o.AuthSchemePreference = []string{"httpBearerAuth"}
		}
	}), nil
}

// Engine runs one Bedrock handler. Mode selects the behaviour that differs
// between the two packages:
//
//   - ModeMessages: messages-first prompt, best-effort structured output in
//     the system prompt, ConverseOptions called once per provider turn, and
//     blocking output concatenated over every turn.
//   - ModeAgent: instructions-first prompt, native Converse OutputConfig
//     structured output, ConverseOptions called once per run (the Strands
//     model-construction analogue), and blocking output from the final turn.
//
// An Engine is safe for concurrent use once built.
type Engine struct {
	Client          *bedrockruntime.Client
	ConverseOptions func(aiserver.AiConfig) *bedrockruntime.ConverseInput
	Mode            aiserver.Mode
	// Tracer is the instrumentation scope (the package import path).
	Tracer string
	// Prefix starts every error the engine creates, e.g. "bedrockmessages".
	Prefix   string
	MaxSteps int
}

func (e *Engine) errorf(format string, args ...any) error {
	return fmt.Errorf(e.Prefix+": "+format, args...)
}

// runUsage keeps Bedrock's field names with cache buckets unfolded: it is the
// handler's return value and aiserver.ParseUsage folds once.
type runUsage struct{ input, output, total, cacheRead, cacheWrite int32 }

func (u *runUsage) add(x *types.TokenUsage) {
	if x == nil {
		return
	}
	u.input += aws.ToInt32(x.InputTokens)
	u.output += aws.ToInt32(x.OutputTokens)
	u.total += aws.ToInt32(x.TotalTokens)
	u.cacheRead += aws.ToInt32(x.CacheReadInputTokens)
	u.cacheWrite += aws.ToInt32(x.CacheWriteInputTokens)
}

func (u runUsage) raw() map[string]any {
	return map[string]any{
		"inputTokens": float64(u.input), "outputTokens": float64(u.output), "totalTokens": float64(u.total),
		"cacheReadInputTokens": float64(u.cacheRead), "cacheWriteInputTokens": float64(u.cacheWrite),
	}
}

func spanUsage(x *types.TokenUsage) aiserver.SpanUsage {
	var u runUsage
	u.add(x)
	return aiserver.SpanUsage{
		Input:     float64(u.input + u.cacheRead + u.cacheWrite),
		Output:    float64(u.output),
		CacheRead: float64(u.cacheRead), CacheCreation: float64(u.cacheWrite),
	}
}

// turnResult is one provider turn, from either Converse or ConverseStream.
type turnResult struct {
	msg   types.Message
	stop  types.StopReason
	usage *types.TokenUsage
}

// run is the tool loop shared by both paths; onText is nil when blocking.
// abandoned reports that the stream consumer stopped iterating.
func (e *Engine) run(ctx context.Context, req aiserver.HandlerRequest, onText func(string) bool) (out string, usage runUsage, abandoned bool, err error) {
	id := ModelID(req.Config.Model)
	agent := e.Mode == aiserver.ModeAgent
	p := aiserver.ComposePrompt(req, e.Mode)
	var output *types.OutputConfig
	if onText == nil { // streaming ignores outputFormat
		if agent {
			if output, err = outputConfig(req.Config.OutputFormat); err != nil {
				err = e.errorf("encode outputFormat: %w", err)
			}
		} else {
			p.System = aiserver.WithOutputFormat(p.System, req.Config.OutputFormat)
		}
	}
	r := aiserver.StartHandlerRun(ctx, aiserver.HandlerSpanOptions{
		Tracer: e.Tracer, Name: SpanName, System: GenAISystem, Operation: SpanName, RequestModel: id,
	}, req, p)
	defer r.EndOnPanic()
	if err != nil {
		r.Fail(err)
		return "", usage, false, err
	}

	msgs, err := toMessages(p.Turns)
	if err != nil {
		r.Fail(err)
		return "", usage, false, err
	}
	var system []types.SystemContentBlock
	if p.System != "" {
		system = []types.SystemContentBlock{&types.SystemContentBlockMemberText{Value: p.System}}
	}
	tools := buildTools(req.Config, req.ToolHandlers)
	inference := inferenceConfig(req.Config.Model.Parameters)
	callerOptions := func() *bedrockruntime.ConverseInput {
		if e.ConverseOptions == nil {
			return nil
		}
		return e.ConverseOptions(req.Config)
	}
	var runOptions *bedrockruntime.ConverseInput
	if agent {
		runOptions = callerOptions()
	}
	steps := e.MaxSteps
	if steps <= 0 {
		steps = DefaultMaxSteps
	}

	var all, last strings.Builder
	finished := false
	for range steps {
		in := &bedrockruntime.ConverseInput{
			ModelId: aws.String(id), Messages: msgs, System: system, InferenceConfig: inference, ToolConfig: tools,
			OutputConfig: output,
		}
		if agent {
			applyConverseOptions(in, runOptions)
		} else {
			applyConverseOptions(in, callerOptions())
		}
		chat := r.StartChat(id)
		res, stopped, terr := e.turn(r.Context(), in, onText)
		usage.add(res.usage)
		if stopped {
			r.Abandon()
			return "", usage, true, nil
		}
		if terr != nil {
			r.Fail(terr)
			return "", usage, false, terr
		}
		r.EndChat(chat, id, spanUsage(res.usage), finishReason(res.stop))

		last.Reset()
		var results []types.ContentBlock
		for _, b := range res.msg.Content {
			switch v := b.(type) {
			case *types.ContentBlockMemberText:
				all.WriteString(v.Value)
				last.WriteString(v.Value)
			case *types.ContentBlockMemberToolUse:
				name, callID := aws.ToString(v.Value.Name), aws.ToString(v.Value.ToolUseId)
				tool := r.StartTool(name, callID) // opened first so a missing tool shows as a failed call, as in JS
				fn, _ := req.ToolHandlers[name].(aiserver.ToolFunc)
				if _, offered := req.Config.Tools[name]; !offered || fn == nil {
					err := e.errorf("%w: %q", ErrToolNotFound, name)
					r.Fail(err)
					return "", usage, false, err
				}
				result, ferr := e.callTool(r.Context(), fn, name, v.Value)
				if ferr != nil {
					r.Fail(ferr)
					return "", usage, false, ferr // tool errors propagate
				}
				r.EndTool(tool, nil)
				results = append(results, &types.ContentBlockMemberToolResult{Value: types.ToolResultBlock{
					ToolUseId: aws.String(callID),
					Content:   []types.ToolResultContentBlock{&types.ToolResultContentBlockMemberText{Value: aiserver.ToolResultString(result)}},
				}})
			}
		}
		if res.stop != types.StopReasonToolUse || len(results) == 0 {
			finished = true
			break
		}
		msgs = append(msgs, res.msg, types.Message{Role: types.ConversationRoleUser, Content: results})
	}
	if !finished {
		err := e.errorf("%w (%d)", ErrMaxSteps, steps)
		r.Fail(err)
		return "", usage, false, err
	}
	out = all.String()
	if agent && onText == nil {
		out = last.String() // the agent's final message; streams keep every turn
	}
	r.Finish(id, out)
	return out, usage, false, nil
}

func (e *Engine) callTool(ctx context.Context, fn aiserver.ToolFunc, name string, use types.ToolUseBlock) (any, error) {
	args, err := documentArgs(use.Input)
	if err != nil {
		return nil, e.errorf("decode input for tool %q: %w", name, err)
	}
	res, err := fn(ctx, args)
	if err != nil {
		return nil, e.errorf("tool %q: %w", name, err)
	}
	return res, nil
}

// turn performs one provider call: Converse when onText is nil, otherwise
// ConverseStream with onText called per text delta. stopped reports that
// onText returned false.
func (e *Engine) turn(ctx context.Context, in *bedrockruntime.ConverseInput, onText func(string) bool) (turnResult, bool, error) {
	if onText == nil {
		out, err := e.Client.Converse(ctx, in)
		if err != nil {
			return turnResult{}, false, e.errorf("converse: %w", err)
		}
		msg, ok := out.Output.(*types.ConverseOutputMemberMessage)
		if !ok {
			return turnResult{usage: out.Usage}, false, e.errorf("converse returned no message (output %T)", out.Output)
		}
		return turnResult{msg: msg.Value, stop: out.StopReason, usage: out.Usage}, false, nil
	}
	out, err := e.Client.ConverseStream(ctx, toStreamInput(in))
	if err != nil {
		return turnResult{}, false, e.errorf("converse stream: %w", err)
	}
	stream := out.GetStream()
	defer stream.Close() // releases the HTTP body on early exit
	acc := newStreamAccumulator(e.Prefix)
	for ev := range stream.Events() {
		delta, err := acc.add(ev)
		if err != nil {
			return acc.result(), false, err
		}
		if delta != "" && !onText(delta) {
			return acc.result(), true, nil
		}
	}
	if err := stream.Err(); err != nil {
		return acc.result(), false, e.errorf("converse stream: %w", err)
	}
	return acc.result(), false, nil
}

// Invoke is the blocking handler: Converse turns until the model stops
// requesting tools. Usage carries Bedrock's raw keys.
func (e *Engine) Invoke(ctx context.Context, req aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
	out, usage, _, err := e.run(ctx, req, nil)
	if err != nil {
		return aiserver.HandlerResult{}, err
	}
	return aiserver.HandlerResult{Output: out, Usage: usage.raw()}, nil
}

// Stream is the streaming handler: a chunk per ConverseStream text delta,
// then one done event with the text of every turn and the run's usage. A
// consumer break ends the run with abandoned spans and no done event.
func (e *Engine) Stream(ctx context.Context, req aiserver.HandlerRequest) iter.Seq2[aiserver.HandlerStreamEvent, error] {
	return func(yield func(aiserver.HandlerStreamEvent, error) bool) {
		out, usage, abandoned, err := e.run(ctx, req, func(t string) bool {
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

// documentArgs decodes a tool-use input document into the JSON argument
// object, with numbers as float64 like encoding/json.
func documentArgs(d document.Interface) (map[string]any, error) {
	if d == nil {
		return map[string]any{}, nil
	}
	b, err := d.MarshalSmithyDocument()
	if err != nil {
		return nil, err
	}
	var args map[string]any
	if err := json.Unmarshal(b, &args); err != nil {
		return nil, err
	}
	return args, nil
}
