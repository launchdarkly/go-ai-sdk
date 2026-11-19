// Package typesafe is the TypeSafe Jev judge handler for aiserver
// (providesFor TypeSafe/messages). It calls the TypeSafe System One REST API
// directly; there is no TypeSafe SDK for Go.
package typesafe

import (
	"context"
	"encoding/json"
	"os"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/handlerkit"
	"github.com/launchdarkly/go-ai-sdk/internal/kitbase"
	"github.com/launchdarkly/go-ai-sdk/internal/sdkapi"
	"github.com/launchdarkly/go-ai-sdk/internal/telemetry"
)

const (
	// PackageName is the Go import path, registered for `$ld:ai:sdk:info`.
	PackageName = "github.com/launchdarkly/go-ai-sdk/typesafe"
	// PackageVersion is this package's release version.
	PackageVersion = "1.0.0" // x-release-please-version
	// SpanName is the root span name.
	SpanName = "invoke_agent"
)

// spanModel names the chat span and the model identity when the config has
// no model name, as the JavaScript handler does.
const spanModel = "jev"

func init() { handlerkit.RegisterAiSdkPackage(PackageName, PackageVersion) }

// Options configure NewHandler.
type Options struct {
	// CaptureContent gates the judged state and Jev's answers on spans.
	// CaptureDefault follows aiserver.Configure, else on.
	CaptureContent aiserver.ContentCapture
}

// NewHandler returns the TypeSafe Jev judge handler (createTypesafeHandler).
// It reads TYPESAFE_API_KEY, TYPESAFE_BASE_URL and TYPESAFE_DEFAULT_MODEL
// when it runs.
func NewHandler(opts Options) (*aiserver.ProviderHandler, error) {
	handlerkit.ReportUsage("typesafe.createTypesafeHandler", PackageName, PackageVersion)
	return sdkapi.CreateHandler(aiserver.ProvidesFor{Provider: "TypeSafe", Mode: aiserver.ModeMessages}, judge, nil, opts.CaptureContent), nil
}

type result struct {
	Key      string  `json:"key"`
	EventKey string  `json:"eventKey"`
	Score    float64 `json:"score"`
	Reason   string  `json:"reason"`
}

func judge(ctx context.Context, req aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
	questions, err := ExtractQuestions(req.Config)
	if err != nil {
		return aiserver.HandlerResult{}, err
	}
	state := buildState(req)
	stateJSON, err := json.Marshal(state)
	if err != nil {
		return aiserver.HandlerResult{}, err
	}
	model := req.Config.Model.Name
	if model == "" {
		model = spanModel
	}
	// The run's context has capture off so the root carries no content: the
	// state and the answer go on the chat span only, gated by ctx.
	run := handlerkit.StartHandlerRun(telemetry.StartInvocation(ctx, telemetry.CaptureOff),
		kitbase.HandlerSpanOptions{Tracer: PackageName, Name: SpanName, System: "typesafe", RequestModel: model}, req, handlerkit.Prompt{})
	defer run.EndOnPanic()
	chat := run.StartChat(model)
	telemetry.SetChatContent(ctx, chat, telemetry.ChatContent{
		InputMessages: []telemetry.ChatMessage{{Role: "user", Parts: []telemetry.MessagePart{telemetry.TextPart(string(stateJSON))}}},
	})

	body := systemOneRequest{Model: wireModel(req.Config.Model.Name), State: state, Questions: map[string]wireQuestion{}}
	for _, q := range questions {
		body.Questions[q.Key] = wireQuestion{Type: q.Type, Instructions: q.Instructions, Criteria: q.Criteria}
	}
	resp, err := systemOne(run.Context(), body)
	if err != nil {
		run.Fail(err)
		return aiserver.HandlerResult{}, err
	}
	results := make([]result, len(questions))
	for i, q := range questions {
		answer := resp.Answers[q.Key]
		score, err := scoreFromAnswer(q, answer)
		if err != nil {
			run.Fail(err)
			return aiserver.HandlerResult{}, err
		}
		results[i] = result{Key: q.Key, EventKey: q.EventKey, Score: score, Reason: reasonFromAnswer(q, answer)}
	}
	out, err := json.Marshal(struct {
		Kind    string   `json:"kind"`
		Results []result `json:"results"`
	}{"typesafe", results})
	if err != nil {
		run.Fail(err)
		return aiserver.HandlerResult{}, err
	}
	output := string(out)
	if telemetry.Enabled(ctx) {
		writeCompletion(chat, output)
	}
	usage := spanUsage(resp.Usage)
	run.EndChat(chat, resp.Model, usage, "")
	run.Finish(resp.Model, output)
	return aiserver.HandlerResult{Output: output, Usage: map[string]any{"input_tokens": usage.Input, "output_tokens": usage.Output}}, nil
}

// wireModel is the model sent: the config's, else TYPESAFE_DEFAULT_MODEL,
// else jev-latest. The request schema requires one.
func wireModel(name string) string {
	if name != "" {
		return name
	}
	if env := os.Getenv("TYPESAFE_DEFAULT_MODEL"); env != "" {
		return env
	}
	return defaultModel
}

// spanUsage reads either token naming; absent fields count as 0, and no
// usage object marks the turn unreported.
func spanUsage(u *systemOneUsage) kitbase.SpanUsage {
	if u == nil {
		return kitbase.SpanUsage{Unreported: true}
	}
	first := func(a, b *float64) float64 {
		if a != nil {
			return *a
		}
		if b != nil {
			return *b
		}
		return 0
	}
	return kitbase.SpanUsage{Input: first(u.InputTokens, u.InputTokensCamel), Output: first(u.OutputTokens, u.OutputTokensCamel)}
}

// writeCompletion writes the answer as the chat span's completion, in the
// same carriers telemetry.SetChatContent uses for output messages.
func writeCompletion(span trace.Span, output string) {
	msgs, _ := json.Marshal([]telemetry.ChatMessage{{Role: "assistant", Parts: []telemetry.MessagePart{telemetry.TextPart(output)}}})
	span.SetAttributes(attribute.String("gen_ai.output.messages", string(msgs)),
		attribute.String("gen_ai.completion.0.role", "assistant"),
		attribute.String("gen_ai.completion.0.content", output))
	span.AddEvent("gen_ai.content.completion", trace.WithAttributes(attribute.String("gen_ai.completion", output)))
}
