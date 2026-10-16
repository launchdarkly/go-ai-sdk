package main

import (
	"context"
	"errors"
	"fmt"
	"image/color"
	"math"
	"os"
	"strings"
	"sync"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/claudeagents"
	"github.com/launchdarkly/go-ai-sdk/langchainagents"
	"github.com/launchdarkly/go-ai-sdk/openaiagents"
)

func registryConfig(key string, skipJudges bool) *aiserver.AIConfig {
	return aiserver.Config(aiserver.ConfigArgs{Key: key, Registry: aiserver.GlobalRegistry, SkipJudges: skipJudges})
}

func registryGraph(key string) *aiserver.AgentGraph {
	return aiserver.Graph(key, aiserver.GraphOptions{Registry: aiserver.GlobalRegistry})
}

// agent demonstrates Config().Invoke.
func agent(ctx context.Context, key, input string) error {
	resp, err := registryConfig(key, false).Invoke(ctx, input, newContext())
	if err != nil {
		return err
	}
	printJSON(resp)
	return writeOutput(resp)
}

// openaiOnly demonstrates a single-handler wrapper: openaiAgents with its
// own tools and no registry.
func openaiOnly(ctx context.Context, key, input string) error {
	resp, err := openaiagents.OpenAIAgents(ctx, key, input, newContext(), &openaiagents.InvokeOptions{
		ToolHandlers: map[string]aiserver.ToolHandler{
			"search-ld-documentation":          aiserver.ToolFunc(searchLDDocumentation),
			"fetch-launchdarkly-documentation": aiserver.ToolFunc(fetchLDDocumentation),
		},
	})
	if err != nil {
		return err
	}
	printJSON(resp)
	return writeOutput(resp)
}

// streaming demonstrates Config().Stream.
func streaming(ctx context.Context, key, input string) error {
	for ev, err := range registryConfig(key, false).Stream(ctx, input, newContext()) {
		if err != nil {
			return err
		}
		if ev.Type == aiserver.EventChunk {
			fmt.Print(ev.Text)
			continue
		}
		fmt.Println()
		printJSON(map[string]any{"usage": ev.Usage, "judgeResults": ev.JudgeResults})
	}
	return nil
}

func sampleHistory() []aiserver.Message {
	return []aiserver.Message{
		{Role: "user", Text: "My name is Ada and I am evaluating feature flag tools."},
		{Role: "assistant", Text: "Nice to meet you, Ada. What matters most in your evaluation?"},
		{Role: "user", Text: "Safe rollouts and experimentation."},
		{Role: "assistant", Text: "Noted: safe rollouts and experimentation."},
	}
}

// history demonstrates Invoke with prior turns.
func history(ctx context.Context, key, input string) error {
	resp, err := registryConfig(key, false).Invoke(ctx, input, newContext(), aiserver.WithHistory(sampleHistory()))
	if err != nil {
		return err
	}
	printJSON(resp)
	return writeOutput(resp)
}

// conversation groups several turns under one conversation id and prints
// inline judge scores.
func conversation(ctx context.Context, key, input string) error {
	id := newConversationID("conversation")
	fmt.Fprintln(os.Stderr, "[conversation]", id)
	ctx = aiserver.WithConversationID(ctx, id)
	user := newMultiContext()
	cfg := registryConfig(key, false)
	var hist []aiserver.Message
	for i, turn := range []string{input, "Can you give a concrete example?", "Summarise that in one sentence."} {
		resp, err := cfg.Invoke(ctx, turn, user, aiserver.WithHistory(hist))
		if err != nil {
			return err
		}
		out := fmt.Sprint(resp.Response)
		fmt.Printf("--- turn %d ---\n%s\n", i+1, out)
		for k, j := range resp.JudgeResults {
			fmt.Fprintf(os.Stderr, "[judge] %s score=%.2f\n", k, j.Score)
		}
		hist = append(hist, aiserver.Message{Role: "user", Text: turn}, aiserver.Message{Role: "assistant", Text: out})
	}
	return nil
}

// graph demonstrates Graph().Invoke.
func graph(ctx context.Context, key, input string) error {
	resp, err := registryGraph(key).Invoke(ctx, input, newContext(), aiserver.WithVariables(map[string]any{"user_id": "user-123"}))
	if err != nil {
		return err
	}
	printJSON(resp)
	return writeOutput(resp)
}

// graphStreaming builds the stream inside a conversation scope and iterates
// it outside, proving call-time binding. Model text goes to stdout; node
// boundaries go to stderr. Writes no JSON file.
func graphStreaming(ctx context.Context, key, input string) error {
	id := newConversationID("graph-streaming")
	fmt.Fprintln(os.Stderr, "[conversation]", id)
	events := registryGraph(key).Stream(aiserver.WithConversationID(ctx, id), input, newContext())
	nodeTotals := 0.0
	for ev, err := range events {
		if err != nil {
			return err
		}
		switch ev.Type {
		case aiserver.EventNodeStart:
			fmt.Fprintln(os.Stderr, "[node_start]", ev.NodeKey)
		case aiserver.EventChunk:
			fmt.Print(ev.Text)
		case aiserver.EventNodeDone:
			nodeTotals += ev.Usage.Total
			fmt.Fprintf(os.Stderr, "[node_done] %s total=%v\n", ev.NodeKey, ev.Usage.Total)
		case aiserver.EventHandoff:
			fmt.Fprintf(os.Stderr, "[handoff] %s -> %s\n", ev.SourceKey, ev.TargetKey)
		case aiserver.EventDone:
			fmt.Printf("\nUsage:\n  input: %v\n  output: %v\n  total: %v\n", ev.Usage.Input, ev.Usage.Output, ev.Usage.Total)
		}
	}
	return nil
}

// graphHistory runs a graph with a generated image in history, once with the
// question as input and once with the question as the last history turn.
func graphHistory(ctx context.Context, key, _ string) error {
	img := aiserver.ContentBlock{Type: "image", Source: &aiserver.ImageSource{Type: "base64", MediaType: "image/png",
		Data: solidColorPNGBase64(color.RGBA{R: 220, A: 255}, 64)}}
	g := registryGraph(key)
	scenarios := []struct {
		name    string
		input   string
		history []aiserver.Message
	}{
		{"history is context only", "What colour is the image I sent?", []aiserver.Message{{Role: "user", Blocks: []aiserver.ContentBlock{img}}}},
		{"history ends with the user turn", "", []aiserver.Message{{Role: "user", Blocks: []aiserver.ContentBlock{img, {Type: "text", Text: "What colour is this image?"}}}}},
	}
	var failed []string
	for _, s := range scenarios {
		resp, err := g.Invoke(ctx, s.input, newContext(), aiserver.WithHistory(s.history))
		if err != nil {
			return fmt.Errorf("%s: %w", s.name, err)
		}
		fmt.Printf("[%s] %s\n", s.name, resp.Response)
		if !strings.Contains(strings.ToLower(resp.Response), "red") {
			failed = append(failed, s.name)
		}
	}
	if len(failed) > 0 {
		return fmt.Errorf("response did not name the image colour: %s", strings.Join(failed, ", "))
	}
	return nil
}

// nativeGraph resolves a graph and runs it with the Claude native adapter.
func nativeGraph(ctx context.Context, key, input string) error {
	def := aiserver.ResolveGraph(ctx, key, newContext(), aiserver.GraphOptions{Registry: aiserver.GlobalRegistry})
	if !def.Enabled {
		return fmt.Errorf("%w: %s", aiserver.ErrGraphDisabled, key)
	}
	resp, err := claudeagents.ToClaudeAgents(def, &claudeagents.AdapterOptions{ToolHandlers: exampleTools}).
		Invoke(ctx, input, aiserver.WithVariables(map[string]any{"user_id": "user-123"}))
	if err != nil {
		return err
	}
	printJSON(resp)
	return writeOutput(resp)
}

// nativeGraphLangChain resolves the graph and runs it as one state graph
// through langchainagents.ToLangGraph. A disabled or missing
// graph is ToLangGraph's error to report.
func nativeGraphLangChain(ctx context.Context, key, input string) error {
	user := newContext()
	def := aiserver.ResolveGraph(ctx, key, user, aiserver.GraphOptions{Registry: aiserver.GlobalRegistry})
	resp, err := langchainagents.ToLangGraph(def, &langchainagents.NativeGraphOptions{ToolHandlers: exampleTools, Context: user}).
		Invoke(ctx, input, aiserver.WithVariables(map[string]any{"user_id": "user-123"}))
	if err != nil {
		return err
	}
	printJSON(resp)
	return writeOutput(resp)
}

// judge invokes with SkipJudges, runs each judge task on its own goroutine,
// tracks the evaluation metric, and verifies the results.
func judge(ctx context.Context, key, input string) error {
	user := newContext()
	resp, err := registryConfig(key, true).Invoke(ctx, input, user)
	if err != nil {
		return err
	}
	if len(resp.JudgeTasks) == 0 {
		return errors.New("no judge tasks returned; does the config have a judgeConfiguration?")
	}
	handlers := aiserver.GlobalRegistry.Handlers()
	type outcome struct {
		task aiserver.JudgeTask
		res  *aiserver.JudgeRunResult
		err  error
	}
	results := make([]outcome, len(resp.JudgeTasks))
	var wg sync.WaitGroup
	for i, task := range resp.JudgeTasks {
		wg.Go(func() {
			res, err := aiserver.RunJudge(ctx, task, handlers)
			results[i] = outcome{task, res, err}
		})
	}
	wg.Wait()
	ok := true
	for _, o := range results {
		if o.err != nil || o.res == nil {
			fmt.Printf("judge %s: failed: %v\n", o.task.ConfigKey, o.err)
			ok = false
			continue
		}
		if o.task.EvaluationMetricKey != "" {
			aiserver.Track(o.task.EvaluationMetricKey, user, o.res.TrackData, o.res.Score)
		}
		checks := map[string]bool{
			"score ∈ [0,1]":     o.res.Score >= 0 && o.res.Score <= 1 && !math.IsNaN(o.res.Score),
			"reasoning present": o.res.Response != "",
			"usage tokens":      o.res.Usage.Total > 0,
			"trackData.runId":   o.res.TrackData.RunID != "",
		}
		fmt.Printf("judge %s: score=%.2f\n", o.task.ConfigKey, o.res.Score)
		for name, pass := range checks {
			fmt.Printf("  [%s] %s\n", map[bool]string{true: "PASS", false: "FAIL"}[pass], name)
			ok = ok && pass
		}
	}
	if !ok {
		return errors.New("judge verification failed")
	}
	return nil
}
