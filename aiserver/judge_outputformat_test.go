package aiserver

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"strings"
	"sync"
	"testing"
)

var judgeSchema = map[string]any{"type": "object", "properties": map[string]any{"label": map[string]any{"type": "string"}}}

// outputFormatLog captures the SDK logger and counts warnings that say a
// judge's outputFormat is ignored.
type outputFormatLog struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (l *outputFormatLog) Write(p []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.buf.Write(p)
}

func (l *outputFormatLog) warnings(judgeKey string) int {
	l.mu.Lock()
	defer l.mu.Unlock()
	n := 0
	for _, line := range strings.Split(l.buf.String(), "\n") {
		if strings.Contains(line, "level=WARN") && strings.Contains(line, "outputFormat") && strings.Contains(line, judgeKey) {
			n++
		}
	}
	return n
}

func captureSDKLog(t *testing.T) *outputFormatLog {
	l := &outputFormatLog{}
	SetLogger(slog.New(slog.NewTextHandler(l, nil)))
	t.Cleanup(func() { SetLogger(nil) })
	return l
}

func assertNoOutputFormat(t *testing.T, where string, cfg AiConfig) {
	t.Helper()
	if cfg.OutputFormat != nil {
		t.Errorf("%s: judge config still has OutputFormat %v", where, cfg.OutputFormat)
	}
	if _, has := cfg.Raw["outputFormat"]; has {
		t.Errorf("%s: judge config Raw still has outputFormat", where)
	}
}

func TestJudgeOutputFormatDroppedInline(t *testing.T) {
	withFake(t, map[string]any{
		"cfg":   enabledConfig("P", "", judgedConfig(nil)),
		"judge": enabledConfig("P", "judge", map[string]any{"outputFormat": judgeSchema}),
	})
	log := captureSDKLog(t)
	var judgeReqs []HandlerRequest
	h := spanJudgeHandler(`{"score":0.5,"reasoning":"r"}`, false, nil, &judgeReqs)
	resp, err := Config(ConfigArgs{Key: "cfg", Handlers: []*ProviderHandler{h}}).Invoke(bg, "q", userCtx)
	if err != nil {
		t.Fatal(err)
	}
	if r, ok := resp.JudgeResults["judge"]; !ok || r.Score != 0.5 {
		t.Fatalf("judge results %v", resp.JudgeResults)
	}
	if len(judgeReqs) != 1 {
		t.Fatalf("judge calls %d", len(judgeReqs))
	}
	got := judgeReqs[0].Config
	assertNoOutputFormat(t, "inline", got)
	if got.Model.Name != "test-model" || got.Provider.Name != "P" || got.Instructions != "be helpful" {
		t.Errorf("judge config lost its model, provider or prompt: %+v", got)
	}
	if n := log.warnings("judge"); n != 1 {
		t.Errorf("warnings naming the judge and outputFormat = %d, want 1:\n%s", n, log.buf.String())
	}
	// The cached source config keeps its outputFormat.
	src, _, _, err := extractVariation(bg, "judge", userCtx, nil, false)
	if err != nil || src.OutputFormat == nil {
		t.Errorf("source judge config changed: %v %v", src.OutputFormat, err)
	}
}

func TestJudgeOutputFormatDroppedDeferred(t *testing.T) {
	withFake(t, map[string]any{
		"cfg":   enabledConfig("P", "", judgedConfig(nil)),
		"judge": enabledConfig("P", "judge", map[string]any{"outputFormat": judgeSchema}),
	})
	log := captureSDKLog(t)
	var judgeReqs []HandlerRequest
	h := spanJudgeHandler(`{"score":0.5,"reasoning":"r"}`, false, nil, &judgeReqs)
	resp, err := Config(ConfigArgs{Key: "cfg", Handlers: []*ProviderHandler{h}, SkipJudges: true}).Invoke(bg, "q", userCtx)
	if err != nil || len(resp.JudgeTasks) != 1 {
		t.Fatalf("tasks=%v err=%v", resp.JudgeTasks, err)
	}
	assertNoOutputFormat(t, "task", resp.JudgeTasks[0].JudgeConfig)
	if n := log.warnings("judge"); n != 1 {
		t.Errorf("warnings = %d, want 1:\n%s", n, log.buf.String())
	}
	b, err := json.Marshal(resp.JudgeTasks[0])
	if err != nil {
		t.Fatal(err)
	}
	var task JudgeTask
	if err := json.Unmarshal(b, &task); err != nil {
		t.Fatal(err)
	}
	out, err := RunJudge(bg, task, []*ProviderHandler{h})
	if err != nil || out == nil || out.Score != 0.5 {
		t.Fatalf("RunJudge = %+v, %v", out, err)
	}
	assertNoOutputFormat(t, "deferred", judgeReqs[0].Config)
}

// A task serialized by an older SDK may still carry outputFormat.
func TestRunJudgeDropsOutputFormatFromOlderTask(t *testing.T) {
	withFake(t, map[string]any{"judge": enabledConfig("P", "judge", nil)})
	var judgeReqs []HandlerRequest
	h := spanJudgeHandler(`{"score":1,"reasoning":"r"}`, false, nil, &judgeReqs)
	task := JudgeTask{
		ConfigKey: "judge", JudgeProvider: "P", JudgeMode: ModeMessages, ActualOutput: "a", UserContext: userCtx,
	}
	task.JudgeConfig.Model.Name, task.JudgeConfig.Provider.Name, task.JudgeConfig.Instructions = "m", "P", "grade"
	task.JudgeConfig.OutputFormat, task.JudgeConfig.Raw = judgeSchema, map[string]any{"outputFormat": judgeSchema}
	out, err := RunJudge(bg, task, []*ProviderHandler{h})
	if err != nil || out == nil || out.Score != 1 {
		t.Fatalf("RunJudge = %+v, %v", out, err)
	}
	assertNoOutputFormat(t, "older task", judgeReqs[0].Config)
}

func TestJudgeOutputFormatDroppedOnAgentFallback(t *testing.T) {
	withFake(t, map[string]any{
		"cfg": enabledConfig("P", "agent", judgedConfig(nil)),
		"judge": enabledConfig("P", "judge", map[string]any{"outputFormat": judgeSchema, "instructions": nil,
			"messages": []any{map[string]any{"role": "system", "content": "grade it"}}}),
	})
	var judgeCfg *AiConfig
	h := createHandler(ProvidesFor{"P", ModeAgent}, func(_ context.Context, req HandlerRequest) (HandlerResult, error) {
		if req.Variables["response_to_evaluate"] == nil {
			return HandlerResult{Output: "answer"}, nil
		}
		judgeCfg = &req.Config
		return HandlerResult{Output: `{"score":1,"reasoning":"r"}`}, nil
	}, nil, CaptureOff)
	if _, err := Config(ConfigArgs{Key: "cfg", Handlers: []*ProviderHandler{h}}).Invoke(bg, "q", userCtx); err != nil {
		t.Fatal(err)
	}
	if judgeCfg == nil {
		t.Fatal("judge not called")
	}
	if judgeCfg.Instructions != "grade it" || judgeCfg.Messages != nil {
		t.Errorf("messages not collapsed: %q %v", judgeCfg.Instructions, judgeCfg.Messages)
	}
	assertNoOutputFormat(t, "agent fallback", *judgeCfg)
}

func TestJudgeWithoutOutputFormatUnchangedAndSilent(t *testing.T) {
	withFake(t, map[string]any{
		"cfg":   enabledConfig("P", "", judgedConfig(nil)),
		"judge": enabledConfig("P", "judge", nil),
	})
	log := captureSDKLog(t)
	var judgeReqs []HandlerRequest
	h := spanJudgeHandler(`{"score":1,"reasoning":"r"}`, false, nil, &judgeReqs)
	if _, err := Config(ConfigArgs{Key: "cfg", Handlers: []*ProviderHandler{h}}).Invoke(bg, "q", userCtx); err != nil {
		t.Fatal(err)
	}
	src, _, _, _ := extractVariation(bg, "judge", userCtx, nil, false)
	if got := judgeReqs[0].Config; got.Instructions != src.Instructions || got.Model.Name != src.Model.Name || len(got.Raw) != len(src.Raw) {
		t.Errorf("config changed: %+v vs %+v", got, src)
	}
	if strings.Contains(log.buf.String(), "outputFormat") {
		t.Errorf("logged about outputFormat:\n%s", log.buf.String())
	}
}
