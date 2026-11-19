package typesafe

import (
	"context"
	"errors"
	"io"
	"math"
	"net/http"
	"os"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/handlerkit"
	"github.com/launchdarkly/go-ai-sdk/internal/judging"
	"github.com/launchdarkly/go-ai-sdk/internal/neverforward"
)

// invoke runs a fresh handler with no content-capture setting.
func invoke(t *testing.T, req aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
	t.Helper()
	h := must(NewHandler(Options{}))
	return h.Invoke(bg, req)
}

func TestFactoryMetadata(t *testing.T) {
	a, errA := NewHandler(Options{})
	b, errB := NewHandler(Options{CaptureContent: aiserver.CaptureOn})
	c, errC := NewHandler(Options{CaptureContent: aiserver.CaptureOff})
	if errA != nil || errB != nil || errC != nil {
		t.Fatalf("NewHandler errors: %v %v %v", errA, errB, errC)
	}
	if a.ProvidesFor != (aiserver.ProvidesFor{Provider: "TypeSafe", Mode: aiserver.ModeMessages}) {
		t.Errorf("ProvidesFor = %+v, want {TypeSafe messages}", a.ProvidesFor)
	}
	if a.CaptureContent != aiserver.CaptureDefault || b.CaptureContent != aiserver.CaptureOn || c.CaptureContent != aiserver.CaptureOff {
		t.Errorf("CaptureContent not passed through: %v %v %v", a.CaptureContent, b.CaptureContent, c.CaptureContent)
	}
	if a == b || a.Invoke == nil {
		t.Error("NewHandler must return independent, callable handlers")
	}
	// Judge-only: there is no stream path.
	if a.Stream != nil {
		t.Error("Stream must be nil")
	}
}

// One POST /v1/systemone with the Bearer key carries every question and the
// model; the body has nothing else.
func TestRequest(t *testing.T) {
	jev := newJev(t, ok(threeAnswers, 100, 20))
	if _, err := invoke(t, aiserver.HandlerRequest{Config: configOf(t, judgeVariation("jev-latest", classifiers)), UserInput: "assistant output"}); err != nil {
		t.Fatal(err)
	}
	reqs := jev.reqs()
	if len(reqs) != 1 {
		t.Fatalf("requests = %d, want one call for every question", len(reqs))
	}
	r := reqs[0]
	if r.method != http.MethodPost || r.path != "/v1/systemone" {
		t.Errorf("request = %s %s, want POST /v1/systemone", r.method, r.path)
	}
	if got := r.header.Get("Authorization"); got != "Bearer test-key" {
		t.Errorf("Authorization = %q, want the TYPESAFE_API_KEY Bearer token", got)
	}
	if got := r.header.Get("Content-Type"); !strings.HasPrefix(got, "application/json") {
		t.Errorf("Content-Type = %q", got)
	}
	keys := slices.Sorted(func(yield func(string) bool) {
		for k := range r.body {
			if !yield(k) {
				return
			}
		}
	})
	if !reflect.DeepEqual(keys, []string{"model", "questions", "state"}) {
		t.Errorf("body keys = %v, want model, questions, state", keys)
	}
	if r.body["model"] != "jev-latest" {
		t.Errorf("model = %v, want config.model.name", r.body["model"])
	}
	want := map[string]any{
		"tone": map[string]any{"type": "noul", "instructions": "Is the tone appropriate?",
			"criteria": map[string]any{"true": "Polite", "false": "Harmful"}},
		"accuracy": map[string]any{"type": "score", "instructions": "How accurate is the response?",
			"criteria": []any{"Great", "Ok", "Bad"}},
		"route": map[string]any{"type": "choice", "instructions": "What is the tone?",
			"criteria": map[string]any{"calm": "Calm", "angry": "Angry"}},
	}
	if !reflect.DeepEqual(r.body["questions"], want) {
		t.Errorf("questions =\n%v\nwant\n%v", r.body["questions"], want)
	}

	t.Run("noul without criteria sends no criteria", func(t *testing.T) {
		jev := newJev(t, ok(`{"q":{"type":"noul","noul":0.9}}`, 1, 1))
		labels := []any{map[string]any{"key": "q", "eventKey": "eq", "instructions": "q?", "type": "noul"}}
		if _, err := invoke(t, aiserver.HandlerRequest{Config: configOf(t, judgeVariation("jev", labels))}); err != nil {
			t.Fatal(err)
		}
		q := jev.reqs()[0].body["questions"].(map[string]any)["q"]
		if !reflect.DeepEqual(q, map[string]any{"type": "noul", "instructions": "q?"}) {
			t.Errorf("question = %v", q)
		}
	})
}

// The model is config.model.name, else TYPESAFE_DEFAULT_MODEL, else jev-latest.
func TestModelFallback(t *testing.T) {
	for name, tc := range map[string]struct{ config, env, want string }{
		"config wins":      {"jev-custom", "jev-env", "jev-custom"},
		"env default":      {"", "jev-env", "jev-env"},
		"built-in default": {"", "", "jev-latest"},
	} {
		t.Run(name, func(t *testing.T) {
			jev := newJev(t, ok(`{"q":{"type":"noul","noul":0.9}}`, 1, 1))
			if tc.env != "" {
				t.Setenv("TYPESAFE_DEFAULT_MODEL", tc.env)
			}
			labels := []any{map[string]any{"key": "q", "eventKey": "eq", "instructions": "q?", "type": "noul"}}
			if _, err := invoke(t, aiserver.HandlerRequest{Config: configOf(t, judgeVariation(tc.config, labels))}); err != nil {
				t.Fatal(err)
			}
			if got := jev.reqs()[0].body["model"]; got != tc.want {
				t.Errorf("model = %v, want %s", got, tc.want)
			}
		})
	}
}

// recordTransport answers every request itself, so nothing is dialled.
type recordTransport struct{ urls []string }

func (r *recordTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	r.urls = append(r.urls, req.URL.String())
	return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": {"application/json"}}, Request: req,
		Body: io.NopCloser(strings.NewReader(`{"model":"jev","answers":{"q":{"type":"noul","noul":0.9}},"usage":{"input_tokens":1,"output_tokens":1}}`))}, nil
}

// With TYPESAFE_BASE_URL unset the client calls the public API. The default
// transport is swapped for a recorder, so the request never leaves the test.
func TestDefaultBaseURL(t *testing.T) {
	t.Setenv("TYPESAFE_BASE_URL", "")
	os.Unsetenv("TYPESAFE_BASE_URL")
	rec := &recordTransport{}
	prev := http.DefaultTransport
	http.DefaultTransport = rec
	t.Cleanup(func() { http.DefaultTransport = prev })
	labels := []any{map[string]any{"key": "q", "eventKey": "eq", "instructions": "q?", "type": "noul"}}
	if _, err := invoke(t, aiserver.HandlerRequest{Config: configOf(t, judgeVariation("jev", labels))}); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(rec.urls, []string{"https://api.typesafe.ai/v1/systemone"}) {
		t.Errorf("requests = %v, want one to https://api.typesafe.ai/v1/systemone", rec.urls)
	}
}

// The key comes only from TYPESAFE_API_KEY. Without it the call fails before
// any request.
func TestMissingAPIKey(t *testing.T) {
	jev := newJev(t, ok(`{"q":{"type":"noul","noul":0.9}}`, 1, 1))
	os.Unsetenv("TYPESAFE_API_KEY") // newJev's t.Setenv restores it
	labels := []any{map[string]any{"key": "q", "eventKey": "eq", "instructions": "q?", "type": "noul"}}
	_, err := invoke(t, aiserver.HandlerRequest{Config: configOf(t, judgeVariation("jev", labels))})
	if err == nil || !strings.Contains(err.Error(), "TYPESAFE_API_KEY") {
		t.Errorf("err = %v, want an error naming TYPESAFE_API_KEY", err)
	}
	if n := len(jev.reqs()); n != 0 {
		t.Errorf("requests = %d, want none without a key", n)
	}
}

// The judged state holds the judge variables that are present, without the
// shared {score, reasoning} formatting block.
func TestState(t *testing.T) {
	labels := []any{map[string]any{"key": "q", "eventKey": "eq", "instructions": "q?", "type": "noul"}}
	state := func(t *testing.T, req aiserver.HandlerRequest) map[string]any {
		t.Helper()
		jev := newJev(t, ok(`{"q":{"type":"noul","noul":0.9}}`, 1, 1))
		req.Config = configOf(t, judgeVariation("jev", labels))
		if _, err := invoke(t, req); err != nil {
			t.Fatal(err)
		}
		s, _ := jev.reqs()[0].body["state"].(map[string]any)
		return s
	}

	t.Run("every variable", func(t *testing.T) {
		s := state(t, aiserver.HandlerRequest{
			UserInput: "the answer",
			Variables: map[string]any{
				"input": "the question", "response_to_evaluate": "the answer",
				"message_history": "the question\n\nthe answer\n\n" + judging.FormattingInstructions,
				"trajectory":      "called search", "expected_output": "an answer",
				"ground_truth_context": "the docs say so",
			},
			History: []aiserver.Message{{Role: "user", Text: "earlier"}},
		})
		for k, want := range map[string]any{
			"input": "the question", "output": "the answer", "trajectory": "called search",
			"expected_output": "an answer", "ground_truth_context": "the docs say so",
			"history": []any{map[string]any{"role": "user", "content": "earlier"}},
		} {
			if !reflect.DeepEqual(s[k], want) {
				t.Errorf("state[%s] = %#v, want %#v", k, s[k], want)
			}
		}
		mh, _ := s["message_history"].(string)
		if !strings.Contains(mh, "the question") || strings.Contains(mh, "valid JSON format") || strings.Contains(mh, `"score"`) {
			t.Errorf("message_history = %q, want the conversation without the formatting block", mh)
		}
		if _, has := s["user_input"]; has {
			t.Error("user_input equal to the output must be absent")
		}
	})
	t.Run("user input that is not the output", func(t *testing.T) {
		s := state(t, aiserver.HandlerRequest{UserInput: "typed", Variables: map[string]any{"response_to_evaluate": "the answer"}})
		if s["user_input"] != "typed" || s["output"] != "the answer" {
			t.Errorf("state = %v", s)
		}
	})
	t.Run("absent and empty variables are left out", func(t *testing.T) {
		s := state(t, aiserver.HandlerRequest{Variables: map[string]any{"input": "", "response_to_evaluate": "the answer"}})
		if !reflect.DeepEqual(s, map[string]any{"output": "the answer"}) {
			t.Errorf("state = %v, want only output", s)
		}
	})
	t.Run("every empty-string variable is left out", func(t *testing.T) {
		s := state(t, aiserver.HandlerRequest{UserInput: "typed", Variables: map[string]any{
			"input": "", "response_to_evaluate": "", "message_history": "", "trajectory": "",
			"expected_output": "", "ground_truth_context": ""}})
		if !reflect.DeepEqual(s, map[string]any{"user_input": "typed"}) {
			t.Errorf("state = %v, want only user_input", s)
		}
	})
}

// The JavaScript handler's main case: one result per label, in label order,
// and the single call's usage.
func TestAnswers(t *testing.T) {
	newJev(t, ok(threeAnswers, 100, 20))
	res, err := invoke(t, aiserver.HandlerRequest{Config: configOf(t, judgeVariation("jev-latest", classifiers)), UserInput: "assistant output"})
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]any{"kind": "typesafe", "results": []any{
		map[string]any{"key": "tone", "eventKey": "$ld:ai:judge:jev:tone", "score": 0.25, "reason": "Harmful"},
		map[string]any{"key": "accuracy", "eventKey": "$ld:ai:judge:jev:accuracy", "score": 0.5, "reason": "Ok"},
		map[string]any{"key": "route", "eventKey": "$ld:ai:judge:jev:route", "score": 0.8, "reason": "Calm"},
	}}
	if got := output(t, res); !reflect.DeepEqual(got, want) {
		t.Errorf("output =\n%v\nwant\n%v", got, want)
	}
	if u := handlerkit.ParseUsage(res.Usage); u.Input != 100 || u.Output != 20 || u.Total != 120 {
		t.Errorf("usage = %+v from %v, want 100 in / 20 out", u, res.Usage)
	}
}

// Score and reason mapping per question type.
func TestScoreAndReason(t *testing.T) {
	noul := map[string]any{"type": "noul", "criteria": map[string]any{"true": "Polite", "false": "Harmful"}}
	bareNoul := map[string]any{"type": "noul"}
	halfNoul := map[string]any{"type": "noul", "criteria": map[string]any{"true": "Polite"}}
	choice := map[string]any{"type": "choice", "criteria": map[string]any{"calm": "A neutral message", "angry": nil}}
	score := map[string]any{"type": "score", "criteria": []any{"Great", "Ok", "Bad"}}
	twoLevel := map[string]any{"type": "score", "criteria": []any{"No", "Yes"}}
	legend := `"legend":{"0":"Great","1":"Ok","2":"Bad"}`
	cases := []struct {
		name     string
		question map[string]any
		answer   string
		score    float64
		reason   string
	}{
		{"noul at 0.5 selects true", noul, `{"type":"noul","noul":0.5}`, 0.5, "Polite"},
		{"noul below 0.5 selects false", noul, `{"type":"noul","noul":0.49}`, 0.49, "Harmful"},
		{"noul without criteria names the side", bareNoul, `{"type":"noul","noul":0.9}`, 0.9, "true"},
		{"noul false side without criteria", bareNoul, `{"type":"noul","noul":0.1}`, 0.1, "false"},
		{"noul side whose text is absent", halfNoul, `{"type":"noul","noul":0.2}`, 0.2, "false"},
		{"choice probability first", choice, `{"type":"choice","choice":"calm","confidence":0.1,"probabilities":{"calm":0.8,"angry":0.2}}`, 0.8, "A neutral message"},
		{"choice confidence next", choice, `{"type":"choice","choice":"angry","confidence":0.6}`, 0.6, "angry"},
		{"choice selected with no number", choice, `{"type":"choice","choice":"angry"}`, 1, "angry"},
		{"score index over levels - 1", score, `{"type":"score","score":1,` + legend + `}`, 0.5, "Ok"},
		{"fractional score, nearest legend entry", score, `{"type":"score","score":1.7,` + legend + `}`, 0.85, "Bad"},
		{"low fractional score", score, `{"type":"score","score":0.4,` + legend + `}`, 0.2, "Great"},
		{"no legend: criteria length and entry", score, `{"type":"score","score":1}`, 0.5, "Ok"},
		{"legend length wins over criteria", score, `{"type":"score","score":2,"legend":{"0":"L0","1":"L1","2":"L2","3":"L3"}}`, 2.0 / 3, "L2"},
		{"clamped above", score, `{"type":"score","score":5,` + legend + `}`, 1, "Bad"},
		{"clamped below", score, `{"type":"score","score":-1,` + legend + `}`, 0, "Great"},
		{"two-level rubric divides by 1", twoLevel, `{"type":"score","score":0.4,"legend":{"0":"No","1":"Yes"}}`, 0.4, "No"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			newJev(t, ok(`{"q":`+tc.answer+`}`, 1, 1))
			label := map[string]any{"key": "q", "eventKey": "eq", "instructions": "q?"}
			for k, v := range tc.question {
				label[k] = v
			}
			res, err := invoke(t, aiserver.HandlerRequest{Config: configOf(t, judgeVariation("jev", []any{label}))})
			if err != nil {
				t.Fatal(err)
			}
			results, _ := output(t, res)["results"].([]any)
			if len(results) != 1 {
				t.Fatalf("results = %v", results)
			}
			r := results[0].(map[string]any)
			if got, _ := r["score"].(float64); math.Abs(got-tc.score) > 1e-9 || r["reason"] != tc.reason {
				t.Errorf("score %v reason %q, want %v %q", r["score"], r["reason"], tc.score, tc.reason)
			}
		})
	}
}

// An invalid classifiers list rejects the call before any request.
func TestInvalidClassifiersRejectBeforeRequest(t *testing.T) {
	jev := newJev(t, ok(`{}`, 1, 1))
	_, err := invoke(t, aiserver.HandlerRequest{Config: configOf(t, judgeVariation("jev", []any{}))})
	if !errors.Is(err, ErrInvalidClassifiers) {
		t.Errorf("err = %v, want ErrInvalidClassifiers", err)
	}
	if n := len(jev.reqs()); n != 0 {
		t.Errorf("requests = %d, want none", n)
	}
}

// An answer without its value rejects the call, as in the JavaScript handler.
func TestMissingAnswerRejects(t *testing.T) {
	for name, rp := range map[string]reply{
		"no answers field":  {body: `{"model":"jev","usage":{"input_tokens":1,"output_tokens":1}}`},
		"no answer":         ok(`{}`, 1, 1),
		"noul without noul": ok(`{"q":{"type":"noul"}}`, 1, 1),
	} {
		t.Run(name, func(t *testing.T) {
			jev := newJev(t, rp)
			labels := []any{map[string]any{"key": "q", "eventKey": "eq", "instructions": "q?", "type": "noul"}}
			_, err := invoke(t, aiserver.HandlerRequest{Config: configOf(t, judgeVariation("jev", labels))})
			// The request must have been made, so the error is the answer's.
			if n := len(jev.reqs()); err == nil || n != 1 {
				t.Errorf("err = %v after %d requests, want an error after the one call", err, n)
			}
		})
	}
}

// The handler forwards no model.parameters key and offers no tools: the
// body is the state, the questions and the model.
func TestNeverForwarded(t *testing.T) {
	jev := newJev(t, ok(threeAnswers, 1, 1))
	v := judgeVariation("jev-latest", classifiers)
	params := map[string]any{"temperature": 0.2, "max_tokens": 100, "top_p": 0.9, "model": "smuggled-model",
		"questions": map[string]any{"smuggled": true}, "state": map[string]any{"smuggled": true}}
	for k, p := range neverforward.Params {
		params[k] = p
	}
	v["model"].(map[string]any)["parameters"] = params
	v["tools"] = map[string]any{"lookup": map[string]any{"name": "lookup", "type": "function", "parameters": map[string]any{"type": "object"}}}
	ran := false
	_, err := invoke(t, aiserver.HandlerRequest{Config: configOf(t, v), UserInput: "assistant output",
		ToolHandlers: map[string]aiserver.ToolHandler{"lookup": aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) { ran = true; return "r", nil })}})
	if err != nil {
		t.Fatal(err)
	}
	r := jev.reqs()[0]
	if len(r.body) != 3 || r.body["model"] != "jev-latest" {
		t.Errorf("body = %v, want only model, questions and state", r.body)
	}
	b := must(jsonText(r.body))
	if strings.Contains(b, "smuggled") || strings.Contains(b, neverforward.Marker) || strings.Contains(b, "lookup") {
		t.Errorf("body carries a model parameter or a tool: %s", b)
	}
	for k := range r.header {
		if strings.Contains(strings.Join(r.header[k], ","), neverforward.Marker) {
			t.Errorf("header %s carries a model parameter", k)
		}
	}
	if ran {
		t.Error("a tool ran")
	}
}

// A retryable status (408, 429, 5xx) is retried; others are not.
func TestRetries(t *testing.T) {
	good := ok(`{"q":{"type":"noul","noul":0.9}}`, 1, 1)
	labels := []any{map[string]any{"key": "q", "eventKey": "eq", "instructions": "q?", "type": "noul"}}
	for _, status := range []int{408, 429, 500, 502, 503} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			jev := newJev(t, reply{status: status, body: `{"detail":"busy"}`, headers: map[string]string{"Retry-After": "0"}}, good)
			if _, err := invoke(t, aiserver.HandlerRequest{Config: configOf(t, judgeVariation("jev", labels))}); err != nil {
				t.Fatalf("err = %v, want success after one retry", err)
			}
			if n := len(jev.reqs()); n != 2 {
				t.Errorf("requests = %d, want 2", n)
			}
		})
	}
	for _, status := range []int{400, 401, 422} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			jev := newJev(t, reply{status: status, body: `{"detail":"bad"}`}, good)
			if _, err := invoke(t, aiserver.HandlerRequest{Config: configOf(t, judgeVariation("jev", labels))}); err == nil {
				t.Error("want an error")
			}
			if n := len(jev.reqs()); n != 1 {
				t.Errorf("requests = %d, want 1 (no retry)", n)
			}
		})
	}
	t.Run("two retries, then the error", func(t *testing.T) {
		jev := newJev(t, reply{status: 500, body: `{"detail":"down"}`, headers: map[string]string{"Retry-After": "0"}})
		if _, err := invoke(t, aiserver.HandlerRequest{Config: configOf(t, judgeVariation("jev", labels))}); err == nil {
			t.Error("want an error")
		}
		if n := len(jev.reqs()); n != 3 {
			t.Errorf("requests = %d, want 3", n)
		}
	})
}

// Retry-After sets the wait before the retry.
func TestRetryAfter(t *testing.T) {
	jev := newJev(t, reply{status: 429, body: `{"detail":"slow down"}`, headers: map[string]string{"Retry-After": "1"}},
		ok(`{"q":{"type":"noul","noul":0.9}}`, 1, 1))
	labels := []any{map[string]any{"key": "q", "eventKey": "eq", "instructions": "q?", "type": "noul"}}
	start := time.Now()
	if _, err := invoke(t, aiserver.HandlerRequest{Config: configOf(t, judgeVariation("jev", labels))}); err != nil {
		t.Fatal(err)
	}
	if n, d := len(jev.reqs()), time.Since(start); n != 2 || d < 950*time.Millisecond {
		t.Errorf("requests = %d after %v, want 2 after Retry-After's 1s", n, d)
	}
}

// Cancelling ctx stops the call and returns the context's error.
func TestCancellation(t *testing.T) {
	release := make(chan struct{})
	jev := newJev(t, ok(`{"q":{"type":"noul","noul":0.9}}`, 1, 1))
	jev.srv.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-release:
		case <-r.Context().Done():
		}
	})
	t.Cleanup(func() { close(release) })
	ctx, cancel := context.WithTimeout(bg, 100*time.Millisecond)
	defer cancel()
	labels := []any{map[string]any{"key": "q", "eventKey": "eq", "instructions": "q?", "type": "noul"}}
	start := time.Now()
	_, err := must(NewHandler(Options{})).Invoke(ctx, aiserver.HandlerRequest{Config: configOf(t, judgeVariation("jev", labels))})
	if !errors.Is(err, context.DeadlineExceeded) || time.Since(start) > 2*time.Second {
		t.Errorf("err = %v after %v, want context.DeadlineExceeded promptly", err, time.Since(start))
	}
}
