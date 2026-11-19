package typesafe

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

const (
	defaultBaseURL = "https://api.typesafe.ai"
	defaultModel   = "jev-latest"
	systemOnePath  = "/v1/systemone"
	clientTimeout  = 10 * time.Second
	maxRetries     = 2
)

// systemOneRequest is the POST /v1/systemone body.
type systemOneRequest struct {
	Model     string                  `json:"model"`
	State     map[string]any          `json:"state"`
	Questions map[string]wireQuestion `json:"questions"`
}

type wireQuestion struct {
	Type         QuestionType `json:"type"`
	Instructions string       `json:"instructions"`
	Criteria     any          `json:"criteria,omitempty"`
}

// systemOneUsage accepts the API's snake_case names and the camelCase ones
// the JavaScript handler also reads.
type systemOneUsage struct {
	InputTokens       *float64 `json:"input_tokens"`
	OutputTokens      *float64 `json:"output_tokens"`
	InputTokensCamel  *float64 `json:"inputTokens"`
	OutputTokensCamel *float64 `json:"outputTokens"`
}

type systemOneResponse struct {
	Model   string          `json:"model"`
	Answers map[string]any  `json:"answers"`
	Usage   *systemOneUsage `json:"usage"`
}

// systemOne makes the one System One call, retrying 408, 429 and 5xx at most
// twice and honouring Retry-After. It reads TYPESAFE_API_KEY and
// TYPESAFE_BASE_URL on every call.
func systemOne(ctx context.Context, body systemOneRequest) (*systemOneResponse, error) {
	key := os.Getenv("TYPESAFE_API_KEY")
	if key == "" {
		return nil, errors.New("typesafe: TYPESAFE_API_KEY is not set")
	}
	base := os.Getenv("TYPESAFE_BASE_URL")
	if base == "" {
		base = defaultBaseURL
	}
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("typesafe: encode request: %w", err)
	}
	client := &http.Client{Timeout: clientTimeout}
	for attempt := 0; ; attempt++ {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(base, "/")+systemOnePath, bytes.NewReader(payload))
		if err != nil {
			return nil, fmt.Errorf("typesafe: %w", err)
		}
		req.Header.Set("Authorization", "Bearer "+key)
		req.Header.Set("Content-Type", "application/json")
		resp, err := client.Do(req)
		if err != nil {
			return nil, fmt.Errorf("typesafe: System One request: %w", err)
		}
		b, readErr := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode/100 == 2 {
			if readErr != nil {
				return nil, fmt.Errorf("typesafe: read response: %w", readErr)
			}
			var out systemOneResponse
			if err := json.Unmarshal(b, &out); err != nil {
				return nil, fmt.Errorf("typesafe: decode response: %w", err)
			}
			return &out, nil
		}
		if attempt >= maxRetries || !retryable(resp.StatusCode) {
			return nil, fmt.Errorf("typesafe: System One returned %d (request id %q): %s",
				resp.StatusCode, resp.Header.Get("x-typesafe-request-id"), strings.TrimSpace(string(b)))
		}
		t := time.NewTimer(retryDelay(attempt, resp.Header.Get("Retry-After")))
		select {
		case <-ctx.Done():
			t.Stop()
			return nil, fmt.Errorf("typesafe: System One request: %w", ctx.Err())
		case <-t.C:
		}
	}
}

func retryable(status int) bool {
	return status == http.StatusRequestTimeout || status == http.StatusTooManyRequests || status >= 500
}

// retryDelay is Retry-After (seconds or an HTTP date) when present, else the
// official SDKs' backoff: 0.5 s doubling to 5 s, with 25 % jitter.
func retryDelay(attempt int, retryAfter string) time.Duration {
	retryAfter = strings.TrimSpace(retryAfter)
	// The upper bound rejects +Inf and values that overflow a time.Duration; NaN fails both checks.
	if s, err := strconv.ParseFloat(retryAfter, 64); err == nil && s >= 0 && s < float64(math.MaxInt64/int64(time.Second)) {
		return time.Duration(s * float64(time.Second))
	}
	if at, err := http.ParseTime(retryAfter); err == nil {
		return max(time.Until(at), 0)
	}
	d := min(500*time.Millisecond<<attempt, 5*time.Second)
	return time.Duration(float64(d) * (1 + 0.25*(2*rand.Float64()-1)))
}
