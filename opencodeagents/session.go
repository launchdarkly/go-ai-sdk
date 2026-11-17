package opencodeagents

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

const (
	httpTimeout    = 30 * time.Second
	messageTimeout = 600 * time.Second
)

// httpError is a route the OpenCode server rejected, or a POST /mcp whose
// server did not connect.
type httpError struct {
	Route  string
	Status int
	Detail string
}

func (e *httpError) Error() string {
	if e.Detail != "" {
		return fmt.Sprintf("opencodeagents: %s: %s", e.Route, e.Detail)
	}
	return fmt.Sprintf("opencodeagents: %s -> %d", e.Route, e.Status)
}

// rule is one session permission rule.
type rule struct {
	Permission string `json:"permission"`
	Action     string `json:"action"`
	Pattern    string `json:"pattern"`
}

// message is the POST /session/{id}/message body.
type message struct {
	Parts  []textPart `json:"parts"`
	Model  modelRef   `json:"model"`
	System string     `json:"system,omitempty"`
	Agent  string     `json:"agent,omitempty"`
}

type textPart struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type modelRef struct {
	ProviderID string `json:"providerID"`
	ModelID    string `json:"modelID"`
}

// runSession opens /event, creates one session with the permission ruleset,
// posts the message and hands each bus event to onEvent until this
// session's idle, a quiet drain window after the POST returned, or onEvent
// returns stop. Leaving for any other reason than a finished turn cancels
// the POST and aborts the session.
func runSession(ctx context.Context, s *server, perm []rule, msg message, onEvent func(map[string]any) (stop bool, err error)) (sessionID string, err error) {
	client := s.client()
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	sseReq, _ := http.NewRequestWithContext(ctx, "GET", s.base+"/event", nil)
	sseReq.Header.Set("Accept", "text/event-stream")
	sse, err := client.Do(sseReq)
	if err != nil {
		return "", fmt.Errorf("opencodeagents: GET /event: %w", err)
	}
	defer sse.Body.Close()
	if sse.StatusCode >= 400 {
		return "", &httpError{Route: "GET /event", Status: sse.StatusCode}
	}
	lines := make(chan string)
	readErr := make(chan error, 1)
	go func() {
		sc := bufio.NewScanner(sse.Body)
		sc.Buffer(make([]byte, 64*1024), 16*1024*1024)
		for sc.Scan() {
			select {
			case lines <- sc.Text():
			case <-ctx.Done():
				return
			}
		}
		readErr <- sc.Err()
	}()

	var created struct {
		ID   string `json:"id"`
		Data struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := postJSON(ctx, client, s.base+"/session", map[string]any{"permission": perm}, httpTimeout, &created); err != nil {
		return "", err
	}
	sessionID = created.ID
	if sessionID == "" {
		sessionID = created.Data.ID
	}
	if sessionID == "" {
		return "", &httpError{Route: "POST /session", Status: 200, Detail: "no session id"}
	}

	finished := false
	defer func() {
		if !finished {
			cancel()
			abort(s, sessionID)
		}
	}()

	route := "/session/" + sessionID + "/message"
	posted := make(chan error, 1)
	go func() { posted <- postJSON(ctx, client, s.base+route, msg, messageTimeout, nil) }()

	var drain <-chan time.Time
	var timer *time.Timer
	resetDrain := func() {
		if timer != nil {
			timer.Stop()
		}
		timer = time.NewTimer(drainFallback)
		drain = timer.C
	}
	defer func() {
		if timer != nil {
			timer.Stop()
		}
	}()
	for {
		select {
		case <-ctx.Done():
			return sessionID, ctx.Err()
		case err := <-posted:
			if err != nil {
				return sessionID, err
			}
			posted = nil
			resetDrain()
		case <-drain:
			finished = true
			return sessionID, nil
		case err := <-readErr:
			if err == nil && posted == nil {
				finished = true
				return sessionID, nil
			}
			if err == nil {
				err = io.ErrUnexpectedEOF
			}
			return sessionID, fmt.Errorf("opencodeagents: GET /event: %w", err)
		case line := <-lines:
			if posted == nil {
				resetDrain()
			}
			ev := parseFrame(line)
			if ev == nil {
				continue
			}
			if sid := eventSession(ev); sid != "" && sid != sessionID {
				continue // another session's event
			}
			if ev["type"] == "session.idle" && eventSession(ev) == sessionID {
				if posted != nil { // the POST normally returns first; its result still counts
					if err := <-posted; err != nil {
						return sessionID, err
					}
				}
				finished = true
				return sessionID, nil
			}
			stop, err := onEvent(ev)
			if err != nil || stop {
				return sessionID, err
			}
		}
	}
}

func abort(s *server, sessionID string) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = postJSON(ctx, s.client(), s.base+"/session/"+sessionID+"/abort", nil, httpTimeout, nil)
}

// postJSON posts body and decodes a 2xx response into out when out is set.
func postJSON(ctx context.Context, client *http.Client, url string, body any, timeout time.Duration, out any) error {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	var r io.Reader = http.NoBody
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return err
		}
		r = bytes.NewReader(b)
	}
	req, _ := http.NewRequestWithContext(ctx, "POST", url, r)
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	route := "POST " + strings.TrimPrefix(url, urlBase(url))
	if err != nil {
		return fmt.Errorf("opencodeagents: %s: %w", route, err) // wraps a context error, for errors.Is
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		return &httpError{Route: route, Status: resp.StatusCode}
	}
	if out == nil {
		_, _ = io.Copy(io.Discard, resp.Body)
		return nil
	}
	if err := json.NewDecoder(resp.Body).Decode(out); err != nil {
		return &httpError{Route: route, Status: resp.StatusCode, Detail: "response is not JSON"}
	}
	return nil
}

func urlBase(url string) string {
	if i := strings.Index(url[len("http://"):], "/"); i >= 0 {
		return url[:len("http://")+i]
	}
	return url
}

// registerBridge attaches the bridge to the running instance. A 200 is not
// enough: the server's status must be "connected".
func registerBridge(ctx context.Context, s *server, name, url, token string) error {
	body := map[string]any{"name": name, "config": map[string]any{
		"type": "remote", "url": url, "headers": map[string]any{"Authorization": "Bearer " + token},
		"oauth": false, "timeout": messageTimeout.Milliseconds(),
	}}
	var status map[string]struct {
		Status string `json:"status"`
	}
	if err := postJSON(ctx, s.client(), s.base+"/mcp", body, httpTimeout, &status); err != nil {
		return err
	}
	if got := status[name].Status; got != "connected" {
		return &httpError{Route: "POST /mcp", Status: 200, Detail: fmt.Sprintf("bridge %q status is %q", name, got)}
	}
	return nil
}

// disconnectBridge is best-effort: the child is stopped right after.
func disconnectBridge(s *server, name string) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = postJSON(ctx, s.client(), s.base+"/mcp/"+name+"/disconnect", nil, httpTimeout, nil)
}

// parseFrame decodes one `data: {json}` SSE line; anything else is nil.
func parseFrame(line string) map[string]any {
	text := strings.TrimSpace(line)
	text = strings.TrimSpace(strings.TrimPrefix(text, "data:"))
	if !strings.HasPrefix(text, "{") {
		return nil
	}
	var ev map[string]any
	if json.Unmarshal([]byte(text), &ev) != nil {
		return nil
	}
	return ev
}

// eventSession is the event's session id: on properties, properties.info or
// properties.part. "" when it has none.
func eventSession(ev map[string]any) string {
	p, _ := ev["properties"].(map[string]any)
	if p == nil {
		s, _ := ev["sessionID"].(string)
		return s
	}
	if s, _ := p["sessionID"].(string); s != "" {
		return s
	}
	for _, k := range []string{"info", "part"} {
		if m, _ := p[k].(map[string]any); m != nil {
			if s, _ := m["sessionID"].(string); s != "" {
				return s
			}
		}
	}
	return ""
}
