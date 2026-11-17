package claudecli

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"testing"
	"time"
)

// The test binary doubles as a fake claude CLI: Options.Path is os.Args[0]
// and Options.Env sets GO_CLAUDE_FAKE=1. The fake plays a fakeScript and
// appends what it saw, one JSON object per line, to GO_CLAUDE_FAKE_RECORD,
// so a killed fake still leaves its record.
func TestMain(m *testing.M) {
	switch os.Getenv("GO_CLAUDE_FAKE") {
	case "1":
		os.Exit(runFake())
	case "sleep": // a grandchild for the process-group test
		time.Sleep(time.Hour)
		os.Exit(0)
	}
	killGrace = 300 * time.Millisecond
	os.Exit(m.Run())
}

type fakeScript struct {
	Version     string     `json:"version"`     // printed for -v; default "2.1.325 (Claude Code)"
	InitError   string     `json:"initError"`   // answer initialize with this error
	InitDelayMs int        `json:"initDelayMs"` // wait before answering initialize
	Steps       []fakeStep `json:"steps"`
}

// fakeStep does one thing, in this order of precedence.
type fakeStep struct {
	Send        json.RawMessage `json:"send,omitempty"`        // write this message to stdout
	SendRaw     string          `json:"sendRaw,omitempty"`     // write this line to stdout verbatim
	Hook        string          `json:"hook,omitempty"`        // send hook_callback for the id registered for this event
	HookInput   json.RawMessage `json:"hookInput,omitempty"`   // its input
	Request     json.RawMessage `json:"request,omitempty"`     // send a control_request with this body and wait for its response
	Stderr      string          `json:"stderr,omitempty"`      // write to stderr
	StderrBytes int             `json:"stderrBytes,omitempty"` // write this many filler bytes to stderr first
	Child       bool            `json:"child,omitempty"`       // start a sleeping grandchild
	Exit        *int            `json:"exit,omitempty"`        // exit now
	Hang        bool            `json:"hang,omitempty"`        // ignore signals and stdin EOF; wait to be killed
}

type fakeEvent struct {
	Kind      string          `json:"kind"` // version, start, recv, eof, child
	Args      []string        `json:"args,omitempty"`
	Env       []string        `json:"env,omitempty"`
	Dir       string          `json:"dir,omitempty"`
	PID       int             `json:"pid,omitempty"`
	Line      json.RawMessage `json:"line,omitempty"`
	AfterInit bool            `json:"afterInit,omitempty"`
}

func runFake() int {
	rec, err := os.OpenFile(os.Getenv("GO_CLAUDE_FAKE_RECORD"), os.O_WRONLY|os.O_APPEND|os.O_CREATE, 0o600)
	if err != nil {
		return 90
	}
	var recMu sync.Mutex
	record := func(e fakeEvent) {
		b, _ := json.Marshal(e)
		recMu.Lock()
		defer recMu.Unlock()
		_, _ = rec.Write(append(b, '\n'))
	}
	var script fakeScript
	if b, err := os.ReadFile(os.Getenv("GO_CLAUDE_FAKE_SCRIPT")); err != nil || json.Unmarshal(b, &script) != nil {
		return 91
	}
	if len(os.Args) == 2 && os.Args[1] == "-v" {
		record(fakeEvent{Kind: "version", Args: os.Args[1:]})
		v := script.Version
		if v == "" {
			v = "2.1.325 (Claude Code)"
		}
		fmt.Println(v)
		return 0
	}
	dir, _ := os.Getwd()
	record(fakeEvent{Kind: "start", Args: os.Args[1:], Env: os.Environ(), Dir: dir, PID: os.Getpid()})

	var outMu sync.Mutex
	send := func(v any) {
		b, _ := json.Marshal(v)
		outMu.Lock()
		defer outMu.Unlock()
		_, _ = os.Stdout.Write(append(b, '\n'))
	}
	sendRaw := func(s string) {
		outMu.Lock()
		defer outMu.Unlock()
		_, _ = os.Stdout.WriteString(s + "\n")
	}

	var initAnswered atomic.Bool
	lines := make(chan map[string]any, 64)
	go func() {
		sc := bufio.NewScanner(os.Stdin)
		sc.Buffer(make([]byte, 1<<20), 1<<24)
		for sc.Scan() {
			line := append([]byte(nil), sc.Bytes()...)
			record(fakeEvent{Kind: "recv", Line: line, AfterInit: initAnswered.Load()})
			var m map[string]any
			_ = json.Unmarshal(line, &m)
			if req, _ := m["request"].(map[string]any); m["type"] == "control_request" && req["subtype"] == "interrupt" {
				send(map[string]any{"type": "control_response", "response": map[string]any{"subtype": "success", "request_id": m["request_id"], "response": map[string]any{}}})
			}
			lines <- m
		}
		record(fakeEvent{Kind: "eof"})
		close(lines)
	}()

	init, ok := <-lines
	if !ok || init["type"] != "control_request" {
		return 92
	}
	if req, _ := init["request"].(map[string]any); req["subtype"] != "initialize" {
		return 93
	}
	time.Sleep(time.Duration(script.InitDelayMs) * time.Millisecond)
	if script.InitError != "" {
		send(map[string]any{"type": "control_response", "response": map[string]any{"subtype": "error", "request_id": init["request_id"], "error": script.InitError}})
		initAnswered.Store(true)
		for range lines {
		}
		return 1
	}
	send(map[string]any{"type": "control_response", "response": map[string]any{"subtype": "success", "request_id": init["request_id"], "response": map[string]any{"commands": []any{}}}})
	initAnswered.Store(true)
	if _, ok := <-lines; !ok { // the user message
		return 94
	}

	hookIDs := map[string]string{}
	if req, _ := init["request"].(map[string]any); req != nil {
		hooks, _ := req["hooks"].(map[string]any)
		for event, v := range hooks {
			if ms, _ := v.([]any); len(ms) > 0 {
				m, _ := ms[0].(map[string]any)
				if ids, _ := m["hookCallbackIds"].([]any); len(ids) > 0 {
					hookIDs[event], _ = ids[0].(string)
				}
			}
		}
	}

	nextID := 0
	request := func(body map[string]any) {
		nextID++
		id := fmt.Sprintf("cli_%d", nextID)
		send(map[string]any{"type": "control_request", "request_id": id, "request": body})
		for m := range lines {
			if resp, _ := m["response"].(map[string]any); m["type"] == "control_response" && resp["request_id"] == id {
				return
			}
		}
	}
	for _, s := range script.Steps {
		switch {
		case s.Send != nil:
			sendRaw(string(s.Send))
		case s.SendRaw != "":
			sendRaw(s.SendRaw)
		case s.Hook != "":
			var in any
			_ = json.Unmarshal(s.HookInput, &in)
			request(map[string]any{"subtype": "hook_callback", "callback_id": hookIDs[s.Hook], "input": in, "tool_use_id": nil})
		case s.Request != nil:
			var body map[string]any
			_ = json.Unmarshal(s.Request, &body)
			request(body)
		case s.Stderr != "" || s.StderrBytes > 0:
			_, _ = os.Stderr.WriteString(strings.Repeat("x", s.StderrBytes) + s.Stderr)
		case s.Child:
			c := exec.Command(os.Args[0])
			c.Env = append(os.Environ(), "GO_CLAUDE_FAKE=sleep")
			if c.Start() == nil {
				record(fakeEvent{Kind: "child", PID: c.Process.Pid})
			}
		case s.Exit != nil:
			return *s.Exit
		case s.Hang:
			signal.Ignore(os.Interrupt, syscall.SIGTERM)
			for range lines {
			}
			time.Sleep(time.Hour)
		}
	}
	for range lines { // a run ends when the SDK closes stdin
	}
	return 0
}

// fakeRecord is what one fake saw. Starts counts runs, not -v checks.
type fakeRecord struct {
	Versions int
	Starts   int
	Args     []string
	Env      []string
	Dir      string
	PID      int
	Recv     []fakeEvent
	EOF      bool
	ChildPID int
}

// newFake writes the script and returns Options that run it, plus a reader
// for the record.
func newFake(t *testing.T, s fakeScript) (Options, func() fakeRecord) {
	t.Helper()
	dir := t.TempDir()
	scriptPath, recPath := dir+"/script.json", dir+"/record.jsonl"
	b, err := json.Marshal(s)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(scriptPath, b, 0o600); err != nil {
		t.Fatal(err)
	}
	opts := Options{
		Path: os.Args[0],
		Env:  []string{"GO_CLAUDE_FAKE=1", "GO_CLAUDE_FAKE_SCRIPT=" + scriptPath, "GO_CLAUDE_FAKE_RECORD=" + recPath},
	}
	read := func() fakeRecord {
		t.Helper()
		var r fakeRecord
		b, err := os.ReadFile(recPath)
		if os.IsNotExist(err) {
			return r
		}
		if err != nil {
			t.Fatal(err)
		}
		for _, line := range strings.Split(strings.TrimSpace(string(b)), "\n") {
			var e fakeEvent
			if err := json.Unmarshal([]byte(line), &e); err != nil {
				t.Fatalf("record line %q: %v", line, err)
			}
			switch e.Kind {
			case "version":
				r.Versions++
			case "start":
				r.Starts++
				r.Args, r.Env, r.Dir, r.PID = e.Args, e.Env, e.Dir, e.PID
			case "recv":
				r.Recv = append(r.Recv, e)
			case "eof":
				r.EOF = true
			case "child":
				r.ChildPID = e.PID
			}
		}
		return r
	}
	return opts, read
}

// msg is a JSON literal for a script step.
func msg(s string) json.RawMessage { return json.RawMessage(s) }

const resultLine = `{"type":"result","subtype":"success","is_error":false,"result":"done","num_turns":1,"stop_reason":"end_turn","usage":{"input_tokens":3,"output_tokens":4},"session_id":"s1"}`

const assistantLine = `{"type":"assistant","message":{"model":"claude-x","content":[{"type":"text","text":"hi"}],"usage":{"input_tokens":1,"output_tokens":2}},"parent_tool_use_id":null,"session_id":"s1"}`

func exitCode(n int) *int { return &n }
