package opencodeagents

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/launchdarkly/go-sdk-common/v3/ldvalue"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

// The handler starts the test binary itself as `opencode`: TestMain turns it
// into a scripted `opencode serve` when LD_OPENCODE_FAKE is set in the child
// environment the handler passes. The fake records what it was sent in a JSON
// state file the test reads back.
func TestMain(m *testing.M) {
	switch os.Getenv("LD_OPENCODE_FAKE") {
	case "serve":
		os.Exit(runFake())
	case "grandchild":
		time.Sleep(time.Hour)
		os.Exit(0)
	}
	os.Exit(m.Run())
}

var bg = context.Background()
var user = ldcontext.New("user-123")

// fakeSpec scripts one fake `opencode serve`.
type fakeSpec struct {
	Events      []map[string]any   `json:"events,omitempty"`      // sent on /event once the message is posted
	Calls       [][]map[string]any `json:"calls,omitempty"`       // per started process, in start order; overrides Events
	HoldMessage bool               `json:"holdMessage,omitempty"` // the message POST blocks until abort or disconnect
	Status      int                `json:"status,omitempty"`      // message POST status; 0 = 200
	MCPStatus   string             `json:"mcpStatus,omitempty"`   // POST /mcp status; "" = connected
	BridgeCalls []bridgeCall       `json:"bridgeCalls,omitempty"` // tools/call requests sent to the bridge before the events
	ExitStderr  string             `json:"exitStderr,omitempty"`  // write this to stderr and exit 1 at start
	Unhealthy   bool               `json:"unhealthy,omitempty"`   // /global/health answers healthy:false
	IgnoreTerm  bool               `json:"ignoreTerm,omitempty"`  // survive SIGTERM
	Grandchild  bool               `json:"grandchild,omitempty"`  // start a long-lived child process
}

type bridgeCall struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

// fakeState is what one fake process recorded.
type fakeState struct {
	PID           int              `json:"pid"`
	GrandchildPID int              `json:"grandchildPid"`
	Argv          []string         `json:"argv"`
	Cwd           string           `json:"cwd"`
	Env           []string         `json:"env"`
	Requests      []string         `json:"requests"` // "METHOD /path", in arrival order
	Sessions      []map[string]any `json:"sessions"` // POST /session bodies
	Messages      []map[string]any `json:"messages"` // POST /session/{id}/message bodies
	Aborts        []string         `json:"aborts"`
	MCP           []map[string]any `json:"mcp"` // POST /mcp bodies
	Disconnects   []string         `json:"disconnects"`
	BridgeStream  string           `json:"bridgeStream"` // first bytes of the bridge's GET stream
	BridgeTools   []map[string]any `json:"bridgeTools"`  // tools/list result
	BridgeResults []map[string]any `json:"bridgeResults"`
	Terminated    bool             `json:"terminated"` // SIGTERM arrived
	AuthFailures  int              `json:"authFailures"`
}

func (s fakeState) env(key string) (string, bool) {
	for _, kv := range s.Env {
		if k, v, ok := strings.Cut(kv, "="); ok && k == key {
			return v, true
		}
	}
	return "", false
}

// fake is the test side of one scripted OpenCode.
type fake struct {
	dir, cfg, stateDir, specPath string
}

func newFake(t *testing.T, spec fakeSpec) *fake {
	t.Helper()
	root := t.TempDir()
	f := &fake{dir: filepath.Join(root, "work"), cfg: filepath.Join(root, "opencode.json"), stateDir: filepath.Join(root, "state"), specPath: filepath.Join(root, "spec.json")}
	for _, d := range []string{f.dir, f.stateDir} {
		if err := os.Mkdir(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	b, _ := json.Marshal(spec)
	if err := os.WriteFile(f.specPath, b, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(f.cfg, []byte(`{}`), 0o600); err != nil {
		t.Fatal(err)
	}
	return f
}

// options run the fake as the handler's command.
func (f *fake) options() Options {
	return Options{
		WorkDir: f.dir, ConfigPath: f.cfg, Command: []string{os.Args[0]},
		Env: []string{"LD_OPENCODE_FAKE=serve", "LD_OPENCODE_FAKE_SPEC=" + f.specPath, "LD_OPENCODE_FAKE_DIR=" + f.stateDir},
	}
}

func (f *fake) handler(t *testing.T) *aiserver.ProviderHandler {
	t.Helper()
	h, err := NewHandler(f.options())
	if err != nil {
		t.Fatalf("NewHandler: %v", err)
	}
	return h
}

// states returns every started process's state, in start order.
func (f *fake) states(t *testing.T) []fakeState {
	t.Helper()
	names, _ := filepath.Glob(filepath.Join(f.stateDir, "*.json"))
	var out []fakeState
	for _, n := range names {
		b, err := os.ReadFile(n)
		if err != nil {
			t.Fatal(err)
		}
		var s fakeState
		if err := json.Unmarshal(b, &s); err != nil {
			t.Fatalf("%s: %v", n, err)
		}
		out = append(out, s)
	}
	slices.SortFunc(out, func(a, b fakeState) int { return strings.Compare(stateOrder(f, a), stateOrder(f, b)) })
	return out
}

func stateOrder(f *fake, s fakeState) string {
	st, err := os.Stat(filepath.Join(f.stateDir, ".order", strconv.Itoa(s.PID)))
	if err != nil {
		return ""
	}
	return st.ModTime().Format(time.RFC3339Nano)
}

// state returns the only process's state.
func (f *fake) state(t *testing.T) fakeState {
	t.Helper()
	all := f.states(t)
	if len(all) != 1 {
		t.Fatalf("want one opencode process, got %d", len(all))
	}
	return all[0]
}

// --- bus events ---

func assistant(id string) map[string]any {
	return map[string]any{"type": "message.updated", "properties": map[string]any{"info": map[string]any{
		"id": id, "sessionID": "ses", "role": "assistant", "time": map[string]any{"created": 1}}}}
}

func userMessage(id string) map[string]any {
	return map[string]any{"type": "message.updated", "properties": map[string]any{"info": map[string]any{
		"id": id, "sessionID": "ses", "role": "user", "time": map[string]any{"created": 1}}}}
}

func completed(id string, in, out int) map[string]any {
	return map[string]any{"type": "message.updated", "properties": map[string]any{"info": map[string]any{
		"id": id, "sessionID": "ses", "role": "assistant", "time": map[string]any{"created": 1, "completed": 2},
		"tokens": map[string]any{"input": in, "output": out, "cache": map[string]any{"read": 0, "write": 0}}}}}
}

func completedNoTokens(id string) map[string]any {
	return map[string]any{"type": "message.updated", "properties": map[string]any{"info": map[string]any{
		"id": id, "sessionID": "ses", "role": "assistant", "time": map[string]any{"created": 1, "completed": 2}}}}
}

func delta(msg, part, text string) map[string]any {
	p := map[string]any{"sessionID": "ses", "messageID": msg, "field": "text", "delta": text}
	if part != "" {
		p["partID"] = part
	}
	return map[string]any{"type": "message.part.delta", "properties": p}
}

func snapshot(msg, part, text string) map[string]any {
	return map[string]any{"type": "message.part.updated", "properties": map[string]any{"part": map[string]any{
		"id": part, "sessionID": "ses", "messageID": msg, "type": "text", "text": text}}}
}

// toolPart is a tool part update. tool may be "{{bridge:<key>}}", which the
// fake replaces with the bridged tool's OpenCode name.
func toolPart(msg, callID, tool, status string, input any, output string) map[string]any {
	state := map[string]any{"status": status}
	if input != nil {
		state["input"] = input
	}
	if status == "completed" {
		state["output"] = output
	}
	if status == "error" {
		state["error"] = output
	}
	return map[string]any{"type": "message.part.updated", "properties": map[string]any{"part": map[string]any{
		"id": "part-" + callID, "sessionID": "ses", "messageID": msg, "type": "tool", "callID": callID, "tool": tool, "state": state}}}
}

func idle(session string) map[string]any {
	p := map[string]any{}
	if session != "" {
		p["sessionID"] = session
	}
	return map[string]any{"type": "session.idle", "properties": p}
}

func sessionError(name, body string) map[string]any {
	return map[string]any{"type": "session.error", "properties": map[string]any{"sessionID": "ses",
		"error": map[string]any{"name": name, "data": map[string]any{"message": body}}}}
}

func otherSession(e map[string]any) map[string]any {
	b, _ := json.Marshal(e)
	var c map[string]any
	_ = json.Unmarshal(bytes.ReplaceAll(b, []byte(`"ses"`), []byte(`"other"`)), &c)
	return c
}

// reply is a one-message assistant turn: role, text deltas, completion, idle.
func reply(text string, in, out int) []map[string]any {
	return []map[string]any{assistant("m1"), delta("m1", "p1", text), completed("m1", in, out), idle("ses")}
}

// --- the fake server (child side) ---

type fakeServer struct {
	spec     fakeSpec
	events   []map[string]any
	stateDir string
	password string

	mu       sync.Mutex
	st       fakeState
	posted   chan struct{} // closed once the message POST arrived
	released chan struct{} // closed on abort
	once     sync.Once
	relOnce  sync.Once
	bridge   struct{ name, url, token string }
}

func runFake() int {
	b, err := os.ReadFile(os.Getenv("LD_OPENCODE_FAKE_SPEC"))
	if err != nil {
		return 2
	}
	var spec fakeSpec
	if json.Unmarshal(b, &spec) != nil {
		return 2
	}
	s := &fakeServer{spec: spec, stateDir: os.Getenv("LD_OPENCODE_FAKE_DIR"), password: os.Getenv("OPENCODE_SERVER_PASSWORD"),
		posted: make(chan struct{}), released: make(chan struct{})}
	wd, _ := os.Getwd()
	s.st = fakeState{PID: os.Getpid(), Argv: os.Args, Cwd: wd, Env: os.Environ()}
	_ = os.MkdirAll(filepath.Join(s.stateDir, ".order"), 0o755)
	prior, _ := filepath.Glob(filepath.Join(s.stateDir, "*.json"))
	_ = os.WriteFile(filepath.Join(s.stateDir, ".order", strconv.Itoa(os.Getpid())), nil, 0o600)
	s.events = spec.Events
	if n := len(prior); len(spec.Calls) > 0 {
		s.events = spec.Calls[min(n, len(spec.Calls)-1)]
	}
	if spec.Grandchild {
		c := exec.Command(os.Args[0])
		c.Env = []string{"LD_OPENCODE_FAKE=grandchild"}
		if c.Start() == nil {
			s.st.GrandchildPID = c.Process.Pid
		}
	}
	s.write()
	if spec.ExitStderr != "" {
		fmt.Fprint(os.Stderr, spec.ExitStderr)
		return 1
	}
	term := make(chan os.Signal, 1)
	signal.Notify(term, syscall.SIGTERM)
	go func() {
		for range term {
			s.mu.Lock()
			s.st.Terminated = true
			s.mu.Unlock()
			s.write()
			if !spec.IgnoreTerm {
				os.Exit(0)
			}
		}
	}()
	port := ""
	for i, a := range os.Args {
		if a == "--port" && i+1 < len(os.Args) {
			port = os.Args[i+1]
		}
	}
	if port == "" {
		return 2
	}
	srv := &http.Server{Addr: "127.0.0.1:" + port, Handler: s}
	_ = srv.ListenAndServe()
	return 1
}

func (s *fakeServer) write() {
	s.mu.Lock()
	b, _ := json.Marshal(s.st)
	s.mu.Unlock()
	p := filepath.Join(s.stateDir, strconv.Itoa(os.Getpid())+".json")
	_ = os.WriteFile(p+".tmp", b, 0o600)
	_ = os.Rename(p+".tmp", p)
}

func (s *fakeServer) record(f func(st *fakeState)) {
	s.mu.Lock()
	f(&s.st)
	s.mu.Unlock()
	s.write()
}

func (s *fakeServer) authorized(r *http.Request) bool {
	u, p, ok := r.BasicAuth()
	return ok && s.password != "" && u == "opencode" && p == s.password
}

func (s *fakeServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if !s.authorized(r) {
		s.record(func(st *fakeState) { st.AuthFailures++ })
		http.Error(w, "unauthorized", http.StatusUnauthorized)
		return
	}
	s.record(func(st *fakeState) { st.Requests = append(st.Requests, r.Method+" "+r.URL.Path) })
	body, _ := io.ReadAll(r.Body)
	var parsed map[string]any
	_ = json.Unmarshal(body, &parsed)
	path := r.URL.Path
	switch {
	case r.Method == "GET" && path == "/global/health":
		writeJSON(w, 200, map[string]any{"healthy": !s.spec.Unhealthy, "version": "fake"})
	case r.Method == "GET" && path == "/event":
		s.serveEvents(w, r)
	case r.Method == "POST" && path == "/session":
		s.record(func(st *fakeState) { st.Sessions = append(st.Sessions, parsed) })
		writeJSON(w, 200, map[string]any{"id": "ses"})
	case r.Method == "POST" && path == "/mcp":
		s.record(func(st *fakeState) { st.MCP = append(st.MCP, parsed) })
		name, _ := parsed["name"].(string)
		status := s.spec.MCPStatus
		if status == "" {
			status = "connected"
			s.connectBridge(parsed)
		}
		writeJSON(w, 200, map[string]any{name: map[string]any{"status": status}})
	case r.Method == "POST" && strings.HasPrefix(path, "/mcp/") && strings.HasSuffix(path, "/disconnect"):
		s.record(func(st *fakeState) {
			st.Disconnects = append(st.Disconnects, strings.TrimSuffix(strings.TrimPrefix(path, "/mcp/"), "/disconnect"))
		})
		writeJSON(w, 200, true)
	case r.Method == "POST" && strings.HasSuffix(path, "/message"):
		s.record(func(st *fakeState) { st.Messages = append(st.Messages, parsed) })
		s.callBridge()
		s.once.Do(func() { close(s.posted) })
		if s.spec.HoldMessage {
			select {
			case <-s.released:
			case <-r.Context().Done():
			}
		}
		status := s.spec.Status
		if status == 0 {
			status = 200
		}
		writeJSON(w, status, map[string]any{})
	case r.Method == "POST" && strings.HasSuffix(path, "/abort"):
		s.record(func(st *fakeState) { st.Aborts = append(st.Aborts, path) })
		s.relOnce.Do(func() { close(s.released) })
		writeJSON(w, 200, true)
	default:
		http.NotFound(w, r)
	}
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

// serveEvents streams the scripted events once the message is posted, then
// keeps the stream open, as OpenCode does, until the client goes away.
func (s *fakeServer) serveEvents(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.WriteHeader(200)
	fl, _ := w.(http.Flusher)
	fmt.Fprint(w, "data: {\"type\":\"server.connected\",\"properties\":{}}\n\n")
	fl.Flush()
	select {
	case <-s.posted:
	case <-r.Context().Done():
		return
	}
	for _, e := range s.events {
		b, _ := json.Marshal(e)
		b = s.bridgeNames(b)
		fmt.Fprintf(w, "data: %s\n\n", b)
		fl.Flush()
	}
	<-r.Context().Done()
}

var bridgeRef = regexp.MustCompile(`\{\{bridge:([^}]*)\}\}`)

func (s *fakeServer) bridgeNames(b []byte) []byte {
	return bridgeRef.ReplaceAllFunc(b, func(m []byte) []byte {
		key := bridgeRef.FindSubmatch(m)[1]
		return []byte(sanitizeName(s.bridge.name) + "_" + sanitizeName(string(key)))
	})
}

var unsafeName = regexp.MustCompile(`[^a-zA-Z0-9_-]`)

func sanitizeName(v string) string { return unsafeName.ReplaceAllString(v, "_") }

// connectBridge does what OpenCode's MCP client does on POST /mcp:
// initialize, open the GET stream, then list the tools.
func (s *fakeServer) connectBridge(body map[string]any) {
	cfg, _ := body["config"].(map[string]any)
	url, _ := cfg["url"].(string)
	headers, _ := cfg["headers"].(map[string]any)
	auth, _ := headers["Authorization"].(string)
	s.bridge.name, _ = body["name"].(string)
	s.bridge.url, s.bridge.token = url, auth
	if _, err := s.rpc("initialize", map[string]any{"protocolVersion": "2025-06-18", "capabilities": map[string]any{}, "clientInfo": map[string]any{"name": "fake"}}); err != nil {
		return
	}
	_, _ = s.rpc("notifications/initialized", nil)
	if req, err := http.NewRequest("GET", url, nil); err == nil {
		req.Header.Set("Authorization", auth)
		req.Header.Set("Accept", "text/event-stream")
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		if resp, err := http.DefaultClient.Do(req.WithContext(ctx)); err == nil {
			line, _ := bufio.NewReader(resp.Body).ReadString('\n')
			s.record(func(st *fakeState) { st.BridgeStream = resp.Header.Get("Content-Type") + "|" + line })
			_ = resp.Body.Close()
		}
	}
	if res, err := s.rpc("tools/list", map[string]any{}); err == nil {
		tools, _ := res["tools"].([]any)
		s.record(func(st *fakeState) {
			for _, t := range tools {
				m, _ := t.(map[string]any)
				st.BridgeTools = append(st.BridgeTools, m)
			}
		})
	}
}

func (s *fakeServer) callBridge() {
	for _, c := range s.spec.BridgeCalls {
		res, err := s.rpc("tools/call", map[string]any{"name": c.Name, "arguments": c.Arguments})
		if err != nil {
			res = map[string]any{"rpcError": err.Error()}
		}
		s.record(func(st *fakeState) { st.BridgeResults = append(st.BridgeResults, res) })
	}
}

var rpcID int

// rpc posts one JSON-RPC message to the bridge. A notification returns nil.
func (s *fakeServer) rpc(method string, params any) (map[string]any, error) {
	msg := map[string]any{"jsonrpc": "2.0", "method": method}
	if !strings.HasPrefix(method, "notifications/") {
		rpcID++
		msg["id"] = rpcID
	}
	if params != nil {
		msg["params"] = params
	}
	b, _ := json.Marshal(msg)
	req, err := http.NewRequest("POST", s.bridge.url, bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", s.bridge.token)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if msg["id"] == nil {
		return nil, nil
	}
	var out struct {
		Result map[string]any `json:"result"`
		Error  map[string]any `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("status %d: %w", resp.StatusCode, err)
	}
	if out.Error != nil {
		return nil, fmt.Errorf("rpc error %v: %v", out.Error["code"], out.Error["message"])
	}
	return out.Result, nil
}

// --- LaunchDarkly and telemetry fakes (test side) ---

type trackCall struct {
	Event string
	Data  map[string]any
}

// fakeLD serves variations by key and records track calls.
type fakeLD struct {
	mu         sync.Mutex
	variations map[string]any
	tracks     []trackCall
}

func (f *fakeLD) JSONVariation(key string, _ ldcontext.Context, def ldvalue.Value) (ldvalue.Value, error) {
	v, ok := f.variations[key]
	if !ok {
		return def, nil
	}
	b, _ := json.Marshal(v)
	return ldvalue.Parse(b), nil
}

func (f *fakeLD) TrackMetric(event string, _ ldcontext.Context, _ float64, data ldvalue.Value) error {
	var d map[string]any
	_ = json.Unmarshal([]byte(data.JSONString()), &d)
	f.mu.Lock()
	defer f.mu.Unlock()
	f.tracks = append(f.tracks, trackCall{event, d})
	return nil
}

func (f *fakeLD) Flush()       {}
func (f *fakeLD) Close() error { return nil }

func (f *fakeLD) events(name string) []trackCall {
	f.mu.Lock()
	defer f.mu.Unlock()
	var out []trackCall
	for _, t := range f.tracks {
		if t.Event == name {
			out = append(out, t)
		}
	}
	return out
}

type env struct {
	ld  *fakeLD
	exp *tracetest.InMemoryExporter
}

// setup installs a fresh fake LD client and an in-memory span exporter, and
// shortens the handler's timings.
func setup(t *testing.T, variations map[string]any) *env {
	t.Helper()
	_ = aiserver.Shutdown(bg)
	ld := &fakeLD{variations: variations}
	if _, err := aiserver.InitWithClient(bg, ld, aiserver.InitOptions{OTLPEndpoint: "http://127.0.0.1:1/v1/traces"}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = aiserver.Shutdown(bg) })
	exp := tracetest.NewInMemoryExporter()
	otel.SetTracerProvider(sdktrace.NewTracerProvider(
		sdktrace.WithSpanProcessor(aiserver.ConversationIDSpanProcessor{}), sdktrace.WithSyncer(exp)))
	short(t, &startupTimeout, 10*time.Second)
	short(t, &shutdownGrace, 2*time.Second)
	short(t, &drainFallback, 300*time.Millisecond)
	return &env{ld: ld, exp: exp}
}

func short(t *testing.T, v *time.Duration, d time.Duration) {
	old := *v
	*v = d
	t.Cleanup(func() { *v = old })
}

func (e *env) spans(name string) []sdktrace.ReadOnlySpan {
	var out []sdktrace.ReadOnlySpan
	for _, s := range e.exp.GetSpans().Snapshots() {
		if s.Name() == name || strings.HasPrefix(s.Name(), name+" ") {
			out = append(out, s)
		}
	}
	return out
}

func attrs(s sdktrace.ReadOnlySpan) map[string]string {
	m := map[string]string{}
	for _, a := range s.Attributes() {
		m[string(a.Key)] = a.Value.Emit()
	}
	return m
}

func cfg(extra func(*aiserver.AiConfig)) aiserver.AiConfig {
	c := aiserver.AiConfig{Model: aiserver.Model{Name: "test-model"}, Instructions: "be brief"}
	c.Provider.Name = "OpenAI"
	if extra != nil {
		extra(&c)
	}
	return c
}

func withTools(keys ...string) func(*aiserver.AiConfig) {
	return func(c *aiserver.AiConfig) {
		c.Tools = map[string]aiserver.Tool{}
		for _, k := range keys {
			c.Tools[k] = aiserver.Tool{Name: k, Description: k + " tool", Parameters: map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}}}
		}
	}
}

// agentVariation is an enabled agent-mode flag variation.
func agentVariation(instructions string, extra map[string]any) map[string]any {
	v := map[string]any{
		"model": map[string]any{"name": "test-model"}, "provider": map[string]any{"name": "OpenAI"},
		"instructions": instructions,
		"_ldMeta":      map[string]any{"enabled": true, "variationKey": "v1", "version": 2, "mode": "agent"},
	}
	for k, x := range extra {
		v[k] = x
	}
	return v
}

// permission is the session's permission ruleset as "action permission" strings.
func permission(t *testing.T, s fakeState) []string {
	t.Helper()
	if len(s.Sessions) != 1 {
		t.Fatalf("want one session, got %d", len(s.Sessions))
	}
	rules, _ := s.Sessions[0]["permission"].([]any)
	var out []string
	for _, r := range rules {
		m, _ := r.(map[string]any)
		if m["pattern"] != "*" {
			t.Errorf("rule pattern = %v, want *", m["pattern"])
		}
		out = append(out, fmt.Sprint(m["action"], " ", m["permission"]))
	}
	return out
}

func invoke(t *testing.T, h *aiserver.ProviderHandler, req aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
	t.Helper()
	if h == nil {
		t.Fatal("no handler")
	}
	ctx, cancel := context.WithTimeout(bg, 20*time.Second)
	defer cancel()
	return h.Invoke(ctx, req)
}
