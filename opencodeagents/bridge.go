package opencodeagents

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"regexp"
	"sync"
	"time"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

// OpenCode has no route that takes a tool definition with a message, so a
// config ToolFunc is served from this per-call loopback MCP server
// (streamable HTTP, JSON-RPC), attached to the child with POST /mcp.
//
// Each request gets exactly one JSON response; the bridge pushes nothing.
// OpenCode's MCP client still opens the optional GET stream after
// initialize and marks the server failed unless it is a live
// text/event-stream (a 405 is not accepted), so GET writes ": connected"
// at once and a keepalive comment until the bridge stops.

const (
	mcpProtocolVersion = "2025-06-18"
	mcpServerName      = "launchdarkly-go-ai-opencode-agents"
)

type bridgeTool struct {
	description string
	parameters  map[string]any
	fn          aiserver.ToolFunc
}

// bridge is a per-call loopback MCP server for the config's ToolFunc tools.
type bridge struct {
	token string
	tools map[string]bridgeTool

	ctx  context.Context // tool calls run under the call's context
	srv  *http.Server
	done chan struct{}
	once sync.Once
}

func newBridge() *bridge {
	tok, _ := secret() // crypto/rand does not fail on supported platforms
	return &bridge{token: tok, tools: map[string]bridgeTool{}, done: make(chan struct{})}
}

// register serves fn under key, with the config's description and schema.
func (b *bridge) register(key, description string, parameters map[string]any, fn aiserver.ToolFunc) {
	b.tools[key] = bridgeTool{description, parameters, fn}
}

// start binds a loopback port and returns the bridge's base URL. Tool calls
// run under ctx.
func (b *bridge) start(ctx context.Context) (string, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return "", fmt.Errorf("opencodeagents: tool bridge: %w", err)
	}
	b.ctx = ctx
	b.srv = &http.Server{Handler: b, ReadHeaderTimeout: httpTimeout}
	go func() { _ = b.srv.Serve(l) }()
	return "http://" + l.Addr().String(), nil
}

// stop closes the listener and ends open event streams.
func (b *bridge) stop() {
	b.once.Do(func() {
		close(b.done)
		if b.srv != nil {
			_ = b.srv.Close()
		}
	})
}

func (b *bridge) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	want := "Bearer " + b.token
	if subtle.ConstantTimeCompare([]byte(r.Header.Get("Authorization")), []byte(want)) != 1 {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
		return
	}
	switch r.Method {
	case "GET":
		b.stream(w, r)
	case "POST":
		var msg map[string]any
		raw, _ := io.ReadAll(io.LimitReader(r.Body, 16<<20))
		_ = json.Unmarshal(raw, &msg)
		resp := b.dispatch(msg)
		if resp == nil {
			w.WriteHeader(http.StatusAccepted)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

func (b *bridge) stream(w http.ResponseWriter, r *http.Request) {
	fl, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(200)
	if _, err := io.WriteString(w, ": connected\n\n"); err != nil {
		return
	}
	fl.Flush()
	t := time.NewTicker(keepaliveInterval)
	defer t.Stop()
	for {
		select {
		case <-b.done:
			return
		case <-r.Context().Done():
			return
		case <-t.C:
			if _, err := io.WriteString(w, ": keepalive\n\n"); err != nil {
				return
			}
			fl.Flush()
		}
	}
}

func rpcResult(id, result any) map[string]any {
	return map[string]any{"jsonrpc": "2.0", "id": id, "result": result}
}

func rpcError(id any, code int, msg string) map[string]any {
	return map[string]any{"jsonrpc": "2.0", "id": id, "error": map[string]any{"code": code, "message": msg}}
}

// dispatch answers one JSON-RPC message; a notification gets nil.
func (b *bridge) dispatch(msg map[string]any) map[string]any {
	method, _ := msg["method"].(string)
	id, hasID := msg["id"]
	if method == "initialize" {
		return rpcResult(id, map[string]any{"protocolVersion": mcpProtocolVersion, "capabilities": map[string]any{"tools": map[string]any{}},
			"serverInfo": map[string]any{"name": mcpServerName, "version": PackageVersion}})
	}
	if method == "" || !hasID || id == nil || len(method) > 14 && method[:14] == "notifications/" {
		return nil
	}
	switch method {
	case "tools/list":
		tools := make([]map[string]any, 0, len(b.tools))
		for name, t := range b.tools {
			schema := t.parameters
			if len(schema) == 0 {
				schema = map[string]any{"type": "object", "properties": map[string]any{}}
			}
			tools = append(tools, map[string]any{"name": name, "description": t.description, "inputSchema": schema})
		}
		return rpcResult(id, map[string]any{"tools": tools})
	case "tools/call":
		params, _ := msg["params"].(map[string]any)
		name, _ := params["name"].(string)
		t, ok := b.tools[name]
		if !ok {
			return rpcError(id, -32602, fmt.Sprintf("Unknown tool: %q", name))
		}
		args, _ := params["arguments"].(map[string]any)
		if args == nil {
			args = map[string]any{}
		}
		text, failed := b.call(t.fn, args)
		return rpcResult(id, map[string]any{"content": []any{map[string]any{"type": "text", "text": text}}, "isError": failed})
	}
	return rpcError(id, -32601, fmt.Sprintf("Unknown method: %q", method))
}

// call runs a tool; an error or a panic becomes an MCP tool error.
func (b *bridge) call(fn aiserver.ToolFunc, args map[string]any) (text string, failed bool) {
	defer func() {
		if p := recover(); p != nil {
			text, failed = fmt.Sprint(p), true
		}
	}()
	ctx := b.ctx
	if ctx == nil {
		ctx = context.Background()
	}
	res, err := fn(ctx, args)
	if err != nil {
		return err.Error(), true
	}
	return aiserver.ToolResultString(res), false
}

var unsafeToolChars = regexp.MustCompile(`[^a-zA-Z0-9_-]`)

// bridgeToolName is the name OpenCode reports for one bridged tool: its own
// MCP naming, sanitize(server) + "_" + sanitize(tool).
func bridgeToolName(server, key string) string {
	return unsafeToolChars.ReplaceAllString(server, "_") + "_" + unsafeToolChars.ReplaceAllString(key, "_")
}
