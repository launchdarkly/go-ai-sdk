package opencodeagents

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/launchdarkly/go-ai-sdk/internal/procgroup"
)

const (
	serverUsername = "opencode"
	probeInterval  = 50 * time.Millisecond
	stderrTail     = 2048
)

// server is one running `opencode serve` child.
type server struct {
	cmd      *exec.Cmd
	base     string // http://127.0.0.1:<port>
	password string
	stderr   *os.File
	exited   chan struct{} // closed once the child has been reaped
	tr       *http.Transport
}

// startServer starts the child and returns once /global/health reports it
// healthy. On any error the child is already stopped.
func startServer(ctx context.Context, opts Options) (*server, error) {
	dir, err := filepath.Abs(opts.WorkDir)
	if err != nil {
		return nil, fmt.Errorf("opencodeagents: working directory: %w", err)
	}
	if st, err := os.Stat(dir); err != nil || !st.IsDir() {
		return nil, fmt.Errorf("opencodeagents: working directory %q does not exist", opts.WorkDir)
	}
	config, err := filepath.Abs(opts.ConfigPath)
	if err != nil {
		return nil, fmt.Errorf("opencodeagents: config file: %w", err)
	}
	if st, err := os.Stat(config); err != nil || !st.Mode().IsRegular() {
		return nil, fmt.Errorf("opencodeagents: config file %q does not exist", opts.ConfigPath)
	}
	password, err := secret()
	if err != nil {
		return nil, err
	}
	port, err := freePort()
	if err != nil {
		return nil, err
	}
	stderr, err := os.CreateTemp("", "opencode-stderr-*") // a pipe nobody reads would block the child
	if err != nil {
		return nil, fmt.Errorf("opencodeagents: stderr file: %w", err)
	}
	command := opts.Command
	if len(command) == 0 {
		command = []string{"opencode"}
	}
	args := append(append([]string{}, command[1:]...), "serve", "--port", strconv.Itoa(port), "--hostname", "127.0.0.1")
	cmd := exec.Command(command[0], args...)
	cmd.Dir = dir
	cmd.Env = childEnv(opts.Env, config, password)
	cmd.Stderr = stderr
	procgroup.Isolate(cmd)
	if err := cmd.Start(); err != nil {
		closeTemp(stderr)
		return nil, fmt.Errorf("opencodeagents: opencode serve failed to start: %w", err)
	}
	s := &server{cmd: cmd, base: "http://127.0.0.1:" + strconv.Itoa(port), password: password, stderr: stderr, exited: make(chan struct{}),
		tr: http.DefaultTransport.(*http.Transport).Clone()}
	go func() { _ = cmd.Wait(); close(s.exited) }()
	if err := s.waitHealthy(ctx); err != nil {
		s.stop()
		return nil, err
	}
	return s, nil
}

// childEnv is env with the three OPENCODE_* values replaced. It is always a
// fresh slice, never nil.
func childEnv(env []string, config, password string) []string {
	out := make([]string, 0, len(env)+3)
	for _, kv := range env {
		k, _, _ := strings.Cut(kv, "=")
		if k != "OPENCODE_CONFIG" && k != "OPENCODE_SERVER_USERNAME" && k != "OPENCODE_SERVER_PASSWORD" {
			out = append(out, kv)
		}
	}
	return append(out, "OPENCODE_CONFIG="+config, "OPENCODE_SERVER_USERNAME="+serverUsername, "OPENCODE_SERVER_PASSWORD="+password)
}

func (s *server) waitHealthy(ctx context.Context) error {
	client := s.client()
	deadline := time.Now().Add(startupTimeout)
	for time.Now().Before(deadline) {
		select {
		case <-s.exited:
			return s.startupError(fmt.Sprintf("opencode serve exited during startup (code %d)", s.cmd.ProcessState.ExitCode()))
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		if s.healthy(ctx, client) {
			return nil
		}
		select {
		case <-time.After(probeInterval):
		case <-s.exited:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return s.startupError("opencode serve did not become healthy in time")
}

func (s *server) healthy(ctx context.Context, client *http.Client) bool {
	ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, "GET", s.base+"/global/health", nil)
	resp, err := client.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	var body struct {
		Healthy bool `json:"healthy"`
	}
	return resp.StatusCode == 200 && json.NewDecoder(resp.Body).Decode(&body) == nil && body.Healthy
}

func (s *server) startupError(msg string) error {
	if tail := s.stderrTail(); tail != "" {
		return fmt.Errorf("opencodeagents: %s\n%s", msg, tail)
	}
	return errors.New("opencodeagents: " + msg)
}

func (s *server) stderrTail() string {
	size, err := s.stderr.Seek(0, io.SeekEnd)
	if err != nil {
		return ""
	}
	off := max(0, size-stderrTail)
	b := make([]byte, size-off)
	if _, err := s.stderr.ReadAt(b, off); err != nil && !errors.Is(err, io.EOF) {
		return ""
	}
	return strings.TrimSpace(strings.ToValidUTF8(string(b), "�"))
}

// stop sends the process group SIGTERM, then SIGKILL after shutdownGrace,
// and waits for the child to be reaped. Anything left in the group is then
// killed.
func (s *server) stop() {
	select {
	case <-s.exited:
	default:
		_ = procgroup.Terminate(s.cmd)
		select {
		case <-s.exited:
		case <-time.After(shutdownGrace):
			_ = procgroup.Kill(s.cmd)
			<-s.exited
		}
	}
	_ = procgroup.Kill(s.cmd) // the group can outlive its leader
	s.tr.CloseIdleConnections()
	closeTemp(s.stderr)
}

// client sends basic auth on every request.
func (s *server) client() *http.Client {
	return &http.Client{Transport: basicAuth{password: s.password, next: s.tr}}
}

type basicAuth struct {
	password string
	next     http.RoundTripper
}

func (b basicAuth) RoundTrip(r *http.Request) (*http.Response, error) {
	r = r.Clone(r.Context())
	r.SetBasicAuth(serverUsername, b.password)
	return b.next.RoundTrip(r)
}

func secret() (string, error) {
	b := make([]byte, 32)
	if _, err := rand.Read(b); err != nil {
		return "", fmt.Errorf("opencodeagents: generate password: %w", err)
	}
	return base64.RawURLEncoding.EncodeToString(b), nil
}

// freePort asks the kernel for a loopback port. Another process can take it
// before the child binds it; the health check then fails the call.
func freePort() (int, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, fmt.Errorf("opencodeagents: free port: %w", err)
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port, nil
}

func closeTemp(f *os.File) {
	_ = f.Close()
	_ = os.Remove(f.Name())
}
