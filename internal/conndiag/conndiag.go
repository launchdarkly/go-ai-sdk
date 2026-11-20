// Package conndiag records the life of each client connection a test's
// *http.Client opens, so a failing test can say who closed a connection that
// a response body was still reading. Tests only; it changes no behaviour.
package conndiag

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptrace"
	"runtime/debug"
	"strings"
	"sync"
	"testing"
)

// Recorder holds the connections and requests seen by one client.
type Recorder struct {
	next  http.RoundTripper
	mu    sync.Mutex
	conns []*Conn
	reqs  []*request
}

type request struct {
	n      int
	ctx    context.Context
	conn   *Conn
	reused bool
	idle   bool
}

// Conn is a recorded client connection.
type Conn struct {
	net.Conn
	rec        *Recorder
	id         int
	once       sync.Once
	closeStack string
	failures   []string
}

// Install wraps c's transport, which must be an *http.Transport, and logs
// the recorder's report when t fails. Register it after the server's Close
// cleanup so the report is written before the server shuts down.
func Install(t testing.TB, c *http.Client) *Recorder {
	t.Helper()
	tr, ok := c.Transport.(*http.Transport)
	if !ok {
		t.Fatalf("conndiag: transport is %T, want *http.Transport", c.Transport)
	}
	r := &Recorder{next: tr}
	dial := tr.DialContext
	if dial == nil {
		dial = (&net.Dialer{}).DialContext
	}
	tr.DialContext = func(ctx context.Context, network, addr string) (net.Conn, error) {
		nc, err := dial(ctx, network, addr)
		if err != nil {
			return nil, err
		}
		r.mu.Lock()
		defer r.mu.Unlock()
		cn := &Conn{Conn: nc, rec: r, id: len(r.conns) + 1}
		r.conns = append(r.conns, cn)
		return cn, nil
	}
	c.Transport = r
	t.Cleanup(func() {
		if t.Failed() {
			t.Log(r.Report())
		}
	})
	return r
}

// RoundTrip records the request's context and connection, then sends it.
func (r *Recorder) RoundTrip(req *http.Request) (*http.Response, error) {
	r.mu.Lock()
	rq := &request{n: len(r.reqs) + 1, ctx: req.Context()}
	r.reqs = append(r.reqs, rq)
	r.mu.Unlock()
	trace := &httptrace.ClientTrace{GotConn: func(info httptrace.GotConnInfo) {
		nc := info.Conn
		if tc, ok := nc.(*tls.Conn); ok {
			nc = tc.NetConn()
		}
		cn, _ := nc.(*Conn)
		r.mu.Lock()
		rq.conn, rq.reused, rq.idle = cn, info.Reused, info.WasIdle
		r.mu.Unlock()
	}}
	return r.next.RoundTrip(req.WithContext(httptrace.WithClientTrace(req.Context(), trace)))
}

// CloseIdleConnections lets httptest.Server.Close reach the transport.
func (r *Recorder) CloseIdleConnections() {
	r.next.(interface{ CloseIdleConnections() }).CloseIdleConnections()
}

// Conns returns the connections dialled so far.
func (r *Recorder) Conns() []*Conn {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]*Conn(nil), r.conns...)
}

// Close records the first closer's stack.
func (c *Conn) Close() error {
	c.once.Do(func() {
		stack := string(debug.Stack())
		c.rec.mu.Lock()
		c.closeStack = stack
		c.rec.mu.Unlock()
	})
	return c.Conn.Close()
}

// Read records every read that fails on a closed connection, with each
// request's context state at that moment.
func (c *Conn) Read(p []byte) (int, error) {
	n, err := c.Conn.Read(p)
	if errors.Is(err, net.ErrClosed) {
		c.rec.mu.Lock()
		c.failures = append(c.failures, fmt.Sprintf("read: %v; requests then: %s", err, c.rec.requestsLocked()))
		c.rec.mu.Unlock()
	}
	return n, err
}

// Report describes every request and connection.
func (r *Recorder) Report() string {
	r.mu.Lock()
	defer r.mu.Unlock()
	var b strings.Builder
	fmt.Fprintf(&b, "conndiag: %d requests: %s\n", len(r.reqs), r.requestsLocked())
	for _, c := range r.conns {
		served := 0
		for _, rq := range r.reqs {
			if rq.conn == c {
				served++
			}
		}
		fmt.Fprintf(&b, "conn %d (%s->%s): served %d requests\n", c.id, c.LocalAddr(), c.RemoteAddr(), served)
		for _, f := range c.failures {
			fmt.Fprintf(&b, "  %s\n", f)
		}
		if c.closeStack == "" {
			b.WriteString("  not closed\n")
		} else {
			fmt.Fprintf(&b, "  closed by:\n%s\n", c.closeStack)
		}
	}
	return b.String()
}

func (r *Recorder) requestsLocked() string {
	var parts []string
	for _, rq := range r.reqs {
		id := 0
		if rq.conn != nil {
			id = rq.conn.id
		}
		parts = append(parts, fmt.Sprintf("#%d conn=%d reused=%t wasIdle=%t ctxErr=%v cause=%v",
			rq.n, id, rq.reused, rq.idle, rq.ctx.Err(), context.Cause(rq.ctx)))
	}
	return strings.Join(parts, "; ")
}
