package conndiag

import (
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"strings"
	"testing"
)

// closeMidBody closes the connection under a response body that is still
// being read, the way the flake looks from the client side.
func closeMidBody(c *Conn) { _ = c.Close() }

func TestRecordsWhoClosedAConnMidBody(t *testing.T) {
	release := make(chan struct{})
	srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, "first")
		w.(http.Flusher).Flush()
		<-release
	}))
	defer srv.Close()
	defer close(release)
	c := srv.Client()
	rec := Install(t, c)

	resp, err := c.Get(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	conns := rec.Conns()
	if len(conns) != 1 {
		t.Fatalf("conns = %d, want 1", len(conns))
	}
	closeMidBody(conns[0])
	if _, err := io.ReadAll(resp.Body); err == nil || !strings.Contains(err.Error(), "use of closed network connection") {
		t.Fatalf("read err = %v, want use of closed network connection", err)
	}

	got := rec.Report()
	for _, want := range []string{
		"conndiag: 1 requests: #1 conn=1 reused=false",
		"served 1 requests",
		"read: ", "use of closed network connection; requests then: #1 conn=1",
		"closed by:", "conndiag.closeMidBody",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("report lacks %q:\n%s", want, got)
		}
	}
}

// TestLogsReportWhenTestFails re-runs itself in a child process where the
// test fails on a read from a closed, reused connection, the way the flake
// fails, and checks the child's output has the report.
func TestLogsReportWhenTestFails(t *testing.T) {
	if os.Getenv("CONNDIAG_FAILING_CHILD") == "1" {
		failOnClosedReusedConn(t)
		return
	}
	cmd := exec.Command(os.Args[0], "-test.run=^TestLogsReportWhenTestFails$", "-test.count=1")
	cmd.Env = append(os.Environ(), "CONNDIAG_FAILING_CHILD=1")
	out, err := cmd.CombinedOutput()
	var exit *exec.ExitError
	if !errors.As(err, &exit) {
		t.Fatalf("child err = %v, want a test failure; output:\n%s", err, out)
	}
	for _, want := range []string{
		"conndiag: 2 requests: #1 conn=1 reused=false",
		"#2 conn=1 reused=true wasIdle=true ctxErr=<nil> cause=<nil>",
		"served 2 requests",
		"use of closed network connection; requests then: #1 conn=1",
		"closed by:", "conndiag.closeMidBody",
	} {
		if !strings.Contains(string(out), want) {
			t.Errorf("child output lacks %q", want)
		}
	}
	if t.Failed() {
		t.Logf("child output:\n%s", out)
	}
}

// failOnClosedReusedConn reads one whole response, then fails reading the
// second response on the same connection after it is closed mid-body.
func failOnClosedReusedConn(t *testing.T) {
	release := make(chan struct{})
	srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, "first")
		if r.URL.Path == "/second" {
			w.(http.Flusher).Flush()
			<-release
		}
	}))
	t.Cleanup(srv.Close)
	t.Cleanup(func() { close(release) })
	c := srv.Client()
	rec := Install(t, c)

	resp, err := c.Get(srv.URL + "/first")
	if err != nil {
		t.Fatal(err)
	}
	_, _ = io.ReadAll(resp.Body)
	_ = resp.Body.Close()
	resp, err = c.Get(srv.URL + "/second")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	closeMidBody(rec.Conns()[0])
	if _, err := io.ReadAll(resp.Body); err != nil {
		t.Errorf("read: %v", err)
	}
}
