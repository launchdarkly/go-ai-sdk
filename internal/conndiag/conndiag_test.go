package conndiag

import (
	"io"
	"net/http"
	"net/http/httptest"
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
