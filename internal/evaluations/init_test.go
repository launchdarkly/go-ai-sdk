package evaluations

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"testing"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/launchdarkly/go-sdk-common/v3/ldvalue"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

// failingTransport fails the test if Init touches the network.
type failingTransport struct{ t *testing.T }

func (f failingTransport) RoundTrip(*http.Request) (*http.Response, error) {
	f.t.Fatal("Init must not perform network I/O")
	return nil, errors.New("unreachable")
}

func TestInitCredentials(t *testing.T) {
	cases := []struct {
		name               string
		envToken, envSDK   string
		optToken, optSDK   string
		byoc               bool
		wantErr            string
		wantToken, wantKey string
	}{
		{name: "env only", envToken: "env-tok", envSDK: "env-sdk", wantToken: "env-tok", wantKey: "env-sdk"},
		{name: "explicit wins", envToken: "env-tok", envSDK: "env-sdk", optToken: "opt-tok", optSDK: "opt-sdk", wantToken: "opt-tok", wantKey: "opt-sdk"},
		{name: "missing token", envSDK: "sdk", wantErr: "LD_API_TOKEN"},
		{name: "missing both names the token first", wantErr: "LD_API_TOKEN"},
		{name: "blank token env", envToken: "   ", envSDK: "sdk", wantErr: "LD_API_TOKEN"},
		{name: "missing SDK key", envToken: "tok", wantErr: "LD_SDK_KEY"},
		{name: "blank SDK key env", envToken: "tok", envSDK: " \t", wantErr: "LD_SDK_KEY"},
		{name: "bring your own client", envToken: "tok", byoc: true, wantToken: "tok", wantKey: ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			e := newEnv(t, 1)
			e.byoc = tc.byoc
			t.Setenv("LD_API_TOKEN", tc.envToken)
			t.Setenv("LD_SDK_KEY", tc.envSDK)
			m, err := Init(Options{APIToken: tc.optToken, SDKKey: tc.optSDK, HTTPClient: &http.Client{Transport: failingTransport{t}}})
			if tc.wantErr != "" {
				if !errors.Is(err, ErrMissingCredential) || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("err = %v, want one naming %s", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if m.api.token != tc.wantToken || m.SDKKey() != tc.wantKey {
				t.Fatalf("token %q key %q", m.api.token, m.SDKKey())
			}
		})
	}
}

// usageClient records $ld:ai:sdk:usage for the helper-usage test.
type usageClient struct{ helpers []string }

func (u *usageClient) JSONVariation(_ string, _ ldcontext.Context, d ldvalue.Value) (ldvalue.Value, error) {
	return d, nil
}

func (u *usageClient) TrackMetric(event string, _ ldcontext.Context, _ float64, data ldvalue.Value) error {
	if event == "$ld:ai:sdk:usage" {
		u.helpers = append(u.helpers, data.GetByKey("helper").StringValue())
	}
	return nil
}
func (u *usageClient) Flush()       {}
func (u *usageClient) Close() error { return nil }

func TestInitReportsHelperUsage(t *testing.T) {
	u := &usageClient{}
	t.Setenv("OTEL_TRACES_EXPORTER", "none")
	if _, err := aiserver.InitWithClient(context.Background(), u, aiserver.InitOptions{}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = aiserver.Shutdown(context.Background()) })
	t.Setenv("LD_API_TOKEN", "tok")
	if _, err := Init(Options{}); err != nil {
		t.Fatal(err)
	}
	if len(u.helpers) != 1 || u.helpers[0] != "experimental.initEvaluations" {
		t.Fatalf("usage helpers = %v", u.helpers)
	}
}

func TestFixedHostsIgnoreLDBaseURI(t *testing.T) {
	e := newEnv(t, 1)
	t.Setenv("LD_BASE_URI", "https://sdk.example.com")
	t.Setenv("LD_API_BASE_URI", "https://api.example.com")
	res := mustRun(t, e.module(), baseArgs(handler("OpenAI", aiserver.ModeMessages, nil)))
	for _, r := range e.requests() {
		if !strings.HasPrefix(r.URL, "https://app.launchdarkly.com/api/v2/") {
			t.Fatalf("request to %s", r.URL)
		}
	}
	if res.URL != "https://app.launchdarkly.com/projects/proj/ai/evaluations/eval-1/runs/run-1" {
		t.Fatalf("url = %s", res.URL)
	}
}
