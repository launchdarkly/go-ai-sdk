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

// failingTransport fails the test if InitEvaluations touches the network.
type failingTransport struct{ t *testing.T }

func (f failingTransport) RoundTrip(*http.Request) (*http.Response, error) {
	f.t.Fatal("InitEvaluations must not perform network I/O")
	return nil, errors.New("unreachable")
}

func TestInitCredentials(t *testing.T) {
	cases := []struct {
		name                         string
		envProject, envToken, envSDK string
		optProject, optToken, optSDK string
		byoc                         bool
		wantErr                      string
		wantProject                  string
		wantToken, wantKey           string
	}{
		{name: "env only", envProject: "env-proj", envToken: "env-tok", envSDK: "env-sdk", wantProject: "env-proj", wantToken: "env-tok", wantKey: "env-sdk"},
		{name: "explicit wins", envProject: "env-proj", envToken: "env-tok", envSDK: "env-sdk", optProject: "opt-proj", optToken: "opt-tok", optSDK: "opt-sdk",
			wantProject: "opt-proj", wantToken: "opt-tok", wantKey: "opt-sdk"},
		{name: "padded explicit project is trimmed", envProject: "env-proj", optProject: " opt-proj ", envToken: "tok", envSDK: "sdk", wantProject: "opt-proj", wantToken: "tok", wantKey: "sdk"},
		{name: "blank explicit project falls back to env", envProject: "env-proj", optProject: "  ", envToken: "tok", envSDK: "sdk", wantProject: "env-proj", wantToken: "tok", wantKey: "sdk"},
		{name: "missing all names the project first", wantErr: "LD_PROJECT_KEY"},
		{name: "missing project", envToken: "tok", envSDK: "sdk", wantErr: "LD_PROJECT_KEY"},
		{name: "blank project env", envProject: " \t", envToken: "tok", envSDK: "sdk", wantErr: "LD_PROJECT_KEY"},
		{name: "only the project names the token", envProject: "proj", wantErr: "LD_API_TOKEN"},
		{name: "blank token env", envProject: "proj", envToken: "   ", envSDK: "sdk", wantErr: "LD_API_TOKEN"},
		{name: "missing SDK key", envProject: "proj", envToken: "tok", wantErr: "LD_SDK_KEY"},
		{name: "blank SDK key env", envProject: "proj", envToken: "tok", envSDK: " \t", wantErr: "LD_SDK_KEY"},
		{name: "bring your own client", envProject: "proj", envToken: "tok", byoc: true, wantProject: "proj", wantToken: "tok", wantKey: ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			e := newEnv(t, 1)
			e.byoc = tc.byoc
			t.Setenv("LD_PROJECT_KEY", tc.envProject)
			t.Setenv("LD_API_TOKEN", tc.envToken)
			t.Setenv("LD_SDK_KEY", tc.envSDK)
			m, err := InitEvaluations(InitEvaluationsOptions{ProjectKey: tc.optProject, APIToken: tc.optToken, SDKKey: tc.optSDK, HTTPClient: &http.Client{Transport: failingTransport{t}}})
			if tc.wantErr != "" {
				if !errors.Is(err, ErrMissingCredential) || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("err = %v, want one naming %s", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if m.projectKey != tc.wantProject || m.api.token != tc.wantToken || m.SDKKey() != tc.wantKey {
				t.Fatalf("project %q token %q key %q", m.projectKey, m.api.token, m.SDKKey())
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
	// A failing InitEvaluations still reports: usage comes before every check.
	t.Setenv("LD_PROJECT_KEY", "")
	t.Setenv("LD_API_TOKEN", "")
	if _, err := InitEvaluations(InitEvaluationsOptions{}); !errors.Is(err, ErrMissingCredential) {
		t.Fatalf("err = %v, want ErrMissingCredential", err)
	}
	if len(u.helpers) != 1 || u.helpers[0] != "client.initEvaluations" {
		t.Fatalf("usage helpers after a failing InitEvaluations = %v", u.helpers)
	}
	// The helper is reported once per process.
	t.Setenv("LD_PROJECT_KEY", "proj")
	t.Setenv("LD_API_TOKEN", "tok")
	if _, err := InitEvaluations(InitEvaluationsOptions{}); err != nil {
		t.Fatal(err)
	}
	if len(u.helpers) != 1 {
		t.Fatalf("usage helpers = %v", u.helpers)
	}
}

func TestHosts(t *testing.T) {
	const def = "https://app.launchdarkly.com"
	cases := []struct {
		name             string
		env              map[string]string
		opts             InitEvaluationsOptions
		wantAPI, wantURL string
	}{
		{name: "default", wantAPI: def, wantURL: def},
		{name: "LD_BASE_URI never moves either host", env: map[string]string{"LD_BASE_URI": "https://relay.example.com"}, wantAPI: def, wantURL: def},
		{name: "env wins over the default", env: map[string]string{"LD_API_BASE_URI": "https://api.example.com", "LD_UI_BASE_URI": "https://ui.example.com"},
			wantAPI: "https://api.example.com", wantURL: "https://ui.example.com"},
		{name: "option wins over env", env: map[string]string{"LD_API_BASE_URI": "https://api.example.com", "LD_UI_BASE_URI": "https://ui.example.com"},
			opts: InitEvaluationsOptions{APIBaseURI: "https://api.opt.com", UIBaseURI: "https://ui.opt.com"}, wantAPI: "https://api.opt.com", wantURL: "https://ui.opt.com"},
		{name: "blank values fall through", env: map[string]string{"LD_API_BASE_URI": "  ", "LD_UI_BASE_URI": "https://ui.example.com"},
			opts: InitEvaluationsOptions{APIBaseURI: " ", UIBaseURI: " "}, wantAPI: def, wantURL: "https://ui.example.com"},
		{name: "trailing slash stripped", env: map[string]string{"LD_UI_BASE_URI": "https://ui.example.com/"},
			opts: InitEvaluationsOptions{APIBaseURI: "https://api.opt.com/"}, wantAPI: "https://api.opt.com", wantURL: "https://ui.example.com"},
		{name: "UI host is not derived from the API host", env: map[string]string{"LD_API_BASE_URI": "https://api.example.com"},
			wantAPI: "https://api.example.com", wantURL: def},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			for _, k := range []string{"LD_BASE_URI", "LD_API_BASE_URI", "LD_UI_BASE_URI"} {
				t.Setenv(k, tc.env[k])
			}
			e := newEnv(t, 1)
			tc.opts.ProjectKey = "proj"
			res := mustRun(t, e.moduleWith(tc.opts), baseArgs(handler("OpenAI", aiserver.ModeMessages, nil)))
			if len(e.requests()) == 0 {
				t.Fatal("no requests recorded")
			}
			for _, r := range e.requests() {
				if !strings.HasPrefix(r.URL, tc.wantAPI+"/api/v2/") {
					t.Fatalf("request to %s, want host %s", r.URL, tc.wantAPI)
				}
			}
			if want := tc.wantURL + "/projects/proj/ai/evaluations/eval-1/runs/run-1"; res.URL != want {
				t.Fatalf("url = %s, want %s", res.URL, want)
			}
		})
	}
}

// A padded project key is trimmed once at InitEvaluations and used trimmed
// in every request path and in the run link.
func TestPaddedProjectKeyIsTrimmedEverywhere(t *testing.T) {
	e := newEnv(t, 1)
	t.Setenv("LD_PROJECT_KEY", "")
	res := mustRun(t, e.moduleWith(InitEvaluationsOptions{ProjectKey: " proj "}), baseArgs(handler("OpenAI", aiserver.ModeMessages, nil)))
	reqs := e.requests()
	if len(reqs) == 0 {
		t.Fatal("no requests recorded")
	}
	for _, r := range reqs {
		if !strings.Contains(r.URL, "/api/v2/projects/proj/") || strings.Contains(r.URL, "%20") {
			t.Fatalf("request to %s, want the trimmed project key", r.URL)
		}
	}
	if res.URL != "https://app.launchdarkly.com/projects/proj/ai/evaluations/eval-1/runs/run-1" {
		t.Fatalf("url = %s", res.URL)
	}
}
