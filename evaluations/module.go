package evaluations

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/bridge"
)

// helperName is the usage ID InitEvaluations reports. The TypeScript and
// Python SDKs report the same ID, so usage aggregates across languages.
const helperName = "client.initEvaluations"

// Seams replaced by tests: the core client singleton, its initialization,
// and judge resolution through flag delivery.
var (
	getClient          = aiserver.GetClient
	initClient         = aiserver.InitClient
	resolveJudgeConfig = func(ctx context.Context, key string, c ldcontext.Context) (map[string]any, error) {
		if bridge.ExtractVariationRaw == nil {
			return nil, fmt.Errorf("flag delivery is not available")
		}
		return bridge.ExtractVariationRaw(ctx, key, c)
	}
	defaultPollInterval = 2 * time.Second
	defaultPollTimeout  = 3 * time.Minute
)

const defaultConcurrency = 10

// InitEvaluationsOptions configure InitEvaluations. Each string option falls
// back to its environment variable; a blank value at either level counts as
// unset.
type InitEvaluationsOptions struct {
	// ProjectKey is the project holding the evaluations, runs, tools,
	// datasets and judges. Default: LD_PROJECT_KEY.
	ProjectKey string
	// APIToken authorizes every management API request. Default: LD_API_TOKEN.
	APIToken string
	// SDKKey initializes the client that emits result events. Default:
	// LD_SDK_KEY. May be omitted only when aiserver.InitWithClient (or
	// InitClient) has already installed a client.
	SDKKey string
	// HTTPClient sends management API requests; its Transport is the
	// request seam. Default http.DefaultClient.
	HTTPClient *http.Client
	// RequestTimeout bounds each attempt. Default 30s.
	RequestTimeout time.Duration
	// MaxAttempts bounds attempts per request, retries included. Default 4.
	MaxAttempts int
	// Logger receives warnings. Default: the SDK logger (aiserver.SetLogger).
	Logger *slog.Logger
	// APIBaseURI is the management API host. Default: LD_API_BASE_URI, then
	// https://app.launchdarkly.com. LD_BASE_URI is never used: it configures
	// flag delivery and may point at a relay proxy.
	APIBaseURI string
	// UIBaseURI is the host of the run link in the result; it carries no API
	// traffic. Default: LD_UI_BASE_URI, then https://app.launchdarkly.com.
	// It is not derived from APIBaseURI.
	UIBaseURI string
}

// EvaluationsModule runs evaluations from code. Create it with InitEvaluations.
type EvaluationsModule struct {
	api        *apiClient
	projectKey string
	uiBase     string
	sdkKey     string
	logger     *slog.Logger
	flushBound int
}

func env(name string) string { return strings.TrimSpace(os.Getenv(name)) }

// host resolves a host option: explicit, then env, then the default, with
// any trailing "/" stripped. A blank value at either level counts as unset.
func host(explicit, name string) string {
	v := option(explicit, name)
	if v == "" {
		v = defaultBaseURI
	}
	return strings.TrimRight(v, "/")
}

// option is the trimmed explicit value, else the trimmed environment value.
func option(explicit, name string) string {
	if v := strings.TrimSpace(explicit); v != "" {
		return v
	}
	return env(name)
}

// InitEvaluations resolves the project key and both credentials and returns the
// module. It performs no network I/O and fails on a missing value in this
// order: project key, token, SDK key.
func InitEvaluations(opts InitEvaluationsOptions) (*EvaluationsModule, error) {
	aiserver.ReportUsage(helperName, "", "")
	project := option(opts.ProjectKey, "LD_PROJECT_KEY")
	if project == "" {
		return nil, fmt.Errorf("evaluations: %w: no LaunchDarkly project key; set LD_PROJECT_KEY or pass InitEvaluationsOptions.ProjectKey", ErrMissingCredential)
	}
	token := option(opts.APIToken, "LD_API_TOKEN")
	if token == "" {
		return nil, fmt.Errorf("evaluations: %w: no LaunchDarkly API access token; set LD_API_TOKEN or pass InitEvaluationsOptions.APIToken", ErrMissingCredential)
	}
	sdkKey := option(opts.SDKKey, "LD_SDK_KEY")
	if sdkKey == "" {
		if _, err := getClient(); err != nil {
			return nil, fmt.Errorf("evaluations: %w: no LaunchDarkly SDK key and no initialized client to emit result events with; "+
				"results reach LaunchDarkly only as SDK events, so set LD_SDK_KEY, pass InitEvaluationsOptions.SDKKey, or call aiserver.InitWithClient first", ErrMissingCredential)
		}
	}
	logger := opts.Logger
	if logger == nil {
		logger = bridge.Logger()
	}
	return &EvaluationsModule{api: newAPIClient(host(opts.APIBaseURI, "LD_API_BASE_URI"), token, opts.HTTPClient, opts.RequestTimeout, opts.MaxAttempts),
		projectKey: project, uiBase: host(opts.UIBaseURI, "LD_UI_BASE_URI"), sdkKey: sdkKey, logger: logger, flushBound: defaultFlushBound}, nil
}

// SDKKey is the key the module initializes a client with; "" when it relies
// on a client the caller installed.
func (m *EvaluationsModule) SDKKey() string { return m.sdkKey }

// resolveClient runs before any management API request.
func (m *EvaluationsModule) resolveClient(ctx context.Context) (aiserver.LDClient, error) {
	if c, err := getClient(); err == nil {
		if m.sdkKey != "" {
			m.logger.Warn("a LaunchDarkly client is already initialized; evaluation events are sent with it and the evaluations SDK key is ignored. Both must point at the project under evaluation.")
		}
		return c, nil
	}
	if m.sdkKey == "" {
		return nil, fmt.Errorf("evaluations: %w: no SDK key and no initialized LaunchDarkly client is available to deliver result events", ErrNoClient)
	}
	c, err := initClient(ctx, aiserver.InitOptions{SDKKey: m.sdkKey})
	if err != nil {
		return nil, fmt.Errorf("evaluations: %w: %w", ErrNoClient, err)
	}
	return c, nil
}

// Run creates an evaluation and a run, generates every dataset row with the
// pool, scores each generated row against the criteria, reports every result
// as an event, and polls the run summary until LaunchDarkly has accounted
// for every row.
func (m *EvaluationsModule) Run(ctx context.Context, args RunOptions) (*EvalRunResult, error) {
	r, err := m.prepare(args)
	if err != nil {
		return nil, err
	}
	if r.client, err = m.resolveClient(ctx); err != nil {
		return nil, err
	}
	return r.execute(ctx)
}
