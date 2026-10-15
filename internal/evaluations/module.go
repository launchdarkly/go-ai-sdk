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

// HelperName is the usage ID Init reports: the one the AI SDKs' usage-ID
// registry assigns to initEvaluations, which Python reports too, so usage
// aggregates across languages.
const HelperName = "experimental.initEvaluations"

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

// Options configure Init. The management API and UI hosts are fixed
// (APIBaseURI, UIBaseURI) and have no option.
type Options struct {
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
}

// Module runs evaluations from code. Create it with Init.
type Module struct {
	api        *apiClient
	sdkKey     string
	logger     *slog.Logger
	flushBound int
}

func env(name string) string { return strings.TrimSpace(os.Getenv(name)) }

// Init resolves both credentials and returns the module. It performs no
// network I/O and fails on a missing credential, token first.
func Init(opts Options) (*Module, error) {
	aiserver.ReportUsage(HelperName, "", "")
	token := strings.TrimSpace(opts.APIToken)
	if token == "" {
		token = env("LD_API_TOKEN")
	}
	if token == "" {
		return nil, fmt.Errorf("evaluations: %w: no LaunchDarkly API access token; set LD_API_TOKEN or pass Options.APIToken", ErrMissingCredential)
	}
	sdkKey := strings.TrimSpace(opts.SDKKey)
	if sdkKey == "" {
		sdkKey = env("LD_SDK_KEY")
	}
	if sdkKey == "" {
		if _, err := getClient(); err != nil {
			return nil, fmt.Errorf("evaluations: %w: no LaunchDarkly SDK key and no initialized client to emit result events with; "+
				"results reach LaunchDarkly only as SDK events, so set LD_SDK_KEY, pass Options.SDKKey, or call aiserver.InitWithClient first", ErrMissingCredential)
		}
	}
	logger := opts.Logger
	if logger == nil {
		logger = bridge.Logger()
	}
	return &Module{api: newAPIClient(token, opts.HTTPClient, opts.RequestTimeout, opts.MaxAttempts),
		sdkKey: sdkKey, logger: logger, flushBound: defaultFlushBound}, nil
}

// SDKKey is the key the module initializes a client with; "" when it relies
// on a client the caller installed.
func (m *Module) SDKKey() string { return m.sdkKey }

// resolveClient runs before any management API request.
func (m *Module) resolveClient(ctx context.Context) (aiserver.LDClient, error) {
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
func (m *Module) Run(ctx context.Context, args RunArgs) (*EvalRunResult, error) {
	r, err := m.prepare(args)
	if err != nil {
		return nil, err
	}
	if r.client, err = m.resolveClient(ctx); err != nil {
		return nil, err
	}
	return r.execute(ctx)
}
