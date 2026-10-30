package aiserver

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/launchdarkly/go-sdk-common/v3/ldvalue"
	ld "github.com/launchdarkly/go-server-sdk/v7"
	"github.com/launchdarkly/go-server-sdk/v7/interfaces"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.26.0"

	"github.com/launchdarkly/go-ai-sdk/internal/bridge"
)

const (
	PackageName    = "github.com/launchdarkly/go-ai-sdk/aiserver"
	PackageVersion = "0.1.0" // x-release-please-version
	sdkLanguage    = "go"
)

// LDClient is the subset of *ld.LDClient the SDK needs. Any value with these
// methods can be passed to InitWithClient (bring your own client).
type LDClient interface {
	JSONVariation(key string, ctx ldcontext.Context, def ldvalue.Value) (ldvalue.Value, error)
	TrackMetric(event string, ctx ldcontext.Context, metric float64, data ldvalue.Value) error
	Flush()
	Close() error
}

// InitOptions are the documented initClient options. Go's typed struct makes
// the "unrecognized option" warning unreachable, so it is not implemented.
type InitOptions struct {
	SDKKey       string
	BaseURI      string
	StreamURI    string
	EventsURI    string
	OTLPEndpoint string
	ServiceName  string
	Environment  string
	WaitFor      time.Duration // initialization timeout; default 10s (matches JavaScript)
}

var (
	// lifecycleMu serialises InitClient, InitWithClient and Shutdown, so
	// concurrent lazy inits share one attempt. It may be held across network
	// waits; nothing on the request path takes it once a client exists.
	lifecycleMu sync.Mutex
	// stateMu guards client and tracerProvider and is held only briefly, so
	// GetClient and tracking never wait on a slow initialization.
	stateMu        sync.Mutex
	client         LDClient
	tracerProvider *sdktrace.TracerProvider

	// newLDClient is the seam tests replace: Go cannot intercept a module
	// import the way JavaScript and Python test mocks do.
	newLDClient = func(sdkKey string, cfg ld.Config, wait time.Duration) (LDClient, error) {
		c, err := ld.MakeCustomClient(sdkKey, cfg, wait)
		if c == nil {
			return nil, err
		}
		return c, err
	}
)

// loggerPtr holds the logger set by SetLogger; nil means slog.Default().
var loggerPtr atomic.Pointer[slog.Logger]

// SetLogger replaces the SDK logger, including the experimental features'.
// It is safe to call at any time; nil restores slog.Default().
func SetLogger(l *slog.Logger) { loggerPtr.Store(l) }

// sdkLogger is the current SDK logger.
func sdkLogger() *slog.Logger {
	if l := loggerPtr.Load(); l != nil {
		return l
	}
	return slog.Default()
}

// InitClient creates the singleton LaunchDarkly client and telemetry. It is
// idempotent; a failed call is not cached.
func InitClient(ctx context.Context, opts InitOptions) (LDClient, error) {
	if c := currentClient(); c != nil { // fast path: every Invoke comes through here
		flushReports(c)
		return c, nil
	}
	lifecycleMu.Lock()
	defer lifecycleMu.Unlock()
	if c := currentClient(); c != nil { // another caller finished while we waited
		flushReports(c)
		return c, nil
	}
	key := sdkKey(opts)
	if key == "" {
		return nil, errorf("LD_SDK_KEY is not set; pass InitOptions.SDKKey or call InitWithClient")
	}
	opts = withEnvDefaults(opts)
	cfg := ld.Config{}
	if opts.BaseURI != "" || opts.StreamURI != "" || opts.EventsURI != "" {
		cfg.ServiceEndpoints = interfaces.ServiceEndpoints{Polling: opts.BaseURI, Streaming: opts.StreamURI, Events: opts.EventsURI}
	}
	wait := opts.WaitFor
	if wait == 0 {
		wait = 10 * time.Second
	}
	c, err := newLDClient(key, cfg, wait)
	if err != nil {
		if c != nil {
			_ = c.Close() // a close error must not replace the init error
		}
		return nil, errorf("LaunchDarkly client failed to initialize: %w", err)
	}
	if err := setupTelemetry(ctx, key, opts); err != nil {
		sdkLogger().Warn("OpenTelemetry setup failed; continuing without tracing", "error", err)
	}
	setClient(c)
	flushReports(c)
	return c, nil
}

func currentClient() LDClient {
	stateMu.Lock()
	defer stateMu.Unlock()
	return client
}

func setClient(c LDClient) {
	stateMu.Lock()
	defer stateMu.Unlock()
	client = c
}

// InitWithClient installs a pre-initialized client (BYOC). Go has no
// overloads, so this is a separate function from InitClient.
func InitWithClient(ctx context.Context, c LDClient, opts InitOptions) (LDClient, error) {
	lifecycleMu.Lock()
	defer lifecycleMu.Unlock()
	if currentClient() == nil {
		if err := setupTelemetry(ctx, sdkKey(opts), opts); err != nil {
			sdkLogger().Warn("OpenTelemetry setup failed; continuing without tracing", "error", err)
		}
	}
	setClient(c)
	flushReports(c)
	return c, nil
}

// GetClient returns the singleton or an error if InitClient has not run.
func GetClient() (LDClient, error) {
	stateMu.Lock()
	defer stateMu.Unlock()
	if client == nil {
		return nil, fmt.Errorf("%w; call InitClient first", ErrNotInitialized)
	}
	return client, nil
}

// Shutdown flushes and closes the client and tracer provider. Idempotent.
func Shutdown(ctx context.Context) error {
	lifecycleMu.Lock()
	defer lifecycleMu.Unlock()
	stateMu.Lock()
	c, tp := client, tracerProvider
	client, tracerProvider = nil, nil
	resetReports()
	stateMu.Unlock()
	for _, hook := range bridge.ShutdownHooks() {
		runShutdownHook(ctx, hook)
	}
	var errs []error
	if tp != nil {
		errs = append(errs, tp.Shutdown(ctx))
	}
	if c != nil {
		safeFlush(c) // a panicking Flush must not skip Close
		errs = append(errs, c.Close())
	}
	return errors.Join(errs...)
}

func safeFlush(c LDClient) {
	defer func() {
		if r := recover(); r != nil {
			sdkLogger().Error("LaunchDarkly client flush failed during shutdown", "panic", r)
		}
	}()
	c.Flush()
}

// runShutdownHook runs experimental cleanup; its failure never fails Shutdown.
func runShutdownHook(ctx context.Context, hook func(context.Context)) {
	defer func() {
		if r := recover(); r != nil {
			sdkLogger().Error("experimental shutdown hook failed", "panic", r)
		}
	}()
	hook(ctx)
}

func init() {
	bridge.Logger = sdkLogger
	bridge.ConfigWithOptions = func(args any, opts bridge.RunOptions) any {
		return &AIConfig{args: args.(ConfigArgs), opts: opts}
	}
	bridge.ConversationID = conversationID
	bridge.WithExactConversationID = func(ctx context.Context, id string) context.Context {
		return context.WithValue(ctx, conversationKey{}, id)
	}
	bridge.TrackDataFor = func(configKey string, config, meta any, runID string, extra map[string]any) map[string]any {
		td := newTrackData(configKey, config.(AiConfig), meta.(VariationMeta), "")
		td.RunID = runID
		var m map[string]any
		b, _ := json.Marshal(td)
		_ = json.Unmarshal(b, &m)
		maps.Copy(m, extra)
		return m
	}
	bridge.ExtractVariationRaw = func(ctx context.Context, key string, c ldcontext.Context) (map[string]any, error) {
		raw, meta, err := readVariation(ctx, key, c, nil, false)
		if err != nil {
			return nil, err
		}
		if !meta.Enabled {
			return nil, fmt.Errorf("%w: variation %s is not enabled", ErrConfigDisabled, key)
		}
		return raw, nil
	}
}

// sdkKey is the SDK key for both init paths: InitOptions.SDKKey, else
// LD_SDK_KEY. Telemetry stamps it as highlight.project_id.
func sdkKey(opts InitOptions) string {
	if opts.SDKKey != "" {
		return opts.SDKKey
	}
	return envValue("LD_SDK_KEY")
}

// DefaultOTLPEndpoint is LaunchDarkly's observability backend, used when
// neither InitOptions.OTLPEndpoint nor OTEL_EXPORTER_OTLP_ENDPOINT is set.
const DefaultOTLPEndpoint = "https://otel.observability.app.launchdarkly.com"

// envValue reads an environment variable, treating a blank value as unset so
// a copied .env with empty entries keeps the built-in defaults.
func envValue(name string) string { return strings.TrimSpace(os.Getenv(name)) }

// withEnvDefaults fills unset options from the environment: LD_BASE_URI,
// LD_STREAM_URI, LD_EVENTS_URI, OTEL_EXPORTER_OTLP_ENDPOINT,
// LD_SERVICE_NAME (default "go-sdk") and LD_ENVIRONMENT.
func withEnvDefaults(o InitOptions) InitOptions {
	for _, f := range []struct {
		dst *string
		env string
	}{
		{&o.BaseURI, "LD_BASE_URI"}, {&o.StreamURI, "LD_STREAM_URI"}, {&o.EventsURI, "LD_EVENTS_URI"},
		{&o.OTLPEndpoint, "OTEL_EXPORTER_OTLP_ENDPOINT"}, {&o.ServiceName, "LD_SERVICE_NAME"}, {&o.Environment, "LD_ENVIRONMENT"},
	} {
		if *f.dst == "" {
			*f.dst = envValue(f.env)
		}
	}
	if o.ServiceName == "" {
		o.ServiceName = "go-sdk"
	}
	return o
}

// setupTelemetry runs under lifecycleMu, so only stateMu-guarded reads and
// writes of tracerProvider can race with it.
func setupTelemetry(ctx context.Context, sdkKey string, opts InitOptions) error {
	stateMu.Lock()
	ready := tracerProvider != nil
	stateMu.Unlock()
	if ready {
		return nil
	}
	opts = withEnvDefaults(opts)
	endpoint := opts.OTLPEndpoint
	if endpoint == "" {
		endpoint = DefaultOTLPEndpoint
	}
	exporterOpts := []otlptracehttp.Option{
		otlptracehttp.WithCompression(otlptracehttp.GzipCompression),
		otlptracehttp.WithEndpointURL(strings.TrimSuffix(endpoint, "/") + "/v1/traces"),
	}
	exp, err := otlptracehttp.New(ctx, exporterOpts...)
	if err != nil {
		return err
	}
	attrs := []attribute.KeyValue{attribute.String("highlight.project_id", sdkKey)}
	attrs = append(attrs, semconv.ServiceName(opts.ServiceName))
	if opts.Environment != "" {
		attrs = append(attrs, semconv.DeploymentEnvironment(opts.Environment))
	}
	tp := sdktrace.NewTracerProvider(
		sdktrace.WithResource(resource.NewWithAttributes(semconv.SchemaURL, attrs...)),
		sdktrace.WithSpanProcessor(ConversationIDSpanProcessor{}), // must precede the exporter
		sdktrace.WithBatcher(exp),
	)
	stateMu.Lock()
	tracerProvider = tp
	stateMu.Unlock()
	otel.SetTracerProvider(tp)
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(propagation.TraceContext{}, propagation.Baggage{}))
	return nil
}

// --- sdk-info and helper usage reporting ---------------------------------

var internalContext = ldcontext.NewBuilder("ld-internal-tracking").Kind("ld_ai").Anonymous(true).Build()

type pkgID struct{ name, version string }

var (
	reportMu      sync.Mutex
	knownPkgs     []pkgID
	reportedPkgs  = map[pkgID]bool{}
	pendingUsage  []usageRecord
	reportedUsage = map[string]bool{}
)

type usageRecord struct{ helper, pkg, version string }

// RegisterAiSdkPackage records a package for `$ld:ai:sdk:info`. Handler
// packages call it from init().
func RegisterAiSdkPackage(name, version string) {
	reportMu.Lock()
	defer reportMu.Unlock()
	id := pkgID{name, version}
	for _, k := range knownPkgs {
		if k == id {
			return
		}
	}
	knownPkgs = append(knownPkgs, id)
}

func init() { RegisterAiSdkPackage(PackageName, PackageVersion) }

// ReportUsage records a public helper call once per client. An empty
// pkg defaults to the core identity.
func ReportUsage(helper, pkg, version string) { reportUsage(helper, pkg, version) }

func reportUsage(helper, pkg, version string) {
	if pkg == "" {
		pkg, version = PackageName, PackageVersion
	}
	reportMu.Lock()
	if reportedUsage[helper] {
		reportMu.Unlock()
		return
	}
	reportMu.Unlock()
	stateMu.Lock()
	c := client
	stateMu.Unlock()
	reportMu.Lock()
	defer reportMu.Unlock()
	if c == nil {
		// Hold each helper once: a client that never initializes must not
		// grow this list on every call.
		for _, u := range pendingUsage {
			if u.helper == helper {
				return
			}
		}
		pendingUsage = append(pendingUsage, usageRecord{helper, pkg, version})
		return
	}
	sendUsage(c, usageRecord{helper, pkg, version})
}

func sendUsage(c LDClient, u usageRecord) {
	if reportedUsage[u.helper] {
		return
	}
	reportedUsage[u.helper] = true // counted even if track fails: no retry loop
	data := ldvalue.ObjectBuild().
		SetString("aiSdkName", PackageName).SetString("aiSdkVersion", PackageVersion).
		SetString("aiSdkLanguage", sdkLanguage).SetString("helper", u.helper).
		SetString("helperPackageName", u.pkg).SetString("helperPackageVersion", u.version).Build()
	safeTrack(c, "$ld:ai:sdk:usage", internalContext, 1, data)
}

func flushReports(c LDClient) {
	reportMu.Lock()
	defer reportMu.Unlock()
	for _, id := range knownPkgs {
		if reportedPkgs[id] {
			continue
		}
		reportedPkgs[id] = true
		data := ldvalue.ObjectBuild().SetString("aiSdkName", id.name).
			SetString("aiSdkVersion", id.version).SetString("aiSdkLanguage", sdkLanguage).Build()
		safeTrack(c, "$ld:ai:sdk:info", internalContext, 1, data)
	}
	for _, u := range pendingUsage {
		sendUsage(c, u)
	}
	pendingUsage = nil
}

func resetReports() {
	reportMu.Lock()
	defer reportMu.Unlock()
	reportedPkgs = map[pkgID]bool{}
	reportedUsage = map[string]bool{}
	pendingUsage = nil // held records are dropped too, as in JavaScript
}

// safeTrack never lets telemetry fail an AI call, including a panicking client.
func safeTrack(c LDClient, event string, ctx ldcontext.Context, metric float64, data ldvalue.Value) {
	defer func() { _ = recover() }()
	if err := c.TrackMetric(event, ctx, metric, data); err != nil {
		sdkLogger().Debug("track failed", "event", event, "error", err)
	}
}

// setClientForTest installs a client without emitting sdk-info.
func setClientForTest(c LDClient) {
	stateMu.Lock()
	defer stateMu.Unlock()
	client = c
}
