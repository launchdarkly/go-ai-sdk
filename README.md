# LaunchDarkly AI SDK for Go

LaunchDarkly AI SDK for Go. Manage prompts, models, tools, and agent graphs as AI
Configs, target and roll them out per user, score quality with judges, and track
cost, latency, and errors. Handlers for OpenAI, Anthropic, Bedrock, and LangChain.

```bash
go get github.com/launchdarkly/go-ai-sdk/aiserver
go get github.com/launchdarkly/go-ai-sdk/claudemessages   # plus the handlers you use
```

Requires Go 1.26.

## Layout

| Package | Tier | What it does |
|---|---|---|
| [`aiserver`](aiserver) | 0 | Core client: `Config`, `Graph`, `ResolveGraph`, `InspectConfig`, registry, lifecycle, judges, telemetry |
| [`aiserver/experimental`](aiserver/experimental) | 0 | Experimental entry point: Agent Skills, Evaluations from Code |
| [`claudemessages`](claudemessages) | 1 | Anthropic Messages API handler — `{Anthropic, messages}` |
| [`openaimessages`](openaimessages) | 1 | OpenAI Responses API handler — `{OpenAI, messages}` |
| [`bedrockmessages`](bedrockmessages) | 1 | Amazon Bedrock Converse handler — `{Bedrock, messages}` |
| [`langchainmessages`](langchainmessages) | 1 | langchaingo handler for any provider — `{*, messages}` |
| [`claudeagents`](claudeagents) | 1 | Anthropic agent handler, `ClaudeGraph`, `ToClaudeAgents` — `{Anthropic, agent}` |
| [`openaiagents`](openaiagents) | 1 | OpenAI agent handler, `OpenAIGraph`, `ToOpenAIAgents` — `{OpenAI, agent}` |
| [`bedrockagents`](bedrockagents) | 1 | Bedrock agent handler, `BedrockGraph` — `{Bedrock, agent}` |
| [`langchainagents`](langchainagents) | 1 | langchaingo + langgraphgo agent handler, `LangChainGraph`, `ToLangGraph` — `{*, agent}` |
| [`examples`](examples) | — | Integration entry point |

`langchainmessages` and `langchainagents` use community Go ports of LangChain
and LangGraph (`tmc/langchaingo`, `smallnest/langgraphgo`). `claudeagents` runs the
Claude Code built-in tools on the host itself (see its README's security note). Each
package README lists what its Go version does differently.

## Quick start

```go
ctx := context.Background()
if _, err := aiserver.InitClient(ctx, aiserver.InitOptions{}); err != nil { // LD_SDK_KEY
	log.Fatal(err)
}
defer aiserver.Shutdown(ctx)
user := ldcontext.New("user-123")

// 1. One handler.
claude := claudemessages.NewHandler(claudemessages.Options{})
resp, err := aiserver.Config(aiserver.ConfigArgs{Key: "summariser", Handlers: []*aiserver.ProviderHandler{claude}}).
	Invoke(ctx, "What is a feature flag?", user,
		aiserver.WithVariables(map[string]any{"tone": "brief"})) // optional; also WithHistory(msgs)
if errors.Is(err, aiserver.ErrConfigDisabled) {
	// the variation is off: serve a fallback
}
fmt.Println(resp.Text()) // string response, or structured output as JSON

// 2. Several providers: the variation's provider and mode choose.
openai := openaimessages.NewHandler(openaimessages.Options{})
cfg := aiserver.Config(aiserver.ConfigArgs{Key: "summariser", Handlers: []*aiserver.ProviderHandler{claude, openai}})
for ev, err := range cfg.Stream(ctx, "Hello", user) {
	if err != nil {
		log.Fatal(err)
	}
	if ev.Type == aiserver.EventChunk {
		fmt.Print(ev.Text)
	}
}

// 3. A multi-agent graph.
agents := []*aiserver.ProviderHandler{claudeagents.NewHandler(claudeagents.Options{}), openaiagents.NewHandler(openaiagents.Options{})}
out, err := aiserver.Graph("travel-agent-flow", aiserver.GraphOptions{Handlers: agents}).
	Invoke(ctx, "Plan a week in Lisbon", user)
```

Failures are sentinel errors for `errors.Is`: `ErrNotInitialized`,
`ErrConfigNotFound`, `ErrConfigDisabled`, `ErrNoHandler`, `ErrGraphDisabled`
(see [aiserver/README.md](aiserver/README.md#errors)).

## Configuration

- **SDK key:** `LD_SDK_KEY`, or `aiserver.InitOptions{SDKKey: ...}`, or bring your
  own client with `aiserver.InitWithClient`.
- **Non-production LaunchDarkly:** `InitOptions{BaseURI, StreamURI, EventsURI}`, or
  the `LD_*_URI` variables below.

An `InitOptions` field that is set wins over its variable. Blank values count as
unset.

| Variable | Required | Description |
|---|---|---|
| `LD_SDK_KEY` | yes, unless `InitOptions.SDKKey` or `InitWithClient` | LaunchDarkly server-side SDK key; also stamped as `highlight.project_id` |
| `LD_BASE_URI` | no | LaunchDarkly polling/base URI for non-production environments |
| `LD_STREAM_URI` | no | LaunchDarkly streaming URI |
| `LD_EVENTS_URI` | no | LaunchDarkly events URI |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | no | OTLP/HTTP trace endpoint (`/v1/traces` is appended); default `https://otel.observability.app.launchdarkly.com` |
| `LD_SERVICE_NAME` | no | `service.name` resource attribute; default `go-sdk` |
| `LD_ENVIRONMENT` | no | `deployment.environment` resource attribute; omitted when unset |
| `LD_ENVIRONMENT_ID` | no | LaunchDarkly environment ID for AI Config Monitoring trace correlation (`environmentId` / `feature_flag.set.id`); omitted when unset |
| `LD_API_TOKEN` | only for Evaluations from Code | LaunchDarkly API access token ([experimental](aiserver/experimental/README.md#evaluations-from-code)) |

- **Providers:** each handler reads its provider's usual environment
  (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, the AWS credential chain /
  `AWS_BEARER_TOKEN_BEDROCK`) unless you inject a client.

## Telemetry

`InitClient` registers a global OpenTelemetry tracer provider: OTLP/HTTP with
gzip to `InitOptions.OTLPEndpoint`, else `OTEL_EXPORTER_OTLP_ENDPOINT`, else
`aiserver.DefaultOTLPEndpoint`; resource attributes `highlight.project_id` (SDK
key), `service.name` (`LD_SERVICE_NAME`, default `go-sdk`) and
`deployment.environment` (`LD_ENVIRONMENT`, when set); W3C propagators; and a span
processor that stamps `gen_ai.conversation.id` from
`aiserver.WithConversationID`. Handler spans carry prompt and completion
content.

## Development

```bash
go mod download   # fetch dependencies
go build ./...    # build every package from source
go test -race ./...
go vet ./...
gofmt -l .
```

Examples run against a real LaunchDarkly environment — see
[examples/README.md](examples/README.md).
