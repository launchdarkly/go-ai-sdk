# experimental

Experimental entry point of the LaunchDarkly AI SDK for Go
(`github.com/launchdarkly/go-ai-sdk/aiserver/experimental`). Every name here is published
but may change in a minor release; none is exported from `aiserver`.

<!-- Each feature owns one section below. Append; do not rewrite another feature's section. -->

## Evaluations from Code

Runs a dataset through your own handler, scores each output with LaunchDarkly AI Judges
and/or local scorers, reports every result to LaunchDarkly as custom events, and waits for
the run verdict. Use it as a CI gate.

```bash
go get github.com/launchdarkly/go-ai-sdk/aiserver/experimental
go get github.com/launchdarkly/go-ai-sdk/openaimessages   # or any handler package for the generation step
```

### Environment variables

| Variable | Required | Description |
| --- | --- | --- |
| `LD_API_TOKEN` | yes, unless `EvaluationsOptions.APIToken` | LaunchDarkly API access token; authorizes every management API request. Checked first. |
| `LD_SDK_KEY` | yes, unless `EvaluationsOptions.SDKKey` or a client is already initialized (`aiserver.InitClient` / `InitWithClient`) | Server-side SDK key for the client that sends result events. When a client is already installed it is used and this key is ignored (with a warning). |

Handlers read their provider's own variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, AWS chain).
Missing credentials fail `InitEvaluations` with `ErrEvalMissingCredential`, before any network I/O.

### Example

```go
evals, err := experimental.InitEvaluations(experimental.EvaluationsOptions{}) // LD_API_TOKEN, LD_SDK_KEY
if err != nil { log.Fatal(err) }
judge, _ := experimental.NewJudge("accuracy", nil)
res, err := evals.Run(ctx, experimental.EvaluationRunArgs{
	ProjectKey: "my-project",
	Key:        "support-bot-" + time.Now().Format("20060102-150405"), // unique per run
	Dataset:    experimental.DatasetKey("support-questions"),
	Handler:    openaimessages.NewHandler(openaimessages.Options{}),
	Generation: &experimental.GenerationConfig{
		Provider: "OpenAI", Model: "gpt-4o",
		Mode:     experimental.GenerationModeCompletion, // required: "completion" or "agent"
		Messages: []aiserver.Message{{Role: "system", Text: "You are a support agent."}},
	},
	Criteria: []experimental.Criterion{judge},
})
if err != nil { log.Fatal(err) }
fmt.Println(res.URL)
if !res.Passed { os.Exit(1) }
```

| Argument | Meaning |
| --- | --- |
| `Dataset` | `DatasetKey("...")` for a hosted dataset, or `InlineRows{...}` (`DatasetRow` values or `map[string]any` with `input`, `expectedOutput`, `variables`, `metadata`, optional `rowIdx`). |
| `Handler` / `Handlers` | Exactly one. Each config (generation and every judge) picks its handler by provider and mode; an exact provider beats `*`; there is no mode fallback. |
| `Generation` / `AIConfig` | Exactly one. `AIConfig: &AIConfigRef{Key, Variation}` reads the latest variation version; `Overrides` replace single fields. |
| `Tools` / `InlineTools` | Library tools (resolved and version-pinned by key) and tools defined in code (`InlineTool{Implementation, Schema, Description}`). Inline keys are lowercase. |
| `Criteria` | `*Judge` (`NewJudge`) and `*Scorer` (`NewScorer`). With `AIConfig`, nil uses the variation's judges and `[]Criterion{}` runs none. |
| `Concurrency` | In-flight handler, judge and scorer calls (default 10). |
| `PollInterval` / `PollTimeout` | Summary polling (default 2s / 3m). Large datasets need a longer timeout. |

The SDK never computes a verdict: it reports scores, and LaunchDarkly rules on them against the
thresholds pinned on the evaluation. `Passed` is true only when no row errored or failed.
The management API host is fixed at `https://app.launchdarkly.com`.

### API reference

```go
func InitEvaluations(opts EvaluationsOptions) (*Evaluations, error)
func (m *Evaluations) Run(ctx context.Context, args EvaluationRunArgs) (*EvalRunResult, error)
func NewJudge(key string, opts *JudgeOptions) (*Judge, error)
func NewScorer(name string, fn ScorerFunc, opts *ScorerOptions) (*Scorer, error)
```

| Type | Key fields |
| --- | --- |
| `EvaluationsOptions` | `APIToken` (default `LD_API_TOKEN`), `SDKKey` (default `LD_SDK_KEY`), `HTTPClient` (default `http.DefaultClient`), `RequestTimeout` (30s per attempt), `MaxAttempts` (4), `Logger` |
| `EvalRunResult` | `Passed` (true iff the final summary has no error and no failed rows), `URL` (the run in the LaunchDarkly UI), `RunID`, `Summary` |
| `RunSummary` | `TotalRows`, `PassedRows`, `FailedRows`, `ErrorRows`, `PendingRows` |

Errors are sentinels for `errors.Is`: `ErrEvalInvalidArgument` (validation, before any I/O),
`ErrEvalMissingCredential`, `ErrEvalNoClient`, `ErrEvalAPI` (non-2xx; `*APIError` carries the
response), `ErrEvalTransport`, `ErrEvalInvalidResponse`, `ErrEvalNotFound`, `ErrEvalInvalidConfig`,
`ErrEvalJudgeResolution`, `ErrEvalUploadFailed`, `ErrEvalPollTimeout`.

## Agent Skills

Version-pinned skill references on AI Configs, verified skill content, and materialization of that
content to disk as `<root>/<key>/SKILL.md`. Install with
`go get github.com/launchdarkly/go-ai-sdk`; POSIX only for materialization.

### Keep a directory in sync with LaunchDarkly

```go
store, err := experimental.NewLaunchDarklySkillStore(os.Getenv("LD_SDK_KEY"), nil) // server-side sdk- key only
if err != nil { log.Fatal(err) }
if _, err := store.Start(); err != nil { log.Fatal(err) }
defer store.Close()
// Gate boot on the first payload. Before it, WriteAllSkills reports
// "unavailable" and deletes nothing.
if !store.WaitForSkills(10 * time.Second) {
	log.Printf("no skill payload yet: failed=%q diagnostics=%+v", store.Failed(), store.Diagnostics())
}
experimental.SetSkillStore(store)

report, watcher, err := experimental.WatchAllSkills(ctx, "/etc/agent/skills", nil)
if err != nil { log.Fatal(err) } // a bad root or option fails here, on your goroutine
defer watcher.Close()
for _, a := range report.Errors() {
	log.Printf("skill %q: %s", a.Key(), a.ErrorMessage()) // Key() is "" for a run-level failure
}
```

A skill revoked in LaunchDarkly is removed from disk within the debounce window (500ms).

### The skills one AI Config uses

```go
res := aiserver.InspectConfig(ctx, "my-agent", ldcontext.New("user-1"))
refs, err := experimental.SkillRefs(res.Config) // an error, never [], when the config failed to resolve
if err != nil { return err }
report, err := experimental.WriteSkills(refs, "/etc/agent/skills", nil)
```

### Read content without touching disk

```go
outcome, err := experimental.GetSkillResult("pdf-extraction", 0) // 0 = newest version
if err != nil { return err } // only ErrNoSkillStore
switch outcome.Reason() {
case experimental.ReasonIntegrityFailure:
	return fmt.Errorf("refusing to run: %s", outcome.Detail()) // fail closed on tampering
case experimental.ReasonOK:
	use(outcome.Skill().Content()) // opaque bytes: the SDK never parses or decodes them
}
```

For local development and tests, `experimental.NewInMemorySkillStore(map[string]any{...})` holds raw
wire objects (`key`, `version`, `content`, `contentHash`, optional `name`, `description`).

### API reference

| Function | Returns | Notes |
| --- | --- | --- |
| `SetSkillStore(store)` | — | nil is ignored; `aiserver.Shutdown` clears it. Does not close a replaced store; a running watcher keeps the store it started with. |
| `SkillRefs(*aiserver.AiConfig)` | `[]SkillReference, error` | Pure: no client, store or telemetry. A nil config, a non-array `skills`, or any malformed entry is an error (never a partial list, which `WriteSkills` with pruning would act on). |
| `GetSkill(key, version)` | `*Skill, error` | nil when missing, at another version, or withheld. |
| `GetSkillResult(key, version)` | `SkillOutcome, error` | `absent`, `integrity_failure`, `ok`, `store_unavailable`, `wrong_version`. |
| `GetSkills(refs)` | `[]Skill, error` | `[]SkillKey`, `[]SkillReference` or mixed `[]SkillRequest`; input order, misses omitted. |
| `AllSkills()` | `[]Skill, error` | Newest version per key. |
| `WriteSkills(skills, root, *WriteSkillsOptions)` | `ReconcileReport, error` | Any slice of `Skill`, `*Skill`, `SkillReference`, `SkillKey` (or mixed `[]SkillInput`). |
| `WriteAllSkills(root, *WriteSkillsOptions)` | `ReconcileReport, error` | Every skill the store holds (the `"*"` key). |
| `WatchSkills` / `WatchAllSkills(ctx, …, *WatchSkillsOptions)` | `ReconcileReport, *SkillWatcher, error` | Needs a store with `AddListener`; stops on `Close` or when `ctx` is done. |
| `NewLaunchDarklySkillStore(sdkKey, *LaunchDarklySkillStoreOptions)` | `*LaunchDarklySkillStore, error` | `Start`, `WaitForSkills`, `IsInitialized`, `Failed`, `Diagnostics`, `Close` (final). |

- `WriteSkillsOptions`: `NoPrune` (pruning is on by default), `Timeout *time.Duration` (nil = 10s,
  0 = already exhausted, negative is an error), `OnUnavailable` (`keep` default; `raise` returns
  `ErrSkillsUnavailable`). `WatchSkillsOptions` embeds it and adds `Debounce` (0 = 500ms) and
  `OnReconcile`.
- `LaunchDarklySkillStoreOptions`: `BaseURI` (default `https://sdk.launchdarkly.com`; given alone it serves
  both endpoints), `StreamURI` (default `https://stream.launchdarkly.com`), `Mode` (`stream`
  default, or `poll`), `PollInterval` (30s), `ReadTimeout` (10s poll, 300s stream),
  `InitialBackoff` (1s), `MaxBackoff` (30s). URIs must be `https`; zero means default, negative is
  an error.
- Constants: `SkillFilename` (`SKILL.md`), `ManifestFilename` (`.launchdarkly-skills.json` — add it
  to `.gitignore`), `ManifestVersion` (1). Errors are sentinels for `errors.Is`.

### Environment variables

The skills API reads none. The examples read `LD_SDK_KEY` (required there) for the store.

### Integrity log record

An integrity failure is logged at ERROR through the SDK logger (`aiserver.SetLogger`, default `slog.Default()`) as
`ld.skills.integrity_failure {sorted-key JSON}`, with the same mapping on the `ld_skills` attribute.
SIEM rules can match the event name and `reason_code`. While Agent Skills is experimental, the event
name, fields and codes may change only in a minor release with a changelog entry under Experimental.

## Shadow requests

A shadow request runs a candidate variation on real traffic without serving it. Use
`ShadowConfig` in place of `aiserver.Config` at the call sites you want to shadow. It takes the
same `aiserver.ConfigArgs` plus `ShadowOptions`, and its `Invoke` and `Stream` return exactly what
`Config`'s return. When the context matches a targeting rule or the fallthrough that has shadows
attached, each sampled shadow runs the same input on its own variation once the served call
succeeds. Its response is discarded. Its events are the normal `$ld:ai:*` events for that
variation, with the served call's `runId` and `isShadow: true`, so you can compare the two on the
usual AI metrics. A shadow never delays or changes the served response, and its failures are only
tracked.

```go
shadowed, err := experimental.ShadowConfig(
	aiserver.ConfigArgs{Key: "my-ai-config", Handlers: []*aiserver.ProviderHandler{openai, anthropic}},
	experimental.ShadowOptions{},
)
if err != nil { log.Fatal(err) }
resp, err := shadowed.Invoke(ctx, "Summarize this document", userCtx)

// Or hand each shadow to your own queue, and run it from a worker.
shadowed, err = experimental.ShadowConfig(args, experimental.ShadowOptions{
	OnTask: func(task experimental.ShadowTask) { queue <- task },
})
result := experimental.RunShadow(ctx, <-queue, handlers) // never fails; see result.Error
```

| Option | Meaning |
| --- | --- |
| `Timeout` | How long each shadow may run before it is tracked as an error with `Error: "timeout"`. Zero means 30 seconds; a negative value makes `ShadowConfig` return an error. |
| `OnTask` | Run shadows yourself. It receives a `ShadowTask` per sampled shadow and should only enqueue it. A `ShadowTask` survives `encoding/json`, except `Config.Raw`. |

Shadows run in a background goroutine by default, detached from the caller's cancellation, and
each one ends at its timeout. `aiserver.Shutdown` does not wait for them. A shadow always runs as
a plain invoke, even when the served call streamed, with no tools and no judges. Pass handlers for
every provider your shadow variations use, since a shadow with no matching handler is skipped.
Shadows need a client with `JSONVariationDetail`, which `*ld.LDClient` has; with any other client
the served call still runs and no shadow does. Plain `aiserver.Config` never runs shadows.
