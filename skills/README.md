# skills — Agent Skills for the LaunchDarkly AI SDK for Go

Version-pinned skill references on AI Configs, verified skill content, and materialization of that
content to disk as `<root>/<key>/SKILL.md`. Part of the core
SDK, in its own package:

```sh
go get github.com/launchdarkly/go-ai-sdk/skills
```

Materialization is POSIX only.

## Keep a directory in sync with LaunchDarkly

```go
store, err := skills.NewLaunchDarklySkillStore(os.Getenv("LD_SDK_KEY"), nil) // server-side sdk- key only
if err != nil { log.Fatal(err) }
if _, err := store.Start(); err != nil { log.Fatal(err) }
defer store.Close()
// Gate boot on the first payload. Before it, WriteAllSkills reports
// "unavailable" and deletes nothing.
if !store.WaitForSkills(10 * time.Second) {
	log.Printf("no skill payload yet: failed=%q diagnostics=%+v", store.Failed(), store.Diagnostics())
}
skills.SetSkillStore(store)

report, watcher, err := skills.WatchAllSkills(ctx, "/etc/agent/skills", nil)
if err != nil { log.Fatal(err) } // a bad root or option fails here, on your goroutine
defer watcher.Close()
for _, a := range report.Errors() {
	log.Printf("skill %q: %s", a.Key(), a.ErrorMessage()) // Key() is "" for a run-level failure
}
```

A skill revoked in LaunchDarkly is removed from disk within the debounce window (500ms).

## The skills one AI Config uses

```go
res := aiserver.InspectConfig(ctx, "my-agent", ldcontext.New("user-1"))
refs, err := skills.SkillRefs(res.Config) // an error, never [], when the config failed to resolve
if err != nil { return err }
report, err := skills.WriteSkills(refs, "/etc/agent/skills", nil)
```

## Read content without touching disk

```go
outcome, err := skills.GetSkillResult("pdf-extraction", 0) // 0 = newest version
if err != nil { return err } // only ErrNoSkillStore
switch outcome.Reason() {
case skills.ReasonIntegrityFailure:
	return fmt.Errorf("refusing to run: %s", outcome.Detail()) // fail closed on tampering
case skills.ReasonOK:
	use(outcome.Skill().Content()) // opaque bytes: the SDK never parses or decodes them
}
```

For local development and tests, `skills.NewInMemorySkillStore(map[string]any{...})` holds raw
wire objects (`key`, `version`, `content`, `contentHash`, optional `name`, `description`).

## API reference

| Function | Returns | Notes |
| --- | --- | --- |
| `SetSkillStore(store)` | — | nil is ignored; `Shutdown` clears it. Does not close a replaced store; a running watcher keeps the store it started with. |
| `SkillRefs(*AiConfig)` | `[]SkillReference, error` | Pure: no client, store or telemetry. A nil config, a non-array `skills`, or any malformed entry is an error (never a partial list, which `WriteSkills` with pruning would act on). |
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

## Environment variables

The skills API reads none. The examples read `LD_SDK_KEY` (required there) for the store.

## Integrity log record

An integrity failure is logged at ERROR through the SDK logger (`aiserver.SetLogger`, default `slog.Default()`) as
`ld.skills.integrity_failure {sorted-key JSON}`, with the same mapping on the `ld_skills` attribute.
SIEM rules can match the event name and `reason_code`. The event name, fields and codes change
only in a major release.
