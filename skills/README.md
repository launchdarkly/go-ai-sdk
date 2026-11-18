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
if err := store.Start(); err != nil { log.Fatal(err) }
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

`ConfigSkills` is the form to use under a watcher: it evaluates the AI Config on every reconcile,
so a config that moves a skill to a new version, or stops referencing one, reaches disk.

```go
// After aiserver.InitClient (ConfigSkills never initializes the client) and SetSkillStore.
src := skills.ConfigSkills("my-agent", ldcontext.New("user-1"))
report, watcher, err := skills.WatchSkillsFrom(ctx, src, "/etc/agent/skills", nil)
if err != nil { log.Fatal(err) }
defer watcher.Close()
```

Each read is a real evaluation (one per reconcile, bounded by the debounce) with a null default.
No client, an evaluation error, the default, or a malformed `skills` list is a failed read: the
reconcile reports a run-level `error` and changes nothing on disk. A disabled variation answers no
skills, so they leave disk. The watcher follows changes to the config key through the LaunchDarkly
client's flag-change signal; with no client, an offline client, or a client without
`GetFlagTracker`, `WatchSkillsFrom` returns `ErrStoreNotWatchable`.

For a one-shot write at boot, `skills.WriteSkillsFrom(ctx, src, root, nil)` reads the source once.
`SkillRefs(aiserver.InspectConfig(...).Config)` with `WriteSkills` is also safe at boot, but do
not hand that list to `WatchSkills`: it is fixed, so the watcher never sees the config change.

Which changes in LaunchDarkly reach disk depends on what the watcher was given:

| Change in LaunchDarkly | `WatchAllSkills` | `WatchSkills` (fixed list) | `WatchSkillsFrom` |
| --- | --- | --- | --- |
| A skill is deleted (possible only once nothing pins it) | pruned | stays on disk, reported `error` on every run | already pruned, when the config stopped referencing it |
| A config moves `pdf-extraction` v2 → v3 | unaffected: `"*"` already follows the newest version | v2 stays on disk, reported `error`; v3 is never written | v3 replaces v2 |
| A config stops referencing a skill | unaffected: `"*"` asks for the whole library | stays on disk | pruned |

### Your own source

```go
src, err := skills.NewSkillSource(func(ctx context.Context) ([]skills.SkillReference, error) {
	return lookUpRefs(ctx) // honour ctx: it carries the reconcile's timeout
}, "planner", "researcher") // the AI Config keys whose changes trigger a read; may be empty
```

Return `[]skills.SkillReference{}` for "no skills" (it prunes). A nil slice, an error, or an entry
with an invalid key or a version below 1 is a failed read, so a bug in the source cannot prune
everything. With no config keys, the watcher needs no client: it re-reads on store changes and
when you call `watcher.Notify()`, which goes through the same debounce.

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
| `SetSkillStore(store)` | — | nil is ignored; `aiserver.Shutdown` clears it. Does not close a replaced store; a running watcher keeps the store it started with. |
| `SkillRefs(*aiserver.AiConfig)` | `[]SkillReference, error` | Needs no client and no store; its only telemetry is its usage event. A nil config, a non-array `skills`, or any malformed entry is an error (never a partial list, which `WriteSkills` with pruning would act on). |
| `GetSkill(key, version)` | `*Skill, error` | nil when missing, at another version, or withheld. |
| `GetSkillResult(key, version)` | `SkillOutcome, error` | `absent`, `integrity_failure`, `ok`, `store_unavailable`, `wrong_version`. |
| `GetSkills(refs)` | `[]Skill, error` | `[]SkillKey`, `[]SkillReference` or mixed `[]SkillRequest`; input order, misses omitted. |
| `AllSkills()` | `[]Skill, error` | Newest version per key. |
| `WriteSkills(skills, root, *WriteSkillsOptions)` | `ReconcileReport, error` | Any slice of `Skill`, `*Skill`, `SkillReference`, `SkillKey` (or mixed `[]SkillInput`). |
| `WriteAllSkills(root, *WriteSkillsOptions)` | `ReconcileReport, error` | Every skill the store holds (the `"*"` key). |
| `WatchSkills` / `WatchAllSkills(ctx, …, *WatchSkillsOptions)` | `ReconcileReport, *SkillWatcher, error` | Needs a store with `AddListener`; stops on `Close` or when `ctx` is done. |
| `ConfigSkills(key, ldcontext.Context)` | `SkillSource` | The skills one AI Config pins for one context, evaluated on every read. A blank key gives a source every call refuses. |
| `NewSkillSource(refs, configKeys...)` | `SkillSource, error` | A nil `refs` or a blank key is `ErrInvalidSkillSource` (wraps `ErrInvalidArgument`). |
| `WriteSkillsFrom(ctx, SkillSource, root, *WriteSkillsOptions)` | `ReconcileReport, error` | Reads the source once, within `Timeout`; a failed read is unavailable retrieval. |
| `WatchSkillsFrom(ctx, SkillSource, root, *WatchSkillsOptions)` | `ReconcileReport, *SkillWatcher, error` | Reads the source on every store change, config-key change and `Notify`. |
| `(*SkillWatcher).Notify()` | — | Schedules a reconcile through the debounce; a no-op after `Close`. |
| `NewLaunchDarklySkillStore(sdkKey, *LaunchDarklySkillStoreOptions)` | `*LaunchDarklySkillStore, error` | `Start`, `WaitForSkills`, `IsInitialized`, `Failed`, `Diagnostics`, `Close` (final). |

- `WriteSkillsOptions`: `NoPrune` (pruning is on by default), `Timeout *time.Duration` (nil = 10s,
  0 = already exhausted, negative is an error), `OnUnavailable` (`keep` default; `raise` returns
  `ErrSkillsUnavailable`). `WatchSkillsOptions` embeds it and adds `Debounce` (0 = 500ms) and
  `OnReconcile`.
- `LaunchDarklySkillStoreOptions`: `BaseURI` (default `https://sdk.launchdarkly.com`; given alone it serves
  both endpoints), `StreamURI` (default `https://stream.launchdarkly.com`), `Mode` (`stream`
  default, or `poll`), `PollInterval` (30s), `ReadTimeout` (10s poll, 300s stream),
  `InitialBackoff` (1s), `MaxBackoff` (30s). URIs must be `https`, except plain `http` to
  `localhost`, `127.0.0.1` or `::1` for a local test double; zero means default, negative is
  an error. A redirect (any 3xx except 304) is never followed, since that would send the SDK key
  to another host: it stops delivery, like a 401, and `Failed()` says why.
- Constants: `SkillFilename` (`SKILL.md`), `ManifestFilename` (`.launchdarkly-skills.json` — add it
  to `.gitignore`), `ManifestVersion` (1). Errors are sentinels for `errors.Is`.

## Environment variables

`NewLaunchDarklySkillStore` reads `LD_BASE_URI` and `LD_STREAM_URI`, the same variables `aiserver.InitClient`
uses, when neither `BaseURI` nor `StreamURI` is set, so a process pointed at a non-production
instance sends skill delivery there too. They are read once, at construction, and follow the
option rules: `LD_BASE_URI` alone moves both endpoints, a blank or whitespace-only value is
unset, and a non-`https` value (other than loopback) is an error naming the variable. Either
option set means neither variable is read. Nothing else in the skills API reads the environment. The
examples read `LD_SDK_KEY` (required there) for the store.

## Integrity log record

An integrity failure is logged at ERROR through the SDK logger (`aiserver.SetLogger`, default `slog.Default()`) as
`ld.skills.integrity_failure {sorted-key JSON}`, with the same mapping on the `ld_skills` attribute.
SIEM rules can match the event name and `reason_code`. The event name, fields and codes change
only in a major release.
