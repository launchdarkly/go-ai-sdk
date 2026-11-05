# skills — agents.md

## Role and tier

Tier 1 by layout (it imports `aiserver`), core by stage, in its own package. Not a handler: no `providesFor`. It imports `aiserver` the way
the provider packages do, and `aiserver` never imports it. `public_test.go` exercises the exported
names from outside the package. The names are pinned in `surface_test.go`'s `skillsSurface`, and
`aiserver/surface_test.go` asserts that none of them is on the `aiserver` root.

**Usage IDs.** The first statement of each public helper is `aiserver.ReportUsage("client.<id>", "", "")`,
before any argument or store check, so a failing call still reports. The IDs match JS and Python:

| ID | Reported by |
| --- | --- |
| `client.allSkills` | `AllSkills` |
| `client.getSkill` | `GetSkill` |
| `client.getSkillResult` | `GetSkillResult` |
| `client.getSkills` | `GetSkills` |
| `client.skillRefs` | `SkillRefs` |
| `client.writeSkills` | `WriteSkills`, `WriteAllSkills` |
| `client.watchSkills` | `WatchSkills`, `WatchAllSkills` |

No helper calls another: the watcher's initial and later reconciles run the unexported
`writeSkills`, `WriteSkills` resolves through `resolveFromStore` and raw objects, and the accessors
share `resolveFromStore`. Keep it that way, or call a non-reporting version. `SetSkillStore`, the
store constructors and methods, and the watcher's methods report nothing. `usage_test.go` checks the
helpers, the watcher's reconciles, `SetSkillStore` and the constructors.

| File | Responsibility |
| --- | --- |
| `doc.go` | Package doc. |
| `types.go` | Immutable value types (unexported fields + getters), closed sets, on-disk constants. |
| `store.go` | `SkillStore` seam and optional halves, `Listener`, module state, `SetSkillStore`, `clearState` (registered with `bridge.OnShutdown`), readiness probe. |
| `telemetry.go` | The only place signals and integrity log records are built (three signals, ten reason codes). |
| `verify.go` | `verifyRawSkill` (shape → size → encoding → hash), `resolveFromStore` (the one retrieval path: key check, then version check), `newestByKey`. |
| `accessors.go`, `memstore.go` | `SkillRefs`, `GetSkill*`, `AllSkills`; `InMemorySkillStore` and the shared `objectSet`. |
| `writeskills.go` | Options, the availability gate, request resolution (`"*"` enumerates raw objects; never `AllSkills`). |
| `materialize_unix.go` | Root validation, manifest load/rewrite, write/adopt/update, prune. `materialize_other.go` refuses non-POSIX. |
| `safefs_unix.go` | Descriptor-pinned primitives on `golang.org/x/sys/unix` and the `fsOps` interception hooks. |
| `fdv2_reader.go`, `fdv2_store.go` | Pure FDv2 protocol reader; transport (HTTP, SSE, backoff, lifecycle). Standard library only. |
| `watch.go` | Watcher goroutine: trailing debounce, context cancellation, detach on close. |

```go
// skills exports
SkillReference SkillKey SkillRequest SkillInput Skill SkillFields SkillOutcome SkillOutcomeReason
ReconcileAction ReconcileActionKind ReconcileReport OnUnavailable SkillStore ListenerStore
ListenerRemover InitializationProber Listener InMemorySkillStore LaunchDarklySkillStore
LaunchDarklySkillStoreOptions LaunchDarklySkillStoreMode StoreDiagnostics WriteSkillsOptions
WatchSkillsOptions SkillWatcher SkillFilename ManifestFilename ManifestVersion Reason* Action*
OnUnavailableKeep OnUnavailableRaise LaunchDarklySkillStoreModeStream LaunchDarklySkillStoreModePoll
ErrNoSkillStore ErrInvalidArgument ErrSkillsUnavailable ErrStoreNotWatchable ErrUnsupportedKind
ErrStoreClosed ErrUnsupportedPlatform NewSkillReference NewSkill NewSkillOutcome NewListener
NewInMemorySkillStore NewLaunchDarklySkillStore SetSkillStore SkillRefs GetSkill GetSkillResult
GetSkills AllSkills WriteSkills WriteAllSkills WatchSkills WatchAllSkills
```

**Lifecycle.** `aiserver.InitClient` is not required for skills; a store is configured with `SetSkillStore`
only (never an `InitOptions` field).
`aiserver.Shutdown` clears the store and emitter through the bridge hook; it does not close a
`LaunchDarklySkillStore`.

**OTel.** Skills emit no spans, and the only `client.track` they cause is their `$ld:ai:sdk:usage` event. The
tracer provider is set up by `aiserver.InitClient`.

**Security posture (non-relaxable).** The root is pinned once per reconcile; every read and write
under it is `*at()`-relative, so swapping a directory for a symlink cannot redirect anything. The
manifest read and the adoption comparison read share one no-follow, non-blocking, `fstat`-checked,
`len+1`-bounded primitive (`readRegularAt`). Only manifest-listed paths under a matching key are
overwritten or pruned; a corrupt manifest or incomplete retrieval suppresses all deletion. Content is
re-verified immediately before writing. The SDK never parses `Skill.Content`; do not add a reader.

**Telemetry seam.** Private emitter (`setEmitterForTesting`), default no-op, `language` = `go`. Store
keys and hashes are shape-checked and redacted before reaching any signal or record.

| Dependency | Why |
| --- | --- |
| `golang.org/x/sys/unix` | `Openat`, `Mkdirat`, `Renameat`, `Unlinkat`, `Fstatat`, `Fchmod`, `Mkfifo` (tests). Already in `go.mod`. |
| standard library | Everything else, including the FDv2 transport. |

**Common pitfalls.**

- Calling `WriteAllSkills` before `WaitForSkills` returns true does not prune (it reports
  unavailable) — but a hand-written store without `IsInitialized` is treated as ready, so its empty
  answer reads as "every skill revoked".
- No `removed` action does not mean nothing is stale: a run with a manifest or retrieval error
  prunes nothing. Check `report.OK()`.
- One watcher (or one `WriteSkills` at a time) per root; interleaved reconciles lose manifest entries.
- `Listener` is a pointer so `RemoveListener` can find it; wrapping the same func twice makes two
  listeners.
- Tests that touch `fsOps`, the umask or module state must not run in parallel.

Check: `gofmt -l skills && go vet ./skills/ && go test -race ./skills/ ./aiserver/`.
