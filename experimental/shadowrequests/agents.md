# shadowrequests — contributor and agent guide

Experimental feature package: its names are documented as Experimental and may change in a minor
release. When it is promoted to core, it moves out of `experimental/`.

## Shadow requests

All code is in `shadow_requests.go`; tests are `shadow_requests_test.go` (package `shadowrequests`,
for `inSample` and the `running` wait group) and `shadow_requests_surface_test.go`.

The core seam is `bridge.ConfigWithOptions(args, bridge.RunOptions{...})`, which returns an
`*aiserver.AIConfig` whose flag evaluation calls `RunOptions.Variation` instead of
`JSONVariation`, and whose tracked runs use `RunID`, merge `TrackData` into every event and add
`SpanAttributes` to the handler's root span. The served call (A) uses it with a `Variation` that
calls `JSONVariationDetail` and strips `shadowConfiguration`; each shadow (B) uses it with a fixed
value, `RunOptions.WrapHandler` for the deadline, `RunOptions.LogJudgeErrors`, `RunOptions.ShadowEvents`
and the shadow track data. Both set `RunOptions.NoUsage`, so only
`experimental.shadowConfig` and `experimental.runShadow` are reported.

`RunOptions.ShadowEvents` puts every event of the run on its shadow key (`bridge.ShadowEventKey`:
`$ld:ai:<rest>` → `$ld:ai-shadow:<rest>`, any other key prefixed, `$ld:ai-shadow:*` unchanged).
Core marks the run's `TrackData` (its unexported `extra`), and `track` maps the key of every event
sent with marked data, so B's run events and its judges' scores follow without a per-call change.
Each judge's own call gets `RunOptions{TrackData, ShadowEvents}` from `AIConfig.judgeRun`: the
judge's own track data and run id plus the five shadow fields, on shadow keys. Usage events never
carry the mark, so they stay on `$ld:ai:sdk:usage`.

Rules that are easy to break: shadows start only after A succeeds (after `Invoke` returns, or at
the stream's `done` event); preparing them never fails A (panics are recovered and logged); B
never gets tools or `shadowConfiguration`; B keeps its `judgeConfiguration`, and core `Invoke` runs
those judges on B's response with B's track data as the base, on the caller's full handler list and
unwrapped handler, so the deadline never covers a judge and a judge failure or panic never fails B; the deadline
wraps only B's handler, so a timeout is tracked once; no shadow event, B's or its judges', is sent on a `$ld:ai:` key; nothing is added to A's events; `aiserver.Shutdown` cancels background shadows through
`bridge.OnShutdown`. Sampling is
`sha1("<id>.<canonical key>")[:15] / 0xFFFFFFFFFFFFFFF`, with the canonical key from
`kitbase.ContextIdentity`.

Check: `go vet ./aiserver/... && go test -race ./internal/bridge/ ./experimental/shadowrequests/ -run 'Shadow|Sampling'`.
