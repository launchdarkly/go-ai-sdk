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
value, `RunOptions.WrapHandler` for the deadline, `RunOptions.LogJudgeErrors` and the shadow track data. Both set `RunOptions.NoUsage`, so only
`experimental.shadowConfig` and `experimental.runShadow` are reported.

Rules that are easy to break: shadows start only after A succeeds (after `Invoke` returns, or at
the stream's `done` event); preparing them never fails A (panics are recovered and logged); B
never gets tools or `shadowConfiguration`; B keeps its `judgeConfiguration`, and core `Invoke` runs
those judges on B's response with B's track data as the base, on the caller's full handler list and
unwrapped handler, so the deadline never covers a judge and a judge failure never fails B; the deadline
wraps only B's handler, so a timeout is tracked once; nothing is added to A's events; `aiserver.Shutdown` cancels background shadows through
`bridge.OnShutdown`. Sampling is
`sha1("<id>.<canonical key>")[:15] / 0xFFFFFFFFFFFFFFF`, with the canonical key from
`kitbase.ContextIdentity`.

Check: `go vet ./aiserver/... && go test -race ./experimental/shadowrequests/ -run 'Shadow|Sampling'`.
