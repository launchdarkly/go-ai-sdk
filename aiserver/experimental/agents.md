# experimental — contributor and agent guide

Experimental entry point. Each feature's names are documented as Experimental. A feature promoted
to core leaves this package: Agent Skills is now package `skills`, and Evaluations from Code is
package `evaluations`.

<!-- Each feature owns one section below. Append; do not rewrite another feature's section. -->

## Shadow requests

All code is in `shadow_requests.go`; tests are `shadow_requests_test.go` (package `experimental`,
for `inSample` and the `running` wait group) and `shadow_requests_surface_test.go`.

The core seam is `bridge.ConfigWithOptions(args, bridge.RunOptions{...})`, which returns an
`*aiserver.AIConfig` whose flag evaluation calls `RunOptions.Variation` instead of
`JSONVariation`, and whose tracked runs use `RunID`, merge `TrackData` into every event and add
`SpanAttributes` to the handler's root span. The served call (A) uses it with a `Variation` that
calls `JSONVariationDetail` and strips `shadowConfiguration`; each shadow (B) uses it with a fixed
value, a deadline wrapped handler and the shadow track data. Both pass `sdkcall.Mark()`, so only
`experimental.shadowConfig` and `experimental.runShadow` are reported.

Rules that are easy to break: shadows start only after A succeeds (after `Invoke` returns, or at
the stream's `done` event); preparing them never fails A (panics are recovered and logged); B
never gets tools, judges or `shadowConfiguration`; the deadline wraps the handler, so a timeout is
tracked once; nothing is added to A's events; `aiserver.Shutdown` cancels background shadows through
`bridge.OnShutdown`. Sampling is
`sha1("<id>.<canonical key>")[:15] / 0xFFFFFFFFFFFFFFF`, with the canonical key from
`aiserver.ContextIdentity`.

Check: `go vet ./aiserver/... && go test -race ./aiserver/experimental/ -run 'Shadow|Sampling'`.
