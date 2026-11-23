# shadowrequests

Shadow requests for the LaunchDarkly AI SDK for Go
(`github.com/launchdarkly/go-ai-sdk/experimental/shadowrequests`). This feature is experimental:
every name here is published but may change in a minor release, and none is exported from
`aiserver`.

## Shadow requests

A shadow request runs a candidate variation on real traffic without serving it. Use
`ShadowConfig` in place of `aiserver.Config` at the call sites you want to shadow. It takes the
same `aiserver.ConfigArgs` plus `ShadowOptions`, and its `Invoke` and `Stream` return exactly what
`Config`'s return. When the context matches a targeting rule or the fallthrough that has shadows
attached, each sampled shadow runs the same input on its own variation once the served call
succeeds. Its response is discarded. A shadow sends the same events a plain `Invoke` of that
variation sends, but on `$ld:ai-shadow:*` keys instead of `$ld:ai:*`, so nothing that reads AI
events counts shadow traffic unless it reads those keys on purpose. The events carry the served
call's `runId` and `isShadow: true`, so you can compare the two runs on the same input. A shadow
never delays or changes the served response, and its failures are only tracked.

| Event the served call sends | The same event from a shadow |
| --- | --- |
| `$ld:ai:<rest>`, for example `$ld:ai:tokens:input` or `$ld:ai:judge:relevance` | `$ld:ai-shadow:<rest>` |
| Any other key, such as a judge's custom metric key `my-relevance` | `$ld:ai-shadow:my-relevance` |
| A key already under `$ld:ai-shadow:` | Unchanged |

This covers the shadow's generation, duration and token events, each of its judges' own model
calls, and their scores, on the background path and through `RunShadow`, including a timeout or
a failed handler. The served call's events, its judges' events, and the `experimental.shadowConfig`
and `experimental.runShadow` usage events keep their usual keys.

```go
shadowed, err := shadowrequests.ShadowConfig(
	aiserver.ConfigArgs{Key: "my-ai-config", Handlers: []*aiserver.ProviderHandler{openai, anthropic}},
	shadowrequests.ShadowOptions{},
)
if err != nil { log.Fatal(err) }
resp, err := shadowed.Invoke(ctx, "Summarize this document", userCtx)

// Or hand each shadow to your own queue, and run it from a worker.
// OnTask runs before Invoke returns, so it must never block.
queue := make(chan shadowrequests.ShadowTask, 100)
shadowed, err = shadowrequests.ShadowConfig(args, shadowrequests.ShadowOptions{
	OnTask: func(task shadowrequests.ShadowTask) {
		select {
		case queue <- task:
		default: // queue full: drop this shadow
		}
	},
})
result := shadowrequests.RunShadow(ctx, <-queue, handlers) // never fails; see result.Error
```

| Option | Meaning |
| --- | --- |
| `Timeout` | How long each shadow may run before it is tracked as an error with `Error: "timeout"`. Zero means 30 seconds; a negative value makes `ShadowConfig` return an error. |
| `OnTask` | Run shadows yourself. It receives a `ShadowTask` per sampled shadow before `Invoke` returns, so it must not block: only enqueue it. A `ShadowTask` survives `encoding/json`, including unmodeled variation fields, and carries the caller's conversation id. `RunShadow` uses the client already set up and never starts one. |

Shadows run in a background goroutine by default, detached from the caller's cancellation, and
each one ends at its timeout. `aiserver.Shutdown` cancels any that are still running, and a
cancelled shadow records nothing. A shadow always runs as
a plain invoke, even when the served call streamed, with no tools. Pass handlers for
every provider your shadow variations use, since a shadow with no matching handler is skipped.

When a shadow variation has judges attached, they score the shadow's response after it succeeds,
the same way they score the served variation's, with each judge's own sampling rate. Their metric
events are on the judge's key mapped into `$ld:ai-shadow:` and carry the shadow's track data (its
`variationKey`, `isShadow` and `shadowRequestId`), so the scores show under the shadow variation
and never mix into the served variation's results. Each judge's own model call is tracked on
`$ld:ai-shadow:*` keys too, with the judge config's track data plus the shadow fields. The
scores are also on `ShadowResult.JudgeResults`, which is nil unless at least one judge scored. The
shadow's timeout covers only its own handler, a judge that fails or panics is logged and skipped
without setting `Error` or affecting the other judges, and `SkipJudges` applies only to the served
variation. Pass handlers for your judges' providers too.
Shadows need a client with `JSONVariationDetail`, which `*ld.LDClient` has; with any other client
the served call still runs and no shadow does. Plain `aiserver.Config` never runs shadows.

Shadows see your variables as JSON values. Each task copies them through `encoding/json`, so a
shadow shares no map with the caller, and a background shadow and a queued one see the same
values. A number arrives as a `float64`, except a whole number beyond ±2^53, which arrives as an
`int64` (a `uint64` above that range) with its exact value. `ShadowTask.Variables` is a
`ShadowVariables`, which decodes this way from JSON, so a task can also be embedded in your own
queue message. Other values arrive as whatever their JSON decodes to, so in
a few cases a shadow renders a prompt differently from the served call: a struct or a typed map
such as `map[string]string` becomes a `map[string]any` that `{{profile.name}}` can reach, and a
`time.Time` becomes a string, so it renders without the quotes the served call puts around it.
