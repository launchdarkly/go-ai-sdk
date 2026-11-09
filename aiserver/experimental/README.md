# experimental

Experimental entry point of the LaunchDarkly AI SDK for Go
(`github.com/launchdarkly/go-ai-sdk/aiserver/experimental`). Every name here is published
but may change in a minor release; none is exported from `aiserver`. Agent Skills and Evaluations
from Code are core, each in its own package: see [`skills`](../../skills) and
[`evaluations`](../../evaluations).

<!-- Each feature owns one section below. Append; do not rewrite another feature's section. -->

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
// OnTask runs before Invoke returns, so it must never block.
queue := make(chan experimental.ShadowTask, 100)
shadowed, err = experimental.ShadowConfig(args, experimental.ShadowOptions{
	OnTask: func(task experimental.ShadowTask) {
		select {
		case queue <- task:
		default: // queue full: drop this shadow
		}
	},
})
result := experimental.RunShadow(ctx, <-queue, handlers) // never fails; see result.Error
```

| Option | Meaning |
| --- | --- |
| `Timeout` | How long each shadow may run before it is tracked as an error with `Error: "timeout"`. Zero means 30 seconds; a negative value makes `ShadowConfig` return an error. |
| `OnTask` | Run shadows yourself. It receives a `ShadowTask` per sampled shadow before `Invoke` returns, so it must not block: only enqueue it. A `ShadowTask` survives `encoding/json`, including unmodeled variation fields, and carries the caller's conversation id. `RunShadow` uses the client already set up and never starts one. |

Shadows run in a background goroutine by default, detached from the caller's cancellation, and
each one ends at its timeout. `aiserver.Shutdown` cancels any that are still running, and a
cancelled shadow records nothing. A shadow always runs as
a plain invoke, even when the served call streamed, with no tools and no judges. Pass handlers for
every provider your shadow variations use, since a shadow with no matching handler is skipped.
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
