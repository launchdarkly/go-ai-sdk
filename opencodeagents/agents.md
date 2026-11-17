# opencodeagents: agent guide

## Role and tier

Tier 1 agent handler, `{*, agent}`. It drives an `opencode serve` child over HTTP; it imports
`aiserver`, `internal/handlerkit`, `internal/kitbase`, `internal/sdkapi` and
`internal/procgroup`, and no provider SDK or OpenCode client.

## File map

| File | Contents |
|---|---|
| `handler.go` | `Options`, `NewHandler`, `OpenCodeAgents`, `SessionError`, tool planning (`planTools`), the permission `ruleset`, prompt flattening, `defaultModel`, the run (`agent.run`). |
| `process.go` | Child start (`startServer`: path checks, env, free port, stderr temp file, health gate), stop (group SIGTERM → SIGKILL), basic-auth client. |
| `session.go` | One session: `GET /event` reader, `POST /session`, the concurrent message POST, drain window, abort; `POST /mcp` register / disconnect; SSE frame parsing. |
| `reducer.go` | Bus events → output text, `chat` / `execute_tool` spans, usage. |
| `bridge.go` | Per-call loopback MCP server for `ToolFunc` tools (JSON-RPC over streamable HTTP, live `GET` stream). |
| `builtins.go` | The 12 `NativeTool` sentinels. |
| `fake_test.go` | The fake `opencode` (the test binary re-execs itself via `TestMain` when `LD_OPENCODE_FAKE=serve`), its state file, bus-event builders, LD and span fakes. |
| `live_test.go` | Build tag `opencode_live`: one bridged-tool turn against a real `opencode` of the pinned version (`pinnedOpenCode`) and a local stub OpenAI-compatible provider; no credentials. CI job `opencode-live` runs it. |
| `handler_test.go`, `contract_test.go`, `spans_test.go`, `bridge_test.go`, `process_unix_test.go`, `usage_test.go` | Tests. |

## Public exports

`PackageName`, `PackageVersion`, `SpanName`; `Options`, `NewHandler`; `InvokeOptions`,
`OpenCodeAgents`; `SessionError`; `OpenCodeBash` … `OpenCodeApplyPatch`. Usage ids
`opencode-agents.createOpenCodeHandler` (`NewHandler`) and `opencode-agents.opencodeAgents`
(`OpenCodeAgents`; it builds its handler without reporting the factory id). `providesFor` is
unexported and uses `kitbase.WildcardProvider`.

## Implementation details

- **Per-call order:** resolve the model (an error fails before any span or child) →
  `StartHandlerRun` (request model = the sent id) → `startServer` → bridge (if any `ToolFunc`
  is offered) and `POST /mcp` → `runSession` → `Finish` / `Fail` / `Abandon`. Deferred, in
  this order: disconnect the bridge, stop the bridge, stop the child.
- **The event stream is opened before the session exists** and read on its own goroutine into
  an unbuffered channel, so nothing is lost while the session is created. Events with another
  session's id are dropped; an event with no session id is kept. Only this session's
  `session.idle` ends the turn.
- **Drain window:** after the message POST returns 2xx, each event restarts a `drainFallback`
  timer; when it fires the turn is done. A POST status ≥ 400 fails at once.
- **Leaving early** (error, cancel, consumer stop) cancels the POST and sends
  `POST /session/{id}/abort` with a fresh 5 s context, because the call's own context may be done.
- **Reducer:** text counts only for messages `message.updated` has marked `assistant`; parts that
  arrive before their role are held. A snapshot emits only the suffix past what that part
  already emitted, and nothing when it does not extend it. Tool spans open on the first
  `running` state with input (or on `completed`/`error` if `running` was skipped), once per call
  id; a sentinel's `TrackNativeTool` fires there. Leftover spans at the end are ended.
- **Stop:** group SIGTERM, wait `shutdownGrace`, group SIGKILL, wait for the reap, then one more
  group SIGKILL for children that outlived the leader. `procgroup` falls back to signalling the
  leader when the group signal fails.
- **Timings the tests shorten:** `startupTimeout`, `shutdownGrace`, `drainFallback`,
  `keepaliveInterval` (package vars).

## Dependencies

Standard library only, plus the module's own packages and `go-sdk-common` (`ldcontext`) and the
OTel trace API.

## Common pitfalls

- **A nil `Options.Env` is refused** (it would leak the parent environment); pass `[]string{}`
  for an empty one.
- **`OpenCodeWrite` alone also allows `edit` and `apply_patch`.** All three are one grant.
- **The bridge's `GET` streams.** OpenCode 1.18.4's MCP client marked the server failed on a 405
  (found by the Python package); 1.19.5 accepts a 405 (checked with the live test). The bridge
  streams for both. A server that does not connect makes `POST /mcp` report a status other than
  `connected`, and the call fails.
- **Bumping OpenCode:** change `pinnedOpenCode` in `live_test.go` and `OPENCODE_VERSION` in the
  `opencode-live` CI job together; the test fails when the installed version differs.
- **`langchainagents` is also `{*, agent}`**: pass one of the two to a handler list.
- **The fake is the test binary.** A test that needs a new OpenCode behaviour extends
  `fakeSpec`, not a mock client.
