# Examples

Runs one scenario against your own LaunchDarkly environment. Each scenario
evaluates an AI Config by flag key, so first create AI Configs in your project
(the reference flag keys below match the JavaScript SDK examples) and point the
command at them. Not a test suite: nothing asserts automatically except `graph-history` and
`judge`, which exit non-zero when their checks fail.

```bash
export LD_SDK_KEY=<your server-side SDK key>
export OPENAI_API_KEY=... ANTHROPIC_API_KEY=...   # whichever providers your AI Configs use
# For Bedrock configs, set the usual AWS credentials (AWS_PROFILE or AWS_ACCESS_KEY_ID /
# AWS_SECRET_ACCESS_KEY, AWS_REGION) or AWS_BEARER_TOKEN_BEDROCK.
go run ./examples <scenario> <flag-key> "<user input>"
```

Optional: `LD_BASE_URI`, `LD_STREAM_URI`, `LD_EVENTS_URI` (non-production
LaunchDarkly), `OTEL_EXPORTER_OTLP_ENDPOINT`, `LD_SERVICE_NAME`,
`LD_ENVIRONMENT`.

| Scenario | Demonstrates | Reference flag |
|---|---|---|
| `agent` | `Config().Invoke` through the global registry | `launch-darkly-documentation-summarizer` |
| `history` | `Invoke` with prior conversation turns | same |
| `conversation` | several turns under one `gen_ai.conversation.id`, inline judge scores on stderr | same |
| `streaming` | `Config().Stream` | same |
| `openai-only` | `openaiagents.OpenAIAgents` with its own tools, no registry | `launch-darkly-documentation-summarizer-open-ai-only` |
| `judge` | `SkipJudges` → `JudgeTasks` → `RunJudge` on goroutines, with four verification checks | `launch-darkly-documentation-summarizer` |
| `graph` | `Graph().Invoke` | `travel-agent-flow` |
| `graph-streaming` | `Graph().Stream` built inside a conversation scope and iterated after it; text on stdout, `[node_start]` / `[node_done]` / `[handoff]` on stderr; no JSON file | `travel-agent-flow` |
| `graph-history` | `Graph().Invoke` with a generated image in history, two shapes; fails unless both answers name the colour | `travel-agent-flow` |
| `native-graph` | `ResolveGraph` + `claudeagents.ToClaudeAgents` | `travel-agent-flow` |
| `native-graph-langchain` | `ResolveGraph` + `langchainagents.ToLangGraph`: the graph runs as one langgraphgo `StateGraph` (community port); tracked for the example context | `travel-agent-flow` |

Failure cases exit 1 with one `Error: …` line, e.g.
`go run ./examples agent doesnt-exist "hi"` and
`go run ./examples graph travel-agent-flow-wrong-key "hi"`.

`native-graph-langchain` runs on community ports (`langchaingo`, `langgraphgo`), not
an official LangGraph; with a missing or disabled graph flag it exits 1 with
`ToLangGraph`'s `ErrGraphDisabled` error.

## Files

| File | Purpose |
|---|---|
| `main.go` | Scenario dispatch, client init and shutdown, exit codes |
| `register.go` | Registers every handler and tool in `aiserver.GlobalRegistry` (`web-search` → `claudeagents.ClaudeWebSearch`; `langchainagents` as the `{*, agent}` fallback) |
| `tools.go` | Tool implementations, keyed exactly as the flag variations name them |
| `scenarios.go` | One function per scenario |
| `utils.go` | `newContext`, `newMultiContext`, `newConversationID`, `writeOutput` (→ `output/`), `solidColorPNGBase64` |
| `user_preferences.json` | Fixture for `get-user-preferences` (embedded) |
