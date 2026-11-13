# Changelog

## [1.0.0](https://github.com/launchdarkly/go-ai-sdk/compare/v0.1.0...v1.0.0) (2026-11-13)


### ⚠ BREAKING CHANGES

* **telemetry:** native graph adapters no longer open a per-node invoke_agent span, so per-node launchdarkly.config.key, summed usage and prompt/completion content are no longer on any span; context identity is on the graph span, per-turn usage on chat spans, tool data on execute_tool spans.
* **telemetry:** the graph span is now launchdarkly.graph and its key attribute launchdarkly.graph.key (was ld.ai.graph / ld.ai.graph.key); update trace queries, alerts and dashboards.
* **handlers:** langchainmessages, langchainagents, claudemessages, bedrockmessages and bedrockagents no longer read camelCase keys (maxTokens, topP, topK, frequencyPenalty, presencePenalty, stopSequences) from model.parameters; use the snake_case keys. In langchainmessages and langchainagents, stop_sequences now wins over stop when both are set.
* **handlers:** `NewHandler` in the claudeagents, claudemessages, langchainagents, langchainmessages, openaiagents and openaimessages packages, `ClaudeGraph`, `LangChainGraph`, `OpenAIGraph`, `ToClaudeAgents`, `ToOpenAIAgents` and `ToLangGraph` now return `(value, error)`. `LaunchDarklySkillStore.Start` now returns only `error`.
* **aiserver:** take the handler kit off the public API ([#60](https://github.com/launchdarkly/go-ai-sdk/issues/60))
* **aiserver:** remove the public SDKCall fields ([#51](https://github.com/launchdarkly/go-ai-sdk/issues/51))
* **shadow-requests:** move shadow requests to experimental/shadowrequests ([#48](https://github.com/launchdarkly/go-ai-sdk/issues/48))
* **AIC-3615:** return resolution errors from InspectConfig and ResolveGraph ([#9](https://github.com/launchdarkly/go-ai-sdk/issues/9))
* **telemetry:** name every handler root span invoke_agent ([#39](https://github.com/launchdarkly/go-ai-sdk/issues/39))
* **evaluations:** Evaluations from Code moves from aiserver/experimental to the evaluations package. Evaluations becomes EvaluationsModule, EvaluationsOptions becomes InitEvaluationsOptions, EvaluationRunArgs becomes RunOptions, InlineTool becomes EvalTool, and ErrEval* sentinels lose the Eval prefix. RunOptions.Tools is now one []EvalTool list; get library tools with EvaluationsModule.Tools().Get. The management API and UI hosts are configurable with APIBaseURI / UIBaseURI or LD_API_BASE_URI / LD_UI_BASE_URI. InlineRows accepts only map[string]any rows. GenerationConfig.Mode is now the GenerationMode type, and DefaultJudgeThreshold, DefaultScorerThreshold and DefaultPassRateThreshold are no longer exported.
* **lifecycle:** keep the installed client when InitWithClient is called again ([#37](https://github.com/launchdarkly/go-ai-sdk/issues/37))
* **skills:** promote Agent Skills to core ([#14](https://github.com/launchdarkly/go-ai-sdk/issues/14))
* **tracing:** add spans ([#21](https://github.com/launchdarkly/go-ai-sdk/issues/21))
* **openaiagents:** openaiagents.BuildOutputType and openaiagents.OutputType are no longer exported.
* **evaluations:** take the project key at InitEvaluations ([#28](https://github.com/launchdarkly/go-ai-sdk/issues/28))
* **evaluations:** resolve discrepancies in inline tool validation ([#20](https://github.com/launchdarkly/go-ai-sdk/issues/20))
* **langchain:** replace community LangChain ports with SDK-owned llms and state-graph packages ([#4](https://github.com/launchdarkly/go-ai-sdk/issues/4))

### Features

* **aiserver:** add NativeGraphRunner, the interface the native graph adapters satisfy ([#50](https://github.com/launchdarkly/go-ai-sdk/issues/50)) ([dce4f9e](https://github.com/launchdarkly/go-ai-sdk/commit/dce4f9e6b938f35db7619931640c2b42ee2b514c))
* **aiserver:** remove the public SDKCall fields ([#51](https://github.com/launchdarkly/go-ai-sdk/issues/51)) ([19d6a54](https://github.com/launchdarkly/go-ai-sdk/commit/19d6a541763541763fb1696df5193a68dc26e42f))
* **aiserver:** take the handler kit off the public API ([#60](https://github.com/launchdarkly/go-ai-sdk/issues/60)) ([75d6f5f](https://github.com/launchdarkly/go-ai-sdk/commit/75d6f5f6e2e6e2ac12f8b7917cb3c615f292e24f))
* **evaluations:** promote Evaluations from Code to its own package ([#29](https://github.com/launchdarkly/go-ai-sdk/issues/29)) ([7148474](https://github.com/launchdarkly/go-ai-sdk/commit/7148474d66620bc5aa3f9419cfffc27dde2474a6))
* **evaluations:** return verdicts for evaluations from code ([#59](https://github.com/launchdarkly/go-ai-sdk/issues/59)) ([d2e717d](https://github.com/launchdarkly/go-ai-sdk/commit/d2e717d0d00cd21d11309694c218d2aa20b2f9b4))
* **handlers:** add MaxRetries to the messages handlers ([#67](https://github.com/launchdarkly/go-ai-sdk/issues/67)) ([95c0a95](https://github.com/launchdarkly/go-ai-sdk/commit/95c0a9577acda9b3b15129bfa72145d25fd0d5f4))
* **judges:** point judge spans at the run they graded ([#40](https://github.com/launchdarkly/go-ai-sdk/issues/40)) ([c52bcb8](https://github.com/launchdarkly/go-ai-sdk/commit/c52bcb8cf9db9524a15207a8b3c7eb0ee105b683))
* **openaiagents:** unexport openaiagents.BuildOutputType ([#16](https://github.com/launchdarkly/go-ai-sdk/issues/16)) ([1acb79b](https://github.com/launchdarkly/go-ai-sdk/commit/1acb79ba8f230ddcf123810960eab8e66972d3ab))
* **skills:** promote Agent Skills to core ([#14](https://github.com/launchdarkly/go-ai-sdk/issues/14)) ([4ac2cf4](https://github.com/launchdarkly/go-ai-sdk/commit/4ac2cf4dec660b5cd7480ee69aeffa0077e68be5))
* **telemetry:** record the flag variant and value on feature_flag events ([#58](https://github.com/launchdarkly/go-ai-sdk/issues/58)) ([e9e59e3](https://github.com/launchdarkly/go-ai-sdk/commit/e9e59e362754ca827c0e75e57de37dfef2c89249))
* **tracing:** add spans ([#21](https://github.com/launchdarkly/go-ai-sdk/issues/21)) ([db57032](https://github.com/launchdarkly/go-ai-sdk/commit/db57032bed329ad4b7eb14c53ee9066f4d093e9e))


### Experimental

* **shadow-requests:** add experimental ShadowConfig wrapper ([#22](https://github.com/launchdarkly/go-ai-sdk/issues/22)) ([39bfe3f](https://github.com/launchdarkly/go-ai-sdk/commit/39bfe3f3dd821871f9b51a3082c04f9736f7f553))
* **shadow-requests:** keep whole numbers exact in shadow variables ([#54](https://github.com/launchdarkly/go-ai-sdk/issues/54)) ([21c65d1](https://github.com/launchdarkly/go-ai-sdk/commit/21c65d17393bd43ad32675f2ba94a3ad1d734d5a))
* **shadow-requests:** move shadow requests to experimental/shadowrequests ([#48](https://github.com/launchdarkly/go-ai-sdk/issues/48)) ([8dbd617](https://github.com/launchdarkly/go-ai-sdk/commit/8dbd6172ad84a478c321448c9e782c41c0d35652))


### Bug Fixes

* **AIC-3615:** return resolution errors from InspectConfig and ResolveGraph ([#9](https://github.com/launchdarkly/go-ai-sdk/issues/9)) ([943ad46](https://github.com/launchdarkly/go-ai-sdk/commit/943ad46a5880161c595effde7b34d8333a241274))
* **aiserver:** hand each handler only the tools its config attaches ([#42](https://github.com/launchdarkly/go-ai-sdk/issues/42)) ([200c45f](https://github.com/launchdarkly/go-ai-sdk/commit/200c45f8eb1b6fce77344faa363cab22916a707e))
* **evaluations:** answer a failed tool call with its error text ([#52](https://github.com/launchdarkly/go-ai-sdk/issues/52)) ([2ad506b](https://github.com/launchdarkly/go-ai-sdk/commit/2ad506b618cf1a2007a72638057e361a2ef733e7))
* **evaluations:** hash RFC 8785 canonical JSON and send contentHash ([#49](https://github.com/launchdarkly/go-ai-sdk/issues/49)) ([9267c63](https://github.com/launchdarkly/go-ai-sdk/commit/9267c6379533f303799494ea3a7154e2521daf39))
* **evaluations:** resolve discrepancies in inline tool validation ([#20](https://github.com/launchdarkly/go-ai-sdk/issues/20)) ([a5b13f5](https://github.com/launchdarkly/go-ai-sdk/commit/a5b13f54b6eba886b36c1af8dd2de641971e2516))
* **evaluations:** take the project key at InitEvaluations ([#28](https://github.com/launchdarkly/go-ai-sdk/issues/28)) ([9ec475c](https://github.com/launchdarkly/go-ai-sdk/commit/9ec475ccaa2b7c6acaa44f86afbde70cec27801f))
* **graph:** report a node's own judge scores on its node_done stream event ([#18](https://github.com/launchdarkly/go-ai-sdk/issues/18)) ([2fc0037](https://github.com/launchdarkly/go-ai-sdk/commit/2fc0037c4c7365fe7e18d11357a8c684aeb36386))
* **handlers:** read model parameters in snake_case only ([#68](https://github.com/launchdarkly/go-ai-sdk/issues/68)) ([1b724dc](https://github.com/launchdarkly/go-ai-sdk/commit/1b724dcb1a1c3e7c021597d8bceac4145ba6f941))
* **handlers:** return an error from handler constructors on invalid options ([#66](https://github.com/launchdarkly/go-ai-sdk/issues/66)) ([acb248a](https://github.com/launchdarkly/go-ai-sdk/commit/acb248ad79b649c78d45c7d1bb58595bae3584f9))
* **judges:** always send input to online judges ([#53](https://github.com/launchdarkly/go-ai-sdk/issues/53)) ([13520d5](https://github.com/launchdarkly/go-ai-sdk/commit/13520d5bfa94d549aa26166fd03f3093be953f50))
* **langchainagents:** give hand-off targets the earlier turns as text ([#81](https://github.com/launchdarkly/go-ai-sdk/issues/81)) ([cc9493f](https://github.com/launchdarkly/go-ai-sdk/commit/cc9493f573e352563008b7a02acef987340634ec))
* **langchain:** replace community LangChain ports with SDK-owned llms and state-graph packages ([#4](https://github.com/launchdarkly/go-ai-sdk/issues/4)) ([6c0f3c0](https://github.com/launchdarkly/go-ai-sdk/commit/6c0f3c0b2472a49ee0be809e952c81dd74aa8b9f))
* **langchain:** write aws.bedrock as the Bedrock provider name ([#62](https://github.com/launchdarkly/go-ai-sdk/issues/62)) ([3c32364](https://github.com/launchdarkly/go-ai-sdk/commit/3c32364726a305b52b7345f3d836339ea4daaf15))
* **lifecycle:** keep the installed client when InitWithClient is called again ([#37](https://github.com/launchdarkly/go-ai-sdk/issues/37)) ([8738c0f](https://github.com/launchdarkly/go-ai-sdk/commit/8738c0f6eb88b5850a2ab926126ccacf82156a40))
* **lifecycle:** treat a negative InitOptions.WaitFor as the 10s default ([#24](https://github.com/launchdarkly/go-ai-sdk/issues/24)) ([9eb26e7](https://github.com/launchdarkly/go-ai-sdk/commit/9eb26e7a19809bc84ce61dcd1861a974f8b2d110))
* **model-parameters:** forward only generation settings from model.parameters ([#26](https://github.com/launchdarkly/go-ai-sdk/issues/26)) ([441ea24](https://github.com/launchdarkly/go-ai-sdk/commit/441ea24e960e4798340cb84fb36e0fd3242df7b2))
* **telemetry:** name every handler root span invoke_agent ([#39](https://github.com/launchdarkly/go-ai-sdk/issues/39)) ([7fe1c79](https://github.com/launchdarkly/go-ai-sdk/commit/7fe1c79fe32d283808f978b3539e198d88f39059))
* **telemetry:** name the graph span launchdarkly.graph ([#63](https://github.com/launchdarkly/go-ai-sdk/issues/63)) ([f237213](https://github.com/launchdarkly/go-ai-sdk/commit/f2372132a4199b4a31ae43f86672c67a1e3d92ed))
* **telemetry:** open no per-node root span in native graphs ([#64](https://github.com/launchdarkly/go-ai-sdk/issues/64)) ([db0f898](https://github.com/launchdarkly/go-ai-sdk/commit/db0f898c57b0f1ed86bcb4b51793ce40da46eb29))
* **telemetry:** write root usage only when a turn reported it ([#61](https://github.com/launchdarkly/go-ai-sdk/issues/61)) ([72ecdae](https://github.com/launchdarkly/go-ai-sdk/commit/72ecdaea9badd4142ea2e9e5ba2ad802a42a305d))

## 0.1.0 (2026-10-30)


### ⚠ BREAKING CHANGES

* **skills:** rename FDv2SkillStore to LaunchDarklySkillStore ([#15](https://github.com/launchdarkly/go-ai-sdk/issues/15))
* **AIC-3570:** re-resolve graph flags on every invoke and stream ([#11](https://github.com/launchdarkly/go-ai-sdk/issues/11))
* **AIC-3614:** hide SDK-only entry points in internal/sdkapi ([#8](https://github.com/launchdarkly/go-ai-sdk/issues/8))
* **AIC-3612:** remove the SDK-owned global registry ([#6](https://github.com/launchdarkly/go-ai-sdk/issues/6))

### Features

* **AIC-3569:** add the Go AI SDK ([#1](https://github.com/launchdarkly/go-ai-sdk/issues/1)) ([b92042b](https://github.com/launchdarkly/go-ai-sdk/commit/b92042b5e39dae13700dbbe80ed51bbcd2600afc))
* **skills:** rename FDv2SkillStore to LaunchDarklySkillStore ([#15](https://github.com/launchdarkly/go-ai-sdk/issues/15)) ([0b87819](https://github.com/launchdarkly/go-ai-sdk/commit/0b87819b1e829c1e99f09d1f3da2c653a79f48fa))


### Bug Fixes

* **AIC-3570:** re-resolve graph flags on every invoke and stream ([#11](https://github.com/launchdarkly/go-ai-sdk/issues/11)) ([ada5a25](https://github.com/launchdarkly/go-ai-sdk/commit/ada5a25b404ddeff634cabb96dc97d2bdbab5598))
* **AIC-3571:** visit every child before its parent in ReverseTraverse ([#12](https://github.com/launchdarkly/go-ai-sdk/issues/12)) ([43f86d0](https://github.com/launchdarkly/go-ai-sdk/commit/43f86d015aca3d3c8fbb46fd54be5801f0f91fd7))
* **AIC-3576:** record deferred judge scores under the evaluated config ([#13](https://github.com/launchdarkly/go-ai-sdk/issues/13)) ([cd71325](https://github.com/launchdarkly/go-ai-sdk/commit/cd7132503721c7ee6e9f6196d4f294f6f13d806d))
* **AIC-3601:** stop multi-edge routing text from inviting a final answer ([#17](https://github.com/launchdarkly/go-ai-sdk/issues/17)) ([e85260c](https://github.com/launchdarkly/go-ai-sdk/commit/e85260c109645fcc231ba50278a83c7505f2c6b1))
* **AIC-3612:** remove the SDK-owned global registry ([#6](https://github.com/launchdarkly/go-ai-sdk/issues/6)) ([6152dc9](https://github.com/launchdarkly/go-ai-sdk/commit/6152dc96d70211b26559a6781a58df36118b7b0d))
* **AIC-3614:** hide SDK-only entry points in internal/sdkapi ([#8](https://github.com/launchdarkly/go-ai-sdk/issues/8)) ([acca582](https://github.com/launchdarkly/go-ai-sdk/commit/acca5827a3a88470b09f487ee76b5fb4f351af9a))
* **AIC-3726:** realign three usage IDs with the 1.0 factory names ([#23](https://github.com/launchdarkly/go-ai-sdk/issues/23)) ([805ece7](https://github.com/launchdarkly/go-ai-sdk/commit/805ece732bf5c9a5a60a9e88f22c7268d8d14984))
* Register every package that reports an AI SDK identity ([#10](https://github.com/launchdarkly/go-ai-sdk/issues/10)) ([cba37e5](https://github.com/launchdarkly/go-ai-sdk/commit/cba37e549df813843362a53b303aa699db841a8f))
