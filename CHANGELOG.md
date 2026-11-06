# Changelog

## [0.2.0](https://github.com/launchdarkly/go-ai-sdk/compare/v0.1.0...v0.2.0) (2026-11-06)


### ⚠ BREAKING CHANGES

* **evaluations:** Evaluations from Code moves from aiserver/experimental to the evaluations package. Evaluations becomes EvaluationsModule, EvaluationsOptions becomes InitEvaluationsOptions, EvaluationRunArgs becomes RunOptions, InlineTool becomes EvalTool, and ErrEval* sentinels lose the Eval prefix. RunOptions.Tools is now one []EvalTool list; get library tools with EvaluationsModule.Tools().Get. The management API and UI hosts are configurable with APIBaseURI / UIBaseURI or LD_API_BASE_URI / LD_UI_BASE_URI. InlineRows accepts only map[string]any rows. GenerationConfig.Mode is now the GenerationMode type, and DefaultJudgeThreshold, DefaultScorerThreshold and DefaultPassRateThreshold are no longer exported.
* **lifecycle:** keep the installed client when InitWithClient is called again ([#37](https://github.com/launchdarkly/go-ai-sdk/issues/37))
* **skills:** promote Agent Skills to core ([#14](https://github.com/launchdarkly/go-ai-sdk/issues/14))
* **tracing:** add spans ([#21](https://github.com/launchdarkly/go-ai-sdk/issues/21))
* **openaiagents:** openaiagents.BuildOutputType and openaiagents.OutputType are no longer exported.
* **evaluations:** take the project key at InitEvaluations ([#28](https://github.com/launchdarkly/go-ai-sdk/issues/28))
* **evaluations:** resolve discrepancies in inline tool validation ([#20](https://github.com/launchdarkly/go-ai-sdk/issues/20))
* **langchain:** replace community LangChain ports with SDK-owned llms and state-graph packages ([#4](https://github.com/launchdarkly/go-ai-sdk/issues/4))

### Features

* **evaluations:** promote Evaluations from Code to its own package ([#29](https://github.com/launchdarkly/go-ai-sdk/issues/29)) ([7148474](https://github.com/launchdarkly/go-ai-sdk/commit/7148474d66620bc5aa3f9419cfffc27dde2474a6))
* **openaiagents:** unexport openaiagents.BuildOutputType ([#16](https://github.com/launchdarkly/go-ai-sdk/issues/16)) ([1acb79b](https://github.com/launchdarkly/go-ai-sdk/commit/1acb79ba8f230ddcf123810960eab8e66972d3ab))
* **skills:** promote Agent Skills to core ([#14](https://github.com/launchdarkly/go-ai-sdk/issues/14)) ([4ac2cf4](https://github.com/launchdarkly/go-ai-sdk/commit/4ac2cf4dec660b5cd7480ee69aeffa0077e68be5))
* **tracing:** add spans ([#21](https://github.com/launchdarkly/go-ai-sdk/issues/21)) ([db57032](https://github.com/launchdarkly/go-ai-sdk/commit/db57032bed329ad4b7eb14c53ee9066f4d093e9e))


### Experimental

* **shadow-requests:** add experimental ShadowConfig wrapper ([#22](https://github.com/launchdarkly/go-ai-sdk/issues/22)) ([39bfe3f](https://github.com/launchdarkly/go-ai-sdk/commit/39bfe3f3dd821871f9b51a3082c04f9736f7f553))


### Bug Fixes

* **evaluations:** resolve discrepancies in inline tool validation ([#20](https://github.com/launchdarkly/go-ai-sdk/issues/20)) ([a5b13f5](https://github.com/launchdarkly/go-ai-sdk/commit/a5b13f54b6eba886b36c1af8dd2de641971e2516))
* **evaluations:** take the project key at InitEvaluations ([#28](https://github.com/launchdarkly/go-ai-sdk/issues/28)) ([9ec475c](https://github.com/launchdarkly/go-ai-sdk/commit/9ec475ccaa2b7c6acaa44f86afbde70cec27801f))
* **graph:** report a node's own judge scores on its node_done stream event ([#18](https://github.com/launchdarkly/go-ai-sdk/issues/18)) ([2fc0037](https://github.com/launchdarkly/go-ai-sdk/commit/2fc0037c4c7365fe7e18d11357a8c684aeb36386))
* **langchain:** replace community LangChain ports with SDK-owned llms and state-graph packages ([#4](https://github.com/launchdarkly/go-ai-sdk/issues/4)) ([6c0f3c0](https://github.com/launchdarkly/go-ai-sdk/commit/6c0f3c0b2472a49ee0be809e952c81dd74aa8b9f))
* **lifecycle:** keep the installed client when InitWithClient is called again ([#37](https://github.com/launchdarkly/go-ai-sdk/issues/37)) ([8738c0f](https://github.com/launchdarkly/go-ai-sdk/commit/8738c0f6eb88b5850a2ab926126ccacf82156a40))
* **lifecycle:** treat a negative InitOptions.WaitFor as the 10s default ([#24](https://github.com/launchdarkly/go-ai-sdk/issues/24)) ([9eb26e7](https://github.com/launchdarkly/go-ai-sdk/commit/9eb26e7a19809bc84ce61dcd1861a974f8b2d110))
* **model-parameters:** forward only generation settings from model.parameters ([#26](https://github.com/launchdarkly/go-ai-sdk/issues/26)) ([441ea24](https://github.com/launchdarkly/go-ai-sdk/commit/441ea24e960e4798340cb84fb36e0fd3242df7b2))

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
