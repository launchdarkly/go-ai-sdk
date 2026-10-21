# Changelog

## 0.1.0 (2026-10-21)


### ⚠ BREAKING CHANGES

* **AIC-3570:** re-resolve graph flags on every invoke and stream ([#11](https://github.com/launchdarkly/go-ai-sdk/issues/11))
* **AIC-3614:** hide SDK-only entry points in internal/sdkapi ([#8](https://github.com/launchdarkly/go-ai-sdk/issues/8))
* **AIC-3612:** remove the SDK-owned global registry ([#6](https://github.com/launchdarkly/go-ai-sdk/issues/6))

### Features

* **AIC-3569:** add the Go AI SDK ([#1](https://github.com/launchdarkly/go-ai-sdk/issues/1)) ([b92042b](https://github.com/launchdarkly/go-ai-sdk/commit/b92042b5e39dae13700dbbe80ed51bbcd2600afc))


### Bug Fixes

* **AIC-3570:** re-resolve graph flags on every invoke and stream ([#11](https://github.com/launchdarkly/go-ai-sdk/issues/11)) ([ada5a25](https://github.com/launchdarkly/go-ai-sdk/commit/ada5a25b404ddeff634cabb96dc97d2bdbab5598))
* **AIC-3571:** visit every child before its parent in ReverseTraverse ([#12](https://github.com/launchdarkly/go-ai-sdk/issues/12)) ([43f86d0](https://github.com/launchdarkly/go-ai-sdk/commit/43f86d015aca3d3c8fbb46fd54be5801f0f91fd7))
* **AIC-3576:** record deferred judge scores under the evaluated config ([#13](https://github.com/launchdarkly/go-ai-sdk/issues/13)) ([cd71325](https://github.com/launchdarkly/go-ai-sdk/commit/cd7132503721c7ee6e9f6196d4f294f6f13d806d))
* **AIC-3612:** remove the SDK-owned global registry ([#6](https://github.com/launchdarkly/go-ai-sdk/issues/6)) ([6152dc9](https://github.com/launchdarkly/go-ai-sdk/commit/6152dc96d70211b26559a6781a58df36118b7b0d))
* **AIC-3614:** hide SDK-only entry points in internal/sdkapi ([#8](https://github.com/launchdarkly/go-ai-sdk/issues/8)) ([acca582](https://github.com/launchdarkly/go-ai-sdk/commit/acca5827a3a88470b09f487ee76b5fb4f351af9a))
* Register every package that reports an AI SDK identity ([#10](https://github.com/launchdarkly/go-ai-sdk/issues/10)) ([cba37e5](https://github.com/launchdarkly/go-ai-sdk/commit/cba37e549df813843362a53b303aa699db841a8f))
