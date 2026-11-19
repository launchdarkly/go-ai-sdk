// Package gatewaymodel builds the Vercel AI Gateway "creator/model" id for an
// AI Config. It is the Go port of the TypeScript and Python Vercel handlers'
// gateway model id mapping, and must stay in sync with both.
package gatewaymodel

import (
	"errors"
	"fmt"
	"regexp"
	"strings"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

// The conditions ID fails on. Each error ID returns wraps one of these and
// names the provider and model.
var (
	// ErrCreatorNotServed means the provider's own models are not in the Gateway catalog.
	ErrCreatorNotServed = errors.New("Vercel AI Gateway currently exposes no models created by this provider")
	// ErrMultipleCreators means the provider hosts several creators and the model names none.
	ErrMultipleCreators = errors.New("provider hosts models from multiple creators")
	// ErrUnknownProvider means the provider has no Gateway creator.
	ErrUnknownProvider = errors.New("cannot map provider to a Vercel AI Gateway creator")
)

// gatewayCreators are the creator prefixes the Gateway's model catalog
// (https://ai-gateway.vercel.sh/v1/models) returns, keyed by normalised name,
// plus a few spellings of the same creators.
var gatewayCreators = map[string]string{
	"alibaba":          "alibaba",
	"amazon":           "amazon",
	"anthropic":        "anthropic",
	"arceeai":          "arcee-ai",
	"bfl":              "bfl",
	"bytedance":        "bytedance",
	"cohere":           "cohere",
	"deepseek":         "deepseek",
	"fishaudio":        "fish-audio",
	"gemini":           "google",
	"google":           "google",
	"googleai":         "google",
	"inception":        "inception",
	"inclusionai":      "inclusionai",
	"inferencenet":     "inference-net",
	"interfaze":        "interfaze",
	"klingai":          "klingai",
	"meta":             "meta",
	"minimax":          "minimax",
	"mistral":          "mistral",
	"mistralai":        "mistral",
	"mixedbread":       "mixedbread",
	"moonshotai":       "moonshotai",
	"morph":            "morph",
	"nvidia":           "nvidia",
	"openai":           "openai",
	"perplexity":       "perplexity",
	"poolside":         "poolside",
	"prodia":           "prodia",
	"quiverai":         "quiverai",
	"recraft":          "recraft",
	"sakana":           "sakana",
	"spacexai":         "spacexai",
	"stepfun":          "stepfun",
	"tencent":          "tencent",
	"thinkingmachines": "thinkingmachines",
	"typesafe":         "typesafe-ai",
	"typesafeai":       "typesafe-ai",
	"voyage":           "voyage",
	"xiaomi":           "xiaomi",
	"xai":              "spacexai",
	"zai":              "zai",
}

// providerCreators maps every LaunchDarkly AI Config provider, keyed by
// normalised name, to its Gateway creator. An empty value means the provider
// hosts models from several creators, or its own models are not in the
// Gateway catalog, so the creator must come from the model. Providers in
// notServed are absent: they never have a creator.
var providerCreators = map[string]string{
	"anthropic":   "anthropic",
	"openai":      "openai",
	"bedrock":     "",
	"azure":       "openai",
	"gemini":      "google",
	"cohere":      "cohere",
	"cortex":      "",
	"cursor":      "",
	"databricks":  "",
	"deepseek":    "deepseek",
	"fireworksai": "",
	"meta":        "meta",
	"mistral":     "mistral",
	"perplexity":  "perplexity",
	"vertex":      "google",
	// Compatibility aliases found in existing configs.
	"google":     "google",
	"googleai":   "google",
	"mistralai":  "mistral",
	"spacexai":   "spacexai",
	"typesafe":   "typesafe-ai",
	"typesafeai": "typesafe-ai",
	"xai":        "spacexai",
}

// notServed are the providers whose own models the Gateway does not serve.
var notServed = map[string]bool{"ai21labs": true, "ibmwatson": true}

// familyCreators name the creator from a model family word, which must be
// followed by "-", "." or the end of the name. Matching ignores case.
var familyCreators = []struct {
	pattern *regexp.Regexp
	creator string
}{
	{regexp.MustCompile(`(?i)^(?:gpt|o[1-9])(?:[-.]|$)`), "openai"},
	{regexp.MustCompile(`(?i)^claude(?:[-.]|$)`), "anthropic"},
	{regexp.MustCompile(`(?i)^gemini(?:[-.]|$)`), "google"},
	{regexp.MustCompile(`(?i)^grok(?:[-.]|$)`), "spacexai"},
	{regexp.MustCompile(`(?i)^(?:command|aya)(?:[-.]|$)`), "cohere"},
	{regexp.MustCompile(`(?i)^deepseek(?:[-.]|$)`), "deepseek"},
	{regexp.MustCompile(`(?i)^llama(?:[-.]|$)`), "meta"},
	{regexp.MustCompile(`(?i)^(?:mistral|mixtral|codestral|pixtral)(?:[-.]|$)`), "mistral"},
	{regexp.MustCompile(`(?i)^sonar(?:[-.]|$)`), "perplexity"},
	{regexp.MustCompile(`(?i)^(?:nova|titan)(?:[-.]|$)`), "amazon"},
	{regexp.MustCompile(`(?i)^qwen(?:[-.]|$)`), "alibaba"},
}

var nonAlphanumeric = regexp.MustCompile(`[^a-z0-9]`)

// key lowercases s and drops every character that is not a letter or digit,
// so "AI21 Labs" and "ai21-labs" both become "ai21labs".
func key(s string) string {
	return nonAlphanumeric.ReplaceAllString(strings.ToLower(s), "")
}

// ID returns the AI Gateway model id for cfg: an explicit "creator/model" name
// unchanged, otherwise the creator read from the model name, then from the
// provider. It returns an error when no single creator can be determined.
func ID(cfg aiserver.AiConfig) (string, error) {
	model, provider := cfg.Model.Name, cfg.Provider.Name
	if strings.Contains(model, "/") {
		return model, nil
	}
	if creator, rest, ok := creatorFromModel(model); ok {
		return creator + "/" + rest, nil
	}

	providerKey := key(provider)
	creator, known := providerCreators[providerKey]
	switch {
	case creator != "":
		return creator + "/" + model, nil
	case notServed[providerKey]:
		return "", fmt.Errorf("%w: %q; inject a direct provider model instead", ErrCreatorNotServed, provider)
	case known:
		return "", fmt.Errorf("%w: %q, so %q cannot be converted to a Vercel creator/model id; store an explicit creator/model id or inject a model",
			ErrMultipleCreators, provider, model)
	}
	if provider == "" {
		provider = "unknown"
	}
	return "", fmt.Errorf("%w: %q; pass a creator/model id in config.model.name or inject a model", ErrUnknownProvider, provider)
}

// creatorFromModel finds the creator in a dotted creator segment (splitting
// there, so a region prefix goes and version dots stay) or, failing that, in
// the model family. The last segment is the model itself and is never read as
// a creator.
func creatorFromModel(model string) (creator, rest string, ok bool) {
	parts := strings.Split(model, ".")
	for i, part := range parts[:len(parts)-1] {
		if c := gatewayCreators[key(part)]; c != "" {
			return c, strings.Join(parts[i+1:], "."), true
		}
	}
	for _, f := range familyCreators {
		if f.pattern.MatchString(model) {
			return f.creator, model, true
		}
	}
	return "", "", false
}
