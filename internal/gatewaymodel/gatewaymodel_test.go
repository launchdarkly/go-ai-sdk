package gatewaymodel

import (
	"errors"
	"strings"
	"testing"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

func config(provider, model string) aiserver.AiConfig {
	var cfg aiserver.AiConfig
	cfg.Provider.Name = provider
	cfg.Model.Name = model
	return cfg
}

// The cases shared with the TypeScript model-id tests and the Python
// test_model_id tests. Keep all three in sync.
func TestMapsEveryLaunchDarklyProvider(t *testing.T) {
	cases := []struct{ provider, model, want string }{
		{"Anthropic", "model", "anthropic/model"},
		{"OpenAI", "model", "openai/model"},
		{"Bedrock", "anthropic.claude-sonnet-4", "anthropic/claude-sonnet-4"},
		{"Azure", "model", "openai/model"},
		{"Gemini", "model", "google/model"},
		{"Cohere", "model", "cohere/model"},
		{"Cortex", "llama-4-scout", "meta/llama-4-scout"},
		{"Cursor", "claude-sonnet-4", "anthropic/claude-sonnet-4"},
		{"Databricks", "llama-4-maverick", "meta/llama-4-maverick"},
		{"DeepSeek", "model", "deepseek/model"},
		{"Fireworks AI", "qwen-3-235b", "alibaba/qwen-3-235b"},
		{"Meta", "model", "meta/model"},
		{"Mistral", "model", "mistral/model"},
		{"Perplexity", "model", "perplexity/model"},
		{"Vertex", "model", "google/model"},
	}
	for _, c := range cases {
		t.Run(c.provider, func(t *testing.T) {
			got, err := ID(config(c.provider, c.model))
			if err != nil || got != c.want {
				t.Fatalf("ID(%q, %q) = %q, %v; want %q", c.provider, c.model, got, err, c.want)
			}
		})
	}
}

func TestFailsForCreatorTheGatewayDoesNotServe(t *testing.T) {
	for _, provider := range []string{"AI21 Labs", "IBM Watson"} {
		t.Run(provider, func(t *testing.T) {
			wantError(t, config(provider, "model"), ErrCreatorNotServed, provider)
		})
	}
}

func TestPreservesExplicitGatewayID(t *testing.T) {
	wantID(t, config("Bedrock", "amazon/nova-pro"), "amazon/nova-pro")
}

func TestRemovesRegionAndDottedCreatorPrefix(t *testing.T) {
	wantID(t, config("Bedrock", "us.anthropic.claude-sonnet-4"), "anthropic/claude-sonnet-4")
}

func TestFailsForMultiCreatorHostWithoutKnownCreator(t *testing.T) {
	wantError(t, config("Databricks", "custom-deployment"), ErrMultipleCreators, `"custom-deployment"`)
}

// Behaviour of the TypeScript and Python mappings that their test files do
// not cover.

func TestKeepsVersionDotsAfterTheCreator(t *testing.T) {
	wantID(t, config("Bedrock", "openai.gpt-5.6-sol"), "openai/gpt-5.6-sol")
}

func TestModelFamilyWinsOverProvider(t *testing.T) {
	wantID(t, config("Perplexity", "llama-3.1-sonar-large-128k-online"), "meta/llama-3.1-sonar-large-128k-online")
	wantID(t, config("OpenAI", "Claude-Sonnet-4"), "anthropic/Claude-Sonnet-4") // family match ignores case
}

// A multi-creator host gives no creator, so these resolve only by the model
// family. One row per family word.
func TestModelFamilyNamesCreatorOnMultiCreatorHost(t *testing.T) {
	cases := []struct{ model, want string }{
		{"gpt", "openai/gpt"},
		{"gpt-5", "openai/gpt-5"},
		{"o1", "openai/o1"},
		{"o3", "openai/o3"},
		{"o3.5", "openai/o3.5"},
		{"O3", "openai/O3"},
		{"o9", "openai/o9"},
		{"o1-mini", "openai/o1-mini"},
		{"claude-x", "anthropic/claude-x"},
		{"gemini-2.5-pro", "google/gemini-2.5-pro"},
		{"grok-4", "spacexai/grok-4"},
		{"command-r", "cohere/command-r"},
		{"aya-x", "cohere/aya-x"},
		{"deepseek-r1", "deepseek/deepseek-r1"},
		{"llama-4", "meta/llama-4"},
		{"mistral-large", "mistral/mistral-large"},
		{"mixtral-8x7b", "mistral/mixtral-8x7b"},
		{"codestral-x", "mistral/codestral-x"},
		{"pixtral-x", "mistral/pixtral-x"},
		{"sonar-pro", "perplexity/sonar-pro"},
		{"nova-pro", "amazon/nova-pro"},
		{"titan-x", "amazon/titan-x"},
		{"qwen-3", "alibaba/qwen-3"},
	}
	for _, c := range cases {
		t.Run(c.model, func(t *testing.T) {
			wantID(t, config("Databricks", c.model), c.want)
		})
	}
}

// A family word must be followed by "-", "." or the end of the name.
func TestModelFamilyNeedsAWordBoundary(t *testing.T) {
	for _, model := range []string{"o10", "o0", "o", "omni-x", "gpt5"} {
		t.Run(model, func(t *testing.T) {
			wantError(t, config("Databricks", model), ErrMultipleCreators, `"`+model+`"`)
		})
	}
}

// Dotted segments are normalised like provider names, and the last segment is
// the model itself, so it never names the creator.
func TestDottedSegments(t *testing.T) {
	wantID(t, config("Bedrock", "US.Anthropic.claude-x"), "anthropic/claude-x")
	wantID(t, config("OpenAI", "anthropic"), "openai/anthropic")
}

// The first creator segment wins and the rest stay in the model, as in
// model-id.ts:98-103 and model_id.py:102-107 (scan left to right, return).
func TestFirstCreatorSegmentWins(t *testing.T) {
	wantID(t, config("Bedrock", "openai.anthropic.x"), "openai/anthropic.x")
}

func TestFailsForBedrockWithoutKnownCreator(t *testing.T) {
	wantError(t, config("Bedrock", "custom-model"), ErrMultipleCreators, `"custom-model"`)
}

// Each of the three errors quotes the provider (and, for a multi-creator
// host, the model) and gives its own fix. A creator the Gateway does not serve
// cannot be fixed with a creator/model id, so that error does not suggest one.
func TestErrorsQuoteNamesAndSayHowToFix(t *testing.T) {
	cases := []struct {
		provider, model string
		sentinel        error
		says            []string
		doesNotSay      string
	}{
		{"AI21 Labs", "model", ErrCreatorNotServed, []string{`"AI21 Labs"`, "inject a direct provider model"}, "creator/model"},
		{"IBM Watson", "model", ErrCreatorNotServed, []string{`"IBM Watson"`, "inject a direct provider model"}, "creator/model"},
		{"ai21-labs", "model", ErrCreatorNotServed, []string{`"ai21-labs"`, "inject a direct provider model"}, "creator/model"},
		{"Bedrock", "custom-model", ErrMultipleCreators, []string{`"Bedrock"`, `"custom-model"`, "explicit creator/model id", "inject a model"}, ""},
		{"Cortex", "custom-model", ErrMultipleCreators, []string{`"Cortex"`, `"custom-model"`, "explicit creator/model id", "inject a model"}, ""},
		{"Cursor", "custom-model", ErrMultipleCreators, []string{`"Cursor"`, `"custom-model"`, "explicit creator/model id", "inject a model"}, ""},
		{"Databricks", "custom-model", ErrMultipleCreators, []string{`"Databricks"`, `"custom-model"`, "explicit creator/model id", "inject a model"}, ""},
		{"Fireworks AI", "custom-model", ErrMultipleCreators, []string{`"Fireworks AI"`, `"custom-model"`, "explicit creator/model id", "inject a model"}, ""},
		{"Acme", "model", ErrUnknownProvider, []string{`"Acme"`, "creator/model id in config.model.name", "inject a model"}, ""},
		{"", "model", ErrUnknownProvider, []string{`"unknown"`, "creator/model id in config.model.name", "inject a model"}, ""},
	}
	for _, c := range cases {
		t.Run(c.provider, func(t *testing.T) {
			got, err := ID(config(c.provider, c.model))
			if !errors.Is(err, c.sentinel) || got != "" {
				t.Fatalf("ID(%q, %q) = %q, %v; want %v", c.provider, c.model, got, err, c.sentinel)
			}
			for _, want := range c.says {
				if !strings.Contains(err.Error(), want) {
					t.Errorf("ID(%q, %q) error %q does not say %s", c.provider, c.model, err, want)
				}
			}
			if c.doesNotSay != "" && strings.Contains(err.Error(), c.doesNotSay) {
				t.Errorf("ID(%q, %q) error %q says %q", c.provider, c.model, err, c.doesNotSay)
			}
		})
	}
}

// Provider names are lowercased and stripped of non-alphanumerics before the
// lookup, and every alias row maps.
func TestProviderAliases(t *testing.T) {
	cases := []struct{ provider, model, want string }{
		{"Google", "model", "google/model"},
		{"Google AI", "model", "google/model"},
		{"Mistral AI", "model", "mistral/model"},
		{"mistral-ai", "model", "mistral/model"},
		{"xAI", "model", "spacexai/model"},
		{"SpaceX AI", "model", "spacexai/model"},
		{"TypeSafe", "jev", "typesafe-ai/jev"},
		{"TypeSafe AI", "jev", "typesafe-ai/jev"},
	}
	for _, c := range cases {
		t.Run(c.provider, func(t *testing.T) {
			wantID(t, config(c.provider, c.model), c.want)
		})
	}
}

func TestFailsForUnknownProvider(t *testing.T) {
	wantError(t, config("Acme", "model"), ErrUnknownProvider, `"Acme"`)
	wantError(t, config("", "model"), ErrUnknownProvider, `"unknown"`)
}

func wantID(t *testing.T, cfg aiserver.AiConfig, want string) {
	t.Helper()
	got, err := ID(cfg)
	if err != nil || got != want {
		t.Fatalf("ID(%q, %q) = %q, %v; want %q", cfg.Provider.Name, cfg.Model.Name, got, err, want)
	}
}

// wantError checks ID fails with sentinel and an error message naming the
// offending value, as the TypeScript and Python messages do.
func wantError(t *testing.T, cfg aiserver.AiConfig, sentinel error, names string) {
	t.Helper()
	got, err := ID(cfg)
	if !errors.Is(err, sentinel) || !strings.Contains(err.Error(), names) {
		t.Fatalf("ID(%q, %q) = %q, %v; want %v naming %s", cfg.Provider.Name, cfg.Model.Name, got, err, sentinel, names)
	}
	if got != "" {
		t.Fatalf("ID(%q, %q) returned id %q with its error", cfg.Provider.Name, cfg.Model.Name, got)
	}
}
