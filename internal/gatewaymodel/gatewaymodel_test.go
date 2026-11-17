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

func TestProviderAliases(t *testing.T) {
	wantID(t, config("xAI", "model"), "spacexai/model")
	wantID(t, config("TypeSafe", "jev"), "typesafe-ai/jev")
	wantID(t, config("Google AI", "model"), "google/model")
	wantID(t, config("mistral-ai", "model"), "mistral/model")
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
