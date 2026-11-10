package main

import (
	"fmt"
	"os"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/bedrockagents"
	"github.com/launchdarkly/go-ai-sdk/bedrockmessages"
	"github.com/launchdarkly/go-ai-sdk/claudeagents"
	"github.com/launchdarkly/go-ai-sdk/claudemessages"
	"github.com/launchdarkly/go-ai-sdk/langchainagents"
	"github.com/launchdarkly/go-ai-sdk/langchainmessages"
	"github.com/launchdarkly/go-ai-sdk/openaiagents"
	"github.com/launchdarkly/go-ai-sdk/openaimessages"
)

// exampleTools are keyed exactly as the LaunchDarkly flag variations name
// them; aliases cover both spellings a flag may use.
var exampleTools = map[string]aiserver.ToolHandler{
	"web-search":                       claudeagents.ClaudeWebSearch,
	"get-user-preferences":             aiserver.ToolFunc(getPreferences),
	"user-preferences-lookup":          aiserver.ToolFunc(getPreferences),
	"search-ld-documentation":          aiserver.ToolFunc(searchLDDocumentation),
	"fetch-ld-documentation":           aiserver.ToolFunc(fetchLDDocumentation),
	"fetch-launchdarkly-documentation": aiserver.ToolFunc(fetchLDDocumentation),
}

// registry holds every handler in this module and every example tool.
var registry *aiserver.Registry

func register() {
	var handlers []*aiserver.ProviderHandler
	// add registers a handler, or reports why it could not be built.
	add := func(name string) func(*aiserver.ProviderHandler, error) {
		return func(h *aiserver.ProviderHandler, err error) {
			if err != nil {
				fmt.Fprintln(os.Stderr, name, "not registered:", err)
				return
			}
			handlers = append(handlers, h)
		}
	}
	add("openai-messages")(openaimessages.NewHandler(openaimessages.Options{}))
	add("openai-agents")(openaiagents.NewHandler(openaiagents.Options{}))
	add("claude-messages")(claudemessages.NewHandler(claudemessages.Options{}))
	add("claude-agents")(claudeagents.NewHandler(claudeagents.Options{}))
	add("langchain-messages")(langchainmessages.NewHandler(langchainmessages.Options{}))
	// {*, agent}: serves agent configs no provider-specific handler claims.
	add("langchain-agents")(langchainagents.NewHandler(langchainagents.Options{}))
	// Bedrock handlers need an AWS configuration; they are skipped when none loads.
	add("bedrock-messages")(bedrockmessages.NewHandler(bedrockmessages.Options{}))
	add("bedrock-agents")(bedrockagents.NewHandler(bedrockagents.Options{}))
	registry = aiserver.NewRegistry(aiserver.RegistryEntries{Handlers: handlers, Tools: exampleTools})
}
