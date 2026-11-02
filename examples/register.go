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
	handlers := []*aiserver.ProviderHandler{
		openaimessages.NewHandler(openaimessages.Options{}),
		openaiagents.NewHandler(openaiagents.Options{}),
		claudemessages.NewHandler(claudemessages.Options{}),
		claudeagents.NewHandler(claudeagents.Options{}),
		langchainmessages.NewHandler(langchainmessages.Options{}),
		// {*, agent}: serves agent configs no provider-specific handler claims.
		langchainagents.NewHandler(langchainagents.Options{}),
	}
	// Bedrock handlers need an AWS configuration; skip them when none loads.
	if h, err := bedrockmessages.NewHandler(bedrockmessages.Options{}); err == nil {
		handlers = append(handlers, h)
	} else {
		fmt.Fprintln(os.Stderr, "bedrock-messages not registered:", err)
	}
	if h, err := bedrockagents.NewHandler(bedrockagents.Options{}); err == nil {
		handlers = append(handlers, h)
	} else {
		fmt.Fprintln(os.Stderr, "bedrock-agents not registered:", err)
	}
	registry = aiserver.NewRegistry(aiserver.RegistryEntries{Handlers: handlers, Tools: exampleTools})
}
