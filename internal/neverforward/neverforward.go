// Package neverforward holds the model.parameters keys no handler may
// forward, for handler tests. Every value carries Marker, so a test can check
// the whole outgoing request for it, renamed or nested keys included.
package neverforward

// Marker appears in every value of Params.
const Marker = "never-forwarded"

// Params has a key from every category that is never forwarded: credentials,
// endpoints and connection settings, request injection, remote tools, data
// location and retention, server-side state, identity and attribution,
// handler wiring, prompt content, and safety configuration.
var Params = map[string]any{
	// credentials
	"api_key": Marker, "apiKey": Marker, "anthropic_api_key": Marker, "openai_api_key": Marker,
	"aws_access_key_id": Marker, "aws_secret_access_key": Marker, "aws_session_token": Marker, "credentials": Marker,
	// endpoints and connection settings
	"base_url": Marker, "baseURL": Marker, "anthropic_api_url": Marker, "endpoint_url": Marker, "organization": Marker,
	"timeout": Marker, "request_timeout": Marker, "max_retries": Marker, "default_headers": map[string]any{"x": Marker},
	"default_query": map[string]any{"x": Marker},
	// request injection
	"headers": map[string]any{"x": Marker}, "extra_headers": map[string]any{"x": Marker},
	"extra_body": map[string]any{"x": Marker}, "extra_query": map[string]any{"x": Marker},
	"model_kwargs": map[string]any{"x": Marker}, "provider_data": map[string]any{"x": Marker},
	// remote tool attachment
	"mcp_servers": []any{map[string]any{"type": "url", "url": Marker, "name": Marker}},
	// data location and retention
	"region": Marker, "inference_geo": Marker, "store": Marker, "zdr_enabled": Marker, "prompt_cache_retention": Marker,
	// server-side state
	"container": Marker, "reuse_last_container": Marker, "previous_response_id": Marker, "response_id": Marker,
	"conversation": Marker,
	// identity and attribution
	"user": Marker, "safety_identifier": Marker, "metadata": map[string]any{"user_id": Marker},
	"request_metadata": map[string]any{"x": Marker},
	// handler and runtime wiring
	"streaming": Marker, "stream_usage": Marker, "include_usage": Marker, "include": []any{Marker},
	"truncation": Marker, "context_management": []any{map[string]any{"type": Marker}}, "modalities": []any{Marker},
	// prompt content beyond the config's instructions
	"prefix_messages": []any{Marker}, "instructions": Marker,
	// safety configuration
	"guardrails": Marker, "guardrail_config": map[string]any{"x": Marker}, "moderation": Marker,
	// a forwarded key with a malformed nested value: a tool_choice with no
	// string type, which every handler drops
	"tool_choice": map[string]any{"name": Marker, "tools": []any{map[string]any{"type": "mcp", "server_url": Marker}}},
}
