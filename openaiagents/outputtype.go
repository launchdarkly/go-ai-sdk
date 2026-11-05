package openaiagents

import (
	"maps"
	"slices"
)

// outputType is the JSON-schema envelope an agent's structured output is
// declared with (the Agents SDK's JsonSchemaDefinition). It is sent
// as the Responses API `text.format`.
type outputType struct {
	// Type is always "json_schema".
	Type string `json:"type"`
	// Name is always "output".
	Name string `json:"name"`
	// Strict is always false.
	Strict bool `json:"strict"`
	// Schema is the normalized JSON Schema.
	Schema map[string]any `json:"schema"`
}

// buildOutputType converts a variation's raw outputFormat JSON Schema into an
// outputType. It returns nil when outputFormat is nil or empty. The schema
// root always gets `type: "object"`. When `properties` is an object,
// `required` is replaced by every property key (sorted), because OpenAI
// Structured Outputs requires every property to be listed; optional fields
// should use a nullable type instead. When `properties` is empty, `required`
// is omitted. The input map is not modified.
func buildOutputType(outputFormat map[string]any) *outputType {
	if len(outputFormat) == 0 {
		return nil
	}
	schema := maps.Clone(outputFormat)
	schema["type"] = "object"
	if props, ok := outputFormat["properties"].(map[string]any); ok {
		if len(props) > 0 {
			schema["required"] = slices.Sorted(maps.Keys(props))
		} else {
			delete(schema, "required")
		}
	}
	return &outputType{Type: "json_schema", Name: "output", Strict: false, Schema: schema}
}
