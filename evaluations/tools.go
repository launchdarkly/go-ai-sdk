package evaluations

import (
	"context"
	"fmt"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

// ToolsClient reads tools from the LaunchDarkly tool library. Get one with
// EvaluationsModule.Tools.
type ToolsClient struct{ m *EvaluationsModule }

// Tools returns the client that reads library tools for this module's
// project.
func (m *EvaluationsModule) Tools() *ToolsClient { return &ToolsClient{m: m} }

// Get reads library tool key now, pins the version it returns and pairs it
// with impl. Pass the result in RunOptions.Tools; Run then issues no tool
// request. Get returns ErrInvalidArgument for a blank or uppercase key or a
// nil impl (before any request), ErrNotFound when the tool does not exist in
// the project, and ErrInvalidResponse when it has no integer version.
func (c *ToolsClient) Get(ctx context.Context, key string, impl aiserver.ToolHandler) (EvalTool, error) {
	if err := validateToolKey(key); err != nil {
		return EvalTool{}, err
	}
	if err := validateToolImplementation(key, impl); err != nil {
		return EvalTool{}, err
	}
	project := c.m.projectKey
	raw, err := c.m.api.get(ctx, "projects/"+segment(project)+"/ai-tools/"+segment(key))
	if err != nil {
		return EvalTool{}, notFound(err, "LaunchDarkly AI tool %q was not found in project %q", key, project)
	}
	obj, err := asObject(raw, fmt.Sprintf("tool %q", key))
	if err != nil {
		return EvalTool{}, err
	}
	version, ok := asInt(obj["version"])
	if !ok {
		return EvalTool{}, fmt.Errorf("evaluations: %w: LaunchDarkly AI tool %q has no integer version", ErrInvalidResponse, key)
	}
	schema, _ := obj["schema"].(map[string]any)
	if schema == nil {
		schema = map[string]any{}
	}
	description, _ := obj["description"].(string)
	pin := &libraryPin{key: key, description: description, schema: schema, version: version, projectKey: project}
	clone, _ := cloneJSON(schema).(map[string]any)
	return EvalTool{Key: key, Implementation: impl, Schema: clone, Description: description, library: pin}, nil
}
