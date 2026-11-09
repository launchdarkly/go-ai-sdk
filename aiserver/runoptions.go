package aiserver

import (
	"context"
	"encoding/json"
	"maps"

	"github.com/launchdarkly/go-ai-sdk/internal/bridge"
)

func init() {
	bridge.ConfigWithOptions = func(args any, opts bridge.RunOptions) any {
		return &AIConfig{args: args.(ConfigArgs), opts: opts}
	}
	bridge.GraphWithOptions = func(key string, opts any, run bridge.RunOptions) any {
		return &AgentGraph{key: key, opts: opts.(GraphOptions), noUsage: run.NoUsage}
	}
	bridge.ConversationID = conversationID
	bridge.WithExactConversationID = func(ctx context.Context, id string) context.Context {
		return context.WithValue(ctx, conversationKey{}, id)
	}
	bridge.TrackDataFor = func(configKey string, config, meta any, runID string, extra map[string]any) map[string]any {
		td := newTrackData(configKey, config.(AiConfig), meta.(VariationMeta), "")
		td.RunID = runID
		var m map[string]any
		b, _ := json.Marshal(td)
		_ = json.Unmarshal(b, &m)
		maps.Copy(m, extra)
		return m
	}
}
