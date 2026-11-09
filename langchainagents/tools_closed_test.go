package langchainagents

import (
	"context"
	"errors"
	"strconv"
	"strings"
	"testing"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

func TestToolsFailClosed(t *testing.T) {
	for _, path := range []string{"invoke", "stream"} {
		for _, name := range []string{"ghost", "privileged"} {
			t.Run(path+"/"+name, func(t *testing.T) {
				turn := toolTurn(name, nil)
				if path == "stream" {
					turn = streamTurn(nil, 1, 1, toolCall(name, nil))
				}
				h, _, _ := setup(t, turn)
				c := instr("I")
				c.Tools = map[string]aiserver.Tool{"safe": {Name: "safe", Type: "function", Parameters: map[string]any{"type": "object"}}}
				ran := false
				req := aiserver.HandlerRequest{Config: c, UserInput: "q", ToolHandlers: map[string]aiserver.ToolHandler{
					"safe":       aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) { return "ok", nil }),
					"privileged": aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) { ran = true; return "secret", nil }),
				}}
				var err error
				if path == "invoke" {
					_, err = h.Invoke(bg, req)
				} else {
					_, err = collect(t, h, req)
				}
				if err == nil || !errors.Is(err, ErrNoToolHandler) || !strings.Contains(err.Error(), strconv.Quote(name)) || ran {
					t.Errorf("err=%v ran=%v", err, ran)
				}
			})
		}
	}
}
