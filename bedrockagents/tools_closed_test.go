package bedrockagents

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"testing"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

func TestToolsFailClosed(t *testing.T) {
	for _, path := range []string{"invoke", "stream"} {
		for _, name := range []string{"ghost", "privileged"} {
			t.Run(path+"/"+name, func(t *testing.T) {
				turn := toolReply(name, `{}`)
				if path == "stream" {
					turn = frames(t, messageStart,
						[2]string{"contentBlockStart", fmt.Sprintf(`{"contentBlockIndex":0,"start":{"toolUse":{"toolUseId":"t1","name":%q}}}`, name)},
						[2]string{"contentBlockDelta", `{"contentBlockIndex":0,"delta":{"toolUse":{"input":"{}"}}}`},
						blockStop(0), messageStop("tool_use"), metadata(1, 1, 0, 0))
				}
				h, _, _ := setup(t, Options{}, turn)
				c := cfg(func(c *aiserver.AiConfig) {
					c.Tools = map[string]aiserver.Tool{"safe": {Name: "safe", Type: "function", Parameters: map[string]any{}}}
				})
				ran := false
				req := aiserver.HandlerRequest{Config: c, UserInput: "q", ToolHandlers: map[string]aiserver.ToolHandler{
					"safe":       aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) { return "ok", nil }),
					"privileged": aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) { ran = true; return "secret", nil }),
				}}
				var err error
				if path == "invoke" {
					_, err = h.Invoke(bg, req)
				} else {
					for _, serr := range h.Stream(bg, req) {
						if serr != nil {
							err = serr
							break
						}
					}
				}
				if err == nil || !errors.Is(err, ErrToolNotFound) || !strings.Contains(err.Error(), strconv.Quote(name)) || ran {
					t.Errorf("err=%v ran=%v", err, ran)
				}
			})
		}
	}
}
