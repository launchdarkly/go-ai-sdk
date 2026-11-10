package openaimessages

import (
	"strconv"
	"strings"
	"testing"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

func TestToolsFailClosed(t *testing.T) {
	for _, path := range []string{"invoke", "stream"} {
		for _, name := range []string{"ghost", "privileged"} {
			t.Run(path+"/"+name, func(t *testing.T) {
				resp := toolResp("r1", name, `{}`)
				if path == "stream" {
					resp = sse(nil, response("r1", callOutput(name, `{}`), 1, 0, 1))
				}
				h, _, _ := setup(t, resp)
				ran := false
				req := aiserver.HandlerRequest{
					Config: cfg(func(c *aiserver.AiConfig) {
						c.Tools = map[string]aiserver.Tool{"safe": funcTool("", map[string]any{"type": "object"})}
					}),
					UserInput: "q",
					ToolHandlers: map[string]aiserver.ToolHandler{
						"safe":       tool(func(map[string]any) (any, error) { return "ok", nil }),
						"privileged": tool(func(map[string]any) (any, error) { ran = true; return "ok", nil }),
					},
				}
				var err error
				if path == "invoke" {
					_, err = h.Invoke(bg, req)
				} else {
					for _, e := range h.Stream(bg, req) {
						if e != nil {
							err = e
							break
						}
					}
				}
				if err == nil {
					t.Fatal("want error, got nil")
				}
				if !strings.Contains(err.Error(), strconv.Quote(name)) {
					t.Errorf("err = %v", err)
				}
				if ran {
					t.Error("unoffered tool ran")
				}
			})
		}
	}
}
