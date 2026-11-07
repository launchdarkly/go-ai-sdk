package claudeagents

import (
	"context"
	"errors"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

func TestToolsFailClosed(t *testing.T) {
	for _, path := range []string{"invoke", "stream"} {
		for _, name := range []string{"ghost", "privileged"} {
			t.Run(path+"/"+name, func(t *testing.T) {
				turn := toolResp(name, `{}`)
				if path == "stream" {
					turn = sse(nil, name, 1, 1)
				}
				e := setup(t, nil, turn)
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
					_, err = e.handler().Invoke(bg, req)
				} else {
					for _, serr := range e.handler().Stream(bg, req) {
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

func TestUnattachedNativeToolNotSent(t *testing.T) {
	handlers := map[string]aiserver.ToolHandler{"reader": ClaudeRead, "shell": ClaudeBash}
	check := func(t *testing.T, req map[string]any) {
		t.Helper()
		if got := toolNames(req); !slices.Contains(got, "Read") || slices.Contains(got, "Bash") {
			t.Errorf("tools sent = %v", got)
		}
	}

	t.Run("handler", func(t *testing.T) {
		e := setup(t, nil, textResp("ok", 1, 1))
		c := cfg(func(c *aiserver.AiConfig) {
			c.Tools = map[string]aiserver.Tool{"reader": {Name: "reader", Type: "function", Parameters: map[string]any{}}}
		})
		if _, err := e.handler().Invoke(bg, aiserver.HandlerRequest{Config: c, UserInput: "q", ToolHandlers: handlers}); err != nil {
			t.Fatal(err)
		}
		check(t, e.api.reqs()[0])
	})

	t.Run("graph", func(t *testing.T) {
		e := setup(t, graphFlags(map[string]any{"tools": readerTools()}), textResp("done", 1, 1))
		if _, err := ToClaudeAgents(resolve(t), &AdapterOptions{Handler: e.opts, ToolHandlers: handlers}).Invoke(bg, "q"); err != nil {
			t.Fatal(err)
		}
		check(t, e.api.reqs()[0])
	})
}
