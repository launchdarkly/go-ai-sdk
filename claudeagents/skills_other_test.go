//go:build !unix

package claudeagents

import (
	"errors"
	"testing"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/skills"
)

// Materialization is unix-only, so a variation with skills fails closed
// before any model request.
func TestSkillsFailClosedOffUnix(t *testing.T) {
	e := setup(t, nil, textResp("must not run", 1, 1))
	skills.SetSkillStore(skills.NewInMemorySkillStore(nil))
	h, err := NewSkillsHandler(SkillsHandlerOptions{Options: e.opts, SkillsTempDir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	c := cfg(func(c *aiserver.AiConfig) {
		c.Raw = map[string]any{"skills": []any{map[string]any{"key": "a", "version": 1}}}
	})
	if _, err := h.Invoke(bg, aiserver.HandlerRequest{Config: c, UserInput: "q"}); !errors.Is(err, skills.ErrUnsupportedPlatform) {
		t.Errorf("err = %v, want skills.ErrUnsupportedPlatform", err)
	}
	if n := len(e.api.reqs()); n != 0 {
		t.Errorf("%d model requests, want 0", n)
	}
}
