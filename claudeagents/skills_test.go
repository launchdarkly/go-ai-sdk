//go:build unix

package claudeagents

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthropics/anthropic-sdk-go"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/skills"
)

// storedSkill is a valid raw skill object for an InMemorySkillStore.
func storedSkill(key string, version int, content string) map[string]any {
	sum := sha256.Sum256([]byte(content))
	return map[string]any{"key": key, "version": version, "content": content, "contentHash": hex.EncodeToString(sum[:]),
		"name": key, "description": "about " + key}
}

// skillStore holds a@1..3 and b@1, so a pin can be told apart from "latest".
func skillStore(t *testing.T) *skills.InMemorySkillStore {
	t.Helper()
	s := skills.NewInMemorySkillStore(nil)
	for _, raw := range []map[string]any{
		storedSkill("a", 1, "a v1"), storedSkill("a", 2, "a v2"), storedSkill("a", 3, "a v3"), storedSkill("b", 1, "b v1"),
	} {
		if err := s.Put(raw); err != nil {
			t.Fatal(err)
		}
	}
	skills.SetSkillStore(s)
	return s
}

type ref struct {
	key     string
	version int
}

func refsValue(refs []ref) []any {
	out := []any{}
	for _, r := range refs {
		out = append(out, map[string]any{"key": r.key, "version": r.version})
	}
	return out
}

// skillsCfg is an Anthropic agent config referencing refs (nil: no "skills").
func skillsCfg(refs []ref) aiserver.AiConfig {
	return cfg(func(c *aiserver.AiConfig) {
		c.Instructions = "inst"
		if refs != nil {
			c.Raw = map[string]any{"skills": refsValue(refs)}
		}
	})
}

func (e *env) skillsHandler(t *testing.T, parent string) *aiserver.ProviderHandler {
	t.Helper()
	h, err := NewSkillsHandler(SkillsHandlerOptions{Options: e.opts, SkillsTempDir: parent})
	if err != nil {
		t.Fatal(err)
	}
	return h
}

// invokeDirs lists the per-invoke directories under parent.
func invokeDirs(t *testing.T, parent string) []string {
	t.Helper()
	m, err := filepath.Glob(filepath.Join(parent, "ld-skills-*"))
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// skillFiles maps key to SKILL.md content in one invoke directory.
func skillFiles(t *testing.T, dir string) map[string]string {
	t.Helper()
	out := map[string]string{}
	m, _ := filepath.Glob(filepath.Join(dir, "launchdarkly", "skills", "*", "SKILL.md"))
	for _, p := range m {
		b, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		out[filepath.Base(filepath.Dir(p))] = string(b)
	}
	return out
}

func toolByName(req map[string]any, name string) map[string]any {
	tools, _ := req["tools"].([]any)
	for _, t := range tools {
		if m := t.(map[string]any); m["name"] == name {
			return m
		}
	}
	return nil
}

// lastToolResult returns the text and is_error of the last tool_result the
// request carries back to the model.
func lastToolResult(t *testing.T, req map[string]any) (string, bool) {
	t.Helper()
	msgs := messages(req)
	for _, b := range msgs[len(msgs)-1]["content"].([]any) {
		blk := b.(map[string]any)
		if blk["type"] != "tool_result" {
			continue
		}
		isErr, _ := blk["is_error"].(bool)
		switch c := blk["content"].(type) {
		case string:
			return c, isErr
		case []any:
			var sb strings.Builder
			for _, x := range c {
				if m, _ := x.(map[string]any); m["type"] == "text" {
					sb.WriteString(m["text"].(string))
				}
			}
			return sb.String(), isErr
		}
	}
	t.Fatal("no tool_result in the request")
	return "", false
}

// run invokes or streams req through h, consuming the whole stream.
func run(h *aiserver.ProviderHandler, path string, req aiserver.HandlerRequest) error {
	if path == "invoke" {
		_, err := h.Invoke(bg, req)
		return err
	}
	for _, err := range h.Stream(bg, req) {
		if err != nil {
			return err
		}
	}
	return nil
}

func reply(path, text string) string {
	if path == "stream" {
		return sse([]string{text}, "", 1, 1)
	}
	return textResp(text, 1, 1)
}

var paths = []string{"invoke", "stream"}

func TestSkillsHandlerRouting(t *testing.T) {
	h, err := NewSkillsHandler(SkillsHandlerOptions{Options: Options{CaptureContent: aiserver.CaptureOff}})
	if err != nil {
		t.Fatal(err)
	}
	if h.ProvidesFor != (aiserver.ProvidesFor{Provider: "Anthropic", Mode: aiserver.ModeAgent}) || h.CaptureContent != aiserver.CaptureOff ||
		h.Invoke == nil || h.Stream == nil {
		t.Errorf("handler = %+v", h)
	}
}

// The factory reports its own ID once and nothing else: not the core
// handler's, and no client.* skills ID for the accessors it calls.
func TestSkillsHandlerUsage(t *testing.T) {
	e := setup(t, nil, textResp("a", 1, 1))
	skillStore(t)
	h := e.skillsHandler(t, t.TempDir())
	e.skillsHandler(t, t.TempDir())
	if err := run(h, "invoke", aiserver.HandlerRequest{Config: skillsCfg([]ref{{"a", 2}}), UserInput: "q"}); err != nil {
		t.Fatal(err)
	}
	var got []string
	for _, id := range e.ld.helpers() {
		if strings.HasPrefix(id, "claude-agents.") || strings.HasPrefix(id, "client.") {
			got = append(got, id)
		}
	}
	if want := []string{"claude-agents.createClaudeAgentsSkillsHandler"}; !reflect.DeepEqual(got, want) {
		t.Errorf("usage IDs = %v, want %v", got, want)
	}
	ev := e.ld.events("$ld:ai:sdk:usage")
	if len(ev) == 0 || ev[0].Data["helperPackageName"] != PackageName {
		t.Errorf("usage payload = %+v", ev)
	}
}

// With no skills the run is the core handler's: same request, nothing
// created, no store needed.
func TestSkillsNoSkills(t *testing.T) {
	for _, path := range paths {
		for name, refs := range map[string][]ref{"absent": nil, "empty": {}} {
			t.Run(path+"/"+name, func(t *testing.T) {
				e := setup(t, nil, reply(path, "core"), reply(path, "skills"))
				parent := t.TempDir()
				c := skillsCfg(refs)
				c.Tools = map[string]aiserver.Tool{"safe": {Name: "safe", Type: "function", Parameters: map[string]any{}}}
				req := aiserver.HandlerRequest{Config: c, UserInput: "q", ToolHandlers: map[string]aiserver.ToolHandler{
					"safe": aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) { return "ok", nil }),
				}}
				if err := run(e.handler(), path, req); err != nil {
					t.Fatal(err)
				}
				if err := run(e.skillsHandler(t, parent), path, req); err != nil {
					t.Fatalf("no store configured, no skills: %v", err)
				}
				r := e.api.reqs()
				if !reflect.DeepEqual(r[0], r[1]) {
					t.Errorf("request differs from the core handler's\ncore   %v\nskills %v", r[0], r[1])
				}
				if d := invokeDirs(t, parent); len(d) != 0 {
					t.Errorf("created %v", d)
				}
			})
		}
	}
}

// During the run each pinned skill's SKILL.md is in the invoke's directory,
// the Skill tool is offered last and lists exactly the variation's skills,
// and the directory is gone afterwards.
func TestSkillsMaterialized(t *testing.T) {
	for _, path := range paths {
		t.Run(path, func(t *testing.T) {
			e := setup(t, nil, reply(path, "done"))
			skillStore(t)
			parent := t.TempDir()
			var during map[string]string
			var dirs []string
			e.api.during = func(map[string]any) {
				dirs = invokeDirs(t, parent)
				if len(dirs) == 1 {
					during = skillFiles(t, dirs[0])
					if _, err := os.Stat(filepath.Join(dirs[0], "launchdarkly", ".claude-plugin")); !os.IsNotExist(err) {
						t.Error("a .claude-plugin directory was written")
					}
				}
			}
			c := skillsCfg([]ref{{"a", 2}, {"b", 1}})
			c.Tools = map[string]aiserver.Tool{"safe": {Name: "safe", Type: "function", Parameters: map[string]any{}}}
			req := aiserver.HandlerRequest{Config: c, UserInput: "q", ToolHandlers: map[string]aiserver.ToolHandler{
				"safe": aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) { return "ok", nil }),
				"read": ClaudeRead,
			}}
			req.Config.Tools["read"] = aiserver.Tool{Name: "read", Type: "function", Parameters: map[string]any{}}
			if err := run(e.skillsHandler(t, parent), path, req); err != nil {
				t.Fatal(err)
			}
			if len(dirs) != 1 {
				t.Fatalf("invoke dirs during the run = %v, want one", dirs)
			}
			if !strings.HasPrefix(filepath.Base(dirs[0]), "ld-skills-") {
				t.Errorf("dir = %s", dirs[0])
			}
			if want := map[string]string{"a": "a v2", "b": "b v1"}; !reflect.DeepEqual(during, want) {
				t.Errorf("SKILL.md files = %v, want %v (pinned versions)", during, want)
			}
			names := toolNames(e.api.reqs()[0])
			if len(names) == 0 || names[len(names)-1] != "Skill" || slices.Index(names, "Skill") != len(names)-1 ||
				!slices.Contains(names, "safe") || !slices.Contains(names, "Read") {
				t.Errorf("tools = %v, want the config's tools then Skill once, last", names)
			}
			tool := toolByName(e.api.reqs()[0], "Skill")
			desc, _ := tool["description"].(string)
			for _, s := range []string{"launchdarkly:a", "about a", "launchdarkly:b", "about b"} {
				if !strings.Contains(desc, s) {
					t.Errorf("Skill description %q lacks %q", desc, s)
				}
			}
			schema, _ := tool["input_schema"].(map[string]any)
			props, _ := schema["properties"].(map[string]any)
			if _, ok := props["skill"]; !ok || !reflect.DeepEqual(schema["required"], []any{"skill"}) {
				t.Errorf("Skill input_schema = %v, want a required \"skill\"", schema)
			}
			if d := invokeDirs(t, parent); len(d) != 0 {
				t.Errorf("not removed after the run: %v", d)
			}
		})
	}
}

// SkillsTempDir empty means os.TempDir().
func TestSkillsDefaultTempDir(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("TMPDIR", tmp)
	e := setup(t, nil, textResp("done", 1, 1))
	skillStore(t)
	var dirs []string
	e.api.during = func(map[string]any) { dirs = invokeDirs(t, tmp) }
	if err := run(e.skillsHandler(t, ""), "invoke", aiserver.HandlerRequest{Config: skillsCfg([]ref{{"a", 1}}), UserInput: "q"}); err != nil {
		t.Fatal(err)
	}
	if len(dirs) != 1 {
		t.Errorf("dirs under os.TempDir() during the run = %v", dirs)
	}
}

// The directory is removed however the run ends.
func TestSkillsDirRemoved(t *testing.T) {
	cases := []struct {
		name string
		resp string
		run  func(h *aiserver.ProviderHandler, req aiserver.HandlerRequest) error
	}{
		{"api error", "STATUS400", func(h *aiserver.ProviderHandler, req aiserver.HandlerRequest) error { return run(h, "invoke", req) }},
		{"stream api error", "STATUS400", func(h *aiserver.ProviderHandler, req aiserver.HandlerRequest) error { return run(h, "stream", req) }},
		{"stream break", sse([]string{"a", "b", "c"}, "", 1, 1), func(h *aiserver.ProviderHandler, req aiserver.HandlerRequest) error {
			for range h.Stream(bg, req) {
				break
			}
			return nil
		}},
		// A run that ends without success: the step limit is reached.
		{"max turns", toolResp("safe", `{}`), func(h *aiserver.ProviderHandler, req aiserver.HandlerRequest) error {
			return maxTurns(run(h, "invoke", oneTurn(req)))
		}},
		{"stream max turns", sse(nil, "safe", 1, 1), func(h *aiserver.ProviderHandler, req aiserver.HandlerRequest) error {
			return maxTurns(run(h, "stream", oneTurn(req)))
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			e := setup(t, nil, tc.resp)
			skillStore(t)
			parent := t.TempDir()
			created := false
			e.api.during = func(map[string]any) { created = len(invokeDirs(t, parent)) == 1 }
			err := tc.run(e.skillsHandler(t, parent), aiserver.HandlerRequest{Config: skillsCfg([]ref{{"a", 1}}), UserInput: "q"})
			if !created {
				t.Fatal("no invoke dir during the run")
			}
			if err != nil {
				t.Error(err)
			}
			if d := invokeDirs(t, parent); len(d) != 0 {
				t.Errorf("not removed: %v", d)
			}
		})
	}
}

// oneTurn allows one model turn and offers a tool the model calls.
func oneTurn(req aiserver.HandlerRequest) aiserver.HandlerRequest {
	req.Config.Model.Parameters = map[string]any{"max_turns": 1.0}
	req.Config.Tools = map[string]aiserver.Tool{"safe": {Name: "safe", Type: "function", Parameters: map[string]any{}}}
	req.ToolHandlers = map[string]aiserver.ToolHandler{
		"safe": aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) { return "ok", nil }),
	}
	return req
}

// maxTurns is nil when err is the step-limit error, so a case can report
// any other outcome.
func maxTurns(err error) error {
	if errors.Is(err, ErrMaxSteps) {
		return nil
	}
	return fmt.Errorf("err = %v, want ErrMaxSteps", err)
}

// syncBuffer is a bytes.Buffer safe for a logger shared across goroutines.
type syncBuffer struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (s *syncBuffer) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.Write(p)
}

func (s *syncBuffer) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.String()
}

// A directory that cannot be removed is logged, and the run's own outcome
// (its result, or its error) is still what the caller gets.
func TestSkillsRemovalFailureIsLogged(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("root ignores directory permissions")
	}
	for _, tc := range []struct{ name, resp string }{{"success", textResp("done", 1, 1)}, {"api error", "STATUS400"}} {
		t.Run(tc.name, func(t *testing.T) {
			e := setup(t, nil, tc.resp)
			skillStore(t)
			var logs syncBuffer
			aiserver.SetLogger(slog.New(slog.NewTextHandler(&logs, &slog.HandlerOptions{Level: slog.LevelWarn})))
			t.Cleanup(func() { aiserver.SetLogger(nil) })
			parent := t.TempDir()
			t.Cleanup(func() { _ = os.Chmod(parent, 0o700) }) // runs before t.TempDir's removal
			created := false
			e.api.during = func(map[string]any) {
				created = len(invokeDirs(t, parent)) == 1
				// The invoke dir's entry cannot be unlinked from a read-only parent.
				if err := os.Chmod(parent, 0o500); err != nil {
					t.Error(err)
				}
			}
			res, err := e.skillsHandler(t, parent).Invoke(bg, aiserver.HandlerRequest{Config: skillsCfg([]ref{{"a", 1}}), UserInput: "q"})
			if !created {
				t.Fatal("no invoke dir during the run")
			}
			if d := invokeDirs(t, parent); len(d) != 1 {
				t.Fatalf("dirs after the run = %v, want the one that could not be removed", d)
			}
			if tc.name == "success" {
				if err != nil || res.Output != "done" {
					t.Errorf("result = %q, %v; want the run's own result", res.Output, err)
				}
			} else if apiErr := (*anthropic.Error)(nil); !errors.As(err, &apiErr) {
				t.Errorf("err = %v, want the run's own API error", err)
			}
			if logs.String() == "" {
				t.Error("the failed removal was not logged")
			}
		})
	}
}

// Two invokes held open at once get distinct directories, and the first
// finishing does not remove the second's.
func TestSkillsConcurrentInvokes(t *testing.T) {
	e := setup(t, nil, textResp("one", 1, 1), textResp("two", 1, 1))
	skillStore(t)
	parent := t.TempDir()
	var mu sync.Mutex
	inFlight := 0
	both := make(chan struct{})
	var seen []int
	e.api.during = func(map[string]any) {
		mu.Lock()
		inFlight++
		if inFlight == 2 {
			close(both)
		}
		mu.Unlock()
		select {
		case <-both:
		case <-time.After(5 * time.Second):
		}
		mu.Lock()
		seen = append(seen, len(invokeDirs(t, parent)))
		mu.Unlock()
	}
	h := e.skillsHandler(t, parent)
	var wg sync.WaitGroup
	errs := make([]error, 2)
	for i := range 2 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			errs[i] = run(h, "invoke", aiserver.HandlerRequest{Config: skillsCfg([]ref{{"a", 1}}), UserInput: "q"})
		}()
	}
	wg.Wait()
	if errs[0] != nil || errs[1] != nil {
		t.Fatalf("errs = %v", errs)
	}
	if !reflect.DeepEqual(seen, []int{2, 2}) {
		t.Errorf("dirs seen by each in-flight run = %v, want [2 2]", seen)
	}
	if d := invokeDirs(t, parent); len(d) != 0 {
		t.Errorf("not removed: %v", d)
	}
}

// Detaching a skill from the variation is how it is revoked: the next
// invoke does not write it.
func TestSkillsDetachedSkillIsNotWritten(t *testing.T) {
	e := setup(t, nil, textResp("one", 1, 1), textResp("two", 1, 1))
	skillStore(t)
	parent := t.TempDir()
	var files []map[string]string
	e.api.during = func(map[string]any) {
		if d := invokeDirs(t, parent); len(d) == 1 {
			files = append(files, skillFiles(t, d[0]))
		}
	}
	h := e.skillsHandler(t, parent)
	for _, refs := range [][]ref{{{"a", 1}, {"b", 1}}, {{"a", 1}}} {
		if err := run(h, "invoke", aiserver.HandlerRequest{Config: skillsCfg(refs), UserInput: "q"}); err != nil {
			t.Fatal(err)
		}
	}
	want := []map[string]string{{"a": "a v1", "b": "b v1"}, {"a": "a v1"}}
	if !reflect.DeepEqual(files, want) {
		t.Errorf("files per run = %v, want %v", files, want)
	}
}

// notReadyStore is a store whose readiness probe says it has no data yet.
type notReadyStore struct{ *skills.InMemorySkillStore }

func (notReadyStore) IsInitialized() (bool, error) { return false, nil }

// A reconcile that is not ok fails the invoke before any model request,
// names each failed skill, and leaves no directory.
func TestSkillsFailClosed(t *testing.T) {
	tampered := storedSkill("w", 1, "withheld")
	tampered["contentHash"] = strings.Repeat("0", 64)
	cases := []struct {
		name    string
		store   func(t *testing.T)
		refs    []ref
		wantKey []string
		notKey  []string
	}{
		{"no store", func(*testing.T) {}, []ref{{"a", 1}}, []string{"a"}, nil},
		{"store not initialized", func(*testing.T) {
			skills.SetSkillStore(notReadyStore{skills.NewInMemorySkillStore(map[string]any{"a": storedSkill("a", 1, "a v1")})})
		}, []ref{{"a", 1}}, []string{"a"}, nil},
		{"withheld skill", func(t *testing.T) {
			s := skillStore(t)
			if err := s.Put(tampered); err != nil {
				t.Fatal(err)
			}
		}, []ref{{"a", 1}, {"w", 1}}, []string{"w"}, []string{"a v1", "withheld"}},
		{"missing pin", func(t *testing.T) { skillStore(t) }, []ref{{"b", 1}, {"a", 5}}, []string{"a"}, []string{"b v1"}},
	}
	for _, path := range paths {
		for _, tc := range cases {
			t.Run(path+"/"+tc.name, func(t *testing.T) {
				e := setup(t, nil, reply(path, "must not run"))
				tc.store(t)
				actionErrs := failedActions(t, tc.refs)
				parent := t.TempDir()
				err := run(e.skillsHandler(t, parent), path, aiserver.HandlerRequest{Config: skillsCfg(tc.refs), UserInput: "q"})
				if err == nil {
					t.Fatal("want an error")
				}
				for _, msg := range actionErrs {
					if !strings.Contains(err.Error(), msg) {
						t.Errorf("err %q lacks the action's error text %q", err, msg)
					}
				}
				if p := pathOutsideInvokeDir(t, err.Error(), parent); p != "" {
					t.Errorf("err %q contains %q, a path outside the invoke dir", err, p)
				}
				for _, k := range tc.wantKey {
					if !strings.Contains(err.Error(), k) {
						t.Errorf("err %q does not name %q", err, k)
					}
				}
				for _, s := range tc.notKey {
					if strings.Contains(err.Error(), s) {
						t.Errorf("err %q contains skill content %q", err, s)
					}
				}
				if n := len(e.api.reqs()); n != 0 {
					t.Errorf("%d model requests, want 0", n)
				}
				if d := invokeDirs(t, parent); len(d) != 0 {
					t.Errorf("not removed: %v", d)
				}
			})
		}
	}
}

// failedActions writes refs with the public WriteSkills into a scratch root
// and returns each error action's text: what the handler's error must carry.
func failedActions(t *testing.T, refs []ref) []string {
	t.Helper()
	var in []skills.SkillReference
	for _, r := range refs {
		in = append(in, skills.NewSkillReference(r.key, r.version))
	}
	rep, err := skills.WriteSkills(in, t.TempDir(), nil)
	if err != nil {
		t.Fatal(err)
	}
	var out []string
	for _, a := range rep.Actions() {
		if a.Action() == skills.ActionError {
			out = append(out, a.ErrorMessage())
		}
	}
	if len(out) == 0 {
		t.Fatal("WriteSkills reported no error action")
	}
	return out
}

// pathOutsideInvokeDir returns a path under parent, or parent itself, that
// msg names outside an ld-skills-* invoke dir ("" if none). Both the given
// and the resolved spelling of parent are checked.
func pathOutsideInvokeDir(t *testing.T, msg, parent string) string {
	t.Helper()
	real, err := filepath.EvalSymlinks(parent)
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range []string{real, parent} {
		msg = regexp.MustCompile(regexp.QuoteMeta(p)+`/ld-skills-[^/\s'"]+`).ReplaceAllString(msg, "")
	}
	for _, p := range []string{real, parent, os.TempDir()} {
		if strings.Contains(msg, p) {
			return p
		}
	}
	return ""
}

// Through config().invoke a fail-closed throw is an ordinary generation error.
func TestSkillsFailClosedIsAGenerationError(t *testing.T) {
	e := setup(t, map[string]any{"cfg": agentVariation("inst", map[string]any{"skills": refsValue([]ref{{"a", 1}})})}, textResp("x", 1, 1))
	h := e.skillsHandler(t, t.TempDir()) // no store configured
	_, err := aiserver.Config(aiserver.ConfigArgs{Key: "cfg", Handlers: []*aiserver.ProviderHandler{h}}).Invoke(bg, "q", user)
	if err == nil {
		t.Fatal("want an error")
	}
	if n := len(e.ld.events("$ld:ai:generation:error")); n != 1 {
		t.Errorf("generation:error events = %d, want 1", n)
	}
	if n := len(e.api.reqs()); n != 0 {
		t.Errorf("%d model requests, want 0", n)
	}
}

// A Skill call returns the listed skill's SKILL.md, gets an execute_tool
// Skill span under invoke_agent, and sends no $ld:ai:tool_call (a config
// tool called in the same run does).
func TestSkillToolCall(t *testing.T) {
	for _, path := range paths {
		t.Run(path, func(t *testing.T) {
			skillTurn := toolResp("Skill", `{"skill":"launchdarkly:a"}`)
			lookupTurn := toolResp("lookup", `{}`)
			e := setup(t, map[string]any{"cfg": agentVariation("inst", map[string]any{
				"skills": refsValue([]ref{{"a", 2}}),
				"tools":  map[string]any{"lookup": map[string]any{"name": "lookup", "type": "function", "parameters": map[string]any{}}},
			})}, skillTurn, lookupTurn, reply(path, "done"))
			skillStore(t)
			h := e.skillsHandler(t, t.TempDir())
			args := aiserver.ConfigArgs{Key: "cfg", Handlers: []*aiserver.ProviderHandler{h}, ToolHandlers: map[string]aiserver.ToolHandler{
				"lookup": aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) { return "looked", nil }),
			}}
			if path == "invoke" {
				if _, err := aiserver.Config(args).Invoke(bg, "q", user); err != nil {
					t.Fatal(err)
				}
			} else {
				for _, err := range aiserver.Config(args).Stream(bg, "q", user) {
					if err != nil {
						t.Fatal(err)
					}
				}
			}
			r := e.api.reqs()
			if len(r) != 3 {
				t.Fatalf("%d model requests, want 3", len(r))
			}
			if text, isErr := lastToolResult(t, r[1]); text != "a v2" || isErr {
				t.Errorf("Skill result = %q (error %v), want the pinned SKILL.md", text, isErr)
			}
			spans := e.spans("execute_tool Skill")
			root := e.spans(SpanName)
			if len(spans) != 1 || len(root) != 1 {
				t.Fatalf("execute_tool Skill spans = %d, invoke_agent spans = %d", len(spans), len(root))
			}
			if attrs(spans[0])["gen_ai.tool.name"] != "Skill" {
				t.Errorf("span attrs = %v", attrs(spans[0]))
			}
			if spans[0].Parent().SpanID() != root[0].SpanContext().SpanID() {
				t.Error("execute_tool Skill is not a child of invoke_agent")
			}
			var keys []any
			for _, ev := range e.ld.events("$ld:ai:tool_call") {
				keys = append(keys, ev.Data["toolKey"])
			}
			if !reflect.DeepEqual(keys, []any{"lookup"}) {
				t.Errorf("tool_call toolKeys = %v, want only the config tool", keys)
			}
		})
	}
}

// The parent is resolved to its real path before use: a symlinked parent
// retargeted mid-run still serves the skill and removes the right directory,
// and a dangling one fails before any model request.
func TestSkillsTempDirSymlink(t *testing.T) {
	t.Run("retargeted", func(t *testing.T) {
		e := setup(t, nil, toolResp("Skill", `{"skill":"launchdarkly:a"}`), textResp("done", 1, 1))
		skillStore(t)
		base := t.TempDir()
		real1, real2, link := filepath.Join(base, "real1"), filepath.Join(base, "real2"), filepath.Join(base, "link")
		for _, d := range []string{real1, real2} {
			if err := os.Mkdir(d, 0o700); err != nil {
				t.Fatal(err)
			}
		}
		if err := os.Symlink(real1, link); err != nil {
			t.Fatal(err)
		}
		created := false
		e.api.during = func(map[string]any) {
			if created {
				return
			}
			created = len(invokeDirs(t, real1)) == 1
			if err := os.Remove(link); err != nil {
				t.Error(err)
			}
			if err := os.Symlink(real2, link); err != nil {
				t.Error(err)
			}
		}
		if err := run(e.skillsHandler(t, link), "invoke", aiserver.HandlerRequest{Config: skillsCfg([]ref{{"a", 1}}), UserInput: "q"}); err != nil {
			t.Fatal(err)
		}
		if !created {
			t.Fatal("no invoke dir in the symlink's target during the run")
		}
		if r := e.api.reqs(); len(r) == 2 {
			if text, isErr := lastToolResult(t, r[1]); text != "a v1" || isErr {
				t.Errorf("Skill result after the retarget = %q (error %v), want a v1", text, isErr)
			}
		}
		if d := append(invokeDirs(t, real1), invokeDirs(t, real2)...); len(d) != 0 {
			t.Errorf("not removed: %v", d)
		}
	})
	t.Run("dangling", func(t *testing.T) {
		e := setup(t, nil, textResp("must not run", 1, 1))
		skillStore(t)
		base := t.TempDir()
		link := filepath.Join(base, "link")
		if err := os.Symlink(filepath.Join(base, "missing"), link); err != nil {
			t.Fatal(err)
		}
		if err := run(e.skillsHandler(t, link), "invoke", aiserver.HandlerRequest{Config: skillsCfg([]ref{{"a", 1}}), UserInput: "q"}); err == nil {
			t.Error("want an error")
		}
		if n := len(e.api.reqs()); n != 0 {
			t.Errorf("%d model requests, want 0", n)
		}
		if _, err := os.Lstat(filepath.Join(base, "missing")); !os.IsNotExist(err) {
			t.Error("the dangling target was created")
		}
	})
}

// With skills and no other tool the model is offered exactly Skill, and a
// Skill call still gets its execute_tool span under invoke_agent.
func TestSkillToolIsTheOnlyTool(t *testing.T) {
	e := setup(t, nil, toolResp("Skill", `{"skill":"launchdarkly:a"}`), textResp("done", 1, 1))
	skillStore(t)
	if err := run(e.skillsHandler(t, t.TempDir()), "invoke", aiserver.HandlerRequest{Config: skillsCfg([]ref{{"a", 1}}), UserInput: "q"}); err != nil {
		t.Fatal(err)
	}
	r := e.api.reqs()
	if got := toolNames(r[0]); !reflect.DeepEqual(got, []string{"Skill"}) {
		t.Errorf("tools = %v, want [Skill]", got)
	}
	if len(r) != 2 {
		t.Fatalf("%d model requests, want 2", len(r))
	}
	if text, isErr := lastToolResult(t, r[1]); text != "a v1" || isErr {
		t.Errorf("Skill result = %q (error %v)", text, isErr)
	}
	spans, root := e.spans("execute_tool Skill"), e.spans(SpanName)
	if len(spans) != 1 || len(root) != 1 {
		t.Fatalf("execute_tool Skill spans = %d, invoke_agent spans = %d", len(spans), len(root))
	}
	if spans[0].Parent().SpanID() != root[0].SpanContext().SpanID() {
		t.Error("execute_tool Skill is not a child of invoke_agent")
	}
}

// The handler owns the tool list: model.parameters.tools does not replace
// or extend it, and Skill is still offered last.
func TestSkillsParameterToolsIgnored(t *testing.T) {
	e := setup(t, nil, textResp("done", 1, 1))
	skillStore(t)
	c := skillsCfg([]ref{{"a", 1}})
	c.Model.Parameters = map[string]any{"tools": []any{map[string]any{"name": "injected", "input_schema": map[string]any{"type": "object"}}}}
	if err := run(e.skillsHandler(t, t.TempDir()), "invoke", aiserver.HandlerRequest{Config: c, UserInput: "q"}); err != nil {
		t.Fatal(err)
	}
	if got := toolNames(e.api.reqs()[0]); !reflect.DeepEqual(got, []string{"Skill"}) {
		t.Errorf("tools = %v, want [Skill]", got)
	}
}

// Only a listed qualified name is served. Anything else is a tool error and
// returns no file content.
func TestSkillToolRejectsUnlistedNames(t *testing.T) {
	for _, name := range []string{"launchdarkly:b", "a", "launchdarkly:../a", "launchdarkly:a/../a", "other:a", ""} {
		t.Run(fmt.Sprintf("%q", name), func(t *testing.T) {
			e := setup(t, nil, toolResp("Skill", fmt.Sprintf(`{"skill":%q}`, name)), textResp("done", 1, 1))
			skillStore(t) // b exists in the store but the variation does not reference it
			if err := run(e.skillsHandler(t, t.TempDir()), "invoke", aiserver.HandlerRequest{Config: skillsCfg([]ref{{"a", 1}}), UserInput: "q"}); err != nil {
				t.Fatal(err)
			}
			r := e.api.reqs()
			if len(r) != 2 {
				t.Fatalf("%d requests", len(r))
			}
			text, isErr := lastToolResult(t, r[1])
			if !isErr || strings.Contains(text, "a v1") || strings.Contains(text, "b v1") {
				t.Errorf("result = %q (error %v), want a tool error with no skill content", text, isErr)
			}
		})
	}
}

// The wrappers and the native graph adapter build the core handler, so a
// variation with skills runs through them without its skills.
func TestSkillsWrappersDoNotMaterialize(t *testing.T) {
	withSkills := agentVariation("inst", map[string]any{"skills": refsValue([]ref{{"a", 1}})})
	flags := map[string]any{"cfg": withSkills, "root": withSkills, "g": map[string]any{"root": "root", "_ldMeta": map[string]any{"enabled": true}}}
	cases := []struct {
		name string
		resp string
		run  func(e *env) error
	}{
		{"ClaudeAgents", textResp("done", 1, 1), func(e *env) error {
			_, err := ClaudeAgents(bg, "cfg", "q", user, &InvokeOptions{Handler: e.opts})
			return err
		}},
		{"ClaudeGraph", sse([]string{"done"}, "", 1, 1), func(e *env) error { // graph() streams each node
			g, err := ClaudeGraph("g", GraphOptions{}, e.opts)
			if err != nil {
				return err
			}
			_, err = g.Invoke(bg, "q", user)
			return err
		}},
		{"ToClaudeAgents", textResp("done", 1, 1), func(e *env) error {
			r, err := ToClaudeAgents(resolve(t), &AdapterOptions{Handler: e.opts})
			if err != nil {
				return err
			}
			_, err = r.Invoke(bg, "q")
			return err
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tmp := t.TempDir()
			t.Setenv("TMPDIR", tmp)
			e := setup(t, flags, tc.resp)
			skillStore(t)
			var dirs []string
			e.api.during = func(map[string]any) { dirs = append(dirs, invokeDirs(t, tmp)...) }
			if err := tc.run(e); err != nil {
				t.Fatal(err)
			}
			r := e.api.reqs()
			if len(r) != 1 {
				t.Fatalf("%d model requests, want 1", len(r))
			}
			if toolByName(r[0], "Skill") != nil {
				t.Error("the wrapper offered the Skill tool")
			}
			if len(dirs) != 0 {
				t.Errorf("the wrapper wrote skills: %v", dirs)
			}
		})
	}
}

// A cancelled context still removes the directory.
func TestSkillsCancelRemovesDir(t *testing.T) {
	e := setup(t, nil, textResp("done", 1, 1))
	skillStore(t)
	parent := t.TempDir()
	ctx, cancel := context.WithCancel(bg)
	created := false
	e.api.during = func(map[string]any) { created = len(invokeDirs(t, parent)) == 1; cancel() }
	_, err := e.skillsHandler(t, parent).Invoke(ctx, aiserver.HandlerRequest{Config: skillsCfg([]ref{{"a", 1}}), UserInput: "q"})
	if !created {
		t.Error("no invoke dir during the run")
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want context.Canceled", err)
	}
	if d := invokeDirs(t, parent); len(d) != 0 {
		t.Errorf("not removed: %v", d)
	}
}
