package skills

import (
	"log/slog"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/bridge"
)

// TESTING.md §3.21a Skill tools: the envelope's optional tools array, in
// author order, with the per-entry shape of a variation's config.tools. Until
// the server ships these run against hand-built envelopes.

func toolEntry(name string) map[string]any {
	return map[string]any{
		"type": "function", "name": name, "description": "d " + name,
		"parameters":       map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}},
		"customParameters": map[string]any{"tags": []any{"x"}},
	}
}

func rawSkillWithTools(key string, version int, content string, tools any) map[string]any {
	raw := rawSkill(key, version, content)
	raw["tools"] = tools
	return raw
}

func storeOf(raws ...map[string]any) *InMemorySkillStore {
	s := NewInMemorySkillStore(nil)
	for _, r := range raws {
		_ = s.Put(r)
	}
	SetSkillStore(s)
	return s
}

func getOne(t *testing.T, raw map[string]any) *Skill {
	t.Helper()
	storeOf(raw)
	s, err := GetSkill(raw["key"].(string), raw["version"].(int))
	if err != nil || s == nil {
		t.Fatalf("GetSkill = %v, %v", s, err)
	}
	return s
}

func toolNames(tools []aiserver.Tool) []string {
	var out []string
	for _, t := range tools {
		out = append(out, t.Name)
	}
	return out
}

func warnings(logs *logCapture) []logEntry {
	var out []logEntry
	for _, e := range logs.all() {
		if e.level == slog.LevelWarn {
			out = append(out, e)
		}
	}
	return out
}

func noIntegritySignal(t *testing.T, rec *recorder, logs *logCapture) {
	t.Helper()
	if len(rec.all()) != 0 || len(logs.records()) != 0 {
		t.Fatalf("bad tool data records no integrity signal or record: %v %v", rec.all(), logs.records())
	}
}

// "Reading tools onto Skill"

func TestSkillToolsOrderIsTheEnvelopesOrder(t *testing.T) {
	rec, logs := setup(t)
	s := getOne(t, rawSkillWithTools("pdf", 3, "# PDF", []any{toolEntry("b"), toolEntry("a"), toolEntry("c")}))
	if got := toolNames(s.Tools()); !slices.Equal(got, []string{"b", "a", "c"}) {
		t.Fatalf("tools = %v", got)
	}
	noIntegritySignal(t, rec, logs)
	if len(warnings(logs)) != 0 {
		t.Fatalf("well-formed tools warn nothing: %v", logs.all())
	}
}

func TestEveryFieldOfAKeptToolArrivesUnchanged(t *testing.T) {
	_, _ = setup(t)
	s := getOne(t, rawSkillWithTools("pdf", 3, "# PDF", []any{toolEntry("get-weather")}))
	e := toolEntry("get-weather")
	want := aiserver.Tool{
		Name: "get-weather", Type: "function", Description: "d get-weather",
		Parameters: e["parameters"].(map[string]any), CustomParameters: e["customParameters"].(map[string]any),
	}
	if got := s.Tools()[0]; !reflect.DeepEqual(got, want) {
		t.Fatalf("tool = %+v, want %+v", got, want)
	}
}

func TestSkillWithoutToolsHasNone(t *testing.T) {
	_, logs := setup(t)
	if got := getOne(t, rawSkill("pdf", 1, "x")).Tools(); got == nil || len(got) != 0 {
		t.Fatalf("no tools field: %#v, want an empty slice", got)
	}
	if got := getOne(t, rawSkillWithTools("pdf", 1, "x", []any{})).Tools(); len(got) != 0 {
		t.Fatalf("empty tools: %v", got)
	}
	if len(warnings(logs)) != 0 {
		t.Fatalf("no warnings expected: %v", logs.all())
	}
}

func TestReturnedToolsShareNoStateWithTheStore(t *testing.T) {
	_, _ = setup(t)
	raw := rawSkillWithTools("pdf", 1, "x", []any{toolEntry("a")})
	s := getOne(t, raw)
	got := s.Tools()
	got[0].Name = "changed"
	got[0].Parameters["type"] = "changed"
	got[0].Parameters["properties"].(map[string]any)["city"] = "changed"
	got[0].CustomParameters["tags"].([]any)[0] = "changed"

	want := toolEntry("a")
	check := func(where string, tools []aiserver.Tool) {
		t.Helper()
		if len(tools) != 1 || tools[0].Name != "a" || !reflect.DeepEqual(tools[0].Parameters, want["parameters"]) ||
			!reflect.DeepEqual(tools[0].CustomParameters, want["customParameters"]) {
			t.Fatalf("%s changed through a returned copy: %+v", where, tools)
		}
	}
	check("the same Skill", s.Tools())
	again, _ := GetSkill("pdf", 1)
	check("a second GetSkill", again.Tools())
	check("the tools a handler is offered", skillTools("pdf", 1))
	if raw["tools"].([]any)[0].(map[string]any)["parameters"].(map[string]any)["type"] != "object" {
		t.Fatal("the store's raw object changed")
	}
}

func TestEveryAccessorCarriesTools(t *testing.T) {
	_, _ = setup(t)
	storeOf(rawSkillWithTools("pdf", 1, "x", []any{toolEntry("b"), toolEntry("a")}))
	want := []string{"b", "a"}
	one, _ := GetSkill("pdf", 1)
	res, _ := GetSkillResult("pdf", 1)
	many, _ := GetSkills([]SkillKey{"pdf"})
	all, _ := AllSkills()
	for name, tools := range map[string][]aiserver.Tool{
		"GetSkill": one.Tools(), "GetSkillResult": res.Skill().Tools(), "GetSkills": many[0].Tools(), "AllSkills": all[0].Tools(),
	} {
		if !slices.Equal(toolNames(tools), want) {
			t.Errorf("%s tools = %v", name, toolNames(tools))
		}
	}
}

func TestHandBuiltSkillHasNoToolsUnlessGiven(t *testing.T) {
	if got := skillOf("pdf", 1, "x").Tools(); got == nil || len(got) != 0 {
		t.Fatalf("NewSkill without tools: %#v, want an empty slice", got)
	}
	in := []aiserver.Tool{{Name: "a", Type: "function", Parameters: map[string]any{"type": "object"}}}
	s := NewSkill(SkillFields{Key: "pdf", Version: 1, Content: []byte("x"), ContentHash: sha("x"), Tools: in})
	in[0].Name = "changed"
	in[0].Parameters["type"] = "changed"
	if got := s.Tools(); got[0].Name != "a" || got[0].Parameters["type"] != "object" {
		t.Fatalf("NewSkill shares its input: %+v", got[0])
	}
}

// "Bad tool data drops only that tool"

func TestToolsNotAnArrayDropsEveryTool(t *testing.T) {
	for name, tools := range map[string]any{"object": map[string]any{"a": toolEntry("a")}, "string": "a", "number": 3.0, "null": nil} {
		t.Run(name, func(t *testing.T) {
			rec, logs := setup(t)
			storeOf(rawSkillWithTools("pdf", 4, "x", tools))
			res, _ := GetSkillResult("pdf", 4)
			if res.Reason() != ReasonOK || len(res.Skill().Tools()) != 0 || string(res.Skill().Content()) != "x" {
				t.Fatalf("want ok with no tools: %v %v", res.Reason(), res.Skill())
			}
			w := warnings(logs)
			if len(w) != 1 || !strings.Contains(w[0].msg, "'pdf' version 4") {
				t.Fatalf("want one warning naming the skill key and version, got %v", logs.all())
			}
			noIntegritySignal(t, rec, logs)
		})
	}
}

func TestBadToolEntryIsDroppedAndTheOthersKeptInOrder(t *testing.T) {
	with := func(name string, edit func(map[string]any)) map[string]any {
		e := toolEntry(name)
		edit(e)
		return e
	}
	cases := map[string]struct {
		entry any
		tool  string
	}{
		"parameters not an object": {with("bad", func(e map[string]any) { e["parameters"] = "x" }), "'bad'"},
		"no parameters":            {with("bad", func(e map[string]any) { delete(e, "parameters") }), "'bad'"},
		"name not a string":        {with("", func(e map[string]any) { e["name"] = 7.0 }), "'tools[1]'"},
		"no name":                  {with("", func(e map[string]any) { delete(e, "name") }), "'tools[1]'"},
		"entry not an object":      {"get-weather", "'tools[1]'"},
		"entry null":               {nil, "'tools[1]'"},
		"type not function":        {with("bad", func(e map[string]any) { e["type"] = "web_search" }), "'bad'"},
		"no type":                  {with("bad", func(e map[string]any) { delete(e, "type") }), "'bad'"},
		"repeated name":            {with("first", func(e map[string]any) { e["description"] = "second" }), "'first'"},
	}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			rec, logs := setup(t)
			s := getOne(t, rawSkillWithTools("pdf", 2, "secret body", []any{toolEntry("first"), c.entry, toolEntry("last")}))
			tools := s.Tools()
			if !slices.Equal(toolNames(tools), []string{"first", "last"}) || tools[0].Description != "d first" {
				t.Fatalf("tools = %+v", tools)
			}
			w := warnings(logs)
			if len(w) != 1 || !strings.Contains(w[0].msg, "'pdf' version 2") || !strings.Contains(w[0].msg, "tool "+c.tool) {
				t.Fatalf("want one warning naming the skill and tool %s, got %v", c.tool, logs.all())
			}
			for _, leak := range []string{"secret body", "d bad", "properties", "city"} {
				if strings.Contains(w[0].msg, leak) || strings.Contains(slogAttrs(w[0]), leak) {
					t.Errorf("warning carries %q: %v", leak, w[0])
				}
			}
			noIntegritySignal(t, rec, logs)
		})
	}
}

func slogAttrs(e logEntry) string {
	var b strings.Builder
	for k, v := range e.attrs {
		b.WriteString(k + "=" + slog.AnyValue(v).String() + " ")
	}
	return b.String()
}

func TestADropWarnsOncePerProcess(t *testing.T) {
	_, logs := setup(t)
	storeOf(rawSkillWithTools("pdf", 1, "x", []any{map[string]any{"name": "bad"}}))
	_, _ = GetSkill("pdf", 1)
	_, _ = GetSkill("pdf", 1)
	_ = skillTools("pdf", 1)
	if w := warnings(logs); len(w) != 1 {
		t.Fatalf("want one warning, got %v", logs.all())
	}
}

// "No integrity hash for tools in v1 (known limit)"

func TestAChangedToolStillVerifies(t *testing.T) {
	_, _ = setup(t)
	raw := rawSkillWithTools("pdf", 1, "same", []any{toolEntry("a")})
	raw["tools"].([]any)[0].(map[string]any)["description"] = "changed"
	storeOf(raw)
	res, _ := GetSkillResult("pdf", 1)
	if res.Reason() != ReasonOK || res.Skill().Tools()[0].Description != "changed" {
		t.Fatalf("outcome = %v %+v", res.Reason(), res.Skill())
	}
}

func TestContentHashIsComputedOverContentOnly(t *testing.T) {
	_, _ = setup(t)
	a := getOne(t, rawSkillWithTools("pdf", 1, "same", []any{toolEntry("a")}))
	b := getOne(t, rawSkillWithTools("pdf", 1, "same", []any{toolEntry("b")}))
	if a.ContentHash() != sha("same") || b.ContentHash() != a.ContentHash() {
		t.Fatalf("hashes %s %s", a.ContentHash(), b.ContentHash())
	}
}

// §3.25 "The envelope's tools is held as delivered."

func TestFDv2EnvelopeHoldsToolsAsDelivered(t *testing.T) {
	_, _ = setup(t)
	h := newHarness(t)
	valid := []any{toolEntry("b"), toolEntry("a")}
	h.commitFull("p",
		putWith("valid:1", map[string]any{"content": "x", "contentHash": sha("x"), "tools": valid}),
		putWith("string:1", map[string]any{"content": "x", "contentHash": sha("x"), "tools": "nope"}),
		putWith("none:1", map[string]any{"content": "x", "contentHash": sha("x")}),
	)
	if got := h.set.get("valid", 1).(map[string]any)["tools"]; !reflect.DeepEqual(got, valid) {
		t.Errorf("valid tools = %#v", got)
	}
	if got := h.set.get("string", 1).(map[string]any)["tools"]; got != "nope" {
		t.Errorf("string tools = %#v", got)
	}
	if _, ok := h.set.get("none", 1).(map[string]any)["tools"]; ok {
		t.Error("an envelope with no tools must reach the store with no tools key")
	}
}

// "Offering skill tools to the provider": the bridge hook core calls.

func TestBridgeSkillToolsIsRegistered(t *testing.T) {
	if _, ok := bridge.SkillTools.(func(string, int) []aiserver.Tool); !ok {
		t.Fatal("skills does not register bridge.SkillTools with the type the core asserts")
	}
}

func TestSkillToolsReadsThePinnedVersion(t *testing.T) {
	_, _ = setup(t)
	storeOf(rawSkillWithTools("s1", 3, "x", []any{toolEntry("v3")}), rawSkillWithTools("s1", 4, "y", []any{toolEntry("v4")}))
	if got := toolNames(skillTools("s1", 3)); !slices.Equal(got, []string{"v3"}) {
		t.Fatalf("tools = %v", got)
	}
}

func TestNoSkillStoreMeansNoSkillToolsAndNoWarning(t *testing.T) {
	_, logs := setup(t)
	if got := skillTools("pdf", 1); got != nil || len(logs.all()) != 0 {
		t.Fatalf("tools = %v, logs = %v", got, logs.all())
	}
}

func TestASkillThatCannotBeReadAddsNoTools(t *testing.T) {
	good := rawSkillWithTools("pdf", 1, "x", []any{toolEntry("a")})
	tampered := rawSkillWithTools("pdf", 1, "x", []any{toolEntry("a")})
	tampered["contentHash"] = sha("other")
	cases := map[SkillOutcomeReason]struct {
		store   SkillStore
		version int
	}{
		ReasonAbsent:           {NewInMemorySkillStore(nil), 1},
		ReasonIntegrityFailure: {NewInMemorySkillStore(map[string]any{"k": tampered}), 1},
		ReasonWrongVersion: {&funcStore{get: func(string, string, int) (any, error) {
			return good, nil
		}}, 2},
		ReasonStoreUnavailable: {&funcStore{get: func(string, string, int) (any, error) { return nil, errStoreDown }}, 1},
	}
	for reason, c := range cases {
		t.Run(string(reason), func(t *testing.T) {
			rec, logs := setup(t)
			SetSkillStore(c.store)
			for range 2 { // two runs, one warning
				if got := skillTools("pdf", c.version); got != nil {
					t.Fatalf("tools = %v", got)
				}
			}
			w := warnings(logs)
			if len(w) != 1 || !strings.Contains(w[0].msg, "'pdf'") || w[0].attrs["reason"] != string(reason) {
				t.Fatalf("want one warning naming the skill and %s, got %v", reason, logs.all())
			}
			// One signal per read, as §3.21 records; skill tools add none.
			if n, want := len(rec.named(signalIntegrityFailure)), map[bool]int{true: 2}[reason == ReasonIntegrityFailure]; n != want {
				t.Fatalf("integrity signals = %d", n)
			}
		})
	}
}

// Go's Tool has plain string and map fields, so a missing description and ""
// are both "", and a wrongly typed optional field is dropped with a warning
// while the tool is kept.
func TestWronglyTypedOptionalToolFieldsAreDroppedWithAWarning(t *testing.T) {
	rec, logs := setup(t)
	badDesc := toolEntry("bad-desc")
	badDesc["description"] = 7.0
	badCustom := toolEntry("bad-custom")
	badCustom["customParameters"] = []any{"x"}
	nulls := toolEntry("nulls")
	nulls["description"], nulls["customParameters"] = nil, nil
	storeOf(rawSkillWithTools("pdf", 1, "x", []any{badDesc, badCustom, nulls}))
	for range 2 {
		s, _ := GetSkill("pdf", 1)
		tools := s.Tools()
		if !slices.Equal(toolNames(tools), []string{"bad-desc", "bad-custom", "nulls"}) ||
			tools[0].Description != "" || tools[1].CustomParameters != nil || tools[2].Description != "" || tools[2].CustomParameters != nil {
			t.Fatalf("tools = %+v", tools)
		}
	}
	w := warnings(logs)
	if len(w) != 2 || !strings.Contains(w[0].msg, "tool 'bad-desc'") || !strings.Contains(w[1].msg, "tool 'bad-custom'") {
		t.Fatalf("want one warning per bad field, once per process, got %v", logs.all())
	}
	noIntegritySignal(t, rec, logs)
}
