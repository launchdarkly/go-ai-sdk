package evaluations

import (
	"encoding/json"
	"fmt"
	"maps"
	"math"
	"reflect"
	"slices"
	"sort"
	"strings"

	"golang.org/x/text/cases"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

// --- handler pool -------------------------------------------------

func validatePool(single *aiserver.ProviderHandler, many []*aiserver.ProviderHandler) ([]*aiserver.ProviderHandler, error) {
	switch {
	case single != nil && many != nil:
		return nil, invalid("pass exactly one of Handler and Handlers, not both")
	case single == nil && many == nil:
		return nil, invalid("pass exactly one of Handler and Handlers")
	case many != nil && len(many) == 0:
		return nil, invalid("Handlers must be a non-empty list")
	}
	pool, name := many, "Handlers"
	if single != nil {
		pool, name = []*aiserver.ProviderHandler{single}, "Handler"
	}
	seen := map[aiserver.ProvidesFor]bool{}
	for i, h := range pool {
		label := name
		if name == "Handlers" {
			label = fmt.Sprintf("Handlers[%d]", i)
		}
		if h == nil || h.Invoke == nil {
			return nil, invalid("%s must be callable; build it with aiserver.CreateHandler or a provider package's NewHandler", label)
		}
		if h.ProvidesFor.Provider == "" {
			return nil, invalid("%s does not declare ProvidesFor; build it with aiserver.CreateHandler or a provider package's NewHandler so it can be selected by provider and mode", label)
		}
		pf := providesFor(h)
		if seen[pf] {
			return nil, invalid("two handlers provide for provider %q in %q mode; the run cannot tell which one to use", pf.Provider, pf.Mode)
		}
		seen[pf] = true
	}
	return pool, nil
}

func providesFor(h *aiserver.ProviderHandler) aiserver.ProvidesFor {
	return aiserver.ProvidesFor{Provider: h.ProvidesFor.Provider, Mode: aiserver.NormalizeMode(string(h.ProvidesFor.Mode))}
}

// selectHandler applies the shared config() rule (aiserver.SelectHandler)
// to the pool with each handler's mode normalized, as the
// duplicate and coverage checks see it; nil when no handler serves provider
// and mode.
func selectHandler(pool []*aiserver.ProviderHandler, provider string, mode aiserver.Mode) *aiserver.ProviderHandler {
	h, _ := aiserver.SelectHandler(pool, provider, mode)
	return h
}

func hasMode(pool []*aiserver.ProviderHandler, mode aiserver.Mode) bool {
	return slices.ContainsFunc(pool, func(h *aiserver.ProviderHandler) bool { return providesFor(h).Mode == mode })
}

func describePool(pool []*aiserver.ProviderHandler) string {
	parts := make([]string, len(pool))
	for i, h := range pool {
		parts[i] = describe(h)
	}
	return strings.Join(parts, ", ")
}

func describe(h *aiserver.ProviderHandler) string {
	pf := providesFor(h)
	return fmt.Sprintf("[%s, %s]", pf.Provider, pf.Mode)
}

func uncovered(provider string, mode aiserver.Mode, pool []*aiserver.ProviderHandler) error {
	return invalid("no handler in the pool provides for generation provider %q in %q mode (a completion generation needs a %q handler, an agent generation an %q handler); the pool provides for: %s",
		provider, mode, aiserver.ModeMessages, aiserver.ModeAgent, describePool(pool))
}

// promptProblem is the prompt rule: the harness never converts one prompt
// field into the other, so it returns the field the handler mode needs, or
// "" when the prompt fits. A prompt with neither field fits both modes.
func promptProblem(mode aiserver.Mode, instructions string, messages []aiserver.Message) string {
	if mode == aiserver.ModeMessages && instructions != "" {
		return "the prompt needs messages because the handler is a messages handler"
	}
	if mode == aiserver.ModeAgent && len(messages) > 0 {
		return "the prompt needs instructions because the handler is an agent handler"
	}
	return ""
}

// validateGenerationMode checks the caller's raw value before normalizing:
// NormalizeMode maps every unknown string, typos included, to messages.
func validateGenerationMode(mode string) (aiserver.Mode, error) {
	switch mode {
	case "":
		return "", invalid("generation.mode is required. Set it to 'completion' or 'agent'.")
	case ModeCompletion, ModeAgent:
		return aiserver.NormalizeMode(mode), nil
	}
	return "", invalid("generation.mode must be 'completion' or 'agent', got %q", mode)
}

func validateGenerationFields(g *GenerationConfig) error {
	if strings.TrimSpace(g.Provider) == "" {
		return invalid("generation.provider is required")
	}
	if strings.TrimSpace(g.Model) == "" {
		return invalid("generation.model is required")
	}
	if g.Instructions != "" && len(g.Messages) > 0 {
		return invalid("generation.instructions and generation.messages are mutually exclusive")
	}
	return nil
}

// --- tools --------------------------------------------------

// runTool is one tool after validation; source is decided here and never
// re-examined downstream.
type runTool struct {
	key         string
	impl        aiserver.ToolHandler
	inline      bool
	description string
	schema      map[string]any
	version     int // library only
}

func isNilTool(h aiserver.ToolHandler) bool {
	switch t := h.(type) {
	case nil:
		return true
	case aiserver.ToolFunc:
		return t == nil
	case *aiserver.NativeTool:
		return t == nil
	}
	return false
}

func (t runTool) sourceName() string {
	if t.inline {
		return "inline tool (InlineTools)"
	}
	return "library tool (Tools)"
}

// validateTools checks both maps before any I/O and returns the tools in
// key order (Go maps have no declaration order). Keys are checked on both
// sources: non-blank, lowercase, and unique under Unicode case folding
// (straße and strasse collide), so the collision check covers library vs
// inline, library vs library and inline vs inline.
func validateTools(library map[string]aiserver.ToolHandler, inline map[string]InlineTool) ([]runTool, error) {
	var out []runTool
	fold := cases.Fold()
	seen := map[string]runTool{}
	checkKey := func(t runTool) error {
		if strings.TrimSpace(t.key) == "" {
			return invalid("tool keys must not be blank: %s has the key %q", t.sourceName(), t.key)
		}
		if lower := strings.ToLower(t.key); t.key != lower {
			return invalid("%s key %q must not use uppercase letters; use %q", t.sourceName(), t.key, lower)
		}
		f := fold.String(t.key)
		if prev, dup := seen[f]; dup {
			if prev.key == t.key {
				return invalid("tool %q is declared both as a %s and as an %s; declare it once", t.key, prev.sourceName(), t.sourceName())
			}
			return invalid("%s %q and %s %q are the same key under case folding; rename one", prev.sourceName(), prev.key, t.sourceName(), t.key)
		}
		seen[f] = t
		return nil
	}
	for _, key := range slices.Sorted(maps.Keys(library)) {
		t := runTool{key: key, impl: library[key]}
		if err := checkKey(t); err != nil {
			return nil, err
		}
		if isNilTool(t.impl) {
			return nil, invalid("tool %q implementation must be an aiserver.ToolFunc or *aiserver.NativeTool", key)
		}
		out = append(out, t)
	}
	for _, key := range slices.Sorted(maps.Keys(inline)) {
		def := inline[key]
		if err := checkKey(runTool{key: key, inline: true}); err != nil {
			return nil, err
		}
		if _, native := def.Implementation.(*aiserver.NativeTool); native {
			return nil, invalid("inline tool %q uses a NativeTool, but a native tool has no schema to define; pass it in Tools instead", key)
		}
		if isNilTool(def.Implementation) {
			return nil, invalid("inline tool %q implementation must be an aiserver.ToolFunc", key)
		}
		if def.Schema == nil {
			return nil, invalid("inline tool %q schema is required and must be a JSON object, got null", key)
		}
		schema, err := strictCopy(def.Schema)
		if err != nil {
			return nil, invalid("inline tool %q schema must be JSON-serializable: %v", key, err)
		}
		out = append(out, runTool{key: key, impl: def.Implementation, inline: true, description: def.Description, schema: schema})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].key < out[j].key })
	return out, nil
}

// strictCopy rejects anything that is not plain JSON — non-finite numbers,
// non-string map keys, structs (time.Time), byte slices, channels — instead
// of letting encoding/json coerce it, then returns a decoded copy.
func strictCopy(v map[string]any) (map[string]any, error) {
	if err := strictJSON(reflect.ValueOf(v), "$"); err != nil {
		return nil, err
	}
	b, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	var out map[string]any
	if err := json.Unmarshal(b, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// cloneJSON deep-copies a decoded JSON value, so each consumer of a
// validated schema (the create body, every row's handler config) holds its
// own copy.
func cloneJSON(v any) any {
	switch t := v.(type) {
	case map[string]any:
		out := make(map[string]any, len(t))
		for k, e := range t {
			out[k] = cloneJSON(e)
		}
		return out
	case []any:
		out := make([]any, len(t))
		for i, e := range t {
			out[i] = cloneJSON(e)
		}
		return out
	}
	return v
}

func strictJSON(v reflect.Value, path string) error {
	if !v.IsValid() {
		return nil
	}
	switch v.Kind() {
	case reflect.Interface, reflect.Pointer:
		if v.IsNil() {
			return nil
		}
		return strictJSON(v.Elem(), path)
	case reflect.String, reflect.Bool,
		reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return nil
	case reflect.Float32, reflect.Float64:
		if f := v.Float(); math.IsNaN(f) || math.IsInf(f, 0) {
			return fmt.Errorf("%s is %v, which is not valid JSON", path, f)
		}
		return nil
	case reflect.Map:
		if v.Type().Key().Kind() != reflect.String {
			return fmt.Errorf("%s has non-string keys (%s)", path, v.Type().Key())
		}
		iter := v.MapRange()
		for iter.Next() {
			if err := strictJSON(iter.Value(), path+"."+iter.Key().String()); err != nil {
				return err
			}
		}
		return nil
	case reflect.Slice, reflect.Array:
		if v.Type().Elem().Kind() == reflect.Uint8 {
			return fmt.Errorf("%s is a byte slice, which is not a JSON value", path)
		}
		for i := range v.Len() {
			if err := strictJSON(v.Index(i), fmt.Sprintf("%s[%d]", path, i)); err != nil {
				return err
			}
		}
		return nil
	}
	return fmt.Errorf("%s has type %s, which is not a JSON value", path, v.Type())
}

// --- inline dataset rows ------------------------------------------

// rawRow is a validated inline row, unrendered, as it is uploaded.
type rawRow struct {
	index     int
	input     string
	expected  *string
	variables map[string]any
	metadata  map[string]any
}

var inlineRowFields = []string{"rowIdx", "input", "expectedOutput", "variables", "metadata"}

func normalizeInlineRows(rows InlineRows) ([]rawRow, error) {
	if len(rows) == 0 {
		return nil, invalid("Inline dataset is empty")
	}
	out := make([]rawRow, len(rows))
	for pos, r := range rows {
		var err error
		switch t := r.(type) {
		case DatasetRow:
			out[pos], err = publicRow(pos, t)
		case *DatasetRow:
			if t == nil {
				return nil, invalid("inline dataset row %d must be a DatasetRow or a map, got nil", pos)
			}
			out[pos], err = publicRow(pos, *t)
		case map[string]any:
			out[pos], err = mappingRow(pos, t)
		default:
			return nil, invalid("inline dataset row %d must be a DatasetRow or a map[string]any, got %T", pos, r)
		}
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

func publicRow(pos int, r DatasetRow) (rawRow, error) {
	if r.RowIndex != 0 && r.RowIndex != pos {
		return rawRow{}, invalid("inline dataset row %d has RowIndex %d; an inline row's index is its position", pos, r.RowIndex)
	}
	if r.Input == "" {
		return rawRow{}, invalid("inline dataset row %d input must be a non-empty string", pos)
	}
	vars, err := rowObject(pos, "variables", r.Variables)
	if err != nil {
		return rawRow{}, err
	}
	meta, err := rowObject(pos, "metadata", r.Metadata)
	if err != nil {
		return rawRow{}, err
	}
	row := rawRow{index: pos, input: r.Input, variables: vars, metadata: meta}
	if r.ExpectedOutput != "" {
		row.expected = &r.ExpectedOutput
	}
	return row, nil
}

func mappingRow(pos int, m map[string]any) (rawRow, error) {
	var unknown []string
	for k := range m {
		if !slices.Contains(inlineRowFields, k) {
			unknown = append(unknown, fmt.Sprintf("%q", k))
		}
	}
	if len(unknown) > 0 {
		sort.Strings(unknown)
		return rawRow{}, invalid("inline dataset row %d has unknown fields: %s. Expected any of: %s",
			pos, strings.Join(unknown, ", "), strings.Join(inlineRowFields, ", "))
	}
	if idx, present := m["rowIdx"]; present && idx != nil {
		if n, ok := intValue(idx); !ok || n != pos {
			return rawRow{}, invalid("inline dataset row %d has rowIdx %#v; an inline row's index is its position", pos, idx)
		}
	}
	input, ok := m["input"].(string)
	if !ok || input == "" {
		return rawRow{}, invalid("inline dataset row %d input must be a non-empty string", pos)
	}
	row := rawRow{index: pos, input: input}
	if e, present := m["expectedOutput"]; present && e != nil {
		s, ok := e.(string)
		if !ok {
			return rawRow{}, invalid("inline dataset row %d expectedOutput must be a string, got %T", pos, e)
		}
		row.expected = &s
	}
	var err error
	if row.variables, err = rowObject(pos, "variables", m["variables"]); err != nil {
		return rawRow{}, err
	}
	if row.metadata, err = rowObject(pos, "metadata", m["metadata"]); err != nil {
		return rawRow{}, err
	}
	return row, nil
}

// intValue accepts Go integers and integral float64 (decoded JSON), and
// rejects bool explicitly.
func intValue(v any) (int, bool) {
	switch t := v.(type) {
	case bool:
		return 0, false
	case float64:
		return asInt(t)
	}
	rv := reflect.ValueOf(v)
	switch rv.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return int(rv.Int()), true
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return int(rv.Uint()), true
	}
	return 0, false
}

func rowObject(pos int, field string, v any) (map[string]any, error) {
	if v == nil {
		return nil, nil
	}
	rv := reflect.ValueOf(v)
	if rv.Kind() != reflect.Map {
		return nil, invalid("inline dataset row %d %s must be an object, got %T", pos, field, v)
	}
	if rv.IsNil() {
		return nil, nil
	}
	if err := strictJSON(rv, field); err != nil {
		return nil, invalid("inline dataset row %d %s must be JSON-encodable without NaN or Infinity: %v", pos, field, err)
	}
	b, err := json.Marshal(v)
	if err != nil {
		return nil, invalid("inline dataset row %d %s must be JSON-encodable: %v", pos, field, err)
	}
	var out map[string]any
	if err := json.Unmarshal(b, &out); err != nil {
		return nil, invalid("inline dataset row %d %s must be an object: %v", pos, field, err)
	}
	return out, nil
}

// --- criteria ---------------------------------------------------

func validateCriteria(criteria []Criterion) error {
	seen := map[string]bool{}
	var dups []string
	for i, c := range criteria {
		if c == nil || reflect.ValueOf(c).IsNil() {
			return invalid("Criteria[%d] must be a *Judge or *Scorer, got nil", i)
		}
		if err := c.validate(); err != nil {
			return err
		}
		id := strings.ToLower(c.criterionType())
		if seen[id] && !slices.Contains(dups, c.criterionType()) {
			dups = append(dups, c.criterionType())
		}
		seen[id] = true
	}
	if len(dups) > 0 {
		return invalid("duplicate evaluation criteria: %q; judge keys and scorer names must be unique within a run (case-insensitive)", dups)
	}
	return nil
}
