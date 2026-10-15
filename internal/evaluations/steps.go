package evaluations

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"strings"
	"time"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

// --- AI Config source ---------------------------------------------

type variationTool struct {
	key     string
	version int
}

// readAIConfig reads the AI Config, its latest variation version and any
// linked model config, running each check as soon as its input arrives.
func (r *runState) readAIConfig(ctx context.Context) error {
	ref, ov := r.args.AIConfig, r.args.Overrides
	if ov == nil {
		ov = &GenerationOverrides{}
	}
	base := "projects/" + r.project() + "/ai-configs/"
	raw, err := r.m.api.get(ctx, base+segment(ref.Key))
	if err != nil {
		return notFound(err, "AI Config %q was not found in project %q", ref.Key, r.args.ProjectKey)
	}
	cfgObj, err := asObject(raw, "AI Config")
	if err != nil {
		return err
	}
	modeName := ModeCompletion // the API default
	if v, present := cfgObj["mode"]; present && v != nil {
		s, ok := v.(string)
		if !ok || (s != ModeAgent && s != ModeCompletion && s != "judge") {
			return fmt.Errorf("evaluations: %w: AI Config %q has unsupported mode %#v", ErrInvalidConfig, ref.Key, v)
		}
		modeName = s
	}
	mode := aiserver.NormalizeMode(modeName)
	if !hasMode(r.pool, mode) {
		return fmt.Errorf("evaluations: %w: AI Config %q is in %q mode, which needs a %q handler; the pool provides for: %s",
			ErrInvalidConfig, ref.Key, modeName, mode, describePool(r.pool))
	}

	raw, err = r.m.api.get(ctx, base+segment(ref.Key)+"/variations/"+segment(ref.Variation))
	if err != nil {
		return notFound(err, "variation %q of AI Config %q was not found in project %q", ref.Variation, ref.Key, r.args.ProjectKey)
	}
	varObj, err := asObject(raw, "AI Config variation")
	if err != nil {
		return err
	}
	latest, err := latestVersion(varObj)
	if err != nil {
		return fmt.Errorf("evaluations: %w: AI Config variation %q/%q %w", ErrInvalidResponse, ref.Key, ref.Variation, err)
	}

	// The mode decides which prompt field to read; the other field is kept
	// only when the mode's field is empty, so the prompt check can report it.
	instructions, _ := latest["instructions"].(string)
	messages := decodeMessages(latest["messages"])
	if mode == aiserver.ModeAgent && instructions != "" {
		messages = nil
	} else if mode == aiserver.ModeMessages && len(messages) > 0 {
		instructions = ""
	}
	if ov.Instructions != "" || ov.Messages != nil { // one prompt slot
		instructions, messages = ov.Instructions, ov.Messages
	}
	if p := promptProblem(mode, instructions, messages); p != "" {
		needs := "messages"
		if mode == aiserver.ModeAgent {
			needs = "instructions"
		}
		return fmt.Errorf("evaluations: %w: AI Config %q variation %q is in %q mode and needs %s: %s",
			ErrInvalidConfig, ref.Key, ref.Variation, modeName, needs, p)
	}

	model, _ := latest["model"].(map[string]any)
	modelName, _ := model["modelName"].(string)
	params, _ := model["parameters"].(map[string]any)
	var provider string
	if mck, present := latest["modelConfigKey"]; present && mck != nil {
		key, ok := mck.(string)
		if !ok {
			return fmt.Errorf("evaluations: %w: AI Config variation %q/%q has a non-string modelConfigKey %#v",
				ErrInvalidResponse, ref.Key, ref.Variation, mck)
		}
		if key != "" {
			var version any
			if v, ok := asInt(latest["modelConfigVersion"]); ok {
				version = v
			}
			raw, err := r.m.api.get(ctx, base+"model-configs/"+segment(key), param{"version", version})
			if err != nil {
				return notFound(err, "model config %q was not found in project %q", key, r.args.ProjectKey)
			}
			mc, err := asObject(raw, "model config")
			if err != nil {
				return err
			}
			provider, _ = mc["provider"].(string)
			if baseParams, ok := mc["params"].(map[string]any); ok {
				merged := maps.Clone(baseParams)
				maps.Copy(merged, params)
				params = merged
			}
			// A linked variation's modelName is the model-config key, not a
			// model ID a provider accepts.
			if id, ok := mc["id"].(string); ok && id != "" {
				modelName = id
			}
		}
	}
	if len(params) == 0 {
		params = nil
	}

	g := GenerationConfig{Provider: provider, Model: modelName, Mode: modeName, Parameters: params,
		Instructions: instructions, Messages: messages}
	if of, ok := latest["outputFormat"].(map[string]any); ok {
		g.OutputFormat = of
	}
	applyOverrides(&g, ov)
	if err := validateGenerationFields(&g); err != nil {
		return err
	}
	h := selectHandler(r.pool, g.Provider, mode)
	if h == nil {
		return uncovered(g.Provider, mode, r.pool)
	}
	r.gen, r.genHandler = g, h

	tools := variationTools(latest)
	if r.args.Tools == nil && r.args.InlineTools == nil && len(tools) > 0 {
		keys := make([]string, len(tools))
		for i, t := range tools {
			keys[i] = t.key
		}
		return fmt.Errorf("evaluations: %w: AI Config variation %q/%q uses tools with no implementation: %q; pass Tools or InlineTools with an implementation for each",
			ErrInvalidConfig, ref.Key, ref.Variation, keys)
	}
	r.pinnedTools = map[string]int{}
	for _, t := range tools {
		r.pinnedTools[t.key] = t.version
		if !r.hasTool(t.key) {
			r.m.logger.Warn("AI Config variation attaches a tool this run does not use", "aiConfig", ref.Key, "variation", ref.Variation, "tool", t.key)
		}
	}
	if r.args.Criteria == nil {
		for _, key := range variationJudges(latest) {
			j, err := NewJudge(key, nil)
			if err != nil {
				return err
			}
			r.criteria = append(r.criteria, j)
		}
		if err := validateCriteria(r.criteria); err != nil {
			return err
		}
	}
	return nil
}

func (r *runState) hasTool(key string) bool {
	for _, t := range r.tools {
		if t.key == key {
			return true
		}
	}
	return false
}

// latestVersion picks the item with the highest integer version.
func latestVersion(obj map[string]any) (map[string]any, error) {
	items, _ := obj["items"].([]any)
	var latest map[string]any
	best := 0
	for _, it := range items {
		m, ok := it.(map[string]any)
		if !ok {
			continue
		}
		if v, ok := asInt(m["version"]); ok && (latest == nil || v > best) {
			latest, best = m, v
		}
	}
	if latest == nil {
		return nil, fmt.Errorf("has no versions")
	}
	return latest, nil
}

func decodeMessages(v any) []aiserver.Message {
	list, ok := v.([]any)
	if !ok || len(list) == 0 {
		return nil
	}
	b, err := json.Marshal(list)
	if err != nil {
		return nil
	}
	var out []aiserver.Message
	if json.Unmarshal(b, &out) != nil {
		return nil
	}
	return out
}

func applyOverrides(g *GenerationConfig, ov *GenerationOverrides) {
	if ov.Provider != "" {
		g.Provider = ov.Provider
	}
	if ov.Model != "" {
		g.Model = ov.Model
	}
	if ov.Parameters != nil {
		merged := maps.Clone(g.Parameters)
		if merged == nil {
			merged = map[string]any{}
		}
		maps.Copy(merged, ov.Parameters)
		g.Parameters = merged
	}
	if ov.PromptSnippets != nil {
		g.PromptSnippets = ov.PromptSnippets
	}
	if ov.OutputFormat != nil {
		g.OutputFormat = ov.OutputFormat
	}
}

func variationTools(v map[string]any) []variationTool {
	list, _ := v["tools"].([]any)
	var out []variationTool
	for _, it := range list {
		m, _ := it.(map[string]any)
		key, ok := m["key"].(string)
		version, vok := asInt(m["version"])
		if ok && vok {
			out = append(out, variationTool{key, version})
		}
	}
	return out
}

func variationJudges(v map[string]any) []string {
	jc, _ := v["judgeConfiguration"].(map[string]any)
	list, _ := jc["judges"].([]any)
	var out []string
	for _, it := range list {
		m, _ := it.(map[string]any)
		if key, ok := m["judgeConfigKey"].(string); ok {
			out = append(out, key)
		}
	}
	return out
}

// --- tools, dataset, records --------------------------------------

// resolveTools pins each library tool's version; inline tools need nothing.
func (r *runState) resolveTools(ctx context.Context) error {
	for i := range r.tools {
		t := &r.tools[i]
		if t.inline {
			continue
		}
		raw, err := r.m.api.get(ctx, "projects/"+r.project()+"/ai-tools/"+segment(t.key))
		if err != nil {
			return notFound(err, "LaunchDarkly AI tool %q was not found in project %q", t.key, r.args.ProjectKey)
		}
		obj, err := asObject(raw, fmt.Sprintf("tool %q", t.key))
		if err != nil {
			return err
		}
		v, ok := asInt(obj["version"])
		if !ok {
			return fmt.Errorf("evaluations: %w: LaunchDarkly AI tool %q has no integer version", ErrInvalidResponse, t.key)
		}
		t.version = v
		t.description, _ = obj["description"].(string)
		t.schema, _ = obj["schema"].(map[string]any)
		if t.schema == nil {
			t.schema = map[string]any{}
		}
		if pinned, ok := r.pinnedTools[t.key]; ok && pinned != v {
			r.m.logger.Warn("AI Config variation pins a different tool version; the run uses the resolved version",
				"tool", t.key, "pinnedVersion", pinned, "resolvedVersion", v)
		}
	}
	return nil
}

func (r *runState) fetchDataset(ctx context.Context) error {
	path := "projects/" + r.project() + "/datasets/" + segment(r.datasetKey)
	raw, err := r.m.api.get(ctx, path)
	if err != nil {
		return notFound(err, "LaunchDarkly dataset %q was not found in project %q", r.datasetKey, r.args.ProjectKey)
	}
	obj, err := asObject(raw, fmt.Sprintf("dataset %q", r.datasetKey))
	if err != nil {
		return err
	}
	if r.datasetID, err = requireString(obj, "id", "dataset"); err != nil {
		return err
	}
	for _, k := range []string{"key", "name"} {
		if s, ok := obj[k].(string); ok && s != "" {
			r.datasetKey = s
			break
		}
	}
	total := -1
	for total < 0 || len(r.rows) < total {
		raw, err := r.m.api.get(ctx, path+"/rows", param{"mode", "all"}, param{"limit", datasetPageSize}, param{"offset", len(r.rows)})
		if err != nil {
			return err
		}
		page, _ := raw.(map[string]any)
		items, ok := page["items"].([]any)
		n, nok := asInt(page["totalCount"])
		if !ok || !nok {
			return fmt.Errorf("evaluations: %w: LaunchDarkly returned invalid rows for dataset %q", ErrInvalidResponse, r.datasetKey)
		}
		total = n
		if len(items) == 0 {
			break
		}
		for _, it := range items {
			item, err := asObject(it, "dataset row")
			if err != nil {
				return err
			}
			idx, ok := asInt(item["rowIndex"])
			if !ok {
				return fmt.Errorf("evaluations: %w: a row of dataset %q is missing its integer rowIndex", ErrInvalidResponse, r.datasetKey)
			}
			r.rows = append(r.rows, renderRow(idx, item["input"], item["expectedOutput"], item["variables"], item["metadata"]))
		}
	}
	if len(r.rows) == 0 {
		return fmt.Errorf("evaluations: %w: dataset %q is empty", ErrInvalidConfig, r.datasetKey)
	}
	if len(r.rows) != total {
		return fmt.Errorf("evaluations: %w: dataset %q returned %d of %d rows", ErrInvalidResponse, r.datasetKey, len(r.rows), total)
	}
	return nil
}

func (r *runState) createEvaluation(ctx context.Context) error {
	g := r.gen
	body := map[string]any{"name": r.args.Key, "generationProvider": g.Provider, "generationModel": g.Model}
	if g.Parameters != nil {
		body["parameters"] = g.Parameters
	}
	switch {
	case g.Instructions != "":
		body["messages"] = []any{map[string]any{"role": "system", "content": g.Instructions}}
	case len(g.Messages) > 0:
		body["messages"] = g.Messages
	default:
		body["messages"] = []any{}
	}
	if g.PromptSnippets != nil {
		body["promptSnippets"] = g.PromptSnippets
	}
	if len(r.tools) > 0 {
		tools := make([]any, len(r.tools))
		for i, t := range r.tools {
			if t.inline {
				tools[i] = map[string]any{"key": t.key, "schema": t.schema, "description": t.description, "source": "inline"}
			} else {
				tools[i] = map[string]any{"key": t.key, "version": t.version, "source": "library"}
			}
		}
		body["tools"] = tools
	}
	if len(r.criteria) > 0 {
		criteria := make([]any, len(r.criteria))
		for i, c := range r.criteria {
			criteria[i] = c.wire()
		}
		body["criteria"] = criteria
	}
	raw, err := r.m.api.post(ctx, "projects/"+r.project()+"/evaluations", body, false)
	if err != nil {
		return err
	}
	obj, err := asObject(raw, "evaluation")
	if err != nil {
		return err
	}
	if r.evaluationID, err = requireString(obj, "id", "evaluation"); err != nil {
		return err
	}
	v, ok := asInt(obj["version"])
	if !ok {
		return fmt.Errorf("evaluations: %w: LaunchDarkly evaluation response is missing integer field \"version\"", ErrInvalidResponse)
	}
	r.evalVersion = v
	r.evaluationKey = r.args.Key
	if name, ok := obj["name"].(string); ok && name != "" {
		r.evaluationKey = name
	}
	return nil
}

func (r *runState) runsPath() string {
	return "projects/" + r.project() + "/evaluations/" + segment(r.evaluationID) + "/runs"
}

func (r *runState) createRun(ctx context.Context) error {
	body := map[string]any{"source": "api"}
	if !r.inline {
		body["datasetId"] = r.datasetID
	}
	raw, err := r.m.api.post(ctx, r.runsPath(), body, false)
	if err != nil {
		return err
	}
	obj, err := asObject(raw, "evaluation run")
	if err != nil {
		return err
	}
	for _, k := range []string{"id", "evaluationId", "state"} {
		s, err := requireString(obj, k, "evaluation run")
		if err != nil {
			return err
		}
		if k == "id" {
			r.runID = s
		}
	}
	return nil
}

// upload sends the inline rows unrendered, 500 at a time and in order;
// every batch lands before the first event is tracked.
func (r *runState) upload(ctx context.Context) error {
	path := r.runsPath() + "/" + segment(r.runID)
	total := len(r.rawRows)
	for start := 0; start < total; start += uploadBatchSize {
		end := min(start+uploadBatchSize, total)
		rows := make([]any, 0, end-start)
		for _, raw := range r.rawRows[start:end] {
			vars := raw.variables
			if vars == nil {
				vars = map[string]any{}
			}
			rows = append(rows, map[string]any{"rowIdx": raw.index, "input": raw.input,
				"expectedOutput": orNil(raw.expected), "variables": vars, "metadata": nilIfEmpty(raw.metadata)})
		}
		if _, err := r.m.api.post(ctx, path+"/dataset-rows", map[string]any{"rows": rows}, true); err != nil {
			// Cancel is the only terminal state a client can set; without it
			// the run sits PENDING with a partial dataset.
			if _, cerr := r.m.api.post(ctx, path+"/cancel", nil, true); cerr != nil {
				r.m.logger.Warn("failed to cancel evaluation run after its inline dataset upload failed", "runId", r.runID, "error", cerr)
			}
			return fmt.Errorf("evaluations: %w: rows %d-%d of %d to evaluation run %s: %w",
				ErrUploadFailed, start, end-1, total, r.runID, err)
		}
	}
	return nil
}

// nilIfEmpty keeps an absent metadata map null on the wire.
func nilIfEmpty(m map[string]any) any {
	if m == nil {
		return nil
	}
	return m
}

// --- summary polling ----------------------------------------------

func parseSummary(raw any) (RunSummary, error) {
	if raw == nil {
		return RunSummary{}, nil
	}
	obj, err := asObject(raw, "evaluation run summary")
	if err != nil {
		return RunSummary{}, err
	}
	counts := obj
	if sc, ok := obj["statusCounts"].(map[string]any); ok {
		counts = sc
	}
	get := func(short, long string) int {
		if v, ok := asInt(counts[short]); ok {
			return v
		}
		v, _ := asInt(counts[long])
		return v
	}
	return RunSummary{TotalRows: get("total", "total_rows"), PassedRows: get("passed", "passed_rows"),
		FailedRows: get("failed", "failed_rows"), ErrorRows: get("error", "error_rows"), PendingRows: get("pending", "pending_rows")}, nil
}

func (s RunSummary) accounted() int { return s.PassedRows + s.FailedRows + s.ErrorRows }

func (s RunSummary) terminal() bool {
	return s.TotalRows > 0 && s.PendingRows == 0 && s.accounted() == s.TotalRows
}

func (r *runState) poll(ctx context.Context) (RunSummary, error) {
	path := r.runsPath() + "/" + segment(r.runID) + "/summary"
	deadline := time.Now().Add(r.timeout)
	for {
		raw, err := r.m.api.get(ctx, path)
		if err != nil {
			return RunSummary{}, err
		}
		s, err := parseSummary(raw)
		if err != nil {
			return RunSummary{}, err
		}
		if s.terminal() {
			return s, nil
		}
		remaining := time.Until(deadline)
		if remaining <= 0 {
			return RunSummary{}, r.timeoutError(s)
		}
		if err := r.m.api.sleep(ctx, min(r.interval, remaining)); err != nil {
			return RunSummary{}, fmt.Errorf("evaluations: summary polling: %w", err)
		}
	}
}

// timeoutError explains a stuck run: almost always events LaunchDarkly
// never received, which are silent on the client side.
func (r *runState) timeoutError(s RunSummary) error {
	var b strings.Builder
	fmt.Fprintf(&b, "timed out after %s waiting for evaluation run %s to be fully accounted (total_rows=%d, accounted_rows=%d, pending_rows=%d). ",
		r.timeout, r.runID, s.TotalRows, s.accounted(), s.PendingRows)
	b.WriteString("Undelivered result events are the likely cause. ")
	if s.accounted() == 0 {
		b.WriteString("No rows were accounted for at all, so nothing arrived: suspect the event transport or an SDK key and API token that point at different projects.")
	} else {
		b.WriteString("Some rows are short: suspect events dropped on a full SDK event buffer or an invalid context, or a criterion result that did not land.")
	}
	if n := len(r.criteria); n > 0 {
		fmt.Fprintf(&b, " The run has %d criteria; a row is counted only once results for all of them have landed.", n)
	}
	return fmt.Errorf("evaluations: %w: %s", ErrPollTimeout, b.String())
}
