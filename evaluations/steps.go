package evaluations

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"math"
	"net/http"
	"strings"
	"time"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/handlerkit"
)

// --- AI Config source ---------------------------------------------

type variationTool struct {
	key     string
	version int
}

// readAIConfig reads the AI Config, the variation version this run uses, and
// any linked model config, running each check as soon as its input arrives.
// An empty Variation is the variation served to {kind: "evaluation", key:
// <evaluation key>}. A missing served _ldMeta.version fails here, before any
// management write.
func (r *runState) readAIConfig(ctx context.Context) error {
	ref, ov := r.args.AIConfig, r.args.Overrides
	if ov == nil {
		ov = &GenerationOverrides{}
	}
	variation := ref.Variation
	var servedVersion int
	if strings.TrimSpace(variation) == "" {
		var err error
		if variation, servedVersion, err = r.servedVariation(ctx); err != nil {
			return err
		}
	}
	base := "projects/" + r.project() + "/ai-configs/"
	raw, err := r.m.api.get(ctx, base+segment(ref.Key))
	if err != nil {
		return notFound(err, "AI Config %q was not found in project %q", ref.Key, r.m.projectKey)
	}
	cfgObj, err := asObject(raw, "AI Config")
	if err != nil {
		return err
	}
	modeName := string(GenerationModeCompletion) // the API default
	if v, present := cfgObj["mode"]; present && v != nil {
		s, ok := v.(string)
		if !ok || (s != string(GenerationModeAgent) && s != string(GenerationModeCompletion) && s != "judge") {
			return fmt.Errorf("evaluations: %w: AI Config %q has unsupported mode %#v", ErrInvalidConfig, ref.Key, v)
		}
		modeName = s
	}
	mode := handlerkit.NormalizeMode(modeName)
	if !hasMode(r.pool, mode) {
		return fmt.Errorf("evaluations: %w: AI Config %q is in %q mode, which needs a %q handler; the pool provides for: %s",
			ErrInvalidConfig, ref.Key, modeName, mode, describePool(r.pool))
	}

	raw, err = r.m.api.get(ctx, base+segment(ref.Key)+"/variations/"+segment(variation))
	if err != nil {
		return notFound(err, "variation %q of AI Config %q was not found in project %q", variation, ref.Key, r.m.projectKey)
	}
	varObj, err := asObject(raw, "AI Config variation")
	if err != nil {
		return err
	}
	var latest map[string]any
	if servedVersion != 0 {
		latest, err = versionByNumber(varObj, servedVersion)
	} else {
		latest, err = latestVersion(varObj)
	}
	if err != nil {
		return fmt.Errorf("evaluations: %w: AI Config variation %q/%q %w", ErrInvalidResponse, ref.Key, variation, err)
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
			ErrInvalidConfig, ref.Key, variation, modeName, needs, p)
	}

	model, _ := latest["model"].(map[string]any)
	modelName, _ := model["modelName"].(string)
	params, _ := model["parameters"].(map[string]any)
	var provider string
	if mck, present := latest["modelConfigKey"]; present && mck != nil {
		key, ok := mck.(string)
		if !ok {
			return fmt.Errorf("evaluations: %w: AI Config variation %q/%q has a non-string modelConfigKey %#v",
				ErrInvalidResponse, ref.Key, variation, mck)
		}
		if key != "" {
			var version any
			if v, ok := asInt(latest["modelConfigVersion"]); ok {
				version = v
			}
			raw, err := r.m.api.get(ctx, base+"model-configs/"+segment(key), param{"version", version})
			if err != nil {
				return notFound(err, "model config %q was not found in project %q", key, r.m.projectKey)
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

	g := GenerationConfig{Provider: provider, Model: modelName, Mode: GenerationMode(modeName), Parameters: params,
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
	if r.args.Tools == nil && len(tools) > 0 {
		keys := make([]string, len(tools))
		for i, t := range tools {
			keys[i] = t.key
		}
		return fmt.Errorf("evaluations: %w: AI Config variation %q/%q uses tools with no implementation: %q; pass Tools with an EvalTool for each",
			ErrInvalidConfig, ref.Key, variation, keys)
	}
	for _, t := range tools {
		rt := r.findTool(t.key)
		switch {
		case rt == nil:
			r.m.logger.Warn("AI Config variation attaches a tool this run does not use", "aiConfig", ref.Key, "variation", variation, "tool", t.key)
		case !rt.inline && rt.version != t.version:
			r.m.logger.Warn("AI Config variation pins a different tool version; the run uses the version Tools().Get pinned",
				"aiConfig", ref.Key, "variation", variation, "tool", t.key, "pinnedVersion", t.version, "runVersion", rt.version)
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
	ver, ok := asInt(latest["version"])
	if !ok {
		return fmt.Errorf("evaluations: %w: AI Config variation %q/%q has no integer version", ErrInvalidResponse, ref.Key, variation)
	}
	// variationVersion is that variation's version, not the evaluation's.
	r.configKey, r.variationKey, r.variationVersion = ref.Key, variation, ver
	return nil
}

// servedVariation asks flag delivery which variation and version this
// evaluation key is served. It does not call the management API.
func (r *runState) servedVariation(ctx context.Context) (string, int, error) {
	key := r.args.AIConfig.Key
	user := ldcontext.NewBuilder(r.args.Key).Kind("evaluation").Build()
	res, err := aiserver.InspectConfig(ctx, key, user)
	if err != nil {
		if errors.Is(err, aiserver.ErrConfigNotFound) {
			return "", 0, fmt.Errorf("evaluations: %w: AI Config %q is not delivered for evaluation %q: %w", ErrNotFound, key, r.args.Key, err)
		}
		return "", 0, fmt.Errorf("evaluations: %w: AI Config %q could not be evaluated for evaluation %q: %w", ErrInvalidConfig, key, r.args.Key, err)
	}
	if res.Meta == nil || !res.Enabled || res.Config == nil || strings.TrimSpace(res.Meta.VariationKey) == "" {
		return "", 0, fmt.Errorf("evaluations: %w: AI Config %q is not delivered for evaluation %q", ErrInvalidConfig, key, r.args.Key)
	}
	if res.Meta.Version < 1 {
		return "", 0, fmt.Errorf("evaluations: %w: AI Config %q served variation %q has no _ldMeta.version", ErrInvalidConfig, key, res.Meta.VariationKey)
	}
	return res.Meta.VariationKey, res.Meta.Version, nil
}

func (r *runState) findTool(key string) *runTool {
	for i := range r.tools {
		if r.tools[i].key == key {
			return &r.tools[i]
		}
	}
	return nil
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

// versionByNumber returns the variation version the run was served. A served
// version older than the newest item is valid.
func versionByNumber(obj map[string]any, version int) (map[string]any, error) {
	items, _ := obj["items"].([]any)
	for _, it := range items {
		m, ok := it.(map[string]any)
		if !ok {
			continue
		}
		if v, ok := asInt(m["version"]); ok && v == version {
			return m, nil
		}
	}
	return nil, fmt.Errorf("has no version %d", version)
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

func (r *runState) fetchDataset(ctx context.Context) error {
	path := "projects/" + r.project() + "/datasets/" + segment(r.datasetKey)
	raw, err := r.m.api.get(ctx, path)
	if err != nil {
		return notFound(err, "LaunchDarkly dataset %q was not found in project %q", r.datasetKey, r.m.projectKey)
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

// evaluationConfigBody is the offline-evaluation create or PATCH snapshot.
// Create includes key and name (name defaults to the key). PATCH omits both
// so a UI rename survives the next reconcile. parameters, promptSnippets,
// tools and criteria are always present, including empty values, so a PATCH
// can clear a previous pin.
func (r *runState) evaluationConfigBody(includeKey bool) map[string]any {
	g := r.gen
	body := map[string]any{"generationProvider": g.Provider, "generationModel": g.Model}
	if includeKey {
		body["key"] = r.args.Key
		body["name"] = r.args.Key
	}
	params := g.Parameters
	if params == nil {
		params = map[string]any{}
	}
	body["parameters"] = params
	switch {
	case g.Instructions != "":
		body["messages"] = []any{map[string]any{"role": "system", "content": g.Instructions}}
	case len(g.Messages) > 0:
		body["messages"] = g.Messages
	default:
		body["messages"] = []any{}
	}
	snippets := g.PromptSnippets
	if snippets == nil {
		snippets = map[string]string{}
	}
	body["promptSnippets"] = snippets
	tools := make([]any, 0, len(r.tools))
	for _, t := range r.tools {
		if t.inline {
			tools = append(tools, map[string]any{"key": t.key, "schema": cloneJSON(t.schema), "description": t.description, "source": "inline"})
		} else {
			tools = append(tools, map[string]any{"key": t.key, "version": t.version, "source": "library"})
		}
	}
	body["tools"] = tools
	criteria := make([]any, 0, len(r.criteria))
	for _, c := range r.criteria {
		criteria = append(criteria, c.wire())
	}
	body["criteria"] = criteria
	if r.configKey != "" {
		body["configKey"] = r.configKey
		body["variationKey"] = r.variationKey
		body["variationVersion"] = r.variationVersion
	}
	return body
}

func (r *runState) offlineCollection() string {
	return "projects/" + r.project() + "/offline-evaluations"
}

// getOfflineEvaluation returns the evaluation, or (nil, nil) on 404. Any
// other status is returned as-is.
func (r *runState) getOfflineEvaluation(ctx context.Context) (map[string]any, error) {
	raw, err := r.m.api.get(ctx, r.offlineCollection()+"/"+segment(r.args.Key))
	if err != nil {
		var apiErr *APIError
		if errors.As(err, &apiErr) && apiErr.Status == http.StatusNotFound {
			return nil, nil
		}
		return nil, err
	}
	return asObject(raw, "offline evaluation")
}

// ensureEvaluation reconciles by key: GET, POST on 404, otherwise PATCH,
// then the caller creates a run on the legacy by-id path. A POST 409 re-GETs
// and continues on the PATCH path. A PATCH 409 re-GETs and retries once.
func (r *runState) ensureEvaluation(ctx context.Context) error {
	existing, err := r.getOfflineEvaluation(ctx)
	if err != nil {
		return err
	}
	if existing == nil {
		raw, postErr := r.m.api.post(ctx, r.offlineCollection(), r.evaluationConfigBody(true), false)
		if postErr == nil {
			return r.applyEvaluationRef(raw)
		}
		var apiErr *APIError
		if !errors.As(postErr, &apiErr) || apiErr.Status != http.StatusConflict {
			return postErr
		}
		existing, err = r.getOfflineEvaluation(ctx)
		if err != nil || existing == nil {
			if err == nil {
				err = postErr
			}
			return fmt.Errorf("evaluations: %w: offline evaluation %q reported a key conflict but could not be re-fetched: %w", ErrAPI, r.args.Key, err)
		}
	}
	raw, err := r.patchOfflineEvaluation(ctx)
	if err != nil {
		return err
	}
	return r.applyEvaluationRef(raw)
}

// patchOfflineEvaluation sends the snapshot. A 409 optimistic-lock collision
// re-GETs and retries once; a second 409 is an error. The call is not
// idempotent, so a 5xx is not replayed.
func (r *runState) patchOfflineEvaluation(ctx context.Context) (any, error) {
	path := r.offlineCollection() + "/" + segment(r.args.Key)
	body := r.evaluationConfigBody(false)
	raw, err := r.m.api.patch(ctx, path, body, false)
	if err == nil {
		return raw, nil
	}
	var apiErr *APIError
	if !errors.As(err, &apiErr) || apiErr.Status != http.StatusConflict {
		return nil, err
	}
	refetched, reErr := r.getOfflineEvaluation(ctx)
	if reErr != nil || refetched == nil {
		if reErr == nil {
			reErr = err
		}
		return nil, fmt.Errorf("evaluations: %w: offline evaluation %q reported an update conflict but could not be re-fetched: %w", ErrAPI, r.args.Key, reErr)
	}
	raw, retryErr := r.m.api.patch(ctx, path, body, false)
	if retryErr == nil {
		return raw, nil
	}
	var retryAPI *APIError
	if errors.As(retryErr, &retryAPI) && retryAPI.Status == http.StatusConflict {
		return nil, fmt.Errorf("evaluations: %w: offline evaluation %q could not be updated because another writer changed it; retry the run or serialize callers that share this key: %w", ErrAPI, r.args.Key, retryErr)
	}
	return nil, retryErr
}

// applyEvaluationRef reads id and version. A non-empty response key is the
// evaluation key; otherwise the caller key is used. name is ignored, because
// PATCH does not send it and a rename can diverge from the key.
func (r *runState) applyEvaluationRef(raw any) error {
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
	if k, ok := obj["key"].(string); ok && k != "" {
		r.evaluationKey = k
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
		return RunSummary{Criteria: []CriterionSummary{}}, nil
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
		FailedRows: get("failed", "failed_rows"), ErrorRows: get("error", "error_rows"), PendingRows: get("pending", "pending_rows"),
		Criteria: parseCriterionSummaries(obj["criterionSummaries"])}, nil
}

// parseCriterionSummaries keeps the object entries in order. A missing or
// non-list value is an empty list, which is also how LaunchDarkly says it
// does not serve verdicts for the project.
func parseCriterionSummaries(raw any) []CriterionSummary {
	list, _ := raw.([]any)
	out := []CriterionSummary{}
	for _, item := range list {
		m, ok := item.(map[string]any)
		if !ok {
			continue
		}
		count := func(k string) int { n, _ := asInt(m[k]); return n }
		typ, _ := m["criterionType"].(string)
		out = append(out, CriterionSummary{
			CriterionType:     typ,
			PassRateThreshold: asOptionalFloat(m["criterionPassRateThreshold"]),
			PassedRows:        count("passed"),
			FailedRows:        count("failed"),
			ErrorRows:         count("error"),
			PassRate:          asOptionalFloat(m["passRate"]),
			PassRateVerdict:   parseVerdict(m["passRateVerdict"]),
		})
	}
	return out
}

func asOptionalFloat(v any) *float64 {
	f, ok := v.(float64)
	if !ok || math.IsNaN(f) || math.IsInf(f, 0) {
		return nil
	}
	return &f
}

// parseVerdict closes the verdict to pass and fail: any other value, which
// LaunchDarkly may add later, is no verdict rather than a coerced one.
func parseVerdict(v any) EvalVerdict {
	switch s, _ := v.(string); EvalVerdict(s) {
	case EvalVerdictPass, EvalVerdictFail:
		return EvalVerdict(s)
	}
	return ""
}

// expectsVerdict reports whether LaunchDarkly will rule on the run: some
// criterion pinned on its evaluation has a pass rate threshold. It reads the
// summary, not the caller's criteria, because those are the thresholds
// LaunchDarkly rules on, and an empty list means no verdicts are served.
func (s RunSummary) expectsVerdict() bool {
	for _, c := range s.Criteria {
		if c.PassRateThreshold != nil {
			return true
		}
	}
	return false
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

// --- run verdict ---------------------------------------------------------

// unjudgedStates are the terminal states besides COMPLETE: a run in one
// never completes, so it never gets a verdict. TEMPORARY_ERROR and STALLED
// can still reach COMPLETE and are waited on.
var unjudgedStates = map[string]bool{"CANCELLED": true, "PERMANENT_ERROR": true}

// runStatus is one read of the run resource. recognized is the verdict when
// it is pass or fail; present is set for any non-null verdict, so an
// unrecognised one (which LaunchDarkly writes once and never changes) ends
// the wait while a missing one does not.
type runStatus struct {
	state      string
	verdict    any
	present    bool
	recognized EvalVerdict
}

func (r *runState) readRun(ctx context.Context) (runStatus, error) {
	raw, err := r.m.api.get(ctx, r.runsPath()+"/"+segment(r.runID))
	if err != nil {
		return runStatus{}, err
	}
	obj, err := asObject(raw, "evaluation run")
	if err != nil {
		return runStatus{}, err
	}
	var st runStatus
	for _, k := range []string{"id", "evaluationId", "state"} {
		s, err := requireString(obj, k, "evaluation run")
		if err != nil {
			return runStatus{}, err
		}
		if k == "state" {
			st.state = s
		}
	}
	st.verdict = obj["verdict"]
	st.present = st.verdict != nil
	st.recognized = parseVerdict(st.verdict)
	return st, nil
}

// awaitVerdict reads the run until LaunchDarkly has recorded its verdict.
// The verdict is written just after the run completes, so the first read
// can miss it. Any outcome but pass or fail is "", and Passed then falls
// back to the row rule: the rows are already accounted for, so only a
// cancelled context makes this an error.
func (r *runState) awaitVerdict(ctx context.Context) (EvalVerdict, error) {
	deadline := time.Now().Add(verdictWaitTimeout)
	for {
		st, err := r.readRun(ctx)
		if err != nil {
			if ctx.Err() != nil {
				return "", fmt.Errorf("evaluations: verdict read: %w", ctx.Err())
			}
			r.m.logger.Warn("could not read the evaluation run verdict; the result is judged on row counts instead",
				"runId", r.runID, "error", err)
			return "", nil
		}
		switch {
		case st.recognized != "":
			return st.recognized, nil
		case st.present:
			r.m.logger.Warn("evaluation run has a verdict this SDK does not recognise; the result is judged on row counts instead",
				"runId", r.runID, "verdict", fmt.Sprintf("%#v", st.verdict))
			return "", nil
		case unjudgedStates[st.state]:
			return "", nil
		}
		remaining := time.Until(deadline)
		if remaining <= 0 {
			r.m.logger.Warn("evaluation run has a pass rate threshold but no verdict was recorded in time; the result is judged on row counts instead",
				"runId", r.runID, "deadline", verdictWaitTimeout, "state", st.state)
			return "", nil
		}
		if err := r.m.api.sleep(ctx, min(r.interval, remaining)); err != nil {
			return "", fmt.Errorf("evaluations: verdict polling: %w", err)
		}
	}
}
