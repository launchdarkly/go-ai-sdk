package evaluations

import (
	"errors"
	"reflect"
	"testing"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

func existingEval(version int, extra map[string]any) map[string]any {
	body := map[string]any{
		"id": "eval-1", "key": "eval-key", "name": "eval-key", "version": version,
		"generationProvider": "OpenAI", "generationModel": "gpt-4o",
		"messages": []any{}, "tools": []any{}, "criteria": []any{},
	}
	for k, v := range extra {
		body[k] = v
	}
	return body
}

func TestRunAlwaysPatchesExistingOfflineEvaluation(t *testing.T) {
	e := newEnv(t, 1)
	e.route("GET", "projects/proj/offline-evaluations/eval-key", resp{body: existingEval(3, nil)})
	e.route("PATCH", "projects/proj/offline-evaluations/eval-key", resp{body: existingEval(4, nil)})
	res := mustRun(t, e.module(), baseArgs(handler("OpenAI", aiserver.ModeMessages, nil)))
	if res.URL != "https://app.launchdarkly.com/projects/proj/ai/evaluations/eval-1/runs/run-1" {
		t.Fatalf("url %s", res.URL)
	}
	assertSequence(t, e,
		"GET projects/proj/datasets/ds", rowsPath,
		offlineGet, "PATCH projects/proj/offline-evaluations/eval-key",
		runPath, summaryPath)
	assertNoURL(t, e, "PUT ")
	if e.count(offlinePost) != 0 {
		t.Fatalf("existing evaluation was created: %v", e.sequence())
	}
	patch := e.body("PATCH", "projects/proj/offline-evaluations/eval-key").(map[string]any)
	if _, has := patch["key"]; has {
		t.Fatal("PATCH must omit key")
	}
	if _, has := patch["name"]; has {
		t.Fatal("PATCH must omit name")
	}
	if !reflect.DeepEqual(patch["parameters"], map[string]any{}) || !reflect.DeepEqual(patch["promptSnippets"], map[string]any{}) ||
		!reflect.DeepEqual(patch["tools"], []any{}) || !reflect.DeepEqual(patch["criteria"], []any{}) {
		t.Fatalf("patch snapshot %s", toJSON(patch))
	}
	for _, k := range []string{"configKey", "variationKey", "variationVersion"} {
		if _, has := patch[k]; has {
			t.Fatalf("generation-only PATCH must omit %s, not clear it: %s", k, toJSON(patch))
		}
	}
	ev := e.ld.events(generationEventName)[0].Data
	if ev["evaluationVersion"] != 4.0 || ev["evaluationKey"] != "eval-key" {
		t.Fatalf("event identity %v", ev)
	}
}

func TestRunHandlesCreateConflictByRegetAndPatch(t *testing.T) {
	e := newEnv(t, 1)
	e.route("GET", "projects/proj/offline-evaluations/eval-key",
		resp{status: 404, body: map[string]any{"message": "not found"}},
		resp{body: existingEval(3, nil)})
	e.route("POST", "projects/proj/offline-evaluations", resp{status: 409, body: map[string]any{"message": "exists"}})
	e.route("PATCH", "projects/proj/offline-evaluations/eval-key", resp{body: existingEval(4, nil)})
	mustRun(t, e.module(), baseArgs(handler("OpenAI", aiserver.ModeMessages, nil)))
	assertSequence(t, e,
		"GET projects/proj/datasets/ds", rowsPath,
		offlineGet, offlinePost, offlineGet,
		"PATCH projects/proj/offline-evaluations/eval-key",
		runPath, summaryPath)
	post := e.body("POST", "projects/proj/offline-evaluations").(map[string]any)
	if post["key"] != "eval-key" || post["name"] != "eval-key" {
		t.Fatalf("create body %s", toJSON(post))
	}
}

func TestRunRaisesWhenCreateConflictRegetIs404(t *testing.T) {
	e := newEnv(t, 1)
	e.route("GET", "projects/proj/offline-evaluations/eval-key", resp{status: 404}, resp{status: 404})
	e.route("POST", "projects/proj/offline-evaluations", resp{status: 409, body: map[string]any{"message": "exists"}})
	runErr(t, e.module(), baseArgs(handler("OpenAI", aiserver.ModeMessages, nil)), ErrAPI, "key conflict but could not be re-fetched")
	if e.count("PATCH ") != 0 {
		t.Fatalf("patched after a failed re-GET: %v", e.sequence())
	}
}

func TestRunRethrowsNon404OnByKeyGet(t *testing.T) {
	e := newEnv(t, 1)
	e.route("GET", "projects/proj/offline-evaluations/eval-key", resp{status: 403, body: "forbidden"})
	err := runErr(t, e.module(), baseArgs(handler("OpenAI", aiserver.ModeMessages, nil)), ErrAPI, "403")
	var apiErr *APIError
	if !errors.As(err, &apiErr) || apiErr.Status != 403 {
		t.Fatalf("err = %v", err)
	}
	if e.count(offlinePost) != 0 || e.count("PATCH ") != 0 {
		t.Fatalf("wrote after a forbidden GET: %v", e.sequence())
	}
}

func TestRunDoesNotRetryReconcilePatchAfter5xx(t *testing.T) {
	e := newEnv(t, 1)
	e.route("GET", "projects/proj/offline-evaluations/eval-key", resp{body: existingEval(3, nil)})
	e.route("PATCH", "projects/proj/offline-evaluations/eval-key",
		resp{status: 503, body: "unavailable"},
		resp{body: existingEval(4, nil)})
	runErr(t, e.module(), baseArgs(handler("OpenAI", aiserver.ModeMessages, nil)), ErrAPI, "503")
	if e.count("PATCH ") != 1 {
		t.Fatalf("PATCH calls: %v", e.sequence())
	}
}

func TestRunRetriesPatchConflictOnce(t *testing.T) {
	e := newEnv(t, 1)
	e.route("GET", "projects/proj/offline-evaluations/eval-key", resp{body: existingEval(3, map[string]any{"name": "Friendly title"})})
	e.route("PATCH", "projects/proj/offline-evaluations/eval-key",
		resp{status: 409, body: map[string]any{"message": "version conflict"}},
		resp{body: existingEval(5, map[string]any{"name": "Friendly title"})})
	mustRun(t, e.module(), baseArgs(handler("OpenAI", aiserver.ModeMessages, nil)))
	assertSequence(t, e,
		"GET projects/proj/datasets/ds", rowsPath,
		offlineGet, "PATCH projects/proj/offline-evaluations/eval-key",
		offlineGet, "PATCH projects/proj/offline-evaluations/eval-key",
		runPath, summaryPath)
	n := 0
	for _, r := range e.requests() {
		if r.Method != "PATCH" {
			continue
		}
		n++
		if _, has := r.Body.(map[string]any)["name"]; has {
			t.Fatal("PATCH retry must still omit name")
		}
	}
	if n != 2 {
		t.Fatalf("PATCH calls = %d", n)
	}
}

func TestRunRaisesWhenPatchConflictsTwice(t *testing.T) {
	e := newEnv(t, 1)
	e.route("GET", "projects/proj/offline-evaluations/eval-key", resp{body: existingEval(3, nil)})
	e.route("PATCH", "projects/proj/offline-evaluations/eval-key", resp{status: 409, body: map[string]any{"message": "version conflict"}})
	runErr(t, e.module(), baseArgs(handler("OpenAI", aiserver.ModeMessages, nil)), ErrAPI, "another writer changed it")
}

func TestRunWrapsFailedRegetAfterPatch409(t *testing.T) {
	e := newEnv(t, 1)
	e.route("GET", "projects/proj/offline-evaluations/eval-key",
		resp{body: existingEval(3, nil)},
		resp{status: 403, body: "forbidden"})
	e.route("PATCH", "projects/proj/offline-evaluations/eval-key", resp{status: 409, body: map[string]any{"message": "version conflict"}})
	err := runErr(t, e.module(), baseArgs(handler("OpenAI", aiserver.ModeMessages, nil)), ErrAPI, "update conflict but could not be re-fetched")
	var apiErr *APIError
	if !errors.As(err, &apiErr) || apiErr.Status != 403 {
		t.Fatalf("err = %v", err)
	}
	if e.count("PATCH ") != 1 {
		t.Fatalf("PATCH calls: %v", e.sequence())
	}
}

func TestRunRequiresIntegerEvaluationVersion(t *testing.T) {
	e := newEnv(t, 1)
	e.route("POST", "projects/proj/offline-evaluations", resp{status: 201, body: map[string]any{"id": "eval-1", "name": "Display", "key": nil}})
	runErr(t, e.module(), baseArgs(handler("OpenAI", aiserver.ModeMessages, nil)), ErrInvalidResponse, `missing integer field "version"`)
}

func TestRunUsesResponseKeyNotName(t *testing.T) {
	e := newEnv(t, 1)
	e.route("POST", "projects/proj/offline-evaluations", resp{status: 201, body: map[string]any{
		"id": "eval-1", "key": "from-key", "name": "Display", "version": 2}})
	mustRun(t, e.module(), baseArgs(handler("OpenAI", aiserver.ModeMessages, nil)))
	ev := e.ld.events(generationEventName)[0].Data
	if ev["evaluationKey"] != "from-key" || ev["evaluationVersion"] != 2.0 {
		t.Fatalf("event identity %v", ev)
	}
}

func TestRunUsesCallerKeyWhenResponseKeyIsNull(t *testing.T) {
	e := newEnv(t, 1)
	e.route("POST", "projects/proj/offline-evaluations", resp{status: 201, body: map[string]any{
		"id": "eval-1", "key": nil, "name": "Display", "version": 2}})
	mustRun(t, e.module(), baseArgs(handler("OpenAI", aiserver.ModeMessages, nil)))
	if got := e.ld.events(generationEventName)[0].Data["evaluationKey"]; got != "eval-key" {
		t.Fatalf("evaluationKey = %v", got)
	}
}

func TestRunRejectsBoolEvaluationVersion(t *testing.T) {
	e := newEnv(t, 1)
	e.route("POST", "projects/proj/offline-evaluations", resp{status: 201, body: map[string]any{
		"id": "eval-1", "key": "eval-key", "version": true}})
	runErr(t, e.module(), baseArgs(handler("OpenAI", aiserver.ModeMessages, nil)), ErrInvalidResponse, "version")
}

func TestRunAcceptsDottedOfflineEvaluationKey(t *testing.T) {
	e := newEnv(t, 1)
	args := baseArgs(handler("OpenAI", aiserver.ModeMessages, nil))
	args.Key = "a.b_c-1"
	mustRun(t, e.module(), args)
	post := e.body("POST", "projects/proj/offline-evaluations").(map[string]any)
	if post["key"] != "a.b_c-1" || post["name"] != "a.b_c-1" {
		t.Fatalf("create body %s", toJSON(post))
	}
	if e.count("GET projects/proj/offline-evaluations/a.b_c-1") != 1 {
		t.Fatalf("sequence %v", e.sequence())
	}
}
