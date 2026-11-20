package shadowrequests

import (
	"context"
	"errors"
	"maps"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/bridge"
)

const (
	usageEvent = "$ld:ai:sdk:usage"
	sdkEvents  = "$ld:ai:sdk:" // usage and SDK info: the caller's, never the shadow's
)

// plainKeys returns the keys, SDK events aside, that a plain
// aiserver.Config Invoke of value sends, with the quality judge resolvable.
func plainKeys(t *testing.T, value map[string]any, handlers []*aiserver.ProviderHandler) []string {
	t.Helper()
	ctx := context.Background()
	f := &shadowLD{value: value, values: map[string]any{"quality": judgeVariation()}}
	if _, err := aiserver.InitWithClient(ctx, f, aiserver.InitOptions{}); err != nil {
		t.Fatal(err)
	}
	_, _ = aiserver.Config(aiserver.ConfigArgs{Key: "cfg", Handlers: handlers}).Invoke(ctx, "hello", shadowUser)
	_ = aiserver.Shutdown(ctx)
	var keys []string
	for _, e := range f.tracks {
		if !strings.HasPrefix(e.event, sdkEvents) {
			keys = append(keys, e.event)
		}
	}
	slices.Sort(keys)
	return keys
}

// plainShadow is the shadow variation as a plain config: no tools, with the
// quality judge.
func plainShadow() map[string]any {
	v := candidate()
	delete(v, "tools")
	v["judgeConfiguration"] = judges(1)
	return v
}

// plainServed is the served variation with the quality judge.
func plainServed() map[string]any {
	v := fallthroughPayload()
	v["judgeConfiguration"] = judges(1)
	return v
}

func shadowKeys(keys []string) []string {
	out := make([]string, len(keys))
	for i, k := range keys {
		out[i] = bridge.ShadowEventKey(k)
	}
	slices.Sort(out)
	return out
}

// split returns the shadow's keys (events with isShadow) and the served
// call's keys, SDK events aside, and the usage helpers reported.
func (f *shadowLD) split() (shadow, served, usage []string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	for _, e := range f.tracks {
		switch {
		case e.event == usageEvent:
			usage = append(usage, e.data["helper"].(string))
		case strings.HasPrefix(e.event, sdkEvents):
		case e.data["isShadow"] == true:
			shadow = append(shadow, e.event)
		default:
			served = append(served, e.event)
		}
	}
	slices.Sort(shadow)
	slices.Sort(served)
	return shadow, served, usage
}

// Every event a shadow sends, B's and its judge's, is on the key a plain
// Invoke sends for the same outcome, mapped into $ld:ai-shadow:. The served
// call's keys are unchanged, and the usage events stay on $ld:ai:sdk:usage.
func TestShadowEventKeys(t *testing.T) {
	failed := func(s *stub) { s.err = errors.New("provider down") }
	for _, o := range []struct {
		name string
		// set makes B's handler produce the outcome; plain makes a plain
		// Invoke's handler produce the same outcome.
		set, plain func(*stub)
	}{
		{"success", func(*stub) {}, func(*stub) {}},
		{"handler failure", failed, failed},
		{"timeout", func(s *stub) { s.delay = time.Second }, failed},
	} {
		for _, deferred := range []bool{false, true} {
			mode := "background"
			if deferred {
				mode = "RunShadow"
			}
			t.Run(o.name+"/"+mode, func(t *testing.T) {
				pj := newJudged()
				o.plain(pj.b)
				wantShadow := shadowKeys(plainKeys(t, plainShadow(), pj.handlers()))
				wantServed := plainKeys(t, plainServed(), newJudged().handlers())

				f := installJudged(t, 1, true)
				j := newJudged()
				o.set(j.b)
				opts := ShadowOptions{Timeout: 50 * time.Millisecond}
				if deferred {
					res := RunShadow(context.Background(), judgedTask(t, mustShadow(t, j.handlers(), opts)), j.handlers())
					if (res.Error == "") != (o.name == "success") {
						t.Fatalf("error %q", res.Error)
					}
				} else {
					invokeAndWait(t, mustShadow(t, j.handlers(), opts))
				}
				time.Sleep(20 * time.Millisecond) // a timed-out handler's late result must track nothing

				shadow, served, usage := f.split()
				if !slices.Equal(shadow, wantShadow) {
					t.Errorf("shadow keys\n got %v\nwant %v", shadow, wantShadow)
				}
				for _, k := range shadow {
					if strings.HasPrefix(k, "$ld:ai:") {
						t.Errorf("shadow key %s is under $ld:ai:", k)
					}
				}
				if !slices.Equal(served, wantServed) {
					t.Errorf("served keys\n got %v\nwant %v", served, wantServed)
				}
				for _, k := range served {
					if strings.HasPrefix(k, bridge.ShadowEventPrefix) {
						t.Errorf("served key %s is under %s", k, bridge.ShadowEventPrefix)
					}
				}
				if deferred && !slices.Contains(usage, "experimental.runShadow") {
					t.Errorf("usage %v, want experimental.runShadow on %s", usage, usageEvent)
				}
			})
		}
	}
}

// A shadow judge's own call sends its generation, duration and token events
// on $ld:ai-shadow: keys, with the judge config's own track data and run id
// plus the five shadow fields. B's events and the score carry B's fields and
// A's run id. The served call's judge is unchanged.
func TestShadowJudgeCallEventsCarryTheShadowFields(t *testing.T) {
	for _, deferred := range []bool{false, true} {
		t.Run(map[bool]string{false: "background", true: "RunShadow"}[deferred], func(t *testing.T) {
			f := installJudged(t, 1, true)
			j := newJudged()
			var tasks []ShadowTask
			opts := ShadowOptions{}
			if deferred {
				opts.OnTask = func(task ShadowTask) { tasks = append(tasks, task) }
			}
			resp := invokeAndWait(t, mustShadow(t, j.handlers(), opts))
			for _, task := range tasks {
				RunShadow(context.Background(), task, j.handlers())
			}
			runID := resp.TrackData.RunID
			want := map[string]any{"isShadow": true, "primaryVariationKey": "gpt-main",
				"shadowRequestId": "shadow-1", "ownerKind": "member", "ownerId": "m-1"}

			judgeCall := map[string]bool{}
			var servedJudgeCall []string
			f.mu.Lock()
			tracks := slices.Clone(f.tracks)
			f.mu.Unlock()
			for _, e := range tracks {
				d := e.data
				switch {
				case strings.HasPrefix(e.event, sdkEvents):
				case strings.HasPrefix(e.event, bridge.ShadowEventPrefix) && d["configKey"] == "quality":
					judgeCall[e.event] = true
					if d["variationKey"] != "judge-v1" || d["runId"] == runID || d["runId"] == "" {
						t.Errorf("%s: shadow judge call data %v, want the judge's variation and its own run id", e.event, d)
					}
					for k, v := range want {
						if d[k] != v {
							t.Errorf("%s: %s = %v, want %v", e.event, k, d[k], v)
						}
					}
				case strings.HasPrefix(e.event, bridge.ShadowEventPrefix):
					// B's run events and the judge's score.
					if d["configKey"] != "cfg" || d["variationKey"] != "claude-candidate" || d["runId"] != runID {
						t.Errorf("%s: shadow data %v, want B's fields and A's run id", e.event, d)
					}
					if e.event == shadowJudgeMetric && d["judgeConfigKey"] != "quality" {
						t.Errorf("score data %v", d)
					}
					for k, v := range want {
						if d[k] != v {
							t.Errorf("%s: %s = %v, want %v", e.event, k, d[k], v)
						}
					}
				default:
					if d["configKey"] == "quality" {
						servedJudgeCall = append(servedJudgeCall, e.event)
					}
					for k := range want {
						if _, ok := d[k]; ok {
							t.Errorf("served event %s carries %s", e.event, k)
						}
					}
				}
			}
			wantCall := []string{"$ld:ai-shadow:duration:total", "$ld:ai-shadow:generation:success",
				"$ld:ai-shadow:tokens:input", "$ld:ai-shadow:tokens:output", "$ld:ai-shadow:tokens:total"}
			if got := slices.Sorted(maps.Keys(judgeCall)); !slices.Equal(got, wantCall) {
				t.Errorf("shadow judge call keys %v, want %v", got, wantCall)
			}
			slices.Sort(servedJudgeCall)
			if !slices.Equal(servedJudgeCall, []string{"$ld:ai:duration:total", "$ld:ai:generation:success",
				"$ld:ai:tokens:input", "$ld:ai:tokens:output", "$ld:ai:tokens:total"}) {
				t.Errorf("served judge call keys %v", servedJudgeCall)
			}
		})
	}
}

// A judge's metric key maps into $ld:ai-shadow: on a shadow whatever its
// form, never twice. The served call's judge sends it as configured.
func TestShadowJudgeMetricKeyMapping(t *testing.T) {
	for _, tc := range []struct{ key, shadow string }{
		{"my-relevance", "$ld:ai-shadow:my-relevance"},
		{"$ld:ai:judge:relevance", "$ld:ai-shadow:judge:relevance"},
		{"$ld:ai-shadow:judge:relevance", "$ld:ai-shadow:judge:relevance"},
	} {
		t.Run(tc.key, func(t *testing.T) {
			f := installJudged(t, 1, true)
			f.values["quality"].(map[string]any)["evaluationMetricKey"] = tc.key
			invokeAndWait(t, mustShadow(t, newJudged().handlers(), ShadowOptions{}))
			var shadow, served int
			for _, e := range f.metrics(tc.shadow) {
				if e.data["isShadow"] == true {
					shadow++
				}
			}
			for _, e := range f.metrics(tc.key) {
				if e.data["isShadow"] == nil {
					served++
				}
			}
			if shadow != 1 || served != 1 {
				t.Fatalf("shadow scores on %s: %d, served scores on %s: %d", tc.shadow, shadow, tc.key, served)
			}
			f.mu.Lock()
			defer f.mu.Unlock()
			for _, e := range f.tracks {
				if strings.Count(e.event, bridge.ShadowEventPrefix) > 1 {
					t.Errorf("key prefixed twice: %s", e.event)
				}
			}
		})
	}
}
