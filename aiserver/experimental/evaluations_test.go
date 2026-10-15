package experimental_test

import (
	"context"
	"errors"
	"slices"
	"testing"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/aiserver/experimental"
)

// The evaluations entry point is reachable, and callable,
// from the experimental package, not only declared there. InitEvaluations
// does no network I/O, so it runs here against a stub client.
func TestEvaluationsWrappersAreCallable(t *testing.T) {
	ctx := context.Background()
	// A client exists, so the usage event is sent now rather than held for
	// the next test's client.
	fake := &fakeLD{}
	if _, err := aiserver.InitWithClient(ctx, fake, aiserver.InitOptions{}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = aiserver.Shutdown(ctx) })
	t.Setenv("LD_API_TOKEN", "")
	t.Setenv("LD_SDK_KEY", "")

	if _, err := experimental.InitEvaluations(experimental.EvaluationsOptions{}); !errors.Is(err, experimental.ErrEvalMissingCredential) {
		t.Fatalf("no API token: %v", err)
	}
	evals, err := experimental.InitEvaluations(experimental.EvaluationsOptions{APIToken: "api-test", SDKKey: "sdk-test"})
	if err != nil || evals == nil || evals.SDKKey() != "sdk-test" {
		t.Fatalf("InitEvaluations: %v %v", evals, err)
	}
	if !slices.Contains(fake.usage(), "experimental.initEvaluations") {
		t.Fatalf("InitEvaluations did not report its helper: %v", fake.usage())
	}

	judge, err := experimental.NewJudge("accuracy", nil)
	if err != nil || judge.Key() != "accuracy" {
		t.Fatalf("NewJudge: %v %v", judge, err)
	}
	scorer, err := experimental.NewScorer("exact", func(context.Context, experimental.DatasetRow, any) (float64, error) {
		return 1, nil
	}, nil)
	if err != nil || scorer.Name() != "exact" {
		t.Fatalf("NewScorer: %v %v", scorer, err)
	}
	if criteria := []experimental.Criterion{judge, scorer}; len(criteria) != 2 {
		t.Fatal("judge and scorer must both be criteria")
	}
}
