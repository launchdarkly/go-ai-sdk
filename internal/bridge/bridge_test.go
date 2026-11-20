package bridge

import "testing"

func TestShadowEventKey(t *testing.T) {
	for in, want := range map[string]string{
		"$ld:ai:tokens:input":           "$ld:ai-shadow:tokens:input",
		"$ld:ai:generation:error":       "$ld:ai-shadow:generation:error",
		"$ld:ai:judge:relevance":        "$ld:ai-shadow:judge:relevance",
		"my-relevance":                  "$ld:ai-shadow:my-relevance",
		"$ld:ai-shadow:judge:relevance": "$ld:ai-shadow:judge:relevance",
		"$ld:ai-shadow:my-relevance":    "$ld:ai-shadow:my-relevance",
	} {
		if got := ShadowEventKey(in); got != want {
			t.Errorf("ShadowEventKey(%q) = %q, want %q", in, got, want)
		}
	}
}
