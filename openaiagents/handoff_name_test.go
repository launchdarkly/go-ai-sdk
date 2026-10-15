package openaiagents

import (
	"strings"
	"testing"
)

// Handoff tool names stay unique and within 64 characters: keys that sanitize
// alike (a.b, a_b) or overflow the limit get a hash suffix of the key.
func TestHandoffToolNames(t *testing.T) {
	root := &agent{name: "root"}
	long := strings.Repeat("x", 80)
	for _, k := range []struct{ name, key string }{{"a_b", "a.b"}, {"a_b", "a_b"}, {long, long}, {"plain", "plain"}} {
		root.addHandoff(&agent{name: k.name}, k.key, "")
	}
	seen := map[string]bool{}
	for _, h := range root.handoffs {
		if len(h.tool) > 64 || seen[h.tool] {
			t.Errorf("bad handoff tool %q (len %d)", h.tool, len(h.tool))
		}
		seen[h.tool] = true
	}
	if root.handoffs[0].tool != handoffPrefix+"a_b" || root.handoffs[3].tool != handoffPrefix+"plain" {
		t.Errorf("ordinary names changed: %q %q", root.handoffs[0].tool, root.handoffs[3].tool)
	}
}
