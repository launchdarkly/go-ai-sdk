package aiserver

import (
	"os"
	"os/exec"
	"strings"
	"testing"
)

// parsing skill content is a security boundary the SDK does not cross,
// so no YAML package is in the import graph of aiserver, test dependencies
// included ("in any group"). The graph is the toolchain's own
// answer, so an indirect dependency counts as much as a direct one.
func TestNoYAMLPackageInTheTransitiveImportGraph(t *testing.T) {
	goBin, err := exec.LookPath("go")
	if err != nil {
		t.Skip("the go command is not on PATH")
	}
	packages := []string{
		"github.com/launchdarkly/go-ai-sdk/aiserver",
	}
	cmd := exec.Command(goBin, append([]string{"list", "-deps", "-test"}, packages...)...)
	// Offline and read-only: the graph comes from go.mod and the module cache.
	cmd.Env = append(os.Environ(), "GOPROXY=off", "GOFLAGS=-mod=readonly")
	out, err := cmd.Output()
	if err != nil {
		var stderr string
		if ee, ok := err.(*exec.ExitError); ok {
			stderr = string(ee.Stderr)
		}
		t.Fatalf("go list failed: %v\n%s", err, stderr)
	}
	deps := strings.Fields(string(out))
	if len(deps) < len(packages) {
		t.Fatalf("go list returned %d packages; the check would pass vacuously", len(deps))
	}
	for _, dep := range deps {
		if strings.Contains(strings.ToLower(dep), "yaml") {
			t.Errorf("%s is in the import graph of aiserver", dep)
		}
	}
}
