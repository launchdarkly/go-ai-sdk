// Package layout holds module-wide tests of the package layout.
package layout

import (
	"os/exec"
	"slices"
	"strings"
	"testing"
)

const module = "github.com/launchdarkly/go-ai-sdk"

// minPackages is the number of packages in the module. Raise it in the same
// change that adds a package, so a listing that silently drops packages fails.
const minPackages = 27

// Stable packages never depend on experimental code, directly or through
// another package.
func TestNoExperimentalImports(t *testing.T) {
	// The module path, not ./..., because go test runs in this directory.
	out, err := exec.Command("go", "list", "-f", "{{.ImportPath}}{{range .Deps}} {{.}}{{end}}", module+"/...").Output()
	if err != nil {
		t.Fatalf("go list: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(out)), "\n")

	var listed []string
	for _, l := range lines {
		listed = append(listed, strings.Fields(l)[0])
	}
	if len(listed) < minPackages {
		t.Fatalf("go list scope wrong: got %d packages, want >= %d", len(listed), minPackages)
	}
	for _, p := range []string{"aiserver", "evaluations", "experimental/shadowrequests", "internal/sdkapi"} {
		if !slices.Contains(listed, module+"/"+p) {
			t.Fatalf("go list scope wrong: %s/%s not listed", module, p)
		}
	}

	for _, v := range experimentalViolations(lines) {
		t.Error(v)
	}
}

// experimentalViolations reads `go list` lines ("pkg dep dep ...") and reports
// each stable package with an experimental package of this module among its
// deps. Other modules' experimental packages (grpc's, say) are not ours.
func experimentalViolations(lines []string) []string {
	var out []string
	for _, l := range lines {
		f := strings.Fields(l)
		if len(f) == 0 || !isStable(f[0]) {
			continue
		}
		for _, dep := range f[1:] {
			if strings.HasPrefix(dep, module+"/") && hasExperimentalElem(dep) {
				out = append(out, f[0]+" depends on "+dep)
			}
		}
	}
	return out
}

func isStable(pkg string) bool {
	return !hasExperimentalElem(pkg) && pkg != module+"/examples" && !strings.HasPrefix(pkg, module+"/examples/")
}

func hasExperimentalElem(pkg string) bool {
	return slices.Contains(strings.Split(pkg, "/"), "experimental")
}

func TestExperimentalViolationsFires(t *testing.T) {
	const (
		ai   = module + "/aiserver"
		ev   = module + "/evaluations"
		x    = module + "/internal/x"
		shad = module + "/experimental/shadowrequests"
	)
	cases := []struct {
		name  string
		lines []string
		want  int
	}{
		{"direct", []string{ai + " " + shad}, 1},
		{"indirect only", []string{ev + " " + x + " " + shad, x + " " + shad}, 2},
		{"package-level experimental", []string{module + "/skills " + module + "/skills/experimental"}, 1},
		{"clean", []string{ai + " " + x, ev + " " + ai}, 0},
		{"experimental imports core", []string{shad + " " + ai, module + "/experimental/other " + shad}, 0},
		{"examples may import experimental", []string{module + "/examples " + shad}, 0},
		{"substring is not an element", []string{ai + " " + module + "/notexperimental"}, 0},
		{"another module's experimental", []string{ai + " google.golang.org/grpc/experimental/stats"}, 0},
	}
	for _, c := range cases {
		if got := experimentalViolations(c.lines); len(got) != c.want {
			t.Errorf("%s: got %d violations %v, want %d", c.name, len(got), got, c.want)
		}
	}
}
