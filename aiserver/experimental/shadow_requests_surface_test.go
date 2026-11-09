package experimental_test

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
)

// shadowSurface is the documented shadow requests surface of the
// experimental entry point.
var shadowSurface = []string{"RunShadow", "ShadowAIConfig", "ShadowConfig", "ShadowOptions", "ShadowResult", "ShadowTask", "ShadowVariables"}

func TestShadowSurfaceIsExactlyTheDocumentedSet(t *testing.T) {
	got := exportedDecls(t, "shadow_requests.go")
	if !slices.Equal(got, shadowSurface) {
		t.Fatalf("experimental shadow requests surface:\n got %v\nwant %v", got, shadowSurface)
	}
	root := exportedDecls(t, coreFiles(t)...)
	for _, name := range shadowSurface {
		if slices.Contains(root, name) {
			t.Errorf("%s leaked onto the aiserver package root", name)
		}
	}
}

// The whole experimental entry point is shadow requests; a new file's names
// must be pinned too.
func TestExperimentalSurfaceIsPinned(t *testing.T) {
	all, _ := filepath.Glob("*.go")
	var files []string
	for _, f := range all {
		if !strings.HasSuffix(f, "_test.go") {
			files = append(files, f)
		}
	}
	if got := exportedDecls(t, files...); !slices.Equal(got, shadowSurface) {
		t.Errorf("experimental entry point has unpinned names:\n got %v\nwant %v", got, shadowSurface)
	}
}

// Core never names experimental code.
func TestCoreDoesNotImportExperimentalCode(t *testing.T) {
	fset := token.NewFileSet()
	for _, file := range coreFiles(t) {
		f, err := parser.ParseFile(fset, file, nil, parser.ImportsOnly)
		if err != nil {
			t.Fatal(err)
		}
		for _, imp := range f.Imports {
			path, _ := strconv.Unquote(imp.Path.Value)
			if strings.Contains(path, "experimental") {
				t.Fatalf("%s imports %s", file, path)
			}
		}
	}
}

func exportedDecls(t *testing.T, files ...string) []string {
	t.Helper()
	fset := token.NewFileSet()
	var out []string
	for _, file := range files {
		f, err := parser.ParseFile(fset, file, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		for name, obj := range f.Scope.Objects {
			if ast.IsExported(name) && obj.Kind != ast.Bad {
				out = append(out, name)
			}
		}
	}
	slices.Sort(out)
	return out
}

func coreFiles(t *testing.T) []string {
	t.Helper()
	all, _ := filepath.Glob("../*.go")
	var out []string
	for _, f := range all {
		if !strings.HasSuffix(f, "_test.go") {
			out = append(out, f)
		}
	}
	if len(out) == 0 {
		t.Fatal("no core files found")
	}
	return out
}
