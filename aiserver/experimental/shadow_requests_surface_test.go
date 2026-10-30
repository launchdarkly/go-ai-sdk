package experimental_test

import (
	"slices"
	"testing"
)

// shadowSurface is the documented shadow requests surface of the
// experimental entry point.
var shadowSurface = []string{"RunShadow", "ShadowAIConfig", "ShadowConfig", "ShadowOptions", "ShadowResult", "ShadowTask"}

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
