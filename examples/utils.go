package main

import (
	"bytes"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
)

func randomHex(n int) string {
	b := make([]byte, n)
	_, _ = rand.Read(b)
	return fmt.Sprintf("%x", b)
}

// newContext returns a fresh user context.
func newContext() ldcontext.Context { return ldcontext.New(randomHex(6)) }

// newMultiContext returns a user + organization multi-context.
func newMultiContext() ldcontext.Context {
	return ldcontext.NewMulti(
		ldcontext.NewWithKind("organization", "example-org:west%region"),
		ldcontext.New("example-user-"+randomHex(4)),
	)
}

// newConversationID is distinct per run so runs do not merge into one
// conversation in LaunchDarkly.
func newConversationID(label string) string { return label + "-" + randomHex(4) }

// writeOutput persists data as JSON under ./output.
func writeOutput(data any) error {
	if err := os.MkdirAll("output", 0o755); err != nil {
		return err
	}
	b, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return err
	}
	name := strings.NewReplacer(":", "-", ".", "-").Replace(time.Now().UTC().Format(time.RFC3339Nano)) + ".json"
	return os.WriteFile(filepath.Join("output", name), b, 0o644)
}

// solidColorPNGBase64 encodes a solid-colour PNG so multimodal examples need
// no binary fixture.
func solidColorPNGBase64(c color.RGBA, size int) string {
	img := image.NewRGBA(image.Rect(0, 0, size, size))
	for y := range size {
		for x := range size {
			img.Set(x, y, c)
		}
	}
	var buf bytes.Buffer
	_ = png.Encode(&buf, img)
	return base64.StdEncoding.EncodeToString(buf.Bytes())
}

func printJSON(v any) {
	b, _ := json.MarshalIndent(v, "", "  ")
	fmt.Println(string(b))
}
