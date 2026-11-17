// Package typesafe is the TypeSafe Jev judge handler for aiserver
// (providesFor TypeSafe/messages). It calls the TypeSafe System One REST API
// directly; there is no TypeSafe SDK for Go.
package typesafe

import (
	"context"
	"errors"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/handlerkit"
	"github.com/launchdarkly/go-ai-sdk/internal/sdkapi"
)

const (
	// PackageName is the Go import path, registered for `$ld:ai:sdk:info`.
	PackageName = "github.com/launchdarkly/go-ai-sdk/typesafe"
	// PackageVersion is this package's release version.
	PackageVersion = "1.0.0" // x-release-please-version
	// SpanName is the root span name.
	SpanName = "invoke_agent"
)

// errNotImplemented marks the test-first stub.
// ponytail: stub until the tests are approved; replaced by the implementation.
var errNotImplemented = errors.New("typesafe: not implemented")

func init() { handlerkit.RegisterAiSdkPackage(PackageName, PackageVersion) }

// Options configure NewHandler.
type Options struct {
	// CaptureContent gates the judged state and Jev's answers on spans.
	// CaptureDefault follows aiserver.Configure, else on.
	CaptureContent aiserver.ContentCapture
}

// NewHandler returns the TypeSafe Jev judge handler (createTypesafeHandler).
// It reads TYPESAFE_API_KEY, TYPESAFE_BASE_URL and TYPESAFE_DEFAULT_MODEL
// when it runs.
func NewHandler(opts Options) (*aiserver.ProviderHandler, error) {
	return sdkapi.CreateHandler(aiserver.ProvidesFor{}, func(context.Context, aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
		return aiserver.HandlerResult{}, errNotImplemented
	}, nil, opts.CaptureContent), nil
}
