package handlerkit

import (
	"reflect"
	"testing"
)

// Every kit function needs its aiserver registration, with the asserted type.
func TestEveryKitFunctionIsRegistered(t *testing.T) {
	for name, f := range map[string]any{
		"startHandlerRun": startHandlerRun, "composePrompt": composePrompt,
		"CallableTools": CallableTools, "ReportUsage": ReportUsage, "RegisterAiSdkPackage": RegisterAiSdkPackage,
		"SetLDSpanAttributesFor": SetLDSpanAttributesFor, "WrapToolHandlers": WrapToolHandlers,
		"ParseUsage": ParseUsage, "NormalizeMode": NormalizeMode, "ParseAiConfig": ParseAiConfig,
		"TrackGraphEvent": TrackGraphEvent, "SelectHandler": SelectHandler, "ImageURL": ImageURL,
		"ContentText": ContentText, "NodeTrackData": NodeTrackData, "GraphTrackData": GraphTrackData,
		"GraphUserContext": GraphUserContext, "TrackNativeTool": TrackNativeTool,
	} {
		if reflect.ValueOf(f).IsNil() {
			t.Errorf("handlerkit.%s is nil: aiserver/kitexport.go does not register it with this type", name)
		}
	}
}
