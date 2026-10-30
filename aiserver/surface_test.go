package aiserver

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// rootSurface pins the exported names of the aiserver package root.
// A new or removed root name fails here, so it shows in review.
var rootSurface = []string{
	"AIConfig", "AgentGraph", "AiConfig", "BuildJudgeTasks", "BuildJudgeTasksArgs", "CallOption", "CallOptions",
	"CallableTools", "CaptureContentEnabled", "CaptureDefault", "CaptureOff", "CaptureOn", "ChatContent", "ChatMessage",
	"Compose", "ComposePrompt", "Config", "ConfigArgs", "Configure", "ContentBlock", "ContentCapture", "ContentText",
	"ContextIdentity", "ContextToMap", "ConversationIDSpanProcessor", "CreateHandler", "DefaultOTLPEndpoint",
	"ErrConfigDisabled", "ErrConfigNotFound", "ErrConfigureLocked", "ErrGraphDisabled", "ErrNoHandler",
	"ErrNotInitialized", "EventChunk", "EventDone", "EventHandoff", "EventNodeDone", "EventNodeStart", "EventType",
	"FlushTelemetry", "FormattingInstructions", "GetClient", "Graph", "GraphDefinition", "GraphEdge", "GraphNode",
	"GraphOptions", "GraphResponse", "GraphSpanName", "GraphStreamEvent", "GraphTopology", "GraphTopologyEdge",
	"HandlerRequest", "HandlerResult", "HandlerRun", "HandlerSpanOptions", "HandlerStreamEvent", "ImageSource",
	"ImageURL", "InitClient", "InitOptions", "InitWithClient", "InputDetails", "InspectConfig", "InspectResult",
	"InvokeFunc", "JudgeConfiguration", "JudgeRef", "JudgeResult", "JudgeRunResult", "JudgeTask", "LDClient", "Message",
	"MessagePart", "Mode", "ModeAgent", "ModeMessages", "Model", "NativeTool", "NewCallOptions", "NewRegistry",
	"NormalizeMode", "OutputFormatInstruction", "PackageName", "PackageVersion", "ParseAiConfig",
	"ParseJSONWithPossibleFences", "ParseTemplate", "ParseUsage", "Prompt", "ProviderHandler", "ProviderResponse",
	"ProvidesFor", "RegisterAiSdkPackage", "Registry", "RegistryEntries", "ReportUsage", "ResolveGraph", "RouteResult",
	"RunJudge", "RunNodeOptions", "SDKOptions", "SelectHandler", "SetChatContent", "SetLDSpanAttributes",
	"SetLDSpanAttributesFor", "SetLogger", "SetToolCallArguments", "SetToolCallResult", "SetUsageSpanAttributes",
	"Shutdown", "SpanCounter", "SpanUsage", "StartHandlerRun", "StreamEvent", "StreamFunc", "TextPart", "TokenUsage",
	"Tool", "ToolCallPart", "ToolFunc", "ToolHandler", "ToolResultString", "TracerPrefix", "Track", "TrackData",
	"TrackGraphEvent", "TraverseVisitor", "VarLD", "VarLDContext", "VariationMeta", "WildcardProvider",
	"WithConversationID", "WithHistory", "WithOutputFormat", "WithSpanAttributes", "WithSpanCounter", "WithVariables",
	"WrapToolHandlers",
}

// exportedNames returns the package-level exported names declared in the
// non-test .go files matching glob.
func exportedNames(t *testing.T, glob string) []string {
	t.Helper()
	files, _ := filepath.Glob(glob)
	fset := token.NewFileSet()
	var out []string
	for _, file := range files {
		if strings.HasSuffix(file, "_test.go") {
			continue
		}
		f, err := parser.ParseFile(fset, file, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		for name := range f.Scope.Objects {
			if ast.IsExported(name) {
				out = append(out, name)
			}
		}
	}
	if len(out) == 0 {
		t.Fatalf("no exported names in %s", glob)
	}
	slices.Sort(out)
	return out
}

func TestRootSurfaceIsPinned(t *testing.T) {
	got := exportedNames(t, "*.go")
	want := slices.Sorted(slices.Values(rootSurface))
	if !slices.Equal(got, want) {
		var added, removed []string
		for _, n := range got {
			if !slices.Contains(want, n) {
				added = append(added, n)
			}
		}
		for _, n := range want {
			if !slices.Contains(got, n) {
				removed = append(removed, n)
			}
		}
		t.Errorf("aiserver root surface changed: added %v, removed %v", added, removed)
	}
}

// No experimental name (skills or evaluations) leaks onto the root.
func TestNoExperimentalNameOnRoot(t *testing.T) {
	root := exportedNames(t, "*.go")
	experimental := exportedNames(t, "experimental/*.go")
	for _, must := range []string{"InitEvaluations", "NewJudge", "NewScorer", "Dataset", "GetSkill", "WriteSkills"} {
		if !slices.Contains(experimental, must) {
			t.Fatalf("experimental surface lost %s; the leak check would be vacuous", must)
		}
	}
	for _, n := range experimental {
		if slices.Contains(root, n) {
			t.Errorf("experimental name %s is on the aiserver root", n)
		}
	}
}
