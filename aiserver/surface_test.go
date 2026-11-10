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
	"CallableTools", "CaptureDefault", "CaptureOff", "CaptureOn",
	"Compose", "ComposePrompt", "Config", "ConfigArgs", "Configure", "ContentBlock", "ContentCapture", "ContentText",
	"ContextIdentity", "ContextToMap", "ConversationIDSpanProcessor", "CreateHandler", "DefaultOTLPEndpoint",
	"ErrConfigDisabled", "ErrConfigNotFound", "ErrConfigureLocked", "ErrGraphDisabled", "ErrNoHandler",
	"ErrNotInitialized", "EventChunk", "EventDone", "EventHandoff", "EventNodeDone", "EventNodeStart", "EventType",
	"FormattingInstructions", "GetClient", "Graph", "GraphDefinition", "GraphEdge", "GraphNode",
	"GraphOptions", "GraphResponse", "GraphSpanName", "GraphStreamEvent", "GraphTopology", "GraphTopologyEdge",
	"HandlerRequest", "HandlerResult", "HandlerRun", "HandlerSpanOptions", "HandlerStreamEvent", "ImageSource",
	"ImageURL", "InitClient", "InitOptions", "InitWithClient", "InputDetails", "InspectConfig", "InspectResult",
	"InvokeFunc", "JudgeConfiguration", "JudgeRef", "JudgeResult", "JudgeRunResult", "JudgeTask", "LDClient", "Message",
	"Mode", "ModeAgent", "ModeMessages", "Model", "NativeGraphRunner", "NativeTool", "NewCallOptions", "NewRegistry",
	"NormalizeMode", "OutputFormatInstruction", "PackageName", "PackageVersion", "ParseAiConfig",
	"ParseJSONWithPossibleFences", "ParseTemplate", "ParseUsage", "Prompt", "ProviderHandler", "ProviderResponse",
	"ProvidesFor", "RegisterAiSdkPackage", "Registry", "RegistryEntries", "ReportUsage", "ResolveGraph", "RouteResult",
	"RunJudge", "RunNodeOptions", "SDKOptions", "SelectHandler", "SetLDSpanAttributes",
	"SetLDSpanAttributesFor", "SetLogger", "SetUsageSpanAttributes",
	"Shutdown", "SpanUsage", "StartHandlerRun", "StreamEvent", "StreamFunc", "TokenUsage",
	"Tool", "ToolFunc", "ToolHandler", "ToolResultString", "TracerPrefix", "Track", "TrackData",
	"TrackGraphEvent", "TraverseVisitor", "VarLD", "VarLDContext", "VariationMeta", "WildcardProvider",
	"WithConversationID", "WithHistory", "WithOutputFormat", "WithVariables",
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

// No experimental name (shadow requests) leaks onto the root.
func TestNoExperimentalNameOnRoot(t *testing.T) {
	root := exportedNames(t, "*.go")
	experimental := exportedNames(t, "../experimental/shadowrequests/*.go")
	for _, must := range []string{"ShadowConfig", "RunShadow"} {
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

// Agent Skills is its own package; none of its names is on the root.
func TestNoSkillsNameOnRoot(t *testing.T) {
	root := exportedNames(t, "*.go")
	skills := exportedNames(t, "../skills/*.go")
	for _, must := range []string{"GetSkill", "WriteSkills", "SkillStore"} {
		if !slices.Contains(skills, must) {
			t.Fatalf("skills surface lost %s; the leak check would be vacuous", must)
		}
	}
	for _, n := range skills {
		if slices.Contains(root, n) {
			t.Errorf("skills name %s is on the aiserver root", n)
		}
	}
}

// handlerKitNames are the handler-kit and drift names that are not public
// API. Handler packages reach the kit through internal/handlerkit and
// internal/kitbase; none of these may come back onto the root.
var handlerKitNames = []string{
	"BuildJudgeTasks", "BuildJudgeTasksArgs", "CallableTools", "ComposePrompt", "ContentText",
	"ContextIdentity", "ContextToMap", "DefaultOTLPEndpoint", "FormattingInstructions", "GraphSpanName",
	"GraphTopology", "GraphTopologyEdge", "HandlerRun", "HandlerSpanOptions", "ImageURL", "NormalizeMode",
	"OutputFormatInstruction", "ParseAiConfig", "ParseJSONWithPossibleFences", "ParseUsage", "Prompt",
	"RegisterAiSdkPackage", "ReportUsage", "SelectHandler", "SetLDSpanAttributes", "SetLDSpanAttributesFor",
	"SetUsageSpanAttributes", "SpanUsage", "StartHandlerRun", "TracerPrefix", "TrackGraphEvent", "VarLD",
	"VarLDContext", "WildcardProvider", "WithOutputFormat", "WrapToolHandlers",
}

// handlerKitMethods are methods of public types that are not public API.
var handlerKitMethods = []string{
	"GraphDefinition.TrackData", "GraphDefinition.UserContext", "GraphNode.TrackData",
	"NativeTool.Track", "ProvidesFor.IsZero",
}

// exportedMethods returns "Type.Method" for every exported method declared in
// the non-test .go files matching glob.
func exportedMethods(t *testing.T, glob string) []string {
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
		for _, d := range f.Decls {
			fd, ok := d.(*ast.FuncDecl)
			if !ok || fd.Recv == nil || !fd.Name.IsExported() {
				continue
			}
			rt := fd.Recv.List[0].Type
			if s, ok := rt.(*ast.StarExpr); ok {
				rt = s.X
			}
			if id, ok := rt.(*ast.Ident); ok {
				out = append(out, id.Name+"."+fd.Name.Name)
			}
		}
	}
	return out
}

func TestNoHandlerKitNameOnRoot(t *testing.T) {
	root := exportedNames(t, "*.go")
	for _, n := range handlerKitNames {
		if slices.Contains(root, n) {
			t.Errorf("handler-kit name %s is on the aiserver root", n)
		}
	}
	methods := exportedMethods(t, "*.go")
	if !slices.Contains(methods, "AIConfig.Invoke") {
		t.Fatal("method listing lost AIConfig.Invoke; the method check would be vacuous")
	}
	for _, m := range handlerKitMethods {
		if slices.Contains(methods, m) {
			t.Errorf("handler-kit method %s is exported from aiserver", m)
		}
	}
}
