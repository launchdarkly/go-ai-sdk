// Command examples is the integration entry point: it runs
// one scenario against a real LaunchDarkly environment.
//
//	go run ./examples <scenario> <flag-key> "<user input>"
package main

import (
	"context"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

type scenario func(ctx context.Context, key, input string) error

var scenarios = map[string]scenario{
	"agent":                  agent,
	"openai-only":            openaiOnly,
	"streaming":              streaming,
	"history":                history,
	"conversation":           conversation,
	"graph":                  graph,
	"graph-streaming":        graphStreaming,
	"graph-history":          graphHistory,
	"native-graph":           nativeGraph,
	"native-graph-langchain": nativeGraphLangChain,
	"judge":                  judge,
}

func usage() {
	names := make([]string, 0, len(scenarios))
	for n := range scenarios {
		names = append(names, n)
	}
	sort.Strings(names)
	fmt.Fprintf(os.Stderr, "usage: go run ./examples <%s> <flag-key> [input]\n", strings.Join(names, "|"))
}

func main() { os.Exit(run()) }

func run() int {
	if len(os.Args) < 3 {
		usage()
		return 2
	}
	run, ok := scenarios[os.Args[1]]
	if !ok {
		usage()
		return 2
	}
	input := strings.Join(os.Args[3:], " ")
	ctx := context.Background()
	if _, err := aiserver.InitClient(ctx, aiserver.InitOptions{ServiceName: "go-ai-sdk-examples"}); err != nil {
		fmt.Fprintln(os.Stderr, "Error:", err)
		return 1
	}
	defer func() { _ = aiserver.Shutdown(ctx) }()
	register()
	if err := run(ctx, os.Args[2], input); err != nil {
		fmt.Fprintln(os.Stderr, "Error:", err)
		return 1
	}
	return 0
}
