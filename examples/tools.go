package main

import (
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"sync"
	"time"
)

//go:embed user_preferences.json
var userPreferences []byte

// getPreferences returns stored preferences for args.user_id.
func getPreferences(_ context.Context, args map[string]any) (any, error) {
	var all map[string]any
	if err := json.Unmarshal(userPreferences, &all); err != nil {
		return fmt.Sprintf("Error reading preferences: %v", err), nil
	}
	id, _ := args["user_id"].(string)
	prefs, ok := all[id]
	if !ok {
		return fmt.Sprintf("No preferences found for user %q.", id), nil
	}
	b, _ := json.Marshal(prefs)
	return string(b), nil
}

const llmsTxtURL = "https://launchdarkly.com/docs/llms.txt"

var (
	httpClient = &http.Client{Timeout: 20 * time.Second}
	llmsOnce   sync.Once
	llmsTxt    string
	llmsErr    error
)

func fetch(ctx context.Context, url, accept string) (string, int, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", 0, err
	}
	req.Header.Set("User-Agent", "LD-Docs-Agent/1.0")
	if accept != "" {
		req.Header.Set("Accept", accept)
	}
	resp, err := httpClient.Do(req)
	if err != nil {
		return "", 0, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	return string(body), resp.StatusCode, err
}

// searchLDDocumentation filters the llms.txt index for lines containing every
// query term and returns up to ten matching page links.
func searchLDDocumentation(ctx context.Context, args map[string]any) (any, error) {
	llmsOnce.Do(func() {
		var status int
		llmsTxt, status, llmsErr = fetch(ctx, llmsTxtURL, "")
		if llmsErr == nil && status != http.StatusOK {
			llmsErr = fmt.Errorf("failed to fetch llms.txt: HTTP %d", status)
		}
	})
	if llmsErr != nil {
		return "Search error: " + llmsErr.Error(), nil
	}
	query, _ := args["query"].(string)
	terms := strings.Fields(strings.ToLower(query))
	var hits []string
	for _, line := range strings.Split(llmsTxt, "\n") {
		if !strings.HasPrefix(line, "- [") {
			continue
		}
		lower, all := strings.ToLower(line), true
		for _, t := range terms {
			all = all && strings.Contains(lower, t)
		}
		if all {
			hits = append(hits, line)
			if len(hits) == 10 {
				break
			}
		}
	}
	if len(hits) == 0 {
		return fmt.Sprintf("No documentation results found for %q.", query), nil
	}
	return strings.Join(hits, "\n"), nil
}

var htmlTag = regexp.MustCompile(`<[^>]+>`)

// fetchLDDocumentation fetches a docs page, preferring its Markdown form.
func fetchLDDocumentation(ctx context.Context, args map[string]any) (any, error) {
	url, _ := args["url"].(string)
	url = strings.TrimSuffix(url, "/")
	md := url
	if !strings.HasSuffix(md, ".md") {
		md += ".md"
	}
	text, status, err := fetch(ctx, md, "text/markdown,text/plain,*/*")
	if err != nil {
		return "Fetch error: " + err.Error(), nil
	}
	if status != http.StatusOK {
		html, fstatus, err := fetch(ctx, url, "")
		if err != nil || fstatus != http.StatusOK {
			return fmt.Sprintf("Failed to fetch %q: HTTP %d", url, status), nil
		}
		plain := strings.Join(strings.Fields(htmlTag.ReplaceAllString(html, " ")), " ")
		return truncate(plain, 8000, ""), nil
	}
	return truncate(text, 10000, "\n\n[…content truncated…]"), nil
}

func truncate(s string, n int, suffix string) string {
	if r := []rune(s); len(r) > n {
		return string(r[:n]) + suffix
	}
	return s
}
