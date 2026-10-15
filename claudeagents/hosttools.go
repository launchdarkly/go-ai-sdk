package claudeagents

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
)

// Host tool limits, matching Claude Code where it documents one.
const (
	bashDefaultTimeout = 2 * time.Minute
	bashMaxTimeout     = 10 * time.Minute
	maxToolOutput      = 30000    // bytes of Bash / Grep / Glob output returned to the model
	maxFileBytes       = 10 << 20 // largest file Read scans or Edit/Grep/NotebookEdit loads
	readDefaultLimit   = 2000     // lines
	maxLineChars       = 2000     // longer lines are cut in Read and Grep output
	globMaxResults     = 100
	binarySniffBytes   = 8000
)

// hostTool is a Claude Code built-in this package executes on the host. It
// is offered to the model as a function tool under the Claude Code name and
// input schema. run returns the model-facing text; a non-nil error makes the
// result an is_error tool result (text, or the error message when text is
// empty) and marks the tool span ERROR. Errors never fail the run.
type hostTool struct {
	description string
	schema      map[string]any
	run         func(ctx context.Context, w *workspace, input json.RawMessage) (string, error)
}

var hostTools = map[string]hostTool{
	"Bash": {
		description: "Executes a bash command in the working directory and returns its combined stdout and stderr. " +
			"The command runs with the host's environment and is not sandboxed. Optional timeout in milliseconds (default 120000, max 600000).",
		schema: object([]string{"command"}, props{
			"command":     str("The command to execute"),
			"timeout":     num("Optional timeout in milliseconds (max 600000)"),
			"description": str("Clear, concise description of what this command does in 5-10 words"),
		}),
		run: runBash,
	},
	"Read": {
		description: "Reads a file inside the working directory. Output is line-numbered (cat -n format). " +
			"By default up to 2000 lines from the start; lines longer than 2000 characters are truncated.",
		schema: object([]string{"file_path"}, props{
			"file_path": str("Path of the file to read (absolute inside the working directory, or relative to it)"),
			"offset":    num("The line number to start reading from (1-based)"),
			"limit":     num("The number of lines to read"),
		}),
		run: runRead,
	},
	"Edit": {
		description: "Performs an exact string replacement in a file inside the working directory. " +
			"The file must have been read with Read in this run first. " +
			"Fails if old_string is not found, or is found more than once unless replace_all is true.",
		schema: object([]string{"file_path", "old_string", "new_string"}, props{
			"file_path":   str("Path of the file to modify"),
			"old_string":  str("The text to replace"),
			"new_string":  str("The text to replace it with (must be different from old_string)"),
			"replace_all": map[string]any{"type": "boolean", "description": "Replace all occurrences of old_string (default false)"},
		}),
		run: runEdit,
	},
	"Write": {
		description: "Writes a file inside the working directory, creating parent directories and overwriting any existing file. " +
			"An existing file must have been read with Read in this run first.",
		schema: object([]string{"file_path", "content"}, props{
			"file_path": str("Path of the file to write"),
			"content":   str("The content to write to the file"),
		}),
		run: runWrite,
	},
	"Glob": {
		description: `Finds files by glob pattern inside the working directory ("**" matches any number of directories, ` +
			`"{a,b}" alternatives). Returns matching paths sorted by modification time, newest first (at most 100).`,
		schema: object([]string{"pattern"}, props{
			"pattern": str(`The glob pattern to match files against, e.g. "**/*.go"`),
			"path":    str("The directory to search in (default: the working directory)"),
		}),
		run: runGlob,
	},
	"Grep": {
		description: "Searches file contents inside the working directory with a regular expression (Go RE2 syntax). " +
			`output_mode "files_with_matches" (default) lists files, "content" shows matching lines, "count" shows match counts.`,
		schema: object([]string{"pattern"}, props{
			"pattern":     str("The regular expression to search for in file contents"),
			"path":        str("File or directory to search in (default: the working directory)"),
			"glob":        str(`Glob that file names must match, e.g. "*.js" or "src/**/*.{ts,tsx}"`),
			"output_mode": map[string]any{"type": "string", "enum": []any{"content", "files_with_matches", "count"}, "description": "Output mode (default files_with_matches)"},
			"-i":          map[string]any{"type": "boolean", "description": "Case insensitive search"},
			"-n":          map[string]any{"type": "boolean", "description": `Show line numbers in "content" mode (default true)`},
			"-A":          num(`Number of lines to show after each match ("content" mode)`),
			"-B":          num(`Number of lines to show before each match ("content" mode)`),
			"-C":          num(`Number of lines to show before and after each match ("content" mode)`),
			"head_limit":  num("Limit output to the first N lines or entries"),
		}),
		run: runGrep,
	},
	"TodoWrite": {
		description: "Creates and updates the structured task list for the current run. Send the whole list each time.",
		schema: object([]string{"todos"}, props{
			"todos": map[string]any{"type": "array", "description": "The updated todo list", "items": object([]string{"content", "status"}, props{
				"content":    str("The task, imperative form"),
				"status":     map[string]any{"type": "string", "enum": []any{"pending", "in_progress", "completed"}},
				"activeForm": str("The task, present continuous form"),
			})},
		}),
		run: runTodoWrite,
	},
	"NotebookEdit": {
		description: "Replaces, inserts or deletes a cell in a Jupyter notebook (.ipynb) inside the working directory.",
		schema: object([]string{"notebook_path", "new_source"}, props{
			"notebook_path": str("Path of the .ipynb file"),
			"cell_id":       str(`ID of the cell to edit (or "cell-N" for the 0-based index). Insert places the new cell after it, or first when omitted`),
			"new_source":    str("The new source for the cell"),
			"cell_type":     map[string]any{"type": "string", "enum": []any{"code", "markdown"}, "description": "The cell type; required for insert"},
			"edit_mode":     map[string]any{"type": "string", "enum": []any{"replace", "insert", "delete"}, "description": "Edit mode (default replace)"},
		}),
		run: runNotebookEdit,
	},
}

type props = map[string]any

func object(required []string, p props) map[string]any {
	req := make([]any, len(required))
	for i, r := range required {
		req[i] = r
	}
	return map[string]any{"type": "object", "properties": p, "required": req}
}

func str(desc string) map[string]any { return map[string]any{"type": "string", "description": desc} }
func num(desc string) map[string]any { return map[string]any{"type": "number", "description": desc} }

// workspace is one run's host state: the confined working directory, the
// files read this run and the TodoWrite list. Tool calls within a run
// execute one at a time.
type workspace struct {
	dir   string // absolute WorkDir as configured
	real  string // dir with symlinks resolved, so absolute paths under either form are accepted
	root  *os.Root
	seen  map[string]time.Time // rel path → modification time when last read or written by a tool
	todos []todo
}

func openWorkspace(dir string) (*workspace, error) {
	if dir == "" {
		return nil, errors.New("claudeagents: no working directory for built-in tools (set Options.WorkDir)")
	}
	root, err := os.OpenRoot(dir)
	if err != nil {
		return nil, fmt.Errorf("claudeagents: open working directory: %w", err)
	}
	real, err := filepath.EvalSymlinks(dir)
	if err != nil {
		real = dir
	}
	return &workspace{dir: dir, real: real, root: root, seen: map[string]time.Time{}}, nil
}

func (w *workspace) close() {
	if w != nil {
		_ = w.root.Close()
	}
}

// rel maps a model-supplied path to a local path under the root. os.Root
// then rejects any symlink that leaves the root.
func (w *workspace) rel(p string) (string, error) {
	if p == "" {
		return "", errors.New("a path is required")
	}
	if slices.Contains(strings.Split(filepath.ToSlash(p), "/"), "..") {
		return "", fmt.Errorf("path %q: \"..\" is not allowed", p)
	}
	if filepath.IsAbs(p) {
		r, ok := within(w.dir, p)
		if !ok {
			r, ok = within(w.real, p)
		}
		if !ok {
			return "", fmt.Errorf("path %q is outside the working directory %s", p, w.dir)
		}
		return r, nil
	}
	if p = filepath.Clean(p); !filepath.IsLocal(p) {
		return "", fmt.Errorf("path %q is outside the working directory %s", p, w.dir)
	}
	return p, nil
}

func within(dir, p string) (string, bool) {
	r, err := filepath.Rel(dir, filepath.Clean(p))
	return r, err == nil && filepath.IsLocal(r)
}

// display is the absolute path reported back to the model.
func (w *workspace) display(rel string) string { return filepath.Join(w.dir, rel) }

// remember records rel's current modification time after a tool read or
// wrote it.
func (w *workspace) remember(rel string) {
	if fi, err := w.root.Stat(rel); err == nil {
		w.seen[rel] = fi.ModTime()
	}
}

// checkRead is Claude Code's read-before-edit rule: an existing file may be
// changed only after this run read it, and only if it has not changed since.
func (w *workspace) checkRead(rel string) error {
	fi, err := w.root.Stat(rel)
	if err != nil {
		return nil // missing: the tool reports it, or Write creates it
	}
	at, ok := w.seen[rel]
	switch {
	case !ok:
		return fmt.Errorf("file %s has not been read yet; read it first before writing to it", w.display(rel))
	case !fi.ModTime().Equal(at):
		return fmt.Errorf("file %s has been modified since it was read; read it again before writing to it", w.display(rel))
	}
	return nil
}

// readSmall loads a regular file of at most maxFileBytes.
func (w *workspace) readSmall(rel string) ([]byte, fs.FileMode, error) {
	fi, err := w.root.Stat(rel)
	if err != nil {
		return nil, 0, err
	}
	if !fi.Mode().IsRegular() {
		return nil, 0, fmt.Errorf("%s is not a regular file", w.display(rel))
	}
	if fi.Size() > maxFileBytes {
		return nil, 0, fmt.Errorf("%s is %d bytes, larger than the %d byte limit", w.display(rel), fi.Size(), maxFileBytes)
	}
	b, err := w.root.ReadFile(rel)
	return b, fi.Mode().Perm(), err
}

func decode(input json.RawMessage, v any) error {
	if len(input) == 0 {
		input = []byte("{}")
	}
	if err := json.Unmarshal(input, v); err != nil {
		return fmt.Errorf("invalid input: %w", err)
	}
	return nil
}

// capWriter keeps the first max bytes written and counts the rest.
type capWriter struct {
	buf     []byte
	max     int
	dropped int
}

func (c *capWriter) Write(p []byte) (int, error) {
	n := min(len(p), c.max-len(c.buf))
	c.buf = append(c.buf, p[:n]...)
	c.dropped += len(p) - n
	return len(p), nil
}

func (c *capWriter) String() string {
	s := strings.ToValidUTF8(string(c.buf), "")
	if c.dropped > 0 {
		s += fmt.Sprintf("\n\n... [output truncated: %d more bytes]", c.dropped)
	}
	return s
}

func capOutput(s string) string {
	c := &capWriter{max: maxToolOutput}
	_, _ = io.WriteString(c, s)
	return c.String()
}

func cutLine(s string) string {
	if r := []rune(s); len(r) > maxLineChars {
		return string(r[:maxLineChars]) + "... [line truncated]"
	}
	return s
}

func runBash(ctx context.Context, w *workspace, input json.RawMessage) (string, error) {
	var in struct {
		Command string  `json:"command"`
		Timeout float64 `json:"timeout"`
	}
	if err := decode(input, &in); err != nil {
		return "", err
	}
	if strings.TrimSpace(in.Command) == "" {
		return "", errors.New("command is required")
	}
	timeout := bashDefaultTimeout
	if in.Timeout > 0 {
		timeout = min(time.Duration(in.Timeout*float64(time.Millisecond)), bashMaxTimeout)
	}
	runCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	cmd := exec.CommandContext(runCtx, "/bin/bash", "-c", in.Command)
	cmd.Dir = w.dir
	out := &capWriter{max: maxToolOutput}
	cmd.Stdout, cmd.Stderr = out, out
	cmd.WaitDelay = 2 * time.Second // background children holding the pipe cannot block the run
	killProcessGroup(cmd)
	err := cmd.Run()
	text := out.String()
	var exit *exec.ExitError
	switch {
	case ctx.Err() != nil: // the run itself was cancelled
		return text, ctx.Err()
	case runCtx.Err() != nil:
		return text + fmt.Sprintf("\nCommand timed out after %s", timeout), fmt.Errorf("bash: timed out after %s", timeout)
	case errors.As(err, &exit):
		return text + fmt.Sprintf("\nExit code %d", exit.ExitCode()), fmt.Errorf("bash: exit code %d", exit.ExitCode())
	case err != nil:
		return "", fmt.Errorf("bash: %w", err)
	}
	if text == "" {
		text = "(no output)"
	}
	return text, nil
}

func runRead(_ context.Context, w *workspace, input json.RawMessage) (string, error) {
	var in struct {
		FilePath string  `json:"file_path"`
		Offset   float64 `json:"offset"`
		Limit    float64 `json:"limit"`
	}
	if err := decode(input, &in); err != nil {
		return "", err
	}
	rel, err := w.rel(in.FilePath)
	if err != nil {
		return "", err
	}
	f, err := w.root.Open(rel)
	if err != nil {
		return "", err
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		return "", err
	}
	if !fi.Mode().IsRegular() {
		return "", fmt.Errorf("%s is not a regular file", w.display(rel))
	}
	br := bufio.NewReader(io.LimitReader(f, maxFileBytes))
	if head, _ := br.Peek(binarySniffBytes); bytes.IndexByte(head, 0) >= 0 {
		return "", fmt.Errorf("%s is a binary file", w.display(rel))
	}
	w.seen[rel] = fi.ModTime()
	offset, limit := max(int(in.Offset), 1), int(in.Limit)
	if limit <= 0 {
		limit = readDefaultLimit
	}
	var b strings.Builder
	n := 0 // lines read
	for n < offset-1+limit {
		line, err := br.ReadString('\n')
		if line != "" {
			if n++; n >= offset {
				fmt.Fprintf(&b, "%6d\t%s\n", n, cutLine(strings.TrimRight(line, "\r\n")))
			}
		}
		if err != nil {
			break
		}
	}
	switch {
	case n == 0:
		return "(the file is empty)", nil
	case b.Len() == 0:
		return fmt.Sprintf("(the file has %d lines; offset %d is past the end)", n, offset), nil
	case fi.Size() > maxFileBytes:
		b.WriteString("... [only the first 10 MiB of this file can be read]\n")
	}
	return b.String(), nil
}

func runEdit(_ context.Context, w *workspace, input json.RawMessage) (string, error) {
	var in struct {
		FilePath   string `json:"file_path"`
		OldString  string `json:"old_string"`
		NewString  string `json:"new_string"`
		ReplaceAll bool   `json:"replace_all"`
	}
	if err := decode(input, &in); err != nil {
		return "", err
	}
	rel, err := w.rel(in.FilePath)
	if err != nil {
		return "", err
	}
	if in.OldString == "" {
		return "", errors.New("old_string must not be empty (use Write to create a file)")
	}
	if in.OldString == in.NewString {
		return "", errors.New("old_string and new_string are the same")
	}
	if err := w.checkRead(rel); err != nil {
		return "", err
	}
	data, perm, err := w.readSmall(rel)
	if err != nil {
		return "", err
	}
	s := string(data)
	n := strings.Count(s, in.OldString)
	switch {
	case n == 0:
		return "", fmt.Errorf("old_string was not found in %s", w.display(rel))
	case n > 1 && !in.ReplaceAll:
		return "", fmt.Errorf("old_string matches %d times in %s; add surrounding context to make it unique, or set replace_all", n, w.display(rel))
	case !in.ReplaceAll:
		n = 1
	}
	if err := w.root.WriteFile(rel, []byte(strings.Replace(s, in.OldString, in.NewString, n)), perm); err != nil {
		return "", err
	}
	w.remember(rel)
	return fmt.Sprintf("The file %s has been updated (%d replacement(s)).", w.display(rel), n), nil
}

func runWrite(_ context.Context, w *workspace, input json.RawMessage) (string, error) {
	var in struct {
		FilePath string  `json:"file_path"`
		Content  *string `json:"content"`
	}
	if err := decode(input, &in); err != nil {
		return "", err
	}
	if in.Content == nil {
		return "", errors.New("content is required")
	}
	rel, err := w.rel(in.FilePath)
	if err != nil {
		return "", err
	}
	if rel == "." {
		return "", errors.New("file_path names the working directory itself")
	}
	if err := w.checkRead(rel); err != nil {
		return "", err
	}
	if err := w.root.MkdirAll(filepath.Dir(rel), 0o755); err != nil {
		return "", err
	}
	_, statErr := w.root.Lstat(rel)
	if err := w.root.WriteFile(rel, []byte(*in.Content), 0o644); err != nil {
		return "", err
	}
	w.remember(rel)
	if statErr == nil {
		return fmt.Sprintf("The file %s has been overwritten.", w.display(rel)), nil
	}
	return fmt.Sprintf("File created successfully at: %s", w.display(rel)), nil
}

// expandBraces expands {a,b} alternatives, e.g. "*.{ts,tsx}" → "*.ts", "*.tsx".
func expandBraces(p string) []string {
	open := strings.IndexByte(p, '{')
	if open < 0 {
		return []string{p}
	}
	depth, start := 0, open+1
	var alts []string
	for i := open; i < len(p); i++ {
		switch p[i] {
		case '{':
			depth++
		case ',':
			if depth == 1 {
				alts, start = append(alts, p[start:i]), i+1
			}
		case '}':
			if depth--; depth == 0 {
				alts = append(alts, p[start:i])
				var out []string
				for _, a := range alts {
					out = append(out, expandBraces(p[:open]+a+p[i+1:])...)
				}
				return out
			}
		}
	}
	return []string{p} // unbalanced: literal
}

// globMatch matches a slash-separated name against a pattern in which "**"
// is a whole segment matching zero or more segments.
func globMatch(pat, name []string) bool {
	if len(pat) == 0 {
		return len(name) == 0
	}
	if pat[0] == "**" {
		for i := 0; i <= len(name); i++ {
			if globMatch(pat[1:], name[i:]) {
				return true
			}
		}
		return false
	}
	if len(name) == 0 {
		return false
	}
	ok, _ := path.Match(pat[0], name[0])
	return ok && globMatch(pat[1:], name[1:])
}

// compileGlob expands braces and validates each segment.
func compileGlob(pattern string) ([][]string, error) {
	var out [][]string
	for _, p := range expandBraces(filepath.ToSlash(pattern)) {
		segs := strings.Split(strings.Trim(p, "/"), "/")
		for _, s := range segs {
			if _, err := path.Match(s, ""); err != nil {
				return nil, fmt.Errorf("invalid glob pattern %q: %w", pattern, err)
			}
		}
		out = append(out, segs)
	}
	return out, nil
}

func matchAny(pats [][]string, name string) bool {
	segs := strings.Split(name, "/")
	for _, p := range pats {
		if globMatch(p, segs) {
			return true
		}
	}
	return false
}

type foundFile struct {
	rel string // slash-separated, relative to the root
	mod time.Time
}

// walkFiles visits regular files and symlinks under base (a root-relative
// slash path), skipping .git directories. sub is the path relative to base.
// Symlinked directories are not followed.
func (w *workspace) walkFiles(base string, visit func(rel, sub string, d fs.DirEntry)) error {
	return fs.WalkDir(w.root.FS(), base, func(p string, d fs.DirEntry, err error) error {
		switch {
		case err != nil && p == base:
			return err
		case err != nil:
			return nil // unreadable entry: skip it
		case d.IsDir() && d.Name() == ".git" && p != base:
			return fs.SkipDir
		case d.IsDir():
			return nil
		}
		sub := p
		if base != "." {
			sub = strings.TrimPrefix(p, base+"/")
		}
		visit(p, sub, d)
		return nil
	})
}

func sortByModTime(files []foundFile) {
	slices.SortFunc(files, func(a, b foundFile) int {
		if c := b.mod.Compare(a.mod); c != 0 {
			return c
		}
		return strings.Compare(a.rel, b.rel)
	})
}

func (w *workspace) searchBase(p string) (string, error) {
	if p == "" {
		return ".", nil
	}
	rel, err := w.rel(p)
	return filepath.ToSlash(rel), err
}

func runGlob(_ context.Context, w *workspace, input json.RawMessage) (string, error) {
	var in struct {
		Pattern string `json:"pattern"`
		Path    string `json:"path"`
	}
	if err := decode(input, &in); err != nil {
		return "", err
	}
	if in.Pattern == "" {
		return "", errors.New("pattern is required")
	}
	base, err := w.searchBase(in.Path)
	if err != nil {
		return "", err
	}
	if filepath.IsAbs(in.Pattern) { // an absolute pattern must lie under the search base
		rel, err := w.rel(in.Pattern)
		if err != nil {
			return "", err
		}
		if in.Pattern = filepath.ToSlash(rel); base != "." {
			if !strings.HasPrefix(in.Pattern, base+"/") {
				return "", fmt.Errorf("pattern is outside path %s", w.display(base))
			}
			in.Pattern = strings.TrimPrefix(in.Pattern, base+"/")
		}
	}
	pats, err := compileGlob(in.Pattern)
	if err != nil {
		return "", err
	}
	var files []foundFile
	err = w.walkFiles(base, func(rel, sub string, d fs.DirEntry) {
		if matchAny(pats, sub) {
			f := foundFile{rel: rel}
			if fi, err := d.Info(); err == nil {
				f.mod = fi.ModTime()
			}
			files = append(files, f)
		}
	})
	if err != nil {
		return "", err
	}
	if len(files) == 0 {
		return "No files found", nil
	}
	sortByModTime(files)
	var b strings.Builder
	for i, f := range files {
		if i == globMaxResults {
			b.WriteString("(Results are truncated. Consider using a more specific path or pattern.)\n")
			break
		}
		b.WriteString(w.display(f.rel) + "\n")
	}
	return capOutput(b.String()), nil
}

func runGrep(_ context.Context, w *workspace, input json.RawMessage) (string, error) {
	var in struct {
		Pattern    string `json:"pattern"`
		Path       string `json:"path"`
		Glob       string `json:"glob"`
		OutputMode string `json:"output_mode"`
		IgnoreCase bool   `json:"-i"`
		LineNums   *bool  `json:"-n"`
		After      int    `json:"-A"`
		Before     int    `json:"-B"`
		Context    int    `json:"-C"`
		HeadLimit  int    `json:"head_limit"`
	}
	if err := decode(input, &in); err != nil {
		return "", err
	}
	if in.Pattern == "" {
		return "", errors.New("pattern is required")
	}
	expr := in.Pattern
	if in.IgnoreCase {
		expr = "(?i)" + expr
	}
	re, err := regexp.Compile(expr)
	if err != nil {
		return "", fmt.Errorf("invalid regular expression: %w", err)
	}
	mode := cmpOr(in.OutputMode, "files_with_matches")
	if !slices.Contains([]string{"content", "files_with_matches", "count"}, mode) {
		return "", fmt.Errorf("unknown output_mode %q", mode)
	}
	var globs [][]string
	if in.Glob != "" {
		if globs, err = compileGlob(in.Glob); err != nil {
			return "", err
		}
	}
	before, after := max(in.Before, in.Context, 0), max(in.After, in.Context, 0)
	base, err := w.searchBase(in.Path)
	if err != nil {
		return "", err
	}
	fi, err := w.root.Stat(base)
	if err != nil {
		return "", err
	}

	var files []foundFile
	var lines []string // content / count entries, in walk order
	search := func(rel, sub string) {
		if globs != nil && !matchAny(globs, sub) && !matchAny(globs, path.Base(sub)) {
			return
		}
		data, _, err := w.readSmall(rel)
		if err != nil || bytes.IndexByte(data[:min(len(data), binarySniffBytes)], 0) >= 0 {
			return // unreadable, too large, escaping symlink, or binary
		}
		text := strings.Split(strings.TrimSuffix(string(data), "\n"), "\n")
		var hits []int
		for i := range text {
			text[i] = strings.TrimSuffix(text[i], "\r")
			if re.MatchString(text[i]) {
				hits = append(hits, i)
			}
		}
		count := len(hits)
		if count == 0 {
			return
		}
		if mode == "content" {
			lines = append(lines, contentLines(w.display(rel), text, hits, before, after, in.LineNums == nil || *in.LineNums)...)
		}
		if mode == "count" {
			lines = append(lines, fmt.Sprintf("%s:%d", w.display(rel), count))
		}
		f := foundFile{rel: rel}
		if st, err := w.root.Stat(rel); err == nil {
			f.mod = st.ModTime()
		}
		files = append(files, f)
	}
	if fi.IsDir() {
		err = w.walkFiles(base, func(rel, sub string, _ fs.DirEntry) { search(rel, sub) })
	} else {
		search(base, path.Base(base))
	}
	if err != nil {
		return "", err
	}

	if mode == "files_with_matches" {
		sortByModTime(files)
		for _, f := range files {
			lines = append(lines, w.display(f.rel))
		}
	}
	if len(lines) == 0 {
		return "No matches found", nil
	}
	total := len(lines)
	if in.HeadLimit > 0 && total > in.HeadLimit {
		lines = lines[:in.HeadLimit]
	}
	out := strings.Join(lines, "\n")
	if mode == "files_with_matches" {
		out = fmt.Sprintf("Found %d file(s)\n%s", total, out)
	}
	if len(lines) < total {
		out += fmt.Sprintf("\n... [%d more not shown (head_limit)]", total-len(lines))
	}
	return capOutput(out), nil
}

// contentLines formats one file's matches like ripgrep: "path:N:line" for a
// match, "path-N-line" for a context line, and "--" between groups that are
// not adjacent when context was asked for.
func contentLines(name string, text []string, hits []int, before, after int, nums bool) []string {
	isHit := map[int]bool{}
	for _, h := range hits {
		isHit[h] = true
	}
	var out []string
	last := -1 // last line index emitted
	for _, h := range hits {
		from, to := max(h-before, last+1), min(h+after, len(text)-1)
		if last >= 0 && from > last+1 && before+after > 0 {
			out = append(out, "--")
		}
		for i := from; i <= to; i++ {
			sep := "-"
			if isHit[i] {
				sep = ":"
			}
			if nums {
				out = append(out, fmt.Sprintf("%s%s%d%s%s", name, sep, i+1, sep, cutLine(text[i])))
			} else {
				out = append(out, name+sep+cutLine(text[i]))
			}
		}
		last = max(last, to)
	}
	return out
}

func cmpOr(s, def string) string {
	if s == "" {
		return def
	}
	return s
}

type todo struct {
	Content    string `json:"content"`
	Status     string `json:"status"`
	ActiveForm string `json:"activeForm,omitempty"`
}

func runTodoWrite(_ context.Context, w *workspace, input json.RawMessage) (string, error) {
	var in struct {
		Todos *[]todo `json:"todos"`
	}
	if err := decode(input, &in); err != nil {
		return "", err
	}
	if in.Todos == nil {
		return "", errors.New("todos is required")
	}
	for i, t := range *in.Todos {
		if strings.TrimSpace(t.Content) == "" {
			return "", fmt.Errorf("todos[%d]: content is required", i)
		}
		if !slices.Contains([]string{"pending", "in_progress", "completed"}, t.Status) {
			return "", fmt.Errorf("todos[%d]: status must be pending, in_progress or completed", i)
		}
	}
	w.todos = slices.Clone(*in.Todos)
	return fmt.Sprintf("Todos have been modified successfully (%d item(s)). Continue to use the todo list to track your progress.", len(w.todos)), nil
}

var cellIndexRE = regexp.MustCompile(`^cell-(\d+)$`)

func runNotebookEdit(_ context.Context, w *workspace, input json.RawMessage) (string, error) {
	var in struct {
		NotebookPath string  `json:"notebook_path"`
		CellID       string  `json:"cell_id"`
		NewSource    *string `json:"new_source"`
		CellType     string  `json:"cell_type"`
		EditMode     string  `json:"edit_mode"`
	}
	if err := decode(input, &in); err != nil {
		return "", err
	}
	rel, err := w.rel(in.NotebookPath)
	if err != nil {
		return "", err
	}
	if !strings.EqualFold(filepath.Ext(rel), ".ipynb") {
		return "", errors.New("notebook_path must be a .ipynb file")
	}
	mode := cmpOr(in.EditMode, "replace")
	if !slices.Contains([]string{"replace", "insert", "delete"}, mode) {
		return "", fmt.Errorf("unknown edit_mode %q", mode)
	}
	if in.CellType != "" && in.CellType != "code" && in.CellType != "markdown" {
		return "", fmt.Errorf("cell_type must be code or markdown, not %q", in.CellType)
	}
	if in.NewSource == nil && mode != "delete" {
		return "", errors.New("new_source is required")
	}
	data, perm, err := w.readSmall(rel)
	if err != nil {
		return "", err
	}
	var nb map[string]any
	if err := json.Unmarshal(data, &nb); err != nil {
		return "", fmt.Errorf("%s is not valid notebook JSON: %w", w.display(rel), err)
	}
	cells, ok := nb["cells"].([]any)
	if !ok {
		return "", fmt.Errorf("%s has no cells array", w.display(rel))
	}

	idx := -1
	if in.CellID != "" {
		for i, c := range cells {
			if m, _ := c.(map[string]any); m != nil && m["id"] == in.CellID {
				idx = i
			}
		}
		if m := cellIndexRE.FindStringSubmatch(in.CellID); idx < 0 && m != nil {
			if i, _ := strconv.Atoi(m[1]); i < len(cells) {
				idx = i
			}
		}
		if idx < 0 {
			return "", fmt.Errorf("cell %q not found in %s", in.CellID, w.display(rel))
		}
	} else if mode != "insert" {
		return "", fmt.Errorf("cell_id is required for edit_mode %s", mode)
	}

	var msg string
	switch mode {
	case "replace":
		cell, ok := cells[idx].(map[string]any)
		if !ok {
			return "", fmt.Errorf("cell %q is malformed", in.CellID)
		}
		cell["source"] = *in.NewSource
		if in.CellType != "" {
			cell["cell_type"] = in.CellType
		}
		if cell["cell_type"] == "code" {
			cell["outputs"], cell["execution_count"] = []any{}, nil // stale after a source change
		} else {
			delete(cell, "outputs")
			delete(cell, "execution_count")
		}
		msg = fmt.Sprintf("Updated cell %s", in.CellID)
	case "insert":
		if in.CellType == "" {
			return "", errors.New("cell_type is required for edit_mode insert")
		}
		cell := map[string]any{"cell_type": in.CellType, "source": *in.NewSource, "metadata": map[string]any{}}
		if in.CellType == "code" {
			cell["outputs"], cell["execution_count"] = []any{}, nil
		}
		major, _ := nb["nbformat"].(float64)
		minor, _ := nb["nbformat_minor"].(float64)
		if major > 4 || (major == 4 && minor >= 5) { // cell ids are part of nbformat 4.5+
			cell["id"] = uuid.NewString()[:8]
		}
		nb["cells"] = slices.Insert(cells, idx+1, any(cell))
		msg = fmt.Sprintf("Inserted %s cell at index %d", in.CellType, idx+1)
		if id, ok := cell["id"].(string); ok {
			msg += " with id " + id
		}
	case "delete":
		nb["cells"] = slices.Delete(cells, idx, idx+1)
		msg = fmt.Sprintf("Deleted cell %s", in.CellID)
	}
	out, err := json.MarshalIndent(nb, "", " ")
	if err != nil {
		return "", err
	}
	if err := w.root.WriteFile(rel, append(out, '\n'), perm); err != nil {
		return "", err
	}
	return msg + " in " + w.display(rel), nil
}
