package claudeagents

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"go.opentelemetry.io/otel/codes"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

func newWS(t *testing.T) (*workspace, string) {
	t.Helper()
	dir := t.TempDir()
	w, err := openWorkspace(dir)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(w.close)
	return w, dir
}

func call(t *testing.T, w *workspace, tool string, input any) (string, error) {
	t.Helper()
	b, _ := json.Marshal(input)
	return hostTools[tool].run(bg, w, b)
}

func mustCall(t *testing.T, w *workspace, tool string, input any) string {
	t.Helper()
	out, err := call(t, w, tool, input)
	if err != nil {
		t.Fatalf("%s(%v): %v (%s)", tool, input, err, out)
	}
	return out
}

func writeFile(t *testing.T, p, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func readFile(t *testing.T, p string) string {
	t.Helper()
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

// Every host built-in is offered under its Claude Code name with the Claude
// Code input schema's required fields.
func TestHostToolSchemas(t *testing.T) {
	want := map[string][]string{
		"Bash": {"command"}, "Read": {"file_path"}, "Edit": {"file_path", "old_string", "new_string"},
		"Write": {"file_path", "content"}, "Glob": {"pattern"}, "Grep": {"pattern"}, "TodoWrite": {"todos"},
		"NotebookEdit": {"notebook_path", "new_source"},
	}
	for _, n := range []*aiserver.NativeTool{ClaudeBash, ClaudeRead, ClaudeEdit, ClaudeWrite, ClaudeGlob, ClaudeGrep, ClaudeTodoWrite, ClaudeNotebookEdit} {
		ht, ok := hostTools[n.ToolName]
		if !ok {
			t.Fatalf("%s has no host implementation", n.ToolName)
		}
		var req []string
		for _, r := range ht.schema["required"].([]any) {
			req = append(req, r.(string))
		}
		if !slices.Equal(req, want[n.ToolName]) || ht.description == "" {
			t.Errorf("%s: required=%v", n.ToolName, req)
		}
		for _, r := range req {
			if _, ok := ht.schema["properties"].(map[string]any)[r]; !ok {
				t.Errorf("%s: required %q has no property", n.ToolName, r)
			}
		}
	}
	if len(hostTools) != 8 {
		t.Errorf("host tools = %d", len(hostTools))
	}
}

func TestDefaultWorkDir(t *testing.T) {
	wd, _ := os.Getwd()
	if got := newAgent(Options{}).workDir; got != wd {
		t.Errorf("default WorkDir = %q, want %q", got, wd)
	}
	if got := newAgent(Options{WorkDir: "sub"}).workDir; got != filepath.Join(wd, "sub") {
		t.Errorf("relative WorkDir = %q", got)
	}
}

// Confinement abuse matrix: every file tool rejects each escape, and nothing
// outside the working directory is read or changed.
func TestConfinement(t *testing.T) {
	w, dir := newWS(t)
	outside := t.TempDir()
	secret := filepath.Join(outside, "secret.txt")
	nb := filepath.Join(outside, "nb.ipynb")
	writeFile(t, secret, "TOP SECRET\n")
	writeFile(t, nb, `{"cells":[{"cell_type":"code","source":"x","id":"a"}],"nbformat":4,"nbformat_minor":5}`)
	if err := os.Symlink(secret, filepath.Join(dir, "link.txt")); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(nb, filepath.Join(dir, "link.ipynb")); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(dir, "linkdir")); err != nil {
		t.Fatal(err)
	}
	writeFile(t, filepath.Join(dir, "sub", "ok.txt"), "fine\n")

	attacks := map[string]struct{ file, nb, dir string }{
		"dotdot":           {"../" + filepath.Base(outside) + "/secret.txt", "../" + filepath.Base(outside) + "/nb.ipynb", ".."},
		"dotdot inside":    {"sub/../../" + filepath.Base(outside) + "/secret.txt", "sub/../x.ipynb", "sub/.."},
		"absolute outside": {secret, nb, outside},
		"symlink file":     {"link.txt", "link.ipynb", ""},
		"symlinked parent": {"linkdir/secret.txt", "linkdir/nb.ipynb", "linkdir"},
	}
	for name, a := range attacks {
		t.Run(name, func(t *testing.T) {
			cases := map[string]any{
				"Read":         map[string]any{"file_path": a.file},
				"Edit":         map[string]any{"file_path": a.file, "old_string": "TOP", "new_string": "PWNED"},
				"Write":        map[string]any{"file_path": a.file, "content": "PWNED"},
				"NotebookEdit": map[string]any{"notebook_path": a.nb, "cell_id": "a", "new_source": "PWNED"},
			}
			if a.dir != "" {
				cases["Glob"] = map[string]any{"pattern": "**/*", "path": a.dir}
				cases["Grep"] = map[string]any{"pattern": "SECRET", "path": a.dir}
			}
			for tool, in := range cases {
				out, err := call(t, w, tool, in)
				if err == nil {
					t.Errorf("%s accepted %v: %q", tool, in, out)
				}
				if strings.Contains(out, "TOP SECRET") {
					t.Errorf("%s leaked the secret: %q", tool, out)
				}
			}
		})
	}
	if readFile(t, secret) != "TOP SECRET\n" || strings.Contains(readFile(t, nb), "PWNED") {
		t.Error("a file outside the working directory was modified")
	}
	// A walk from the root does not follow the escaping symlinks either.
	if out := mustCall(t, w, "Grep", map[string]any{"pattern": "SECRET"}); out != "No matches found" {
		t.Errorf("grep followed a symlink out: %q", out)
	}
	if out := mustCall(t, w, "Glob", map[string]any{"pattern": "**/secret.txt"}); out != "No files found" {
		t.Errorf("glob followed a symlink out: %q", out)
	}
	// Absolute paths inside the working directory are accepted.
	if out := mustCall(t, w, "Read", map[string]any{"file_path": filepath.Join(dir, "sub", "ok.txt")}); !strings.Contains(out, "fine") {
		t.Errorf("absolute inside path: %q", out)
	}
}

// The lexical check rejects escapes on its own, before os.Root sees the path
// (each layer is tested separately: TestConfinement exercises os.Root).
func TestRelRejects(t *testing.T) {
	w, dir := newWS(t)
	for _, p := range []string{"", "..", "../x", "a/../b", "a/../../x", "/etc/passwd", filepath.Dir(dir), dir + "/../x"} {
		if r, err := w.rel(p); err == nil {
			t.Errorf("rel(%q) = %q, want error", p, r)
		}
	}
	for p, want := range map[string]string{"a/b": "a/b", "./a": "a", dir: ".", filepath.Join(dir, "x/y"): "x/y", filepath.Join(w.real, "z"): "z"} {
		if r, err := w.rel(p); err != nil || r != filepath.FromSlash(want) {
			t.Errorf("rel(%q) = %q, %v; want %q", p, r, err, want)
		}
	}
}

func TestBash(t *testing.T) {
	w, dir := newWS(t)
	t.Setenv("LD_HOSTTOOL_TEST", "inherited")
	out := mustCall(t, w, "Bash", map[string]any{"command": "pwd -P; echo $LD_HOSTTOOL_TEST; echo err >&2"})
	real, _ := filepath.EvalSymlinks(dir)
	if !strings.Contains(out, real) || !strings.Contains(out, "inherited") || !strings.Contains(out, "err") {
		t.Errorf("bash out = %q", out)
	}
	out, err := call(t, w, "Bash", map[string]any{"command": "echo partial; exit 3"})
	if err == nil || !strings.Contains(out, "partial") || !strings.Contains(out, "Exit code 3") {
		t.Errorf("non-zero exit: out=%q err=%v", out, err)
	}
	if _, err := call(t, w, "Bash", map[string]any{"command": "  "}); err == nil {
		t.Error("empty command accepted")
	}
	if out := mustCall(t, w, "Bash", map[string]any{"command": "true"}); out != "(no output)" {
		t.Errorf("no-output = %q", out)
	}
}

func TestBashTimeout(t *testing.T) {
	w, _ := newWS(t)
	start := time.Now()
	// The background child holds the output pipe; the process-group kill and WaitDelay still return promptly.
	out, err := call(t, w, "Bash", map[string]any{"command": "echo started; sleep 30 & sleep 30", "timeout": 300})
	if err == nil || !strings.Contains(out, "timed out") || !strings.Contains(out, "started") {
		t.Errorf("timeout: out=%q err=%v", out, err)
	}
	if d := time.Since(start); d > 5*time.Second {
		t.Errorf("timeout took %s", d)
	}
	// Cancelling the run stops the command too.
	ctx, cancel := context.WithTimeout(bg, 200*time.Millisecond)
	defer cancel()
	if _, err := hostTools["Bash"].run(ctx, w, json.RawMessage(`{"command":"sleep 30"}`)); !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("ctx cancel: %v", err)
	}
}

func TestBashOutputCap(t *testing.T) {
	w, _ := newWS(t)
	out := mustCall(t, w, "Bash", map[string]any{"command": "head -c 100000 /dev/zero | tr '\\0' a"})
	if !strings.HasPrefix(out, strings.Repeat("a", maxToolOutput)) || !strings.Contains(out, "[output truncated: 70000 more bytes]") {
		t.Errorf("cap: len=%d tail=%q", len(out), out[len(out)-60:])
	}
	if len(out) > maxToolOutput+100 {
		t.Errorf("output not capped: %d", len(out))
	}
}

func TestRead(t *testing.T) {
	w, dir := newWS(t)
	writeFile(t, filepath.Join(dir, "f.txt"), "one\ntwo\r\nthree\nfour")
	if out := mustCall(t, w, "Read", map[string]any{"file_path": "f.txt"}); out != "     1\tone\n     2\ttwo\n     3\tthree\n     4\tfour\n" {
		t.Errorf("read = %q", out)
	}
	if out := mustCall(t, w, "Read", map[string]any{"file_path": "f.txt", "offset": 2, "limit": 2}); out != "     2\ttwo\n     3\tthree\n" {
		t.Errorf("offset/limit = %q", out)
	}
	if out := mustCall(t, w, "Read", map[string]any{"file_path": "f.txt", "offset": 9}); !strings.Contains(out, "has 4 lines") {
		t.Errorf("past end = %q", out)
	}
	writeFile(t, filepath.Join(dir, "long.txt"), strings.Repeat("x", 3000)+"\n")
	if out := mustCall(t, w, "Read", map[string]any{"file_path": "long.txt"}); !strings.Contains(out, "[line truncated]") || len(out) > 2100 {
		t.Errorf("long line len=%d", len(out))
	}
	writeFile(t, filepath.Join(dir, "many.txt"), strings.Repeat("l\n", 2500))
	if out := mustCall(t, w, "Read", map[string]any{"file_path": "many.txt"}); strings.Count(out, "\n") != readDefaultLimit {
		t.Errorf("default limit: %d lines", strings.Count(out, "\n"))
	}
	writeFile(t, filepath.Join(dir, "empty.txt"), "")
	if out := mustCall(t, w, "Read", map[string]any{"file_path": "empty.txt"}); !strings.Contains(out, "empty") {
		t.Errorf("empty = %q", out)
	}
	writeFile(t, filepath.Join(dir, "bin"), "a\x00b")
	for _, in := range []map[string]any{{"file_path": "bin"}, {"file_path": "missing.txt"}, {"file_path": "."}, {}} {
		if _, err := call(t, w, "Read", in); err == nil {
			t.Errorf("read %v: want error", in)
		}
	}
}

func TestEdit(t *testing.T) {
	w, dir := newWS(t)
	p := filepath.Join(dir, "f.go")
	writeFile(t, p, "a := 1\nb := 1\n")
	if err := os.Chmod(p, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := call(t, w, "Edit", map[string]any{"file_path": "f.go", "old_string": "a := 1", "new_string": "a := 2"}); err == nil || !strings.Contains(err.Error(), "not been read") {
		t.Errorf("edit before read: %v", err)
	}
	mustCall(t, w, "Read", map[string]any{"file_path": p})
	mustCall(t, w, "Edit", map[string]any{"file_path": "f.go", "old_string": "a := 1", "new_string": "a := 2"})
	if got := readFile(t, p); got != "a := 2\nb := 1\n" {
		t.Errorf("edit = %q", got)
	}
	if fi, _ := os.Stat(p); fi.Mode().Perm() != 0o600 {
		t.Errorf("mode changed to %v", fi.Mode())
	}
	if _, err := call(t, w, "Edit", map[string]any{"file_path": "f.go", "old_string": " := ", "new_string": "="}); err == nil || !strings.Contains(err.Error(), "2 times") {
		t.Errorf("non-unique: %v", err)
	}
	out := mustCall(t, w, "Edit", map[string]any{"file_path": "f.go", "old_string": " := ", "new_string": "=", "replace_all": true})
	if got := readFile(t, p); got != "a=2\nb=1\n" || !strings.Contains(out, "2 replacement") {
		t.Errorf("replace_all = %q (%s)", got, out)
	}
	for _, in := range []map[string]any{
		{"file_path": "f.go", "old_string": "zzz", "new_string": "y"},
		{"file_path": "f.go", "old_string": "a", "new_string": "a"},
		{"file_path": "f.go", "old_string": "", "new_string": "y"},
		{"file_path": "nope.go", "old_string": "a", "new_string": "b"},
	} {
		if _, err := call(t, w, "Edit", in); err == nil {
			t.Errorf("edit %v: want error", in)
		}
	}
	if got := readFile(t, p); got != "a=2\nb=1\n" {
		t.Errorf("failed edits changed the file: %q", got)
	}
	if err := os.Chtimes(p, time.Time{}, time.Now().Add(time.Hour)); err != nil { // changed outside the tools
		t.Fatal(err)
	}
	if _, err := call(t, w, "Edit", map[string]any{"file_path": "f.go", "old_string": "a=2", "new_string": "a=3"}); err == nil || !strings.Contains(err.Error(), "modified since") {
		t.Errorf("edit after outside change: %v", err)
	}
}

func TestWrite(t *testing.T) {
	w, dir := newWS(t)
	out := mustCall(t, w, "Write", map[string]any{"file_path": "a/b/c.txt", "content": "hi"})
	if readFile(t, filepath.Join(dir, "a/b/c.txt")) != "hi" || !strings.Contains(out, "created") {
		t.Errorf("create: %q", out)
	}
	out = mustCall(t, w, "Write", map[string]any{"file_path": filepath.Join(dir, "a/b/c.txt"), "content": ""})
	if readFile(t, filepath.Join(dir, "a/b/c.txt")) != "" || !strings.Contains(out, "overwritten") {
		t.Errorf("overwrite: %q", out)
	}
	writeFile(t, filepath.Join(dir, "old.txt"), "keep")
	if _, err := call(t, w, "Write", map[string]any{"file_path": "old.txt", "content": "x"}); err == nil || readFile(t, filepath.Join(dir, "old.txt")) != "keep" {
		t.Errorf("overwrite before read: %v", err)
	}
	for _, in := range []map[string]any{{"file_path": "x.txt"}, {"file_path": ".", "content": "x"}, {"file_path": "a", "content": "x"}} {
		if _, err := call(t, w, "Write", in); err == nil {
			t.Errorf("write %v: want error", in)
		}
	}
}

func TestGlob(t *testing.T) {
	w, dir := newWS(t)
	old := time.Now().Add(-time.Hour)
	for i, f := range []string{"main.go", "pkg/a.go", "pkg/deep/b.go", "pkg/c.ts", "web/x.tsx", ".git/HEAD.go"} {
		writeFile(t, filepath.Join(dir, f), "x")
		mt := old.Add(time.Duration(i) * time.Minute)
		_ = os.Chtimes(filepath.Join(dir, f), mt, mt)
	}
	got := strings.Fields(mustCall(t, w, "Glob", map[string]any{"pattern": "**/*.go"}))
	want := []string{filepath.Join(dir, "pkg/deep/b.go"), filepath.Join(dir, "pkg/a.go"), filepath.Join(dir, "main.go")}
	if !slices.Equal(got, want) {
		t.Errorf("**/*.go (newest first, .git skipped) = %v", got)
	}
	if got := strings.Fields(mustCall(t, w, "Glob", map[string]any{"pattern": "*.go"})); !slices.Equal(got, want[2:]) {
		t.Errorf("*.go = %v", got)
	}
	if got := strings.Fields(mustCall(t, w, "Glob", map[string]any{"pattern": "**/*.{ts,tsx}"})); len(got) != 2 {
		t.Errorf("braces = %v", got)
	}
	if got := strings.Fields(mustCall(t, w, "Glob", map[string]any{"pattern": "*.go", "path": "pkg"})); !slices.Equal(got, want[1:2]) {
		t.Errorf("path = %v", got)
	}
	if got := strings.Fields(mustCall(t, w, "Glob", map[string]any{"pattern": filepath.Join(dir, "pkg", "**", "b.go")})); !slices.Equal(got, want[:1]) {
		t.Errorf("absolute pattern = %v", got)
	}
	if out := mustCall(t, w, "Glob", map[string]any{"pattern": "*.rs"}); out != "No files found" {
		t.Errorf("none = %q", out)
	}
	for i := range globMaxResults + 5 {
		writeFile(t, filepath.Join(dir, "many", strings.Repeat("f", i+1)), "")
	}
	if out := mustCall(t, w, "Glob", map[string]any{"pattern": "many/*"}); strings.Count(out, "\n") != globMaxResults+1 || !strings.Contains(out, "truncated") {
		t.Errorf("result cap: %d lines", strings.Count(out, "\n"))
	}
	for _, in := range []map[string]any{{"pattern": "["}, {}, {"pattern": "*", "path": "missing"}} {
		if _, err := call(t, w, "Glob", in); err == nil {
			t.Errorf("glob %v: want error", in)
		}
	}
}

func TestGrep(t *testing.T) {
	w, dir := newWS(t)
	writeFile(t, filepath.Join(dir, "a.go"), "func Foo() {}\nfoo := 1\n")
	writeFile(t, filepath.Join(dir, "src/b.ts"), "const Foo = 1\n")
	writeFile(t, filepath.Join(dir, "bin.dat"), "Foo\x00")
	a, b := filepath.Join(dir, "a.go"), filepath.Join(dir, "src/b.ts")
	if out := mustCall(t, w, "Grep", map[string]any{"pattern": "Foo"}); !strings.HasPrefix(out, "Found 2 file(s)\n") || !strings.Contains(out, a) || !strings.Contains(out, b) {
		t.Errorf("files_with_matches = %q", out)
	}
	if out := mustCall(t, w, "Grep", map[string]any{"pattern": "foo", "-i": true, "output_mode": "content", "glob": "*.go"}); out != a+":1:func Foo() {}\n"+a+":2:foo := 1" {
		t.Errorf("content = %q", out)
	}
	if out := mustCall(t, w, "Grep", map[string]any{"pattern": "foo", "-i": true, "output_mode": "content", "-n": false, "path": "a.go", "head_limit": 1}); !strings.HasPrefix(out, a+":func Foo() {}\n...") {
		t.Errorf("no line numbers / head_limit = %q", out)
	}
	writeFile(t, filepath.Join(dir, "ctx.txt"), "1\nhit\n3\n4\n5\nhit\nhit\n8\n")
	c := filepath.Join(dir, "ctx.txt")
	if out := mustCall(t, w, "Grep", map[string]any{"pattern": "hit", "output_mode": "content", "path": "ctx.txt", "-C": 1}); out != c+"-1-1\n"+c+":2:hit\n"+c+"-3-3\n--\n"+c+"-5-5\n"+c+":6:hit\n"+c+":7:hit\n"+c+"-8-8" {
		t.Errorf("-C = %q", out)
	}
	if out := mustCall(t, w, "Grep", map[string]any{"pattern": "hit", "output_mode": "content", "path": "ctx.txt", "-A": 2, "-B": 0, "-n": false}); out != c+":hit\n"+c+"-3\n"+c+"-4\n--\n"+c+":hit\n"+c+":hit\n"+c+"-8" {
		t.Errorf("-A = %q", out)
	}
	if out := mustCall(t, w, "Grep", map[string]any{"pattern": "o", "output_mode": "count", "glob": "src/**/*.{ts,js}"}); out != b+":1" {
		t.Errorf("count = %q", out)
	}
	if out := mustCall(t, w, "Grep", map[string]any{"pattern": "nomatch"}); out != "No matches found" {
		t.Errorf("none = %q", out)
	}
	for _, in := range []map[string]any{{"pattern": "("}, {}, {"pattern": "x", "output_mode": "bogus"}, {"pattern": "x", "path": "missing"}} {
		if _, err := call(t, w, "Grep", in); err == nil {
			t.Errorf("grep %v: want error", in)
		}
	}
}

func TestTodoWrite(t *testing.T) {
	w, _ := newWS(t)
	todos := []map[string]any{{"content": "a", "status": "pending"}, {"content": "b", "status": "in_progress", "activeForm": "Doing b"}}
	if out := mustCall(t, w, "TodoWrite", map[string]any{"todos": todos}); !strings.Contains(out, "2 item") {
		t.Errorf("out = %q", out)
	}
	if len(w.todos) != 2 || w.todos[1].ActiveForm != "Doing b" {
		t.Errorf("todos = %+v", w.todos)
	}
	for _, in := range []map[string]any{{}, {"todos": []map[string]any{{"content": "", "status": "pending"}}}, {"todos": []map[string]any{{"content": "x", "status": "done"}}}} {
		if _, err := call(t, w, "TodoWrite", in); err == nil {
			t.Errorf("todo %v: want error", in)
		}
	}
	if len(w.todos) != 2 {
		t.Error("an invalid list replaced the stored one")
	}
}

func TestNotebookEdit(t *testing.T) {
	w, dir := newWS(t)
	p := filepath.Join(dir, "n.ipynb")
	writeFile(t, p, `{"cells":[{"cell_type":"code","id":"c1","source":"print(1)","outputs":[{"x":1}],"execution_count":3,"metadata":{}},
{"cell_type":"markdown","id":"c2","source":"# hi","metadata":{}}],"metadata":{"kernel":"py"},"nbformat":4,"nbformat_minor":5}`)
	cells := func() []map[string]any {
		var nb struct{ Cells []map[string]any }
		_ = json.Unmarshal([]byte(readFile(t, p)), &nb)
		return nb.Cells
	}
	mustCall(t, w, "NotebookEdit", map[string]any{"notebook_path": "n.ipynb", "cell_id": "c1", "new_source": "print(2)"})
	if c := cells()[0]; c["source"] != "print(2)" || len(c["outputs"].([]any)) != 0 || c["execution_count"] != nil {
		t.Errorf("replace = %v", c)
	}
	out := mustCall(t, w, "NotebookEdit", map[string]any{"notebook_path": "n.ipynb", "cell_id": "c1", "new_source": "new", "cell_type": "markdown", "edit_mode": "insert"})
	if c := cells(); len(c) != 3 || c[1]["source"] != "new" || c[1]["cell_type"] != "markdown" || c[1]["id"] == nil || !strings.Contains(out, "index 1") {
		t.Errorf("insert = %v (%s)", c, out)
	}
	mustCall(t, w, "NotebookEdit", map[string]any{"notebook_path": "n.ipynb", "new_source": "first", "cell_type": "code", "edit_mode": "insert"})
	if c := cells(); c[0]["source"] != "first" {
		t.Errorf("insert at start = %v", c[0])
	}
	mustCall(t, w, "NotebookEdit", map[string]any{"notebook_path": "n.ipynb", "cell_id": "cell-0", "edit_mode": "delete"})
	if c := cells(); len(c) != 3 || c[0]["id"] != "c1" {
		t.Errorf("delete by index = %v", c)
	}
	if !strings.Contains(readFile(t, p), `"kernel": "py"`) {
		t.Error("notebook metadata lost")
	}
	writeFile(t, filepath.Join(dir, "x.txt"), "{}")
	for _, in := range []map[string]any{
		{"notebook_path": "x.txt", "new_source": "a"},
		{"notebook_path": "n.ipynb", "cell_id": "nope", "new_source": "a"},
		{"notebook_path": "n.ipynb", "new_source": "a"},
		{"notebook_path": "n.ipynb", "cell_id": "c1", "new_source": "a", "edit_mode": "insert"},
		{"notebook_path": "n.ipynb", "cell_id": "c1", "new_source": "a", "edit_mode": "move"},
		{"notebook_path": "n.ipynb", "cell_id": "c1", "new_source": "a", "cell_type": "raw"},
		{"notebook_path": "n.ipynb", "cell_id": "c1"},
	} {
		if _, err := call(t, w, "NotebookEdit", in); err == nil {
			t.Errorf("notebook %v: want error", in)
		}
	}
}

// toolResult returns the first tool_result block of a request's last message.
func toolResult(t *testing.T, req map[string]any) (string, bool) {
	t.Helper()
	msgs := messages(req)
	for _, b := range msgs[len(msgs)-1]["content"].([]any) {
		blk := b.(map[string]any)
		if blk["type"] != "tool_result" {
			continue
		}
		var text strings.Builder
		for _, c := range blk["content"].([]any) {
			text.WriteString(c.(map[string]any)["text"].(string))
		}
		isErr, _ := blk["is_error"].(bool)
		return text.String(), isErr
	}
	t.Fatal("no tool_result in request")
	return "", false
}

func readerTools() map[string]any {
	return map[string]any{"reader": map[string]any{"name": "reader", "type": "function", "parameters": map[string]any{}}}
}

// Through the handler: offered as Read, executed in WorkDir, tracked under the
// LD key, traced as execute_tool; a tool error goes back to the model.
func TestHostToolThroughHandler(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "notes.txt"), "remember the milk\n")
	e := setup(t, map[string]any{"cfg": agentVariation("x", map[string]any{"tools": readerTools()})},
		toolResp("Read", `{"file_path":"notes.txt"}`), toolResp("Read", `{"file_path":"../etc/passwd"}`), textResp("milk", 1, 1))
	e.opts.WorkDir = dir
	res, err := ClaudeAgents(bg, "cfg", "what?", user, &InvokeOptions{Handler: e.opts,
		ToolHandlers: map[string]aiserver.ToolHandler{"reader": ClaudeRead}})
	if err != nil || res.Response != "milk" {
		t.Fatalf("res=%+v err=%v", res, err)
	}
	r := e.api.reqs()
	if got := toolNames(r[0]); !slices.Equal(got, []string{"Read"}) {
		t.Errorf("offered tools = %v", got)
	}
	tool := r[0]["tools"].([]any)[0].(map[string]any)
	if _, ok := tool["input_schema"].(map[string]any)["properties"].(map[string]any)["file_path"]; !ok {
		t.Errorf("Read schema = %v", tool["input_schema"])
	}
	if out, isErr := toolResult(t, r[1]); isErr || out != "     1\tremember the milk\n" {
		t.Errorf("tool result = %q isErr=%v", out, isErr)
	}
	if out, isErr := toolResult(t, r[2]); !isErr || !strings.Contains(out, "not allowed") {
		t.Errorf("escape result = %q isErr=%v", out, isErr)
	}
	tc := e.ld.events("$ld:ai:tool_call")
	if len(tc) != 2 || tc[0].Data["toolKey"] != "reader" {
		t.Errorf("tool_call = %+v", tc)
	}
	spans := e.spans("execute_tool Read")
	if len(spans) != 2 || spans[0].Status().Code != codes.Ok || spans[1].Status().Code != codes.Error {
		t.Errorf("tool spans = %d", len(spans))
	}
}

func TestHostToolConfigErrors(t *testing.T) {
	c := cfg(func(c *aiserver.AiConfig) {
		c.Tools = map[string]aiserver.Tool{"Read": {Name: "Read"}, "reader": {Name: "reader"}}
	})
	e := setup(t, nil)
	e.opts.WorkDir = t.TempDir()
	_, err := e.handler().Invoke(bg, aiserver.HandlerRequest{Config: c, ToolHandlers: map[string]aiserver.ToolHandler{
		"reader": ClaudeRead, "Read": aiserver.ToolFunc(func(context.Context, map[string]any) (any, error) { return "", nil }),
	}})
	if err == nil || !strings.Contains(err.Error(), "both a function tool and a built-in") {
		t.Errorf("name clash: %v", err)
	}
	e.opts.WorkDir = filepath.Join(t.TempDir(), "missing")
	_, err = e.handler().Invoke(bg, aiserver.HandlerRequest{Config: c, ToolHandlers: map[string]aiserver.ToolHandler{"reader": ClaudeRead}})
	if err == nil || !strings.Contains(err.Error(), "working directory") || len(e.api.reqs()) != 0 {
		t.Errorf("missing WorkDir: %v", err)
	}
}

// ToClaudeAgents: a node with ClaudeRead runs it, tracked for that node.
func TestToClaudeAgentsHostTool(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "a.txt"), "graph data\n")
	e := setup(t, graphFlags(map[string]any{"tools": readerTools()}),
		toolResp("Read", `{"file_path":"a.txt"}`), textResp("done", 1, 1))
	e.opts.WorkDir = dir
	res, err := ToClaudeAgents(resolve(t), &AdapterOptions{Handler: e.opts,
		ToolHandlers: map[string]aiserver.ToolHandler{"reader": ClaudeRead}}).Invoke(bg, "q")
	if err != nil || res.Response != "done" {
		t.Fatalf("%v %v", res, err)
	}
	if out, _ := toolResult(t, e.api.reqs()[1]); !strings.Contains(out, "graph data") {
		t.Errorf("tool result = %q", out)
	}
	tc := e.ld.events("$ld:ai:tool_call")
	if len(tc) != 1 || tc[0].Data["toolKey"] != "reader" || tc[0].Data["configKey"] != "root" {
		t.Errorf("tool_call = %+v", tc)
	}
}
