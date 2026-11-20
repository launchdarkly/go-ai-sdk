package opencodeagents

import (
	"errors"
	"fmt"
	"strings"

	"go.opentelemetry.io/otel/trace"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
	"github.com/launchdarkly/go-ai-sdk/internal/handlerkit"
	"github.com/launchdarkly/go-ai-sdk/internal/kitbase"
)

// errToolFailed marks a failed tool part's span. OpenCode's error text is not
// recorded: it can carry the provider's response body.
var errToolFailed = errors.New("opencodeagents: tool call failed")

type heldText struct {
	part, text string
	snapshot   bool
}

// reducer folds one session's bus events into output text, chat and
// execute_tool spans and usage.
//
// Text counts only for messages message.updated has marked assistant; text
// and tool parts that arrive before their message's role are held until it
// does. A part snapshot is that part's whole text so far, so only its new
// suffix is emitted. A delta without a part id is dropped. One chat span
// per assistant message and one execute_tool span per tool call id.
type reducer struct {
	run    kitbase.Run
	model  string                          // the sent model id
	names  map[string]string               // OpenCode tool name → config key
	native map[string]*aiserver.NativeTool // OpenCode tool name → sentinel to track
	onText func(string) bool

	roles     map[string]string
	heldText  map[string][]heldText
	heldTools map[string][]map[string]any
	partText  map[string]string
	chats     map[string]trace.Span
	closed    map[string]bool
	tools     map[string]trace.Span
	toolsDone map[string]bool

	out     strings.Builder
	usage   kitbase.SpanUsage
	sawUse  bool
	stopped bool // onText asked to stop
}

func newReducer(run kitbase.Run, model string, names map[string]string, native map[string]*aiserver.NativeTool, onText func(string) bool) *reducer {
	return &reducer{run: run, model: model, names: names, native: native, onText: onText,
		roles: map[string]string{}, heldText: map[string][]heldText{}, heldTools: map[string][]map[string]any{},
		partText: map[string]string{}, chats: map[string]trace.Span{}, closed: map[string]bool{},
		tools: map[string]trace.Span{}, toolsDone: map[string]bool{}}
}

// handle takes one event already filtered to this session (or none).
func (r *reducer) handle(ev map[string]any) error {
	p, _ := ev["properties"].(map[string]any)
	switch ev["type"] {
	case "session.error":
		name := ""
		if e, _ := p["error"].(map[string]any); e != nil {
			name, _ = e["name"].(string)
		}
		return &SessionError{Name: name}
	case "message.part.delta":
		if p == nil || p["field"] != "text" {
			return nil
		}
		d, _ := p["delta"].(string)
		msg, _ := p["messageID"].(string)
		part, _ := p["partID"].(string)
		if d != "" && msg != "" && part != "" {
			r.text(msg, heldText{part: part, text: d})
		}
	case "message.part.updated":
		part, _ := p["part"].(map[string]any)
		switch part["type"] {
		case "text":
			msg, _ := part["messageID"].(string)
			id, _ := part["id"].(string)
			text, _ := part["text"].(string)
			if msg != "" && id != "" && text != "" {
				r.text(msg, heldText{part: id, text: text, snapshot: true})
			}
		case "tool":
			r.tool(part)
		}
	case "message.updated":
		if info, _ := p["info"].(map[string]any); info != nil {
			r.message(info)
		}
	}
	return nil
}

func (r *reducer) text(msg string, t heldText) {
	switch r.roles[msg] {
	case "":
		r.heldText[msg] = append(r.heldText[msg], t)
	case "assistant":
		r.apply(t)
	}
}

func (r *reducer) apply(t heldText) {
	if r.stopped {
		return
	}
	add := t.text
	if t.snapshot {
		already := r.partText[t.part]
		if !strings.HasPrefix(t.text, already) {
			return
		}
		add = t.text[len(already):]
		r.partText[t.part] = t.text
	} else {
		r.partText[t.part] += t.text
	}
	if add == "" {
		return
	}
	r.out.WriteString(add)
	if r.onText != nil && !r.onText(add) {
		r.stopped = true
	}
}

func (r *reducer) tool(part map[string]any) {
	msg, _ := part["messageID"].(string)
	switch r.roles[msg] {
	case "":
		r.heldTools[msg] = append(r.heldTools[msg], part)
	case "assistant":
		r.applyTool(part)
	}
}

func (r *reducer) applyTool(part map[string]any) {
	state, _ := part["state"].(map[string]any)
	status, _ := state["status"].(string)
	id, _ := part["callID"].(string)
	if id == "" {
		id, _ = part["id"].(string)
	}
	name, _ := part["tool"].(string)
	if id == "" || state == nil || r.toolsDone[id] {
		return
	}
	input, hasInput := state["input"]
	switch status {
	case "running":
		if hasInput && name != "" {
			r.openTool(id, name, input)
		}
	case "completed", "error":
		if name != "" {
			r.openTool(id, name, input)
		}
		s := r.tools[id]
		if s == nil {
			return
		}
		delete(r.tools, id)
		r.toolsDone[id] = true
		if status == "error" {
			r.run.EndTool(s, errToolFailed)
			return
		}
		r.run.ToolResult(s, state["output"])
		r.run.EndTool(s, nil)
	}
}

func (r *reducer) openTool(id, name string, input any) {
	if r.tools[id] != nil {
		return
	}
	if n := r.native[name]; n != nil {
		handlerkit.TrackNativeTool(n)
	}
	key := name
	if k, ok := r.names[name]; ok {
		key = k
	}
	s := r.run.StartTool(key, id)
	r.run.ToolArguments(s, input)
	r.tools[id] = s
}

func (r *reducer) message(info map[string]any) {
	id, _ := info["id"].(string)
	role, _ := info["role"].(string)
	if id == "" || role == "" {
		return
	}
	r.roles[id] = role
	held, tools := r.heldText[id], r.heldTools[id]
	delete(r.heldText, id)
	delete(r.heldTools, id)
	if role != "assistant" {
		return
	}
	if r.chats[id] == nil && !r.closed[id] {
		r.chats[id] = r.run.StartChat(r.model)
	}
	for _, t := range held {
		r.apply(t)
	}
	for _, p := range tools {
		r.applyTool(p)
	}
	tm, _ := info["time"].(map[string]any)
	if _, done := tm["completed"]; !done || r.closed[id] {
		return
	}
	r.closed[id] = true
	u := kitbase.SpanUsage{Unreported: true}
	if tok, ok := info["tokens"].(map[string]any); ok {
		cache, _ := tok["cache"].(map[string]any)
		u = kitbase.SpanUsage{Input: num(tok["input"]), Output: num(tok["output"]), CacheRead: num(cache["read"]), CacheCreation: num(cache["write"])}
		r.usage.Add(u)
		r.sawUse = true
		u.Input += u.CacheRead + u.CacheCreation // spans report cache-inclusive input
	}
	r.run.EndChat(r.chats[id], r.model, u, "")
	delete(r.chats, id)
}

// close ends the turn: held parts of messages whose role never arrived are
// applied as assistant parts only for tools, as OpenCode runs those anyway,
// and spans still open are ended.
func (r *reducer) close() {
	for msg, parts := range r.heldTools {
		if r.roles[msg] == "" {
			for _, p := range parts {
				r.applyTool(p)
			}
		}
	}
	for id, s := range r.tools {
		r.run.EndTool(s, fmt.Errorf("%w: unfinished", errToolFailed))
		delete(r.tools, id)
	}
	for id, s := range r.chats {
		r.run.EndChat(s, r.model, kitbase.SpanUsage{Unreported: true}, "")
		delete(r.chats, id)
	}
}

// usageMap is the run's usage for handlerkit.ParseUsage, or nil when no
// assistant message reported tokens.
func (r *reducer) usageMap() map[string]any {
	if !r.sawUse {
		return nil
	}
	return map[string]any{"input_tokens": r.usage.Input, "output_tokens": r.usage.Output,
		"cache_read_input_tokens": r.usage.CacheRead, "cache_creation_input_tokens": r.usage.CacheCreation}
}

func num(v any) float64 {
	f, _ := v.(float64)
	return f
}
