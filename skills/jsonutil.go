package skills

import (
	"bytes"
	"encoding/json"
	"fmt"
	"unicode/utf16"
	"unicode/utf8"
)

// marshalSortedASCII serializes v the way Python's json.dumps(sort_keys=True)
// does, so two SDKs writing the same file or log line produce the same bytes:
// map keys sorted (encoding/json already sorts them), no HTML escaping, and
// every non-ASCII character escaped as \uXXXX (astral characters as a
// surrogate pair). indent selects the two-space manifest form; otherwise the
// output is compact. There is no trailing newline.
func marshalSortedASCII(v any, indent bool) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if indent {
		enc.SetIndent("", "  ")
	}
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	raw := bytes.TrimSuffix(buf.Bytes(), []byte("\n"))
	out := make([]byte, 0, len(raw))
	for len(raw) > 0 {
		r, size := utf8.DecodeRune(raw)
		raw = raw[size:]
		if r < utf8.RuneSelf {
			out = append(out, byte(r))
			continue
		}
		if r > 0xFFFF {
			hi, lo := utf16.EncodeRune(r)
			out = fmt.Appendf(out, `\u%04x\u%04x`, hi, lo)
			continue
		}
		out = fmt.Appendf(out, `\u%04x`, r)
	}
	return out, nil
}

// decodeJSONStringVerbatim decodes one JSON string literal without the
// laundering encoding/json applies: a lone surrogate escape such as "\ud800"
// becomes its 3-byte WTF-8 form and raw invalid UTF-8 bytes are kept, instead
// of both turning into U+FFFD. Verification then sees content that has no
// UTF-8 encoding and withholds it with not_utf8 rather than hashing a
// substitute. ok is false when raw is not a JSON string literal.
func decodeJSONStringVerbatim(raw []byte) (string, bool) {
	raw = bytes.TrimSpace(raw)
	if len(raw) < 2 || raw[0] != '"' || raw[len(raw)-1] != '"' {
		return "", false
	}
	body := raw[1 : len(raw)-1]
	out := make([]byte, 0, len(body))
	for i := 0; i < len(body); i++ {
		c := body[i]
		if c == '"' || c < 0x20 {
			return "", false
		}
		if c != '\\' {
			out = append(out, c)
			continue
		}
		i++
		if i >= len(body) {
			return "", false
		}
		switch body[i] {
		case '"', '\\', '/':
			out = append(out, body[i])
		case 'b':
			out = append(out, '\b')
		case 'f':
			out = append(out, '\f')
		case 'n':
			out = append(out, '\n')
		case 'r':
			out = append(out, '\r')
		case 't':
			out = append(out, '\t')
		case 'u':
			cp, ok := hex4(body, i+1)
			if !ok {
				return "", false
			}
			i += 4
			if cp >= 0xD800 && cp < 0xDC00 && i+2 < len(body) && body[i+1] == '\\' && body[i+2] == 'u' {
				if lo, ok := hex4(body, i+3); ok && lo >= 0xDC00 && lo <= 0xDFFF {
					out = utf8.AppendRune(out, utf16.DecodeRune(rune(cp), rune(lo)))
					i += 6
					continue
				}
			}
			if utf16.IsSurrogate(rune(cp)) {
				out = append(out, 0xE0|byte(cp>>12), 0x80|byte(cp>>6)&0x3F, 0x80|byte(cp)&0x3F)
				continue
			}
			out = utf8.AppendRune(out, rune(cp))
		default:
			return "", false
		}
	}
	return string(out), true
}

func hex4(b []byte, at int) (uint16, bool) {
	if at+4 > len(b) {
		return 0, false
	}
	var v uint16
	for _, c := range b[at : at+4] {
		v <<= 4
		switch {
		case c >= '0' && c <= '9':
			v |= uint16(c - '0')
		case c >= 'a' && c <= 'f':
			v |= uint16(c-'a') + 10
		case c >= 'A' && c <= 'F':
			v |= uint16(c-'A') + 10
		default:
			return 0, false
		}
	}
	return v, true
}
