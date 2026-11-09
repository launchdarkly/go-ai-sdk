package evaluations

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"testing"
	"time"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

var hex64 = regexp.MustCompile(`^[0-9a-f]{64}$`)

func sha256Hex(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:])
}

func TestCanonicalJSON(t *testing.T) {
	negZero := math.Copysign(0, -1)
	cases := []struct {
		name  string
		value any
		bytes string
	}{
		{"ASCII identity",
			map[string]any{"projectKey": "proj", "evaluationId": "eval-1", "evaluationRunId": "run-1", "runId": "run-1", "rowIndex": 1},
			`{"evaluationId":"eval-1","evaluationRunId":"run-1","projectKey":"proj","rowIndex":1,"runId":"run-1"}`},
		{"non-ASCII is written as is; only the quote, backslash and controls are escaped",
			map[string]any{"projectKey": "p<>&é東\u2028\u2029\U0001F600\n\x1f\"\\", "evaluationId": "e", "evaluationRunId": "r", "runId": "r", "rowIndex": 2, "criterionType": "Accuracy"},
			"{\"criterionType\":\"Accuracy\",\"evaluationId\":\"e\",\"evaluationRunId\":\"r\",\"projectKey\":\"p<>&é東\u2028\u2029\U0001F600" + `\n\u001f\"\\` + "\",\"rowIndex\":2,\"runId\":\"r\"}"},
		// U+1F600 is the surrogate pair D83D DE00, so it sorts before U+FF61.
		{"keys sort by UTF-16 code unit", map[string]any{"z": 1, "\uff61": 2, "\U0001F600": 3}, "{\"z\":1,\"\U0001F600\":3,\"\uff61\":2}"},
		{"-0 is 0", []any{negZero, float32(negZero)}, `[0,0]`},
		{"an integer above 2^53 is the nearest double",
			[]any{int64(1<<53 + 1), uint64(1<<53 + 1), json.Number("9007199254740993")},
			`[9007199254740992,9007199254740992,9007199254740992]`},
		// RFC 8785 rejects NaN and Infinity; the SDK contract writes null, as
		// JSON.stringify does. Don't change this to an error.
		{"non-finite numbers are null",
			map[string]any{"nan": math.NaN(), "inf": math.Inf(1), "ninf": float32(math.Inf(-1)), "huge": json.Number("1e400")},
			`{"huge":null,"inf":null,"nan":null,"ninf":null}`},
		{"controls", "\b\f\t\r\x01\x7f", `"\b\f\t\r\u0001` + "\x7f\""},
		{"nil, typed nil and nested values",
			map[string]any{"a": nil, "b": map[string]any(nil), "c": []any{true, false, nil, "x", 1.5}},
			`{"a":null,"b":null,"c":[true,false,null,"x",1.5]}`},
		{"other types are hashed as the JSON they marshal to", struct {
			B []int          `json:"b"`
			A map[string]int `json:"a"`
		}{[]int{1, 2}, map[string]int{"y": 2, "x": 1}}, `{"a":{"x":1,"y":2},"b":[1,2]}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b, err := canonicalJSON(tc.value)
			if err != nil || string(b) != tc.bytes {
				t.Fatalf("canonical\n got %s (%v)\nwant %s", b, err, tc.bytes)
			}
		})
	}
	// eventId digests. The second was computed with the Python SDK's canonical_json.
	for i, digest := range []string{
		"c8c1bc7d6ea3f69748037db8acbb329b7f833f89ae63a2c255764995f35a8034",
		"431488bfe6b426666e571458fab1fcd418b12880a9d51323cfc1123e3c1290b6",
	} {
		if id, _ := eventID(cases[i].value.(map[string]any)); id != digest || sha256Hex(cases[i].bytes) != digest {
			t.Errorf("%s: eventId %s, want %s", cases[i].name, id, digest)
		}
	}
}

// A Go string cannot hold a lone surrogate: decoding one gives U+FFFD, which
// is written as is, and so is invalid UTF-8.
func TestCanonicalJSONLoneSurrogate(t *testing.T) {
	var s string
	if err := json.Unmarshal([]byte(`"a\ud800b"`), &s); err != nil {
		t.Fatal(err)
	}
	for _, in := range []string{s, "a\xffb"} {
		if b, _ := canonicalJSON(in); string(b) != "\"a�b\"" {
			t.Fatalf("%q: got %q", in, b)
		}
	}
}

// RFC 8785 Appendix B, "Number Serialization Samples". Its NaN and Infinity
// rows have no JSON form and are covered above.
func TestCanonicalJSONNumbers(t *testing.T) {
	for _, tc := range []struct {
		bits uint64
		want string
	}{
		{0x0000000000000000, "0"}, {0x8000000000000000, "0"},
		{0x0000000000000001, "5e-324"}, {0x8000000000000001, "-5e-324"},
		{0x7fefffffffffffff, "1.7976931348623157e+308"}, {0xffefffffffffffff, "-1.7976931348623157e+308"},
		{0x4340000000000000, "9007199254740992"}, {0xc340000000000000, "-9007199254740992"},
		{0x4430000000000000, "295147905179352830000"},
		{0x44b52d02c7e14af5, "9.999999999999997e+22"}, {0x44b52d02c7e14af6, "1e+23"}, {0x44b52d02c7e14af7, "1.0000000000000001e+23"},
		{0x444b1ae4d6e2ef4e, "999999999999999700000"}, {0x444b1ae4d6e2ef4f, "999999999999999900000"}, {0x444b1ae4d6e2ef50, "1e+21"},
		{0x3eb0c6f7a0b5ed8c, "9.999999999999997e-7"}, {0x3eb0c6f7a0b5ed8d, "0.000001"},
		{0x41b3de4355555553, "333333333.3333332"}, {0x41b3de4355555554, "333333333.33333325"},
		{0x41b3de4355555555, "333333333.3333333"}, {0x41b3de4355555556, "333333333.3333334"},
		{0x41b3de4355555557, "333333333.33333343"},
		{0xbecbf647612f3696, "-0.0000033333333333333333"},
		{0x43143ff3c1cb0959, "1424953923781206.2"},
	} {
		if b, _ := canonicalJSON(math.Float64frombits(tc.bits)); string(b) != tc.want {
			t.Errorf("%016x: got %s, want %s", tc.bits, b, tc.want)
		}
	}
}

// The RFC 8785 reference implementation's vectors (testdata/jcs/README.md).
func TestCanonicalJSONReferenceVectors(t *testing.T) {
	inputs, err := filepath.Glob("testdata/jcs/input/*.json")
	if err != nil || len(inputs) != 6 {
		t.Fatalf("vectors %v %v", inputs, err)
	}
	for _, in := range inputs {
		t.Run(filepath.Base(in), func(t *testing.T) {
			raw, err := os.ReadFile(in)
			if err != nil {
				t.Fatal(err)
			}
			want, err := os.ReadFile(filepath.Join("testdata/jcs/expected", filepath.Base(in)))
			if err != nil {
				t.Fatal(err)
			}
			dec := json.NewDecoder(bytes.NewReader(raw))
			dec.UseNumber()
			var v any
			if err := dec.Decode(&v); err != nil {
				t.Fatal(err)
			}
			if got, err := canonicalJSON(v); err != nil || !bytes.Equal(got, bytes.TrimRight(want, "\n")) {
				t.Fatalf("got  %s (%v)\nwant %s", got, err, want)
			}
		})
	}
}

func TestContentHash(t *testing.T) {
	gen := func(status string, output any, usage map[string]any, errMsg string) *genResult {
		return &genResult{status: status, output: output, usage: usage, errMsg: errMsg, started: time.Unix(0, 0), elapsed: time.Second}
	}
	hashOf := func(g *genResult) string {
		t.Helper()
		f, err := generationFields(g)
		if err != nil {
			t.Fatal(err)
		}
		return f["contentHash"].(string)
	}
	// Shared with the JavaScript and Python SDKs' tests: the same content
	// hashes the same in every language.
	for _, tc := range []struct {
		name string
		g    *genResult
		want string
	}{
		{"non-ASCII output with usage", gen(statusComplete, "café 東京 \U0001F600", map[string]any{"input_tokens": 1, "output_tokens": 2}, ""),
			"c413938130f9e5d632edbd07d529139b942e1c43c62fe3fb9ae09552dc7e7f71"},
		{"errored row", gen(statusError, nil, nil, "handler raised: boom"),
			"04e26552bf66f7afa68ba6307c9f9c162abd2bbd963cb7e42d9dcea337152848"},
		{"structured output, no usage", gen(statusComplete, map[string]any{"b": 1, "a": []any{true, nil, "x"}}, nil, ""),
			"504de7cbb5f7d8b3ef260a99c69460c6b7f4fb3ea449c16cedde0ac0e120008c"},
		{"numbers", gen(statusComplete, map[string]any{"score": 1.0, "p": 0.1, "tiny": 1e-5, "big": 1e16,
			"list": []any{3, math.Copysign(0, -1), 2.5e-7}, "neg": -2}, map[string]any{"input_tokens": 7, "output_tokens": 9}, ""),
			"ae920410735c7fe62767503ea2121965d264b0d26abe396736d03be6aac77ae7"},
		{"UTF-16 key order", gen(statusComplete, map[string]any{"z": 1, "\uff61": 2, "\U0001F600": 3}, nil, ""),
			sha256Hex("{\"error\":null,\"output\":{\"z\":1,\"\U0001F600\":3,\"\uff61\":2},\"status\":\"COMPLETE\"}")},
	} {
		if got := hashOf(tc.g); got != tc.want {
			t.Errorf("%s: %s, want %s", tc.name, got, tc.want)
		}
	}
	same := gen(statusComplete, "same", nil, "")
	moved := gen(statusComplete, "same", nil, "")
	moved.row.index, moved.started, moved.elapsed = 4, time.Unix(99, 0), 5*time.Millisecond
	if hashOf(moved) != hashOf(same) || hashOf(gen(statusComplete, "different", nil, "")) == hashOf(same) {
		t.Fatal("contentHash must cover what was generated, not when or for which row")
	}
}

// A non-finite output is sent as null, as JavaScript sends it, and hashed as
// that null.
func TestNonFiniteOutputIsSentAsNull(t *testing.T) {
	for _, tc := range []struct {
		output any
		sent   any
		digest string // computed with the Python SDK's canonical_json
	}{
		{math.NaN(), nil, "15632e7076cae5309f58fd3087ff7a89855b824e6be3908b9bed310ed5bde6b4"},
		{map[string]any{"s": math.Inf(1)}, map[string]any{"s": nil}, "3d0bca1cf3f69ee1db80a9fdf936372861026cae03ddc0746f152389b2a5c572"},
	} {
		e := newEnv(t, 1)
		mustRun(t, e.module(), baseArgs(handler("OpenAI", aiserver.ModeMessages, func(context.Context, aiserver.HandlerRequest) (aiserver.HandlerResult, error) {
			return aiserver.HandlerResult{Output: tc.output}, nil
		})))
		evs := e.ld.events(generationEventName)
		if len(evs) != 1 {
			t.Fatalf("output %v: %d generation events, logs %s", tc.output, len(evs), e.logs)
		}
		d := evs[0].Data
		if out, has := d["output"]; !has || !sameJSON(out, tc.sent) {
			t.Errorf("output %v sent as %v (present %v)", tc.output, out, has)
		}
		if d["contentHash"] != tc.digest {
			t.Errorf("output %v: contentHash %v", tc.output, d["contentHash"])
		}
	}
}
