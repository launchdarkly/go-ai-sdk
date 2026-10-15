package langchainwire

import "testing"

// langchaingo's OpenAI client streams tool-call deltas through the text
// callback as a JSON array of calls; only that shape is filtered out.
func TestIsToolCallChunk(t *testing.T) {
	for chunk, want := range map[string]bool{
		`[{"id":"c1","type":"function","function":{"name":"a","arguments":"{}"}}]`: true,
		`[{"function":{"name":"a"}},{"function":{"name":"b"}}]`:                    true,
		`[{"function":{"name":"a"}},{"name":"b"}]`:                                 false,
		`[]`:              false,
		`[1, 2]`:          false,
		`[not json`:       false,
		`{"function":{}}`: false,
		`Hello`:           false,
		``:                false,
	} {
		if got := isToolCallChunk([]byte(chunk)); got != want {
			t.Errorf("isToolCallChunk(%q) = %v, want %v", chunk, got, want)
		}
	}
}
