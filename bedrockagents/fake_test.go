package bedrockagents

import (
	"bytes"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/aws/protocol/eventstream"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/launchdarkly/go-sdk-common/v3/ldcontext"
	"github.com/launchdarkly/go-sdk-common/v3/ldvalue"

	"github.com/launchdarkly/go-ai-sdk/aiserver"
)

// fakeLD serves byKey[key], else variation, and records track calls.
type fakeLD struct {
	variation map[string]any
	byKey     map[string]map[string]any
	mu        sync.Mutex
	events    []trackedEvent
	keys      []string // every flag key evaluated
}

type trackedEvent struct {
	key  string
	data ldvalue.Value
}

func (f *fakeLD) JSONVariation(key string, _ ldcontext.Context, _ ldvalue.Value) (ldvalue.Value, error) {
	f.mu.Lock()
	f.keys = append(f.keys, key)
	f.mu.Unlock()
	v, ok := f.byKey[key]
	if !ok {
		v = f.variation
	}
	b, _ := json.Marshal(v)
	return ldvalue.Parse(b), nil
}

func (f *fakeLD) TrackMetric(key string, _ ldcontext.Context, _ float64, data ldvalue.Value) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.events = append(f.events, trackedEvent{key, data})
	return nil
}
func (f *fakeLD) Flush()       {}
func (f *fakeLD) Close() error { return nil }

func (f *fakeLD) usageHelpers() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	var out []string
	for _, e := range f.events {
		if e.key == "$ld:ai:sdk:usage" {
			out = append(out, e.data.GetByKey("helper").StringValue())
		}
	}
	return out
}

func (f *fakeLD) requestedKeys() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.keys...)
}

// checkUsagePayloads asserts every usage event so far names this package's
// constants as the helper package and the core client as the AI SDK.
func (f *fakeLD) checkUsagePayloads(t *testing.T) {
	t.Helper()
	f.mu.Lock()
	defer f.mu.Unlock()
	n := 0
	for _, e := range f.events {
		if e.key != "$ld:ai:sdk:usage" {
			continue
		}
		n++
		d := e.data
		if d.GetByKey("helperPackageName").StringValue() != PackageName || d.GetByKey("helperPackageVersion").StringValue() != PackageVersion ||
			d.GetByKey("aiSdkName").StringValue() != aiserver.PackageName || d.GetByKey("aiSdkVersion").StringValue() != aiserver.PackageVersion {
			t.Errorf("usage payload = %v", d)
		}
	}
	if n == 0 {
		t.Error("no usage events to check")
	}
}

// reply is one canned Bedrock response.
type reply struct {
	status int
	body   string // JSON body; for streams, the event-stream frames
	stream bool
}

// recorded is one request the fake received.
type recorded struct {
	modelID, op, auth, raw string
	body                   map[string]any
}

// fakeBedrock is an HTTPS Bedrock Runtime: it replays replies in order and
// records each request (HTTPS because bearer auth refuses plain HTTP).
type fakeBedrock struct {
	mu       sync.Mutex
	replies  []reply
	requests []recorded
	srv      *httptest.Server
}

func newFakeBedrock(t *testing.T, replies ...reply) *fakeBedrock {
	t.Helper()
	f := &fakeBedrock{replies: replies}
	f.srv = httptest.NewTLSServer(f)
	t.Cleanup(f.srv.Close)
	return f
}

func (f *fakeBedrock) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	b, _ := io.ReadAll(r.Body)
	var body map[string]any
	_ = json.Unmarshal(b, &body)
	// Path is /model/{modelId}/{converse|converse-stream}.
	parts := strings.Split(strings.TrimPrefix(r.URL.EscapedPath(), "/model/"), "/")
	id, _ := url.PathUnescape(parts[0])
	op := ""
	if len(parts) > 1 {
		op = parts[1]
	}
	f.mu.Lock()
	f.requests = append(f.requests, recorded{modelID: id, op: op, auth: r.Header.Get("Authorization"), raw: string(b), body: body})
	i := len(f.requests) - 1
	f.mu.Unlock()
	if i >= len(f.replies) {
		w.Header().Set("X-Amzn-ErrorType", "InternalServerException")
		http.Error(w, `{"message":"no more replies"}`, 500)
		return
	}
	rep := f.replies[i]
	switch {
	case rep.status >= 400:
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Amzn-ErrorType", "ValidationException")
		w.WriteHeader(rep.status)
	case rep.stream:
		w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
	default:
		w.Header().Set("Content-Type", "application/json")
	}
	_, _ = io.WriteString(w, rep.body)
}

func (f *fakeBedrock) reqs() []recorded {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]recorded(nil), f.requests...)
}

// client returns an injected client pointed at the fake with static SigV4
// credentials and no retries.
func (f *fakeBedrock) client() *bedrockruntime.Client {
	return bedrockruntime.New(bedrockruntime.Options{
		Region: "us-west-2", BaseEndpoint: aws.String(f.srv.URL), HTTPClient: f.srv.Client(),
		Credentials:      credentials.NewStaticCredentialsProvider("AKIDINJECTED", "SECRET", ""),
		RetryMaxAttempts: 1,
	})
}

// isolateAWSEnv makes factory-built clients reach the fake: endpoint and CA
// bundle via the standard AWS env vars, static credentials, no shared files,
// no IMDS, and no ambient bearer token.
func isolateAWSEnv(t *testing.T, f *fakeBedrock) {
	t.Helper()
	dir := t.TempDir()
	ca := filepath.Join(dir, "ca.pem")
	pemBytes := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: f.srv.Certificate().Raw})
	if err := os.WriteFile(ca, pemBytes, 0o600); err != nil {
		t.Fatal(err)
	}
	for k, v := range map[string]string{
		"AWS_CONFIG_FILE": filepath.Join(dir, "none-config"), "AWS_SHARED_CREDENTIALS_FILE": filepath.Join(dir, "none-creds"),
		"AWS_PROFILE": "", "AWS_ACCESS_KEY_ID": "AKIDENV", "AWS_SECRET_ACCESS_KEY": "SECRET", "AWS_SESSION_TOKEN": "",
		"AWS_EC2_METADATA_DISABLED": "true", "AWS_BEARER_TOKEN_BEDROCK": "", "AWS_REGION": "us-east-1",
		"AWS_DEFAULT_REGION": "", "AWS_ENDPOINT_URL": f.srv.URL, "AWS_ENDPOINT_URL_BEDROCK_RUNTIME": f.srv.URL,
		"AWS_CA_BUNDLE": ca, "AWS_MAX_ATTEMPTS": "1",
	} {
		t.Setenv(k, v)
	}
}

func textReply(text string, in, out int) reply {
	return reply{body: fmt.Sprintf(`{"output":{"message":{"role":"assistant","content":[{"text":%q}]}},"stopReason":"end_turn",
"usage":{"inputTokens":%d,"outputTokens":%d,"totalTokens":%d},"metrics":{"latencyMs":1}}`, text, in, out, in+out)}
}

func toolReply(name, input string) reply {
	return reply{body: fmt.Sprintf(`{"output":{"message":{"role":"assistant","content":[{"toolUse":{"toolUseId":"tu-%s","name":%q,"input":%s}}]}},
"stopReason":"tool_use","usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2},"metrics":{"latencyMs":1}}`, name, name, input)}
}

func errorReply() reply { return reply{status: 400, body: `{"message":"bad request"}`} }

// frames encodes ConverseStream events as AWS event-stream messages.
func frames(t *testing.T, events ...[2]string) reply {
	t.Helper()
	var buf bytes.Buffer
	enc := eventstream.NewEncoder()
	for _, e := range events {
		err := enc.Encode(&buf, eventstream.Message{Headers: eventstream.Headers{
			{Name: ":message-type", Value: eventstream.StringValue("event")},
			{Name: ":event-type", Value: eventstream.StringValue(e[0])},
			{Name: ":content-type", Value: eventstream.StringValue("application/json")},
		}, Payload: []byte(e[1])})
		if err != nil {
			t.Fatal(err)
		}
	}
	return reply{body: buf.String(), stream: true}
}

func textDelta(i int, s string) [2]string {
	return [2]string{"contentBlockDelta", fmt.Sprintf(`{"contentBlockIndex":%d,"delta":{"text":%q}}`, i, s)}
}

func blockStop(i int) [2]string {
	return [2]string{"contentBlockStop", fmt.Sprintf(`{"contentBlockIndex":%d}`, i)}
}

func messageStop(reason string) [2]string {
	return [2]string{"messageStop", fmt.Sprintf(`{"stopReason":%q}`, reason)}
}

func metadata(in, out, cacheRead, cacheWrite int) [2]string {
	return [2]string{"metadata", fmt.Sprintf(`{"usage":{"inputTokens":%d,"outputTokens":%d,"totalTokens":%d,"cacheReadInputTokens":%d,"cacheWriteInputTokens":%d},"metrics":{"latencyMs":1}}`,
		in, out, in+out+cacheRead+cacheWrite, cacheRead, cacheWrite)}
}

var messageStart = [2]string{"messageStart", `{"role":"assistant"}`}

func streamText(t *testing.T, deltas []string, in, out int) reply {
	evs := [][2]string{messageStart}
	for _, d := range deltas {
		evs = append(evs, textDelta(0, d))
	}
	evs = append(evs, blockStop(0), messageStop("end_turn"), metadata(in, out, 0, 0))
	return frames(t, evs...)
}
