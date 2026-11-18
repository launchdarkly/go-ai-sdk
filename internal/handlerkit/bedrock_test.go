package handlerkit

import "testing"

func TestBedrockModelID(t *testing.T) {
	const id = "anthropic.claude-sonnet-4-5"
	const arn = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc"
	cases := []struct{ id, region, want string }{
		{id, "us", "us." + id},
		{id, "global", "global." + id},
		{"us." + id, "us", "us." + id},
		{"eu." + id, "us", "eu." + id},
		{"us." + id, "eu", "us." + id},
		{arn, "us", arn},
		// The region is trimmed and loses trailing dots; empty means no prefix.
		{id, "", id},
		{id, "   ", id},
		{id, " . ", id},
		{id, " us ", "us." + id},
		{id, "us.", "us." + id},
	}
	// An id that already has any known profile prefix is not prefixed again.
	for _, p := range []string{"global.", "us-gov.", "eu.", "apac.", "au.", "jp.", "ca.", "in."} {
		cases = append(cases, struct{ id, region, want string }{p + id, "us", p + id})
	}
	for _, tc := range cases {
		if got := BedrockModelID(tc.id, tc.region); got != tc.want {
			t.Errorf("BedrockModelID(%q, %q) = %q, want %q", tc.id, tc.region, got, tc.want)
		}
	}
}
