package handlerkit

import "strings"

// bedrockProfilePrefixes are the cross-region inference-profile prefixes
// Bedrock defines (see the Bedrock model cards).
var bedrockProfilePrefixes = []string{"global.", "us.", "us-gov.", "eu.", "apac.", "au.", "jp.", "ca.", "in."}

// BedrockModelID returns a Bedrock model id with region prepended once as its
// cross-region inference-profile prefix: ("anthropic.claude-x", "us") is
// "us.anthropic.claude-x". region is trimmed and loses trailing dots. The id is
// returned unchanged when region is then empty, when the id already starts with
// a known profile prefix or with region, and for an ARN. region is the short
// profile prefix ("us", "global"), never an AWS Region. Callers decide whether
// the model is a Bedrock one.
func BedrockModelID(id, region string) string {
	prefix := strings.TrimRight(strings.TrimSpace(region), ".")
	if prefix == "" || id == "" || strings.HasPrefix(id, prefix+".") || strings.HasPrefix(id, "arn:") {
		return id
	}
	for _, p := range bedrockProfilePrefixes {
		if strings.HasPrefix(id, p) {
			return id
		}
	}
	return prefix + "." + id
}
