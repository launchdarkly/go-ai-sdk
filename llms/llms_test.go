package llms

// textParts is a message of role with one text part per string.
func textParts(role ChatMessageType, parts ...string) MessageContent {
	m := MessageContent{Role: role}
	for _, p := range parts {
		m.Parts = append(m.Parts, TextContent{Text: p})
	}
	return m
}
