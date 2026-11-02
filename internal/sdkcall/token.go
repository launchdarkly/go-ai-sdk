// Package sdkcall is the unforgeable mark handler packages pass so aiserver
// skips $ld:ai:sdk:usage. It lives under internal/, so a module outside this
// repo cannot import it or build a non-zero Token.
package sdkcall

// Token suppresses usage reporting when Marked.
type Token struct{ marked bool }

// Mark returns the token handler packages set on ConfigArgs and GraphOptions.
func Mark() Token { return Token{marked: true} }

// Marked reports whether Mark produced this token.
func (t Token) Marked() bool { return t.marked }
