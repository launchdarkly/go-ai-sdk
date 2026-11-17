//go:build !unix

package claudecli

import "testing"

// waitGone is a no-op where pids cannot be signalled.
func waitGone(*testing.T, int) {}
