package handlerkit

import "fmt"

// CheckMaxRetries validates a messages handler's MaxRetries option at
// construction. A negative count is refused (the vendor SDKs panic on one), and
// so is a count given together with a caller-supplied client or model, which
// the count cannot reach: supplied names that option ("Client", "Model",
// "ModelFactory"), or is "" when there is none.
func CheckMaxRetries(pkg string, n *int, supplied string) error {
	if n == nil {
		return nil
	}
	if *n < 0 {
		return fmt.Errorf("%s: MaxRetries must be non-negative, got %d", pkg, *n)
	}
	if supplied != "" {
		return fmt.Errorf("%s: MaxRetries applies only to what the handler builds; configure retries on Options.%s instead", pkg, supplied)
	}
	return nil
}
