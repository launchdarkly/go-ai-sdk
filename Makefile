.PHONY: test lint vet format format-check build examples vulncheck

# Run all tests with the race detector
test:
	go test -race ./...

# Check formatting and vet
lint: format-check vet

vet:
	go vet ./...

# Format all files
format:
	gofmt -w .

# Fail if any file is not gofmt-formatted
format-check:
	@out="$$(gofmt -l .)"; if [ -n "$$out" ]; then echo "gofmt needed:"; echo "$$out"; exit 1; fi

# Report known vulnerabilities in code this module actually calls
vulncheck:
	go run golang.org/x/vuln/cmd/govulncheck@v1.1.4 ./...

# Build every package, including the examples
build:
	go build ./...

# Run an example scenario: make examples ARGS='agent <config-key> "What is a feature flag?"'
examples:
	go run ./examples $(ARGS)
