//go:build windows

package main

import (
	"context"
	"os"
)

// loginEnv is the process environment: Explorer starts apps with the user's
// PATH and variables from the registry, as a new terminal gets them.
func loginEnv(context.Context) ([]string, error) { return os.Environ(), nil }
