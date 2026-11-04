//go:build !windows

package main

import (
	"os"
	"testing"
)

// lockFolder makes a folder the app cannot write in.
func lockFolder(t *testing.T, dir string) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o500); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.Chmod(dir, 0o700) })
}
