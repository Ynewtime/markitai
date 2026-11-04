//go:build !windows

package main

import (
	"os"
	"testing"
)

// lockFolder makes a folder the app cannot write in, whether or not it
// was there. Root writes in any folder, so the test cannot hold there.
func lockFolder(t *testing.T, dir string) {
	t.Helper()
	if os.Geteuid() == 0 {
		t.Skip("root writes in a read-only folder")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(dir, 0o500); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.Chmod(dir, 0o700) })
}
