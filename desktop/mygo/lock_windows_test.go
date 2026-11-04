package main

import (
	"os"
	"os/exec"
	"testing"
)

// lockFolder makes a folder the app cannot write in. Windows ignores a
// folder's read-only attribute when files are made in it, so an access
// rule denies everyone (S-1-1-0) writing there until the test ends.
func lockFolder(t *testing.T, dir string) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command("icacls", dir, "/deny", "*S-1-1-0:(W)").CombinedOutput(); err != nil {
		t.Fatalf("icacls: %v\n%s", err, out)
	}
	t.Cleanup(func() {
		if out, err := exec.Command("icacls", dir, "/remove:d", "*S-1-1-0").CombinedOutput(); err != nil {
			t.Errorf("icacls: %v\n%s", err, out)
		}
	})
}
