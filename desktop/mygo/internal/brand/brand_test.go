package brand

import (
	"bytes"
	"os"
	"testing"
)

func TestIconMatchesDesktopIcon(t *testing.T) {
	want, err := os.ReadFile("../../../icon.png")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(want, Icon) {
		t.Error("internal/brand/icon.png differs from desktop/icon.png: run desktop/mygo/build.py, which copies it")
	}
}
