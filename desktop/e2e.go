//go:build e2e

package main

import (
	"os"
	"path/filepath"
	"strconv"

	"github.com/egoist/mygo"
	_ "github.com/egoist/mygo/plugins/automation"
)

// An e2e build, driven by scripts/desktop_e2e.py over MYGO_AUTOMATION. With
// MARKITAI_DESKTOP_E2E=DIR it keeps what it writes under DIR: its own name (a
// single-instance lock apart from an installed Markitai), user data, cache,
// logs and downloads. Save panels answer DIR/saved/<name>, or cancel when
// MARKITAI_DESKTOP_E2E_SAVE=cancel; the busy-quit question answers
// MARKITAI_DESKTOP_E2E_QUIT (when-done, the default; now; cancel) and
// writes the count it was asked about to DIR/asked.
func init() {
	dir := os.Getenv("MARKITAI_DESKTOP_E2E")
	if dir == "" {
		return
	}
	mygo.App.SetName("Markitai E2E")
	for name, sub := range map[mygo.PathName]string{mygo.PathUserData: "data", mygo.PathCache: "cache",
		mygo.PathLogs: "logs", mygo.PathDownloads: "downloads"} {
		mygo.App.SetPath(name, filepath.Join(dir, sub))
	}
	saveAs = func(_ *mygo.Window, suggested string) string {
		if os.Getenv("MARKITAI_DESKTOP_E2E_SAVE") == "cancel" {
			return ""
		}
		saved := filepath.Join(dir, "saved")
		if os.MkdirAll(saved, 0o700) != nil {
			return ""
		}
		return filepath.Join(saved, filepath.Base(suggested))
	}
	askQuit = func(_ *mygo.Window, _ Texts, n int) quitChoice {
		_ = os.WriteFile(filepath.Join(dir, "asked"), []byte(strconv.Itoa(n)), 0o600)
		switch os.Getenv("MARKITAI_DESKTOP_E2E_QUIT") {
		case "now":
			return quitStopNow
		case "cancel":
			return quitCancel
		}
		return quitWhenDone
	}
}
