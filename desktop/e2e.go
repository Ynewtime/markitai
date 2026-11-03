//go:build e2e

package main

import (
	"bytes"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"

	"github.com/egoist/mygo"
	_ "github.com/egoist/mygo/plugins/automation"
)

// An e2e build, driven by scripts/desktop_e2e.py over MYGO_AUTOMATION. With
// MARKITAI_DESKTOP_E2E=DIR it keeps what it writes under DIR: its own name (a
// single-instance lock apart from an installed Markitai), user data, cache,
// logs and downloads. The web view's data (localStorage, cookies) lives
// outside DIR, under the app's identifier, so each launch clears it before
// the first page loads; on macOS it refuses to start unless its identifier
// is its own (ending in .e2e), never clearing an installed Markitai's. Save
// panels answer DIR/saved/<name>, or cancel when
// MARKITAI_DESKTOP_E2E_SAVE=cancel; the busy-quit question answers
// MARKITAI_DESKTOP_E2E_QUIT (when-done, the default; now; cancel) and
// writes the count it was asked about, or "unknown", to DIR/asked. What
// would open in the browser or mail app is appended to DIR/external.
func init() {
	dir := os.Getenv("MARKITAI_DESKTOP_E2E")
	if dir == "" {
		return
	}
	if id := bundleIdentifier(); runtime.GOOS == "darwin" && !strings.HasSuffix(id, ".e2e") {
		log.Fatalf("markitai desktop e2e: the identifier %q is not an e2e build's, whose web data it would clear", id)
	}
	mygo.App.SetName("Markitai E2E")
	for name, sub := range map[mygo.PathName]string{mygo.PathUserData: "data", mygo.PathCache: "cache",
		mygo.PathLogs: "logs", mygo.PathDownloads: "downloads"} {
		mygo.App.SetPath(name, filepath.Join(dir, sub))
	}
	// Registered before main's start, so it runs first.
	mygo.App.WhenReady(func() {
		if err := mygo.App.ClearBrowsingData(); err != nil {
			log.Fatalf("markitai desktop e2e: clearing the web view's data: %v", err)
		}
	})
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
	openExternal = func(url string) {
		if f, err := os.OpenFile(filepath.Join(dir, "external"), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600); err == nil {
			_, _ = f.WriteString(url + "\n")
			_ = f.Close()
		}
	}
	askQuit = func(_ *mygo.Window, _ Texts, b busy) quitChoice {
		asked := "unknown"
		if b.Known {
			asked = strconv.Itoa(b.N)
		}
		_ = os.WriteFile(filepath.Join(dir, "asked"), []byte(asked), 0o600)
		switch os.Getenv("MARKITAI_DESKTOP_E2E_QUIT") {
		case "now":
			return quitStopNow
		case "cancel":
			return quitCancel
		}
		return quitWhenDone
	}
}

// bundleIdentifier is CFBundleIdentifier in the Info.plist of the app bundle
// that holds this program; "" outside one.
func bundleIdentifier() string {
	exe, err := os.Executable()
	if err != nil {
		return ""
	}
	plist, err := os.ReadFile(filepath.Join(filepath.Dir(filepath.Dir(exe)), "Info.plist"))
	if err != nil {
		return ""
	}
	_, rest, _ := bytes.Cut(plist, []byte("<key>CFBundleIdentifier</key>"))
	_, rest, _ = bytes.Cut(rest, []byte("<string>"))
	id, _, _ := bytes.Cut(rest, []byte("</string>"))
	return string(bytes.TrimSpace(id))
}
