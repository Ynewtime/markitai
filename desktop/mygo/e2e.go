//go:build e2e

package main

import (
	"log"
	"os"
	"runtime"
	"strings"
	"time"

	"github.com/egoist/mygo"
	_ "github.com/egoist/mygo/plugins/automation"
)

// This file is only in builds made with build.py --e2e, which keep mygo's
// inspector: the automation plugin then drives the window through the
// socket MYGO_AUTOMATION names (internal/e2e). MARKITAI_MYGO_E2E_THEME
// starts the window light or dark without changing the system's setting.
// MARKITAI_MYGO_E2E_PAGE_EVENT names a file the driver writes a key name
// to; the app then dispatches that key's keydown in the preview's page.
func init() {
	switch os.Getenv("MARKITAI_MYGO_E2E_THEME") {
	case "light":
		mygo.Theme.SetSource(mygo.ThemeLight)
	case "dark":
		mygo.Theme.SetSource(mygo.ThemeDark)
	}
	if trigger := os.Getenv("MARKITAI_MYGO_E2E_PAGE_EVENT"); trigger != "" {
		e2ePreview = func(page *mygo.Page) { go pageEvents(trigger, page) }
	}
}

// pageEvents waits for the driver's file at trigger, removes it, and
// dispatches the keydown it names in page, as the system would deliver the
// key: the check of the page's script (previewJS) and its mygo.call
// where the driver cannot press keys, as on macOS without the
// Accessibility permission (DESIGN 16).
func pageEvents(trigger string, page *mygo.Page) {
	mod := "ctrlKey"
	if runtime.GOOS == "darwin" {
		mod = "metaKey"
	}
	keys := map[string]string{
		"zoom-in": `document.dispatchEvent(new KeyboardEvent("keydown", {code: "Equal", key: "=", ` + mod + `: true, bubbles: true, cancelable: true}))`,
	}
	for range time.Tick(100 * time.Millisecond) {
		name, err := os.ReadFile(trigger)
		if err != nil {
			continue
		}
		os.Remove(trigger)
		code, ok := keys[strings.TrimSpace(string(name))]
		if !ok {
			log.Printf("e2e page event %q: unknown", name)
			continue
		}
		if _, err := page.Eval(code); err != nil {
			log.Printf("e2e page event %q: %v", name, err)
		}
	}
}
