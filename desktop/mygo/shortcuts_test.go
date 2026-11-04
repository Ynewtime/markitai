package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/egoist/mygo/ui"

	"markitai.local/mygo/internal/l10n"
	"markitai.local/mygo/internal/model"
)

// key is a section 9.2 key, sent to the window or pressed in the preview
// and forwarded by its script.
type key func(h *harness, act action, mods ui.Modifiers, k ui.Key)

func native(h *harness, _ action, mods ui.Modifiers, k ui.Key) {
	h.tt.Key(mods, k)
	h.settle()
}

func forwarded(h *harness, act action, _ ui.Modifiers, _ ui.Key) {
	keys := &previewKeys{goos: h.a.goos, press: h.a.pressed}
	keys.Press(string(act))
	h.settle()
	h.settle()
}

func TestShortcuts(t *testing.T) {
	for name, press := range map[string]key{"native": native, "forwarded": forwarded} {
		t.Run(name, func(t *testing.T) {
			h := start(t, l10n.En)
			pdf := h.inHome(pdfWarnings)
			h.convert(pdf)
			done := h.item("policy-assets.pdf").State.(model.Done)

			press(h, actHistory, ui.Cmd, ui.Key2)
			if h.a.router.Path() != pathHistory {
				t.Errorf("Go to History: %s", h.a.router.Path())
			}
			press(h, actSettings, ui.Cmd, ui.KeyComma)
			if h.a.router.Path() != pathSettings {
				t.Errorf("Settings: %s", h.a.router.Path())
			}
			press(h, actConvert, ui.Cmd, ui.Key1)
			if h.a.router.Path() != pathConvert {
				t.Errorf("Go to Convert: %s", h.a.router.Path())
			}

			press(h, actToggleView, ui.Cmd|ui.Shift, ui.KeyM)
			if h.a.detail != showMarkdown || !h.has(h.T("detail.markdown")) {
				t.Error("Switch Preview and Markdown")
			}
			press(h, actToggleView, ui.Cmd|ui.Shift, ui.KeyM)
			press(h, actCopy, ui.Cmd|ui.Shift, ui.KeyC)
			if h.tt.Clipboard() != done.Result.Markdown {
				t.Error("Copy Markdown")
			}
			h.desk.saveTo = filepath.Join(h.home, "Saved", "copy.md")
			os.MkdirAll(filepath.Dir(h.desk.saveTo), 0o755)
			press(h, actSaveAs, ui.Cmd|ui.Shift, ui.KeyS)
			h.until("the saved file", func() bool { _, err := os.Stat(h.desk.saveTo); return err == nil })
			if len(h.desk.suggests) != 1 || h.desk.suggests[0] != filepath.Base(done.Folder)+".md" {
				t.Errorf("Save As suggested %q", h.desk.suggests)
			}
			press(h, actShow, ui.Cmd|ui.Shift, ui.KeyR)
			if len(h.desk.shown) != 1 || h.desk.shown[0] != done.Result.MarkdownPath {
				t.Errorf("Show in folder %q", h.desk.shown)
			}

			press(h, actZoomIn, ui.Cmd, ui.KeyEqual)
			press(h, actZoomIn, ui.Cmd|ui.Shift, ui.KeyEqual)
			if h.a.settings.Zoom != 1.2 || h.savedSettings()["view.zoom"] != 1.2 {
				t.Errorf("Zoom In: %v", h.a.settings.Zoom)
			}
			press(h, actZoomOut, ui.Cmd, ui.KeyMinus)
			if h.a.settings.Zoom != 1.1 || h.pane.zoom != 1.1 {
				t.Errorf("Zoom Out: %v, preview %v", h.a.settings.Zoom, h.pane.zoom)
			}
			press(h, actActualSize, ui.Cmd, ui.Key0)
			if h.a.settings.Zoom != 1 {
				t.Errorf("Actual Size: %v", h.a.settings.Zoom)
			}

			h.desk.files = []string{h.write("chosen.md", "# Chosen\n")}
			press(h, actChoose, ui.Cmd, ui.KeyO)
			if h.item("chosen.md") == nil {
				t.Error("Choose Files")
			}
			h.until("chosen", func() bool { return model.Count[model.Converting](&h.a.queue) == 0 })

			press(h, actSearch, ui.Cmd, ui.KeyF)
			h.rest()
			if h.a.router.Path() != pathHistory || !h.tt.Focused(h.T("history.search")) {
				t.Errorf("Search History: %s", h.a.router.Path())
			}
			press(h, actAddURL, ui.Cmd, ui.KeyL)
			h.rest()
			if h.a.router.Path() != pathConvert || !h.tt.Focused(h.T("convert.url.label")) {
				t.Errorf("Add Web Address: %s", h.a.router.Path())
			}
		})
	}
}

func TestRetryShortcut(t *testing.T) {
	h := start(t, l10n.En)
	h.convert(h.inHome(pngNoText))
	h.tt.Key(ui.Cmd, ui.KeyR)
	h.tt.Frame()
	if _, failed := h.item("english.png").State.(model.Failed); failed {
		t.Error("Retry did not queue the item")
	}
	waitState[model.Failed](h, "english.png")
	h.settle()
	(&previewKeys{goos: h.a.goos, press: h.a.pressed}).Press(string(actRetry))
	if len(h.posts) != 0 {
		t.Error("the preview forwarded a key it does not have")
	}
}

func TestLinuxCloseAndQuit(t *testing.T) {
	h := start(t, l10n.En)
	h.a.goos = "linux"
	h.tt.Key(ui.Cmd, ui.KeyW)
	h.tt.Key(ui.Cmd, ui.KeyQ)
	h.settle()
	if h.desk.closes != 1 || h.desk.quits != 1 {
		t.Errorf("closes %d, quits %d", h.desk.closes, h.desk.quits)
	}
	forwarded(h, actClose, 0, 0)
	if h.desk.closes != 2 {
		t.Error("the preview did not forward Close")
	}
}

func TestShortcutsWaitForTheAlert(t *testing.T) {
	h := start(t, l10n.En)
	h.convert(h.write("a.md", "# A\n"))
	h.a.asking = askingClearHistory
	h.tt.Frame()
	h.tt.Key(ui.Cmd, ui.Key2)
	h.settle()
	if h.a.router.Path() != pathConvert {
		t.Error("a shortcut ran under the alert")
	}
	h.tt.Key(0, ui.KeyEscape)
	h.settle()
	if h.a.asking != askingNone || len(h.a.history.Entries) != 1 {
		t.Error("Escape did not cancel the alert")
	}
}

func TestPreviewKeysScript(t *testing.T) {
	for _, goos := range []string{"darwin", "windows", "linux"} {
		js := previewKeysJS(goos)
		for _, b := range bindings(goos) {
			if strings.Contains(js, `"action":"`+string(b.act)+`"`) != b.page {
				t.Errorf("%s: %s forwarded is not %v", goos, b.act, b.page)
			}
		}
		mod := map[string]string{"darwin": "e.metaKey ===", "windows": "e.ctrlKey ===", "linux": "e.ctrlKey ==="}[goos]
		if !strings.Contains(js, mod) || !strings.Contains(js, `mygo?.call("PreviewKeys.Press"`) {
			t.Errorf("%s script %s", goos, js)
		}
	}
	if strings.Contains(previewKeysJS("darwin"), `"KeyQ"`) || !strings.Contains(previewKeysJS("linux"), `"KeyQ"`) {
		t.Error("Quit is forwarded on Linux only")
	}
}

func TestMenuAccelerators(t *testing.T) {
	want := map[action]string{
		actChoose: "CmdOrCtrl+O", actAddURL: "CmdOrCtrl+L", actSaveAs: "CmdOrCtrl+Shift+S", actShow: "CmdOrCtrl+Shift+R",
		actCopy: "CmdOrCtrl+Shift+C", actSearch: "CmdOrCtrl+F", actConvert: "CmdOrCtrl+1", actHistory: "CmdOrCtrl+2",
		actToggleView: "CmdOrCtrl+Shift+M", actActualSize: "CmdOrCtrl+0", actZoomIn: "CmdOrCtrl+=", actZoomOut: "CmdOrCtrl+-",
		actRetry: "CmdOrCtrl+R", actSettings: "CmdOrCtrl+,",
	}
	for act, acc := range want {
		if got := accelerator(act); got != acc {
			t.Errorf("%s: %q, want %q", act, got, acc)
		}
	}
}
