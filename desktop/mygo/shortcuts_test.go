package main

import (
	"math"
	"os"
	"path/filepath"
	"slices"
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
	keys := &previewService{goos: h.a.goos, press: h.a.pressed}
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
			if want := filepath.Join(h.a.output(), filepath.Base(done.Folder)+".md"); len(h.desk.starts) != 1 || h.desk.starts[0] != want {
				t.Errorf("Save As started at %q, want %q", h.desk.starts, want)
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

// TestFocusFromThePreview leaves the web view out of the frame a key
// forwarded from it focuses a field in: mygo gives the keyboard back to
// the content only as it hides the web view (DESIGN 14). The next frame
// shows it again.
func TestFocusFromThePreview(t *testing.T) {
	for _, step := range []struct {
		act   action
		field string
	}{{actAddURL, "convert.url.label"}, {actSearch, "history.search"}} {
		h := start(t, l10n.En)
		h.convert(h.inHome(pdfWarnings))
		if step.act == actSearch {
			h.click(h.T("nav.history"))
			h.click(h.entryLabel("policy-assets.pdf"))
		}
		h.until("the preview", func() bool { _, ok := h.tt.Find(h.T("a11y.preview")); return ok })
		r := h.find(h.T("a11y.preview"))
		x, y := r.X+r.W/2, r.Y+r.H/2
		if h.tt.WebViewAt(x, y) != h.pane.View() {
			t.Fatalf("%s: no web view", step.act)
		}
		var shown []bool
		h.frame = func() { shown = append(shown, h.tt.WebViewAt(x, y) != nil) }
		(&previewService{goos: h.a.goos, press: h.a.pressed}).Press(string(step.act))
		h.settle()
		h.frame = nil
		// The frame the key focuses the field in leaves the web view out;
		// the next one brings it back.
		if !slices.Contains(shown, false) || h.tt.WebViewAt(x, y) != h.pane.View() || !h.tt.Focused(h.T(step.field)) {
			t.Errorf("%s: placed %v, then %v; field focused %v", step.act, shown, h.tt.WebViewAt(x, y) != nil, h.tt.Focused(h.T(step.field)))
		}
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
	(&previewService{goos: h.a.goos, press: h.a.pressed}).Press(string(actRetry))
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

func TestPreviewScript(t *testing.T) {
	for _, goos := range []string{"darwin", "windows", "linux"} {
		js := previewJS(goos)
		for _, b := range bindings(goos) {
			if strings.Contains(js, `"action":"`+string(b.act)+`"`) != b.page {
				t.Errorf("%s: %s forwarded is not %v", goos, b.act, b.page)
			}
		}
		mod := map[string]string{"darwin": "e.metaKey ===", "windows": "e.ctrlKey ===", "linux": "e.ctrlKey ==="}[goos]
		if !strings.Contains(js, mod) || !strings.Contains(js, `mygo?.call("Preview." + method`) {
			t.Errorf("%s script %s", goos, js)
		}
	}
	if strings.Contains(previewJS("darwin"), `"KeyQ"`) || !strings.Contains(previewJS("linux"), `"KeyQ"`) {
		t.Error("Quit is forwarded on Linux only")
	}
	// WebView2 alone zooms the page on Ctrl and the wheel: there the
	// wheel steps view.zoom.
	for _, goos := range []string{"darwin", "windows", "linux"} {
		js := previewJS(goos)
		wheel := strings.Contains(js, `addEventListener("wheel"`)
		if wheel != (goos == "windows") || wheel && (!strings.Contains(js, "passive: false") || !strings.Contains(js, `"zoom_in"`) || !strings.Contains(js, `"zoom_out"`)) {
			t.Errorf("%s: the wheel script %v", goos, wheel)
		}
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

// TestWheelZoom steps view.zoom once a notch of Ctrl and the wheel over the
// preview at any zoom: WebView2 gives the page a notch as 120 over the
// page's zoom, in whole pixels (DESIGN 8.3). Turning the wheel back starts
// the count again.
func TestWheelZoom(t *testing.T) {
	h := start(t, l10n.En)
	h.convert(h.inHome(pdfWarnings))
	page := &previewService{goos: "windows", wheel: h.a.wheeled}
	// wheel turns the wheel by notches as the page at its zoom gets them.
	wheel := func(notches float64) {
		page.Wheel(math.Trunc(120 * notches / h.pane.zoom))
		h.settle()
		h.settle()
	}
	want := func(step string, z float64) {
		t.Helper()
		if h.a.settings.Zoom != z || h.pane.zoom != z {
			t.Errorf("%s: view.zoom %v, the page's %v; want %v", step, h.a.settings.Zoom, h.pane.zoom, z)
		}
	}
	wheel(1)
	want("a notch up at 1", 1.1)
	h.a.zoom(2.3)
	h.settle()
	for _, z := range []float64{2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3} {
		wheel(1)
		want("a notch up", z)
	}
	wheel(-1)
	want("a notch down at 3", 2.9)
	wheel(0.5)
	wheel(-1)
	want("half a notch up, then a notch down", 2.8)
	wheel(-0.5)
	want("half a notch down", 2.8)
	wheel(-0.5)
	want("another half", 2.7)
}
