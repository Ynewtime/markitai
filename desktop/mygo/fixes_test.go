package main

import (
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
	"unicode/utf16"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/transfer"
	"github.com/egoist/mygo/ui"

	"markitai.local/mygo/internal/core"
	"markitai.local/mygo/internal/l10n"
	"markitai.local/mygo/internal/model"
)

// TestPasteTakesOnlyWebLinks: pasted text becomes an item only when it is
// one http or https URL with its scheme (DESIGN 7.1), as in Native.
func TestPasteTakesOnlyWebLinks(t *testing.T) {
	h := start(t, l10n.En)
	for _, text := range []string{"report.pdf", "v1.2", "example.com/page", "ftp://example.com/a", "https://example.com/a b"} {
		h.tt.SetClipboard(text)
		h.tt.Key(ui.Cmd, ui.KeyV)
		h.settle()
		if len(h.a.queue.Items) != 0 {
			t.Fatalf("pasting %q added %+v", text, h.a.queue.Items)
		}
	}
	h.tt.SetClipboard("  HTTPS://Example.com/notes\n")
	h.tt.Key(ui.Cmd, ui.KeyV)
	h.settle()
	if len(h.a.queue.Items) != 1 || h.a.queue.Items[0].Source.Kind != model.URL {
		t.Fatalf("a web link was not added: %+v", h.a.queue.Items)
	}
}

// TestDropLinkFromChromium: Chromium on Linux drags a link as
// text/x-moz-url alone, in UTF-16, the URL then its title (10.1).
func TestDropLinkFromChromium(t *testing.T) {
	utf16le := func(s string) []byte {
		var b []byte
		for _, u := range utf16.Encode([]rune(s)) {
			b = append(b, byte(u), byte(u>>8))
		}
		return b
	}
	if got := mozURLLink(utf16le("https://example.com/article\nAn article")); got != "https://example.com/article" {
		t.Errorf("UTF-16 %q", got)
	}
	if got := mozURLLink([]byte("https://example.com/b\r\nB")); got != "https://example.com/b" {
		t.Errorf("UTF-8 %q", got)
	}
	h := start(t, l10n.En)
	if !containsFormat(dropFormats.Formats, mozURL) {
		t.Fatal("the content area does not take text/x-moz-url")
	}
	h.a.dropped(transfer.New(transfer.NewItem(transfer.Bytes(mozURL, utf16le("javascript:alert(1)\nNo")))))
	h.settle()
	if len(h.a.queue.Items) != 0 {
		t.Fatalf("a script link was added: %+v", h.a.queue.Items)
	}
	h.a.dropped(transfer.New(transfer.NewItem(transfer.Bytes(mozURL, utf16le("https://example.com/article\nAn article")))))
	h.settle()
	if len(h.a.queue.Items) != 1 || h.a.queue.Items[0].Source != (model.Source{Kind: model.URL, Value: "https://example.com/article"}) {
		t.Fatalf("items %+v", h.a.queue.Items)
	}
}

// TestDropOnThePreview adds what the preview's page forwards as a drop on
// the content area adds it: the files, else the web links alone
// (DESIGN 8.3).
func TestDropOnThePreview(t *testing.T) {
	h := start(t, l10n.En)
	page := &previewService{goos: h.a.goos, press: h.a.pressed, drop: h.a.droppedOnPreview}
	page.Drop(nil, []string{"javascript:alert(1)", "file:///etc/hosts", "two words"})
	h.settle()
	if len(h.a.queue.Items) != 0 {
		t.Fatalf("items %+v", h.a.queue.Items)
	}
	page.Drop(nil, []string{"https://example.com/article"})
	h.settle()
	if len(h.a.queue.Items) != 1 || h.a.queue.Items[0].Source != (model.Source{Kind: model.URL, Value: "https://example.com/article"}) {
		t.Fatalf("items %+v", h.a.queue.Items)
	}
	page.Drop([]string{h.write("a.md", "# A\n")}, []string{"https://example.com/other"})
	h.settle()
	if h.item("a.md") == nil || len(h.a.queue.Items) != 2 {
		t.Fatalf("items %+v", h.a.queue.Items)
	}
}

func containsFormat(fs []transfer.Format, f transfer.Format) bool {
	for _, g := range fs {
		if g == f {
			return true
		}
	}
	return false
}

// TestNoticeStaysOnItsPage: a notice shows on the page it was given on,
// for 8 s from then, and in the stacked layout's detail too (6.7).
func TestNoticeStaysOnItsPage(t *testing.T) {
	h := start(t, l10n.En)
	dir := filepath.Join(h.home, "Downloads", "Scans")
	os.MkdirAll(dir, 0o755)
	h.a.addPaths([]string{dir})
	h.settle()
	notice := h.T("notice.folder", "name", "Scans")
	h.wantText(notice)
	h.click(h.T("nav.settings"))
	h.rest()
	h.wantNoText(notice)
	h.click(h.T("nav.convert"))
	h.rest()
	h.wantText(notice)
	// The notice's 8 s run from when it was given, not from a frame that
	// showed it later.
	h.a.notice.at = h.tt.Now().Add(-7 * time.Second)
	h.tt.Advance(1500 * time.Millisecond)
	h.tt.Frame()
	h.wantNoText(notice)

	// A notice given in a stacked detail shows there.
	h.convert(h.write("a.md", "# A\n"))
	h.tt.SetSize(800, 780)
	h.rest()
	h.click(h.rowLabel("a.md"))
	h.rest()
	if h.a.router.Path() != pathConvertItem {
		t.Fatalf("path %s", h.a.router.Path())
	}
	h.a.setNotice(h.T("error.save_failed"))
	h.tt.Frame()
	h.wantText(h.T("error.save_failed"))
}

// TestSaveAsAndFolderDialogs start in the output folder, or its nearest
// folder that exists, never by a bare name in the working directory, the
// data folder. A name without an extension takes .md and replaces
// nothing the dialog did not ask about; a failure shows on the page Save
// As began on (DESIGN 6.7).
func TestSaveAsAndFolderDialogs(t *testing.T) {
	h := start(t, l10n.En)
	h.convert(h.write("a.md", "# A\n"))
	done := h.item("a.md").State.(model.Done)
	saved := filepath.Join(h.home, "Saved")
	os.MkdirAll(saved, 0o755)
	notes := filepath.Join(saved, "notes.md")
	saveAs := func(answer string) {
		h.desk.saveTo = answer
		h.tt.Key(ui.Cmd|ui.Shift, ui.KeyS)
		h.settle()
	}

	saveAs(filepath.Join(saved, "notes"))
	h.until("notes.md", func() bool { b, _ := os.ReadFile(notes); return string(b) == done.Result.Markdown })
	if want := filepath.Join(h.a.output(), filepath.Base(done.Folder)+".md"); h.desk.starts[0] != want {
		t.Errorf("Save As started at %q, want %q", h.desk.starts[0], want)
	}

	// The dialog answers once the person has gone on to Settings.
	os.WriteFile(notes, []byte("mine"), 0o644)
	var answer func()
	h.desk.post = func(fn func()) { answer = fn }
	saveAs(filepath.Join(saved, "notes"))
	h.desk.post = func(fn func()) { h.posts <- fn }
	h.click(h.T("nav.settings"))
	if answer == nil {
		t.Fatal("Save As showed no dialog")
	}
	h.posts <- answer
	h.until("the failure", func() bool { return h.a.notice.text == h.T("error.save_failed") })
	if b, _ := os.ReadFile(notes); string(b) != "mine" || h.a.notice.page != pathConvert {
		t.Fatalf("notes.md %q, the notice on %q", b, h.a.notice.page)
	}
	h.click(h.T("nav.convert"))
	h.wantText(h.T("error.save_failed"))

	saveAs(notes)
	h.until("notes.md replaced", func() bool { b, _ := os.ReadFile(notes); return string(b) == done.Result.Markdown })

	missing := filepath.Join(h.home, "Gone", "Markitai")
	h.a.settings.OutputFolder = &missing
	h.desk.saveTo = ""
	h.tt.Key(ui.Cmd|ui.Shift, ui.KeyS)
	h.settle()
	h.a.chooseOutput()
	h.settle()
	starts := h.desk.starts[len(h.desk.starts)-2:]
	if want := []string{filepath.Join(h.home, filepath.Base(done.Folder)+".md"), h.home}; !slices.Equal(starts, want) {
		t.Errorf("with the output folder gone the dialogs started at %q, want %q", starts, want)
	}
}

// failingPreview is the core whose preview_html fails while fail holds.
type failingPreview struct {
	coreLib
	fail *bool
}

func (f failingPreview) PreviewHTML(markdown, theme string, baseDir, filesURL *string) (string, error) {
	if *f.fail {
		return "", errors.New("internal_error: test")
	}
	return f.coreLib.PreviewHTML(markdown, theme, baseDir, filesURL)
}

// TestPreviewFailure: a preview the core cannot render says so, and is
// rendered again the next time the Preview tab shows the item (DESIGN 8).
func TestPreviewFailure(t *testing.T) {
	h := start(t, l10n.En)
	fail := true
	h.a.core = failingPreview{h.a.core, &fail}
	h.convert(h.write("a.md", "# A\n"))
	h.click(h.rowLabel("a.md"))
	h.until("the failure", func() bool { return h.has(h.T("preview.failed")) })
	loads := len(h.pane.html)
	h.rest()
	h.wantText(h.T("preview.failed"))
	if len(h.pane.html) != loads {
		t.Error("the failed document was loaded")
	}

	fail = false
	h.click(h.T("detail.markdown"))
	h.click(h.T("detail.preview"))
	h.until("the preview", func() bool { return len(h.pane.html) > loads })
	h.settle()
	h.wantNoText(h.T("preview.failed"))
	if !strings.Contains(h.pane.html[len(h.pane.html)-1], "<h1") {
		t.Errorf("the document %.120q", h.pane.html[len(h.pane.html)-1])
	}
}

// TestRemoveKeyLeavesTextFields: while the URL field or History's search
// has the focus, ⌘⌫ deletes to the line's start there: the menu's Remove
// gives up the key until the focus leaves (DESIGN 11).
func TestRemoveKeyLeavesTextFields(t *testing.T) {
	h := start(t, l10n.En)
	h.a.newMenuBar(h.a.pressed)
	remove := h.a.menu.items[actRemove]
	h.convert(h.write("a.md", "# A\n"))
	h.click(h.rowLabel("a.md"))
	if remove.Accelerator != removeAccelerator {
		t.Fatalf("Remove's key %q", remove.Accelerator)
	}
	h.click(h.T("convert.url.label"))
	h.tt.Frame()
	if remove.Accelerator != "" {
		t.Errorf("Remove kept %q in the URL field", remove.Accelerator)
	}
	h.click(h.rowLabel("a.md"))
	h.tt.Frame()
	if remove.Accelerator != removeAccelerator {
		t.Errorf("Remove's key %q after the field", remove.Accelerator)
	}
	h.click(h.T("nav.history"))
	h.click(h.T("history.search"))
	h.tt.Frame()
	if remove.Accelerator != "" {
		t.Errorf("Remove kept %q in the search field", remove.Accelerator)
	}
}

// TestMenuOrder follows DESIGN 11: the item commands in File after Save
// As and Show, the zoom items In, Out, Actual Size.
func TestMenuOrder(t *testing.T) {
	h := start(t, l10n.En)
	bar := h.a.newMenuBar(h.a.pressed)
	labels := map[string][]string{}
	for _, top := range bar.Items() {
		for _, it := range top.Submenu {
			label := it.Label
			switch {
			case it.ID != "":
				label = it.ID
			case it.Role != "":
				label = string(it.Role)
			}
			if label == "" {
				label = "-"
			}
			labels[string(top.Role)] = append(labels[string(top.Role)], label)
		}
	}
	file := strings.Join(labels["fileMenu"], " ")
	if !strings.HasSuffix(file, "- "+string(mygo.RoleClose)+" save_as show - retry remove") {
		t.Errorf("File %q", file)
	}
	edit := strings.Join(labels["editMenu"], " ")
	if !strings.HasSuffix(edit, "- copy - search") {
		t.Errorf("Edit %q", edit)
	}
	view := strings.Join(labels["viewMenu"], " ")
	if !strings.HasSuffix(view, "zoom_in zoom_out actual_size") || strings.Contains(view, "retry") {
		t.Errorf("View %q", view)
	}
}

// countingConfig counts the core's ConfigFile calls.
type countingConfig struct {
	coreLib
	n *int
}

func (c countingConfig) ConfigFile() (core.ConfigFile, error) {
	*c.n++
	return c.coreLib.ConfigFile()
}

// TestSettingsReadsConfigOnce: Settings asks the core for the configuration
// file as it opens and after Create or Open, not on every frame (DESIGN 6.6).
func TestSettingsReadsConfigOnce(t *testing.T) {
	h := start(t, l10n.En)
	n := 0
	h.a.core = countingConfig{h.a.core, &n}
	h.click(h.T("nav.settings"))
	h.rest()
	h.tt.Advance(time.Second)
	h.rest()
	if n != 1 {
		t.Errorf("opening Settings read the configuration %d times", n)
	}
	h.click(h.T("settings.config.create"))
	h.rest()
	h.rest()
	// openConfig asks once itself, then the page once more.
	if n != 3 {
		t.Errorf("after Create the configuration was read %d times in all", n)
	}
	h.wantText(h.T("settings.config.open"))
	h.click(h.T("nav.convert"))
	h.click(h.T("nav.settings"))
	h.rest()
	if n != 4 {
		t.Errorf("coming back to Settings read the configuration %d times in all", n)
	}
}
