package main

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/egoist/mygo/transfer"
	"github.com/egoist/mygo/ui"

	"markitai.local/mygo/internal/l10n"
	"markitai.local/mygo/internal/model"
)

func TestEmptyConvert(t *testing.T) {
	for _, lang := range []l10n.Lang{l10n.En, l10n.ZhHans} {
		t.Run(string(lang), func(t *testing.T) {
			h := start(t, lang)
			// DESIGN 4.3: ~ for the home folder on macOS and Linux, full
			// paths on Windows.
			folder := "~" + string(filepath.Separator) + filepath.Join("Documents", "Markitai")
			if runtime.GOOS == "windows" {
				folder = filepath.Join(h.home, "Documents", "Markitai")
			}
			for _, s := range []string{
				h.T("convert.empty.title"), h.T("convert.empty.body", "folder", folder), h.T("convert.choose"),
				h.T("convert.or"), h.T("convert.url.add"), h.T("nav.convert"), h.T("nav.history"), h.T("nav.settings"),
			} {
				h.wantText(s)
			}
			if !h.tt.Focused(h.T("convert.choose")) {
				t.Error("Choose Files is not focused")
			}
			h.find(h.T("a11y.drop_zone"))
			h.find(h.T("convert.url.label"))
		})
	}
}

func TestURLField(t *testing.T) {
	h := start(t, l10n.En)
	h.click(h.T("convert.url.label"))
	h.tt.Type("not a link")
	h.tt.Key(0, ui.KeyEnter)
	h.settle()
	h.wantText(h.T("convert.url.invalid"))
	h.tt.Type("x")
	h.settle()
	h.wantNoText(h.T("convert.url.invalid"))

	h.tt.Key(ui.Cmd, ui.KeyA)
	h.tt.Type("example.com")
	h.click(h.T("convert.url.add"))
	if len(h.a.queue.Items) != 1 || h.a.queue.Items[0].Source != (model.Source{Kind: model.URL, Value: "https://example.com"}) {
		t.Fatalf("items %+v", h.a.queue.Items)
	}
	if h.a.url != "" {
		t.Errorf("the field keeps %q", h.a.url)
	}
}

func TestDropToDone(t *testing.T) {
	h := start(t, l10n.En)
	pdf := h.inHome(pdfWarnings)
	h.drop(h.T("a11y.drop_zone"), pdf)
	done := waitState[model.Done](h, "policy-assets.pdf")
	h.settle()

	name := "policy-assets.pdf, " + h.a.doneText(done.Result.Duration, len(done.Result.Warnings))
	h.find(name)
	if !h.announced(h.T("a11y.done", "title", "policy-assets.pdf")) {
		t.Errorf("announcements %q", h.tt.Announcements())
	}
	if len(done.Result.Warnings) == 0 {
		t.Fatal("the fixture gave no warning")
	}

	h.until("the preview to load", func() bool { return len(h.pane.html) > 0 })
	h.settle()
	if got, want := h.pane.base[len(h.pane.base)-1], folderURL(done.Folder); got != want {
		t.Errorf("base URL %q, want %q", got, want)
	}
	if got := h.pane.backgrounds; len(got) == 0 || got[len(got)-1] != "#ffffff" {
		t.Errorf("the web view is filled with %v, want the light detail's #ffffff last", got)
	}
	if html := h.pane.html[len(h.pane.html)-1]; !strings.Contains(html, `data-theme="light"`) {
		t.Errorf("the document is not light: %.200s", html)
	}
	r := h.find(h.T("a11y.preview"))
	if h.tt.WebViewAt(r.X+r.W/2, r.Y+r.H/2) != h.pane.View() {
		t.Error("the web view does not show in the Preview tab")
	}

	h.click(h.T("detail.markdown"))
	if got := h.markdownText(); got != done.Result.Markdown {
		t.Errorf("Markdown text %q, want %q", got, done.Result.Markdown)
	}

	h.click(h.T("detail.copy"))
	if h.tt.Clipboard() != done.Result.Markdown {
		t.Error("Copy Markdown did not copy the Markdown")
	}
	if !h.announced(h.T("notice.copied")) {
		t.Error("Copied was not announced")
	}
	h.click(h.a.showLabel())
	if len(h.desk.shown) != 1 || h.desk.shown[0] != done.Result.MarkdownPath {
		t.Errorf("shown %q", h.desk.shown)
	}
	if _, err := os.Stat(done.Result.MarkdownPath); err != nil {
		t.Error(err)
	}
}

func TestDropURLAndFolder(t *testing.T) {
	h := start(t, l10n.En)
	link, err := transfer.URLData("https://example.com/article")
	if err != nil {
		t.Fatal(err)
	}
	// The Tester sends file drops only: the drop of a link comes through
	// the app's handler of ui.DropData.
	h.a.dropped(link)
	h.settle()
	if len(h.a.queue.Items) != 1 || h.a.queue.Items[0].Source.Kind != model.URL {
		t.Fatalf("items %+v", h.a.queue.Items)
	}

	dir := filepath.Join(h.home, "Downloads", "Scans")
	os.MkdirAll(dir, 0o755)
	h.drop(h.T("a11y.drop_zone"), dir)
	h.wantText(h.T("notice.folder", "name", "Scans"))
	if !h.announced(h.T("notice.folder", "name", "Scans")) {
		t.Error("the notice was not announced")
	}
	if len(h.a.queue.Items) != 1 {
		t.Errorf("the folder was added: %d items", len(h.a.queue.Items))
	}
}

func TestPaste(t *testing.T) {
	h := start(t, l10n.En)
	pdf := h.inHome(pdfTable)
	data, err := transfer.FileData(pdf)
	if err != nil {
		t.Fatal(err)
	}
	h.tt.SetClipboardData(data)
	h.tt.Key(ui.Cmd, ui.KeyV)
	h.settle()
	if h.item("default-table.pdf") == nil {
		t.Fatalf("clipboard files were not added: %+v", h.a.queue.Items)
	}

	h.tt.SetClipboard("https://example.com/notes")
	h.tt.Key(ui.Cmd, ui.KeyV)
	h.settle()
	if h.item("example.com/notes") == nil {
		t.Fatalf("a clipboard URL was not added: %+v", h.a.queue.Items)
	}

	h.click(h.T("convert.url.label"))
	h.tt.SetClipboard("https://example.com/typed")
	h.tt.Key(ui.Cmd, ui.KeyV)
	h.settle()
	if h.a.url != "https://example.com/typed" || len(h.a.queue.Items) != 2 {
		t.Errorf("paste in the field: field %q, %d items", h.a.url, len(h.a.queue.Items))
	}
}

func TestTwoAtATime(t *testing.T) {
	h := start(t, l10n.En)
	paths := []string{h.write("a.md", "# A\n"), h.write("b.md", "# B\n"), h.write("c.md", "# C\n")}
	r := h.find(h.T("a11y.drop_zone"))
	h.tt.DropFiles(r.X+r.W/2, r.Y+r.H/2, paths...)
	h.tt.Frame()
	if n, q := model.Count[model.Converting](&h.a.queue), model.Count[model.Queued](&h.a.queue); n != 2 || q != 1 {
		t.Fatalf("%d converting, %d queued", n, q)
	}
	h.find("a.md, " + h.T("state.converting"))
	h.find("b.md, " + h.T("state.converting"))
	h.find("c.md, " + h.T("state.queued"))
	h.until("all done", func() bool { return model.Count[model.Done](&h.a.queue) == 3 })
}

func TestDuplicate(t *testing.T) {
	h := start(t, l10n.En)
	path := h.write("a.md", "# A\n")
	r := h.find(h.T("a11y.drop_zone"))
	h.tt.DropFiles(r.X+r.W/2, r.Y+r.H/2, path)
	h.tt.Frame()
	r = h.find(h.T("a11y.drop_zone"))
	h.tt.DropFiles(r.X+r.W/2, r.Y+r.H/2, path)
	h.tt.Frame()
	h.wantText(h.T("notice.duplicate", "name", "a.md"))
	if len(h.a.queue.Items) != 1 {
		t.Errorf("%d items", len(h.a.queue.Items))
	}
	h.until("done", func() bool { return model.Count[model.Done](&h.a.queue) == 1 })
	h.tt.Advance(8 * 1e9)
	h.tt.Frame()
	h.wantNoText(h.T("notice.duplicate", "name", "a.md"))
}

func TestFailure(t *testing.T) {
	h := start(t, l10n.En)
	png := h.inHome(pngNoText)
	h.drop(h.T("a11y.drop_zone"), png)
	failed := waitState[model.Failed](h, "english.png")
	h.settle()
	if failed.Code != "image_only" {
		t.Fatalf("code %q: %s", failed.Code, failed.Message)
	}
	sentence := h.T("error.image_only", "settings", h.T("nav.settings"))
	h.wantText(sentence)
	h.wantText(h.T("detail.retry"))
	h.wantText(h.T("detail.details"))
	h.find("english.png, " + h.T("row.failed", "reason", sentence))
	if !h.announced(h.T("a11y.failed", "title", "english.png")) {
		t.Errorf("announcements %q", h.tt.Announcements())
	}
	entries, _ := os.ReadDir(h.a.output())
	if len(entries) != 0 {
		t.Errorf("the claimed folder stayed: %v", entries)
	}
	h.click(h.T("action.open_settings", "settings", h.T("nav.settings")))
	if h.a.router.Path() != pathSettings {
		t.Errorf("page %s", h.a.router.Path())
	}

	h.click(h.T("nav.convert"))
	h.a.addPaths([]string{filepath.Join(h.home, "Downloads", "gone.pdf")})
	h.settle()
	gone := waitState[model.Failed](h, "gone.pdf")
	if gone.Code != "not_found" {
		t.Errorf("a missing file: %q %s", gone.Code, gone.Message)
	}
	h.rest()
	h.click("gone.pdf, " + h.T("row.failed", "reason", h.T("error.not_found")))
	h.wantText(h.T("error.not_found"))
	h.click(h.T("convert.remove"))
	if h.item("gone.pdf") != nil {
		t.Error("Remove from List kept the item")
	}
}

func TestRetry(t *testing.T) {
	h := start(t, l10n.En)
	h.drop(h.T("a11y.drop_zone"), h.inHome(pngNoText))
	waitState[model.Failed](h, "english.png")
	h.settle()
	if h.item("english.png").Options.Config != model.ConfigUser {
		t.Fatal("options")
	}
	h.a.settings.Config = model.ConfigBuiltin
	if err := h.tt.Click(h.T("detail.retry")); err != nil {
		t.Fatal(err)
	}
	it := h.item("english.png")
	if _, failed := it.State.(model.Failed); failed || it.Options.Config != model.ConfigBuiltin {
		t.Errorf("retry: state %T, options %+v", it.State, it.Options)
	}
	waitState[model.Failed](h, "english.png")
}

func TestRemoveAndClearFinished(t *testing.T) {
	h := start(t, l10n.En)
	a, b, c := h.write("a.md", "# A\n"), h.write("b.md", "# B\n"), h.write("c.md", "# C\n")
	h.drop(h.T("a11y.drop_zone"), a)
	h.until("a done", func() bool { return model.Count[model.Done](&h.a.queue) == 1 })
	r := h.find(h.T("a11y.drop_zone"))
	h.tt.DropFiles(r.X+r.W/2, r.Y+r.H/2, b, c)
	h.rest()
	// Rows are b, c (converting), a (done), newest first.
	if err := h.tt.RightClick("b.md, " + h.T("state.converting")); err != nil {
		t.Fatal(err)
	}
	if err := h.tt.ChooseMenuItem(h.T("convert.remove")); err == nil {
		h.settle()
	}
	if h.item("b.md") == nil {
		t.Fatal("a converting row was removed")
	}
	h.tt.CloseMenu()

	h.until("all done", func() bool { return model.Count[model.Done](&h.a.queue) == 3 })
	h.rest()
	h.click(h.item("c.md").Title + ", " + h.a.subtitle(h.item("c.md")))
	mods, key := removeKey(h.a.goos)
	h.tt.Key(mods, key)
	h.settle()
	if h.item("c.md") != nil {
		t.Fatal("the selected row stayed")
	}
	if sel := h.a.queue.Item(h.a.queue.Selected); sel == nil || sel.Title != "a.md" {
		t.Errorf("the selection went to %+v, want the next row", sel)
	}
	h.click(h.T("convert.clear_finished"))
	if len(h.a.queue.Items) != 0 {
		t.Errorf("%d items after Clear Finished", len(h.a.queue.Items))
	}
	h.wantText(h.T("convert.empty.title"))
	if _, err := os.Stat(filepath.Join(h.a.output(), "a")); err != nil {
		t.Errorf("Clear Finished touched the files: %v", err)
	}
}
