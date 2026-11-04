package main

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/egoist/mygo/ui"

	"markitai.local/mygo/internal/l10n"
	"markitai.local/mygo/internal/model"
)

// mygoText is what mygo itself names in its widgets: the split's divider
// for assistive technology, in the app's language.
var mygoText = map[l10n.Lang][]string{
	l10n.En:     {"Divider"},
	l10n.ZhHans: {"分隔条"},
}

// catalog matches a language's copy: each value, its placeholders any
// text.
type catalog []*regexp.Regexp

func newCatalog(lang l10n.Lang) catalog {
	var c catalog
	placeholder := regexp.MustCompile(`\\\{[a-z_]+\\\}`)
	pattern := func(k string) string { return placeholder.ReplaceAllString(regexp.QuoteMeta(lang.Raw(k)), ".+") }
	for _, k := range l10n.Keys() {
		c = append(c, regexp.MustCompile("^"+pattern(k)+"$"))
	}
	// The quit alert's body joins two sentences (6.7).
	sep := " "
	if lang == l10n.ZhHans {
		sep = ""
	}
	for _, k := range []string{"quit.body.one", "quit.body.other"} {
		c = append(c, regexp.MustCompile("^"+pattern(k)+sep+pattern("quit.body.queued")+"$"))
	}
	return c
}

func (c catalog) has(s string) bool {
	for _, re := range c {
		if re.MatchString(s) {
			return true
		}
	}
	return false
}

// copyWalk checks that every text of the frame is the catalog's, joined
// as rows and footers join them, or a value the app shows as it is.
func (h *harness) copyWalk(screen string, c catalog, values map[string]bool) {
	h.t.Helper()
	ok := func(s string) bool { return s == "" || values[s] || c.has(s) }
	// joined is a text of parts that rows, names and footers join with
	// ", " and " · ", each part one of the above.
	var joined func(s string) bool
	joined = func(s string) bool {
		if ok(s) {
			return true
		}
		for _, sep := range []string{", ", " · "} {
			for i := strings.Index(s, sep); i >= 0; {
				if ok(s[:i]) && joined(s[i+len(sep):]) {
					return true
				}
				next := strings.Index(s[i+len(sep):], sep)
				if next < 0 {
					break
				}
				i += len(sep) + next
			}
		}
		return false
	}
	for _, s := range h.tt.Texts() {
		if !joined(s) {
			h.t.Errorf("%s: %q is not in the %s copy", screen, s, h.a.lang)
		}
	}
}

// values are the texts the app shows as they are: names, paths,
// versions, the core's warnings and messages, and mygo's own.
func (h *harness) values() map[string]bool {
	v := map[string]bool{h.lib.Version(): true, h.a.version: true, h.a.query: true, h.a.abbreviated(h.a.output()): true}
	for _, s := range mygoText[h.a.lang] {
		v[s] = true
	}
	if cfg, err := h.lib.ConfigFile(); err == nil {
		v[h.a.abbreviated(cfg.Path)] = true
	}
	for _, it := range h.a.queue.Items {
		v[it.Title], v[h.a.abbreviated(it.Source.Value)], v[it.Source.Value] = true, true, true
		switch s := it.State.(type) {
		case model.Done:
			for _, w := range s.Result.Warnings {
				v[w] = true
			}
		case model.Failed:
			v[s.Message] = true
		}
	}
	for _, e := range h.a.history.Entries {
		v[e.Title], v[h.a.abbreviated(e.Source)] = true, true
		for _, w := range e.Warnings {
			v[w] = true
		}
	}
	return v
}

// TestCopy walks every screen in both languages: each text it shows comes
// from strings.json.
func TestCopy(t *testing.T) {
	for _, lang := range []l10n.Lang{l10n.En, l10n.ZhHans} {
		t.Run(string(lang), func(t *testing.T) {
			h := start(t, lang)
			c := newCatalog(lang)
			walk := func(screen string) {
				h.t.Helper()
				h.rest()
				h.copyWalk(screen, c, h.values())
			}
			walk("empty Convert")
			h.click(h.T("nav.history"))
			walk("empty History")
			h.click(h.T("nav.settings"))
			walk("Settings")
			h.click(h.T("nav.convert"))
			h.click(h.T("convert.url.label"))
			h.tt.Type("nope")
			h.tt.Key(0, ui.KeyEnter)
			walk("invalid address")

			h.convert(h.inHome(pngNoText))
			h.convert(h.inHome(pdfWarnings))
			walk("failed item")
			h.click(h.rowLabel("policy-assets.pdf"))
			walk("Done with warnings")
			h.click(h.a.lang.P("state.warnings", len(h.item("policy-assets.pdf").State.(model.Done).Result.Warnings)))
			h.click(h.T("detail.markdown"))
			walk("Markdown, warnings open")
			h.click(h.T("detail.copy"))
			walk("copied")
			h.a.addPaths([]string{h.home})
			h.settle()
			walk("folder notice")
			h.startConverting(h.inHome(pdfTable), h.write("a.md", "# A\n"), h.write("b.md", "# B\n"))
			walk("converting and queued")
			h.a.mayQuit()
			walk("quit alert")
			h.tt.Key(0, ui.KeyEscape)
			h.until("conversions", func() bool { return model.Count[model.Converting](&h.a.queue) == 0 })

			h.click(h.T("nav.history"))
			h.click(h.entryLabel("policy-assets.pdf"))
			h.until("the entry", func() bool { return h.has(h.T("detail.preview")) })
			walk("History with detail")
			h.click(h.T("history.search"))
			h.tt.Type("zzz")
			walk("no matches")
			h.tt.Key(ui.Cmd, ui.KeyA)
			h.tt.Key(0, ui.KeyBackspace)
			h.click(h.T("history.clear"))
			walk("clear alert")
			h.tt.Key(0, ui.KeyEscape)

			h.tt.SetSize(800, 780)
			walk("stacked History")
			h.click(h.T("nav.convert"))
			h.click(h.rowLabel("policy-assets.pdf"))
			walk("stacked detail")
			h.tt.SetSize(640, 480)
			walk("narrow window")

			md := h.entry("a.md").MarkdownPath
			os.Remove(md)
			h.tt.SetSize(1200, 780)
			h.click(h.T("nav.history"))
			h.click(h.entryLabel("a.md"))
			h.until("the missing file", func() bool { return h.has(h.T("detail.missing.title")) })
			walk("missing Markdown")
		})
	}
}

func TestFatalScreen(t *testing.T) {
	for _, lang := range []l10n.Lang{l10n.En, l10n.ZhHans} {
		d := &fakeDesk{}
		a := &app{lang: lang, desk: d, goos: "darwin", fatal: true}
		tt := ui.NewTester(a.view, 1200, 780)
		for _, s := range []string{lang.T("fatal.abi.title"), lang.T("fatal.abi.body"), lang.T("quit.confirm")} {
			if !tt.HasText(s) {
				t.Errorf("%s: no %q in %q", lang, s, tt.Texts())
			}
		}
		if _, ok := tt.Find(lang.T("nav.convert")); ok {
			t.Errorf("the fatal screen shows the sidebar: %q", tt.Texts())
		}
		tt.Key(ui.Cmd, ui.Key2)
		tt.Click(lang.T("quit.confirm"))
		if d.quits != 1 {
			t.Errorf("quits %d", d.quits)
		}
	}
}

func TestAccessibility(t *testing.T) {
	h := start(t, l10n.En)
	for _, name := range []string{h.T("a11y.drop_zone"), h.T("convert.url.label"), h.T("convert.choose")} {
		h.find(name)
	}
	h.convert(h.inHome(pdfWarnings))
	h.click(h.rowLabel("policy-assets.pdf"))
	for _, name := range []string{
		h.T("a11y.conversions"), h.T("a11y.drop_zone"), h.T("convert.clear_finished"), h.T("a11y.actions"),
		h.T("detail.view"), h.T("detail.copy"), h.T("detail.save_as"), h.a.showLabel(), h.T("a11y.warnings"),
		h.T("a11y.preview"), h.rowLabel("policy-assets.pdf"),
	} {
		h.find(name)
	}
	h.click(h.T("detail.markdown"))
	h.find(h.T("a11y.markdown_text"))
	if !h.announced(h.T("a11y.done", "title", "policy-assets.pdf")) {
		t.Errorf("announcements %q", h.tt.Announcements())
	}
	h.click(h.T("nav.history"))
	h.rest()
	h.find(h.T("nav.history"))
	h.find(h.T("history.search"))
	h.find(h.T("history.clear"))
	h.find(h.entryLabel("policy-assets.pdf"))
	if !h.announced(h.T("nav.history")) {
		t.Errorf("the page change was not announced: %q", h.tt.Announcements())
	}
}

func TestImages(t *testing.T) {
	h := start(t, l10n.En)
	dir := filepath.Join(h.home, "Downloads")
	os.MkdirAll(dir, 0o755)
	md := h.write("pictures.md", "# Pictures\n\n![remote](https://example.com/remote.png)\n")
	h.convert(md)
	done := h.item("pictures.md").State.(model.Done)
	h.until("the preview", func() bool { return len(h.pane.html) > 0 })
	if got := h.pane.base[0]; got != folderURL(done.Folder) {
		t.Errorf("base %q, want the result folder %q", got, folderURL(done.Folder))
	}
	if !strings.Contains(h.pane.html[0], "img-remote") {
		t.Errorf("a remote image was not left out: %.300s", h.pane.html[0])
	}
	h.tt.SetDark(true)
	h.settle()
	h.until("the dark preview", func() bool { return len(h.pane.html) > 1 })
	if !strings.Contains(h.pane.html[1], `data-theme="dark"`) {
		t.Error("the preview did not follow the dark appearance")
	}
	// The page paints the detail's background itself (DESIGN 8.1).
	for i, want := range []string{"#ffffff", "#18181b"} {
		page := h.pane.html[i]
		style := "<style>html { background: " + want + "; }</style>\n</head>"
		if !strings.Contains(page, style) || strings.Index(page, style) > strings.Index(page, "<body>") {
			t.Errorf("document %d does not end its head with the %s background: %.400s", i, want, page)
		}
	}
}

// TestLinuxPreviewComesBack: on Linux the web view comes back into the
// content one DIP short for a frame (DESIGN 14). The Tester runs the frame
// that asks for and the next, so this checks the end: the full frame, still
// built.
func TestLinuxPreviewComesBack(t *testing.T) {
	h := start(t, l10n.En)
	h.a.goos = "linux"
	h.convert(h.write("notes.md", "# Notes\n\nHello.\n"))
	h.until("the preview", func() bool { return len(h.pane.html) > 0 })
	h.settle()
	full := h.find(h.T("a11y.preview"))
	h.click(h.T("detail.markdown"))
	if h.a.previews.placed {
		t.Fatal("the Markdown tab still builds the web view")
	}
	h.click(h.T("detail.preview"))
	if got := h.find(h.T("a11y.preview")); got != full {
		t.Errorf("the preview is at %+v after it came back, want %+v", got, full)
	}
	if !h.a.previews.placed || !h.a.previews.wasPlaced {
		t.Error("the frames after it came back do not build the web view")
	}
}
