package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/egoist/mygo/ui"

	"markitai.local/mygo/internal/core"
	"markitai.local/mygo/internal/l10n"
	"markitai.local/mygo/internal/model"
	"markitai.local/mygo/internal/testenv"
)

// Repository fixtures the tests convert.
const (
	pdfWarnings = "crates/markitai-core/src/formats/pdf/fixtures/sanitize/policy-assets.pdf"
	pdfTable    = "crates/markitai-core/src/formats/pdf/fixtures/default-table.pdf"
	pngNoText   = "crates/markitai-core/src/ocr/fixtures/english.png"
)

// fakeDesk records what the app asked of the desktop, and answers its
// dialogs later, as the system's do.
type fakeDesk struct {
	post     func(func())
	files    []string
	folder   string
	saveTo   string
	shown    []string
	opened   []string
	external []string
	openErr  error
	suggests []string
	quits    int
	closes   int
	shows    int
}

func (d *fakeDesk) chooseFiles(done func([]string)) {
	if paths := d.files; len(paths) > 0 {
		d.post(func() { done(paths) })
	}
}

func (d *fakeDesk) chooseFolder(done func(string)) {
	if dir := d.folder; dir != "" {
		d.post(func() { done(dir) })
	}
}

func (d *fakeDesk) saveAs(name string, done func(string)) {
	d.suggests = append(d.suggests, name)
	if path := d.saveTo; path != "" {
		d.post(func() { done(path) })
	}
}

func (d *fakeDesk) showItemInFolder(path string) { d.shown = append(d.shown, path) }
func (d *fakeDesk) openPath(path string) error {
	d.opened = append(d.opened, path)
	return d.openErr
}
func (d *fakeDesk) openExternal(url string) error {
	d.external = append(d.external, url)
	return nil
}
func (d *fakeDesk) showWindow()  { d.shows++ }
func (d *fakeDesk) closeWindow() { d.closes++ }
func (d *fakeDesk) quit()        { d.quits++ }

// fakeView stands for the web view: the Tester places it and never calls
// it (section 14).
type fakeView struct{ ui.NativeWebView }

// fakePreview records the documents the app loads.
type fakePreview struct {
	view *fakeView
	html []string
	base []string
	// inFolder is the file each load names in the result folder.
	inFolder []string
	// backgrounds are the colors the web view was filled with, in order.
	backgrounds []string
	zoom        float64
	zooms       int
}

func (p *fakePreview) View() ui.NativeWebView { return p.view }
func (p *fakePreview) Load(html, baseURL, inFolder string, done func()) {
	p.html = append(p.html, html)
	p.base = append(p.base, baseURL)
	p.inFolder = append(p.inFolder, inFolder)
	done()
}
func (p *fakePreview) SetZoom(z float64)        { p.zoom, p.zooms = z, p.zooms+1 }
func (p *fakePreview) SetBackground(hex string) { p.backgrounds = append(p.backgrounds, hex) }

// offline is the real core with the network left out: a web page fails
// to download at once, as with no connection, so no test sends a request.
type offline struct{ *core.Lib }

func (o offline) ConvertJSON(request any) (json.RawMessage, error) {
	if r, ok := request.(model.ConvertRequest); ok && strings.Contains(r.Source, "://") {
		return nil, &core.Error{Code: "fetch_error", Message: "offline test: " + r.Source}
	}
	return o.Lib.ConvertJSON(request)
}

type harness struct {
	t     *testing.T
	a     *app
	tt    *ui.Tester
	posts chan func()
	desk  *fakeDesk
	pane  *fakePreview
	home  string
	lib   *core.Lib
}

// start runs the app's view in a Tester over the real core, in an
// isolated home whose environment the core prepared.
func start(t *testing.T, lang l10n.Lang) *harness {
	t.Helper()
	return startWith(t, lang, nil)
}

// startWith lets prepare write the app's files before it starts.
func startWith(t *testing.T, lang l10n.Lang, prepare func(home string, f files)) *harness {
	t.Helper()
	home := testenv.Isolate(t)
	lib, err := core.Load(testenv.Library(t))
	if err != nil {
		t.Fatal(err)
	}
	data := filepath.Join(home, "data")
	if _, err := lib.PrepareEnvironment(data); err != nil {
		t.Fatal(err)
	}
	f := files{data: data, settings: filepath.Join(data, "settings.json"), history: filepath.Join(data, "history.json"),
		output: filepath.Join(home, "Documents", "Markitai")}
	if prepare != nil {
		prepare(home, f)
	}
	h := &harness{t: t, posts: make(chan func(), 1024), home: home, lib: lib, pane: &fakePreview{view: &fakeView{}}}
	post := func(fn func()) { h.posts <- fn }
	h.desk = &fakeDesk{post: post}
	h.a = newApp(lang, offline{lib}, h.desk, h.pane, post, f)
	h.a.version = "2.3.2"
	h.tt = ui.NewTester(h.a.view, 1200, 780)
	h.tt.SetLocale(string(lang))
	h.tt.Frame()
	return h
}

// until runs the work that comes back to the UI thread, as the window's
// Update does, until ok holds.
func (h *harness) until(what string, ok func() bool) {
	h.t.Helper()
	deadline := time.After(60 * time.Second)
	for !ok() {
		select {
		case f := <-h.posts:
			f()
			h.tt.Frame()
		case <-deadline:
			h.t.Fatalf("timed out waiting for %s; texts %q", what, h.tt.Texts())
		}
	}
}

// settle runs the work already waiting.
func (h *harness) settle() {
	for {
		select {
		case f := <-h.posts:
			f()
			h.tt.Frame()
		default:
			h.tt.Frame()
			return
		}
	}
}

// rest lets the motion of the last change end: the Tester's clock stands
// still until Advance.
func (h *harness) rest() {
	h.tt.Advance(time.Second)
	h.tt.Frame()
}

func (h *harness) find(label string) ui.Rect {
	h.t.Helper()
	r, ok := h.tt.Find(label)
	if !ok {
		h.t.Fatalf("no element %q; texts %q", label, h.tt.Texts())
	}
	return r
}

// scrollToEnd scrolls the page under the element named label to its end,
// as a person scrolls to the controls below.
func (h *harness) scrollToEnd(label string) {
	h.t.Helper()
	r := h.find(label)
	h.tt.Scroll(r.X+r.W/2, r.Y+r.H/2, 0, 10000)
	h.settle()
}

func (h *harness) click(label string) {
	h.t.Helper()
	if err := h.tt.Click(label); err != nil {
		h.t.Fatalf("click %q: %v; texts %q", label, err, h.tt.Texts())
	}
	h.settle()
}

// drop drops paths on the element named label, as a file manager does.
func (h *harness) drop(label string, paths ...string) {
	h.t.Helper()
	r := h.find(label)
	if !h.tt.DropFiles(r.X+r.W/2, r.Y+r.H/2, paths...) {
		h.t.Fatalf("the drop on %q was not taken", label)
	}
	h.settle()
}

func (h *harness) has(s string) bool { return h.tt.HasText(s) }

func (h *harness) wantText(s string) {
	h.t.Helper()
	if !h.has(s) {
		h.t.Errorf("no text %q; texts %q", s, h.tt.Texts())
	}
}

func (h *harness) wantNoText(s string) {
	h.t.Helper()
	if h.has(s) {
		h.t.Errorf("text %q shows", s)
	}
}

func (h *harness) T(key string, args ...any) string { return h.a.lang.T(key, args...) }

// inHome copies a fixture into the isolated home's Downloads folder.
func (h *harness) inHome(rel string) string {
	h.t.Helper()
	data, err := os.ReadFile(testenv.Fixture(h.t, rel))
	if err != nil {
		h.t.Fatal(err)
	}
	dst := filepath.Join(h.home, "Downloads", filepath.Base(rel))
	os.MkdirAll(filepath.Dir(dst), 0o755)
	if err := os.WriteFile(dst, data, 0o644); err != nil {
		h.t.Fatal(err)
	}
	return dst
}

// write makes a file in the isolated home.
func (h *harness) write(name, content string) string {
	h.t.Helper()
	path := filepath.Join(h.home, "Downloads", name)
	os.MkdirAll(filepath.Dir(path), 0o755)
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		h.t.Fatal(err)
	}
	return path
}

func (h *harness) item(title string) *model.Item {
	for _, it := range h.a.queue.Items {
		if it.Title == title {
			return it
		}
	}
	return nil
}

// stateOf waits for the item titled title to reach state S.
func waitState[S model.State](h *harness, title string) S {
	h.t.Helper()
	var s S
	h.until(title+" to change state", func() bool {
		it := h.item(title)
		if it == nil {
			return false
		}
		var ok bool
		s, ok = it.State.(S)
		return ok
	})
	return s
}

// markdownText reads the Markdown tab's text, which Tester.Texts leaves
// out, with Select All and Copy.
func (h *harness) markdownText() string {
	h.t.Helper()
	r := h.find(h.T("a11y.markdown_text"))
	h.tt.ClickAt(r.X+r.W/2, r.Y+r.H/2)
	h.tt.Key(ui.Cmd, ui.KeyA)
	h.tt.Key(ui.Cmd, ui.KeyC)
	return h.tt.Clipboard()
}

func (h *harness) announced(s string) bool {
	for _, a := range h.tt.Announcements() {
		if strings.Contains(a, s) {
			return true
		}
	}
	return false
}
