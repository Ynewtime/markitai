package main

import (
	"encoding/json"
	"errors"
	"log"
	"maps"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"slices"
	"strings"
	"sync"
	"unicode/utf8"

	"github.com/egoist/mygo"
)

// A status page's buttons reach Go as window.open of a markitai-web: URL, a
// target="_blank" link to one, which SetWindowOpenHandler sees without
// saying which frame asked; so does Export PDF (openPrint). The app honors
// such a request only in a phase whose page offers it (offers in view.go).
// What the preload says of the page goes through the one service the app
// binds, MarkitaiWeb (pages.go), which mygo's bridge offers the main frame
// alone.
const (
	openRetry   = "markitai-web:retry"
	openStopNow = "markitai-web:stop-now"
	openShowLog = "markitai-web:show-log"
)

// openPrint is Export PDF's request, offered while the workbench shows,
// beside converted content: a link in that content reaches window.open too
// (the snapshot frame lets pop-ups out of its sandbox). It carries a key of
// this launch that only the preload knows, which replaces window.print in
// the main frame alone and before the workbench's scripts.
var openPrint = "markitai-web:print?" + string(newToken())

// request is what a window.open asks of the app.
type request int

const (
	reqDenied request = iota
	reqExternal
	reqPrint
	reqRetry
	reqStopNow
	reqShowLog
)

var requests = map[string]request{
	openPrint:   reqPrint,
	openRetry:   reqRetry,
	openStopNow: reqStopNow,
	openShowLog: reqShowLog,
}

// route classifies a URL a page opens or navigates to. Setting a handler
// replaces mygo's default, which opened http(s) in the browser, so route
// keeps that, and hands mailto to the mail app.
func route(raw string) request {
	if r, ok := requests[raw]; ok {
		return r
	}
	u, err := url.Parse(raw)
	switch {
	case err != nil:
		return reqDenied
	case (u.Scheme == "http" || u.Scheme == "https") && u.Host != "":
		return reqExternal
	case u.Scheme == "mailto" && u.Opaque != "":
		return reqExternal
	}
	return reqDenied
}

// stays reports whether the main frame may go to raw in phase p: to the
// page render shows for p, within the workbench of the ready child, or a
// status page where paint loads it (atStatusPage). mygo asks about the app's
// own loads too when two overlap, as when serve fails at once and the
// Failed page follows the Starting page: it marks only the first as the
// app's.
func stays(p Phase, raw string) bool {
	if r, ok := p.(Ready); ok {
		return origin(raw) == r.Child.origin()
	}
	return atStatusPage(runtime.GOOS, raw)
}

// atStatusPage reports whether raw is where a status page loads on goos:
// paint gives the web view its HTML with no base URL, at about:blank,
// which WebView2 asks about as the data: URL it makes of the HTML. No page
// can take the window to a data: URL itself there: Chromium refuses a
// page's top-level navigation to one.
func atStatusPage(goos, raw string) bool {
	return raw == "about:blank" || goos == "windows" && strings.HasPrefix(raw, "data:text/html")
}

// origin is raw's scheme://host, that of the page that made it for a blob:
// URL.
func origin(raw string) string {
	u, err := url.Parse(raw)
	if err == nil && strings.EqualFold(u.Scheme, "blob") {
		u, err = url.Parse(u.Opaque)
	}
	if err != nil {
		return ""
	}
	return strings.ToLower(u.Scheme + "://" + u.Host)
}

// keepsDocuments is whether the window also keeps its document loads by
// content rules (documentRules). WebView2 may send a navigation's GET while
// the app is still deciding whether it goes, as the documentation of
// NavigationStarting's Cancel says: willNavigate keeps the window where it
// is, but the other site is asked all the same. Content rules match every
// request before it is sent, so a document elsewhere is never asked for. The
// web views of macOS and Linux ask before they request.
var keepsDocuments = runtime.GOOS == "windows"

// documentRules are content rules that let documents load from origin
// alone: the first blocks every document, the second undoes that for
// origin. Documents the app gives the window itself, at about: and data:
// URLs, never reach content rules.
func documentRules(origin string) string {
	rules, _ := json.Marshal([]map[string]map[string]any{
		{"trigger": {"url-filter": ".*", "resource-type": []string{"document"}}, "action": {"type": "block"}},
		{"trigger": {"url-filter": "^" + regexp.QuoteMeta(origin) + "/", "resource-type": []string{"document"}},
			"action": {"type": "ignore-previous-rules"}},
	})
	return string(rules)
}

// a4Margins are the workbench's @page margins, 14 mm by 12 mm, in inches.
var a4Margins = mygo.Margins{Top: 14 / 25.4, Right: 12 / 25.4, Bottom: 14 / 25.4, Left: 12 / 25.4}

// window is the app's only window; closing it quits the app (main.go).
type window struct {
	w         *mygo.Window
	sc        *Sidecar
	mem       *memory
	logPath   string
	downloads string

	mu      sync.Mutex
	phases  chan Phase // the phase to show next; the newest replaces an unshown one
	paints  int        // the status pages painted so far
	latest  sight      // the page painted last
	settled bool       // the main frame said it shows latest, and nothing was painted since
	kept    string     // the origin whose documents the content rules keep; paint's alone

	// turn takes the preload's calls one at a time, which mygo makes
	// each in a goroutine of its own: Look and Key.
	turn   sync.Mutex
	looks  reports    // the workbench's look reported last
	wheels wheelCount // Ctrl with the wheel's notches not yet a step
}

// sight is a page the window paints: a status page by its paint, numbered
// from 1, or the workbench of a child by its token.
type sight struct {
	Paint int
	Token Token
}

// background is the web view's own color, the workbench's background in
// its light and dark themes, which shows before a page draws.
const background = "light-dark(#fafafa, #09090b)"

// minWidth and minHeight are the window's smallest size: the workbench lays
// itself out for a phone from 480 wide down, and below 300 high a tiling
// window manager's tile would leave too little of the page. A tile smaller
// than the window gets its content squeezed, as Hyprland does.
const minWidth, minHeight = 480, 300

func openWindow(sc *Sidecar, mem *memory, logPath, downloads string) *window {
	w := &window{sc: sc, mem: mem, logPath: logPath, downloads: downloads, phases: make(chan Phase, 1)}
	mygo.Bind(&MarkitaiWeb{w: w})
	w.w = mygo.NewWindow(mygo.WindowOptions{
		Title:           "Markitai Web",
		Width:           1280,
		Height:          860,
		MinWidth:        minWidth,
		MinHeight:       minHeight,
		BackgroundColor: background,
		StateKey:        "main",
		// The main frame stays on the app's pages (willNavigate): every
		// origin it shows may call MarkitaiWeb, the workbench's of a port
		// chosen after the window opens among them.
		Page: mygo.PageOptions{PreloadScript: preload, PreloadAllFrames: true, TrustedOrigins: []string{"*"}},
	})
	page := w.w.Page()
	// Trusted pages get every permission they ask for by default; the
	// workbench asks for none, and gets none, as before it was trusted.
	page.SetPermissionHandler(func(mygo.PermissionRequest) bool { return false })
	page.SetWindowOpenHandler(w.open)
	page.OnWillNavigate(w.willNavigate)
	page.OnWillDownload(w.willDownload)
	page.OnDownloadDone(w.downloadDone)
	page.OnContextMenu(contextMenu)
	page.OnPageTitleUpdated(func(e *mygo.TitleEvent) {
		e.PreventDefault()
		w.w.SetTitle(windowTitle(e.Title))
	})
	page.OnRenderProcessGone(func(string) { w.Show(sc.Phase()) })
	go w.paint()
	return w
}

// Show displays p. It never waits: the sidecar calls it while the main
// thread may be waiting for the sidecar to stop.
func (w *window) Show(p Phase) {
	w.mu.Lock()
	defer w.mu.Unlock()
	select {
	case <-w.phases:
	default:
	}
	w.phases <- p
}

func (w *window) paint() {
	for p := range w.phases {
		w.mu.Lock()
		paint := w.paints + 1
		w.mu.Unlock()
		pg := render(p, shownTexts(), paint)
		var s sight
		switch {
		case pg.URL != "":
			s = sight{Token: p.(Ready).Child.Token}
		case pg.HTML != "":
			s = sight{Paint: paint}
		default:
			continue
		}
		w.mu.Lock()
		w.paints = max(w.paints, s.Paint)
		w.latest, w.settled = s, false
		w.mu.Unlock()
		if pg.URL != "" {
			w.keepDocuments(origin(pg.URL))
			_ = w.w.Page().LoadURL(pg.URL)
		} else {
			w.w.Page().LoadHTML(pg.HTML, "")
		}
	}
}

// shown is what the main frame says it shows, as each page shows, also as
// the history brings back one it showed before: mygo loads the history's
// pages unasked (willNavigate), as with the mouse's back button or a
// context menu's Back where the web view has them. A page other than the
// one painted last, a status page whose buttons the phase no longer
// honors or a workbench whose server is gone, is painted over with the
// phase's. Until the latest page says it shows, what the others say is
// the window's own loads overtaking each other, and goes unanswered.
func (w *window) shown(s sight) {
	w.mu.Lock()
	current := s == w.latest
	stale := w.settled && !current
	w.settled = current
	w.mu.Unlock()
	if stale {
		w.Show(w.sc.Phase())
	}
}

// look takes the workbench's language and theme (Look), when they are the
// latest reported. The theme is kept for the next launch after the turn,
// which the shortcuts share: saving can take seconds on Windows (replace).
func (w *window) look(doc string, n int, l Look) {
	w.turn.Lock()
	if !w.looks.newer(doc, n) {
		w.turn.Unlock()
		return
	}
	if l.Lang != "" {
		if workbenchLang(mygo.App.Locale()) != l.Lang {
			mygo.App.SetLocale(locale(l.Lang))
		}
		chose.Store(true)
	}
	if source := themeSource(l.Theme); mygo.Theme.Source() != source {
		mygo.Theme.SetSource(source)
	}
	w.mem.Set(func(m *Memory) { m.Theme = l.Theme })
	w.turn.Unlock()
	if err := w.mem.Save(); err != nil {
		log.Printf("markitai web: %v", err)
	}
}

// key does what a shortcut keys lists for this system does.
func (w *window) key(k key) {
	if !slices.Contains(slices.Collect(maps.Values(keys(runtime.GOOS))), k) {
		return
	}
	w.turn.Lock()
	defer w.turn.Unlock()
	page := w.w.Page()
	switch k {
	case keyZoomIn, keyZoomOut:
		page.SetZoomFactor(nextZoom(page.ZoomFactor(), k == keyZoomIn))
	case keyZoomReset:
		page.SetZoomFactor(1)
	case keyFullScreen:
		w.w.SetFullScreen(!w.w.IsFullScreen())
	case keyQuit:
		mygo.App.Quit()
	}
}

// wheel zooms a step for each notch of Ctrl with the wheel (wheelCount),
// wheelDeltaY of the page, where the preload takes keys.
func (w *window) wheel(delta float64) {
	if len(keys(runtime.GOOS)) == 0 {
		return
	}
	w.turn.Lock()
	defer w.turn.Unlock()
	page := w.w.Page()
	steps := w.wheels.add(runtime.GOOS, notches(runtime.GOOS, delta, page.ZoomFactor()))
	for ; steps != 0; steps -= sign(steps) {
		page.SetZoomFactor(nextZoom(page.ZoomFactor(), steps > 0))
	}
}

// sign is 1 for a positive n, -1 for a negative one.
func sign(n int) int {
	if n < 0 {
		return -1
	}
	return 1
}

// keepDocuments applies documentRules for o where keepsDocuments, before the
// window loads the workbench there. The rules stay through the status pages
// that follow, which they never see. If they cannot be compiled, willNavigate
// alone keeps the window.
func (w *window) keepDocuments(o string) {
	if !keepsDocuments || o == w.kept {
		return
	}
	rules, err := mygo.CompileContentRules("markitai-documents", documentRules(o))
	if err != nil {
		return
	}
	w.w.Page().SetContentRules(rules)
	w.kept = o
}

// open answers every window.open; no request opens a window. A link the
// user followed opens in the browser or mail app; one a script opened by
// itself, as a pop-up, opens nowhere.
func (w *window) open(req mygo.WindowOpenRequest) *mygo.WindowOptions {
	r := route(req.URL)
	if r == reqExternal {
		if req.UserInitiated {
			go openExternal(req.URL)
		}
		return nil
	}
	if !slices.Contains(offers(w.sc.Phase()), r) {
		return nil
	}
	switch r {
	case reqPrint:
		go w.exportPDF()
	case reqRetry:
		w.sc.Retry()
	case reqStopNow:
		w.sc.Stop(Now)
	case reqShowLog:
		go showInFolder(w.logPath)
	}
	return nil
}

// willNavigate keeps the main frame on the app's pages. Elsewhere, as a link
// without a target or a URL dropped on the window would take it, it does
// not go: what the user followed opens in the browser or mail app instead.
func (w *window) willNavigate(e *mygo.NavigateEvent) {
	if stays(w.sc.Phase(), e.URL) {
		return
	}
	e.PreventDefault()
	if e.UserInitiated && route(e.URL) == reqExternal {
		go openExternal(e.URL)
	}
}

// editItems are the web view's items a text field's context menu keeps.
var editItems = []string{mygo.ContextMenuUndo, mygo.ContextMenuRedo, mygo.ContextMenuCut, mygo.ContextMenuCopy,
	mygo.ContextMenuPaste, mygo.ContextMenuPasteAndMatchStyle, mygo.ContextMenuDelete, mygo.ContextMenuSelectAll,
	mygo.ContextMenuSpellingGuess, mygo.ContextMenuEmoji}

// contextMenu keeps the web view's context menu to what an app's text has:
// a text field's editing items, Copy over a selection,
// and no menu elsewhere. A browser's items, as Back, Reload, Print, Save As
// or Share, never show; the items it keeps say what they do in the app's
// language.
func contextMenu(e *mygo.ContextMenuEvent) {
	if e.Menu == nil {
		return
	}
	labels := shownTexts().MenuItems
	shows := false
	for _, it := range e.Menu.Items() {
		switch {
		case it.Type == mygo.MenuItemSeparator:
		case menuKeeps(e.Editable, e.SelectionText != "", it.ID):
			shows = true
			if label := labels[it.ID]; label != "" {
				it.SetLabel(label)
			}
		default:
			it.SetVisible(false)
		}
	}
	if !shows {
		e.PreventDefault()
	}
}

// menuKeeps is whether the context menu keeps the web view's item id: a
// text field's keeps its editing items, a selection Copy alone, anything
// else none.
func menuKeeps(editable, selection bool, id string) bool {
	if editable {
		return slices.Contains(editItems, id)
	}
	return selection && id == mygo.ContextMenuCopy
}

// windowTitle names the app in the page's title, as the taskbar, the Dock
// and the window switcher show it: the workbench's "… · Markitai" ends in
// "Markitai Web".
func windowTitle(title string) string {
	switch {
	case strings.TrimSpace(title) == "":
		return "Markitai Web"
	case strings.HasSuffix(title, "Markitai Web"):
		return title
	case strings.HasSuffix(title, "Markitai"):
		return title + " Web"
	}
	return title + " · Markitai Web"
}

// exportPDF is the workbench's Export PDF: A4 with the workbench's margins
// and backgrounds, afterprint at once so the workbench restores its page,
// then a Save panel named after the title the workbench set for printing.
// The page is white to its edges, as the workbench's print styles promise:
// WebView2 paints the margins, which no document draws, in the dark
// canvas color under a dark color-scheme, else in the web view's own
// background, so both are light while it prints.
func (w *window) exportPDF() {
	page := w.w.Page()
	title, _ := mygo.EvalAs[string](page, "document.title")
	_ = w.w.SetBackgroundColor("#ffffff")
	_, _ = page.Eval(`document.documentElement.style.setProperty("color-scheme", "light", "important")`)
	pdf, err := page.PrintToPDF(mygo.PDFOptions{PageSize: mygo.PageA4, Margins: &a4Margins, Background: true})
	_, _ = page.Eval(`document.documentElement.style.removeProperty("color-scheme"); dispatchEvent(new Event("afterprint"))`)
	_ = w.w.SetBackgroundColor(background)
	if err == nil {
		path := saveAs(w.w, filepath.Join(w.downloads, fileName(title)+".pdf"))
		if path == "" {
			return
		}
		err = os.WriteFile(path, pdf, 0o644)
	}
	if err != nil {
		w.warn(shownTexts().SaveFailed, err)
	}
}

// willDownload asks where to save, as a browser set to ask does: the
// workbench's downloads are exports the user just chose. e.Path is the
// default, a numbered name in Downloads.
func (w *window) willDownload(e *mygo.DownloadEvent) {
	path := saveAs(w.w, e.Path)
	if path == "" {
		e.PreventDefault()
		return
	}
	e.Path = path
}

func (w *window) downloadDone(d *mygo.Download) {
	if d.Err != nil && !errors.Is(d.Err, mygo.ErrCanceled) {
		go w.warn(shownTexts().DownloadFailed, d.Err)
	}
}

func (w *window) warn(message string, err error) {
	_, _ = mygo.Dialog.Message(mygo.MessageOptions{Parent: w.w, Type: mygo.MessageWarning, Message: message,
		Detail: err.Error()})
}

func (w *window) focus() {
	w.w.Restore()
	w.w.Show()
}

// openExternal hands url to the system's handler for its scheme, and logs
// why it could not (not the link, which may carry a secret). A variable so
// the e2e build can record it instead.
var openExternal = func(url string) {
	if err := openURL(url); err != nil {
		log.Printf("markitai web: open a link: %v", err)
	}
}

// saveAs asks where to save, starting at suggested; "" when canceled. A
// variable so the e2e build can answer without a panel.
var saveAs = func(parent *mygo.Window, suggested string) string {
	return askSavePath(suggested, func(at string) (string, error) {
		return mygo.Dialog.Save(mygo.SaveDialogOptions{Parent: parent, DefaultPath: at, CreateDirectories: true,
			Filters: saveFilters(suggested)})
	}, func(path string) bool {
		t := shownTexts()
		r, err := mygo.Dialog.Message(mygo.MessageOptions{Parent: parent, Type: mygo.MessageWarning,
			Message: t.Exists(filepath.Base(path)), Detail: t.ExistsDetail, Buttons: []string{t.Replace, t.Cancel},
			DefaultButton: 1, CancelButton: 1, DestructiveButtons: []int{0}})
		return err == nil && r.Button == 0
	})
}

// askSavePath is the path the user chose with ask, the panel, starting at
// suggested; "" when canceled. A name typed without an extension of
// suggested's format (formatExts) gets suggested's, as the panel of macOS
// gives it, also one whose last dot is part of the name ("notes 1.5", or
// "page.html", as Explorer shows "page.html.md"): one that names a file
// already there is asked about with replace, the panel having asked about
// the name as typed, and the panel comes back at it unless replaced.
func askSavePath(suggested string, ask func(at string) (string, error), replace func(path string) bool) string {
	ext := filepath.Ext(suggested)
	for at := suggested; ; {
		path, err := ask(at)
		if err != nil || path == "" {
			return ""
		}
		if ext == "" || slices.Contains(formatExts(ext), strings.ToLower(filepath.Ext(path))) {
			return path
		}
		path += ext
		if _, err := os.Lstat(path); err != nil || replace(path) {
			return path
		}
		at = path
	}
}

// formatExts are the extensions, as ".md", of the format whose extension
// ext is, any case.
func formatExts(ext string) []string {
	ext = strings.ToLower(ext)
	if ext == ".md" {
		return []string{".md", ".markdown"}
	}
	return []string{ext}
}

// saveFilters is the panel's file type for suggested's extension, which
// names its format, the extension in its name, as "Markdown (*.md)", as
// Explorer hides it; none without one.
func saveFilters(suggested string) []mygo.FileFilter {
	ext := strings.ToLower(strings.TrimPrefix(filepath.Ext(suggested), "."))
	if ext == "" {
		return nil
	}
	name, ok := formats[ext]
	if !ok {
		name = strings.ToUpper(ext)
	}
	return []mygo.FileFilter{{Name: name + " (*." + ext + ")", Extensions: []string{ext}}}
}

// formats name the formats the workbench saves, by extension.
var formats = map[string]string{"pdf": "PDF", "md": "Markdown", "zip": "ZIP", "html": "HTML", "json": "JSON",
	"txt": "Text"}

// fileName makes a page title a file name on every system: no separators or
// reserved characters, and room for an extension within 255 bytes.
func fileName(title string) string {
	name := strings.Map(func(r rune) rune {
		if r < 0x20 || r == 0x7f || strings.ContainsRune(`<>:"/\|?*`, r) {
			return '-'
		}
		return r
	}, title)
	for len(name) > 200 {
		_, size := utf8.DecodeLastRuneInString(name)
		name = name[:len(name)-size]
	}
	name = strings.Trim(name, " .")
	if name == "" {
		return "Markitai Web"
	}
	return name
}
