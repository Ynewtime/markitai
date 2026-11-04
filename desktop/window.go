package main

import (
	"encoding/json"
	"errors"
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

// A page reaches Go through one channel: window.open of a markitai-desktop:
// URL, or a target="_blank" link to one, which SetWindowOpenHandler sees
// without saying which frame asked. The app binds no Go service; it honors
// a request only in a phase whose page offers it (offers in view.go).
const (
	openRetry   = "markitai-desktop:retry"
	openStopNow = "markitai-desktop:stop-now"
	openShowLog = "markitai-desktop:show-log"
)

// openPrint is Export PDF's request, offered while the workbench shows,
// beside converted content: a link in that content reaches window.open too
// (the snapshot frame lets pop-ups out of its sandbox). It carries a key of
// this launch that only the preload knows, which runs in the main frame
// alone and before the workbench's scripts.
var openPrint = "markitai-desktop:print?" + string(newToken())

// preload replaces window.print, which the web view ignores (it returns at
// once, with no panel and no afterprint), so the workbench's PDF export
// reaches exportPDF.
var preload = `window.print = () => { window.open("` + openPrint + `"); };`

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
	texts     Texts
	logPath   string
	downloads string

	mu     sync.Mutex
	phases chan Phase // the phase to show next; the newest replaces an unshown one
	kept   string     // the origin whose documents the content rules keep; paint's alone
}

func openWindow(sc *Sidecar, t Texts, logPath, downloads string) *window {
	w := &window{sc: sc, texts: t, logPath: logPath, downloads: downloads, phases: make(chan Phase, 1)}
	w.w = mygo.NewWindow(mygo.WindowOptions{
		Title:           "Markitai Web",
		Width:           1280,
		Height:          860,
		MinWidth:        480,
		MinHeight:       400,
		BackgroundColor: "light-dark(#fafafa, #09090b)",
		StateKey:        "main",
		Page:            mygo.PageOptions{PreloadScript: preload},
	})
	page := w.w.Page()
	page.SetWindowOpenHandler(w.open)
	page.OnWillNavigate(w.willNavigate)
	page.OnWillDownload(w.willDownload)
	page.OnDownloadDone(w.downloadDone)
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
		switch pg := render(p, w.texts); {
		case pg.URL != "":
			w.keepDocuments(origin(pg.URL))
			_ = w.w.Page().LoadURL(pg.URL)
		case pg.HTML != "":
			w.w.Page().LoadHTML(pg.HTML, "")
		}
	}
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
		mygo.Shell.ShowItemInFolder(w.logPath)
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

// exportPDF is the workbench's Export PDF: A4 with the workbench's margins
// and backgrounds, afterprint at once so the workbench restores its page,
// then a Save panel named after the title the workbench set for printing.
func (w *window) exportPDF() {
	page := w.w.Page()
	title, _ := mygo.EvalAs[string](page, "document.title")
	pdf, err := page.PrintToPDF(mygo.PDFOptions{PageSize: mygo.PageA4, Margins: &a4Margins, Background: true})
	_, _ = page.Eval(`dispatchEvent(new Event("afterprint"))`)
	if err == nil {
		path := saveAs(w.w, filepath.Join(w.downloads, fileName(title)+".pdf"))
		if path == "" {
			return
		}
		err = os.WriteFile(path, pdf, 0o644)
	}
	if err != nil {
		w.warn(w.texts.SaveFailed, err)
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
		go w.warn(w.texts.DownloadFailed, d.Err)
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

// openExternal hands url to the system's handler for its scheme. A
// variable so the e2e build can record it instead.
var openExternal = func(url string) { _ = mygo.Shell.OpenExternal(url) }

// saveAs asks where to save, starting at suggested; "" when canceled. A
// variable so the e2e build can answer without a panel.
var saveAs = func(parent *mygo.Window, suggested string) string {
	path, err := mygo.Dialog.Save(mygo.SaveDialogOptions{Parent: parent, DefaultPath: suggested, CreateDirectories: true})
	if err != nil {
		return ""
	}
	return path
}

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
