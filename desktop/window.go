package main

import (
	"errors"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"unicode/utf8"

	"github.com/egoist/mygo"
)

// A page reaches Go through one channel: window.open of a markitai-desktop:
// URL, or a target="_blank" link to one, which SetWindowOpenHandler sees.
// The app binds no Go service and trusts no origin.
const (
	openPrint   = "markitai-desktop:print"
	openRetry   = "markitai-desktop:retry"
	openStopNow = "markitai-desktop:stop-now"
	openShowLog = "markitai-desktop:show-log"
)

// preload replaces window.print, which the web view ignores (it returns at
// once, with no panel and no afterprint), so the workbench's PDF export
// reaches exportPDF.
const preload = `window.print = () => { window.open("markitai-desktop:print"); };`

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

// route classifies a window.open URL. Setting a handler replaces mygo's
// default, which opened http(s) in the browser, so route keeps that.
func route(raw string) request {
	if r, ok := requests[raw]; ok {
		return r
	}
	if u, err := url.Parse(raw); err == nil && (u.Scheme == "http" || u.Scheme == "https") && u.Host != "" {
		return reqExternal
	}
	return reqDenied
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
}

func openWindow(sc *Sidecar, t Texts, logPath, downloads string) *window {
	w := &window{sc: sc, texts: t, logPath: logPath, downloads: downloads, phases: make(chan Phase, 1)}
	w.w = mygo.NewWindow(mygo.WindowOptions{
		Title:           "Markitai",
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
			_ = w.w.Page().LoadURL(pg.URL)
		case pg.HTML != "":
			w.w.Page().LoadHTML(pg.HTML, "")
		}
	}
}

// open answers every window.open; no request opens a window.
func (w *window) open(req mygo.WindowOpenRequest) *mygo.WindowOptions {
	r := route(req.URL)
	if r != reqExternal && r != reqDenied && !w.ownPage() {
		return nil
	}
	switch r {
	case reqExternal:
		go func() { _ = mygo.Shell.OpenExternal(req.URL) }()
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

// ownPage reports whether the window shows one of the app's pages: a status
// page or the workbench of our ready child. A page the workbench navigated
// to elsewhere asks nothing of the app.
func (w *window) ownPage() bool {
	at := w.w.Page().URL()
	if at == "about:blank" {
		return true
	}
	r, ok := w.sc.Phase().(Ready)
	return ok && strings.HasPrefix(at, r.Child.origin()+"/")
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
		return "Markitai"
	}
	return name
}
