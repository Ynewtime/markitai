package main

import (
	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"
	"log"
	"runtime"
)

// systemDesk is the desktop through mygo. Its dialogs block, so they run
// on goroutines of their own and answer through post.
type systemDesk struct {
	win  func() *mygo.Window
	post func(func())
	// chooseTitle titles the open dialog: convert.choose (6.2).
	chooseTitle string
}

// A dialog that fails to show is one cancelled, as far as the app goes: the
// log says why.

func (d systemDesk) chooseFiles(done func([]string)) {
	go func() {
		paths, err := mygo.Dialog.Open(mygo.OpenDialogOptions{Parent: d.win(), Title: d.chooseTitle, Multiple: true})
		if err != nil {
			log.Printf("choose files: %v", err)
		}
		if err == nil && len(paths) > 0 {
			d.post(func() { done(paths) })
		}
	}()
}

func (d systemDesk) chooseFolder(start string, done func(string)) {
	go func() {
		paths, err := mygo.Dialog.Open(mygo.OpenDialogOptions{Parent: d.win(), DefaultPath: start, Directory: true, CreateDirectories: true})
		if err != nil {
			log.Printf("choose folder: %v", err)
		}
		if err == nil && len(paths) > 0 {
			d.post(func() { done(paths[0]) })
		}
	}()
}

// saveAs lists Markdown files; a name without an extension takes .md
// from the app, as Windows and GTK leave it (section 6.7).
func (d systemDesk) saveAs(start string, done func(string)) {
	go func() {
		path, err := mygo.Dialog.Save(mygo.SaveDialogOptions{Parent: d.win(), DefaultPath: start,
			Filters: []mygo.FileFilter{{Name: "Markdown", Extensions: []string{"md"}}}})
		if err != nil {
			log.Printf("save as: %v", err)
		}
		if err == nil && path != "" {
			d.post(func() { done(path) })
		}
	}()
}

func (systemDesk) quit() { mygo.App.Quit() }

func (d systemDesk) showWindow() {
	if w := d.win(); w != nil {
		w.Show()
		w.Focus()
	}
}

func (d systemDesk) closeWindow() {
	if w := d.win(); w != nil {
		w.Close()
	}
}

// webPreview is the preview pane on the window's web view. base is the
// document's base URL, which navigation compares against.
type webPreview struct {
	view *mygo.WebView
	base string
}

func (p *webPreview) View() ui.NativeWebView { return p.view }

// Load loads html. On macOS WebKit lets the page read the folder's
// images only after a load with read access to it: loadHTMLString's base
// URL grants none, loadFileURL grants the file's folder, and a later
// loadHTMLString whose base URL is inside it keeps that access. So the
// page first loads the Markdown file beside them, which the view never
// shows: it builds the web view only once done runs.
func (p *webPreview) Load(html, baseURL, inFolder string, done func()) {
	p.base = baseURL
	page := p.view.Page()
	show := func() {
		var off func()
		off = page.OnDidFinishLoad(func() {
			off()
			done()
		})
		page.LoadHTML(html, baseURL)
	}
	if runtime.GOOS != "darwin" || inFolder == "" {
		show()
		return
	}
	var granted func()
	granted = page.OnDidFinishLoad(func() {
		granted()
		show()
	})
	if err := page.LoadFile(inFolder); err != nil {
		granted()
		show()
	}
}

func (p *webPreview) SetZoom(f float64) { p.view.Page().SetZoomFactor(f) }

func (p *webPreview) SetBackground(hex string) {
	if err := p.view.SetBackgroundColor(hex); err != nil {
		log.Printf("preview background %s: %v", hex, err)
	}
}
