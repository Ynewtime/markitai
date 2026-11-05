package main

import (
	"context"
	"log"
	"runtime"
	"time"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/egoist/mygo"
)

// drops hands the files of a drop to the workbench. WebKitGTK 2.52 and
// later keep them from the page (DataTransfer::allowsFileAccess, off
// outside Cocoa since WebKit 303828@main; WebKit bugs 320301 and 323277): a
// drag of files reaches it as text/uri-list alone. The app reads their
// paths from the drag's data as WebKit receives it, before the page sees
// the drag. The preload asks how many there are (Dragged) and, when such a
// drag drops on the workbench, for the files (Drop): the app clicks the
// workbench's drop input and answers the file chooser that opens with the
// paths, so the files enter the page as chosen files do, a folder walked as
// the Folder tool's is. A click by Page.Eval counts as the user's; one in the
// page's drop handler opens no chooser. Main thread only.
var drops struct {
	paths       []string // the files of the drag over the page
	answer      []string // what the file chooser gets while Drop clicks
	viewType    uintptr  // WebKitWebView's type, once WebKitGTK is loaded
	selectFiles func(request unsafe.Pointer, files **byte)
}

// dragMark marks a drag whose data the app has read: WebKit asks for each
// of a drag's targets as it enters the page, and the first answer starts
// the drag's paths afresh. chooserMark marks a web view whose file chooser
// the app answers.
const dragMark, chooserMark = "markitai-web-drag", "markitai-web-chooser"

// gValueSize is a GValue's size: its type, then two words of data.
const gValueSize = 24

// dropClick clicks the workbench's drop input, which the workbench
// disables while it ignores drops, as with a dialog open.
const dropClick = `document.querySelector("input[data-drop-input]:enabled")?.click()`

// dropWait bounds the click: an answer WebKitGTK never sends would keep the
// paths for the next chooser the user opens.
const dropWait = 5 * time.Second

// watchDrops reads the paths of each drag over a web view, and answers its
// file chooser while Drop clicks. GTK has started (toolkitReady).
func watchDrops() {
	if err := loadGLib(); err != nil {
		log.Printf("file drops: %v", err)
		return
	}
	received := gtk.signalLookup("drag-data-received", gtk.widgetType())
	if received == 0 {
		log.Print("file drops: no drag-data-received signal")
		return
	}
	// gboolean (*run_file_chooser) (WebKitWebView *, WebKitFileChooserRequest *,
	// gpointer): TRUE says the app answered.
	chooser := purego.NewCallback(func(_, request unsafe.Pointer, _ uintptr) uintptr {
		if drops.answer == nil {
			return 0 // the user's own choice, in WebKit's chooser
		}
		files := make([]*byte, 0, len(drops.answer)+1)
		for _, path := range drops.answer {
			files = append(files, &append([]byte(path), 0)[0])
		}
		files = append(files, nil)
		drops.selectFiles(request, &files[0])
		runtime.KeepAlive(files)
		drops.answer = nil
		return 1
	})
	// drag-data-received's values: the widget, the drag, x, y, the data,
	// info and time.
	hook := purego.NewCallback(func(_, _ uintptr, values unsafe.Pointer, _ uintptr) uintptr {
		widget, view := gtk.valueGetObject(values), webViewType()
		if widget == nil || view == 0 || gtk.isA(widget, view) == 0 {
			return 1
		}
		if gtk.objectGetData(widget, chooserMark) == 0 {
			gtk.objectSetData(widget, chooserMark, 1)
			gtk.signalConnect(widget, "run-file-chooser", chooser, 0, 0, 0)
		}
		if drag := gtk.valueGetObject(unsafe.Add(values, gValueSize)); drag != nil && gtk.objectGetData(drag, dragMark) == 0 {
			gtk.objectSetData(drag, dragMark, 1)
			drops.paths = nil
		}
		if uris := gtk.selectionURIs(gtk.valueGetBoxed(unsafe.Add(values, 4*gValueSize))); uris != nil {
			drops.paths = filePaths(uris)
			glib.strfreev(uris)
		}
		return 1
	})
	gtk.addEmissionHook(received, 0, hook, 0, 0)
}

// webViewType is WebKitWebView's type, 0 until a window has shown a page:
// mygo loads WebKitGTK, 4.1 or 4.0, with its symbols global only then,
// after GTK starts.
func webViewType() uintptr {
	if drops.viewType != 0 {
		return drops.viewType
	}
	viewType, err := purego.Dlsym(purego.RTLD_DEFAULT, "webkit_web_view_get_type")
	if err != nil {
		return 0
	}
	selectFiles, err := purego.Dlsym(purego.RTLD_DEFAULT, "webkit_file_chooser_request_select_files")
	if err != nil {
		return 0
	}
	var getType func() uintptr
	purego.RegisterFunc(&getType, viewType)
	purego.RegisterFunc(&drops.selectFiles, selectFiles)
	drops.viewType = getType()
	return drops.viewType
}

// filePaths is the local files of uris, a NULL-ended list of URIs.
func filePaths(uris **byte) []string {
	paths := []string{}
	for p := uris; *p != nil; p = (**byte)(unsafe.Add(unsafe.Pointer(p), unsafe.Sizeof(p))) {
		if path := glib.filenameFromURI(*p, nil, nil); path != nil {
			paths = append(paths, takeString(path))
		}
	}
	return paths
}

// dragged is how many files the drag over the page carries (drops).
func (w *window) dragged() int {
	n := 0
	mygo.RunOnMain(func() { n = len(drops.paths) })
	return n
}

// dropFiles hands the files of the drag dropped on the workbench to it
// (drops).
func (w *window) dropFiles() {
	var paths []string
	mygo.RunOnMain(func() {
		paths, drops.paths = drops.paths, nil
		if len(paths) > 0 {
			drops.answer = paths
		}
	})
	if len(paths) == 0 {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), dropWait)
	_, err := w.w.Page().EvalContext(ctx, dropClick)
	cancel()
	mygo.RunOnMain(func() { drops.answer = nil })
	if err != nil {
		log.Printf("markitai web: drop: %v", err)
	}
}
