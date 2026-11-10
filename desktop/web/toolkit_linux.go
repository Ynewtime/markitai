package main

import (
	"log"
	"os"
	"path/filepath"
	"strings"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/egoist/mygo"
)

// nameApp names the app by its identifier: the program name and the app's
// name, so
// that the window's class (Wayland app ID) matches dev.markitai.web.desktop
// and the app's folders, WebKitGTK's data and the single-instance lock all
// go under dev.markitai.web. Before anything reads a path.
func nameApp() {
	mygo.App.SetName(appID)
	if glib, err := purego.Dlopen("libglib-2.0.so.0", purego.RTLD_NOW|purego.RTLD_GLOBAL); err != nil {
		log.Printf("g_set_prgname: %v", err)
	} else {
		var setPrgname func(string)
		purego.RegisterLibFunc(&setPrgname, glib, "g_set_prgname")
		setPrgname(appID)
	}
}

// prepareToolkit readies GTK before mygo starts it, once the app knows its
// languages and theme.
//
//   - GTK's and WebKitGTK's own words, as those of GTK's file chooser, are
//     in the app's language through gettext's LANGUAGE: GTK
//     reads it as it starts, so a language the workbench chooses later
//     reaches them at the next launch. serve and the programs the app opens
//     keep the user's (userEnv).
//   - The installed desktop entry names the app to the portal.
//   - The file chooser is the desktop's, through the portal, when the
//     desktop has a portal file chooser: it follows the desktop's theme
//     and size, and leaves GTK's settings alone. Else GTK's own shows,
//     within the screen (toolkitReady).
//   - GTK's own widgets, as context menus and dialogs, follow the system's
//     dark appearance while the app follows the system: mygo reads it from
//     the portal but tells GTK only as the app sets its theme's source, so
//     the app sets it again whenever the appearance changes.
//   - mygo sets WEBKIT_DISABLE_DMABUF_RENDERER for WebKitGTK as App.Run
//     starts, which serve and the programs the app opens do without.
func prepareToolkit() {
	noteUserEnv("WEBKIT_DISABLE_DMABUF_RENDERER")
	setAppEnv("LANGUAGE", toolkitLanguage(mygo.App.Locale()))
	nameToPortal()
	if os.Getenv("GTK_USE_PORTAL") == "" {
		if err := portalFileChooser(); err != nil {
			log.Printf("file chooser: GTK's own, as the portal has none: %v", err)
		} else {
			setAppEnv("GTK_USE_PORTAL", "1")
		}
	}
	// Main thread only. A change is applied after GTK's notification
	// returns: setting the source from inside it, which calls GTK from a
	// callback of GTK's, never returned. Setting it
	// tells GTK, which reports a change again; applying skips that one.
	var applying, pending bool
	mygo.Theme.OnUpdated(func() {
		if applying || pending || mygo.Theme.Source() != mygo.ThemeSystem {
			return
		}
		pending = true
		go mygo.RunOnMain(func() {
			pending, applying = false, true
			mygo.Theme.SetSource(mygo.ThemeSystem)
			applying = false
		})
	})
}

// toolkitLanguage is the gettext language of a language of the app.
func toolkitLanguage(locale string) string {
	switch workbenchLang(locale) {
	case "zh":
		return "zh_CN"
	case "ja":
		return "ja"
	}
	return "en"
}

// nameToPortal points mygo at the app's installed desktop entry, which it
// registers with the portal as the app's ID, unless the desktop launched
// the app from one: mygo otherwise names it after the executable,
// markitai-web, which no entry is.
func nameToPortal() {
	if os.Getenv("GIO_LAUNCHED_DESKTOP_FILE") != "" || os.Getenv("BAMF_DESKTOP_FILE_HINT") != "" {
		return
	}
	dirs := []string{os.Getenv("XDG_DATA_HOME")}
	if dirs[0] == "" {
		home, _ := os.UserHomeDir()
		dirs[0] = filepath.Join(home, ".local", "share")
	}
	data := os.Getenv("XDG_DATA_DIRS")
	if data == "" {
		data = "/usr/local/share:/usr/share"
	}
	for _, dir := range append(dirs, strings.Split(data, ":")...) {
		entry := filepath.Join(dir, "applications", appID+".desktop")
		if _, err := os.Stat(entry); err == nil && dir != "" {
			setAppEnv("BAMF_DESKTOP_FILE_HINT", entry)
			return
		}
	}
}

// portalFileChooser asks the desktop's portal for its file chooser's
// version: GTK, told to use the portal, waits for an answer that never
// comes from a portal without one. The portal starts if it has to, which
// the timeout allows for.
func portalFileChooser() error {
	if err := loadGLib(); err != nil {
		return err
	}
	return sessionCall("org.freedesktop.portal.Desktop", "/org/freedesktop/portal/desktop", "org.freedesktop.DBus.Properties", "Get",
		tuple(glib.variantNewString("org.freedesktop.portal.FileChooser"), glib.variantNewString("version")), 3000)
}

// gdkRectangle and gdkGeometry are GDK's.
type gdkRectangle struct{ x, y, width, height int32 }

type gdkGeometry struct {
	minWidth, minHeight, maxWidth, maxHeight   int32
	baseWidth, baseHeight, widthInc, heightInc int32
	minAspect, maxAspect                       float64
	winGravity                                 int32
}

// fileChooserMargin is the room GTK's own file chooser leaves around it
// on its screen's work area and in its window.
const fileChooserMargin = 48

// fileChooserChars is the fewest characters GTK's own file chooser's
// entries ask room for: its name entry still shows a short name whole.
const fileChooserChars = 12

// toolkitReady readies what the app adds to GTK once GTK has started, as
// the signals it watches exist from then on: its own file chooser kept
// within its window and screen (sizeFileChoosers), and a drop's files handed
// to the page (watchDrops).
func toolkitReady() {
	if err := loadGTK(); err != nil {
		log.Printf("gtk: %v", err)
		return
	}
	gtk.classRef(gtk.widgetType())
	sizeFileChoosers()
	watchDrops()
}

// sizeFileChoosers keeps GTK's own file chooser within its window and the
// work area of its window's screen, as it opens 1203 by 950 by default,
// past a 1280 by 720 screen, and a tiling window manager centers it on its
// window, as Hyprland does a floating dialog: one wider than a window tiled
// at the screen's edge would reach past it. realize gives each file chooser
// dialog a largest size before it shows. Its name and search entries ask
// for 45 characters, which keeps it at least 736 pixels wide, wider than a
// window tiled at half a 1280-pixel screen: they ask for fewer, down to
// fileChooserChars, until it fits.
func sizeFileChoosers() {
	dialog, container, entry := gtk.fileChooserType(), gtk.containerType(), gtk.entryType()
	var children []unsafe.Pointer
	// void (*GtkCallback) (GtkWidget *widget, gpointer data)
	collect := purego.NewCallback(func(w unsafe.Pointer, _ uintptr) {
		children = append(children, w)
	})
	// entries is the entries inside w, its title bar's among them.
	entries := func(w unsafe.Pointer) []unsafe.Pointer {
		var found []unsafe.Pointer
		for queue := []unsafe.Pointer{w}; len(queue) > 0; {
			next := queue[0]
			queue = queue[1:]
			if gtk.isA(next, entry) != 0 {
				found = append(found, next)
			}
			if gtk.isA(next, container) != 0 {
				children = children[:0]
				gtk.forall(next, collect, 0)
				queue = append(queue, children...)
			}
		}
		return found
	}
	least := func(w unsafe.Pointer) int32 {
		var width, natural int32
		gtk.preferredWidth(w, &width, &natural)
		return width
	}
	realize := gtk.signalLookup("realize", gtk.widgetType())
	if realize == 0 {
		log.Print("file chooser size: no realize signal")
		return
	}
	// gboolean (*GSignalEmissionHook) (GSignalInvocationHint *ihint,
	// guint n_param_values, const GValue *param_values, gpointer data):
	// the first value holds the widget.
	hook := purego.NewCallback(func(_, _ uintptr, values unsafe.Pointer, _ uintptr) uintptr {
		w := gtk.valueGetObject(values)
		if w == nil || gtk.isA(w, dialog) == 0 {
			return 1
		}
		d := gtk.displayDefault()
		var monitor unsafe.Pointer
		width, height := int32(1<<20), int32(1<<20)
		if parent := gtk.transientFor(w); parent != nil {
			gtk.windowSize(parent, &width, &height)
			if win := gtk.widgetWindow(parent); win != nil {
				monitor = gtk.monitorAtWindow(d, win)
			}
		}
		if monitor == nil {
			monitor = gtk.monitorAt(d, 0)
		}
		if monitor == nil {
			return 1
		}
		var area gdkRectangle
		gtk.workarea(monitor, &area)
		g := gdkGeometry{
			maxWidth:  max(min(area.width, width)-2*fileChooserMargin, 1),
			maxHeight: max(min(area.height, height)-2*fileChooserMargin, 1),
		}
		gtk.setGeometryHints(w, nil, &g, 1<<2) // GDK_HINT_MAX_SIZE
		if least(w) > g.maxWidth {
			found := entries(w)
			for chars := int32(40); chars >= fileChooserChars && least(w) > g.maxWidth; chars -= 4 {
				for _, e := range found {
					if gtk.entryWidthChars(e) > chars {
						gtk.setEntryWidth(e, chars)
					}
				}
			}
		}
		return 1
	})
	gtk.addEmissionHook(realize, 0, hook, 0, 0)
}

// openURL opens url with its default program, in the user's environment:
// a launch context gives it the user's values of appEnv. mygo's Shell
// launches with the app's.
func openURL(url string) error {
	var err error
	mygo.RunOnMain(func() { err = launchAsUser(url) })
	return err
}

// showInFolder asks the file manager to show path, as mygo does, and else
// opens its folder in the user's environment: mygo's own fallback opens it
// with the app's.
func showInFolder(path string) {
	mygo.RunOnMain(func() {
		if err := loadGLib(); err != nil {
			mygo.Shell.ShowItemInFolder(path)
			return
		}
		s := glib.variantTypeNew("s")
		items := []unsafe.Pointer{glib.variantNewString(fileURI(path))}
		uris := glib.variantNewArray(s, &items[0], 1)
		glib.variantTypeFree(s)
		err := sessionCall("org.freedesktop.FileManager1", "/org/freedesktop/FileManager1", "org.freedesktop.FileManager1",
			"ShowItems", tuple(uris, glib.variantNewString("")), -1)
		if err != nil {
			if err := launchAsUser(fileURI(filepath.Dir(path))); err != nil {
				log.Printf("markitai web: show %s: %v", path, err)
			}
		}
	})
}

// closeQuestions answers the message dialogs that parent holds up, as the
// quit's question, as their Cancel: the app quits as Quit When Done does
// when the session ends (terminate), and GTK destroys a dialog left up with
// its window, which mygo then destroys again.
func closeQuestions(parent *mygo.Window) {
	if err := loadGTK(); err != nil {
		log.Printf("markitai web: close the question: %v", err)
		return
	}
	mygo.RunOnMain(func() {
		win := parent.NativeHandle()
		list := gtk.listToplevels()
		for l := list; l != nil; l = l.next {
			if gtk.isA(l.data, gtk.messageDialogType()) != 0 && uintptr(gtk.transientFor(l.data)) == win {
				gtk.dialogResponse(l.data, -4) // GTK_RESPONSE_DELETE_EVENT
			}
		}
		gtk.listFree(list)
	})
}
