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

// nameApp names the app by its identifier, as Native and Mygo are named
// (desktop/DESIGN.md section 14): the program name and the app's name, so
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
//     in the app's language through gettext's LANGUAGE, as Mygo's are: GTK
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
func prepareToolkit() {
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
	// callback of GTK's, never returned in Mygo's Linux run. Setting it
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
	err := sessionCall("org.freedesktop.portal.Desktop", "/org/freedesktop/portal/desktop", "org.freedesktop.DBus.Properties", "Get",
		tuple(glib.variantNewString("org.freedesktop.portal.FileChooser"), glib.variantNewString("version")), 3000)
	if err != nil {
		log.Printf("markitai web: GTK's own file chooser, the portal has none: %v", err)
	}
	return err
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
// on its screen's work area.
const fileChooserMargin = 48

// toolkitReady keeps GTK's own file chooser within the work area of its
// window's screen, as it opens 1203 by 950 by default, past a 1280 by 720
// screen: realize, a signal that exists once GTK has started, gives each
// file chooser dialog a largest size before it shows.
func toolkitReady() {
	gtk, err := purego.Dlopen("libgtk-3.so.0", purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		log.Printf("file chooser size: %v", err)
		return
	}
	gobject, err := purego.Dlopen("libgobject-2.0.so.0", purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		log.Printf("file chooser size: %v", err)
		return
	}
	var (
		widgetType, dialogType func() uintptr
		classRef               func(t uintptr) unsafe.Pointer
		signalLookup           func(name string, owner uintptr) uint32
		addEmissionHook        func(signal, detail uint32, hook, data, destroy uintptr) uint64
		isA                    func(instance unsafe.Pointer, t uintptr) int32
		valueGetObject         func(v unsafe.Pointer) unsafe.Pointer
		transientFor           func(w unsafe.Pointer) unsafe.Pointer
		widgetWindow           func(w unsafe.Pointer) unsafe.Pointer
		displayDefault         func() unsafe.Pointer
		monitorAtWindow        func(d, w unsafe.Pointer) unsafe.Pointer
		monitorAt              func(d unsafe.Pointer, n int32) unsafe.Pointer
		workarea               func(m unsafe.Pointer, r *gdkRectangle)
		setGeometryHints       func(w, geometryWidget unsafe.Pointer, g *gdkGeometry, mask int32)
	)
	for _, f := range []struct {
		fn   any
		lib  uintptr
		name string
	}{
		{&widgetType, gtk, "gtk_widget_get_type"},
		{&dialogType, gtk, "gtk_file_chooser_dialog_get_type"},
		{&classRef, gobject, "g_type_class_ref"},
		{&signalLookup, gobject, "g_signal_lookup"},
		{&addEmissionHook, gobject, "g_signal_add_emission_hook"},
		{&isA, gobject, "g_type_check_instance_is_a"},
		{&valueGetObject, gobject, "g_value_get_object"},
		{&transientFor, gtk, "gtk_window_get_transient_for"},
		{&widgetWindow, gtk, "gtk_widget_get_window"},
		{&displayDefault, gtk, "gdk_display_get_default"},
		{&monitorAtWindow, gtk, "gdk_display_get_monitor_at_window"},
		{&monitorAt, gtk, "gdk_display_get_monitor"},
		{&workarea, gtk, "gdk_monitor_get_workarea"},
		{&setGeometryHints, gtk, "gtk_window_set_geometry_hints"},
	} {
		purego.RegisterLibFunc(f.fn, f.lib, f.name)
	}
	dialog := dialogType()
	classRef(widgetType())
	realize := signalLookup("realize", widgetType())
	if realize == 0 {
		log.Print("file chooser size: no realize signal")
		return
	}
	// gboolean (*GSignalEmissionHook) (GSignalInvocationHint *ihint,
	// guint n_param_values, const GValue *param_values, gpointer data):
	// the first value holds the widget.
	hook := purego.NewCallback(func(_, _ uintptr, values unsafe.Pointer, _ uintptr) uintptr {
		w := valueGetObject(values)
		if w == nil || isA(w, dialog) == 0 {
			return 1
		}
		d := displayDefault()
		var monitor unsafe.Pointer
		if parent := transientFor(w); parent != nil {
			if win := widgetWindow(parent); win != nil {
				monitor = monitorAtWindow(d, win)
			}
		}
		if monitor == nil {
			monitor = monitorAt(d, 0)
		}
		if monitor == nil {
			return 1
		}
		var area gdkRectangle
		workarea(monitor, &area)
		g := gdkGeometry{maxWidth: max(area.width-2*fileChooserMargin, 1), maxHeight: max(area.height-2*fileChooserMargin, 1)}
		setGeometryHints(w, nil, &g, 1<<2) // GDK_HINT_MAX_SIZE
		return 1
	})
	addEmissionHook(realize, 0, hook, 0, 0)
}

// openURL opens url with its default program, in the user's environment:
// a launch context gives it the user's values of appEnv. mygo's Shell
// launches with the app's.
func openURL(url string) error {
	var err error
	mygo.RunOnMain(func() { err = launchAsUser(url) })
	return err
}
