package main

import (
	"log"
	"os"

	"github.com/ebitengine/purego"
	"github.com/egoist/mygo"
)

// prepareToolkit readies GTK before mygo starts it.
//
//   - The program name and the app's name are its identifier, so that the
//     window's class (Wayland app ID) matches dev.markitai.mygo.desktop and
//     WebKitGTK's data, the content rules and the cookies all go under
//     dev.markitai.mygo, beside the app's own folders (section 2.1).
//   - The file chooser is the desktop's, through the portal, as Native's:
//     it follows the desktop's theme and size.
//   - GTK's own widgets, as context menus, follow the system's dark
//     appearance: mygo reads it from the portal but tells GTK only when
//     the app sets its theme's source, so the app sets it, and again
//     whenever the appearance changes (section 10.4).
func prepareToolkit() {
	mygo.App.SetName(appID)
	if glib, err := purego.Dlopen("libglib-2.0.so.0", purego.RTLD_NOW|purego.RTLD_GLOBAL); err != nil {
		log.Printf("g_set_prgname: %v", err)
	} else {
		var setPrgname func(string)
		purego.RegisterLibFunc(&setPrgname, glib, "g_set_prgname")
		setPrgname(appID)
	}
	if os.Getenv("GTK_USE_PORTAL") == "" {
		os.Setenv("GTK_USE_PORTAL", "1")
	}
	if mygo.Theme.Source() != mygo.ThemeSystem {
		return // a test build's fixed appearance (e2e.go)
	}
	mygo.Theme.SetSource(mygo.ThemeSystem)
	// Main thread only. A change is applied after GTK's notification
	// returns: setting the source from inside it, which calls GTK from a
	// callback of GTK's, never returned in the Linux run. Setting it tells
	// GTK, which reports a change again; applying skips that one.
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
