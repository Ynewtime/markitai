package main

import (
	"github.com/ebitengine/purego/objc"
)

// showAbout shows AppKit's standard About panel with what Native's shows:
// the app's name, its version, the core's version as credits and the
// copyright. Version is given empty: AppKit otherwise adds the build
// number, the same 2.3.2, in brackets after the version, and mygo's
// ShowAboutPanel leaves out options given empty (internal/darwin/app.go:452-469),
// so the app asks AppKit itself, on the main thread a menu's Click runs on.
func showAbout(name, credits, copyright string) {
	sel := objc.RegisterName
	str := func(s string) objc.ID {
		return objc.ID(objc.GetClass("NSString")).Send(sel("stringWithUTF8String:"), s)
	}
	opts := objc.ID(objc.GetClass("NSMutableDictionary")).Send(sel("dictionary"))
	set := func(key string, value objc.ID) { opts.Send(sel("setObject:forKey:"), value, str(key)) }
	set("ApplicationName", str(name))
	set("Version", str(""))
	set("Copyright", str(copyright))
	set("Credits", objc.ID(objc.GetClass("NSAttributedString")).Send(sel("alloc")).Send(sel("initWithString:"), str(credits)).Send(sel("autorelease")))
	app := objc.ID(objc.GetClass("NSApplication")).Send(sel("sharedApplication"))
	app.Send(sel("activateIgnoringOtherApps:"), true)
	app.Send(sel("orderFrontStandardAboutPanelWithOptions:"), opts)
}
