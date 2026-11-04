package main

import (
	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"
)

// menuBar is the macOS menu bar of section 9.1: its items run the app's
// actions, and follow the state each frame.
type menuBar struct {
	items   map[action]*mygo.MenuItem
	enabled map[action]bool
	remove  string
	// removeKey is Remove's key equivalent as the menu has it now.
	removeKey string
}

// removeAccelerator is Remove's key equivalent, ⌘⌫, which a text field
// takes instead while it has the focus (section 11).
const removeAccelerator = "CmdOrCtrl+Backspace"

// accelerator writes a binding's keys as a menu does.
func accelerator(act action) string {
	for _, b := range bindings("darwin") {
		if b.act != act || !b.native {
			continue
		}
		s := ""
		if b.mods&modCmd != 0 {
			s += "CmdOrCtrl+"
		}
		if b.mods&modShift != 0 {
			s += "Shift+"
		}
		if b.mods&modAlt != 0 {
			s += "Alt+"
		}
		return s + map[action]string{
			actChoose: "O", actAddURL: "L", actConvert: "1", actHistory: "2", actSettings: ",",
			actToggleView: "M", actCopy: "C", actSaveAs: "S", actShow: "R", actRetry: "R", actSearch: "F",
			actZoomIn: "=", actZoomOut: "-", actActualSize: "0",
		}[act]
	}
	return ""
}

// newMenuBar builds the menu bar; press runs an action in the next frame.
func (a *app) newMenuBar(press func(action)) *mygo.Menu {
	bar := &menuBar{items: map[action]*mygo.MenuItem{}, enabled: map[action]bool{}}
	item := func(act action, label string) *mygo.MenuItem {
		it := &mygo.MenuItem{ID: string(act), Label: label, Accelerator: accelerator(act),
			Click: func(*mygo.MenuItem, *mygo.Window) { press(act) }}
		bar.items[act] = it
		return it
	}
	remove := item(actRemove, a.lang.T("convert.remove"))
	remove.Accelerator = removeAccelerator
	bar.removeKey = removeAccelerator
	about := &mygo.MenuItem{Label: a.lang.T("menu.about.mygo"), Click: func(*mygo.MenuItem, *mygo.Window) {
		mygo.App.ShowAboutPanel(mygo.AboutPanelOptions{ApplicationName: "Markitai Mygo", ApplicationVersion: a.version,
			Copyright: a.lang.T("settings.about.copyright")})
	}}
	settings := item(actSettings, a.lang.T("menu.settings"))
	a.menu = bar
	return mygo.NewMenu([]*mygo.MenuItem{
		{Role: mygo.RoleAppMenu, Submenu: []*mygo.MenuItem{
			about, mygo.Separator(), settings, mygo.Separator(),
			{Role: mygo.RoleServices}, mygo.Separator(),
			{Role: mygo.RoleHide}, {Role: mygo.RoleHideOthers}, {Role: mygo.RoleUnhide}, mygo.Separator(),
			{Role: mygo.RoleQuit},
		}},
		// The File and Edit menus place their items as Native's do (shared
		// DESIGN 11): Close (and AppKit's Close All) before the item's
		// commands, Copy Markdown after the pasteboard's.
		{Role: mygo.RoleFileMenu, Submenu: []*mygo.MenuItem{
			item(actChoose, a.lang.T("convert.choose")), item(actAddURL, a.lang.T("menu.add_url")), mygo.Separator(),
			{Role: mygo.RoleClose},
			item(actSaveAs, a.lang.T("detail.save_as")), item(actShow, a.showLabel()), mygo.Separator(),
			item(actRetry, a.lang.T("detail.retry")), remove,
		}},
		{Role: mygo.RoleEditMenu, Submenu: []*mygo.MenuItem{
			{Role: mygo.RoleUndo}, {Role: mygo.RoleRedo}, mygo.Separator(),
			{Role: mygo.RoleCut}, {Role: mygo.RoleCopy}, {Role: mygo.RolePaste},
			{Role: mygo.RoleSelectAll}, mygo.Separator(),
			item(actCopy, a.lang.T("detail.copy")), mygo.Separator(),
			item(actSearch, a.lang.T("history.search")),
		}},
		{Role: mygo.RoleViewMenu, Submenu: []*mygo.MenuItem{
			item(actConvert, a.lang.T("nav.convert")), item(actHistory, a.lang.T("nav.history")), mygo.Separator(),
			item(actToggleView, a.lang.T("menu.toggle_view")), mygo.Separator(),
			item(actZoomIn, a.lang.T("menu.zoom_in")), item(actZoomOut, a.lang.T("menu.zoom_out")),
			item(actActualSize, a.lang.T("menu.actual_size")),
		}},
		{Role: mygo.RoleWindowMenu},
	})
}

// syncMenu enables the menu bar's items as the buttons are, names Remove
// for the page shown, and leaves ⌘⌫ to a text field with the focus, where
// it deletes to the start of the line.
func (a *app) syncMenu(c *ui.Context) {
	if a.menu == nil {
		return
	}
	key := removeAccelerator
	if c.FocusWithin(a.urlField) || c.FocusWithin(a.searchField) {
		key = ""
	}
	if key != a.menu.removeKey {
		a.menu.items[actRemove].SetAccelerator(key)
		a.menu.removeKey = key
	}
	for act, it := range a.menu.items {
		on := a.enabled(act)
		if was, ok := a.menu.enabled[act]; !ok || was != on {
			it.SetEnabled(on)
			a.menu.enabled[act] = on
		}
	}
	label := a.lang.T("convert.remove")
	if a.onHistory() {
		label = a.lang.T("history.remove")
	}
	if label != a.menu.remove {
		a.menu.items[actRemove].SetLabel(label)
		a.menu.remove = label
	}
}
