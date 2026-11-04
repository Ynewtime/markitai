package main

import (
	"encoding/json"
	"math"
	"strings"

	"github.com/egoist/mygo/ui"

	"markitai.local/mygo/internal/model"
)

// action names a user action of section 9.2. A shortcut, a menu item, a
// button and a key forwarded from the preview all run one.
type action string

const (
	actChoose     action = "choose"
	actAddURL     action = "add_url"
	actPaste      action = "paste"
	actConvert    action = "convert"
	actHistory    action = "history"
	actSettings   action = "settings"
	actToggleView action = "toggle_view"
	actCopy       action = "copy"
	actSaveAs     action = "save_as"
	actShow       action = "show"
	actRetry      action = "retry"
	actRemove     action = "remove"
	actSearch     action = "search"
	actZoomIn     action = "zoom_in"
	actZoomOut    action = "zoom_out"
	actActualSize action = "actual_size"
	actBack       action = "back"
	actClose      action = "close"
	actQuit       action = "quit"
)

// run does an action in a frame.
func (a *app) run(c *ui.Context, act action) {
	switch act {
	case actChoose:
		a.chooseFiles()
	case actAddURL:
		a.show(pathConvert)
		a.urlField.Focus()
	case actPaste:
		a.paste(c)
	case actConvert:
		a.show(pathConvert)
	case actHistory:
		a.show(pathHistory)
	case actSettings:
		a.show(pathSettings)
	case actToggleView:
		a.toggleView()
	case actCopy:
		a.copyMarkdown(c)
	case actSaveAs:
		a.saveAs()
	case actShow:
		a.showInFolder()
	case actRetry:
		if t := a.target(); t != nil && !t.entry {
			a.retry(t.id)
		}
	case actRemove:
		if a.enabled(actRemove) {
			a.remove(a.target().id)
		}
	case actSearch:
		a.show(pathHistory)
		a.searchField.Focus()
	case actZoomIn:
		a.zoom(a.settings.Zoom + model.ZoomStep)
	case actZoomOut:
		a.zoom(a.settings.Zoom - model.ZoomStep)
	case actActualSize:
		a.zoom(1)
	case actBack:
		if strings.HasSuffix(a.router.Path(), "/item") {
			a.router.Back()
		}
	case actClose:
		a.desk.closeWindow()
	case actQuit:
		a.desk.quit()
	}
}

// enabled says whether an action does anything now: buttons and menu
// items follow it.
func (a *app) enabled(act action) bool {
	t := a.target()
	_, done := t.done()
	switch act {
	case actToggleView, actCopy, actSaveAs, actShow:
		return done
	case actRetry:
		if t == nil || t.entry {
			return false
		}
		_, failed := t.state.(model.Failed)
		return failed
	case actRemove:
		if t == nil {
			return false
		}
		_, busy := t.state.(model.Converting)
		return !busy
	}
	return true
}

// mod is a modifier of a binding: Cmd is ⌘ on macOS and Ctrl elsewhere.
type mod uint8

const (
	modCmd mod = 1 << iota
	modShift
	modAlt
	// modAnyShift accepts the key with or without Shift.
	modAnyShift
)

func (m mod) ui() ui.Modifiers {
	var out ui.Modifiers
	if m&modCmd != 0 {
		out |= ui.Cmd
	}
	if m&modShift != 0 {
		out |= ui.Shift
	}
	if m&modAlt != 0 {
		out |= ui.Alt
	}
	return out
}

// binding is one key of section 9.2. native keys are c.Shortcut in the
// root view; page keys are forwarded from the preview (8.3).
type binding struct {
	act    action
	mods   mod
	key    ui.Key
	code   string // KeyboardEvent.code
	native bool
	page   bool
}

// bindings are the keys on goos.
func bindings(goos string) []binding {
	b := []binding{
		{actChoose, modCmd, ui.KeyO, "KeyO", true, true},
		{actAddURL, modCmd, ui.KeyL, "KeyL", true, true},
		{actPaste, modCmd, ui.KeyV, "KeyV", true, false},
		{actConvert, modCmd, ui.Key1, "Digit1", true, true},
		{actHistory, modCmd, ui.Key2, "Digit2", true, true},
		{actSettings, modCmd, ui.KeyComma, "Comma", true, true},
		{actToggleView, modCmd | modShift, ui.KeyM, "KeyM", true, true},
		{actCopy, modCmd | modShift, ui.KeyC, "KeyC", true, true},
		{actSaveAs, modCmd | modShift, ui.KeyS, "KeyS", true, true},
		{actShow, modCmd | modShift, ui.KeyR, "KeyR", true, true},
		{actRetry, modCmd, ui.KeyR, "KeyR", true, false},
		{actSearch, modCmd, ui.KeyF, "KeyF", true, true},
		{actZoomIn, modCmd | modAnyShift, ui.KeyEqual, "Equal", true, true},
		{actZoomOut, modCmd, ui.KeyMinus, "Minus", true, true},
		{actActualSize, modCmd, ui.Key0, "Digit0", true, true},
		{actBack, 0, ui.KeyEscape, "Escape", true, true},
	}
	// The router takes its own back keys natively; the page forwards them.
	if goos == "darwin" {
		b = append(b, binding{actBack, modCmd, ui.KeyBracketLeft, "BracketLeft", false, true})
	} else {
		b = append(b, binding{actBack, modAlt, ui.KeyLeft, "ArrowLeft", false, true})
	}
	if goos == "linux" {
		b = append(b,
			binding{actClose, modCmd, ui.KeyW, "KeyW", true, true},
			binding{actQuit, modCmd, ui.KeyQ, "KeyQ", true, true})
	}
	return b
}

// removeKey is the key that removes the selected row of a focused list.
func removeKey(goos string) (ui.Modifiers, ui.Key) {
	if goos == "darwin" {
		return ui.Cmd, ui.KeyBackspace
	}
	return 0, ui.KeyDelete
}

// shortcuts runs the native keys of section 9.2.
func (a *app) shortcuts(c *ui.Context) {
	for _, b := range bindings(a.goos) {
		if !b.native {
			continue
		}
		hit := c.Shortcut(b.mods.ui(), b.key)
		if b.mods&modAnyShift != 0 && !hit {
			hit = c.Shortcut(b.mods.ui()|ui.Shift, b.key)
		}
		if hit {
			a.run(c, b.act)
			return
		}
	}
}

// previewJS is the preview's script (section 8.3). The page's keys of
// section 9.2 go to Preview.Press, as does the mouse's back button; links
// and files dropped on the page go to Preview.Drop; on Windows, Ctrl with
// the wheel goes to Preview.Wheel. The document itself can be tabbed to.
func previewJS(goos string) string {
	type key struct {
		Code   string `json:"code"`
		Mod    bool   `json:"mod"`
		Shift  *bool  `json:"shift"`
		Alt    bool   `json:"alt"`
		Action action `json:"action"`
	}
	var keys []key
	for _, b := range bindings(goos) {
		if !b.page {
			continue
		}
		k := key{Code: b.code, Mod: b.mods&modCmd != 0, Alt: b.mods&modAlt != 0, Action: b.act}
		if b.mods&modAnyShift == 0 {
			shift := b.mods&modShift != 0
			k.Shift = &shift
		}
		keys = append(keys, k)
	}
	table, _ := json.Marshal(keys)
	mod, other := "e.ctrlKey", "e.metaKey"
	if goos == "darwin" {
		mod, other = "e.metaKey", "e.ctrlKey"
	}
	// WebView2 zooms the page itself on Ctrl and the wheel, apart from
	// view.zoom (DESIGN 8): the page hands the wheel to the app instead,
	// which steps view.zoom (wheeled).
	wheel := ""
	if goos == "windows" {
		wheel = `
  addEventListener("wheel", (e) => {
    if (!e.ctrlKey || e.metaKey || !e.wheelDeltaY) return;
    e.preventDefault();
    e.stopPropagation();
    call("Wheel", e.wheelDeltaY);
  }, {capture: true, passive: false});`
	}
	return `(() => {
  const call = (method, ...args) => globalThis.mygo?.call("Preview." + method, ...args);
  const keys = ` + string(table) + `;
  // A key goes by the US key that types what it types, as mygo takes the
  // native keys (platform.KeyForRune), on any keyboard layout: AZERTY
  // types A at KeyQ's place and a comma at KeyM's. A key that types none,
  // as a Cyrillic letter, goes by its place.
  const us = {"-": "Minus", "=": "Equal", ",": "Comma", ".": "Period", "/": "Slash", ";": "Semicolon", "'": "Quote",
    "[": "BracketLeft", "]": "BracketRight", "\\": "Backslash", "\x60": "Backquote"};
  const place = (e) => /^[a-z]$/i.test(e.key) ? "Key" + e.key.toUpperCase() : /^[0-9]$/.test(e.key) ? "Digit" + e.key : us[e.key] ?? e.code;
  addEventListener("keydown", (e) => {
    if (` + other + `) return;
    const code = place(e);
    for (const k of keys) {
      if (code === k.code && ` + mod + ` === k.mod && e.altKey === k.alt && (k.shift === null || e.shiftKey === k.shift)) {
        e.preventDefault();
        e.stopPropagation();
        call("Press", k.action);
        return;
      }
    }
  }, true);
  // The mouse's back button is the app's Back and its forward button does
  // nothing: neither moves the page's own history. WebKitGTK gives the
  // page neither as such (toolkitReady takes the back button there).
  for (const type of ["mousedown", "mouseup", "auxclick"]) {
    addEventListener(type, (e) => {
      if (e.button !== 3 && e.button !== 4) return;
      e.preventDefault();
      e.stopPropagation();
      if (type === "mouseup" && e.button === 3) call("Press", "back");
    }, true);
  }
  // What is dropped on the page goes to the app, as on the list (10.1),
  // and the page never opens it: links here, files once mygo has their
  // paths (mygo:file-drop). A drag from the page itself is the page's.
  const kinds = ["Files", "text/uri-list", "text/x-moz-url", "text/plain"];
  let inside = false;
  addEventListener("dragstart", () => { inside = true; }, true);
  addEventListener("dragend", () => { inside = false; }, true);
  const takes = (dt) => !inside && !!dt && Array.from(dt.types).some((t) => kinds.includes(t));
  addEventListener("dragover", (e) => {
    if (!takes(e.dataTransfer)) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
  }, true);
  addEventListener("drop", (e) => {
    const dt = e.dataTransfer;
    if (!takes(dt)) return;
    e.preventDefault();
    if (Array.from(dt.types).includes("Files")) return;
    for (const type of kinds.slice(1)) {
      const lines = (dt.getData(type) || "").split(/\r?\n/).map((l) => l.trim()).filter((l) => l && !l.startsWith("#"));
      // text/x-moz-url is a URL and its title, a line each.
      const links = type === "text/x-moz-url" ? lines.filter((_, i) => i % 2 === 0) : lines;
      if (links.length > 0) {
        call("Drop", [], links);
        return;
      }
    }
  }, true);
  globalThis.mygo?.on("mygo:file-drop", (e) => call("Drop", e.paths, []));
  // The document takes the focus, and its arrows scroll it, when Tab
  // comes to the page and it has nothing else to stop at.
  const focusable = () => { if (document.body) document.body.tabIndex = 0; };
  if (document.readyState === "loading") addEventListener("DOMContentLoaded", focusable, {once: true});
  else focusable();` + wheel + `
})();`
}

// previewService is what the preview's script calls (8.3).
type previewService struct {
	goos  string
	press func(action)
	drop  func(paths, links []string)
	wheel func(delta float64)
}

// Press runs a forwarded key's action. It runs on a goroutine of its own.
func (p *previewService) Press(name string) {
	for _, b := range bindings(p.goos) {
		if b.page && string(b.act) == name {
			p.press(b.act)
			return
		}
	}
}

// Drop takes the files, else the links, dropped on the page. It runs on a
// goroutine of its own.
func (p *previewService) Drop(paths, links []string) {
	p.drop(paths, links)
}

// Wheel takes Ctrl with the wheel over the page, on Windows: the event's
// wheelDeltaY. It runs on a goroutine of its own.
func (p *previewService) Wheel(delta float64) {
	p.wheel(delta)
}

// wheelSlack is how far short of a notch the wheel may come and still
// step: the page gets the wheel in whole pixels, so at zoom z a notch can
// come up to z/120 of a notch short.
const wheelSlack = 0.05

// wheeled steps view.zoom once for each notch of Ctrl and the wheel over
// the preview (8.3). WebView2 gives the page the wheel in the page's own
// pixels, 120 a notch divided by the page's zoom: at 2.0 a notch came as
// 60, and counting 120 of them stepped every other notch (found in the
// Windows run). So delta times the zoom the web view has is the wheel's
// own. Turning the wheel the other way starts the count again.
func (a *app) wheeled(delta float64) {
	a.post(func() {
		z := a.previews.zoom
		if z == 0 {
			z = a.settings.Zoom
		}
		n := delta * z / 120
		if n*a.wheel < 0 {
			a.wheel = 0
		}
		a.wheel += n
		for math.Abs(a.wheel) >= 1-wheelSlack {
			act, notch := actZoomIn, 1.0
			if a.wheel < 0 {
				act, notch = actZoomOut, -1
			}
			a.later = append(a.later, act)
			a.wheel -= notch
		}
	})
}

// pressed queues an action from outside a frame: the next frame runs it.
func (a *app) pressed(act action) {
	a.post(func() { a.later = append(a.later, act) })
}

func (a *app) runLater(c *ui.Context) {
	later := a.later
	a.later = nil
	for _, act := range later {
		// A History entry's Markdown may still be on its way; its arrival
		// brings the next frame.
		if t := a.target(); t != nil && t.loading && (act == actCopy || act == actSaveAs || act == actShow) {
			a.later = append(a.later, act)
			continue
		}
		// A key forwarded from the preview, or a menu item chosen while it
		// has the keyboard: the web view keeps the keyboard from the field
		// this focuses, unless it leaves this frame (8.3).
		if act == actAddURL || act == actSearch {
			// Work posted now runs once this frame has ended: the next
			// one shows the web view again.
			a.previews.hold = true
			a.post(func() { a.previews.hold = false })
		}
		a.run(c, act)
	}
}
