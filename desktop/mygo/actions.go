package main

import (
	"encoding/json"
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

// previewKeysJS forwards the page's keys of section 8.3 to PreviewKeys.Press.
func previewKeysJS(goos string) string {
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
	// view.zoom (DESIGN 8): the wheel steps view.zoom instead, a notch, or
	// a touchpad's 100 pixels, a step.
	wheel := ""
	if goos == "windows" {
		wheel = `
  let wheel = 0;
  addEventListener("wheel", (e) => {
    if (!e.ctrlKey || e.metaKey || e.deltaY === 0) return;
    e.preventDefault();
    e.stopPropagation();
    if (e.deltaMode !== 0 || Math.abs(e.deltaY) >= 50) {
      wheel = e.deltaY;
    } else {
      wheel += e.deltaY;
      if (Math.abs(wheel) < 100) return;
    }
    globalThis.mygo?.call("PreviewKeys.Press", wheel < 0 ? "zoom_in" : "zoom_out");
    wheel = 0;
  }, {capture: true, passive: false});`
	}
	return `(() => {
  const keys = ` + string(table) + `;
  addEventListener("keydown", (e) => {
    if (` + other + `) return;
    for (const k of keys) {
      if (e.code === k.code && ` + mod + ` === k.mod && e.altKey === k.alt && (k.shift === null || e.shiftKey === k.shift)) {
        e.preventDefault();
        e.stopPropagation();
        globalThis.mygo?.call("PreviewKeys.Press", k.action);
        return;
      }
    }
  }, true);` + wheel + `
})();`
}

// previewKeys is the service the preview's script calls (8.3).
type previewKeys struct {
	goos  string
	press func(action)
}

// Press runs a forwarded key's action. It runs on a goroutine of its own.
func (p *previewKeys) Press(name string) {
	for _, b := range bindings(p.goos) {
		if b.page && string(b.act) == name {
			p.press(b.act)
			return
		}
	}
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
		a.run(c, act)
	}
}
