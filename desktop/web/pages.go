package main

import (
	"math"
	"runtime"
	"strings"
)

// MarkitaiWeb is what the preload tells the app from the main frame, the
// one frame where mygo's bridge exists: the page the window shows, the
// workbench's look, the keys the preload takes, and on Linux the drops whose
// files the page does not get. The window's pages are
// the app's own (willNavigate keeps it on them), which is why the window
// trusts every origin to call it (window.go). A link in converted content
// cannot call it: the snapshot frame has no bridge. Each method leaves the
// app as it is for arguments it does not know.
type MarkitaiWeb struct{ w *window }

// Shown says the main frame shows a page: a status page by its paint
// (data-paint), or the workbench by the token it holds.
func (s *MarkitaiWeb) Shown(paint int, token string) {
	s.w.shown(sight{Paint: paint, Token: Token(token)})
}

// Look says the workbench's language and theme as it keeps them, at its
// start and whenever it stores another: the n-th report of the document
// doc. mygo runs each call in a goroutine of its own, so reports can come
// out of order; the window takes the latest alone (reports).
func (s *MarkitaiWeb) Look(lang, theme, doc string, n int) { s.w.look(doc, n, lookOf(lang, theme)) }

// Key is a shortcut the preload takes on Linux and Windows, which have no
// menu bar to hold them (keys).
func (s *MarkitaiWeb) Key(name string) { s.w.key(key(name)) }

// Wheel is Ctrl with the wheel over the page on Linux and Windows: the
// event's wheelDeltaY, 120 a notch as the page measures it (notches).
func (s *MarkitaiWeb) Wheel(delta float64) { s.w.wheel(delta) }

// Dragged is how many files the drag over the workbench carries that the
// page does not get, on Linux (drops): it gets none of them.
func (s *MarkitaiWeb) Dragged() int { return s.w.dragged() }

// Drop hands the workbench the files of such a drag, dropped on it.
func (s *MarkitaiWeb) Drop() { s.w.dropFiles() }

// key names a shortcut of keys.
type key string

const (
	keyZoomIn     key = "zoom-in"
	keyZoomOut    key = "zoom-out"
	keyZoomReset  key = "zoom-reset"
	keyFullScreen key = "full-screen"
	keyQuit       key = "quit"
)

// keys are the shortcuts of macOS's menu bar (mygo's default menu: Quit,
// View's Actual Size, Zoom In, Zoom Out and Full Screen) as Linux and
// Windows have them, which the preload takes from the page: keydown's key,
// with Ctrl. Windows quits with Alt+F4, which closes the window. Ctrl with
// the wheel zooms too, a
// step a notch as the keys step (wheelCount): WebView2 would zoom by
// itself, apart from the factor the keys change.
func keys(goos string) map[string]key {
	switch goos {
	case "linux":
		return map[string]key{"=": keyZoomIn, "+": keyZoomIn, "-": keyZoomOut, "0": keyZoomReset,
			"F11": keyFullScreen, "q": keyQuit}
	case "windows":
		return map[string]key{"=": keyZoomIn, "+": keyZoomIn, "-": keyZoomOut, "0": keyZoomReset,
			"F11": keyFullScreen}
	}
	return nil
}

// zoomSteps are the zoom factors of mygo's View menu on macOS (Zoom In and
// Zoom Out step through them; menu.go of mygo), so the keys of every system
// zoom alike.
var zoomSteps = []float64{0.25, 0.33, 0.5, 0.67, 0.75, 0.8, 0.9, 1, 1.1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5}

// nextZoom is the step after f, up or down.
func nextZoom(f float64, up bool) float64 {
	if up {
		for _, s := range zoomSteps {
			if s > f+0.001 {
				return s
			}
		}
		return zoomSteps[len(zoomSteps)-1]
	}
	for i := len(zoomSteps) - 1; i >= 0; i-- {
		if zoomSteps[i] < f-0.001 {
			return zoomSteps[i]
		}
	}
	return zoomSteps[0]
}

// notches is how many notches of the wheel a page's wheelDeltaY is, the
// page zoomed by zoom. WebView2 gives the page 120 a notch divided by the
// page's zoom, 60 at 2.0; its deltaY
// also changes with the system's lines a notch, wheelDeltaY does not.
// WebKitGTK gives 120 for each unit of GDK's smooth scroll, whatever the
// zoom: a notch of the wheel is one unit or more, as the compositor scales
// it (Hyprland's is 1.5, 180 to the page).
func notches(goos string, delta, zoom float64) float64 {
	if goos == "windows" {
		return delta * zoom / 120
	}
	return delta / 120
}

// wheelSlack is how far short of a notch the wheel may come and still
// step: WebView2 gives the page whole pixels, so at zoom z a notch can
// come up to z/120 of a notch short.
const wheelSlack = 0.05

// wheelCount counts the notches of Ctrl with the wheel into zoom steps,
// one a notch: smaller turns, as of a touchpad or a fine wheel, add up to
// a notch first, and turning the other way starts the count again. A
// Windows notch is 1 (notches), and an event that holds several, which
// Chromium makes of quick turns, steps once for each. A Linux notch may
// be 1 or 1.5 by the compositor: an event steps once, whatever it holds
// over a notch.
type wheelCount struct {
	n float64
}

// add counts n notches of goos's wheel: the steps they make, positive to
// zoom in.
func (c *wheelCount) add(goos string, n float64) int {
	if n*c.n < 0 {
		c.n = 0
	}
	c.n += n
	steps := 0
	for math.Abs(c.n) >= 1-wheelSlack {
		step := 1
		if c.n < 0 {
			step = -1
		}
		steps += step
		if goos != "windows" {
			c.n = 0
			break
		}
		c.n -= float64(step)
	}
	return steps
}

// preloadScript runs in every document of the window, its frames' too,
// before their own scripts. A frame, as the snapshot of converted content,
// gets only the wheel guard: WebView2 would zoom it by itself. The main
// frame:
//
//   - replaces window.print, which the web view ignores (it returns at
//     once, with no panel and no afterprint), so the workbench's PDF export
//     reaches exportPDF through window.open of openPrint;
//   - says at pageshow, also as the history brings a page back, which page
//     it shows: data-paint of a status page, the token of the workbench, as
//     the address brings it (#token=) and its sessionStorage keeps it
//     (markitai.service-token, src/api/token.ts of the workbench);
//   - reports the workbench's look as it starts and as it stores another
//     (markitai.lang, markitai.theme, as boot.js reads them), numbered in
//     the document, which the workbench itself never sends anywhere: in a
//     browser and under the Python server it works as it does without the
//     app;
//   - keeps the mouse's back and forward buttons from leaving the page;
//   - on Linux, takes a drag of files the workbench does not get (types
//     without Files): it asks how many files it carries (Dragged), shows
//     the workbench's veil (markitai:drag) and accepts the drop, unless
//     the drop input is disabled, then hands it over (Drop);
//   - takes the keys keys lists, and Ctrl with the wheel (Wheel), where
//     it lists any.
func preloadScript(goos string) string {
	drops := ""
	if goos == "linux" {
		drops = preloadDrops
	}
	return `(() => {
const call = (method, ...args) => { try { window.mygo.call("MarkitaiWeb." + method, ...args).catch(() => {}); } catch {} };
const keys = ` + jsKeys(keys(goos)) + `;
const zooms = Object.keys(keys).length > 0;
if (window.top !== window) {
  if (zooms) addEventListener("wheel", (e) => { if (e.ctrlKey) e.preventDefault(); }, { capture: true, passive: false });
  return;
}
window.print = () => { window.open("` + openPrint + `"); };
const workbench = location.protocol === "http:";
let token = new URLSearchParams(location.hash.slice(1)).get("token");
if (token === null && workbench) { try { token = sessionStorage.getItem("markitai.service-token"); } catch {} }
const doc = Math.random().toString(36).slice(2);
let looks = 0;
const look = () => {
  let lang = "", theme = "";
  try { lang = localStorage.getItem("markitai.lang") || ""; theme = localStorage.getItem("markitai.theme") || ""; } catch {}
  call("Look", lang, theme, doc, ++looks);
};
if (workbench) {
  for (const name of ["setItem", "removeItem", "clear"]) {
    const original = Storage.prototype[name];
    Storage.prototype[name] = function (...args) {
      const result = original.apply(this, args);
      try { if (this === localStorage && (name === "clear" || /^markitai\.(lang|theme)$/.test(args[0]))) look(); } catch {}
      return result;
    };
  }
}
addEventListener("pageshow", () => {
  call("Shown", Number(document.body && document.body.dataset.paint) || 0, workbench ? token || "" : "");
  if (workbench) look();
});
for (const type of ["mousedown", "mouseup", "auxclick"]) {
  addEventListener(type, (e) => { if (e.button === 3 || e.button === 4) e.preventDefault(); }, true);
}
` + drops + `if (!zooms) return;
addEventListener("keydown", (e) => {
  if (e.isComposing || e.altKey || e.metaKey) return;
  const plain = e.key.length === 1 && !e.shiftKey ? e.key.toLowerCase() : e.key;
  const name = e.key === "F11" ? (e.ctrlKey || e.shiftKey ? undefined : keys.F11) : e.ctrlKey ? keys[plain] : undefined;
  if (!name) return;
  e.preventDefault();
  call("Key", name);
}, true);
addEventListener("wheel", (e) => {
  if (!e.ctrlKey) return;
  e.preventDefault();
  if (e.wheelDeltaY) call("Wheel", e.wheelDeltaY);
}, { capture: true, passive: false });
})();`
}

// preloadDrops is the preload's part for drops on Linux, on the workbench
// (preloadScript, drops).
const preloadDrops = `if (workbench) {
  let depth = 0, carried = 0;
  const handed = (e) => { const t = e.dataTransfer ? [...e.dataTransfer.types] : []; return !t.includes("Files") && t.includes("text/uri-list"); };
  const veil = (shown) => dispatchEvent(new CustomEvent("markitai:drag", { detail: shown ? "enter" : "leave" }));
  addEventListener("dragenter", (e) => {
    if (!handed(e) || depth++) return;
    carried = 0;
    try { window.mygo.call("MarkitaiWeb.Dragged").then((n) => { if (depth && n > 0) { carried = n; veil(true); } }, () => {}); } catch {}
  }, true);
  addEventListener("dragleave", (e) => {
    if (!handed(e) || !depth || --depth) return;
    if (carried) veil(false);
    carried = 0;
  }, true);
  addEventListener("dragover", (e) => {
    const input = document.querySelector("input[data-drop-input]");
    if (!carried || !input || !handed(e)) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = input.disabled ? "none" : "copy";
  }, true);
  addEventListener("drop", (e) => {
    if (!handed(e)) return;
    const had = carried;
    depth = 0;
    carried = 0;
    if (!had) return;
    e.preventDefault();
    veil(false);
    call("Drop");
  }, true);
}
`

// jsKeys writes keys as a JavaScript object of keydown's key to the
// shortcut's name.
func jsKeys(keys map[string]key) string {
	var b strings.Builder
	b.WriteString("{")
	first := true
	for _, k := range []string{"=", "+", "-", "0", "F11", "q"} {
		name, ok := keys[k]
		if !ok {
			continue
		}
		if !first {
			b.WriteString(", ")
		}
		first = false
		b.WriteString(`"` + k + `": "` + string(name) + `"`)
	}
	b.WriteString("}")
	return b.String()
}

// preload is this system's preloadScript.
var preload = preloadScript(runtime.GOOS)
