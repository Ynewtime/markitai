package main

import (
	"runtime"
	"strings"
)

// MarkitaiWeb is what the preload tells the app from the main frame, the
// one frame where mygo's bridge exists: the page the window shows, the
// workbench's look, and the keys the preload takes. The window's pages are
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
// with Ctrl. Windows quits with Alt+F4, which closes the window, as Native
// does (desktop/DESIGN.md section 11). Ctrl with the wheel zooms too, in
// steps as the keys do: WebView2 would zoom by itself, apart from the
// factor the keys change.
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
//   - takes the keys keys lists, and Ctrl with the wheel, where it lists
//     any.
func preloadScript(goos string) string {
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
if (!zooms) return;
addEventListener("keydown", (e) => {
  if (e.isComposing || e.altKey || e.metaKey) return;
  const plain = e.key.length === 1 && !e.shiftKey ? e.key.toLowerCase() : e.key;
  const name = e.key === "F11" ? (e.ctrlKey || e.shiftKey ? undefined : keys.F11) : e.ctrlKey ? keys[plain] : undefined;
  if (!name) return;
  e.preventDefault();
  call("Key", name);
}, true);
let wheel = 0;
addEventListener("wheel", (e) => {
  if (!e.ctrlKey) return;
  e.preventDefault();
  wheel += e.deltaY;
  if (Math.abs(wheel) < 40) return;
  call("Key", wheel < 0 ? "` + string(keyZoomIn) + `" : "` + string(keyZoomOut) + `");
  wheel = 0;
}, { capture: true, passive: false });
})();`
}

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
