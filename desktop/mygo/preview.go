package main

import (
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/egoist/mygo/ui"

	"markitai.local/mygo/internal/model"
)

// previewPane is the window's one web view (section 8.1). View is nil
// when the system has no web engine.
type previewPane interface {
	View() ui.NativeWebView
	// Load shows html at baseURL and calls done once it has loaded.
	// inFolder is a file in the folder baseURL names, the Markdown file.
	Load(html, baseURL, inFolder string, done func())
	SetZoom(f float64)
	// SetBackground fills the web view with a #rrggbb color until the page
	// paints.
	SetBackground(hex string)
}

// previewKey names a rendered document: an item in a theme.
type previewKey struct {
	id   model.ItemID
	dark bool
}

// previewPage is a document as the web view shows it: on the detail's
// background, a #rrggbb color.
type previewPage struct {
	key        previewKey
	background string
}

// previews renders documents off the UI thread, keeps the last 8, and
// tracks which one the web view shows (8.2).
type previews struct {
	pane      previewPane
	html      map[previewKey]string
	order     []previewKey
	rendering map[previewKey]bool
	loading   previewPage
	shown     previewPage
	zoom      float64
	// background is the color the web view was last filled with.
	background string
	// placed and wasPlaced say whether this frame and the frame before it
	// built the web view. view resets placed as each frame starts.
	placed, wasPlaced bool
	// failed is the document the core could not render, which is never
	// kept. failedShown and failedWasShown say whether this frame and the
	// frame before it showed that failure: shown again after it was not,
	// the document renders again.
	failed                      *previewKey
	failedShown, failedWasShown bool
}

const previewCacheSize = 8

func (p *previews) put(k previewKey, html string) {
	if p.html == nil {
		p.html = map[previewKey]string{}
	}
	p.html[k] = html
	p.order = append(slices.DeleteFunc(p.order, func(o previewKey) bool { return o == k }), k)
	for len(p.order) > previewCacheSize {
		delete(p.html, p.order[0])
		p.order = p.order[1:]
	}
}

// preview is the Preview tab: the web view once the document for this
// item and theme has loaded, and nothing before, so the old document never
// shows under the new title.
func (a *app) preview(c *ui.Context, t *target, d model.Done) {
	l := &a.look
	p := &a.previews
	if p.pane == nil || p.pane.View() == nil {
		key := "preview.unavailable"
		if a.goos == "windows" {
			key = "preview.webview2_missing"
		}
		ui.Column(c).Grow(1).Basis(0).Center().Padding(24).Children(func() {
			ui.Text(c, a.lang.T(key)).TextColor(l.theme.TextMuted).TextAlign(ui.Center)
		})
		return
	}
	// The web view sits under mygo's surface, which leaves a hole where it
	// shows. Until the page paints, the web view shows its own color, else
	// the window's backing: fill it with the detail's Background.
	bg := fmt.Sprintf("#%02x%02x%02x", l.theme.Background.R, l.theme.Background.G, l.theme.Background.B)
	if bg != p.background {
		p.pane.SetBackground(bg)
		p.background = bg
	}
	if z := a.settings.Zoom; z != p.zoom {
		p.pane.SetZoom(z)
		p.zoom = z
	}
	k := previewKey{t.id, l.theme.Dark}
	page := previewPage{k, bg}
	html, ok := p.html[k]
	if !ok && p.failed != nil && *p.failed == k {
		if p.failedWasShown {
			p.failedShown = true
			ui.Column(c).Grow(1).Basis(0).Center().Padding(24).Children(func() {
				ui.Text(c, a.lang.T("preview.failed")).TextAlign(ui.Center)
			})
			return
		}
		p.failed = nil
	}
	switch {
	case !ok:
		a.render(k, d)
	case p.shown != page && p.loading != page:
		p.loading = page
		base := folderURL(d.Folder)
		p.pane.Load(onBackground(html, bg), base, d.Result.MarkdownPath, func() {
			a.post(func() {
				if p.loading == page {
					p.shown = page
				}
			})
		})
	}
	if p.shown == page {
		web := ui.WebView(c.Key("preview"), p.pane.View()).Grow(1).Basis(0).Label(a.lang.T("a11y.preview"))
		// On Linux a web view that comes back into the content keeps the
		// size it had when it left, and only a change of size while it
		// shows reaches GTK (DESIGN 14). One DIP short for its first frame,
		// it changes size on the next.
		if a.goos == "linux" && !p.wasPlaced {
			web.Margin(0, 0, 1, 0)
			c.Invalidate()
		}
		p.placed = true
		return
	}
	ui.Box(c).Grow(1).Basis(0)
}

// onBackground gives the core's document, whose page is transparent, the
// detail's background color. WebKitGTK paints a transparent page in the
// dark color scheme #1e1e1e, not the web view's color (DESIGN 8.1), so the
// page paints the color itself. The core's style allows inline styles, and
// its head ends before the Markdown, which it escapes.
func onBackground(html, color string) string {
	return strings.Replace(html, "</head>", "<style>html { background: "+color+"; }</style>\n</head>", 1)
}

// render asks the core for the item's document in the theme.
func (a *app) render(k previewKey, d model.Done) {
	p := &a.previews
	if p.rendering[k] {
		return
	}
	if p.rendering == nil {
		p.rendering = map[previewKey]bool{}
	}
	p.rendering[k] = true
	theme := "light"
	if k.dark {
		theme = "dark"
	}
	md, folder := d.Result.Markdown, d.Folder
	go func() {
		html, err := a.core.PreviewHTML(md, theme, &folder)
		a.post(func() {
			delete(p.rendering, k)
			if err != nil {
				fmt.Fprintf(os.Stderr, "preview_html: %v\n", err)
				p.failed, p.failedShown = &k, true
				return
			}
			p.put(k, html)
		})
	}()
}

// folderURL is a folder's file: URL with a trailing slash, the preview's
// base URL.
func folderURL(dir string) string {
	p := filepath.ToSlash(dir)
	if !strings.HasPrefix(p, "/") {
		p = "/" + p
	}
	if !strings.HasSuffix(p, "/") {
		p += "/"
	}
	return (&url.URL{Scheme: "file", Path: p}).String()
}

// navigation decides a navigation of the preview (8.1): the document's
// own fragments stay, web and mail links open outside, the rest is
// refused.
func navigation(base, target string) (allow, external bool) {
	if target == base || strings.HasPrefix(target, base+"#") {
		return true, false
	}
	u, err := url.Parse(target)
	if err != nil {
		return false, false
	}
	switch strings.ToLower(u.Scheme) {
	case "http", "https", "mailto":
		return false, true
	}
	return false, false
}
