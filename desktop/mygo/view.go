package main

import (
	"strings"
	"time"

	"github.com/egoist/mygo/transfer"
	"github.com/egoist/mygo/ui"

	"markitai.local/mygo/internal/model"
)

// stackBelow is the content width under which list and detail stack
// (section 6.10).
const stackBelow = 760

const sidebarWidth = 200

// mozURL is a link's URL and title, a line each: Chromium and Firefox
// offer it, and Chromium on Linux offers no URI list for a link.
const mozURL transfer.Format = "text/x-moz-url"

// dropFormats are what the content area takes from a drag (10.1).
var dropFormats = transfer.DropOptions{Formats: []transfer.Format{transfer.FileList, transfer.URIList, mozURL}}

// view builds the window from the state, on the UI thread.
func (a *app) view(c *ui.Context) {
	a.previews.wasPlaced, a.previews.placed = a.previews.placed, false
	a.previews.failedWasShown, a.previews.failedShown = a.previews.failedShown, false
	if c.Preferences().ReduceMotion {
		c.DisableAnimations()
	}
	a.look = makeLook(*c.Theme(), c.Preferences())
	c.SetTheme(&a.look.theme)
	for _, s := range a.announce {
		c.Announce(s)
	}
	a.announce = nil
	if a.fatal {
		a.fatalView(c)
		return
	}
	a.runLater(c)
	if a.asking == askingNone {
		a.shortcuts(c)
	}
	a.followWidth(c)
	if page := top(a.router.Path()); page != a.page {
		if page == pathSettings {
			a.config.read = false
		}
		a.page = page
	}
	ui.Row(c).Fill().AlignItems(ui.Stretch).Children(func() {
		a.sidebar(c)
		content := ui.Column(c.Key("content")).Grow(1).Basis(0).MinWidth(0).AlignItems(ui.Stretch)
		dragging := false
		if a.page != pathSettings {
			if paths := content.DroppedFiles(); len(paths) > 0 {
				a.addPaths(paths)
			}
			if d, ok := ui.DropData(content, dropFormats); ok {
				a.dropped(d.Data)
			}
			_, over := ui.DataDragOver(content, dropFormats)
			dragging = over || content.FileDragOver()
		}
		content.Children(func() {
			// Pages move only as the stacked layout pushes and pops; any
			// other change shows at once (section 7), as does the stacked
			// item page that a wider window replaces with the list.
			a.router.Transition = ui.TransitionNone
			if a.stacked(c) && top(a.router.Path()) == top(a.built) {
				a.router.Transition = ui.TransitionAuto
			}
			a.router.View(c, func(r *ui.Route) { a.route(c, r, dragging) }).Grow(1)
			a.built = a.router.Path()
		})
	})
	a.alerts(c)
	a.syncMenu(c)
}

// top is the sidebar's page of a path.
func top(path string) string {
	if i := strings.Index(path[1:], "/"); i >= 0 {
		return path[:i+1]
	}
	return path
}

// stacked reports whether list and detail stack at this window's width.
func (a *app) stacked(c *ui.Context) bool {
	w, _ := c.Size()
	return w-sidebarWidth < stackBelow
}

// followWidth leaves an item page that no longer applies: the window grew
// past the stacked width, or the item is gone (6.10).
func (a *app) followWidth(c *ui.Context) {
	path := a.router.Path()
	if !strings.HasSuffix(path, "/item") {
		return
	}
	if !a.stacked(c) || a.target() == nil {
		a.router.Replace(top(path))
	}
}

func (a *app) route(c *ui.Context, r *ui.Route, dragging bool) {
	// The router builds a page going away too, as the next one comes in;
	// handles bind to the controls of the page shown only.
	a.leaving = r.Path() != a.router.Path()
	defer func() { a.leaving = false }()
	switch {
	case r.Match(pathConvert):
		r.Title(a.lang.T("nav.convert"))
		a.convertPage(c, dragging, false)
	case r.Match(pathConvertItem):
		r.Title(a.target().name())
		a.convertPage(c, dragging, true)
	case r.Match(pathHistory):
		r.Title(a.lang.T("nav.history"))
		a.historyPage(c, false)
	case r.Match(pathHistoryItem):
		r.Title(a.target().name())
		a.historyPage(c, true)
	case r.Match(pathSettings):
		r.Title(a.lang.T("nav.settings"))
		a.settingsPage(c)
	}
}

// listAndDetail lays out a list column and the detail: side by side, or
// in the stacked layout the one the page is.
func (a *app) listAndDetail(c *ui.Context, item bool, list func(stacked bool), detail func(stacked bool)) {
	l := &a.look
	if a.stacked(c) {
		if item {
			ui.Column(c).Fill().Background(l.theme.Background).Children(func() { detail(true) })
		} else {
			ui.Column(c).Fill().Background(l.listFill).Children(func() { list(true) })
		}
		return
	}
	ui.Split(c, &a.listWidth, func() {
		ui.Column(c).Fill().Background(l.listFill).BorderWidth(0, 1, 0, 0).BorderColor(l.rule).Children(func() { list(false) })
	}, func() {
		ui.Column(c).Fill().Background(l.theme.Background).Children(func() { detail(false) })
	}).Fill()
	a.listWidth = min(max(a.listWidth, 260), 420)
}

// sidebar is the navigation: the pages, then Settings at the bottom, two
// sidebars on one page path (section 14), each a tree named for assistive
// technology (section 11).
func (a *app) sidebar(c *ui.Context) {
	l := &a.look
	ui.Themed(c, &l.quiet).Width(sidebarWidth).Shrink(0).Background(l.sidebarFill).BorderWidth(0, 1, 0, 0).BorderColor(l.rule).
		AlignItems(ui.Stretch).Children(func() {
		page := a.page
		if ui.Sidebar(c.Key("nav"), &page, func() {
			a.sidebarItem(c, pathConvert, iconConvert, "nav.convert")
			a.sidebarItem(c, pathHistory, iconHistory, "nav.history")
		}).Grow(1).Background(l.sidebarFill).Label(a.lang.T("app.name.mygo")).Changed() {
			a.show(page)
		}
		footer := a.page
		if ui.Sidebar(c.Key("nav.footer"), &footer, func() {
			a.sidebarItem(c, pathSettings, iconSettings, "nav.settings")
		}).Background(l.sidebarFill).Label(a.lang.T("nav.settings")).Changed() {
			a.show(footer)
		}
	})
}

// sidebarItem draws its glyph and the chosen item's bar itself: mygo
// paints an item's glyph in the accent, which the quiet theme makes the
// chosen fill (ui/sidebar.go:180-200).
func (a *app) sidebarItem(c *ui.Context, id string, icon *ui.SVG, key string) {
	l := &a.look
	chosen := a.page == id
	color := l.theme.TextMuted
	if chosen {
		color = l.theme.Text
	}
	accent := l.theme.Accent
	ui.SidebarItem(c, id, nil, a.lang.T(key)).PaddingStart(8 + 16 + 8).Draw(func(p *ui.Painter, r ui.Rect) {
		if chosen {
			p.Fill(ui.Rect{X: r.X, Y: r.Y, W: 2, H: r.H}, accent, 1)
		}
		p.Icon(icon, ui.Rect{X: r.X + 8, Y: r.Y + (r.H-16)/2, W: 16, H: 16}, color)
	})
}

// header is a list column's header: the page title and its buttons.
func (a *app) header(c *ui.Context, title string, buttons func()) {
	l := &a.look
	ui.Row(c).Height(52).PaddingX(12).Gap(8).AlignItems(ui.Center).Shrink(0).BorderWidth(0, 0, 1, 0).BorderColor(l.rule).Children(func() {
		ui.Text(c, title).FontSize(l.size(17)).FontWeight(600).Role(ui.RoleHeading).Level(1).SingleLine()
		ui.Spacer(c)
		buttons()
	})
}

// iconButton is an icon-only button named by its copy key.
func (a *app) iconButton(c *ui.Context, key, label string, icon *ui.SVG, disabled bool) ui.Element {
	return ui.Button(c.Key(key), "").Label(label).Tooltip(label).Disabled(disabled).Size(32, 32).Padding(0).Children(func() {
		ui.Icon(c, icon).FontSize(16)
	})
}

// noticeLine shows the notice given on page for 8 s (6.7).
func (a *app) noticeLine(c *ui.Context, page string) {
	if a.notice.text == "" || a.notice.page != page {
		return
	}
	// A clock behind the notice's, as the Tester's, starts it now.
	if a.notice.at.After(c.Now()) {
		a.notice.at = c.Now()
	}
	left := 8*time.Second - c.Now().Sub(a.notice.at)
	if left <= 0 {
		a.notice = notice{}
		return
	}
	c.After(left)
	l := &a.look
	ui.Row(c.Key("notice")).Gap(6).AlignItems(ui.Center).Role(ui.RoleStatus).
		Children(func() {
			ui.Icon(c, iconWarning).FontSize(14).TextColor(l.theme.Warning)
			ui.Text(c, a.notice.text).Grow(1).Basis(0)
		})
}

// alerts shows the alert asked for (6.7).
func (a *app) alerts(c *ui.Context) {
	switch a.asking {
	case askingQuit:
		open := true
		switch ui.AlertDialog(c, &open, a.lang.T("quit.title"), a.quitBody(), a.lang.T("quit.confirm"), a.lang.T("common.cancel")) {
		case 0:
			a.confirmQuit()
		case 1, ui.AlertDismissed:
			a.asking = askingNone
		}
	case askingClearHistory:
		open := true
		n := len(a.history.Entries)
		switch ui.AlertDialog(c, &open, a.lang.P("history.clear.title", n), a.lang.T("history.clear.body"),
			a.lang.T("history.clear.confirm"), a.lang.T("common.cancel")) {
		case 0:
			a.clearHistory()
		case 1, ui.AlertDismissed:
			a.asking = askingNone
		}
	}
}

// quitBody follows the queue: how many convert, and whether files wait.
func (a *app) quitBody() string {
	body := a.lang.P("quit.body", model.Count[model.Converting](&a.queue))
	if model.Count[model.Queued](&a.queue) > 0 {
		sep := " "
		if a.lang != "en" {
			sep = ""
		}
		body += sep + a.lang.T("quit.body.queued")
	}
	return body
}

// fatalView is the only screen of an app whose core does not match
// (6.9).
func (a *app) fatalView(c *ui.Context) {
	l := &a.look
	if c.Shortcut(ui.Cmd, ui.KeyQ) {
		a.desk.quit()
	}
	ui.Column(c).Fill().Center().Gap(12).Padding(40).Background(l.theme.Background).Children(func() {
		ui.Text(c, a.lang.T("fatal.abi.title")).FontSize(l.size(17)).FontWeight(600).Role(ui.RoleHeading).Level(1)
		ui.Text(c, a.lang.T("fatal.abi.body")).TextAlign(ui.Center) // primary text (DESIGN 12)
		ui.PrimaryButton(c.Key("fatal.quit"), a.lang.T("quit.confirm")).AutoFocus().OnClick(a.desk.quit)
	})
}
