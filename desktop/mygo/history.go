package main

import (
	"github.com/egoist/mygo/ui"

	"markitai.local/mygo/internal/model"
)

// historyPage is the History page of 6.5.
func (a *app) historyPage(c *ui.Context, item bool) {
	l := &a.look
	if len(a.history.Entries) == 0 {
		ui.Column(c).Fill().Background(l.theme.Background).AlignItems(ui.Stretch).Children(func() {
			a.historyHeader(c)
			ui.Column(c).Grow(1).Basis(0).Center().Gap(8).Padding(24).Children(func() {
				ui.Text(c, a.lang.T("history.empty.title")).FontSize(l.size(22)).FontWeight(600).TextAlign(ui.Center)
				ui.Text(c, a.lang.T("history.empty.body")).TextColor(l.theme.TextMuted).TextAlign(ui.Center)
				a.noticeLine(c, pathHistory)
			})
		})
		return
	}
	a.listAndDetail(c, item, func(stacked bool) { a.historyColumn(c, stacked) }, func(stacked bool) {
		a.detailPane(c, a.target(), stacked)
	})
}

// historyHeader is the page title and Clear History, disabled with
// nothing to clear (6.5).
func (a *app) historyHeader(c *ui.Context) {
	a.header(c, a.lang.T("nav.history"), func() {
		label := a.lang.T("history.clear")
		if a.iconButton(c, "history.clear", label, iconTrash, len(a.history.Entries) == 0).Clicked() {
			a.asking = askingClearHistory
		}
	})
}

func (a *app) historyColumn(c *ui.Context, stacked bool) {
	a.historyHeader(c)
	ui.Column(c).Grow(1).Basis(0).Padding(12).Gap(8).AlignItems(ui.Stretch).Children(func() {
		search := a.lang.T("history.search")
		field := ui.SearchField(c.Key("history.search"), &a.query).Label(search).Placeholder(search).FillWidth()
		if !a.leaving {
			field.Bind(&a.searchField)
		}
		a.noticeLine(c, pathHistory)
		a.historyRows(c, stacked)
	})
}

// historyRows is the History list: a ui.List whose choice is held by the
// entry's ID, so that it follows the entry through searches.
func (a *app) historyRows(c *ui.Context, stacked bool) {
	if a.leaving && !stacked {
		// One List a frame shows a ListState (ui/list.go:474). Side by
		// side, the page shown has the list too: a window that widens
		// while the stacked pages move builds both that way.
		ui.Box(c).Grow(1).Basis(0)
		return
	}
	l := &a.look
	entries := a.filtered()
	a.historySel = -1
	for i, e := range entries {
		if e.ID == a.historyID {
			a.historySel = i
		}
	}
	a.historyList.Selected = &a.historySel
	a.historyList.Key = func(i int) any { return entries[i].ID }
	a.historyList.Label = func(i int) string { return entries[i].Title + ", " + a.entrySubtitle(c, entries[i]) }
	clicked := -1
	ui.Themed(c, &l.quiet).Background(ui.Transparent).Grow(1).Basis(0).AlignItems(ui.Stretch).Children(func() {
		list := ui.List(c.Key("history.list"), &a.historyList, len(entries), func(i int) {
			if a.entryRow(c, entries[i], i == a.historySel).Clicked() {
				clicked = i
			}
		}).Grow(1).Basis(0).Label(a.lang.T("nav.history")).Children(func() {
			if len(entries) == 0 {
				ui.Text(c, a.lang.T("history.no_results", "query", a.query)).TextColor(l.theme.TextMuted).Padding(12)
			}
		})
		mods, key := removeKey(a.goos)
		if list.Shortcut(mods, key) {
			a.run(c, actRemove)
		}
		chosen := a.historySel >= 0 && a.historySel < len(entries)
		switch {
		case clicked >= 0:
			// The row takes the click, as a Convert row does: chosen
			// already or not, it opens its page in the stacked layout,
			// where Up and Down only move the choice (6.10).
			a.historyID = entries[clicked].ID
			list.Focus()
			if stacked {
				a.router.Push(pathHistoryItem)
			}
		case list.Changed() && chosen:
			a.historyID = entries[a.historySel].ID
		case list.Submitted() && chosen && stacked:
			a.historyID = entries[a.historySel].ID
			a.router.Push(pathHistoryItem)
		}
	})
}

// entryRow is a History row, drawn as a Convert row (6.5).
func (a *app) entryRow(c *ui.Context, e model.Entry, selected bool) ui.Element {
	l := &a.look
	row := ui.Row(c).Height(44).PaddingX(10).Gap(8).AlignItems(ui.Center).FillWidth()
	a.leadingBar(row, selected)
	row.ContextMenu(func(m *ui.Menu) { a.entryMenu(c, m, e) })
	row.Children(func() {
		kind := iconFile
		if e.Kind == model.URL {
			kind = iconGlobe
		}
		ui.Icon(c, kind).FontSize(16).TextColor(l.theme.TextMuted)
		ui.Column(c).Grow(1).Basis(0).Gap(1).Children(func() {
			if e.Kind == model.File {
				middleText(c, e.Title, l.b, l.theme.Text)
			} else {
				ui.Text(c, e.Title).SingleLine()
			}
			ui.Text(c, a.entrySubtitle(c, e)).FontSize(l.b - 2).SingleLine().TextColor(l.theme.TextMuted)
		})
		if len(e.Warnings) > 0 {
			ui.Icon(c, iconWarning).FontSize(14).TextColor(l.theme.Warning)
		}
	})
	return row
}

func (a *app) entrySubtitle(c *ui.Context, e model.Entry) string {
	text := a.lang.Finished(e.FinishedAt, c.Now())
	if n := len(e.Warnings); n > 0 {
		text += " · " + a.lang.P("state.warnings", n)
	}
	return text
}

// entryMenu is a History row's context menu (DESIGN 6.5).
func (a *app) entryMenu(c *ui.Context, m *ui.Menu, e model.Entry) {
	pick := func(act action) {
		a.historyID = e.ID
		if t := a.target(); t != nil && t.loading {
			a.later = append(a.later, act)
			return
		}
		a.run(c, act)
	}
	md := a.markdowns[e.ID]
	a.doneMenu(m, md != nil && md.missing, pick)
	m.Separator()
	mods, key := removeKey(a.goos)
	if m.Item(a.lang.T("history.remove")).Shortcut(mods, key).Chosen() {
		a.removeEntry(e.ID)
	}
}
