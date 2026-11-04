package main

import (
	"slices"
	"time"

	"github.com/egoist/mygo/ui"

	"markitai.local/mygo/internal/l10n"
	"markitai.local/mygo/internal/model"
)

// rowMotion fades a row in and out while the rows below slide (section 7).
var rowMotion = ui.ElementTransition{Duration: 150 * time.Millisecond, Ease: ui.EaseOut, Enter: &ui.Motion{}, Exit: &ui.Motion{}}

// dropMotion fades the drop frame's border to the accent.
var dropMotion = ui.ElementTransition{Colors: true, Duration: 120 * time.Millisecond}

func (a *app) convertPage(c *ui.Context, dragging, item bool) {
	if len(a.queue.Items) == 0 {
		a.emptyConvert(c, dragging)
		return
	}
	a.listAndDetail(c, item, func(stacked bool) { a.convertList(c, dragging, stacked) }, func(stacked bool) {
		a.detailPane(c, a.target(), stacked)
	})
}

// dropFrame draws the dashed frame of the drop zone and drop bar, solid in
// the accent while a drag is over the window.
func (a *app) dropFrame(e ui.Element, dragging bool) ui.Element {
	l := &a.look
	e = e.Radius(12).Role(ui.RoleGroup).Label(a.lang.T("a11y.drop_zone")).Transition(dropMotion)
	if dragging {
		return e.Border(1.5, l.theme.Accent)
	}
	return e.Border(1.5, l.theme.Border).BorderStyle(ui.BorderDashed)
}

// emptyConvert is 6.2: the page title over a drop zone 560 wide, or as wide
// as the room, centered in a scroll that reaches its URL row at the
// smallest window.
func (a *app) emptyConvert(c *ui.Context, dragging bool) {
	l := &a.look
	ui.Column(c).Fill().Background(l.theme.Background).AlignItems(ui.Stretch).Children(func() {
		a.header(c, a.lang.T("nav.convert"), func() {})
		// The scroll is as tall as the zone, in the middle of the page, up
		// to the page's height, past which it scrolls: inside a scroll no
		// percentage of its height resolves, so it cannot center the zone
		// itself.
		ui.Column(c).Grow(1).Basis(0).Justify(ui.Center).AlignItems(ui.Stretch).Children(func() {
			ui.Scroll(c.Key("convert.empty")).MaxHeightPercent(100).Children(func() {
				ui.Column(c).FillWidth().Padding(24).AlignItems(ui.Center).Children(func() {
					zone := ui.Column(c.Key("drop")).FillWidth().MaxWidth(560).Padding(40).Gap(12).AlignItems(ui.Center)
					a.dropFrame(zone, dragging).Children(func() {
						ui.Icon(c, iconAddFiles).FontSize(48).TextColor(l.theme.TextMuted)
						heading := a.lang.T("convert.empty.title")
						if dragging {
							heading = a.lang.T("convert.drop.release")
						}
						ui.Text(c, heading).FontSize(l.size(22)).FontWeight(600).TextAlign(ui.Center).ContentTransition(ui.ContentTransition{})
						ui.Text(c, a.lang.T("convert.empty.body", "folder", a.abbreviated(a.output()))).TextColor(l.theme.TextMuted).TextAlign(ui.Center)
						ui.PrimaryButton(c.Key("convert.choose"), a.lang.T("convert.choose")).AutoFocus().OnClick(a.chooseFiles)
						ui.Row(c).FillWidth().Gap(12).AlignItems(ui.Center).Children(func() {
							ui.Box(c).Height(1).Grow(1).Background(l.rule)
							ui.Text(c, a.lang.T("convert.or")).TextColor(l.theme.TextMuted)
							ui.Box(c).Height(1).Grow(1).Background(l.rule)
						})
						a.urlRow(c, false)
						a.noticeLine(c, pathConvert)
					})
				})
			})
		})
	})
}

// urlRow is the URL field and its Add button. The list's row leads with
// the link glyph (6.3); the empty page's has none (6.2).
func (a *app) urlRow(c *ui.Context, glyph bool) {
	l := &a.look
	ui.Column(c).FillWidth().Gap(6).Children(func() {
		ui.Row(c).FillWidth().Gap(8).AlignItems(ui.Center).Children(func() {
			ui.Row(c).Grow(1).Basis(0).Height(32).PaddingX(8).Gap(6).AlignItems(ui.Center).Radius(l.theme.Radius).
				Background(l.theme.Background).Border(1, l.theme.Border).Children(func() {
				if glyph {
					ui.Icon(c, iconLink).FontSize(16).TextColor(l.theme.TextMuted)
				}
				field := ui.TextInputBase(c.Key("convert.url"), &a.url).Grow(1).
					Placeholder(a.lang.T("convert.url.placeholder")).Label(a.lang.T("convert.url.label"))
				if !a.leaving {
					field.Bind(&a.urlField)
				}
				if a.urlInvalid {
					field.Error(a.lang.T("convert.url.invalid"))
				}
				if field.Changed() {
					a.urlInvalid = false
				}
				if field.Submitted() {
					a.addURL()
				}
			})
			if ui.Button(c.Key("convert.url.add"), a.lang.T("convert.url.add")).Height(32).Clicked() {
				a.addURL()
			}
		})
		if a.urlInvalid {
			ui.Row(c).Gap(6).AlignItems(ui.Center).Children(func() {
				ui.Icon(c, iconFailed).FontSize(14).TextColor(l.theme.Danger)
				ui.Text(c, a.lang.T("convert.url.invalid")).Grow(1).Basis(0)
			})
		}
	})
}

// convertList is the list column of 6.3.
func (a *app) convertList(c *ui.Context, dragging, stacked bool) {
	l := &a.look
	a.header(c, a.lang.T("nav.convert"), func() {
		label := a.lang.T("convert.clear_finished")
		if a.iconButton(c, "convert.clear_finished", label, iconClear, !a.queue.AnyFinished()).Clicked() {
			a.clearFinished()
		}
	})
	ui.Column(c).Grow(1).Basis(0).Padding(12).Gap(8).AlignItems(ui.Stretch).Children(func() {
		bar := ui.Row(c.Key("drop")).Height(48).PaddingX(12).Gap(8).AlignItems(ui.Center).Shrink(0)
		a.dropFrame(bar, dragging).Children(func() {
			ui.Icon(c, iconAddFiles).FontSize(18).TextColor(l.theme.TextMuted)
			label := a.lang.T("convert.drop.bar")
			if dragging {
				label = a.lang.T("convert.drop.release")
			}
			ui.Text(c, label).TextColor(l.theme.TextMuted).SingleLine().Grow(1).Basis(0).ContentTransition(ui.ContentTransition{})
			if ui.Button(c.Key("convert.choose"), a.lang.T("convert.choose")).Clicked() {
				a.chooseFiles()
			}
		})
		a.urlRow(c, true)
		a.noticeLine(c, pathConvert)
		a.conversions(c, stacked)
	})
}

// conversions is the Convert list: a scroll of keyed rows, which come and
// go with transitions that ui.List does not give its rows. It is one Tab
// stop whose arrows move the selection.
func (a *app) conversions(c *ui.Context, stacked bool) {
	l := &a.look
	ui.Themed(c, &l.quiet).Background(ui.Transparent).Grow(1).Basis(0).AlignItems(ui.Stretch).Children(func() {
		list := ui.Scroll(c.Key("conversions")).Grow(1).Basis(0).Focusable().Role(ui.RoleList).Label(a.lang.T("a11y.conversions"))
		mods, key := removeKey(a.goos)
		switch {
		case list.Shortcut(0, ui.KeyUp):
			a.queue.Move(-1)
		case list.Shortcut(0, ui.KeyDown):
			a.queue.Move(1)
		case list.Shortcut(0, ui.KeyEnter):
			if stacked && a.queue.Item(a.queue.Selected) != nil {
				a.router.Push(pathConvertItem)
			}
		case list.Shortcut(mods, key):
			a.run(c, actRemove)
		}
		var chosen model.ItemID
		var active ui.Element
		list.Children(func() {
			// A row's context menu may remove it while the rows build.
			for _, it := range slices.Clone(a.queue.Items) {
				selected := it.ID == a.queue.Selected
				row := a.row(c, it, selected)
				if row.Clicked() {
					chosen = it.ID
				}
				if selected {
					active = row
				}
			}
		})
		if active.Valid() {
			list.ActiveDescendant(active)
		}
		if chosen != "" {
			a.queue.Selected = chosen
			list.Focus()
			if stacked {
				a.router.Push(pathConvertItem)
			}
		}
	})
}

// row is one item of the Convert list (6.3).
func (a *app) row(c *ui.Context, it *model.Item, selected bool) ui.Element {
	l := &a.look
	subtitle := a.subtitle(it)
	row := ui.Row(c.Key(string(it.ID))).Height(44).PaddingX(10).Gap(8).Radius(6).AlignItems(ui.Center).Shrink(0).
		Role(ui.RoleListItem).Label(it.Title + ", " + subtitle).Selected(selected).Transition(rowMotion)
	fill := ui.Transparent
	switch {
	case selected:
		fill = l.chosenQuiet
	case row.Hovered():
		fill = l.rowHover
	}
	row.Background(row.AnimateColor("fill", fill, ui.Animation{Duration: 120 * time.Millisecond, Ease: ui.EaseOut})).TextColor(l.theme.Text)
	a.leadingBar(row, selected)
	row.ContextMenu(func(m *ui.Menu) { a.rowMenu(c, m, it) })
	row.Children(func() {
		kind := iconFile
		if it.Source.Kind == model.URL {
			kind = iconGlobe
		}
		ui.Icon(c, kind).FontSize(16).TextColor(l.theme.TextMuted)
		ui.Column(c).Grow(1).Basis(0).Gap(1).Children(func() {
			if it.Source.Kind == model.File {
				middleText(c, it.Title, l.b, l.theme.Text)
			} else {
				ui.Text(c, it.Title).SingleLine()
			}
			ui.Text(c, subtitle).FontSize(l.b - 2).SingleLine().TextColor(l.theme.TextMuted).ContentTransition(ui.ContentTransition{})
		})
		switch s := it.State.(type) {
		case model.Queued:
			ui.Icon(c, iconQueued).FontSize(14).TextColor(l.theme.TextMuted).ContentTransition(ui.ContentTransition{})
		case model.Converting:
			ui.Spinner(c).Label(a.lang.T("state.converting"))
		case model.Done:
			if len(s.Result.Warnings) > 0 {
				ui.Icon(c, iconWarning).FontSize(14).TextColor(l.theme.Warning).ContentTransition(ui.ContentTransition{})
			}
		case model.Failed:
			ui.Icon(c, iconFailed).FontSize(14).TextColor(l.theme.Danger).ContentTransition(ui.ContentTransition{})
		}
	})
	return row
}

// leadingBar draws a chosen row's 2 px bar in the accent (5.1).
func (a *app) leadingBar(e ui.Element, on bool) {
	if !on {
		return
	}
	accent := a.look.theme.Accent
	e.Draw(func(p *ui.Painter, r ui.Rect) { p.Fill(ui.Rect{X: r.X, Y: r.Y, W: 2, H: r.H}, accent, 1) })
}

func (a *app) subtitle(it *model.Item) string {
	switch s := it.State.(type) {
	case model.Queued:
		return a.lang.T("state.queued")
	case model.Converting:
		return a.lang.T("state.converting")
	case model.Done:
		return a.doneText(s.Result.Duration, len(s.Result.Warnings))
	case model.Failed:
		return a.lang.T("row.failed", "reason", a.sentence(s))
	}
	return ""
}

func (a *app) doneText(duration float64, warnings int) string {
	text := a.lang.T("state.done", "duration", a.lang.Duration(duration))
	if warnings > 0 {
		text += " · " + a.lang.P("state.warnings", warnings)
	}
	return text
}

// sentence is a failure's mapped sentence (DESIGN 9).
func (a *app) sentence(f model.Failed) string {
	key, _ := l10n.ErrorKey(f.Code)
	return a.lang.T(key, "settings", a.lang.T("nav.settings"), "folder", a.abbreviated(f.Folder), "reason", f.Message)
}

// showLabel is Show in folder in this OS's words.
func (a *app) showLabel() string {
	switch a.goos {
	case "darwin":
		return a.lang.T("detail.show.macos")
	case "windows":
		return a.lang.T("detail.show.windows")
	}
	return a.lang.T("detail.show.linux")
}

// rowMenu is a Convert row's context menu (DESIGN 6.3). Its items act on
// the row, which they select first.
func (a *app) rowMenu(c *ui.Context, m *ui.Menu, it *model.Item) {
	pick := func(act action) {
		a.queue.Selected = it.ID
		a.run(c, act)
	}
	switch it.State.(type) {
	case model.Done:
		a.doneMenu(m, false, pick)
		m.Separator()
	case model.Failed:
		if m.Item(a.lang.T("detail.retry")).Shortcut(ui.Cmd, ui.KeyR).Chosen() {
			pick(actRetry)
		}
		m.Separator()
	}
	_, busy := it.State.(model.Converting)
	mods, key := removeKey(a.goos)
	if m.Item(a.lang.T("convert.remove")).Shortcut(mods, key).Disabled(busy).Chosen() {
		pick(actRemove)
	}
}

// doneMenu holds Copy, Save As and Show, disabled for a History entry whose
// file is gone.
func (a *app) doneMenu(m *ui.Menu, missing bool, pick func(action)) {
	if m.Item(a.lang.T("detail.copy")).Shortcut(ui.Cmd|ui.Shift, ui.KeyC).Disabled(missing).Chosen() {
		pick(actCopy)
	}
	if m.Item(a.lang.T("detail.save_as")).Shortcut(ui.Cmd|ui.Shift, ui.KeyS).Disabled(missing).Chosen() {
		pick(actSaveAs)
	}
	if m.Item(a.showLabel()).Shortcut(ui.Cmd|ui.Shift, ui.KeyR).Disabled(missing).Chosen() {
		pick(actShow)
	}
}
