package main

import (
	"time"

	"github.com/egoist/mygo/ui"

	"markitai.local/mygo/internal/l10n"
	"markitai.local/mygo/internal/model"
)

// viewPill is the identity of the view switch's chosen-segment pill,
// which flies between the segments (section 7).
type viewPill struct{}

// detailPane is the detail of 6.4: its bar, content and footer.
func (a *app) detailPane(c *ui.Context, t *target, stacked bool) {
	l := &a.look
	a.detailBar(c, t, stacked)
	if t == nil {
		ui.Column(c).Grow(1).Basis(0).Center().Padding(24).Children(func() {
			ui.Text(c, a.lang.T("detail.empty")).TextColor(l.theme.TextMuted).TextAlign(ui.Center)
		})
		return
	}
	d, done := t.done()
	ui.Column(c).Grow(1).Basis(0).Padding(24).Gap(16).AlignItems(ui.Stretch).Children(func() {
		if stacked {
			// The list column, which shows the notice side by side, is
			// another page here.
			a.noticeLine(c, a.page)
		}
		if t.missing {
			ui.Text(c, a.lang.T("detail.missing.title")).FontSize(l.size(17)).FontWeight(600).Role(ui.RoleHeading).Level(2)
			// Informative text is primary text (DESIGN 12).
			ui.Text(c, a.lang.T("detail.missing.body", "path", a.abbreviated(t.state.(model.Done).Result.MarkdownPath))).Selectable()
			return
		}
		a.detailHeader(c, t)
		switch s := t.state.(type) {
		case model.Done:
			if !done {
				return
			}
			a.warnings(c, t.id, s.Result.Warnings)
			a.body(c, t, d)
		case model.Converting:
			ui.Text(c, a.lang.T("detail.converting.note")).TextColor(l.theme.TextMuted)
		case model.Queued:
			ui.Text(c, a.lang.T("detail.queued")).TextColor(l.theme.TextMuted)
		case model.Failed:
			a.failure(c, t, s)
		}
	})
	if done {
		a.footer(c, d)
	}
}

// detailBar is the detail's toolbar: Back in the stacked layout, the view
// switch and the actions of a Done item.
func (a *app) detailBar(c *ui.Context, t *target, stacked bool) {
	l := &a.look
	_, done := t.done()
	missing := t != nil && t.missing
	ui.Toolbar(c, func() {
		if stacked {
			if ui.Button(c.Key("detail.back"), "").Label(a.lang.T("common.back")).Height(32).Children(func() {
				ui.Row(c).Gap(4).AlignItems(ui.Center).Children(func() {
					ui.Icon(c, iconChevron).FontSize(16).Rotate(180)
					ui.Text(c, a.lang.T("common.back"))
				})
			}).Clicked() {
				a.router.Back()
			}
		}
		ui.Spacer(c)
		if done {
			a.viewSwitch(c)
		}
		ui.Spacer(c)
		if done || missing {
			copyIcon := iconCopy
			if a.copied {
				if a.copiedAt.IsZero() || a.copiedAt.After(c.Now()) {
					a.copiedAt = c.Now()
				}
				if left := 1500*time.Millisecond - c.Now().Sub(a.copiedAt); left > 0 {
					copyIcon = iconCheck
					c.After(left)
				} else {
					a.copied = false
				}
			}
			label := a.lang.T("detail.copy")
			if ui.Button(c.Key("detail.copy"), "").Label(label).Tooltip(label).Disabled(!done).Size(32, 32).Padding(0).Children(func() {
				ui.Icon(c, copyIcon).FontSize(16)
			}).Clicked() {
				a.copyMarkdown(c)
			}
			if a.iconButton(c, "detail.save_as", a.lang.T("detail.save_as"), iconSaveAs, !done).Clicked() {
				a.saveAs()
			}
			if a.iconButton(c, "detail.show", a.showLabel(), iconFolder, !done).Clicked() {
				a.showInFolder()
			}
		}
	}).Label(a.lang.T("a11y.actions")).Height(52).PaddingX(16).Shrink(0).BorderWidth(0, 0, 1, 0).BorderColor(l.rule)
}

// viewSwitch is the Preview and Markdown switch: a segmented control whose
// chosen-segment pill moves between the segments.
func (a *app) viewSwitch(c *ui.Context) {
	l := &a.look
	seg := ui.SegmentedBase(c.Key("detail.view"), (*int)(&a.detail), 2)
	// The pill shares the chosen segment's grid cell and is built first:
	// an Absolute pill would paint over the labels (ui/paint.go:360-372).
	seg.Track.Label(a.lang.T("detail.view")).Height(32).Padding(2).Radius(6).Background(l.theme.Surface).Border(1, l.theme.Border).
		Grid().Columns(2).RowTracks(ui.Fr(1)).Children(func() {
		ui.Box(c.Key("pill")).ColumnStart(int(a.detail)+1).RowStart(1).Radius(4).Background(l.theme.Background).
			Shadow(0, 1, 2, 0, ui.RGBA(0, 0, 0, 0.12)).PassThrough().
			Transition(ui.ElementTransition{Animation: &ui.Snappy, Match: viewPill{}})
		for i, key := range []string{"detail.preview", "detail.markdown"} {
			label := a.lang.T(key)
			seg.Segment(i).ColumnStart(i + 1).RowStart(1).PaddingX(12).Radius(4).Label(label).AlignItems(ui.Center).Justify(ui.Center).
				Children(func() { ui.Text(c, label).SingleLine() })
		}
	})
}

// detailHeader is the title, the source line and the status line.
func (a *app) detailHeader(c *ui.Context, t *target) {
	l := &a.look
	ui.Column(c).Gap(4).Shrink(0).Children(func() {
		ui.Text(c, t.title).FontSize(l.size(17)).FontWeight(600).Selectable().Role(ui.RoleHeading).Level(2)
		ui.Row(c).Gap(6).AlignItems(ui.Center).Children(func() {
			kind, src := iconFile, a.abbreviated(t.source.Value)
			if t.source.Kind == model.URL {
				kind, src = iconGlobe, t.source.Value
			}
			ui.Icon(c, kind).FontSize(14).TextColor(l.theme.TextMuted)
			ui.Text(c, src).FontSize(l.b - 2).TextColor(l.theme.TextMuted).SingleLine().Selectable().Grow(1).Basis(0)
		})
		ui.Row(c).Gap(6).AlignItems(ui.Center).Children(func() {
			switch s := t.state.(type) {
			case model.Done:
				ui.Icon(c, iconDone).FontSize(16).TextColor(l.theme.Success)
				text := a.lang.T("state.done", "duration", a.lang.Duration(s.Result.Duration))
				if t.entry {
					text += " · " + a.lang.Finished(t.finished, c.Now())
				}
				ui.Text(c, text)
			case model.Converting:
				elapsed := max(c.Now().Sub(s.StartedAt), 0)
				c.After(time.Second - elapsed%time.Second)
				ui.Spinner(c).Label(a.lang.T("state.converting"))
				ui.Text(c, a.lang.T("detail.converting", "elapsed", l10n.Elapsed(int(elapsed.Seconds()))))
			case model.Queued:
				ui.Icon(c, iconQueued).FontSize(16).TextColor(l.theme.TextMuted)
				ui.Text(c, a.lang.T("state.queued"))
			case model.Failed:
				ui.Icon(c, iconFailed).FontSize(16).TextColor(l.theme.Danger)
				ui.Text(c, a.lang.T("state.failed"))
			}
		})
	})
}

// warnings is the disclosure of 6.4, collapsed by default and kept per
// item for the session.
func (a *app) warnings(c *ui.Context, id model.ItemID, warnings []string) {
	if len(warnings) == 0 {
		return
	}
	l := &a.look
	open := a.warningsOpen[id]
	if open == nil {
		open = new(bool)
		a.warningsOpen[id] = open
	}
	ui.Column(c.Key("warnings")).Shrink(0).Radius(12).Background(l.theme.Surface).Padding(12, 16).
		Role(ui.RoleGroup).Label(a.lang.T("a11y.warnings")).AlignItems(ui.Stretch).Children(func() {
		w := ui.CollapsibleBase(c, open)
		w.Trigger.Row().Gap(6).AlignItems(ui.Center).Expanded(*open).Children(func() {
			ui.Icon(c, iconWarning).FontSize(14).TextColor(l.theme.Warning)
			ui.Text(c, a.lang.P("state.warnings", len(warnings))).Grow(1).Basis(0)
			ui.Icon(c, iconChevron).FontSize(14).TextColor(l.theme.TextMuted).Rotate(90 * w.Progress())
		})
		w.Panel(func() {
			ui.Column(c).Gap(4).PaddingY(8).Children(func() {
				for _, s := range warnings {
					ui.Text(c, s).FontSize(l.b - 2).Selectable()
				}
			})
		})
	})
}

// body is a Done item's Preview or Markdown.
func (a *app) body(c *ui.Context, t *target, d model.Done) {
	l := &a.look
	if a.detail == showPreview {
		a.preview(c, t, d)
		return
	}
	text := d.Result.Markdown
	ui.TextArea(c.Key("markdown"), &text).ReadOnly(true).Font("monospace").FontSize((l.b - 1) * float32(a.settings.Zoom)).
		Label(a.lang.T("a11y.markdown_text")).Padding(16).Grow(1).Basis(0)
}

// footer is a Done item's duration, and its tokens and cost when AI ran.
func (a *app) footer(c *ui.Context, d model.Done) {
	l := &a.look
	text := a.lang.T("footer.duration", "duration", a.lang.Duration(d.Result.Duration))
	if u := d.Result.Usage; u != nil {
		text += " · " + a.lang.T("footer.tokens", "count", a.lang.Grouped(u.InputTokens+u.OutputTokens))
		switch u.Cost {
		case model.CostComplete:
			text += " · " + l10n.USD(u.CostUSD)
		case model.CostPartial:
			text += " · " + a.lang.T("footer.cost.at_least", "cost", l10n.USD(u.CostUSD))
		default:
			text += " · " + a.lang.T("footer.cost.unknown")
		}
	}
	ui.Row(c).Height(28).PaddingX(24).AlignItems(ui.Center).Shrink(0).BorderWidth(1, 0, 0, 0).BorderColor(l.rule).Children(func() {
		ui.Text(c, text).FontSize(l.b - 2).TextColor(l.theme.TextMuted).SingleLine()
	})
}

// failure is the error block of section 12: the sentence, Retry and the
// code's second button, then the raw message.
func (a *app) failure(c *ui.Context, t *target, f model.Failed) {
	l := &a.look
	ui.Text(c, a.sentence(f))
	_, second := l10n.ErrorKey(f.Code)
	ui.Row(c).Gap(8).Children(func() {
		if ui.Button(c.Key("detail.retry"), a.lang.T("detail.retry")).Clicked() {
			a.retry(t.id)
		}
		if second == l10n.NoAction {
			return
		}
		label := a.lang.T(string(second), "settings", a.lang.T("nav.settings"))
		if ui.Button(c.Key("detail.second"), label).Clicked() {
			switch second {
			case l10n.RemoveItem:
				a.remove(t.id)
			case l10n.OpenSettings:
				a.show(pathSettings)
			case l10n.OpenConfig:
				a.openConfig()
			}
		}
	})
	ui.Text(c, a.lang.T("detail.details")).FontSize(l.b - 2).TextColor(l.theme.TextMuted)
	ui.Column(c).Radius(12).Background(l.theme.Surface).Padding(12, 16).Children(func() {
		ui.Text(c, f.Message).Font("monospace").FontSize(l.b - 1).Selectable()
	})
}
