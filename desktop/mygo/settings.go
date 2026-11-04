package main

import (
	"sync"

	"github.com/egoist/mygo/ui"

	"markitai.local/mygo/internal/brand"
	"markitai.local/mygo/internal/model"
)

var appIcon = sync.OnceValue(func() *ui.Bitmap {
	b, err := ui.DecodeBitmap(brand.Icon)
	if err != nil {
		return nil
	}
	return b
})

// settingsPage is the Settings page of 6.6.
func (a *app) settingsPage(c *ui.Context) {
	l := &a.look
	ui.Scroll(c.Key("settings")).Fill().Background(l.theme.Background).Children(func() {
		ui.Column(c).FillWidth().AlignItems(ui.Center).Children(func() {
			ui.Column(c).FillWidth().MaxWidth(640).Padding(24).Gap(24).AlignItems(ui.Stretch).Children(func() {
				ui.Text(c, a.lang.T("nav.settings")).FontSize(l.size(17)).FontWeight(600).Role(ui.RoleHeading).Level(1)
				a.noticeLine(c, pathSettings)
				a.conversionGroup(c)
				a.outputGroup(c)
				a.configGroup(c)
				a.aboutGroup(c)
			})
		})
	})
}

// group is a heading above a box whose rows rule separates.
func (a *app) group(c *ui.Context, key string, rows ...func()) {
	l := &a.look
	ui.Column(c).Gap(8).AlignItems(ui.Stretch).Children(func() {
		ui.Text(c, a.lang.T(key)).FontWeight(600).Role(ui.RoleHeading).Level(2)
		ui.Column(c).Radius(12).Background(l.theme.Surface).PaddingX(16).AlignItems(ui.Stretch).Children(func() {
			for i, row := range rows {
				if i > 0 {
					ui.Box(c).Height(1).Background(l.rule)
				}
				row()
			}
		})
	})
}

// settingRow is a title and its note on the left, a control on the
// right. With the control disabled, which dims it, the title and note dim
// too, so the whole row reads as off, as GNOME's insensitive rows do.
func (a *app) settingRow(c *ui.Context, title, note string, disabled bool, control func()) {
	l := &a.look
	ui.Row(c).MinHeight(52).PaddingY(10).Gap(12).AlignItems(ui.Center).Children(func() {
		text := ui.Column(c).Grow(1).Basis(0).Gap(2)
		if disabled {
			text.Opacity(0.5)
		}
		text.Children(func() {
			ui.Text(c, title)
			if note != "" {
				ui.Text(c, note).FontSize(l.b - 2).TextColor(l.theme.TextMuted)
			}
		})
		control()
	})
}

func (a *app) conversionGroup(c *ui.Context) {
	s := &a.settings
	var rows []func()
	for _, sw := range []struct {
		key   string
		on    *bool
		needs bool
	}{
		{"settings.llm", &s.LLM, false},
		{"settings.ocr", &s.OCR, false},
		{"settings.screenshot", &s.Screenshot, false},
		{"settings.desc", &s.Desc, true},
		{"settings.alt", &s.Alt, true},
	} {
		rows = append(rows, func() {
			off := sw.needs && !s.LLM
			a.settingRow(c, a.lang.T(sw.key), a.lang.T(sw.key+".note"), off, func() {
				if ui.Switch(c.Key(sw.key), sw.on).Label(a.lang.T(sw.key)).Disabled(off).Changed() {
					a.saveSettings()
				}
			})
		})
	}
	a.group(c, "settings.conversion", rows...)
}

func (a *app) outputGroup(c *ui.Context) {
	l := &a.look
	a.group(c, "settings.output", func() {
		ui.Row(c).MinHeight(52).PaddingY(10).Gap(12).AlignItems(ui.Center).Children(func() {
			ui.Column(c).Grow(1).Basis(0).Gap(2).Children(func() {
				ui.Text(c, a.lang.T("settings.output.folder"))
				ui.Text(c, a.lang.T("settings.output.note")).FontSize(l.b - 2).TextColor(l.theme.TextMuted)
				ui.Text(c, a.abbreviated(a.output())).FontSize(l.b - 2).Selectable()
			})
			if ui.Button(c.Key("settings.output.change"), a.lang.T("settings.output.change")).Clicked() {
				a.chooseOutput()
			}
			if ui.Button(c.Key("settings.output.show"), a.lang.T("settings.output.show")).Clicked() {
				a.showOutput()
			}
		})
	})
}

func (a *app) configGroup(c *ui.Context) {
	l := &a.look
	if !a.config.read {
		a.config.file, a.config.err = a.core.ConfigFile()
		a.config.read = true
	}
	cfg, err := a.config.file, a.config.err
	userNote := a.lang.T("settings.config.user.missing")
	if err == nil && cfg.Exists {
		userNote = a.abbreviated(cfg.Path)
	}
	a.group(c, "settings.config", func() {
		before := a.settings.Config
		ui.RadioGroup(c.Key("settings.config"), func() {
			for _, choice := range []struct {
				key, note string
				value     model.ConfigSource
			}{
				{"settings.config.user", userNote, model.ConfigUser},
				{"settings.config.builtin", a.lang.T("settings.config.builtin.note"), model.ConfigBuiltin},
			} {
				ui.Radio(c.Key(choice.key), &a.settings.Config, choice.value, a.lang.T(choice.key))
				ui.Text(c, choice.note).FontSize(l.b - 2).TextColor(l.theme.TextMuted).PaddingStart(26).Selectable()
			}
		}).Label(a.lang.T("settings.config")).Gap(4).PaddingY(12)
		if a.settings.Config != before {
			a.saveSettings()
		}
	}, func() {
		key := "settings.config.create"
		if err == nil && cfg.Exists {
			key = "settings.config.open"
		}
		ui.Row(c).MinHeight(52).AlignItems(ui.Center).Children(func() {
			if ui.Button(c.Key("settings.config.file"), a.lang.T(key)).Clicked() {
				a.openConfig()
			}
		})
	})
}

func (a *app) aboutGroup(c *ui.Context) {
	l := &a.look
	value := func(key, v string) {
		ui.Row(c).Gap(8).Children(func() {
			ui.Text(c, a.lang.T(key)).TextColor(l.theme.TextMuted)
			ui.Text(c, v).Selectable()
		})
	}
	a.group(c, "settings.about", func() {
		ui.Row(c).PaddingY(16).Gap(16).AlignItems(ui.Center).Children(func() {
			if icon := appIcon(); icon != nil {
				ui.Image(c, icon).Size(48, 48)
			}
			ui.Column(c).Gap(4).Children(func() {
				ui.Text(c, a.lang.T("app.name.mygo")).FontSize(l.size(17)).FontWeight(600)
				value("settings.about.app", a.version)
				value("settings.about.core", a.coreVersion)
				ui.Text(c, a.lang.T("settings.about.copyright")).FontSize(l.b - 2).TextColor(l.theme.TextMuted)
			})
		})
	})
}
