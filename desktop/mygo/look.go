package main

import "github.com/egoist/mygo/ui"

// look is the frame's theme (section 5): the window's, the quiet copy the
// sidebar and the lists build with, and the region fills.
type look struct {
	theme, quiet                                       ui.Theme
	sidebarFill, listFill, rule, rowHover, chosenQuiet ui.Color
	// b is the body size; scale the desktop's text scale for the others.
	b, scale float32
}

// makeLook builds the look from the system's theme of the frame, which
// follows the appearance, high contrast and text size.
func makeLook(system ui.Theme, p ui.Preferences) look {
	t := system
	l := look{scale: p.TextScale}
	if l.scale <= 0 {
		l.scale = 1
	}
	focus := float32(0.55)
	if t.Dark {
		t.Background, t.Surface, t.SurfaceHover, t.SurfacePressed = ui.Hex("#18181b"), ui.Hex("#27272a"), ui.Hex("#3f3f46"), ui.Hex("#52525b")
		t.Border, t.Text, t.TextMuted = ui.Hex("#3f3f46"), ui.Hex("#fafafa"), ui.Hex("#a1a1aa")
		t.Accent, t.AccentHover, t.AccentPressed = ui.Hex("#71717a"), ui.Hex("#52525b"), ui.Hex("#3f3f46")
		l.sidebarFill, l.listFill, l.rule, l.rowHover, l.chosenQuiet = ui.Hex("#09090b"), ui.Hex("#09090b"), ui.Hex("#27272a"), ui.Hex("#18181b"), ui.Hex("#27272a")
		focus = 0.6
	} else {
		t.Background, t.Surface, t.SurfaceHover, t.SurfacePressed = ui.Hex("#ffffff"), ui.Hex("#f4f4f5"), ui.Hex("#e4e4e7"), ui.Hex("#d4d4d8")
		t.Border, t.Text, t.TextMuted = ui.Hex("#d4d4d8"), ui.Hex("#18181b"), ui.Hex("#52525b")
		t.Accent, t.AccentHover, t.AccentPressed = ui.Hex("#18181b"), ui.Hex("#27272a"), ui.Hex("#3f3f46")
		l.sidebarFill, l.listFill, l.rule, l.rowHover, l.chosenQuiet = ui.Hex("#f4f4f5"), ui.Hex("#fafafa"), ui.Hex("#e4e4e7"), ui.Hex("#f4f4f5"), ui.Hex("#e4e4e7")
	}
	t.AccentText = ui.RGB(255, 255, 255)
	t.Focus = t.Accent.Alpha(focus)
	t.Radius = 6
	if p.HighContrast {
		t.Border = t.Border.Mix(t.Text, 0.45)
		l.rule = l.rule.Mix(t.Text, 0.45)
		t.TextMuted = t.TextMuted.Mix(t.Text, 0.4)
		t.Focus = t.Accent
	}
	l.theme = t
	q := t
	q.Accent, q.AccentHover, q.AccentPressed, q.SurfacePressed = l.chosenQuiet, l.chosenQuiet, l.chosenQuiet, l.chosenQuiet
	q.AccentText = t.Text
	q.SurfaceHover = l.rowHover
	l.quiet = q
	l.b = t.FontSize
	return l
}

// size scales a type size of section 5.3 with the desktop's text size.
func (l *look) size(v float32) float32 { return v * l.scale }
