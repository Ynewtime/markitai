package main

import "github.com/egoist/mygo/ui"

// middleText draws s on one line, cut in its middle where it does not
// fit, as file names are (section 14): mygo cuts text only at its end.
func middleText(c *ui.Context, s string, size float32, color ui.Color) ui.Element {
	_, h := c.MeasureText(0, ui.Span{Text: "Ag", Size: size})
	return ui.Box(c).Height(h).MinWidth(0).Grow(1).Draw(func(p *ui.Painter, r ui.Rect) {
		p.Text(r.X, r.Y, middleCut(s, r.W, func(x string) float32 {
			w, _ := p.MeasureText(0, ui.Span{Text: x, Size: size})
			return w
		}), size, color)
	})
}

// middleCut keeps the most of s's start and end that fits width, with an
// ellipsis between them.
func middleCut(s string, width float32, measure func(string) float32) string {
	if measure(s) <= width {
		return s
	}
	runes := []rune(s)
	cut := func(k int) string { return string(runes[:(k+1)/2]) + "…" + string(runes[len(runes)-k/2:]) }
	lo, hi := 0, len(runes)-1
	for lo < hi {
		mid := (lo + hi + 1) / 2
		if measure(cut(mid)) <= width {
			lo = mid
		} else {
			hi = mid - 1
		}
	}
	return cut(lo)
}
