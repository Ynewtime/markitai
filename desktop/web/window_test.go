package main

import "testing"

func TestFitSizeKeepsTheWindowWithinTheWorkArea(t *testing.T) {
	for _, c := range []struct {
		name                  string
		areaWidth, areaHeight int
		width, height         int
	}{
		{"a screen larger than the window", 1920, 1080, startWidth, startHeight},
		// Xvfb's 1280 by 800, with no window manager: 860 would put its bottom 60 below the screen.
		{"a screen 800 high", 1280, 800, 1280, 800},
		{"a work area under a panel", 1366, 736, 1280, 736},
		{"no work area told", 0, 0, startWidth, startHeight},
		{"one smaller than the window's smallest size", 320, 240, minWidth, minHeight},
	} {
		width, height := fitSize(startWidth, startHeight, c.areaWidth, c.areaHeight)
		if width != c.width || height != c.height {
			t.Errorf("%s: %dx%d, want %dx%d", c.name, width, height, c.width, c.height)
		}
	}
}
