package main

import "github.com/egoist/mygo"

// contentMinimum keeps the content at least 640 by 480 (section 6.10):
// mygo gives macOS MinWidth and MinHeight as the window's least frame,
// title bar included (setMinSize:), so the frame's own height is added.
func contentMinimum(win *mygo.Window) {
	b, c := win.Bounds(), win.ContentBounds()
	win.SetMinimumSize(minWidth+b.Width-c.Width, minHeight+b.Height-c.Height)
}
