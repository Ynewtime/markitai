package main

import (
	"syscall"
	"unsafe"

	"github.com/egoist/mygo"
)

var (
	procGetWindowRect   = syscall.NewLazyDLL("user32.dll").NewProc("GetWindowRect")
	procGetClientRect   = syscall.NewLazyDLL("user32.dll").NewProc("GetClientRect")
	procGetDpiForWindow = syscall.NewLazyDLL("user32.dll").NewProc("GetDpiForWindow")
)

// contentMinimum keeps the content at least 640 by 480 (section 6.10):
// mygo gives Windows MinWidth and MinHeight as the window's least size,
// frame and title bar included (MinTrackSize), so the frame is added. It
// is measured in pixels and rounded up to whole DIPs: Window.Bounds and
// ContentBounds round each size, and at 192 dpi a frame 71 pixels high
// read as 35 DIPs, which left the content 959 pixels high.
func contentMinimum(win *mygo.Window) {
	hwnd := win.NativeHandle()
	var w, c struct{ left, top, right, bottom int32 }
	procGetWindowRect.Call(hwnd, uintptr(unsafe.Pointer(&w)))
	procGetClientRect.Call(hwnd, uintptr(unsafe.Pointer(&c)))
	dpi, _, _ := procGetDpiForWindow.Call(hwnd)
	if dpi == 0 {
		dpi = 96
	}
	dips := func(px int32) int { return (int(px)*96 + int(dpi) - 1) / int(dpi) }
	win.SetMinimumSize(minWidth+dips(w.right-w.left-c.right), minHeight+dips(w.bottom-w.top-c.bottom))
}
