package main

import (
	"testing"

	"github.com/egoist/mygo"
)

func TestFitWindow(t *testing.T) {
	main := mygo.Rectangle{X: 0, Y: 0, Width: 710, Height: 640}
	right := mygo.Rectangle{X: 710, Y: -200, Width: 1920, Height: 1040}
	for _, c := range []struct {
		name     string
		in, want mygo.WindowState
	}{
		{"inside stays", mygo.WindowState{X: 10, Y: 20, Width: 640, Height: 480}, mygo.WindowState{X: 10, Y: 20, Width: 640, Height: 480}},
		{"first launch centered off the top", mygo.WindowState{X: -244, Y: -69, Width: 1200, Height: 780}, mygo.WindowState{X: 0, Y: 0, Width: 710, Height: 640}},
		{"a corner showing", mygo.WindowState{X: -700, Y: 590, Width: 800, Height: 600}, mygo.WindowState{X: 0, Y: 40, Width: 710, Height: 600}},
		{"larger than the screen", mygo.WindowState{X: -1000, Y: -500, Width: 1500, Height: 1000}, mygo.WindowState{X: 0, Y: 0, Width: 710, Height: 640}},
		{"on another display", mygo.WindowState{X: 2400, Y: 700, Width: 400, Height: 300}, mygo.WindowState{X: 2230, Y: 540, Width: 400, Height: 300}},
		{"on no display", mygo.WindowState{X: -5000, Y: -5000, Width: 800, Height: 600}, mygo.WindowState{X: 0, Y: 0, Width: 710, Height: 600}},
		{"maximized keeps its flag", mygo.WindowState{X: 0, Y: 0, Width: 710, Height: 640, Maximized: true}, mygo.WindowState{X: 0, Y: 0, Width: 710, Height: 640, Maximized: true}},
	} {
		if got := fitWindow(c.in, []mygo.Rectangle{main, right}); got != c.want {
			t.Errorf("%s: %+v, want %+v", c.name, got, c.want)
		}
	}
	if w, h := firstSize(main); w != 710 || h != 640 {
		t.Errorf("first size in 710x640: %dx%d", w, h)
	}
	if w, h := firstSize(mygo.Rectangle{Width: 2560, Height: 1400}); w != firstWidth || h != firstHeight {
		t.Errorf("first size on a large screen: %dx%d", w, h)
	}
}

// TestKeepFrame: a zoomed state whose frame covers its work area is saved
// by the frame (macOS, the review's 1728x994 at 0,33); a window zoomed from
// a smaller frame, and one full screen, keep their flags.
func TestKeepFrame(t *testing.T) {
	areas := []mygo.Rectangle{{X: 0, Y: 33, Width: 1728, Height: 994}, {X: 1728, Y: -400, Width: 2560, Height: 1415}}
	for _, c := range []struct {
		name     string
		in, want mygo.WindowState
	}{
		{"fills the work area", mygo.WindowState{X: 0, Y: 33, Width: 1728, Height: 994, Maximized: true}, mygo.WindowState{X: 0, Y: 33, Width: 1728, Height: 994}},
		{"fills the second display's", mygo.WindowState{X: 1728, Y: -400, Width: 2560, Height: 1415, Maximized: true}, mygo.WindowState{X: 1728, Y: -400, Width: 2560, Height: 1415}},
		{"zoomed from a smaller frame", mygo.WindowState{X: 100, Y: 120, Width: 900, Height: 640, Maximized: true}, mygo.WindowState{X: 100, Y: 120, Width: 900, Height: 640, Maximized: true}},
		{"full screen", mygo.WindowState{X: 0, Y: 33, Width: 1728, Height: 994, Maximized: true, FullScreen: true}, mygo.WindowState{X: 0, Y: 33, Width: 1728, Height: 994, Maximized: true, FullScreen: true}},
		{"not zoomed", mygo.WindowState{X: 0, Y: 33, Width: 1728, Height: 994}, mygo.WindowState{X: 0, Y: 33, Width: 1728, Height: 994}},
	} {
		if got := keepFrame(c.in, areas); got != c.want {
			t.Errorf("%s: %+v, want %+v", c.name, got, c.want)
		}
	}
}
