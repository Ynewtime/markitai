package main

import (
	"encoding/json"
	"errors"
	"io/fs"
	"os"

	"github.com/egoist/mygo"
)

// firstWidth and firstHeight are the window's size at its first launch
// (DESIGN 14), as far as the work area allows.
const firstWidth, firstHeight = 1200, 780

// firstSize is the first launch's size in the primary display's work
// area: mygo centers the window there, so it shows whole.
func firstSize(area mygo.Rectangle) (width, height int) {
	width, height = firstWidth, firstHeight
	if area.Width > 0 && area.Height > 0 {
		width, height = min(width, area.Width), min(height, area.Height)
	}
	return width, height
}

// windowStates keeps the window's place in window-state.json in the data
// folder, as mygo's own store does, and gives it back inside the work
// area of the display it is on: mygo restores a place of which a corner
// shows, as a title bar above the screen (DESIGN 14).
type windowStates struct{ path string }

func (s windowStates) LoadWindowStates() (map[string]mygo.WindowState, error) {
	data, err := os.ReadFile(s.path)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	} else if err != nil {
		return nil, err
	}
	var states map[string]mygo.WindowState
	if err := json.Unmarshal(data, &states); err != nil {
		return nil, err // the window gets its options; the next save mends the file
	}
	var areas []mygo.Rectangle
	for _, d := range mygo.Screen.Displays() {
		areas = append(areas, d.WorkArea)
	}
	for key, st := range states {
		states[key] = fitWindow(st, areas)
	}
	return states, nil
}

func (s windowStates) SaveWindowStates(states map[string]mygo.WindowState) error {
	data, err := json.MarshalIndent(states, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(s.path+".tmp", data, 0o644); err != nil {
		return err
	}
	return os.Rename(s.path+".tmp", s.path)
}

// fitWindow puts a saved place inside the work area it overlaps most, the
// first area when it overlaps none: no larger than the area, and moved in
// as far as it reaches out.
func fitWindow(st mygo.WindowState, areas []mygo.Rectangle) mygo.WindowState {
	if len(areas) == 0 || st.Width <= 0 || st.Height <= 0 {
		return st
	}
	area, best := areas[0], 0
	for _, a := range areas {
		w := min(a.X+a.Width, st.X+st.Width) - max(a.X, st.X)
		h := min(a.Y+a.Height, st.Y+st.Height) - max(a.Y, st.Y)
		if w > 0 && h > 0 && w*h > best {
			area, best = a, w*h
		}
	}
	if area.Width <= 0 || area.Height <= 0 {
		return st
	}
	st.Width, st.Height = min(st.Width, area.Width), min(st.Height, area.Height)
	st.X = min(max(st.X, area.X), area.X+area.Width-st.Width)
	st.Y = min(max(st.Y, area.Y), area.Y+area.Height-st.Height)
	return st
}
