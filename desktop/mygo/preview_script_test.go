package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"testing"
)

// pageStep is an event for the preview's script: a DOM event fired at the
// window, or a mygo page event (on).
type pageStep struct {
	Fire  string         `json:"fire,omitempty"`
	Props map[string]any `json:"props,omitempty"`
	On    string         `json:"on,omitempty"`
	Data  any            `json:"data,omitempty"`
}

// pageOutcome is what a step did: whether the script prevented the
// event's default, the drop effect it chose, and the calls it made.
type pageOutcome struct {
	Prevented  bool    `json:"prevented"`
	DropEffect string  `json:"dropEffect"`
	Calls      [][]any `json:"calls"`
}

// pageStandIn runs the preview's script in node over a window with no
// page, and fires the steps of stdin at it. A dataTransfer is given as
// {types, data}.
const pageStandIn = `
const fs = require("node:fs");
const listeners = {}, events = {};
let calls = [];
globalThis.addEventListener = (type, fn) => { (listeners[type] ??= []).push(fn); };
globalThis.document = { readyState: "loading", body: { tabIndex: -1 } };
globalThis.mygo = {
  call: (...args) => { calls.push(args); return Promise.resolve(); },
  on: (name, fn) => { events[name] = fn; },
};
new Function(fs.readFileSync(process.argv[2], "utf8"))();
const out = [];
for (const step of JSON.parse(fs.readFileSync(0, "utf8"))) {
  calls = [];
  const e = { defaultPrevented: false, preventDefault() { this.defaultPrevented = true; }, stopPropagation() {}, ...step.props };
  if (e.dataTransfer) {
    const d = e.dataTransfer;
    e.dataTransfer = { types: d.types, dropEffect: "none", getData: (t) => (d.data ?? {})[t] ?? "" };
  }
  if (step.on) events[step.on]?.(step.data);
  else for (const fn of listeners[step.fire] ?? []) fn(e);
  out.push({ prevented: e.defaultPrevented, dropEffect: e.dataTransfer?.dropEffect ?? "", calls });
}
out.push({ prevented: false, dropEffect: "", calls: [["tabIndex", document.body.tabIndex]] });
console.log(JSON.stringify(out));
`

// runPreviewJS fires steps at previewJS(goos), and returns what each did,
// then the body's tabIndex once the document has loaded.
func runPreviewJS(t *testing.T, goos string, steps []pageStep) []pageOutcome {
	t.Helper()
	node, err := exec.LookPath("node")
	if err != nil {
		t.Skip("node is not installed: the preview's script cannot run")
	}
	dir := t.TempDir()
	script, standIn := filepath.Join(dir, "preview.js"), filepath.Join(dir, "page.js")
	if err := os.WriteFile(script, []byte(previewJS(goos)), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(standIn, []byte(pageStandIn), 0o644); err != nil {
		t.Fatal(err)
	}
	in, _ := json.Marshal(append(steps, pageStep{Fire: "DOMContentLoaded"}))
	cmd := exec.Command(node, standIn, script)
	cmd.Stdin = bytes.NewReader(in)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("node: %v\n%s", err, stderr.String())
	}
	var got []pageOutcome
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatalf("%v: %s", err, out)
	}
	return got
}

func keyDown(code, key string, mod, shift bool, goos string) pageStep {
	return pageStep{Fire: "keydown", Props: map[string]any{
		"code": code, "key": key, "ctrlKey": mod && goos != "darwin", "metaKey": mod && goos == "darwin", "altKey": false, "shiftKey": shift,
	}}
}

func press(act action) [][]any { return [][]any{{"Preview.Press", string(act)}} }

// TestPreviewKeysOnAnyLayout forwards a key by what it types, as the
// native keys take it, and by its place when it types no US key's
// character (DESIGN 8.3): AZERTY's Ctrl+A, at KeyQ's place, is the page's
// Select All, not Quit.
func TestPreviewKeysOnAnyLayout(t *testing.T) {
	type keyCase struct {
		name       string
		code, key  string
		mod, shift bool
		want       [][]any
	}
	for _, goos := range []string{"linux", "windows", "darwin"} {
		cases := []keyCase{
			{"AZERTY select all", "KeyQ", "a", true, false, nil},
			{"AZERTY undo", "KeyW", "z", true, false, nil},
			{"AZERTY add web address", "KeyL", "l", true, false, press(actAddURL)},
			{"AZERTY switch view", "Semicolon", "M", true, true, press(actToggleView)},
			{"AZERTY settings", "KeyM", ",", true, false, press(actSettings)},
			{"AZERTY zoom out", "Digit6", "-", true, false, press(actZoomOut)},
			{"AZERTY Convert", "Digit1", "&", true, false, press(actConvert)},
			{"QWERTZ undo", "KeyY", "z", true, false, nil},
			{"US search", "KeyF", "f", true, false, press(actSearch)},
			{"Russian search", "KeyF", "а", true, false, press(actSearch)},
			{"US zoom in", "Equal", "=", true, false, press(actZoomIn)},
			{"letter alone", "KeyL", "l", false, false, nil},
		}
		if goos == "linux" {
			cases = append(cases, keyCase{"AZERTY quit", "KeyA", "q", true, false, press(actQuit)},
				keyCase{"AZERTY close", "KeyZ", "w", true, false, press(actClose)})
		}
		var steps []pageStep
		for _, c := range cases {
			steps = append(steps, keyDown(c.code, c.key, c.mod, c.shift, goos))
		}
		got := runPreviewJS(t, goos, steps)
		for i, c := range cases {
			if !reflect.DeepEqual(got[i].Calls, c.want) && !(len(got[i].Calls) == 0 && len(c.want) == 0) || got[i].Prevented != (len(c.want) > 0) {
				t.Errorf("%s %s: called %v, prevented %v; want %v", goos, c.name, got[i].Calls, got[i].Prevented, c.want)
			}
		}
	}
}

// TestPreviewMouseButtons makes the back button the app's Back and the
// forward button nothing: the page's history never moves (DESIGN 8.3).
func TestPreviewMouseButtons(t *testing.T) {
	for _, goos := range []string{"linux", "windows", "darwin"} {
		var steps []pageStep
		for _, button := range []int{3, 4, 0} {
			for _, typ := range []string{"mousedown", "mouseup", "auxclick"} {
				steps = append(steps, pageStep{Fire: typ, Props: map[string]any{"button": button}})
			}
		}
		got := runPreviewJS(t, goos, steps)
		for i, button := range []int{3, 4, 0} {
			for j, typ := range []string{"mousedown", "mouseup", "auxclick"} {
				o := got[3*i+j]
				var want [][]any
				if button == 3 && typ == "mouseup" {
					want = press(actBack)
				}
				if o.Prevented != (button != 0) || len(o.Calls) != len(want) || len(want) > 0 && !reflect.DeepEqual(o.Calls, want) {
					t.Errorf("%s: button %d %s prevented %v, called %v", goos, button, typ, o.Prevented, o.Calls)
				}
			}
		}
	}
}

// TestPreviewDrops takes what is dropped on the page to the app, as on the
// list (DESIGN 8.3): links from their URI list, Chromium's text/x-moz-url
// or plain text, and files once mygo has their paths. The page opens
// neither, and keeps a drag of its own.
func TestPreviewDrops(t *testing.T) {
	dt := func(types []string, data map[string]string) map[string]any {
		return map[string]any{"dataTransfer": map[string]any{"types": types, "data": data}}
	}
	link := "https://example.com/a"
	steps := []pageStep{
		{Fire: "dragover", Props: dt([]string{"text/uri-list", "text/plain"}, nil)},
		{Fire: "drop", Props: dt([]string{"text/uri-list", "text/plain"}, map[string]string{"text/uri-list": "# a comment\r\n" + link + "\r\n", "text/plain": link})},
		{Fire: "drop", Props: dt([]string{"text/x-moz-url"}, map[string]string{"text/x-moz-url": link + "\nA page"})},
		{Fire: "drop", Props: dt([]string{"text/plain"}, map[string]string{"text/plain": link})},
		{Fire: "dragover", Props: dt([]string{"Files"}, nil)},
		{Fire: "drop", Props: dt([]string{"Files"}, nil)},
		{On: "mygo:file-drop", Data: map[string]any{"paths": []string{"/tmp/a.pdf"}, "x": 1, "y": 2}},
		{Fire: "dragstart"},
		{Fire: "dragover", Props: dt([]string{"text/uri-list"}, nil)},
		{Fire: "drop", Props: dt([]string{"text/uri-list"}, map[string]string{"text/uri-list": link})},
		{Fire: "dragend"},
		{Fire: "dragover", Props: dt([]string{"text/html"}, nil)},
	}
	drop := func(paths, links []any) [][]any { return [][]any{{"Preview.Drop", paths, links}} }
	want := []pageOutcome{
		{Prevented: true, DropEffect: "copy"},
		{Prevented: true, Calls: drop([]any{}, []any{link})},
		{Prevented: true, Calls: drop([]any{}, []any{link})},
		{Prevented: true, Calls: drop([]any{}, []any{link})},
		{Prevented: true, DropEffect: "copy"},
		{Prevented: true},
		{Calls: drop([]any{"/tmp/a.pdf"}, []any{})},
		{},
		{},
		{},
		{},
		{},
	}
	for _, goos := range []string{"linux", "windows", "darwin"} {
		got := runPreviewJS(t, goos, steps)
		for i, w := range want {
			g := got[i]
			if g.Prevented != w.Prevented || w.DropEffect != "" && g.DropEffect != w.DropEffect || len(g.Calls) != len(w.Calls) || len(w.Calls) > 0 && !reflect.DeepEqual(g.Calls, w.Calls) {
				t.Errorf("%s step %d (%s%s): got %+v, want %+v", goos, i, steps[i].Fire, steps[i].On, g, w)
			}
		}
	}
}

// TestPreviewDocumentFocusable lets Tab stop at the page, whose arrows then
// scroll it, though it has no link (DESIGN 8.3).
func TestPreviewDocumentFocusable(t *testing.T) {
	got := runPreviewJS(t, "linux", nil)
	if last := got[len(got)-1].Calls; !reflect.DeepEqual(last, [][]any{{"tabIndex", float64(0)}}) {
		t.Errorf("the body's tabIndex: %v", last)
	}
}

// TestPreviewWheelZoom hands Ctrl with the wheel to the app on Windows,
// in place of WebView2's own zoom, and leaves the wheel alone without Ctrl
// and on the other systems (DESIGN 8.3).
func TestPreviewWheelZoom(t *testing.T) {
	wheel := func(d int, ctrl bool) pageStep {
		return pageStep{Fire: "wheel", Props: map[string]any{"wheelDeltaY": d, "deltaY": -d / 4, "ctrlKey": ctrl, "metaKey": false}}
	}
	steps := []pageStep{wheel(120, true), wheel(-52, true), wheel(120, false)}
	want := [][][]any{{{"Preview.Wheel", float64(120)}}, {{"Preview.Wheel", float64(-52)}}, nil}
	got := runPreviewJS(t, "windows", steps)
	for i, w := range want {
		if len(got[i].Calls) != len(w) || len(w) > 0 && !reflect.DeepEqual(got[i].Calls, w) || got[i].Prevented != (w != nil) {
			t.Errorf("step %d: called %v, prevented %v; want %v", i, got[i].Calls, got[i].Prevented, w)
		}
	}
	for _, goos := range []string{"linux", "darwin"} {
		if got := runPreviewJS(t, goos, steps[:1]); got[0].Prevented || len(got[0].Calls) > 0 {
			t.Errorf("%s takes the wheel: %+v", goos, got[0])
		}
	}
}
