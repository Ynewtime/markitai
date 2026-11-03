// Command Markitai is the desktop app: the markitai workbench in a native
// window, served by the markitai program bundled with it.
package main

import (
	"context"
	"log"
	"path/filepath"
	"runtime"
	"sync/atomic"
	"time"

	"github.com/egoist/mygo"
)

// idleStop is how long the quit waits here for a child with nothing
// running: an idle serve exits in milliseconds.
const idleStop = 5 * time.Second

type desktop struct {
	sc    *Sidecar
	win   *window
	texts Texts

	leaving bool        // main thread only: the quit under way goes on
	asking  atomic.Bool // the busy-quit question is up, or its answer is being carried out
}

func main() {
	if !mygo.App.RequestSingleInstanceLock() {
		return
	}
	d := &desktop{}
	mygo.App.OnSecondInstance(func([]string, string) { d.focus() })
	mygo.App.OnBeforeQuit(d.beforeQuit)
	mygo.App.OnQuit(d.cleanup)
	mygo.App.WhenReady(d.start)
	if err := mygo.App.Run(); err != nil {
		log.Fatal(err)
	}
}

// start shows the window on the Starting page at once; the sidecar resolves
// the login environment and starts serve behind it.
func (d *desktop) start() {
	d.texts = textsFor(mygo.App.PreferredLanguages())
	resources, _ := mygo.App.Path(mygo.PathResources)
	data, _ := mygo.App.Path(mygo.PathUserData)
	logs, _ := mygo.App.Path(mygo.PathLogs)
	downloads, _ := mygo.App.Path(mygo.PathDownloads)
	logPath := filepath.Join(logs, "serve.log")
	d.sc = NewSidecar(Config{
		Program:    programPath(resources, mygo.App.IsPackaged()),
		LogPath:    logPath,
		MemoryPath: filepath.Join(data, "desktop.json"),
		Texts:      d.texts,
		Env:        loginEnv,
	}, func(p Phase) { d.win.Show(p) })
	d.win = openWindow(d.sc, d.texts, logPath, downloads)
	d.win.w.OnClose(d.close)
	go d.sc.Run()
}

// programPath is the bundled markitai: bin/ of the app's resources, or of
// resources/<goos>-<goarch>/ under `go run`, where they are the project's
// resources directory as it is.
func programPath(resources string, packaged bool) string {
	name := "markitai"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	if packaged {
		return filepath.Join(resources, "bin", name)
	}
	return filepath.Join(resources, runtime.GOOS+"-"+runtime.GOARCH, "bin", name)
}

// gate is what a request to quit does.
type gate int

const (
	gateLeave gate = iota // no child of ours: quit
	gateDrain             // nothing unfinished: stop the child here, then quit
	gateAsk               // conversions unfinished, or no count: ask first; this quit is canceled
	gateFocus             // a stop or the question is under way: show the window
)

// busy is how many conversions a stop would wait for, running or queued, and
// uploads still arriving that would start some; not Known when the server
// could not say.
type busy struct {
	N     int
	Known bool
}

// quitGate decides from the sidecar's phase; unfinished is asked only of a
// ready child. A count that fails asks too: conversions are never stopped
// unasked.
func quitGate(p Phase, asking bool, unfinished func() (int, error)) (gate, busy) {
	switch p.(type) {
	case Stopped:
		return gateLeave, busy{}
	case Stopping:
		return gateFocus, busy{}
	}
	if asking {
		return gateFocus, busy{}
	}
	if _, ok := p.(Ready); ok {
		n, err := unfinished()
		if err != nil {
			return gateAsk, busy{}
		}
		if n > 0 {
			return gateAsk, busy{N: n, Known: true}
		}
	}
	return gateDrain, busy{}
}

// beforeQuit is the one gate for every way out: Cmd+Q and the Quit item,
// closing the window (close), SIGINT and SIGTERM, logout. It runs on the
// main thread, which a drain here holds, so only an idle stop happens here,
// for up to idleStop; one that waits for conversions goes through the
// question and the Stopping page. An idle stop completes here rather than
// canceling the quit: mygo answers a canceled quit to the system as a
// refusal, which would also cancel a logout. A child still draining after
// idleStop is finishing work the count could not see, such as a conversion
// an upload started just after it, and a draining serve no longer answers a
// count: the quit goes on as Quit When Done, on the Stopping page with Stop
// Now, rather than stopping that work unasked.
func (d *desktop) beforeQuit(e *mygo.QuitEvent) {
	if d.leaving || d.sc == nil {
		d.leaving = true
		return
	}
	g, b := quitGate(d.sc.Phase(), d.asking.Load(), func() (int, error) {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		return d.sc.Unfinished(ctx)
	})
	switch g {
	case gateLeave:
		d.leaving = true
	case gateDrain:
		if drainHere(d.sc, idleStop) {
			d.leaving = true
			return
		}
		e.PreventDefault()
		d.asking.Store(true)
		go d.quitWhenStopped()
	case gateAsk:
		e.PreventDefault()
		d.asking.Store(true)
		go d.askAndQuit(b)
	case gateFocus:
		e.PreventDefault()
		d.focus()
	}
}

// drainHere stops a child with nothing counted and waits up to limit for it
// to exit; whether it did. One still draining is left to finish, never told
// to stop now.
func drainHere(sc *Sidecar, limit time.Duration) bool {
	sc.Stop(WhenDone)
	return sc.Wait(limit)
}

// close makes closing the window a quit on every system: the window is the
// app, and no server may outlive it unseen.
func (d *desktop) close(e *mygo.CloseEvent) {
	if d.leaving {
		return
	}
	e.PreventDefault()
	mygo.App.Quit()
}

func (d *desktop) askAndQuit(b busy) {
	switch askQuit(d.win.w, d.texts, b) {
	case quitCancel:
		d.asking.Store(false)
		return
	case quitWhenDone:
		d.sc.Stop(WhenDone)
	case quitStopNow:
		d.sc.Stop(Now)
	}
	d.quitWhenStopped()
}

func (d *desktop) quitWhenStopped() {
	<-d.sc.Done()
	mygo.App.Quit()
}

type quitChoice int

const (
	quitWhenDone quitChoice = iota
	quitStopNow
	quitCancel
)

// askQuit asks what to do with unfinished conversions. A variable so the
// e2e build can answer without a dialog.
var askQuit = func(parent *mygo.Window, t Texts, b busy) quitChoice {
	r, err := mygo.Dialog.Message(mygo.MessageOptions{
		Parent:             parent,
		Type:               mygo.MessageWarning,
		Message:            t.Unfinished(b),
		Detail:             t.QuitDetail,
		Buttons:            []string{t.QuitWhenDone, t.StopAndQuit, t.Cancel},
		DefaultButton:      0,
		CancelButton:       2,
		DestructiveButtons: []int{1},
	})
	if err != nil {
		return quitCancel
	}
	switch r.Button {
	case 0:
		return quitWhenDone
	case 1:
		return quitStopNow
	}
	return quitCancel
}

// cleanup runs as the app ends, also when mygo ends it after its 3 s grace:
// a child still running is stopped at once.
func (d *desktop) cleanup() {
	if d.sc == nil {
		return
	}
	if _, ok := d.sc.Phase().(Stopped); !ok {
		d.sc.Stop(Now)
		d.sc.Wait(killGrace + time.Second)
	}
}

func (d *desktop) focus() {
	if d.win != nil {
		d.win.focus()
	}
}
