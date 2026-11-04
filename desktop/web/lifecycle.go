package main

// Phase is what the bundled markitai server is doing. Sidecar.Run is its only
// writer, through next; the window shows the phase published last.
type Phase interface{ isPhase() }

// Preparing lasts until a child can start: at launch, until the login
// environment is known; on a retry, until Dying, the child that failed, has
// exited. Dying is 0 when no child of ours runs.
type Preparing struct{ Dying int }

// Starting: Child runs and has not yet answered as a supervised server.
type Starting struct{ Child child }

// Ready: Child answers on its port; the window shows the workbench.
type Ready struct{ Child child }

// Failed: no server of ours answers. Tail is the end of what the child wrote
// to serve.log. Dying is the child that never answered, told to stop and
// not yet exited; 0 when no child of ours runs.
type Failed struct {
	Cause Cause
	Tail  string
	Dying int
}

// Stopping: the child PID was asked to stop and has not exited yet.
type Stopping struct {
	Mode Mode
	PID  int
}

// Stopped is terminal: no child of ours runs and the app may quit.
type Stopped struct{}

func (Preparing) isPhase() {}
func (Starting) isPhase()  {}
func (Ready) isPhase()     {}
func (Failed) isPhase()    {}
func (Stopping) isPhase()  {}
func (Stopped) isPhase()   {}

// URL is the only address the window loads: the server's own origin, which
// the server's Origin guard and the workbench's location.origin both need,
// with the token in the fragment the workbench moves to sessionStorage.
func (r Ready) URL() string { return r.Child.origin() + "/#token=" + string(r.Child.Token) }

// Cause says why no server of ours runs.
type Cause interface{ isCause() }

// Exited: the child exited by itself, or was ended from outside; serve.log
// says why (a busy home or port, a broken configuration). Signal names the
// Unix signal that ended it, as SIGKILL, and Code is then -1.
type Exited struct {
	Code   int
	Signal string
}

// NotReady: the child ran but never answered as a supervised server.
type NotReady struct{}

// NoProgram: the app has no markitai at Path, as in a checkout never staged.
type NoProgram struct{ Path string }

// SpawnFailed: the system refused to start the child.
type SpawnFailed struct{ Err string }

func (Exited) isCause()      {}
func (NotReady) isCause()    {}
func (NoProgram) isCause()   {}
func (SpawnFailed) isCause() {}

// Mode is how a child stops.
type Mode int

const (
	// WhenDone: serve refuses new work, ends queued items, finishes running
	// conversions and exits.
	WhenDone Mode = iota
	// Now ends the child at once, its runtimes (Chromium, LibreOffice) with
	// it; running conversions are lost.
	Now
)

// child is one markitai serve this app started, in its MARKITAI_HOME.
type child struct {
	PID   int
	Port  Port
	Token Token
	Home  string
}

func (c child) origin() string { return "http://127.0.0.1:" + c.Port.String() }

// Events carry the PID of the child they concern, so those of an earlier
// child, still dying after a retry, fall through next unanswered.
type event interface{ isEvent() }

type (
	evPrepared    struct{} // the login environment is known
	evSpawned     struct{ Child child }
	evSpawnFailed struct{ Cause Cause }
	evReady       struct{ PID int }
	evNotReady    struct {
		PID  int
		Tail string
	}
	evExited struct {
		PID  int
		Exit Exited
		Tail string
	}
	evRetry struct{}
	evStop  struct{ Mode Mode }
)

func (evPrepared) isEvent()    {}
func (evSpawned) isEvent()     {}
func (evSpawnFailed) isEvent() {}
func (evReady) isEvent()       {}
func (evNotReady) isEvent()    {}
func (evExited) isEvent()      {}
func (evRetry) isEvent()       {}
func (evStop) isEvent()        {}

// Effects are what a transition asks of the shell in sidecar.go.
type effect interface{ isEffect() }

type (
	doPrepare struct{} // resolve the login environment, then evPrepared
	doSpawn   struct{} // start a child, answered at once by evSpawned or evSpawnFailed
	doStop    struct {
		PID  int
		Mode Mode
	}
	doRemember struct{ Port Port }
)

func (doPrepare) isEffect()  {}
func (doSpawn) isEffect()    {}
func (doStop) isEffect()     {}
func (doRemember) isEffect() {}

// start is where every launch begins.
func start() (Phase, []effect) { return Preparing{}, []effect{doPrepare{}} }

// next is the whole lifecycle; anything it does not list leaves the phase as
// it is, with no effect. d is a child still exiting, c the current one.
//
//	from                     event                  to                       effects
//	Preparing{}              evPrepared             Preparing{}              doSpawn
//	Preparing{d}             evExited{d}            Preparing{}              doSpawn
//	Preparing                evSpawned{c}           Starting{c}
//	Preparing                evSpawnFailed{cause}   Failed{cause}
//	Starting{c}              evReady{c}             Ready{c}                 doRemember{port}
//	Starting{c}              evNotReady{c}          Failed{NotReady, d: c}   doStop{c, Now}
//	Starting{c}, Ready{c}    evExited{c}            Failed{Exited}
//	Starting{c}, Ready{c}    evStop{mode}           Stopping{mode, c}        doStop{c, mode}
//	Failed{d}                evExited{d}            Failed{}
//	Failed{}                 evRetry                Preparing{}              doSpawn
//	Failed{d}                evRetry                Preparing{d}
//	Preparing{}, Failed{}    evStop                 Stopped
//	Preparing{d}, Failed{d}  evStop                 Stopping{Now, d}
//	Stopping{WhenDone, c}    evStop{Now}            Stopping{Now, c}         doStop{c, Now}
//	Stopping{c}              evExited{c}            Stopped
func next(p Phase, e event) (Phase, []effect) {
	switch p := p.(type) {
	case Preparing:
		switch e := e.(type) {
		case evPrepared:
			return p, []effect{doSpawn{}}
		case evExited:
			if e.PID == p.Dying {
				return Preparing{}, []effect{doSpawn{}}
			}
		case evSpawned:
			return Starting{Child: e.Child}, nil
		case evSpawnFailed:
			return Failed{Cause: e.Cause}, nil
		case evStop:
			return stopped(p.Dying), nil
		}
	case Starting:
		c := p.Child
		switch e := e.(type) {
		case evReady:
			if e.PID == c.PID {
				return Ready{Child: c}, []effect{doRemember{Port: c.Port}}
			}
		case evNotReady:
			if e.PID == c.PID {
				return Failed{Cause: NotReady{}, Tail: e.Tail, Dying: c.PID}, []effect{doStop{PID: c.PID, Mode: Now}}
			}
		case evExited:
			if e.PID == c.PID {
				return Failed{Cause: e.Exit, Tail: e.Tail}, nil
			}
		case evStop:
			return Stopping{Mode: e.Mode, PID: c.PID}, []effect{doStop{PID: c.PID, Mode: e.Mode}}
		}
	case Ready:
		c := p.Child
		switch e := e.(type) {
		case evExited:
			if e.PID == c.PID {
				return Failed{Cause: e.Exit, Tail: e.Tail}, nil
			}
		case evStop:
			return Stopping{Mode: e.Mode, PID: c.PID}, []effect{doStop{PID: c.PID, Mode: e.Mode}}
		}
	case Failed:
		switch e := e.(type) {
		case evExited:
			if e.PID == p.Dying {
				return Failed{Cause: p.Cause, Tail: p.Tail}, nil
			}
		case evRetry:
			if p.Dying == 0 {
				return Preparing{}, []effect{doSpawn{}}
			}
			return Preparing{Dying: p.Dying}, nil
		case evStop:
			return stopped(p.Dying), nil
		}
	case Stopping:
		switch e := e.(type) {
		case evExited:
			if e.PID == p.PID {
				return Stopped{}, nil
			}
		case evStop:
			if p.Mode == WhenDone && e.Mode == Now {
				return Stopping{Mode: Now, PID: p.PID}, []effect{doStop{PID: p.PID, Mode: Now}}
			}
		}
	}
	return p, nil
}

// stopped is where a stop leads with no live child but perhaps dying, one
// already told to stop now: Stopped once it has exited.
func stopped(dying int) Phase {
	if dying == 0 {
		return Stopped{}
	}
	return Stopping{Mode: Now, PID: dying}
}
