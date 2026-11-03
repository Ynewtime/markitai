package main

import (
	"reflect"
	"testing"
)

func TestNext(t *testing.T) {
	c := child{PID: 7, Port: 51234, Token: "t"}
	old := child{PID: 6, Port: 51000, Token: "o"}
	cases := []struct {
		name    string
		from    Phase
		ev      event
		want    Phase
		effects []effect
	}{
		{"launch spawns once the environment is known", Starting{}, evPrepared{},
			Starting{}, []effect{doSpawn{}}},
		{"a spawned child is tracked", Starting{}, evSpawned{Child: c}, Starting{Child: c}, nil},
		{"a missing program fails", Starting{}, evSpawnFailed{Cause: NoProgram{Path: "/x"}},
			Failed{Cause: NoProgram{Path: "/x"}}, nil},
		{"a supervised answer is ready and remembers the port", Starting{Child: c}, evReady{PID: 7},
			Ready{Child: c}, []effect{doRemember{Port: 51234}}},
		{"an earlier child's answer is ignored", Starting{Child: c}, evReady{PID: 6}, Starting{Child: c}, nil},
		{"an early exit fails with the tail", Starting{Child: c}, evExited{PID: 7, Code: 1, Tail: "busy"},
			Failed{Cause: Exited{Code: 1}, Tail: "busy"}, nil},
		{"a child that never answers is stopped now", Starting{Child: c}, evNotReady{PID: 7, Tail: "x"},
			Failed{Cause: NotReady{}, Tail: "x"}, []effect{doStop{PID: 7, Mode: Now}}},
		{"a crash while ready fails", Ready{Child: c}, evExited{PID: 7, Code: -1, Tail: "t"},
			Failed{Cause: Exited{Code: -1}, Tail: "t"}, nil},
		{"an earlier child's exit is ignored while ready", Ready{Child: c}, evExited{PID: 6}, Ready{Child: c}, nil},
		{"retry spawns again", Failed{Cause: Exited{Code: 1}}, evRetry{}, Starting{}, []effect{doSpawn{}}},
		{"retry while ready does nothing", Ready{Child: c}, evRetry{}, Ready{Child: c}, nil},
		{"a stop before any child ends at once", Starting{}, evStop{Mode: WhenDone}, Stopped{}, nil},
		{"the environment after a stop spawns nothing", Stopped{}, evPrepared{}, Stopped{}, nil},
		{"a stop after a failure ends at once", Failed{Cause: NotReady{}}, evStop{Mode: Now}, Stopped{}, nil},
		{"a stop when done drains the ready child", Ready{Child: c}, evStop{Mode: WhenDone},
			Stopping{Mode: WhenDone, Child: c}, []effect{doStop{PID: 7, Mode: WhenDone}}},
		{"a stop now ends the ready child", Ready{Child: c}, evStop{Mode: Now},
			Stopping{Mode: Now, Child: c}, []effect{doStop{PID: 7, Mode: Now}}},
		{"a stop reaches a child still starting", Starting{Child: c}, evStop{Mode: WhenDone},
			Stopping{Mode: WhenDone, Child: c}, []effect{doStop{PID: 7, Mode: WhenDone}}},
		{"stop now hurries a drain", Stopping{Mode: WhenDone, Child: c}, evStop{Mode: Now},
			Stopping{Mode: Now, Child: c}, []effect{doStop{PID: 7, Mode: Now}}},
		{"a drain is not asked twice", Stopping{Mode: WhenDone, Child: c}, evStop{Mode: WhenDone},
			Stopping{Mode: WhenDone, Child: c}, nil},
		{"stop now is not undone", Stopping{Mode: Now, Child: c}, evStop{Mode: WhenDone},
			Stopping{Mode: Now, Child: c}, nil},
		{"the child's exit ends the stop", Stopping{Mode: WhenDone, Child: c}, evExited{PID: 7},
			Stopped{}, nil},
		{"an earlier child's exit does not end the stop", Stopping{Mode: Now, Child: c}, evExited{PID: old.PID},
			Stopping{Mode: Now, Child: c}, nil},
		{"readiness during a stop is ignored", Stopping{Mode: WhenDone, Child: c}, evReady{PID: 7},
			Stopping{Mode: WhenDone, Child: c}, nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, effects := next(tc.from, tc.ev)
			if !reflect.DeepEqual(got, tc.want) || !reflect.DeepEqual(effects, tc.effects) {
				t.Errorf("next(%#v, %#v) = %#v, %#v; want %#v, %#v", tc.from, tc.ev, got, effects, tc.want, tc.effects)
			}
		})
	}
}

func TestLaunchStartsByPreparing(t *testing.T) {
	p, effects := start()
	if p != (Starting{}) || !reflect.DeepEqual(effects, []effect{doPrepare{}}) {
		t.Errorf("start() = %#v, %#v", p, effects)
	}
}

func TestReadyURLCarriesTheTokenInTheFragment(t *testing.T) {
	r := Ready{Child: child{PID: 1, Port: 51234, Token: "abc"}}
	if got := r.URL(); got != "http://127.0.0.1:51234/#token=abc" {
		t.Errorf("URL() = %q", got)
	}
}

func TestQuitGate(t *testing.T) {
	c := child{PID: 7, Port: 51234, Token: "t"}
	cases := []struct {
		name    string
		phase   Phase
		asking  bool
		running int
		want    gate
		n       int
	}{
		{"stopped leaves", Stopped{}, false, 0, gateLeave, 0},
		{"a stop under way shows the window", Stopping{Mode: WhenDone, Child: c}, false, 0, gateFocus, 0},
		{"an open question shows the window", Ready{Child: c}, true, 3, gateFocus, 0},
		{"an idle ready child is drained here", Ready{Child: c}, false, 0, gateDrain, 0},
		{"running conversions ask", Ready{Child: c}, false, 2, gateAsk, 2},
		{"a starting child is drained here", Starting{Child: c}, false, 5, gateDrain, 0},
		{"a failure ends the sidecar", Failed{Cause: NotReady{}}, false, 0, gateDrain, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			g, n := quitGate(tc.phase, tc.asking, func() int { return tc.running })
			if g != tc.want || n != tc.n {
				t.Errorf("quitGate = %v, %d; want %v, %d", g, n, tc.want, tc.n)
			}
		})
	}
}
