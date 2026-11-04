package main

import (
	"errors"
	"reflect"
	"testing"
)

func TestNext(t *testing.T) {
	c := child{PID: 7, Port: 51234, Token: "t"}
	const old = 6
	cases := []struct {
		name    string
		from    Phase
		ev      event
		want    Phase
		effects []effect
	}{
		{"launch spawns once the environment is known", Preparing{}, evPrepared{},
			Preparing{}, []effect{doSpawn{}}},
		{"a spawned child is tracked", Preparing{}, evSpawned{Child: c}, Starting{Child: c}, nil},
		{"a missing program fails", Preparing{}, evSpawnFailed{Cause: NoProgram{Path: "/x"}},
			Failed{Cause: NoProgram{Path: "/x"}}, nil},
		{"a supervised answer is ready and remembers the port", Starting{Child: c}, evReady{PID: 7},
			Ready{Child: c}, []effect{doRemember{Port: 51234}}},
		{"an earlier child's answer is ignored", Starting{Child: c}, evReady{PID: old}, Starting{Child: c}, nil},
		{"an early exit fails with the tail", Starting{Child: c}, evExited{PID: 7, Code: 1, Tail: "busy"},
			Failed{Cause: Exited{Code: 1}, Tail: "busy"}, nil},
		{"a child that never answers fails, stopped now and still exiting", Starting{Child: c},
			evNotReady{PID: 7, Tail: "x"}, Failed{Cause: NotReady{}, Tail: "x", Dying: 7}, []effect{doStop{PID: 7, Mode: Now}}},
		{"its exit leaves the failure", Failed{Cause: NotReady{}, Tail: "x", Dying: 7}, evExited{PID: 7, Code: -1, Tail: "y"},
			Failed{Cause: NotReady{}, Tail: "x"}, nil},
		{"a crash while ready fails", Ready{Child: c}, evExited{PID: 7, Code: -1, Tail: "t"},
			Failed{Cause: Exited{Code: -1}, Tail: "t"}, nil},
		{"an earlier child's exit is ignored while ready", Ready{Child: c}, evExited{PID: old}, Ready{Child: c}, nil},
		{"retry spawns again", Failed{Cause: Exited{Code: 1}}, evRetry{}, Preparing{}, []effect{doSpawn{}}},
		{"retry waits for the failed child to exit", Failed{Cause: NotReady{}, Dying: old}, evRetry{},
			Preparing{Dying: old}, nil},
		{"its exit lets the retry spawn", Preparing{Dying: old}, evExited{PID: old}, Preparing{}, []effect{doSpawn{}}},
		{"another exit does not", Preparing{Dying: old}, evExited{PID: 5}, Preparing{Dying: old}, nil},
		{"retry while ready does nothing", Ready{Child: c}, evRetry{}, Ready{Child: c}, nil},
		{"a stop before any child ends at once", Preparing{}, evStop{Mode: WhenDone}, Stopped{}, nil},
		{"the environment after a stop spawns nothing", Stopped{}, evPrepared{}, Stopped{}, nil},
		{"a stop after a failure ends at once", Failed{Cause: NotReady{}}, evStop{Mode: Now}, Stopped{}, nil},
		{"a stop after a failure waits for the child still exiting", Failed{Cause: NotReady{}, Dying: old},
			evStop{Mode: WhenDone}, Stopping{Mode: Now, PID: old}, nil},
		{"a stop while a retry waits waits too", Preparing{Dying: old}, evStop{Mode: WhenDone},
			Stopping{Mode: Now, PID: old}, nil},
		{"a stop when done drains the ready child", Ready{Child: c}, evStop{Mode: WhenDone},
			Stopping{Mode: WhenDone, PID: 7}, []effect{doStop{PID: 7, Mode: WhenDone}}},
		{"a stop now ends the ready child", Ready{Child: c}, evStop{Mode: Now},
			Stopping{Mode: Now, PID: 7}, []effect{doStop{PID: 7, Mode: Now}}},
		{"a stop reaches a child still starting", Starting{Child: c}, evStop{Mode: WhenDone},
			Stopping{Mode: WhenDone, PID: 7}, []effect{doStop{PID: 7, Mode: WhenDone}}},
		{"stop now hurries a drain", Stopping{Mode: WhenDone, PID: 7}, evStop{Mode: Now},
			Stopping{Mode: Now, PID: 7}, []effect{doStop{PID: 7, Mode: Now}}},
		{"a drain is not asked twice", Stopping{Mode: WhenDone, PID: 7}, evStop{Mode: WhenDone},
			Stopping{Mode: WhenDone, PID: 7}, nil},
		{"stop now is not undone", Stopping{Mode: Now, PID: 7}, evStop{Mode: WhenDone},
			Stopping{Mode: Now, PID: 7}, nil},
		{"the child's exit ends the stop", Stopping{Mode: WhenDone, PID: 7}, evExited{PID: 7},
			Stopped{}, nil},
		{"an earlier child's exit does not end the stop", Stopping{Mode: Now, PID: 7}, evExited{PID: old},
			Stopping{Mode: Now, PID: 7}, nil},
		{"readiness during a stop is ignored", Stopping{Mode: WhenDone, PID: 7}, evReady{PID: 7},
			Stopping{Mode: WhenDone, PID: 7}, nil},
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
	if p != (Preparing{}) || !reflect.DeepEqual(effects, []effect{doPrepare{}}) {
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
	noCount := errors.New("no control file")
	cases := []struct {
		name    string
		phase   Phase
		asking  bool
		running int
		err     error
		want    gate
		busy    busy
	}{
		{"stopped leaves", Stopped{}, false, 0, nil, gateLeave, busy{}},
		{"a stop under way shows the window", Stopping{Mode: WhenDone, PID: 7}, false, 0, nil, gateFocus, busy{}},
		{"an open question shows the window", Ready{Child: c}, true, 3, nil, gateFocus, busy{}},
		{"an idle ready child is drained here", Ready{Child: c}, false, 0, nil, gateDrain, busy{}},
		{"unfinished conversions ask", Ready{Child: c}, false, 2, nil, gateAsk, busy{N: 2, Known: true}},
		{"a count that fails asks, saying it is unknown", Ready{Child: c}, false, 0, noCount, gateAsk, busy{}},
		{"a starting child is drained here", Starting{Child: c}, false, 5, nil, gateDrain, busy{}},
		{"a failure ends the sidecar", Failed{Cause: NotReady{}}, false, 0, nil, gateDrain, busy{}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			g, b := quitGate(tc.phase, tc.asking, func() (int, error) { return tc.running, tc.err })
			if g != tc.want || b != tc.busy {
				t.Errorf("quitGate = %v, %+v; want %v, %+v", g, b, tc.want, tc.busy)
			}
		})
	}
}

func TestQuitQuestionSaysUnfinished(t *testing.T) {
	cases := []struct {
		texts Texts
		busy  busy
		want  string
	}{
		{en, busy{N: 1, Known: true}, "1 conversion is unfinished."},
		{en, busy{N: 3, Known: true}, "3 conversions are unfinished."},
		{en, busy{}, "Conversions may be unfinished; Markitai Web could not count them."},
		{zh, busy{N: 3, Known: true}, "还有 3 个转换未完成。"},
		{zh, busy{}, "可能还有未完成的转换，Markitai Web 无法统计数量。"},
	}
	for _, tc := range cases {
		if got := tc.texts.Unfinished(tc.busy); got != tc.want {
			t.Errorf("%s Unfinished(%+v) = %q; want %q", tc.texts.Lang, tc.busy, got, tc.want)
		}
	}
}
