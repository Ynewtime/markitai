package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"
)

// fakeServeMode makes the test binary stand in for `markitai serve
// --supervised`: ready serves capabilities with the token check and
// supervised true; on stdin, "drain" or its end make it exit 0, "now" exit
// 129 at once. exit fails at once; unsupervised never says supervised;
// stubborn ignores drain and stdin's end, as a serve with long conversions
// would; deaf ignores stdin, as a hung serve would. Modes combine with
// commas. sleep stands in for a login shell whose startup files take a
// minute.
const fakeServeMode = "MARKITAI_DESKTOP_FAKE_SERVE"

func TestMain(m *testing.M) {
	if mode := os.Getenv(fakeServeMode); mode != "" {
		os.Exit(fakeServe(mode, os.Args[1:]))
	}
	os.Exit(m.Run())
}

func fakeServe(mode string, args []string) int {
	if mode == "sleep" {
		time.Sleep(time.Minute)
		return 0
	}
	i := slices.Index(args, "--port")
	if len(args) == 0 || args[0] != "serve" || !slices.Contains(args, "--supervised") || i < 0 || i+1 >= len(args) {
		fmt.Fprintln(os.Stderr, "fake serve: unexpected arguments", args)
		return 2
	}
	for _, kv := range os.Environ() {
		if strings.HasPrefix(kv, "MYGO_") {
			fmt.Fprintln(os.Stderr, "fake serve: the app passed", kv)
			return 2
		}
	}
	if wd, _ := os.Getwd(); !sameDir(wd, os.Getenv("MARKITAI_HOME")) {
		fmt.Fprintf(os.Stderr, "fake serve: working directory %s is not MARKITAI_HOME\n", wd)
		return 2
	}
	token := os.Getenv("MARKITAI_SERVE_TOKEN")
	fmt.Fprintf(os.Stderr, "  • Access token: %s\n", token)
	modes := strings.Split(mode, ",")
	if mode == "exit" {
		fmt.Fprintln(os.Stderr, "fake serve: another serve instance is using this MARKITAI_HOME")
		return 1
	}
	running, _ := strconv.Atoi(os.Getenv("MARKITAI_DESKTOP_FAKE_RUNNING"))
	authed := func(answer func() any) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			if r.Header.Get("Authorization") != "Bearer "+token {
				http.Error(w, `{"code":"unauthorized"}`, http.StatusUnauthorized)
				return
			}
			_ = json.NewEncoder(w).Encode(answer())
		}
	}
	mux := http.NewServeMux()
	mux.Handle("/api/capabilities", authed(func() any {
		return map[string]any{"version": "test", "supervised": !slices.Contains(modes, "unsupervised")}
	}))
	secret := "fake-control-secret"
	mux.HandleFunc("/api/update/control", func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Markitai-Control") != secret {
			http.Error(w, `{"code":"control_forbidden"}`, http.StatusForbidden)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"pid": os.Getpid(), "waiting_jobs": running})
	})
	l, err := net.Listen("tcp", "127.0.0.1:"+args[i+1])
	if err != nil {
		fmt.Fprintln(os.Stderr, "fake serve: the port is already in use")
		return 1
	}
	go func() { _ = http.Serve(l, mux) }()
	ctl, _ := json.Marshal(control{Format: 1, PID: os.Getpid(), Secret: secret})
	if os.MkdirAll("serve", 0o700) != nil || os.WriteFile(filepath.Join("serve", "control"), ctl, 0o600) != nil {
		fmt.Fprintln(os.Stderr, "fake serve: cannot write the control file")
		return 1
	}
	if slices.Contains(modes, "deaf") {
		select {}
	}
	stubborn := slices.Contains(modes, "stubborn")
	for lines := bufio.NewScanner(os.Stdin); lines.Scan(); {
		switch lines.Text() {
		case "now":
			fmt.Fprintln(os.Stderr, "fake serve: now; stopped at once")
			return 129
		case "drain":
			if !stubborn {
				fmt.Fprintln(os.Stderr, "fake serve: drain; drained")
				return 0
			}
		}
	}
	if stubborn {
		select {}
	}
	fmt.Fprintln(os.Stderr, "fake serve: stdin closed; drained")
	return 0
}

func sameDir(a, b string) bool {
	a, errA := filepath.EvalSymlinks(a)
	b, errB := filepath.EvalSymlinks(b)
	return errA == nil && errB == nil && a == b
}

type harness struct {
	t      *testing.T
	sc     *Sidecar
	cfg    Config
	phases chan Phase
}

// newHarness runs a sidecar whose program is this test binary as fake serve.
func newHarness(t *testing.T, mode string, extra ...string) *harness {
	return startHarness(t, harnessConfig(t, mode, extra...))
}

func harnessConfig(t *testing.T, mode string, extra ...string) Config {
	dir := t.TempDir()
	env := append([]string{"PATH=" + os.Getenv("PATH"), "HOME=" + dir,
		"MARKITAI_HOME=" + filepath.Join(dir, "markitai-home"), fakeServeMode + "=" + mode,
		"MYGO_AUTOMATION=e2e.sock"}, extra...)
	return Config{
		Program:      os.Args[0],
		LogPath:      filepath.Join(dir, "logs", "serve.log"),
		MemoryPath:   filepath.Join(dir, "data", "desktop.json"),
		Texts:        en,
		Env:          func(context.Context) ([]string, error) { return env, nil },
		ReadyTimeout: 3 * time.Second,
	}
}

func startHarness(t *testing.T, cfg Config) *harness {
	h := &harness{t: t, cfg: cfg, phases: make(chan Phase, 64)}
	h.sc = NewSidecar(cfg, func(p Phase) { h.phases <- p })
	go h.sc.Run()
	t.Cleanup(func() {
		h.sc.Stop(Now)
		if !h.sc.Wait(10 * time.Second) {
			t.Error("the sidecar did not stop")
		}
	})
	return h
}

// await returns the first phase shown from now on that ok accepts.
func (h *harness) await(what string, ok func(Phase) bool) Phase {
	h.t.Helper()
	timeout := time.After(15 * time.Second)
	for {
		select {
		case p := <-h.phases:
			if ok(p) {
				return p
			}
		case <-timeout:
			h.t.Fatalf("no %s phase; the last is %#v", what, h.sc.Phase())
		}
	}
}

func is[T Phase](p Phase) bool { _, ok := p.(T); return ok }

func (h *harness) ready() Ready { return h.await("Ready", is[Ready]).(Ready) }

func TestSidecarStartsAndDrains(t *testing.T) {
	h := newHarness(t, "ready")
	r := h.ready()
	if r.Child.Port == 0 || r.Child.Port == cliPort || len(r.Child.Token) != 64 {
		t.Errorf("ready child %+v", r.Child)
	}
	for token, want := range map[Token]int{r.Child.Token: http.StatusOK, "wrong": http.StatusUnauthorized} {
		var caps map[string]any
		err := get(context.Background(), r.Child, "/api/capabilities", "Authorization", "Bearer "+string(token), &caps)
		if (want == http.StatusOK) != (err == nil) {
			t.Errorf("capabilities with token %v: %v", token, err)
		}
	}
	h.sc.Stop(WhenDone)
	h.await("Stopping", func(p Phase) bool { return p == Stopping{Mode: WhenDone, PID: r.Child.PID} })
	h.await("Stopped", is[Stopped])
	if m := loadMemory(h.cfg.MemoryPath); m.Port != r.Child.Port {
		t.Errorf("remembered %v; ready on %d", m, r.Child.Port)
	}
	if !h.sc.Wait(time.Second) {
		t.Error("Done is open after Stopped")
	}
	if log, _ := os.ReadFile(h.cfg.LogPath); !strings.Contains(string(log), "drain; drained") {
		t.Errorf("the child did not hear drain on stdin:\n%s", log)
	}
}

func TestSidecarKeepsThePortAcrossLaunches(t *testing.T) {
	first := newHarness(t, "ready")
	port := first.ready().Child.Port
	first.sc.Stop(WhenDone)
	first.await("Stopped", is[Stopped])
	second := startHarness(t, first.cfg)
	if got := second.ready().Child.Port; got != port {
		t.Errorf("second launch on %d; the first was on %d", got, port)
	}
}

func TestSidecarEarlyExitShowsTheLogTail(t *testing.T) {
	h := newHarness(t, "exit")
	starting := h.await("Starting", is[Starting])
	f := h.await("Failed", is[Failed]).(Failed)
	if f.Cause != (Exited{Code: 1}) || !strings.Contains(f.Tail, "another serve instance is using this MARKITAI_HOME") {
		t.Errorf("failed %#v", f)
	}
	if strings.Contains(f.Tail, string(starting.(Starting).Child.Token)) || !strings.Contains(f.Tail, "Access token: [redacted]") {
		t.Errorf("the tail shows the token:\n%s", f.Tail)
	}
	h.sc.Retry()
	again := h.await("Starting again", is[Starting])
	if again.(Starting).Child.PID == starting.(Starting).Child.PID {
		t.Error("retry reused the child")
	}
	h.await("Failed again", is[Failed])
}

func TestSidecarStopsAChildThatNeverAnswers(t *testing.T) {
	h := newHarness(t, "unsupervised")
	s := h.await("Starting", is[Starting]).(Starting)
	f := h.await("Failed", is[Failed]).(Failed)
	if f.Cause != (NotReady{}) {
		t.Errorf("failed %#v", f)
	}
	deadline := time.Now().Add(10 * time.Second)
	for alive(s.Child.PID) {
		if time.Now().After(deadline) {
			t.Fatalf("the unready child %d still runs", s.Child.PID)
		}
		time.Sleep(50 * time.Millisecond)
	}
}

func TestSidecarStopsOnlyOnceTheFailedChildIsGone(t *testing.T) {
	h := newHarness(t, "unsupervised,deaf")
	s := h.await("Starting", is[Starting]).(Starting)
	h.await("Failed", is[Failed])
	h.sc.Stop(WhenDone)
	h.await("Stopped", is[Stopped])
	if alive(s.Child.PID) {
		t.Errorf("Stopped while child %d still runs", s.Child.PID)
	}
}

func TestSidecarRetriesOnceTheFailedChildIsGone(t *testing.T) {
	h := newHarness(t, "unsupervised,deaf")
	old := h.await("Starting", is[Starting]).(Starting)
	h.await("Failed", is[Failed])
	h.sc.Retry()
	again := h.await("Starting with a new child", func(p Phase) bool {
		s, ok := p.(Starting)
		return ok && s.Child.PID != old.Child.PID
	}).(Starting)
	if alive(old.Child.PID) {
		t.Errorf("retry spawned %d while %d still runs", again.Child.PID, old.Child.PID)
	}
}

// A stop now is the line "now"; serve's exit 129 after it is the stop, not
// a failure.
func TestSidecarStopsNowThroughStdin(t *testing.T) {
	h := newHarness(t, "ready")
	h.ready()
	h.sc.Stop(Now)
	if p := h.await("the end of the stop", func(p Phase) bool { return is[Stopped](p) || is[Failed](p) }); !is[Stopped](p) {
		t.Errorf("a stop now ended in %#v", p)
	}
	if log, _ := os.ReadFile(h.cfg.LogPath); !strings.Contains(string(log), "now; stopped at once") {
		t.Errorf("the child did not hear now on stdin:\n%s", log)
	}
}

func TestSidecarStoppedWhilePreparingLeavesNoShell(t *testing.T) {
	cfg := harnessConfig(t, "ready")
	shell := make(chan int, 1)
	cfg.Env = func(ctx context.Context) ([]string, error) {
		cmd := exec.CommandContext(ctx, os.Args[0])
		cmd.Env = append(os.Environ(), fakeServeMode+"=sleep")
		if err := cmd.Start(); err != nil {
			shell <- 0
			return nil, err
		}
		shell <- cmd.Process.Pid
		return nil, cmd.Wait()
	}
	h := startHarness(t, cfg)
	pid := <-shell
	if pid == 0 {
		t.Fatal("the stand-in login shell did not start")
	}
	h.sc.Stop(WhenDone)
	h.await("Stopped", is[Stopped])
	deadline := time.Now().Add(2 * time.Second)
	for alive(pid) {
		if time.Now().After(deadline) {
			t.Fatalf("the login shell %d outlived the stop", pid)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func TestSidecarSaysWhyTheLoginEnvironmentFailed(t *testing.T) {
	cfg := harnessConfig(t, "exit")
	login := cfg.Env
	cfg.Env = func(ctx context.Context) ([]string, error) {
		env, _ := login(ctx)
		return env, errors.New("/bin/tcsh: exit status 1: Unknown option")
	}
	f := startHarness(t, cfg).await("Failed", is[Failed]).(Failed)
	said := "could not read your login shell's environment (/bin/tcsh: exit status 1: Unknown option)"
	if !strings.Contains(f.Tail, said) || !strings.Contains(f.Tail, "another serve instance") {
		t.Errorf("the Failed page's tail does not say why:\n%s", f.Tail)
	}
	if log, _ := os.ReadFile(cfg.LogPath); !strings.Contains(string(log), said) {
		t.Errorf("serve.log does not say why:\n%s", log)
	}
}

func TestSidecarKeepsOneLogPerLaunch(t *testing.T) {
	h := newHarness(t, "exit")
	h.await("Failed", is[Failed])
	h.sc.Retry()
	h.await("Starting", is[Starting])
	h.await("Failed again", is[Failed])
	previous := strings.TrimSuffix(h.cfg.LogPath, ".log") + ".previous.log"
	if _, err := os.Stat(previous); err == nil {
		t.Errorf("a retry rotated this launch's log into %s", previous)
	}
	log, _ := os.ReadFile(h.cfg.LogPath)
	if n := strings.Count(string(log), "another serve instance"); n != 2 {
		t.Errorf("serve.log holds %d of this launch's 2 attempts:\n%s", n, log)
	}
}

// A launch while a serve of the launch before still runs, as after a crash:
// the earlier serve, still writing, writes to serve.previous.log, and
// serve.log is this launch's alone, on Windows too.
func TestALaunchMovesAsideTheLogOfAServeStillRunning(t *testing.T) {
	earlier := newHarness(t, "stubborn")
	earlier.ready()
	later := startHarness(t, earlier.cfg)
	later.ready()
	earlier.sc.Stop(Now)
	earlier.await("Stopped", is[Stopped])
	previous := strings.TrimSuffix(earlier.cfg.LogPath, ".log") + ".previous.log"
	for path, stopped := range map[string]bool{previous: true, earlier.cfg.LogPath: false} {
		log, _ := os.ReadFile(path)
		if strings.Count(string(log), "Access token") != 1 || strings.Contains(string(log), "now; stopped at once") != stopped {
			t.Errorf("%s is not one serve's log, its last words there %v:\n%s", filepath.Base(path), stopped, log)
		}
	}
}

// Stop Now and Quit on the Stopping page: "now" follows "drain" on the
// same stdin, and the child stops at once, long before killGrace.
func TestSidecarStopsNowAChildThatWillNotDrain(t *testing.T) {
	h := newHarness(t, "stubborn")
	r := h.ready()
	h.sc.Stop(WhenDone)
	h.await("Stopping", is[Stopping])
	if h.sc.Wait(500 * time.Millisecond) {
		t.Fatal("a child ignoring drain stopped by itself")
	}
	h.sc.Stop(Now)
	h.await("Stopping now", func(p Phase) bool { return p == Stopping{Mode: Now, PID: r.Child.PID} })
	if !h.sc.Wait(killGrace / 2) {
		t.Fatal("the child did not stop at once after drain")
	}
	if log, _ := os.ReadFile(h.cfg.LogPath); !strings.Contains(string(log), "now; stopped at once") {
		t.Errorf("the child did not hear now after drain:\n%s", log)
	}
}

// A quit with nothing counted drains here; a child still draining after
// the limit is finishing work the count did not see, such as a conversion
// an upload started just after it, and is not told "now".
func TestAnIdleQuitLeavesWorkTheCountMissedToFinish(t *testing.T) {
	h := newHarness(t, "stubborn")
	r := h.ready()
	if drainHere(h.sc, 300*time.Millisecond) {
		t.Fatal("drainHere took a child still draining for stopped")
	}
	time.Sleep(300 * time.Millisecond)
	if p := h.sc.Phase(); p != (Stopping{Mode: WhenDone, PID: r.Child.PID}) || !alive(r.Child.PID) {
		t.Errorf("after the idle drain, the phase is %#v and the child alive is %v; want it draining", p, alive(r.Child.PID))
	}
	if log, _ := os.ReadFile(h.cfg.LogPath); strings.Contains(string(log), "now; stopped at once") {
		t.Errorf("the idle drain told the child to stop now:\n%s", log)
	}
}

func TestSidecarCountsUnfinishedConversions(t *testing.T) {
	h := newHarness(t, "ready", "MARKITAI_DESKTOP_FAKE_RUNNING=2")
	h.ready()
	if n, err := h.sc.Unfinished(context.Background()); n != 2 || err != nil {
		t.Errorf("Unfinished = %d, %v; want 2", n, err)
	}
}

func TestQuitAsksWhenTheCountFails(t *testing.T) {
	h := newHarness(t, "ready", "MARKITAI_DESKTOP_FAKE_RUNNING=2")
	r := h.ready()
	if err := os.Remove(filepath.Join(r.Child.Home, "serve", "control")); err != nil {
		t.Fatal(err)
	}
	g, b := quitGate(h.sc.Phase(), false, func() (int, error) { return h.sc.Unfinished(context.Background()) })
	if g != gateAsk || b.Known {
		t.Errorf("with the count unknown, the quit gate is %v, %+v; want gateAsk, unknown", g, b)
	}
}

func TestSidecarAsksOnlyItsOwnServer(t *testing.T) {
	h := newHarness(t, "ready", "MARKITAI_DESKTOP_FAKE_RUNNING=2")
	r := h.ready()
	path := filepath.Join(r.Child.Home, "serve", "control")
	other, _ := json.Marshal(control{Format: 1, PID: r.Child.PID + 1, Secret: "fake-control-secret"})
	if err := os.WriteFile(path, other, 0o600); err != nil {
		t.Fatal(err)
	}
	if n, err := h.sc.Unfinished(context.Background()); err == nil {
		t.Errorf("Unfinished = %d from another server's control file; want an error", n)
	}
}

func TestSidecarWithoutProgram(t *testing.T) {
	cfg := harnessConfig(t, "ready")
	cfg.Program = filepath.Join(t.TempDir(), "missing", "markitai")
	f := startHarness(t, cfg).await("Failed", is[Failed]).(Failed)
	if f.Cause != (NoProgram{Path: cfg.Program}) {
		t.Errorf("failed %#v", f)
	}
}

func TestSidecarStoppedBeforeTheEnvironmentSpawnsNothing(t *testing.T) {
	release := make(chan struct{})
	cfg := harnessConfig(t, "ready")
	login := cfg.Env
	cfg.Env = func(ctx context.Context) ([]string, error) { <-release; return login(ctx) }
	late := startHarness(t, cfg)
	late.sc.Stop(WhenDone)
	late.await("Stopped", is[Stopped])
	close(release)
	time.Sleep(200 * time.Millisecond)
	if _, err := os.Stat(cfg.LogPath); err == nil {
		t.Error("a child started after the stop")
	}
}

func TestLogTail(t *testing.T) {
	if got := logTail([]byte("partial line\nwhole\nlast\n"), true, ""); got != "whole\nlast" {
		t.Errorf("cut tail = %q", got)
	}
	if got := logTail([]byte("first\nAccess token: abc\n"), false, "abc"); got != "first\nAccess token: [redacted]" {
		t.Errorf("tail = %q", got)
	}
}
