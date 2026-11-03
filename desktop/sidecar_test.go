package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"
)

// fakeServeMode makes the test binary stand in for `markitai serve
// --supervised`: ready serves capabilities with the token check and
// supervised true, and exits 0 once stdin closes; exit fails at once;
// unsupervised never says supervised; stubborn ignores stdin's end.
const fakeServeMode = "MARKITAI_DESKTOP_FAKE_SERVE"

func TestMain(m *testing.M) {
	if mode := os.Getenv(fakeServeMode); mode != "" {
		os.Exit(fakeServe(mode, os.Args[1:]))
	}
	os.Exit(m.Run())
}

func fakeServe(mode string, args []string) int {
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
		return map[string]any{"version": "test", "supervised": mode != "unsupervised"}
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
	if mode == "stubborn" {
		select {}
	}
	_, _ = io.Copy(io.Discard, os.Stdin)
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
		Lang:         "en",
		Env:          func(context.Context) []string { return env },
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
	h.await("Stopping", func(p Phase) bool { return p == Stopping{Mode: WhenDone, Child: r.Child} })
	h.await("Stopped", is[Stopped])
	if m := loadMemory(h.cfg.MemoryPath); m.Port != r.Child.Port {
		t.Errorf("remembered %v; ready on %d", m, r.Child.Port)
	}
	if !h.sc.Wait(time.Second) {
		t.Error("Done is open after Stopped")
	}
	if log, _ := os.ReadFile(h.cfg.LogPath); !strings.Contains(string(log), "stdin closed; drained") {
		t.Errorf("the child did not drain on stdin's end:\n%s", log)
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
	starting := h.await("Starting with a child", func(p Phase) bool { s, ok := p.(Starting); return ok && s.Child.PID != 0 })
	f := h.await("Failed", is[Failed]).(Failed)
	if f.Cause != (Exited{Code: 1}) || !strings.Contains(f.Tail, "another serve instance is using this MARKITAI_HOME") {
		t.Errorf("failed %#v", f)
	}
	if strings.Contains(f.Tail, string(starting.(Starting).Child.Token)) || !strings.Contains(f.Tail, "Access token: [redacted]") {
		t.Errorf("the tail shows the token:\n%s", f.Tail)
	}
	h.sc.Retry()
	again := h.await("Starting with a new child", func(p Phase) bool { s, ok := p.(Starting); return ok && s.Child.PID != 0 })
	if again.(Starting).Child.PID == starting.(Starting).Child.PID {
		t.Error("retry reused the child")
	}
	h.await("Failed again", is[Failed])
}

func TestSidecarStopsAChildThatNeverAnswers(t *testing.T) {
	h := newHarness(t, "unsupervised")
	s := h.await("Starting with a child", func(p Phase) bool { s, ok := p.(Starting); return ok && s.Child.PID != 0 }).(Starting)
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

func TestSidecarStopsNowAChildThatWillNotDrain(t *testing.T) {
	h := newHarness(t, "stubborn")
	r := h.ready()
	h.sc.Stop(WhenDone)
	h.await("Stopping", is[Stopping])
	if h.sc.Wait(500 * time.Millisecond) {
		t.Fatal("a child ignoring stdin's end stopped by itself")
	}
	h.sc.Stop(Now)
	h.await("Stopping now", func(p Phase) bool { return p == Stopping{Mode: Now, Child: r.Child} })
	h.await("Stopped", is[Stopped])
}

func TestSidecarCountsRunningConversions(t *testing.T) {
	h := newHarness(t, "ready", "MARKITAI_DESKTOP_FAKE_RUNNING=2")
	h.ready()
	if n := h.sc.Running(context.Background()); n != 2 {
		t.Errorf("Running = %d; want 2", n)
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
	if n := h.sc.Running(context.Background()); n != 0 {
		t.Errorf("Running = %d from another server's control file; want 0", n)
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
	cfg.Env = func(ctx context.Context) []string { <-release; return login(ctx) }
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
