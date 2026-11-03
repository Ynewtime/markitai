package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync/atomic"
	"time"
)

const (
	readyTimeout = 60 * time.Second
	readyPoll    = 100 * time.Millisecond
	// killGrace is how long a child stopped Now has to exit before it is
	// killed outright.
	killGrace = 5 * time.Second
	tailBytes = 4096
)

// Config is what the sidecar needs from the app, resolved at launch.
type Config struct {
	// Program is the bundled markitai.
	Program string
	// LogPath receives the child's stdout and stderr, a file per spawn: a
	// pipe would turn serve's writes after the app quit into crashes.
	LogPath string
	// MemoryPath is PathUserData/desktop.json.
	MemoryPath string
	// Lang is "en" or "zh", the child's LANG when the login environment has none.
	Lang string
	// Env returns the login environment; it may take seconds.
	Env func(context.Context) []string
	// ReadyTimeout defaults to readyTimeout.
	ReadyTimeout time.Duration
}

// Sidecar owns the bundled markitai serve: it starts it with a fresh token,
// notices it answer or die, and stops it. Callers see Phases and ask for
// Retry and Stop.
type Sidecar struct {
	cfg    Config
	show   func(Phase)
	events chan event
	done   chan struct{}
	last   atomic.Pointer[Phase]

	// Run's alone. env is written by the doPrepare goroutine before it posts
	// evPrepared, the only event after which Run reads it.
	phase  Phase
	env    []string
	memory Memory
	procs  map[int]*proc
}

type proc struct {
	cmd   *exec.Cmd
	stdin io.Closer
}

// NewSidecar returns a sidecar whose Run will call show with each new phase;
// show must not wait for the main thread, which may be waiting for Done.
func NewSidecar(cfg Config, show func(Phase)) *Sidecar {
	if cfg.ReadyTimeout == 0 {
		cfg.ReadyTimeout = readyTimeout
	}
	s := &Sidecar{cfg: cfg, show: show, events: make(chan event, 16), done: make(chan struct{}),
		procs: map[int]*proc{}}
	var p Phase = Starting{}
	s.last.Store(&p)
	return s
}

// Run is the loop, the only writer of the phase and of Memory. It returns
// once the phase is Stopped.
func (s *Sidecar) Run() {
	s.memory = loadMemory(s.cfg.MemoryPath)
	p, effects := start()
	for _, e := range s.apply(p, effects) {
		s.step(e)
	}
	for {
		if _, ok := s.phase.(Stopped); ok {
			close(s.done)
			return
		}
		s.step(<-s.events)
	}
}

// step runs e through next and does what the transition asks. A spawn
// answers at once and its answer is taken before any queued event, so no
// event can meet a child the phase does not know.
func (s *Sidecar) step(e event) {
	for queue := []event{e}; len(queue) > 0; queue = queue[1:] {
		if exited, ok := queue[0].(evExited); ok {
			delete(s.procs, exited.PID)
		}
		p, effects := next(s.phase, queue[0])
		queue = append(queue, s.apply(p, effects)...)
	}
}

func (s *Sidecar) apply(p Phase, effects []effect) []event {
	if p != s.phase {
		s.phase = p
		s.last.Store(&p)
		s.show(p)
	}
	var answers []event
	for _, f := range effects {
		if e := s.do(f); e != nil {
			answers = append(answers, e)
		}
	}
	return answers
}

func (s *Sidecar) do(f effect) event {
	switch f := f.(type) {
	case doPrepare:
		go func() {
			s.env = s.cfg.Env(context.Background())
			s.post(evPrepared{})
		}()
	case doSpawn:
		return s.spawn()
	case doStop:
		s.stop(f.PID, f.Mode)
	case doRemember:
		s.memory.Port = f.Port
		if err := s.memory.save(s.cfg.MemoryPath); err != nil {
			log.Printf("markitai desktop: %v", err)
		}
	}
	return nil
}

// spawn starts `markitai serve --supervised` on a loopback port with a new
// token, its working directory MARKITAI_HOME (serve reads ./markitai.json and
// ./.env), and stdin a pipe whose closing asks it to finish and exit.
func (s *Sidecar) spawn() event {
	failed := func(err error) event { return evSpawnFailed{Cause: SpawnFailed{Err: err.Error()}} }
	if _, err := os.Stat(s.cfg.Program); err != nil {
		return evSpawnFailed{Cause: NoProgram{Path: s.cfg.Program}}
	}
	userHome := value(s.env, "HOME")
	if userHome == "" {
		userHome, _ = os.UserHomeDir()
	}
	home := markitaiHome(s.env, userHome)
	if err := os.MkdirAll(home, 0o700); err != nil {
		return failed(err)
	}
	port, err := pickPort(s.memory.Port)
	if err != nil {
		return failed(err)
	}
	logFile, err := rotateLog(s.cfg.LogPath)
	if err != nil {
		return failed(err)
	}
	defer logFile.Close()
	token := newToken()
	cmd := exec.Command(s.cfg.Program, "serve", "--supervised", "--host", "127.0.0.1", "--port", port.String())
	cmd.Dir = home
	cmd.Env = childEnv(s.env, token, s.cfg.Lang, home)
	cmd.Stdout, cmd.Stderr = logFile, logFile
	cmd.SysProcAttr = procAttr()
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return failed(err)
	}
	if err := cmd.Start(); err != nil {
		return failed(err)
	}
	c := child{PID: cmd.Process.Pid, Port: port, Token: token, Home: home}
	s.procs[c.PID] = &proc{cmd: cmd, stdin: stdin}
	ctx, cancel := context.WithCancel(context.Background())
	go s.awaitReady(ctx, c)
	go func() {
		code := exitCode(cmd.Wait())
		cancel()
		s.post(evExited{PID: c.PID, Code: code, Tail: s.tail(token)})
	}()
	return evSpawned{Child: c}
}

// awaitReady posts evReady once the child answers its capabilities with the
// token and says it is supervised: only our child knows the token, so no
// localized startup text is parsed. It posts evNotReady after ReadyTimeout.
func (s *Sidecar) awaitReady(ctx context.Context, c child) {
	deadline := time.After(s.cfg.ReadyTimeout)
	tick := time.NewTicker(readyPoll)
	defer tick.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-deadline:
			s.post(evNotReady{PID: c.PID, Tail: s.tail(c.Token)})
			return
		case <-tick.C:
			var caps struct {
				Supervised bool `json:"supervised"`
			}
			if get(ctx, c, "/api/capabilities", "Authorization", "Bearer "+string(c.Token), &caps) == nil &&
				caps.Supervised {
				s.post(evReady{PID: c.PID})
				return
			}
		}
	}
}

func (s *Sidecar) stop(pid int, mode Mode) {
	p := s.procs[pid]
	if p == nil {
		return
	}
	_ = p.stdin.Close()
	if mode == Now {
		_ = stopNow(p.cmd.Process)
		time.AfterFunc(killGrace, func() { _ = p.cmd.Process.Kill() })
	}
}

func (s *Sidecar) tail(token Token) string {
	f, err := os.Open(s.cfg.LogPath)
	if err != nil {
		return ""
	}
	defer f.Close()
	cut := false
	if info, err := f.Stat(); err == nil && info.Size() > tailBytes {
		_, _ = f.Seek(-tailBytes, io.SeekEnd)
		cut = true
	}
	b, _ := io.ReadAll(f)
	return logTail(b, cut, token)
}

// logTail is the end of a log for the status page: whole lines only (cut: b
// starts mid-line), without the token serve prints at startup.
func logTail(b []byte, cut bool, token Token) string {
	if cut {
		if i := bytes.IndexByte(b, '\n'); i >= 0 {
			b = b[i+1:]
		}
	}
	text := strings.TrimSpace(string(b))
	if token != "" {
		text = strings.ReplaceAll(text, string(token), token.String())
	}
	return text
}

func rotateLog(path string) (*os.File, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return nil, err
	}
	_ = os.Rename(path, strings.TrimSuffix(path, ".log")+".previous.log")
	return os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600)
}

func exitCode(err error) int {
	var exit *exec.ExitError
	if errors.As(err, &exit) {
		return exit.ExitCode()
	}
	if err != nil {
		return -1
	}
	return 0
}

func (s *Sidecar) post(e event) {
	select {
	case s.events <- e:
	case <-s.done:
	}
}

// Retry starts a new child after a failure.
func (s *Sidecar) Retry() { s.post(evRetry{}) }

// Stop stops the child, or with none, ends the sidecar at once.
func (s *Sidecar) Stop(m Mode) { s.post(evStop{Mode: m}) }

// Phase is the phase published last.
func (s *Sidecar) Phase() Phase { return *s.last.Load() }

// Done closes once the phase is Stopped.
func (s *Sidecar) Done() <-chan struct{} { return s.done }

// Wait reports whether the sidecar stopped within d.
func (s *Sidecar) Wait(d time.Duration) bool {
	select {
	case <-s.done:
		return true
	case <-time.After(d):
		return false
	}
}

// Running counts the conversions our ready child has running or queued; 0
// when it has none or does not answer.
func (s *Sidecar) Running(ctx context.Context) int {
	r, ok := s.Phase().(Ready)
	if !ok {
		return 0
	}
	n, err := waiting(ctx, r.Child)
	if err != nil {
		log.Printf("markitai desktop: running conversions: %v", err)
	}
	return n
}

// control is the part of MARKITAI_HOME/serve/control, written by serve for
// `markitai update`, that the app reads.
type control struct {
	Format int    `json:"format"`
	PID    int    `json:"pid"`
	Secret string `json:"secret"`
}

// waiting asks c the question `markitai update` asks before a restart, how
// many conversions a stop waits for: GET /api/update/control with the
// secret c wrote to its home. The history cannot say: it lists finished
// jobs only.
func waiting(ctx context.Context, c child) (int, error) {
	b, err := os.ReadFile(filepath.Join(c.Home, "serve", "control"))
	if err != nil {
		return 0, err
	}
	var ctl control
	if err := json.Unmarshal(b, &ctl); err != nil {
		return 0, err
	}
	if ctl.Format != 1 || ctl.PID != c.PID {
		return 0, fmt.Errorf("the control file is another server's (format %d, pid %d)", ctl.Format, ctl.PID)
	}
	var status struct {
		WaitingJobs int `json:"waiting_jobs"`
	}
	err = get(ctx, c, "/api/update/control", "X-Markitai-Control", ctl.Secret, &status)
	return status.WaitingJobs, err
}

// api talks to our child only, on loopback: never through a proxy.
var api = &http.Client{Transport: &http.Transport{Proxy: nil}, Timeout: 2 * time.Second}

// get decodes the answer to GET path on c, authorized by header.
func get(ctx context.Context, c child, path, header, value string, v any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.origin()+path, nil)
	if err != nil {
		return err
	}
	req.Header.Set(header, value)
	resp, err := api.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("GET %s: %s", path, resp.Status)
	}
	return json.NewDecoder(resp.Body).Decode(v)
}
