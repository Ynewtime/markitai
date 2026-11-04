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
	"strconv"
	"strings"
	"sync/atomic"
	"time"
)

const (
	readyTimeout = 60 * time.Second
	readyPoll    = 100 * time.Millisecond
	// killGrace is how long a child told "now" has to exit before it is
	// killed outright, as a hung one would not.
	killGrace = 5 * time.Second
	tailBytes = 4096
)

// Config is what the sidecar needs from the app, resolved at launch.
type Config struct {
	// Program is the bundled markitai.
	Program string
	// LogPath receives the children's stdout and stderr, a file rather than
	// a pipe, which would turn serve's writes after the app quit into
	// crashes. It holds one launch: the previous one's is beside it as
	// serve.previous.log.
	LogPath string
	// Memory is PathUserData/web.json, which the window shares.
	Memory *memory
	// Texts are the app's words as they are now: their Lang is the child's
	// LANG when the login environment has none.
	Texts func() Texts
	// Env returns the login environment, or a fallback and why; it may take
	// seconds, and ends early when its context does.
	Env func(context.Context) ([]string, error)
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

	// Run's alone. env and envErr are written by the doPrepare goroutine
	// before it posts evPrepared, the only event after which Run reads them.
	ctx     context.Context // ends with Run
	phase   Phase
	env     []string
	envErr  error
	procs   map[int]*proc
	rotated bool // serve.log is this launch's
}

type proc struct {
	cmd   *exec.Cmd
	stdin io.Writer
}

// NewSidecar returns a sidecar whose Run will call show with each new phase;
// show must not wait for the main thread, which may be waiting for Done.
func NewSidecar(cfg Config, show func(Phase)) *Sidecar {
	if cfg.ReadyTimeout == 0 {
		cfg.ReadyTimeout = readyTimeout
	}
	s := &Sidecar{cfg: cfg, show: show, events: make(chan event, 16), done: make(chan struct{}),
		procs: map[int]*proc{}}
	var p Phase = Preparing{}
	s.last.Store(&p)
	return s
}

// Run is the loop, the only writer of the phase and of the port. It returns
// once the phase is Stopped, ending the login shell if it still runs.
func (s *Sidecar) Run() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	s.ctx = ctx
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
			s.env, s.envErr = s.cfg.Env(s.ctx)
			s.post(evPrepared{})
		}()
	case doSpawn:
		return s.spawn()
	case doStop:
		s.stop(f.PID, f.Mode)
	case doRemember:
		if err := s.cfg.Memory.Update(func(m *Memory) { m.Port = f.Port }); err != nil {
			log.Printf("markitai web: %v", err)
		}
	}
	return nil
}

// spawn starts `markitai serve --supervised` on a loopback port with a new
// token, its working directory MARKITAI_HOME (serve reads ./markitai.json and
// ./.env), and stdin a pipe that carries the app's requests (stop): the
// pipe ends only with the app, and serve then drains by itself. A login
// environment that failed is said first in the child's part of serve.log,
// which the Failed page shows.
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
	port, err := pickPort(s.cfg.Memory.Get().Port)
	if err != nil {
		return failed(err)
	}
	logFile, err := s.openLog()
	if err != nil {
		return failed(err)
	}
	defer logFile.Close()
	info, err := logFile.Stat()
	if err != nil {
		return failed(err)
	}
	from := info.Size()
	texts := s.cfg.Texts()
	if s.envErr != nil {
		fmt.Fprintln(logFile, texts.LoginEnvFailed(s.envErr.Error()))
	}
	token := newToken()
	cmd := exec.Command(s.cfg.Program, "serve", "--supervised", "--host", "127.0.0.1", "--port", port.String())
	cmd.Dir = home
	cmd.Env = childEnv(s.env, token, texts.Lang, home)
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
	ctx, cancel := context.WithCancel(s.ctx)
	go s.awaitReady(ctx, c, from)
	go func() {
		exit := exitOf(cmd.Wait())
		cancel()
		s.post(evExited{PID: c.PID, Exit: exit, Tail: s.tail(token, from)})
	}()
	return evSpawned{Child: c}
}

// openLog opens serve.log for a child to append to. The launch's first
// child moves the previous launch's log aside; a retry's appends, so the log
// is the launch's. A serve of an earlier launch still writing there, as
// after a crash, writes on to the moved log (openAppend); a rename refused
// all the same leaves both in one file.
func (s *Sidecar) openLog() (*os.File, error) {
	if err := os.MkdirAll(filepath.Dir(s.cfg.LogPath), 0o700); err != nil {
		return nil, err
	}
	if !s.rotated {
		s.rotated = true
		_ = os.Rename(s.cfg.LogPath, strings.TrimSuffix(s.cfg.LogPath, ".log")+".previous.log")
	}
	return openAppend(s.cfg.LogPath)
}

// awaitReady posts evReady once the child answers its capabilities with the
// token and says it is supervised: only our child knows the token, so no
// localized startup text is parsed. It posts evNotReady after ReadyTimeout,
// with the child's part of the log from offset from.
func (s *Sidecar) awaitReady(ctx context.Context, c child, from int64) {
	deadline := time.After(s.cfg.ReadyTimeout)
	tick := time.NewTicker(readyPoll)
	defer tick.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-deadline:
			s.post(evNotReady{PID: c.PID, Tail: s.tail(c.Token, from)})
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

// stop asks the child to stop with a line on its stdin, the same on every
// system: "drain" to finish and exit, "now" to stop at once (serve exits
// 129). A child stopped now that still runs killGrace later is killed.
func (s *Sidecar) stop(pid int, mode Mode) {
	p := s.procs[pid]
	if p == nil {
		return
	}
	if mode == WhenDone {
		_, _ = io.WriteString(p.stdin, "drain\n")
		return
	}
	_, _ = io.WriteString(p.stdin, "now\n")
	time.AfterFunc(killGrace, func() { _ = p.cmd.Process.Kill() })
}

// tail is the end of the log from offset from on, what one child wrote.
func (s *Sidecar) tail(token Token, from int64) string {
	f, err := os.Open(s.cfg.LogPath)
	if err != nil {
		return ""
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return ""
	}
	start := max(from, info.Size()-tailBytes)
	if _, err := f.Seek(start, io.SeekStart); err != nil {
		return ""
	}
	b, _ := io.ReadAll(f)
	return logTail(b, start > from, token)
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

// exitOf is how a child's Wait ended.
func exitOf(err error) Exited {
	var exit *exec.ExitError
	if errors.As(err, &exit) {
		if name := signalName(exit.ProcessState); name != "" {
			return Exited{Code: -1, Signal: name}
		}
		return Exited{Code: exit.ExitCode()}
	}
	if err != nil {
		return Exited{Code: -1}
	}
	return Exited{}
}

// codeText writes an exit code as goos does: Windows' codes from 0x80000000
// on are NTSTATUS values, as 0xC0000005, or a TerminateProcess with -1, which
// read as hexadecimal.
func codeText(goos string, code int) string {
	if goos == "windows" && code >= 0x80000000 {
		return fmt.Sprintf("0x%08X", uint32(code))
	}
	return strconv.Itoa(code)
}

func (s *Sidecar) post(e event) {
	select {
	case s.events <- e:
	case <-s.done:
	}
}

// Retry starts a new child after a failure.
func (s *Sidecar) Retry() { s.post(evRetry{}) }

// Stop stops the child, or with none, ends the sidecar at once; with a child
// still exiting, once it has.
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

// Unfinished counts what our ready child would finish before it stops: the
// conversions it has running or queued, and uploads still arriving; or says
// why it cannot; 0 without a ready child.
func (s *Sidecar) Unfinished(ctx context.Context) (int, error) {
	r, ok := s.Phase().(Ready)
	if !ok {
		return 0, nil
	}
	return waiting(ctx, r.Child)
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
