//go:build e2e

// Package e2e drives the built Markitai Mygo app in a real window through
// mygo's automation plugin (build.py --e2e). Each run launches the app with
// HOME, MARKITAI_HOME and every user folder in a folder of its own, converts
// repository fixtures, walks Convert, History and Settings, quits while a
// conversion runs, and writes window captures, the app's log, timings and
// the files the app wrote into MARKITAI_MYGO_E2E_OUT.
//
//	MARKITAI_MYGO_APP=<executable> MARKITAI_MYGO_E2E_OUT=<folder> go test -tags e2e ./internal/e2e
//
// MARKITAI_MYGO_E2E_RUNS lists the runs, such as "en-light,zh-Hans-dark"
// (the default is all four). It never touches the clipboard, opens no
// dialog, Finder, file manager or browser.
package e2e

import (
	"bufio"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io/fs"
	"math"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"

	"markitai.local/mygo/internal/l10n"
	"markitai.local/mygo/internal/testenv"
)

type rect struct{ X, Y, Width, Height float64 }

type node struct {
	ID, Parent, Role, Key, Label, Value string
	States, Actions                     []string
	Bounds                              rect
}

type client struct {
	t    *testing.T
	conn net.Conn
	r    *bufio.Reader
	n    int
}

func (c *client) call(method string, params map[string]any) (json.RawMessage, error) {
	c.n++
	line, _ := json.Marshal(map[string]any{"id": c.n, "method": method, "params": params})
	c.conn.SetDeadline(time.Now().Add(90 * time.Second))
	if _, err := c.conn.Write(append(line, '\n')); err != nil {
		return nil, err
	}
	answer, err := c.r.ReadBytes('\n')
	if err != nil {
		return nil, err
	}
	var res struct {
		Result json.RawMessage
		Error  *struct{ Code, Message string }
	}
	if err := json.Unmarshal(answer, &res); err != nil {
		return nil, err
	}
	if res.Error != nil {
		return nil, fmt.Errorf("%s: %s %s", method, res.Error.Code, res.Error.Message)
	}
	return res.Result, nil
}

func (c *client) must(method string, params map[string]any) json.RawMessage {
	c.t.Helper()
	res, err := c.call(method, params)
	if err != nil {
		c.t.Fatal(err)
	}
	return res
}

func (c *client) tree() []node {
	c.t.Helper()
	var res struct{ Nodes []node }
	json.Unmarshal(c.must("tree", nil), &res)
	return res.Nodes
}

// wait polls the window's elements until ok finds what it waits for.
func (c *client) wait(what string, timeout time.Duration, ok func([]node) bool) []node {
	c.t.Helper()
	deadline := time.Now().Add(timeout)
	for {
		nodes := c.tree()
		if ok(nodes) {
			return nodes
		}
		if time.Now().After(deadline) {
			var labels []string
			for _, n := range nodes {
				if n.Label != "" {
					labels = append(labels, n.Role+":"+n.Label)
				}
			}
			c.t.Fatalf("timed out waiting for %s; elements %q", what, labels)
		}
		time.Sleep(250 * time.Millisecond)
	}
}

func find(nodes []node, ok func(node) bool) (node, bool) {
	i := slices.IndexFunc(nodes, ok)
	if i < 0 {
		return node{}, false
	}
	return nodes[i], true
}

func byKey(key string) func(node) bool { return func(n node) bool { return n.Key == key } }

func byLabel(role, label string) func(node) bool {
	return func(n node) bool { return n.Label == label && (role == "" || n.Role == role) }
}

func (c *client) act(n node, action string, text ...string) {
	c.t.Helper()
	p := map[string]any{"id": n.ID, "action": action}
	if len(text) > 0 {
		p["text"] = text[0]
	}
	c.must("act", p)
	c.settle()
}

// press finds an element and presses it.
func (c *client) press(what string, ok func(node) bool) {
	c.t.Helper()
	n, found := find(c.wait(what, 20*time.Second, func(ns []node) bool { _, f := find(ns, ok); return f }), ok)
	if !found {
		c.t.Fatalf("no %s", what)
	}
	c.act(n, "press")
}

func (c *client) settle() {
	c.call("idle", map[string]any{"timeout": 1500})
	time.Sleep(300 * time.Millisecond)
}

func (c *client) capture(path string) {
	c.t.Helper()
	c.settle()
	var res struct {
		PNG           string
		Width, Height int
	}
	json.Unmarshal(c.must("capture", nil), &res)
	data, err := base64.StdEncoding.DecodeString(res.PNG)
	if err != nil {
		c.t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		c.t.Fatal(err)
	}
	c.t.Logf("capture %s: %dx%d", filepath.Base(path), res.Width, res.Height)
}

// session is one launch of the app with a home of its own.
type session struct {
	t         *testing.T
	lang      l10n.Lang
	name, out string
	home, exe string
	// temp is the app's temporary folder on Windows, outside home.
	temp       string
	env        []string
	cmd        *exec.Cmd
	exited     chan error
	c          *client
	launchedAt time.Time
	report     map[string]any
	// pageEvent is the file through which the app dispatches a key in the
	// preview's page (e2e.go), for pageKey without a key program.
	pageEvent string
}

// isolatedEnv is this process's environment with every user folder moved
// into home, and Windows' temporary folder to temp; the display and session
// bus stay, which the window needs.
func isolatedEnv(home, temp string, extra ...string) []string {
	moved := map[string]string{
		"HOME":            home,
		"USERPROFILE":     home,
		"MARKITAI_HOME":   filepath.Join(home, ".markitai"),
		"XDG_DATA_HOME":   filepath.Join(home, ".local", "share"),
		"XDG_CONFIG_HOME": filepath.Join(home, ".config"),
		"XDG_CACHE_HOME":  filepath.Join(home, ".cache"),
		"XDG_STATE_HOME":  filepath.Join(home, ".local", "state"),
		"APPDATA":         filepath.Join(home, "AppData", "Roaming"),
		"LOCALAPPDATA":    filepath.Join(home, "AppData", "Local"),
		// Unix keeps its own temporary folder, which the system clears.
		"TEMP": temp,
		"TMP":  temp,
	}
	var env []string
	for _, kv := range os.Environ() {
		k, _, _ := strings.Cut(kv, "=")
		if _, ok := moved[strings.ToUpper(k)]; ok || strings.EqualFold(k, "TERM") || strings.HasPrefix(k, "MYGO_") || strings.HasPrefix(k, "MARKITAI_") {
			continue
		}
		env = append(env, kv)
	}
	for k, v := range moved {
		os.MkdirAll(v, 0o755)
		env = append(env, k+"="+v)
	}
	return append(env, extra...)
}

func launch(t *testing.T, run string, args func(home string) []string) *session {
	t.Helper()
	exe := os.Getenv("MARKITAI_MYGO_APP")
	out := os.Getenv("MARKITAI_MYGO_E2E_OUT")
	if exe == "" || out == "" {
		t.Skip("MARKITAI_MYGO_APP and MARKITAI_MYGO_E2E_OUT name the app and the evidence folder")
	}
	lang, theme, _ := strings.Cut(run, "-light")
	if theme = "light"; lang == run {
		lang, _, _ = strings.Cut(run, "-dark")
		theme = "dark"
	}
	s := &session{t: t, lang: l10n.Lang(lang), name: run, out: filepath.Join(out, run), exe: exe, report: map[string]any{"run": run, "goos": runtime.GOOS}}
	os.MkdirAll(s.out, 0o755)
	s.home = filepath.Join(s.out, "home")
	if err := os.MkdirAll(s.home, 0o755); err != nil {
		t.Fatal(err)
	}
	// A folder of this run's own under a short path: the automation socket,
	// and Windows' temporary folder. mygo's single-instance lock is a socket
	// in the temporary folder, and mygo skips the lock on Windows when that
	// path reaches 100 bytes, as it does under the run's home on a runner
	// (DESIGN 14).
	short, err := os.MkdirTemp("", "mkmygo-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(short) })
	sock := filepath.Join(short, "a.sock")
	s.temp = filepath.Join(short, "temp")
	s.pageEvent = filepath.Join(s.out, "page-event")
	s.env = isolatedEnv(s.home, s.temp, "MYGO_AUTOMATION="+sock, "MYGO_LOCALE="+lang, "MARKITAI_MYGO_E2E_THEME="+theme,
		"MARKITAI_MYGO_E2E_PAGE_EVENT="+s.pageEvent)
	logf, err := os.Create(filepath.Join(s.out, "app.log"))
	if err != nil {
		t.Fatal(err)
	}
	s.cmd = exec.Command(exe, args(s.home)...)
	s.cmd.Env, s.cmd.Stdout, s.cmd.Stderr = s.env, logf, logf
	s.launchedAt = time.Now()
	if err := s.cmd.Start(); err != nil {
		t.Fatal(err)
	}
	s.exited = make(chan error, 1)
	go func() { s.exited <- s.cmd.Wait(); logf.Close() }()
	t.Cleanup(func() {
		select {
		case <-s.exited:
		default:
			s.cmd.Process.Kill()
			<-s.exited
		}
		s.writeReport()
	})
	deadline := time.Now().Add(30 * time.Second)
	for {
		conn, err := net.Dial("unix", sock)
		if err == nil {
			s.c = &client{t: t, conn: conn, r: bufio.NewReader(conn)}
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("no automation socket after 30 s: %v (is this a build.py --e2e build?)", err)
		}
		time.Sleep(20 * time.Millisecond)
	}
	for {
		var res struct{ Windows []struct{ Visible bool } }
		raw, err := s.c.call("windows", nil)
		if err == nil && json.Unmarshal(raw, &res) == nil && len(res.Windows) > 0 && res.Windows[0].Visible {
			s.report["window_visible_ms"] = time.Since(s.launchedAt).Milliseconds()
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("no visible window after 30 s: %v", err)
		}
		time.Sleep(20 * time.Millisecond)
	}
	return s
}

func (s *session) writeReport() {
	s.report["launched_at_ms"] = s.launchedAt.UnixMilli()
	if log, err := os.ReadFile(filepath.Join(s.out, "app.log")); err == nil {
		for _, line := range strings.Split(string(log), "\n") {
			var ms int64
			if _, rest, ok := strings.Cut(line, "first frame at "); ok && s.report["first_frame_ms"] == nil {
				fmt.Sscanf(rest, "%d", &ms)
				s.report["first_frame_ms"] = ms - s.launchedAt.UnixMilli()
			}
		}
	}
	data, _ := json.MarshalIndent(s.report, "", "  ")
	os.WriteFile(filepath.Join(s.out, "report.json"), append(data, '\n'), 0o644)
}

// shot writes what the window's content draws (name.png) and, with
// MARKITAI_MYGO_E2E_WINDOW_SHOT naming a program that takes a process ID
// and a PNG path, what the system composites, web view included
// (name.system.png). A program ending in .ps1 runs in Windows PowerShell.
func (s *session) shot(name string) {
	s.c.capture(filepath.Join(s.out, name+".png"))
	if prog := os.Getenv("MARKITAI_MYGO_E2E_WINDOW_SHOT"); prog != "" {
		time.Sleep(500 * time.Millisecond)
		out, err := tool(prog, fmt.Sprint(s.cmd.Process.Pid), filepath.Join(s.out, name+".system.png"))
		if err != nil {
			s.t.Logf("system capture %s: %v %s", name, err, out)
		}
	}
}

// tool runs a program the environment names; one ending in .ps1 runs in
// Windows PowerShell.
func tool(prog string, args ...string) ([]byte, error) {
	if strings.EqualFold(filepath.Ext(prog), ".ps1") {
		prog, args = "powershell.exe", append([]string{"-NoProfile", "-ExecutionPolicy", "Bypass", "-File", prog}, args...)
	}
	return exec.Command(prog, args...).CombinedOutput()
}

// pageKey presses Zoom In while the preview page holds the keyboard, as the
// keyboard would, through MARKITAI_MYGO_E2E_PAGE_KEY: a program that takes
// a process ID and the word zoom-in and sends that process's window the
// platform's Zoom In keys (Ctrl+=; Cmd+= on macOS). Without one, the app
// dispatches that keydown in the page itself (e2e.go): the page's script and
// its mygo.call run, the system's delivery of the key is not checked. Only
// the page's injected script can act on the key (section 8.3), so view.zoom
// must go up by exactly 0.1: no change means the script or its mygo.call
// did not run, 0.2 that the key ran twice.
func (s *session) pageKey() {
	t, c := s.t, s.c
	before := s.zoom()
	wv, ok := find(c.tree(), byKey("preview"))
	if !ok {
		t.Error("no preview page to give the keyboard")
		return
	}
	c.act(wv, "focus")
	time.Sleep(500 * time.Millisecond)
	if prog := os.Getenv("MARKITAI_MYGO_E2E_PAGE_KEY"); prog != "" {
		out, err := tool(prog, fmt.Sprint(s.cmd.Process.Pid), "zoom-in")
		s.report["page_key_tool"] = strings.TrimSpace(string(out))
		if err != nil {
			t.Errorf("the page key: %v %s", err, out)
			return
		}
	} else {
		s.report["page_key_tool"] = "none: keydown dispatched in the page by the app"
		if err := os.WriteFile(s.pageEvent, []byte("zoom-in"), 0o644); err != nil {
			t.Errorf("the page event: %v", err)
			return
		}
	}
	for deadline := time.Now().Add(5 * time.Second); time.Now().Before(deadline) && s.zoom() == before; {
		time.Sleep(100 * time.Millisecond)
	}
	time.Sleep(time.Second)
	after := s.zoom()
	s.report["page_key_zoom"] = fmt.Sprintf("%.1f to %.1f", before, after)
	if math.Round((after-before)*10) != 1 {
		t.Errorf("Zoom In in the page took view.zoom from %.1f to %.1f, want one step", before, after)
	}
	s.shot("03b-page-zoom-in")
}

// zoom is view.zoom in the app's settings.json, 1 before the file exists.
func (s *session) zoom() float64 {
	for _, data := range s.settingsFiles() {
		var v struct {
			Zoom float64 `json:"view.zoom"`
		}
		if json.Unmarshal([]byte(data), &v) == nil && v.Zoom > 0 {
			return v.Zoom
		}
	}
	return 1
}

// rss is the app's resident memory in KiB, from ps or tasklist.
func (s *session) rss() int64 {
	var kib int64
	if runtime.GOOS == "windows" {
		out, _ := exec.Command("tasklist", "/FI", fmt.Sprintf("PID eq %d", s.cmd.Process.Pid), "/FO", "CSV", "/NH").Output()
		fields := strings.Split(strings.TrimSpace(string(out)), `","`)
		if len(fields) >= 5 {
			fmt.Sscanf(strings.NewReplacer(",", "", ".", "", " K", "", `"`, "").Replace(fields[4]), "%d", &kib)
		}
		return kib
	}
	out, _ := exec.Command("ps", "-o", "rss=", "-p", fmt.Sprint(s.cmd.Process.Pid)).Output()
	fmt.Sscanf(strings.TrimSpace(string(out)), "%d", &kib)
	return kib
}

func (s *session) waitExit(timeout time.Duration) error {
	select {
	case err := <-s.exited:
		s.exited <- err
		return err
	case <-time.After(timeout):
		return fmt.Errorf("the app did not exit in %v", timeout)
	}
}

// files lists what the app wrote under its home, relative and slash
// separated, leaving out the fixtures the run put there, then what it left
// in its temporary folder on Windows, as "temp/...".
func (s *session) files() []string {
	var paths []string
	walk := func(root, prefix string) {
		filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
			if err != nil || d.IsDir() {
				return nil
			}
			rel, _ := filepath.Rel(root, p)
			rel = prefix + filepath.ToSlash(rel)
			if !strings.HasPrefix(rel, "Downloads/") {
				paths = append(paths, rel)
			}
			return nil
		})
	}
	walk(s.home, "")
	if runtime.GOOS == "windows" {
		walk(s.temp, "temp/")
	}
	return paths
}

func copyFixture(t *testing.T, home, rel string) string {
	data, err := os.ReadFile(testenv.Fixture(t, rel))
	if err != nil {
		t.Fatal(err)
	}
	dst := filepath.Join(home, "Downloads", filepath.Base(rel))
	os.MkdirAll(filepath.Dir(dst), 0o755)
	os.WriteFile(dst, data, 0o644)
	return dst
}

const features = `# Preview features

| Format | Pages |
|:--|--:|
| PDF | 12 |
| Word | 3 |

- [x] Tables
- [ ] Task lists

A footnote reference.[^1] A [link](https://example.com/) and a remote image ![remote](https://example.com/remote.png).

<b>raw html stays text</b>

[^1]: The footnote.
`

const article = `<!doctype html><html><head><meta charset="utf-8"><title>Local article</title></head>
<body><article><h1>Local article</h1><p>The e2e driver serves this page from 127.0.0.1.</p>
<p>It has a second paragraph, so the page reads as an article.</p></article></body></html>`

// server serves the article and a page that never answers until the run
// ends, which keeps a conversion Converting.
func server(t *testing.T) string {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	stop := make(chan struct{})
	mux := http.NewServeMux()
	mux.HandleFunc("/article.html", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		fmt.Fprint(w, article)
	})
	mux.HandleFunc("/slow.html", func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
		case <-stop:
		case <-time.After(3 * time.Minute):
		}
	})
	srv := &http.Server{Handler: mux}
	go srv.Serve(ln)
	t.Cleanup(func() { close(stop); srv.Close() })
	return "http://" + ln.Addr().String()
}

func runs() []string {
	if r := os.Getenv("MARKITAI_MYGO_E2E_RUNS"); r != "" {
		return strings.Split(r, ",")
	}
	return []string{"en-light", "zh-Hans-dark", "en-dark", "zh-Hans-light"}
}

func TestRealWindow(t *testing.T) {
	base := server(t)
	for _, run := range runs() {
		t.Run(run, func(t *testing.T) { realWindow(t, run, base) })
	}
}

func realWindow(t *testing.T, run, base string) {
	var assets, table, feat, folder string
	s := launch(t, run, func(home string) []string {
		assets = copyFixture(t, home, "crates/markitai-core/src/formats/pdf/fixtures/sanitize/policy-assets.pdf")
		table = copyFixture(t, home, "crates/markitai-core/src/formats/pdf/fixtures/default-table.pdf")
		feat = filepath.Join(home, "Downloads", "features.md")
		os.WriteFile(feat, []byte(features), 0o644)
		folder = filepath.Join(home, "Downloads", "A folder")
		os.MkdirAll(folder, 0o755)
		return []string{assets, folder}
	})
	l, c := s.lang, s.c
	row := func(title string, state ...string) func(node) bool {
		return func(n node) bool {
			if n.Role != "listItem" || !strings.HasPrefix(n.Label, title+", ") {
				return false
			}
			return len(state) == 0 || slices.ContainsFunc(state, func(p string) bool { return strings.Contains(n.Label, p) })
		}
	}
	done := strings.Split(l.Raw("state.done"), "{")[0]
	failed := strings.Split(l.Raw("row.failed"), "{")[0]
	finished := func(title string) func([]node) bool {
		return func(ns []node) bool { _, ok := find(ns, row(title, done, failed)); return ok }
	}

	nodes := c.wait("policy-assets.pdf finished", 2*time.Minute, finished("policy-assets.pdf"))
	r, _ := find(nodes, row("policy-assets.pdf"))
	s.report["policy_assets_row"] = r.Label
	notice, ok := find(nodes, func(n node) bool { return strings.Contains(n.Label, l.T("notice.folder", "name", "A folder")) })
	s.report["folder_notice"] = notice.Label
	if !ok {
		t.Errorf("no folder notice")
	}
	// The detail builds the web view once the page has loaded; the first
	// page of a launch also waits for the web engine to start.
	doneAt := time.Now()
	nodes = c.wait("the preview web view in the Done detail", 30*time.Second, func(ns []node) bool { _, ok := find(ns, byKey("preview")); return ok })
	s.report["preview_after_done_ms"] = time.Since(doneAt).Milliseconds()
	wv, _ := find(nodes, byKey("preview"))
	s.report["preview_bounds"] = wv.Bounds
	s.shot("01-convert-done-preview")
	time.Sleep(3 * time.Second)
	s.report["rss_kib_after_first_done"] = s.rss()

	c.press("the Markdown segment", byLabel("", l.T("detail.markdown")))
	s.shot("02-convert-done-markdown")
	if _, ok := find(c.tree(), byKey("markdown")); !ok {
		t.Errorf("no Markdown text after switching")
	}
	c.press("the Preview segment", byLabel("", l.T("detail.preview")))

	// The second instance hands its files to the first and exits.
	handOver, cancel := context.WithTimeout(t.Context(), time.Minute)
	defer cancel()
	second := exec.CommandContext(handOver, s.exe, table, feat)
	second.Env = s.env
	secondStarted := time.Now()
	if err := second.Run(); handOver.Err() != nil {
		t.Fatalf("the second launch did not exit within a minute: it did not hand over")
	} else if err != nil {
		t.Errorf("the second launch: %v", err)
	}
	s.report["second_instance_exit_ms"] = time.Since(secondStarted).Milliseconds()
	c.wait("the second launch's files", 2*time.Minute, func(ns []node) bool {
		return finished("default-table.pdf")(ns) && finished("features.md")(ns)
	})
	c.press("the features.md row", row("features.md"))
	time.Sleep(2 * time.Second)
	s.shot("03-features-preview")
	s.pageKey()

	field, ok := find(c.tree(), byKey("convert.url"))
	if !ok {
		t.Fatal("no URL field")
	}
	c.act(field, "setValue", base+"/article.html")
	c.press("Add", byKey("convert.url.add"))
	nodes = c.wait("the URL item finished", 2*time.Minute, func(ns []node) bool {
		return finished("Local article")(ns) || finished(strings.TrimPrefix(base, "http://")+"/article.html")(ns)
	})
	for _, n := range nodes {
		if n.Role == "listItem" {
			s.report["rows"] = append(s.reportList("rows"), n.Label)
		}
	}
	s.shot("04-convert-url")

	c.press("History", byLabel("", l.T("nav.history")))
	s.shot("05-history")
	search, ok := find(c.tree(), func(n node) bool { return n.Role == "textField" && n.Label == l.T("history.search") })
	if ok {
		c.act(search, "setValue", "policy")
		s.shot("06-history-search")
		c.act(search, "setValue", "zzz-no-match")
		s.shot("07-history-no-match")
		c.act(search, "setValue", "")
	} else {
		t.Errorf("no History search field")
	}

	c.press("Settings", byLabel("", l.T("nav.settings")))
	s.shot("08-settings")
	c.press("the OCR switch", byKey("settings.ocr"))
	s.report["settings_files"] = s.settingsFiles()

	c.press("Convert", byLabel("", l.T("nav.convert")))
	field, _ = find(c.tree(), byKey("convert.url"))
	c.act(field, "setValue", base+"/slow.html")
	c.press("Add", byKey("convert.url.add"))
	converting := strings.Split(l.Raw("state.converting"), "{")[0]
	c.wait("the slow item converting", 30*time.Second, func(ns []node) bool {
		_, ok := find(ns, row(strings.TrimPrefix(base, "http://")+"/slow.html", converting))
		return ok
	})
	s.shot("09-converting")
	s.quitWhileConverting(base)
	s.report["files"] = s.files()
	s.checkFiles()
}

func (s *session) reportList(key string) []string {
	v, _ := s.report[key].([]string)
	return v
}

func (s *session) settingsFiles() map[string]string {
	found := map[string]string{}
	filepath.WalkDir(s.home, func(p string, d fs.DirEntry, err error) error {
		if err == nil && !d.IsDir() && d.Name() == "settings.json" {
			data, _ := os.ReadFile(p)
			rel, _ := filepath.Rel(s.home, p)
			found[filepath.ToSlash(rel)] = string(data)
		}
		return nil
	})
	return found
}

// quitWhileConverting asks to quit, cancels, asks again and quits: the
// claimed folder of the converting item is gone after the app exits.
func (s *session) quitWhileConverting(base string) {
	t, c, l := s.t, s.c, s.lang
	ask := func() bool {
		switch runtime.GOOS {
		case "darwin":
			for _, quit := range []string{"Quit Markitai Mygo", "退出 Markitai Mygo", "退出Markitai Mygo"} {
				if _, err := c.call("menu", map[string]any{"path": []string{"Markitai Mygo", quit}}); err == nil {
					return true
				}
			}
			return false
		case "linux":
			_, err := c.call("input", map[string]any{"events": []map[string]any{{"type": "key", "key": "Ctrl+Q"}}})
			return err == nil
		case "windows":
			// taskkill without /F posts WM_CLOSE, what the close button and
			// Alt+F4 send.
			return exec.Command("taskkill", "/PID", fmt.Sprint(s.cmd.Process.Pid)).Run() == nil
		}
		return false
	}
	if !ask() {
		s.report["quit"] = "no quit path the driver can reach on " + runtime.GOOS
		t.Log(s.report["quit"])
		return
	}
	alert := func(ns []node) bool {
		_, ok := find(ns, func(n node) bool { return n.Role == "alertDialog" })
		return ok
	}
	c.wait("the quit alert", 15*time.Second, alert)
	s.shot("10-quit-alert")
	c.press("Cancel", byLabel("button", l.T("common.cancel")))
	if alert(c.tree()) {
		t.Error("the alert stayed after Cancel")
	}
	if err := s.waitExit(time.Second); err == nil {
		t.Fatal("the app quit after Cancel")
	}
	if !ask() {
		t.Fatal("the second quit could not be asked")
	}
	c.wait("the quit alert again", 15*time.Second, alert)
	c.press("Quit", byLabel("button", l.T("quit.confirm")))
	err := s.waitExit(20 * time.Second)
	s.report["quit"] = fmt.Sprintf("asked, canceled, asked, quit: exit %v", err)
	if err != nil && !strings.Contains(err.Error(), "exit status") {
		t.Errorf("quit: %v", err)
	}
	slow := strings.ReplaceAll(strings.TrimPrefix(base, "http://"), ":", "_")
	matches, _ := filepath.Glob(filepath.Join(s.home, "Documents", "Markitai", "*slow*"))
	s.report["slow_folders_after_quit"] = matches
	if len(matches) > 0 {
		t.Errorf("the converting item's folder stayed after Quit: %v (%s)", matches, slow)
	}
}

// checkFiles reports what the app wrote outside its own folders (section
// 2.1, with the web data on Linux), the output folder and the markitai
// home, and fails when any of it is another version's.
func (s *session) checkFiles() {
	own := map[string][]string{
		"darwin": {"Library/Application Support/dev.markitai.mygo/", "Library/Caches/dev.markitai.mygo/", "Documents/Markitai/", ".markitai/"},
		"linux": {".local/share/dev.markitai.mygo/", ".config/dev.markitai.mygo/", ".cache/dev.markitai.mygo/",
			".config/Markitai Mygo/", ".local/share/markitai-mygo/", ".cache/markitai-mygo/", "Documents/Markitai/", ".markitai/"},
		"windows": {"AppData/Local/Markitai Mygo/", "Documents/Markitai/", ".markitai/"},
	}[runtime.GOOS]
	var other []string
	for _, f := range s.files() {
		if !slices.ContainsFunc(own, func(p string) bool { return strings.HasPrefix(f, p) }) {
			other = append(other, f)
		}
	}
	s.report["files_outside_own_folders"] = other
	for _, f := range other {
		if strings.Contains(f, "dev.markitai.web") || strings.Contains(f, "dev.markitai.native") || strings.Contains(f, "Markitai Web") || strings.Contains(f, "Markitai Native") {
			t := s.t
			t.Errorf("wrote another version's file %s", f)
		}
	}
}
