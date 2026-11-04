package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"
	"unicode/utf16"

	"github.com/egoist/mygo/transfer"
	"github.com/egoist/mygo/ui"

	"markitai.local/mygo/internal/core"
	"markitai.local/mygo/internal/l10n"
	"markitai.local/mygo/internal/model"
)

// coreLib is the markitai core the app calls; *core.Lib in the app.
type coreLib interface {
	ClaimOutputFolder(parent, source string) (core.Folder, error)
	ConvertJSON(request any) (json.RawMessage, error)
	PreviewHTML(markdown, theme string, baseDir *string) (string, error)
	ConfigFile() (core.ConfigFile, error)
	Version() string
}

// desk is the desktop around the window. Its dialogs answer later, on the
// UI thread, and not at all when canceled.
type desk interface {
	chooseFiles(done func(paths []string))
	chooseFolder(done func(path string))
	saveAs(name string, done func(path string))
	showItemInFolder(path string)
	openPath(path string) error
	openExternal(url string) error
	showWindow()
	closeWindow()
	quit()
}

// asking is the alert the window shows.
type asking uint8

const (
	askingNone asking = iota
	askingQuit
	askingClearHistory
)

// detailView is the tab of a Done item's detail.
type detailView int

const (
	showPreview detailView = iota
	showMarkdown
)

// Router paths of the pages (section 4).
const (
	pathConvert     = "/convert"
	pathConvertItem = "/convert/item"
	pathHistory     = "/history"
	pathHistoryItem = "/history/item"
	pathSettings    = "/settings"
)

// files are where the app keeps its own data (section 2.1).
type files struct {
	data, settings, history string
	// output is the default output folder, used while settings say none.
	output string
}

// notice is the one-line notice of section 6.7: it shows on the page it
// was given on, for 8 s from then.
type notice struct {
	text string
	at   time.Time
	page string
}

// app is the window's state, read and changed on the UI thread only.
type app struct {
	lang        l10n.Lang
	core        coreLib
	desk        desk
	pane        previewPane
	post        func(func())
	files       files
	version     string
	coreVersion string
	goos        string
	fatal       bool

	settings model.Settings
	queue    model.Queue
	history  *model.History

	router       *ui.Router
	page         string
	look         look
	listWidth    float32
	detail       detailView
	warningsOpen map[model.ItemID]*bool
	query        string
	historyID    model.ItemID
	historyList  ui.ListState
	historySel   int
	url          string
	urlInvalid   bool
	urlField     ui.Handle
	searchField  ui.Handle
	leaving      bool
	notice       notice
	copied       bool
	copiedAt     time.Time
	asking       asking
	quitting     bool

	announce  []string
	later     []action
	previews  previews
	markdowns map[model.ItemID]*markdownFile
	menu      *menuBar
	// config is the configuration file Settings shows, read again when
	// Settings opens, after Create or Open, and when the window comes back
	// (6.6), not on every frame.
	config struct {
		file core.ConfigFile
		err  error
		read bool
	}
}

func newApp(lang l10n.Lang, lib coreLib, d desk, pane previewPane, post func(func()), f files) *app {
	a := &app{
		lang: lang, core: lib, desk: d, pane: pane, post: post, files: f, goos: runtime.GOOS,
		router:       ui.NewRouter(pathConvert),
		page:         pathConvert,
		listWidth:    320,
		warningsOpen: map[model.ItemID]*bool{},
		markdowns:    map[model.ItemID]*markdownFile{},
		historySel:   -1,
		settings:     model.LoadSettings(f.settings),
	}
	if lib != nil {
		a.coreVersion = lib.Version()
	}
	h, reset, err := model.LoadHistory(f.history, time.Now())
	if err != nil {
		fmt.Fprintf(os.Stderr, "history: %v\n", err)
	}
	if h == nil {
		h = &model.History{}
	}
	a.history = h
	if reset != "" {
		a.setNotice(a.lang.T("history.reset", "name", reset))
	}
	a.previews.pane = pane
	return a
}

// output is the folder conversions write into (DESIGN 4.3).
func (a *app) output() string {
	if a.settings.OutputFolder != nil {
		return *a.settings.OutputFolder
	}
	return a.files.output
}

// abbreviated shows a path with the home folder as ~ on macOS and Linux
// (DESIGN 4.3).
func (a *app) abbreviated(path string) string {
	if a.goos == "windows" {
		return path
	}
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		return path
	}
	if path == home {
		return "~"
	}
	if rest, ok := strings.CutPrefix(path, home+string(filepath.Separator)); ok {
		return "~" + string(filepath.Separator) + rest
	}
	return path
}

func (a *app) setNotice(text string) {
	a.notice = notice{text: text, at: time.Now(), page: a.page}
	a.announce = append(a.announce, text)
}

// addPaths adds files chosen, dropped, pasted or opened with the app.
func (a *app) addPaths(paths []string) {
	sources, folders := model.FileSources(paths)
	a.add(sources, folders)
}

// add puts sources in the Convert list and shows Convert (7.1). Its
// notices go with the page it leaves the window on.
func (a *app) add(sources []model.Source, folders []string) {
	a.notice = notice{}
	var dups []model.Source
	if len(sources) > 0 {
		_, dups = a.queue.Add(sources, a.settings.Options(), time.Now())
		a.show(pathConvert)
	}
	switch len(folders) {
	case 0:
	case 1:
		a.setNotice(a.lang.T("notice.folder", "name", folders[0]))
	default:
		a.setNotice(a.lang.P("notice.folders", len(folders)))
	}
	if len(dups) > 0 {
		a.setNotice(a.lang.T("notice.duplicate", "name", dups[0].Title()))
	}
	if len(sources) > 0 {
		a.pump()
	}
}

// addURL adds the URL field's text, or shows why it cannot.
func (a *app) addURL() {
	s, ok := model.ParseURL(a.url)
	if !ok {
		a.urlInvalid = true
		return
	}
	a.url = ""
	a.add([]model.Source{s}, nil)
}

// dropped takes a drop on the content area: files, else the http and
// https links of its URI list, else of its text/x-moz-url (10.1).
func (a *app) dropped(d transfer.Data) {
	if paths, err := d.Files(); err == nil && len(paths) > 0 {
		a.addPaths(paths)
		return
	}
	links, _ := d.URLs()
	if len(links) == 0 {
		if b, err := d.Read(mozURL); err == nil {
			links = []string{mozURLLink(b)}
		}
	}
	var sources []model.Source
	for _, l := range links {
		if s, ok := webLink(l); ok {
			sources = append(sources, s)
		}
	}
	a.add(sources, nil)
}

// mozURLLink is the URL of a text/x-moz-url: its first line, in the
// UTF-16 the browsers write or in UTF-8.
func mozURLLink(b []byte) string {
	text := string(b)
	// UTF-16 has its byte order mark, or a URL's first, ASCII letter
	// leaves a zero byte after it.
	if len(b) >= 2 && (b[1] == 0 || b[0] == 0xff && b[1] == 0xfe) {
		units := make([]uint16, len(b)/2)
		for i := range units {
			units[i] = uint16(b[2*i]) | uint16(b[2*i+1])<<8
		}
		text = string(utf16.Decode(units))
	}
	text = strings.TrimPrefix(text, "\ufeff")
	line, _, _ := strings.Cut(text, "\n")
	return strings.TrimRight(line, "\r\x00")
}

// paste adds the clipboard's files, else its text when it is one http or
// https URL with its scheme (7.1): other text with a dot is no link.
func (a *app) paste(c *ui.Context) {
	d := c.Services().ReadClipboardData(transfer.FileList, transfer.Text)
	if paths, err := d.Files(); err == nil && len(paths) > 0 {
		a.addPaths(paths)
		return
	}
	text, _ := d.Read(transfer.Text)
	if s, ok := webLink(string(text)); ok {
		a.add([]model.Source{s}, nil)
	}
}

// webLink is text that is one http or https URL, scheme and all.
func webLink(text string) (model.Source, bool) {
	text = strings.TrimSpace(text)
	lower := strings.ToLower(text)
	if !strings.HasPrefix(lower, "http://") && !strings.HasPrefix(lower, "https://") || strings.ContainsAny(text, " \n\t\r") {
		return model.Source{}, false
	}
	return model.ParseURL(text)
}

// pump starts Queued items while a slot is free.
func (a *app) pump() {
	jobs, failed := a.queue.Start(a.claim, a.output(), time.Now())
	for _, it := range failed {
		a.announce = append(a.announce, a.lang.T("a11y.failed", "title", it.Title))
	}
	for _, j := range jobs {
		go a.convert(j)
	}
}

func (a *app) claim(source string) (folder, code, message string) {
	f, err := a.core.ClaimOutputFolder(a.output(), source)
	if err != nil {
		var ce *core.Error
		if errors.As(err, &ce) {
			return "", ce.Code, ce.Message
		}
		return "", "output_unavailable", err.Error()
	}
	return f.Path, "", ""
}

// convert runs one job off the UI thread and posts its outcome.
func (a *app) convert(j model.Job) {
	var raw json.RawMessage
	var code, message string
	func() {
		defer func() {
			if r := recover(); r != nil {
				code, message = "internal_error", fmt.Sprint(r)
			}
		}()
		var err error
		raw, err = a.core.ConvertJSON(j.Request)
		var ce *core.Error
		switch {
		case errors.As(err, &ce):
			code, message = ce.Code, ce.Message
		case err != nil:
			code, message = "internal_error", err.Error()
		}
	}()
	state := model.Outcome(j.Folder, raw, code, message)
	a.post(func() { a.finish(j, state) })
}

// finish records a job's outcome on the UI thread (DESIGN 4.1).
func (a *app) finish(j model.Job, state model.State) {
	if _, failed := state.(model.Failed); failed {
		os.RemoveAll(j.Folder)
	}
	it := a.queue.Finish(j.ID, state)
	title := ""
	if it != nil {
		title = it.Title
	}
	switch s := state.(type) {
	case model.Done:
		if it != nil {
			if err := a.history.Add(model.NewEntry(it, s, a.coreVersion, time.Now())); err != nil {
				fmt.Fprintf(os.Stderr, "history: %v\n", err)
			}
		}
		a.announce = append(a.announce, a.lang.T("a11y.done", "title", title))
	case model.Failed:
		a.announce = append(a.announce, a.lang.T("a11y.failed", "title", title))
	}
	a.pump()
	if a.asking == askingQuit && model.Count[model.Converting](&a.queue) == 0 {
		a.asking = askingNone
		a.quitting = true
		a.desk.quit()
	}
}

// mayQuit answers a close or quit request: yes unless an item converts,
// in which case the window shows and asks (10.3).
func (a *app) mayQuit() bool {
	if a.quitting {
		return true
	}
	if model.Count[model.Converting](&a.queue) > 0 {
		a.asking = askingQuit
		a.desk.showWindow()
		return false
	}
	a.quitting = true
	return true
}

// confirmQuit is the quit alert's Quit: the folders claimed for items
// still converting go, and the app quits.
func (a *app) confirmQuit() {
	a.asking = askingNone
	a.removeConvertingFolders()
	a.quitting = true
	a.desk.quit()
}

func (a *app) removeConvertingFolders() {
	for _, f := range a.queue.ConvertingFolders() {
		os.RemoveAll(f)
	}
}

// show goes to a page, as the sidebar does.
func (a *app) show(path string) {
	if path == pathSettings && a.page != pathSettings {
		a.config.read = false
	}
	a.page = path
	a.router.Replace(path)
}

// remove takes an item out of the Convert list, or an entry out of
// History, whichever page shows it.
func (a *app) remove(id model.ItemID) {
	if a.onHistory() {
		a.removeEntry(id)
		return
	}
	a.queue.Remove(id)
	if a.queue.Item(a.queue.Selected) == nil && a.router.Path() == pathConvertItem {
		a.router.Back()
	}
}

func (a *app) removeEntry(id model.ItemID) {
	filtered := a.filtered()
	i := -1
	for k, e := range filtered {
		if e.ID == id {
			i = k
		}
	}
	if err := a.history.Remove(id); err != nil {
		fmt.Fprintf(os.Stderr, "history: %v\n", err)
	}
	delete(a.markdowns, id)
	if a.historyID != id {
		return
	}
	a.historyID = ""
	switch {
	case i >= 0 && i+1 < len(filtered):
		a.historyID = filtered[i+1].ID
	case i > 0:
		a.historyID = filtered[i-1].ID
	}
	if a.historyID == "" && a.router.Path() == pathHistoryItem {
		a.router.Back()
	}
}

func (a *app) retry(id model.ItemID) {
	if a.queue.Retry(id, a.settings.Options()) {
		a.pump()
	}
}

func (a *app) clearFinished() {
	a.queue.ClearFinished()
	if a.queue.Item(a.queue.Selected) == nil && a.router.Path() == pathConvertItem {
		a.router.Back()
	}
}

func (a *app) clearHistory() {
	a.asking = askingNone
	if err := a.history.Clear(); err != nil {
		fmt.Fprintf(os.Stderr, "history: %v\n", err)
	}
	a.historyID = ""
	clear(a.markdowns)
}

func (a *app) onHistory() bool { return strings.HasPrefix(a.router.Path(), pathHistory) }

// target is what the detail shows: the selected item of Convert, or the
// selected entry of History as a Done item.
type target struct {
	id       model.ItemID
	title    string
	source   model.Source
	state    model.State
	finished time.Time
	entry    bool
	// missing is a History entry whose Markdown file is gone; loading one
	// still being read.
	missing, loading bool
}

func (a *app) target() *target {
	switch {
	case strings.HasPrefix(a.router.Path(), pathConvert):
		it := a.queue.Item(a.queue.Selected)
		if it == nil {
			return nil
		}
		return &target{id: it.ID, title: it.Title, source: it.Source, state: it.State}
	case a.onHistory():
		e := a.history.Entry(a.historyID)
		if e == nil || !a.matches(*e) {
			return nil
		}
		t := &target{id: e.ID, title: e.Title, source: model.Source{Kind: e.Kind, Value: e.Source}, finished: e.FinishedAt, entry: true}
		md := a.markdown(*e)
		t.missing, t.loading = md.missing, !md.read
		t.state = model.Done{Folder: e.Folder, Result: model.Result{Markdown: md.text, MarkdownPath: e.MarkdownPath,
			Duration: e.Duration, Warnings: e.Warnings, Usage: e.Usage, Title: e.Title}}
		return t
	}
	return nil
}

func (t *target) name() string {
	if t == nil {
		return ""
	}
	return t.title
}

// done returns the target's Done state when its Markdown can be used.
func (t *target) done() (model.Done, bool) {
	if t == nil || t.missing || t.loading {
		return model.Done{}, false
	}
	d, ok := t.state.(model.Done)
	return d, ok
}

// markdownFile is a History entry's Markdown, read off the UI thread.
type markdownFile struct {
	text          string
	read, missing bool
}

func (a *app) markdown(e model.Entry) *markdownFile {
	if md := a.markdowns[e.ID]; md != nil {
		return md
	}
	md := &markdownFile{}
	a.markdowns[e.ID] = md
	id, path := e.ID, e.MarkdownPath
	go func() {
		data, err := os.ReadFile(path)
		a.post(func() {
			if a.markdowns[id] != md {
				return
			}
			md.read, md.missing, md.text = true, err != nil, withoutFrontMatter(string(data))
		})
	}()
	return md
}

// withoutFrontMatter is a written Markdown file's text after its YAML
// front matter, as the core leaves it out of a result's markdown (DESIGN
// 3.3), so that History shows, copies and saves what Convert did.
func withoutFrontMatter(text string) string {
	rest, ok := strings.CutPrefix(text, "---\n")
	if !ok {
		return text
	}
	for offset := 0; offset < len(rest); {
		line := rest[offset:]
		if end := strings.IndexByte(line, '\n'); end >= 0 {
			line = line[:end+1]
		}
		if line == "---\n" || line == "---" {
			return strings.TrimLeft(rest[offset+len(line):], "\n")
		}
		offset += len(line)
	}
	return text
}

// matches applies History's search: a case-insensitive substring of the
// title or the source.
func (a *app) matches(e model.Entry) bool {
	q := strings.ToLower(strings.TrimSpace(a.query))
	return q == "" || strings.Contains(strings.ToLower(e.Title), q) || strings.Contains(strings.ToLower(e.Source), q)
}

func (a *app) filtered() []model.Entry {
	var out []model.Entry
	for _, e := range a.history.Entries {
		if a.matches(e) {
			out = append(out, e)
		}
	}
	return out
}

func (a *app) copyMarkdown(c *ui.Context) {
	d, ok := a.target().done()
	if !ok {
		return
	}
	c.WriteClipboard(d.Result.Markdown)
	a.copied, a.copiedAt = true, time.Time{}
	a.announce = append(a.announce, a.lang.T("notice.copied"))
}

func (a *app) showInFolder() {
	if d, ok := a.target().done(); ok {
		a.desk.showItemInFolder(d.Result.MarkdownPath)
	}
}

// saveAs writes the shown Markdown where the user says, with the
// conversion's .markitai folder next to it when the text points into it
// (DESIGN 7.2).
func (a *app) saveAs() {
	d, ok := a.target().done()
	if !ok {
		return
	}
	a.desk.saveAs(filepath.Base(d.Folder)+".md", func(path string) {
		go func() {
			err := save(d, path)
			a.post(func() {
				if err != nil {
					fmt.Fprintf(os.Stderr, "save as: %v\n", err)
					a.setNotice(a.lang.T("error.save_failed"))
				}
			})
		}()
	})
}

func save(d model.Done, path string) error {
	if err := os.WriteFile(path, []byte(d.Result.Markdown), 0o644); err != nil {
		return err
	}
	if !strings.Contains(d.Result.Markdown, ".markitai/") {
		return nil
	}
	return copyMissing(filepath.Join(d.Folder, ".markitai"), filepath.Join(filepath.Dir(path), ".markitai"))
}

// copyMissing copies the files of src into dst that dst lacks, never
// replacing one.
func copyMissing(src, dst string) error {
	return filepath.WalkDir(src, func(p string, e os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(src, p)
		to := filepath.Join(dst, rel)
		if e.IsDir() {
			return os.MkdirAll(to, 0o755)
		}
		data, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		f, err := os.OpenFile(to, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
		if errors.Is(err, os.ErrExist) {
			return nil
		}
		if err != nil {
			return err
		}
		if _, err := f.Write(data); err != nil {
			f.Close()
			return err
		}
		return f.Close()
	})
}

func (a *app) toggleView() {
	if _, ok := a.target().done(); ok {
		a.detail = 1 - a.detail
	}
}

func (a *app) zoom(z float64) {
	z = model.ClampZoom(z)
	if z == a.settings.Zoom {
		return
	}
	a.settings.Zoom = z
	a.saveSettings()
}

func (a *app) saveSettings() {
	if err := a.settings.Save(a.files.settings); err != nil {
		fmt.Fprintf(os.Stderr, "settings: %v\n", err)
	}
}

// setOutput takes a folder for the output when the app can write there
// (DESIGN 6.6). Taking one clears the notice about a folder refused before.
func (a *app) setOutput(dir string) {
	unwritable := a.lang.T("settings.output.unwritable")
	if !writable(dir) {
		a.setNotice(unwritable)
		return
	}
	if a.notice.text == unwritable {
		a.notice = notice{}
	}
	a.settings.OutputFolder = &dir
	a.saveSettings()
}

func writable(dir string) bool {
	f, err := os.CreateTemp(dir, ".markitai-write-test-*")
	if err != nil {
		return false
	}
	name := f.Name()
	f.Close()
	return os.Remove(name) == nil
}

func (a *app) showOutput() {
	dir := a.output()
	os.MkdirAll(dir, 0o755)
	a.desk.showItemInFolder(dir)
}

// openConfig opens the markitai configuration file, creating it first
// when it is not there (DESIGN 6.6).
func (a *app) openConfig() {
	defer func() { a.config.read = false }()
	cfg, err := a.core.ConfigFile()
	if err == nil && !cfg.Exists {
		if err = os.MkdirAll(filepath.Dir(cfg.Path), 0o755); err == nil {
			err = os.WriteFile(cfg.Path, []byte("{}\n"), 0o644)
		}
	}
	if err == nil {
		err = a.desk.openPath(cfg.Path)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "configuration: %v\n", err)
		a.setNotice(a.lang.T("error.open_failed", "name", filepath.Base(cfg.Path)))
	}
}

func (a *app) chooseFiles() {
	a.desk.chooseFiles(func(paths []string) {
		if len(paths) > 0 {
			a.addPaths(paths)
		}
	})
}

func (a *app) chooseOutput() {
	a.desk.chooseFolder(func(dir string) {
		if dir != "" {
			a.setOutput(dir)
		}
	})
}
