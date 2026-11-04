package main

import (
	"errors"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"github.com/egoist/mygo"
	_ "github.com/egoist/mygo/locales/zh-Hans"
	"github.com/egoist/mygo/ui"

	"markitai.local/mygo/internal/core"
	"markitai.local/mygo/internal/l10n"
)

// noNetwork blocks every web and socket request from the preview (8.1).
const noNetwork = `[{"trigger":{"url-filter":"^https?://"},"action":{"type":"block"}},` +
	`{"trigger":{"url-filter":"^wss?://"},"action":{"type":"block"}}]`

// minWidth and minHeight are the content's least size (section 6.10).
const minWidth, minHeight = 640, 480

// main launches the app in section 2.3's order.
func main() {
	home, err := os.UserHomeDir()
	if err != nil {
		log.Fatal(err)
	}
	// Relative file arguments name paths from where the app was launched,
	// before prepare_environment moves the working directory.
	wd, _ := os.Getwd()

	// The language comes first: the system's own text takes it before
	// anything asks the system for a language (10.5).
	preferred := mygo.App.PreferredLanguages()
	if tag := os.Getenv("MYGO_LOCALE"); tag != "" {
		preferred = append([]string{tag}, preferred...)
	}
	lang := l10n.Choose(preferred)
	localizeSystemText(lang, preferred)

	data, cache, settings := mygoFolders(runtime.GOOS, home, os.Getenv)
	mygo.App.SetPath(mygo.PathUserData, data)
	mygo.App.SetPath(mygo.PathCache, cache)
	prepareToolkit()
	if !singleInstance(data, wd) {
		return
	}
	mygo.App.SetLanguages(string(l10n.En), string(l10n.ZhHans))
	mygo.App.SetLocale(string(lang))
	l10n.ChangeMygoText()
	log.Printf("language %s, from %v", lang, preferred)

	resources, _ := mygo.App.Path(mygo.PathResources)
	lib, err := core.Load(core.Find(resources))
	fatal := errors.Is(err, core.ErrABI)
	if err != nil && !fatal {
		log.Fatal(err)
	}

	var win *mygo.Window
	post := func(fn func()) { win.Update(fn) }
	d := systemDesk{win: func() *mygo.Window { return win }, post: post, chooseTitle: lang.T("convert.choose")}
	var a *app
	if fatal {
		log.Print(err)
		a = &app{lang: lang, desk: d, goos: runtime.GOOS, fatal: true}
	} else {
		if err := os.MkdirAll(data, 0o755); err != nil {
			log.Fatal(err)
		}
		env, err := lib.PrepareEnvironment(data)
		if err != nil {
			log.Printf("prepare_environment: %v", err)
		} else {
			log.Printf("prepare_environment: %s", env.LoginEnvironment)
		}
		a = newApp(lang, lib, d, nil, post, files{
			data: data, settings: settings, history: filepath.Join(data, "history.json"),
			output: filepath.Join(documentsFolder(), "Markitai"),
		})
		a.version = mygo.App.Version()
		mygo.BindAs("Preview", &previewService{goos: runtime.GOOS, press: a.pressed, drop: a.droppedOnPreview, wheel: a.wheeled})
	}

	opened := fileArgs(os.Args[1:], wd)
	if runtime.GOOS == "darwin" {
		mygo.App.OnOpenFile(func(path string) {
			switch {
			case a.fatal:
				// The fatal window converts nothing (section 9).
			case win == nil:
				opened = append(opened, path)
			default:
				win.Update(func() { a.addPaths([]string{path}) })
			}
		})
	}
	mygo.App.OnSecondInstance(func(args []string, wd string) {
		paths := fileArgs(args, wd)
		if win == nil {
			// Before the window: its files wait with the first launch's.
			if !a.fatal {
				opened = append(opened, paths...)
			}
			return
		}
		win.Update(func() {
			if len(paths) > 0 && !a.fatal {
				a.addPaths(paths)
			}
			d.showWindow()
		})
	})
	mygo.App.OnBeforeQuit(func(e *mygo.QuitEvent) {
		if !a.fatal && !a.mayQuit() {
			e.PreventDefault()
			win.Invalidate()
		}
	})
	mygo.App.OnQuit(func() {
		if !a.fatal {
			a.removeConvertingFolders()
		}
	})

	mygo.App.WhenReady(func() {
		// After Run gave the backend the app's languages, which the
		// system's dialogs on Windows do not all resolve (10.5).
		localizeSystemDialogs()
		var back func()
		if !fatal {
			back = func() { a.pressed(actBack) }
		}
		toolkitReady(back)
		// The window shows once the preview's web view is in it: on Linux,
		// mygo makes room for the first web view by moving the content's
		// widget, and a shown window that is not active yet loses its
		// keyboard focus that way (DESIGN 14).
		width, height := firstSize(mygo.Screen.PrimaryDisplay().WorkArea)
		win = mygo.NewWindow(mygo.WindowOptions{
			Title: lang.T("app.name.mygo"), Width: width, Height: height, MinWidth: minWidth, MinHeight: minHeight,
			StateKey: "main", Content: ui.View(a.view), Hidden: true,
		})
		contentMinimum(win)
		defer win.Show()
		log.Printf("window created at %d ms since the epoch", time.Now().UnixMilli())
		win.OnReadyToShow(func() { log.Printf("first frame at %d ms since the epoch", time.Now().UnixMilli()) })
		a.closing(win)
		// Settings reads the configuration file again when the window
		// comes back, as the user may have made it meanwhile (6.6).
		win.OnFocus(func() {
			a.config.read = false
			win.Invalidate()
		})
		if fatal {
			if runtime.GOOS == "darwin" {
				mygo.App.SetMenu(mygo.NewMenu([]*mygo.MenuItem{{Role: mygo.RoleAppMenu, Submenu: []*mygo.MenuItem{{Role: mygo.RoleQuit}}}}))
			}
			return
		}
		// On the main thread mygo keeps handling events while WebKit
		// compiles; a goroutine's call would wait for this one, which would
		// wait for it.
		rules, err := mygo.CompileContentRules("preview-no-network", noNetwork)
		if err != nil {
			log.Printf("content rules: %v", err)
		}
		a.attachPreview(win, rules)
		if runtime.GOOS == "darwin" {
			mygo.App.SetMenu(a.newMenuBar(a.pressed))
		}
		if len(opened) > 0 {
			paths := opened
			win.Update(func() { a.addPaths(paths) })
		}
	})
	mygo.App.SetWindowStateStore(windowStates{path: filepath.Join(data, "window-state.json")})
	if err := mygo.App.Run(); err != nil {
		log.Fatal(err)
	}
}

// closing follows section 10.3: on macOS closing the window hides it and
// the Dock icon shows it again; elsewhere closing it quits. The fatal
// window has nothing to come back to: closing it quits everywhere, as mygo
// does when the last window closes and nothing keeps the app.
func (a *app) closing(win *mygo.Window) {
	if runtime.GOOS != "darwin" {
		win.OnClose(func(e *mygo.CloseEvent) {
			if !a.fatal && !a.mayQuit() {
				e.PreventDefault()
				win.Invalidate()
			}
		})
		return
	}
	if a.fatal {
		return
	}
	win.OnClose(func(e *mygo.CloseEvent) {
		if !a.quitting {
			e.PreventDefault()
			win.Hide()
		}
	})
	mygo.App.OnWindowAllClosed(func() {})
	mygo.App.OnActivate(func(visible bool) {
		if !visible {
			win.Show()
			win.Focus()
		}
	})
}

// e2ePreview, which only e2e builds set (e2e.go), gets the preview's page
// once the window has it.
var e2ePreview func(page *mygo.Page)

// attachPreview makes the window's web view (8.1). Without one, the
// Preview tab says so.
func (a *app) attachPreview(win *mygo.Window, rules *mygo.ContentRules) {
	page := mygo.PageOptions{DevTools: mygo.DevToolsDisabled, Session: mygo.NewEphemeralSession(), PreloadScript: previewJS(runtime.GOOS)}
	if rules != nil {
		page.ContentRules = []*mygo.ContentRules{rules}
	}
	view, err := win.NewWebView(mygo.WebViewOptions{Page: page})
	if err != nil {
		log.Printf("web view: %v", err)
		return
	}
	pane := &webPreview{view: view}
	if e2ePreview != nil {
		e2ePreview(view.Page())
	}
	view.Page().OnWillNavigate(func(e *mygo.NavigateEvent) {
		allow, external := navigation(pane.base, e.URL)
		if !allow {
			e.PreventDefault()
		}
		if external {
			if err := a.desk.openExternal(e.URL); err != nil {
				log.Printf("open %s: %v", e.URL, err)
			}
		}
	})
	copyLabel := a.lang.T("common.copy")
	view.Page().OnContextMenu(func(e *mygo.ContextMenuEvent) {
		if e.SelectionText == "" || e.Menu == nil || e.Menu.ItemByID(mygo.ContextMenuCopy) == nil {
			e.PreventDefault()
			return
		}
		for _, it := range e.Menu.Items() {
			if it.ID == mygo.ContextMenuCopy {
				it.SetLabel(copyLabel)
			} else {
				it.SetVisible(false)
			}
		}
	})
	a.pane = pane
	a.previews.pane = pane
}
