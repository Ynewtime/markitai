# Markitai Mygo: design specification

Markitai Mygo is the third desktop version of Markitai. Markitai Web
(`desktop/web/`) wraps `markitai serve` in a window. Markitai Native
(`desktop/native/macos`, `desktop/native/windows`, `desktop/native/linux`)
draws each platform's own controls. Markitai Mygo is one Go codebase that
draws its own interface with mygo's GPU-rendered `ui` package and runs on
macOS, Windows and Linux with the same look. It calls the Rust core in
process.

This file is the single source for the implementer and the reviewers. It
builds on [the shared product specification](../DESIGN.md),
`desktop/DESIGN.md`, cited below as "DESIGN" with its section numbers.
Section 3 here lists which DESIGN sections apply unchanged. This file
replaces every per-platform control column of DESIGN.

mygo facts carry a citation relative to the mygo v0.7.4 module root, such as
`ui/feedback.go:176`. A fact marked **(unverified)** was not confirmed in
mygo's source and has a check in section 17.

## 1. Scope

Scope is DESIGN 1, unchanged. Nothing in this file adds a feature.

## 2. Technology and identity

| Item | Value |
|---|---|
| Display name | `Markitai Mygo`: bundle and executable name, window title, app menu, Dock, taskbar, launcher entry, About title. Sentences in the copy keep saying "Markitai" |
| Identifier | `dev.markitai.mygo` |
| macOS | `Markitai Mygo.app`, minimum macOS 13 |
| Windows | `Markitai Mygo.exe`, x86-64, WebView2 Runtime for the preview |
| Linux | binary and Debian package `markitai-mygo`, x86-64. GTK 3 and WebKitGTK 4.1 (`libgtk-3-0`, `libwebkit2gtk-4.1-0`, `cmd/mygo/deb.go:74-81`). Before `App.Run`, `main` names the app `dev.markitai.mygo` with `App.SetName` and GLib's `g_set_prgname` (`toolkit_linux.go`), as Markitai Native names itself `dev.markitai.native`: that is the window's class (Wayland app ID), the single-instance key and the name of every folder the app and mygo keep (section 2.1). The packages install the desktop file as `dev.markitai.mygo.desktop`, with `StartupWMClass=dev.markitai.mygo`, so launchers and docks match it to the window (section 15) |
| Source | `desktop/mygo/`, its own Go module `markitai.local/mygo`. Go version and mygo version equal `desktop/web/go.mod`'s (Go 1.27.3, `github.com/egoist/mygo v0.7.4`), with `tool github.com/egoist/mygo/cmd/mygo`. Code names inside it say "mygo" for this version, never "desktop" or "native" |
| UI | mygo `ui`, drawn by mygo on every platform. The only system views are the preview web view (section 8) and the system file pickers |
| Core | `crates/markitai-app` built as a cdylib, loaded with `github.com/ebitengine/purego`, `CGO_ENABLED=0`. The C ABI is DESIGN 3.1 and 3.3, unchanged. The library sits in `resources/<goos>-<goarch>/` (`libmarkitai_app.dylib`, `markitai_app.dll`, `libmarkitai_app.so`), which `mygo build` merges into the bundle's resources (`cmd/mygo/resources.go:104-112`) |
| Markitai configuration | Shared with the other versions: `~/.markitai`, or `MARKITAI_HOME` |

`desktop/mygo/mygo.json`:

```json
{
  "name": "Markitai Mygo",
  "slug": "markitai-mygo",
  "identifier": "dev.markitai.mygo",
  "version": "<workspace version>",
  "copyright": "Copyright (c) 2026 Ynewtime",
  "icon": "../icon.png",
  "out": "build",
  "macos": {
    "minimumSystemVersion": "13.0",
    "infoPlist": {
      "CFBundleDevelopmentRegion": "en",
      "CFBundleLocalizations": ["en", "zh-Hans"],
      "NSDocumentsFolderUsageDescription": "<plist.documents, en>",
      "NSDesktopFolderUsageDescription": "<plist.desktop, en>",
      "NSDownloadsFolderUsageDescription": "<plist.downloads, en>",
      "CFBundleDocumentTypes": [{"CFBundleTypeRole": "Viewer", "LSHandlerRank": "None", "LSItemContentTypes": ["public.data"]}]
    }
  },
  "linux": {"comment": "Convert documents and web pages to Markdown", "categories": ["Office", "Utility"], "command": "markitai-mygo",
            "maintainer": "Ynewtime <27655771+Ynewtime@users.noreply.github.com>"}
}
```

`icon` names `desktop/icon.png` itself, from which `mygo build` makes the
bundle's `AppIcon.icns`, the `.exe`'s icon and the Linux icons. The About
panel and Settings' About draw `internal/brand/icon.png`, a copy inside the
module, because `go:embed` cannot reach a parent folder; `build.py` refreshes
the copy, and its `--check` and `brand_test.go` fail when it differs. `linux.webview` stays unset,
which means true: a `ui.WebView` needs WebKitGTK (`cmd/mygo/deb.go:37-41`).
`macos.infoPlist` overrides the keys mygo writes (`cmd/mygo/bundle.go:307-309`).
`build.py` fails when `version` differs from the workspace version.
`linux.comment` is Native's desktop-file `Comment`, which `linux_entry.py`
checks, and `linux.maintainer` the repository's public commit identity, which
the Debian package and the PKGBUILD name.

### 2.1 Coexistence with the other versions

The three versions run side by side on one machine. Markitai Web is
`dev.markitai.web`, Markitai Native is `dev.markitai.native`.

| Concern | Markitai Mygo |
|---|---|
| Data folder (`data_dir`, working directory) | macOS `~/Library/Application Support/dev.markitai.mygo`. Windows `%LOCALAPPDATA%\Markitai Mygo`. Linux `$XDG_DATA_HOME/dev.markitai.mygo` (`~/.local/share/...`) |
| History (DESIGN 4.4 shape) | `<data folder>/history.json` |
| Settings (DESIGN 4.5 keys except `window.*`) | macOS and Windows `<data folder>/settings.json`. Linux `$XDG_CONFIG_HOME/dev.markitai.mygo/settings.json`. Writes follow DESIGN 4.4's temporary-file-and-rename rule |
| Window frame | mygo's `WindowOptions.StateKey` `"main"`, kept by the app's `App.SetWindowStateStore` in `<data folder>/window-state.json` (`windowstate.go`), written through a temporary file and a rename. As it reads the frame back, the store fits it to the work area it overlaps most: the size at most the work area's, the frame inside it (section 6.1). mygo itself restores any frame of which a corner shows (`window_state.go:113-151`) |
| mygo's own folders | Before `App.Run`, `App.SetPath(PathUserData, <data folder>)` and `App.SetPath(PathCache, <cache folder>)` (`paths.go:72-80`). Without this, mygo names them after the display name (`paths.go:90-95`). The cache folder is macOS `~/Library/Caches/dev.markitai.mygo`, Windows `%LOCALAPPDATA%\Markitai Mygo\Cache`, Linux `$XDG_CACHE_HOME/dev.markitai.mygo`. On Linux, mygo's web data does not follow these paths (row "Web data (Linux)") |
| WebView2 data (Windows) | `%LOCALAPPDATA%\Markitai Mygo\WebView2`, which mygo derives from the display name (`internal/windows/app.go:185-189`). The GPU driver keeps its own shader cache, Intel's in `%USERPROFILE%\AppData\LocalLow\Intel\ShaderCache`, shared by every app that draws through it |
| Single instance | `App.RequestSingleInstanceLock()`. A packaged app's key is `identifier + "\n" + name` (`single_instance.go:164-172`), so the three versions never share a lock. On Windows its socket is in the data folder, or under the temporary folder when the data folder takes no file (section 10.6) |
| macOS defaults domain | `dev.markitai.mygo`, used only for `AppleLanguages` (section 10.5) |
| WebKit's content rules (macOS) | `~/Library/WebKit/dev.markitai.mygo/ContentRuleLists/`, the compiled `preview-no-network` list (256 bytes). mygo compiles into `WKContentRuleListStore.defaultStore` (`internal/darwin/content_rules.go:61`), which WebKit keeps by identifier in the account's real home, not under `HOME`, so test runs with an isolated `HOME` still write it there. The preview's session is ephemeral (`NewEphemeralSession`, `internal/darwin/session.go:44`), so no other web data is kept |
| Web data (Linux) | mygo keeps the compiled content rules in `$XDG_CONFIG_HOME/<App.Name()>/content-rules/` and the default web context's cookies in `$XDG_CONFIG_HOME/<App.Name()>/cookies.sqlite`, not under `PathUserData` (`internal/linux/content_rules.go:84-89`, `internal/linux/cookies.go:92-103`), and that context's WebKitGTK data manager keeps its data in `$XDG_DATA_HOME/<program name>/` and `$XDG_CACHE_HOME/<program name>/`. With the name and the program name both `dev.markitai.mygo` (section 2), all of it is in `$XDG_CONFIG_HOME/dev.markitai.mygo/`, `$XDG_DATA_HOME/dev.markitai.mygo/` and `$XDG_CACHE_HOME/dev.markitai.mygo/`, beside the app's own files (found in the Linux runs). The preview's own session is ephemeral (`internal/linux/session.go:59`) |
| Shared files (Linux) | Files that every app drawing through the same libraries shares, under no app's name. The GPU driver's shader cache, Mesa's in `$XDG_CACHE_HOME/mesa_shader_cache/`. Once a file dialog has shown, through the portal or not, as `GtkFileChooserNative` makes GTK's own chooser either way: `$XDG_DATA_HOME/gvfs-metadata/`, which `gvfsd-metadata` writes about a minute after the dialog opens, and `$XDG_DATA_HOME/recently-used.xbel`, which GTK writes as the app quits (found in the Linux review: an empty tree and an empty list). The e2e run lists them and fails only on another version's names, or on Linux on `Markitai Mygo` or `markitai-mygo` (section 16) |
| Output folders | The shared default output folder (DESIGN 4.3) is safe to share: `claim_output_folder` creates each conversion's folder with an atomic `create_dir` (DESIGN 3.3), so no two apps get one folder |

### 2.2 Code structure and data shapes

The spike's packages stay and grow:

| Package | Holds |
|---|---|
| `internal/core` | The purego binding of DESIGN 3.1 (spike `core.go`) |
| `internal/l10n` | `strings.json` lookup, plurals, formats of DESIGN 4.2 |
| `internal/model` | `Item`, `State` (`Queued`, `Converting`, `Done`, `Failed`), `Queue` (DESIGN 4.1, spike `queue.go`), and new `History` (DESIGN 4.4: add, remove, clear, 500 limit, atomic write, unreadable rename) and `Settings` (DESIGN 4.5) with their files |
| `internal/brand` | The embedded app icon (section 2) |
| `internal/testenv` | The isolated environment of section 15 for tests, the repository's fixtures and the core library to test against |
| `internal/e2e` | The real-window driver, behind the `e2e` build tag (section 16) |
| `main` | `main.go` (launch, section 2.3), `app.go` (state and actions), one file per screen, `preview.go` (section 8), `menu.go` (section 9.1), `desk.go` (dialogs, shell and the web view behind small interfaces that tests fake), `actions.go` (actions, keys and the preview's script, section 8.3), `env.go` (the user's environment for programs the app opens, section 10.5), files for one OS's needs (`about_*`, `documents_*`, `instance_*`, `lang_*`, `shell_*`, `toolkit_*`, `window_*`, `gio_linux.go`), `e2e.go` (the `e2e` build's automation plugin and theme override) |

The window's state is one `app` value read and changed on the UI thread. Work
done elsewhere (a conversion, a preview render, a dialog's answer) returns
through `Window.Update`. Every user action is one method on `app`, such as
`a.copyMarkdown()`, so a shortcut, a menu item, a context-menu item, a button
and a key forwarded from the preview all call the same method.

```text
app {
  lang, core, desk                    injected: tests pass fakes
  settings: model.Settings            DESIGN 4.5
  queue: model.Queue                  DESIGN 4.1
  history: model.History              DESIGN 4.4
  router: *ui.Router                  pages, section 4
  view: { listWidth, detailView (preview|markdown), zoom, warningsOpen[id],
          query, notice, copiedAt, asking (none|quit|clearHistory) }
  preview: previewPane                section 8
}
```

### 2.3 Launch

1. Keep the working directory: relative file arguments name paths from it,
   and later steps move it.
2. Choose the language, before anything asks the system for one, and give it
   to the system's own text: on macOS in the argument domain, on Linux in
   `LANGUAGE` (section 10.5).
3. Compute the folders of section 2.1 and call `App.SetPath` for them. On
   Linux (`prepareToolkit`, `toolkit_linux.go`): name the app and the program
   `dev.markitai.mygo` (section 2); set `BAMF_DESKTOP_FILE_HINT` to the
   installed `dev.markitai.mygo.desktop` when the desktop did not launch the
   app from an entry (section 14); ask the portal for its `FileChooser`
   version, waiting up to 3 s, and set `GTK_USE_PORTAL=1` only when it
   answers (section 10.5); follow the system's dark style (section 10.4).
   Each variable the app sets for its toolkit, and
   `WEBKIT_DISABLE_DMABUF_RENDERER`, which mygo sets for WebKitGTK in
   `App.Run` (`internal/linux/app.go:112-116`), first has the user's value
   recorded for the programs the app opens (section 10.5, `env.go`).
4. `App.RequestSingleInstanceLock()`. A second launch hands its arguments to the
   first and exits (`single_instance.go:50-73`). On Windows the lock is taken
   in the data folder when it takes a file (section 10.6).
5. `App.SetLanguages("en", "zh-Hans")` and `App.SetLocale` (section 10.5),
   then `l10n.ChangeMygoText()`, which registers the app's changes to
   mygo's own strings (section 14).
6. `core.Load(core.Find(resources))`. It checks the ABI version before it
   looks up the other functions, so a library without one of them is refused
   the same way: an ABI mismatch shows only the fatal screen (section 6.9).
7. `PrepareEnvironment(data folder)`, before `App.Run`. It blocks up to 10 s
   (DESIGN 3.3).
8. Load settings and history.
9. `mygo.BindAs("Preview", &previewService{...})` (section 8.3, `ipc.go:105-110`),
   and `App.SetWindowStateStore` (section 2.1).
10. `App.WhenReady`: on Windows give the system dialogs their languages
   (section 10.5); on Linux hook GTK's signals (`toolkitReady`: the file
   chooser's size, section 14, and the mouse's back button over the
   preview, section 8.3); make the window hidden (`WindowOptions.Hidden`) at
   the size of section 6.1, with the content's minimum of section 6.10,
   compile the content rules (section 8.1), make the preview's web view, and
   on macOS the menu bar. Then show the window. Add the files
   from `os.Args[1:]` (section 10.1). The window shows only after its web view is made, because
   of a Linux focus defect (section 14). The first launch with a new data
   folder compiles the content rules before that (measured on the Linux
   machine: the window showed about 1.25 s after launch, and about 0.3 s once
   the compiled rules were kept). On the Windows machine, at 192 dpi, the
   window showed about 0.6 s after launch with a new data folder and about
   0.12 s with one it had used.

## 3. Relation to desktop/DESIGN.md

| DESIGN section | In Markitai Mygo |
|---|---|
| 1 Scope | Applies |
| 2 Technology | Replaced by section 2 |
| 3 The `markitai-app` crate | Applies. Built as a cdylib on all three platforms |
| 4.1 to 4.2 Items, result fields | Apply |
| 4.3 Output folder | Applies. The default is mygo's `PathDocuments` + `Markitai` (`paths.go:142-171` reads `user-dirs.dirs` on Linux) |
| 4.4 History | Shape, limits and operations apply. The file is in section 2.1 |
| 4.5 Settings | Keys apply except `window.*` (section 2.1) |
| 4.6 App folders | Replaced by section 2.1 |
| 5 Information architecture | Replaced by section 4 |
| 6 Screens | Content, states and copy keys apply. Controls, sizes and placement are section 6 here |
| 7 Behavior | Applies. Platform rows are replaced by section 10 |
| 8 Preview web view | Rules apply. The mygo realization is section 8 |
| 9 Errors | Applies. `{settings}` is `nav.settings` |
| 10 Copy | Applies, plus the keys in section 13 |
| 11 Keyboard shortcuts | Actions apply. Keys per OS are section 9.2 |
| 12 Accessibility | Applies. The mygo mapping is section 11 |
| 13 Visual tokens | 13.1 ramp, 13.3 meanings, 13.4 scale and 13.6 meanings apply. The single look is section 5 |
| 14 Window sizes | Replaced by section 6.10 |
| 15 Preview stylesheet | Applies unchanged |
| 16 Checks | Replaced by section 17 |

## 4. Information architecture

| Place | How |
|---|---|
| Convert | Sidebar item, first. Router path `/convert`, `/convert/item` in the stacked layout |
| History | Sidebar item, second. `/history`, `/history/item` |
| Settings | Sidebar footer item `nav.settings`, a page of the main window, `/settings`. Also ⌘, or Ctrl+, and on macOS the app menu's `menu.settings` |
| About | The last group of Settings. On macOS also the app menu's `menu.about.mygo`, which shows AppKit's standard About panel (`showAbout`, `about_darwin.go`; section 14) |
| Start page | `/convert`, always |

One `ui.Router` (`ui/router.go:62`) holds the pages. Choosing a sidebar item
calls `router.Replace` with its path, so sidebar changes do not pile up in the
history. Only the stacked layout pushes (section 6.10).

## 5. Look

One look on every platform. mygo draws all of it, so the values below are
exact on every OS.

### 5.1 Theme

The app builds its theme each frame from `*c.Theme()`, which already follows
the system's appearance, accent, high contrast and text size
(`ui/preferences.go:50-79`). It overrides the neutrals with DESIGN 13.1's ramp,
then calls `c.SetTheme`.

| `ui.Theme` field | Light | Dark | Use |
|---|---|---|---|
| `Background` | `#ffffff` | 900 `#18181b` | Detail pane, dialogs, fields |
| `Surface` | 100 `#f4f4f5` | 800 `#27272a` | Buttons, segmented track, boxes (warnings, raw error, Settings groups) |
| `SurfaceHover` | 200 `#e4e4e7` | 700 `#3f3f46` | Hovered controls |
| `SurfacePressed` | 300 `#d4d4d8` | 600 `#52525b` | Pressed controls |
| `Border` | 300 `#d4d4d8` | 700 `#3f3f46` | Control outlines, drop frames |
| `Text` | 900 `#18181b` | 50 `#fafafa` | Primary text |
| `TextMuted` | 600 `#52525b` | 400 `#a1a1aa` | Secondary text only (DESIGN 12) |
| `Accent`, `AccentHover`, `AccentPressed`, `AccentText` | the ink of section 5.2 | the ink of section 5.2 | Primary button, switches, marks, chosen-segment pill, drag highlight, the chosen row's leading bar |
| `Focus` | the accent at 55%: 3.6:1 or more on every light fill | 400 `#a1a1aa`: 4.0:1 or more on every dark fill, where the dark accent was about 2:1 | Focus ring, at least 3:1 against what it sits on (WCAG 1.4.11) |
| `Danger`, `Warning`, `Success` | mygo's (`#dc2626`, `#d97706`, `#16a34a`) | mygo's (`#ef4444`, `#f59e0b`, `#22c55e`) | Status glyphs only (DESIGN 13.3) |
| `Radius` | 6 | 6 | Buttons, fields, rows |

Region fills, which are not theme fields:

| Token | Light | Dark | Use |
|---|---|---|---|
| `sidebarFill` | 100 `#f4f4f5` | 950 `#09090b` | Sidebar |
| `listFill` | 50 `#fafafa` | 950 `#09090b` | List column |
| `rule` | 200 `#e4e4e7` | 800 `#27272a` | Hairlines between regions and rows of a group |
| `rowHover` | 100 `#f4f4f5` | 900 `#18181b` | Hovered row of a list or sidebar |
| `chosenQuiet` | 200 `#e4e4e7` | 800 `#27272a` | Chosen row of a list, chosen sidebar item |

Chosen rows and items follow the workbench (`.row.is-selected` in
`crates/markitai-cli/src/server/web/src/styles/app.css`): a quiet fill,
`chosenQuiet`, with a 2 px bar in `Accent` at the leading edge, and every part
in its normal color (title `Text`, subtitle and glyph `TextMuted`), focused or
not. The fill is never the accent. `ui.List` paints chosen rows with `Accent`
and `AccentText` (`ui/list.go:703-708`, `ui/table.go:629-640`) and `ui.Sidebar`
paints its chosen item with `Accent` while focused and `SurfacePressed`
otherwise (`ui/sidebar.go:184-200`). So the sidebar and both lists build inside
`ui.Themed` with a copy of the theme whose `Accent`, `AccentHover`,
`AccentPressed` and `SurfacePressed` are `chosenQuiet`, `AccentText` is `Text`,
`SurfaceHover` is `rowHover`, and `Focus` keeps the theme's, with
`Background(ui.Transparent)` (`ui/context.go:351-360`); the row or item draws
its leading bar itself in the real accent. Sidebar glyphs are `TextMuted`, and
`Text` on the chosen item; never the accent. mygo paints hover with
`SurfaceHover` (`ui/list.go:725-729`). `ui.Sidebar` paints `Surface` under
itself, so each also takes `Background(sidebarFill)` (`ui/sidebar.go:42`).
Secondary text passes 4.5:1 on every fill it sits on: 600 on 200 is 6.09:1,
400 on 800 is 5.81:1. The Convert list draws its rows the same way.

With `Preferences().HighContrast`, after the overrides, the app applies mygo's
own rule again: `Border` and `rule` mix 45% toward `Text`, `TextMuted` 40%, and
`Focus` is opaque (`ui/preferences.go:68-73`).

### 5.2 Accent

The accent is the brand ink of DESIGN 13.2 on every OS, whatever accent the
desktop reports, so the three versions look alike: light 900 `#18181b` (hover
800, pressed 700), dark 500 `#71717a` (hover 600, pressed 700), with white
`AccentText` (4.83:1 or better on every state). The system accent
(`Preferences().Accent`) is not used; on macOS its usual pair, white on
`#007aff`, is 4.0:1, below 4.5:1 for small text. The accent fills the primary
button, switches, radio and check marks, the segmented control's chosen-segment
pill, the drag highlight, and the leading bar of a chosen row or item, and in
the light theme the focus ring (section 5.1). Text is never drawn in the
accent.

### 5.3 Type

`B` is the theme's `FontSize`: 13 on macOS and 14 elsewhere
(`ui/theme.go:97-102`), times the desktop's text scale. The font is the
system's (`Theme.Font` empty).

| Role | Size and weight |
|---|---|
| Empty-state title | 22, 600 |
| Page title (column header, Settings) | 17, 600 |
| Detail title | 17, 600 |
| Settings group heading | `B`, 600 |
| Body, row title, buttons | `B` |
| Row subtitle, notes, footer, raw error caption | `B - 2` |
| Monospace (raw error, Markdown text) | `B - 1`, `Font("monospace")` (`docs/ui/text.md:173`). The Markdown text also scales with `view.zoom` |

Sizes other than `B` scale with the text scale too: multiply by
`TextScale` from `Preferences()`.

### 5.4 Spacing and sizes

Spacing comes from DESIGN 13.4's scale (4, 8, 12, 16, 20, 24, 32, 48).

| Use | Value |
|---|---|
| Sidebar width | 200. Each `ui.Sidebar` keeps mygo's padding (`ui/sidebar.go:42`) |
| List column | 320 default, 260 to 420 (section 14), padding 12 |
| Detail content inset | 24 |
| Between related controls | 8 |
| Between groups | 24 |
| Column header and detail bar | 52 high, hairline `rule` below |
| List row | 44 high, radius 6, padding 0 10, gap 8 |
| Drop zone | 560 wide (at most the room), padding 40, gap 12, radius 12, 1.5 px dashed `Border` |
| Drop bar | 48 high, radius 12, same frame |
| Text field, buttons | 32 high, `Radius` 6 |
| Boxes (warnings, raw error, Settings groups) | radius 12, `Surface` fill, padding 12 16 |
| Footer | 28 high, hairline `rule` above |

### 5.5 Icons

One SVG set for every OS: 24-unit viewBox, 2-unit round strokes in
`currentColor`, parsed once with `ui.MustParseSVG` and drawn with `ui.Icon`,
which takes the text color. The spike's `icons.go` is the set, one icon per
DESIGN 13.6 meaning, plus `check` (copied), `chevron` (disclosure, Back) and
`trash` (Clear History). Sizes: 16 in rows and buttons, 14 for the trailing
status glyph, 18 in the drop bar, 48 in the drop zone. The Converting meaning
is `ui.Spinner`.

The app icon is `desktop/icon.png`. Settings > About draws it from an embedded
copy (`go:embed`), so no image is read from disk at run time.

## 6. Screens

The DESIGN 6 tables say what each region holds and in which state. This
section says how mygo draws it.

### 6.1 Window

| Region | mygo | Size and tokens |
|---|---|---|
| Window | `mygo.NewWindow(WindowOptions{Title: app.name.mygo, Width, Height, MinWidth: 640, MinHeight: 480, StateKey: "main", Content: ui.View(a.view)})` (`window.go:61-206`), then `contentMinimum` (section 6.10). The first launch's size is 1200 × 780, or the primary display's work area where that is smaller (`Screen.PrimaryDisplay().WorkArea`), so the title bar never starts off the screen; a saved frame is fitted to its work area (section 2.1). The system's own title bar (`TitleBarDefault`) on every OS | |
| Maximized | On macOS the store saves a maximized window whose frame covers a work area by its frame alone (`keepFrame`, `windowstate.go`): AppKit calls any window at its standard frame zoomed, as one fitted to the screen is, and zooming it again on the next launch made it smaller (found in the macOS review: 1728 × 994 came back as 1708 × 974). On Windows and Linux the store saves what mygo reports. A tiling compositor on Linux reports a tiled window as maximized, so mygo saves `maximized: true`, and the next launch is tiled as before (found in the Linux review) | |
| Root | `ui.Row` filling the window, children stretched | |
| Sidebar | `ui.Themed` (section 5.1) 200 wide, `sidebarFill`, hairline `rule` at its right. A `ui.Sidebar` with `nav.convert` and `nav.history` grows, and a second `ui.Sidebar` with `nav.settings` sits at the bottom. Both bind the same `*string` page path (section 14) | Items as mygo draws them |
| Content | `router.View`, growing | |
| List and detail | `ui.Split(c, &listWidth, list, detail)` with the clamp of section 14 | List column `listFill`, detail `Background` |
| Column header | `ui.Row` 52 high, padding 0 12: page title (`RoleHeading`, level 1), `Spacer`, icon buttons | |
| Detail bar | `ui.Toolbar` 52 high, padding 0 16, `Label(a11y.actions)`: Back (stacked layout only), `Spacer`, view switch, `Spacer`, action buttons | |

### 6.2 Convert, empty

The column header of section 6.1 with the page title `nav.convert` and no
buttons, as Native's empty Convert has, then DESIGN 6.2's content, centered in
the rest of the content area:

| Element | mygo |
|---|---|
| Placement | A column growing below the header, `Justify(Center)`, holds a `ui.Scroll` with `MaxHeightPercent(100)`, which holds the zone in a column with padding 24. The scroll is as tall as the zone up to the page's height, and scrolls past it, as at the minimum size. Inside a scroll no percentage of its height resolves (`ui/layout.go:581-583`), so the column outside it centers the zone |
| Drop zone | `ui.Column` at most 560 wide and narrower when the page is, centered, with the section 5.4 frame, `Role(RoleGroup)`, `Label(a11y.drop_zone)`, the drop target of section 10.1 |
| Glyph | Add files, 48, `TextMuted` |
| Title | Empty-state title. Reads `convert.drop.release` while a drag is over the window |
| Body | `TextMuted`, centered, with the abbreviated output folder |
| Primary button | `ui.PrimaryButton(convert.choose).AutoFocus()` |
| `convert.or` divider | `ui.Row`: a `Box` 1 high growing in `rule`, the word in `TextMuted`, another such `Box` (section 14) |
| URL row | Field and `convert.url.add` button (section 6.3), without the link glyph (DESIGN 6.2) |
| Error | Failed glyph in `Danger` and `convert.url.invalid` in `Text`, until the text changes. The field also gets `Error(convert.url.invalid)` |
| Notices | Section 6.7, below the URL row |

### 6.3 Convert, with items

The list column, top to bottom, gap 8:

| Element | mygo |
|---|---|
| Column header | `nav.convert`, then an icon button Clear Finished (`Label` and `Tooltip` `convert.clear_finished`, disabled when nothing is Done or Failed) |
| Drop bar | `ui.Row` with the section 5.4 frame, padding 0 12: Add files glyph 18 `TextMuted`, `convert.drop.bar` (or `convert.drop.release`) growing, a `ui.Button(convert.choose)` |
| URL row | `ui.Row` gap 8: a field box (32 high, `Background`, 1 px `Border`, `Radius`) holding the link glyph 16 `TextMuted` and `ui.TextInputBase` with `Placeholder(convert.url.placeholder)` and `Label(convert.url.label)`, then `ui.Button(convert.url.add)`. Enter submits |
| Notice | Section 6.7 |
| List | `ui.Scroll` growing, `Focusable()`, `Role(RoleList)`, `Label(a11y.conversions)`, one Tab stop. Up and Down move the selection, `ActiveDescendant` is the selected row. Rows are keyed by item id |

The Convert list is a scroll of keyed rows, not a `ui.List`, because its rows
come and go with transitions (section 7). It holds what one session adds, so it
needs no virtualization.

Row anatomy (DESIGN 6.3), 44 high, inset 0 from the column's padding:

| Part | mygo |
|---|---|
| Leading | File or web page glyph, 16, `TextMuted` |
| Title | File names with middle truncation (section 14), URLs `SingleLine()` with the end cut, `B` |
| Subtitle | `SingleLine()`, `B - 2`, `TextMuted` |
| Trailing | Queued glyph 14 `TextMuted`, `ui.Spinner` named `state.converting`, Warning glyph 14 `Warning`, or Failed glyph 14 `Danger` |
| Fill | `Selected(true)`, then `chosenQuiet` with the 2 px leading bar in the accent when selected, else `rowHover` while hovered (section 5.1) |
| Accessibility | `Role(RoleListItem)`, `Label("<title>, <subtitle>")`, `Selected` |
| Context menu | `Element.ContextMenu` with DESIGN 6.3's items. `detail.show.*` is section 13's per-OS key |

### 6.4 Detail

DESIGN 6.4's table applies. The detail column is a `ui.Column` filling the
pane: detail bar, content, footer.

| Part | mygo |
|---|---|
| View switch | `ui.SegmentedBase` with two segments `detail.preview`, `detail.markdown`, `Label(detail.view)`, track in `Surface` with 1 px `Border`, radius 6, padding 2. The chosen segment's pill is a `Box` in `Background` with a 1 px shadow that moves between segments (section 7) |
| Actions | Icon-only `ui.Button`s, 32 by 32: Copy Markdown, Save As…, Show in folder. Each has `Label` and `Tooltip` equal to its copy key |
| Header | Column gap 4: detail title (`Selectable()`, `RoleHeading` level 2), source line (kind glyph 14, abbreviated path or URL, `B - 2`, `TextMuted`, `SingleLine()`, `Selectable()`), status line (glyph and text, `B`): Done `state.done`, Converting `ui.Spinner` and `detail.converting` with the elapsed `m:ss`, updated by `c.After` each second, Queued `state.queued`, Failed `state.failed` |
| Warnings | A box (section 5.4) named `a11y.warnings` holding `ui.CollapsibleBase`. The trigger row is the Warning glyph, `state.warnings.*` growing, and the chevron, which turns 90° with the panel's `Progress()`. The panel lists each warning as its own selectable `B - 2` text. Open state is per item for the session |
| Body, Done | Preview: the web view element of section 8, growing, or `preview.failed` centered in `Text` when the page could not be rendered (section 8.2). Markdown: `ui.TextArea` with `ReadOnly(true)`, `Font("monospace")`, `FontSize((B - 1) * zoom)`, word wrap, `Label(a11y.markdown_text)`, padding 16, growing. Both sit inside the content inset |
| Body, Converting | `detail.converting.note` in `TextMuted` |
| Body, Queued | `detail.queued`, `TextMuted` |
| Body, Failed | Section 12's error block |
| No selection | `detail.empty`, centered, `TextMuted` |
| Footer | `ui.Row` 28 high, padding 0 24, `B - 2`, `TextMuted`, `SingleLine()` |

History entries use the Done column (DESIGN 6.4), their status line followed
by ` · ` and the finish date of section 6.5. The Markdown is read from
`markdown_path` on a goroutine, through the core's `markdown_body` (DESIGN
3.3), so the entry shows, copies and saves what its Convert item did.
A missing file shows `detail.missing.title` (detail title style) and
`detail.missing.body` in `Text` (DESIGN 12), and only Remove from History
stays enabled.

### 6.5 History

| Element | mygo |
|---|---|
| Column header | `nav.history`, then an icon button Clear History (trash glyph, `Label` and `Tooltip` `history.clear`, disabled when History is empty) |
| Search | `ui.SearchField(c, &query)` with `Label(history.search)` and `Placeholder(history.search)`, full width, below the header (`docs/ui/search-field.md`). Filters on `Changed`: case-insensitive substring of title or source |
| List | `ui.List(c, &historyList, n, row)` growing, `Label(nav.history)`. `ListState.Key` is the entry id, `Label` the row name, `Selected` the selected index among the filtered entries (`docs/ui/list.md`). Rows are 44 high and drawn as section 6.3's rows |
| Row | Kind glyph, title, subtitle = finish date (DESIGN 6.5: `date.today`, `date.yesterday`, else the app language's medium date, with its short time) and ` · ` `state.warnings.*` when there are warnings, trailing Warning glyph |
| Context menu | DESIGN 6.5's items |
| Empty | The column header, its Clear History disabled, then `history.empty.title` (empty-state title) and `history.empty.body`, centered below it, without a split |
| No matches | `history.no_results` as the list's child, shown while it has no rows (`docs/ui/list.md`, Layout), padding 12, `TextMuted` |
| Detail | Section 6.4, Done column |

### 6.6 Settings

A page in the content area: `ui.Scroll` filling it, holding a column of at most
640, padding 24, gap 24. The page title `nav.settings` (page title style,
`RoleHeading` level 1) comes first. Each group is a heading (`RoleHeading`
level 2) above a box (section 5.4) whose rows are separated by `rule`
hairlines. Rows are at least 52 high: title and note on the left (`B`, then
`B - 2` `TextMuted`), the control on the right. A disabled row dims as a
whole: its title and note at 50% opacity with its switch, as Native's rows do.
The page asks the core for the configuration file (`ConfigFile`) as it opens,
after Create or Open, and when the window comes back to the front, not on every
frame (DESIGN 6.6).

| Group | Row | mygo |
|---|---|---|
| `settings.conversion` | `settings.llm`, `settings.ocr`, `settings.screenshot`, `settings.desc`, `settings.alt`, each with its `.note` | `ui.Switch` with `Label` = the row title. `settings.desc` and `settings.alt` are `Disabled` while `settings.llm` is off |
| `settings.output` | `settings.output.folder`, note, then the abbreviated path (`Selectable()`) | `ui.Button(settings.output.change)` opens `mygo.Dialog.Open` with `Directory: true` (`dialog.go:34-35`), `CreateDirectories: true` and `DefaultPath` the output folder, or its nearest folder that exists (section 6.7). `ui.Button(settings.output.show)` calls `Shell.ShowItemInFolder` |
| `settings.config` | `settings.config.user` with the file's path (or `settings.config.user.missing`) as its note, `settings.config.builtin` with its note | `ui.RadioGroup` of two `ui.Radio`, `Label(settings.config)` (`ui/api_gen.go:1644-1654`) |
| | `settings.config.open` or `settings.config.create` | `ui.Button`. Open calls `Shell.OpenPath`. Create writes `{}` and a newline, then opens |
| `settings.about` | App icon 48, `app.name.mygo` (page title style), `settings.about.app` and the app version, `settings.about.core` and `markitai_app_version()`, `settings.about.copyright` | Text rows, values `Selectable()` |

DESIGN 6.6's rules for the configuration note, Create, Open and an unwritable
folder apply.

### 6.7 Dialogs and notices

| Need | mygo |
|---|---|
| Quit with running conversions | `ui.AlertDialog(c, &askingQuit, quit.title, body, quit.confirm, common.cancel)`. The body is `quit.body.*`, then `quit.body.queued` when items are Queued, after a space in English and directly in zh-Hans. The last button, `common.cancel`, is the default: accent, focused, Enter. Escape cancels (`ui/feedback.go:161-218`) |
| Clear History | `ui.AlertDialog(c, &askingClear, history.clear.title.*, history.clear.body, history.clear.confirm, common.cancel)` |
| Notice (folder refused, duplicate, history reset, save or open failed) | A `ui.Row` below the URL row, gap 6: Warning glyph 14 in `Warning`, the text in `Text`. `Role(RoleStatus)`, and `c.Announce` of its text when it appears. It belongs to the page that gave it: Convert's below its URL row, History's below its header, Settings' on Settings, and in the stacked layout at the top of that page's detail, so a failed Save As shows where it was asked. It clears 8 s after it was given (`c.After`), not after it first shows, or at the next add, and `settings.output.unwritable` also when a folder is taken |
| Copied | The Copy Markdown button's glyph becomes `check` for 1.5 s, and `c.Announce(notice.copied)` |
| File pickers | `mygo.Dialog.Open` (multiple files; or one folder for Settings' Change…) and `mygo.Dialog.Save` with `Parent` set, which makes them sheets on macOS (`dialog.go:24-25`, `internal/darwin/dialog.go:19-28`). They block, so they run on a goroutine and post the answer back (`dialog.go:9-10`). Change… and Save As… start at an absolute `DefaultPath` (section 10.2): a bare name starts GTK's chooser in the working directory, the data folder (`internal/linux/dialog.go:37-48`, `filepath.Dir` of a name is `.`). Save As… lists Markdown files (`Filters: [{Markdown, [md]}]`). A dialog that fails to show is a cancel, with the error in the log (`desk.go`) |

The two alerts are drawn in the window, so their text follows the app's
language on every OS. mygo draws no destructive style. The button's words
carry the meaning.

### 6.8 Errors in the detail

See section 12.

### 6.9 Fatal

On an ABI mismatch the view draws only a centered column: `fatal.abi.title`
(page title style), `fatal.abi.body` in `Text` (DESIGN 12), and
`ui.PrimaryButton(quit.confirm)`, which calls `App.Quit`. No sidebar and no
shortcuts other than Quit. Closing the window quits on every OS, macOS too,
and files handed to the app (Dock, Open With, a second launch) are ignored.

### 6.10 Window sizes and narrow widths

| | Value |
|---|---|
| Default | 1200 × 780 |
| Minimum | The content area's, 640 × 480, on every OS (DESIGN 14). mygo gives `MinWidth` and `MinHeight` to GTK as the content's least size (`internal/linux/window.go:390-404`), and to macOS and Windows as the window's, frame and title bar included (`setMinSize:`, `internal/darwin/window.go:772-774`; `MinTrackSize`, `internal/windows/window.go:477-482`). There `contentMinimum` adds the frame to the window's minimum, so the content keeps 640 × 480. On macOS (`window_darwin.go`) the frame is the difference between `Window.Bounds` and `Window.ContentBounds`, in AppKit's points; without it a titled 640 × 480 frame held 640 × 448. On Windows (`window_windows.go`) it is measured in pixels (`GetWindowRect`, `GetClientRect`, `GetDpiForWindow`) and rounded up to whole DIPs: mygo's `Bounds` and `ContentBounds` round each size (`toDIP`, `internal/windows/window.go:863`), and at 192 dpi a frame 71 pixels high read as 35 DIPs and left the content 959 pixels high. The build before this round's fix held 1254 × 889 pixels of content at its least size (found in the Windows run). A tiling compositor may give the window less: Hyprland tiled it 621 wide (found in the Linux review) |
| Navigation | Never collapses. At the minimum width the sidebar keeps 200 and leaves 440 for the page. The empty Convert page's zone narrows to fit it, and fits without scrolling at 640 × 480 (section 6.2) |
| Stacked list and detail | When the content width (`c.Size()` minus 200, `ui/context.go:411`) is below 760, Convert and History stack |

Stacked layout:

- The list column fills the content area. Choosing a row with a click, the
  chosen row's included, Enter or a context menu's default pushes
  `/convert/item` or `/history/item`. Up and Down only move the selection.
- The detail page has a Back button at the leading end of its detail bar:
  chevron glyph and `common.back`, `Label(common.back)`.
- Back, Esc, ⌘[ on macOS and Alt+Left elsewhere (the router's own keys,
  `docs/ui/navigation.md:84-87`), and a mouse's back button, over the
  preview too (section 8.3), return to the list with the row still selected.
  Over the preview the forward button does nothing; over the native UI it is
  the router's Forward (`ui/router.go:590-607`).
- Leaving the stacked width while on an `/item` page replaces it with the list
  page at once (section 7). The selection is kept. Only one `ui.List` shows a
  `ListState` in a frame (`ui/list.go:474`), so History's page going away
  builds no list while the window is not stacked (`history.go`).
- Empty Convert, empty History and Settings never stack (DESIGN 14).

## 7. Motion

DESIGN 12 is the rule: list insertion and removal, and the motion Markitai
Native's system controls make for the same change (owner decision, 2026-11-04). The app
adds nothing else. Each animated change, with the mygo call that makes it:

| Change | Element | mygo | Native's counterpart | Timing |
|---|---|---|---|---|
| Row enters (Convert) | The new row and those below | `Transition(rowMotion)` with `rowMotion = ElementTransition{Duration: 150ms, Ease: EaseOut, Position: true, Enter: &Motion{}, Exit: &Motion{}}`: it fades in while the rows below slide down. `Position` alone animates, so a row's colors, as a selection's fill, change at once (`ui/transition.go:30-34`) | The list's insertion | 150 ms, ease out |
| Row leaves (Convert) | The removed row and those below | Same transition: the row fades out in place while those below slide up | The list's removal | 150 ms, ease out |
| Segmented switch | The chosen-segment pill | A `Box` built in the chosen segment with `Transition(ElementTransition{Animation: &ui.Snappy, Match: viewPill{}})`: it flies from the old segment to the new (`docs/ui/transitions.md`, Shared elements). The body swaps at once | The segmented control's selection | `ui.Snappy` spring |
| Warnings disclosure | Panel and chevron | `CollapsibleBase` grows the panel (`ui/collapsible.go:38`). The chevron's `Rotate` is `90 * Progress()` | The disclosure's expansion and its arrow | 200 ms |
| Stacked push and pop | Content | `router.Transition` is `TransitionAuto` only while the layout is stacked and the page stays under the same sidebar item, as `/history` and `/history/item`: the deeper page slides in from the right and back out to the right (`ui/router.go:119-123`) | The navigation stack's push and pop | 250 ms, ease out |
| Converting | Spinner | `ui.Spinner` | The platform's spinner | Continuous |

Every other change shows at once: a sidebar page change, and the stacked item
page that a wider window replaces with the list (`TransitionNone` otherwise,
`view.go`), hover and selection fills, the drop highlight and its heading,
a row's status glyph and subtitle, a notice, and Copy's check. History's rows
have no transitions: `ui.List` places them without (`docs/ui/transitions.md:167-169`).
mygo draws `AlertDialog` at once (`ui/feedback.go:176-219`). mygo's own
controls keep the motion they draw themselves, as a switch's knob
(`ui/widgets.go:283`), as Native's system controls do.

Reduce Motion. While `c.Preferences().ReduceMotion` is true, the view's first
call is `c.DisableAnimations()`. That frame then shows every value,
transition, color, pose, content change and router page at its target at once
(`ui/context.go:473-482`, `ui/transition.go:659`, `ui/transition.go:824`,
`ui/motion.go:322`, `ui/router.go:638`, `ui/content.go:83`). This removes the
fades that mygo keeps under Reduce Motion (`ui/api_gen.go:6114-6117`), as
DESIGN 12 asks. Spinners keep turning: `Loop` ignores both
(`ui/widgets.go:62-84`, `ui/preferences.go:11-15`).

## 8. Preview

### 8.1 The web view

The window makes one web view at `WhenReady`, never in the view
(`docs/ui/web-view.md:69-72`):

```go
preview, err := win.NewWebView(mygo.WebViewOptions{
	Page: mygo.PageOptions{
		DevTools:      mygo.DevToolsDisabled,
		Session:       mygo.NewEphemeralSession(),
		ContentRules:  []*mygo.ContentRules{noNetwork},
		PreloadScript: previewJS(runtime.GOOS),
	},
})
```

The detail builds `ui.WebView(c, preview).Grow(1).Label(a11y.preview)` only
while a Done item's Preview tab shows, except for the one frame a focus
action leaves it out (section 8.3). An element the view does not build
hides its web view and keeps its page (`ui/webview.go:26-27`). On Linux, the
first frame that builds it again gives it a bottom margin of 1 DIP and asks
for another frame, because of a GTK sizing defect (section 14).

| DESIGN 8 concern | mygo |
|---|---|
| Load | `Page.LoadHTML(html, baseURL)` with `baseURL` the `file://` URL of the result folder with a trailing slash (`window.go:1345-1351`). macOS calls `loadHTMLString:baseURL:` (`internal/darwin/window.go:1379-1388`), Linux `webkit_web_view_load_html` (`internal/linux/window.go:1100-1103`), Windows serves the document at the base URL (`internal/windows/webview.go:667-684`). On macOS `loadHTMLString:baseURL:` grants the page no read access to the folder, and `loadFileURL:allowingReadAccessToURL:` grants it only for a file WebKit can show: on macOS 15.7 a Markdown file becomes a download into the Downloads folder, with its permission prompt on the main thread (measured). So on macOS `baseURL` is `markitai-preview://<hash of the folder>/`, a scheme `previewFolders` serves through `Protocol.Handle` (`protocol.go:53`) from inside the folder only, and `preview_html` gets the same URL as `files_url`. Section 18 has the outcome per OS |
| Scripts | The page's own scripts never run: the document's CSP `default-src 'none'` blocks them, and `preview_html` escapes raw HTML (DESIGN 3.3). The only script is `previewJS` (section 8.3), injected by the host, which the CSP does not govern (section 18, claim 2) |
| Network | The CSP, plus `noNetwork = CompileContentRules("preview-no-network", rules)` blocking `^https?://` and `^wss?://` (`content_rules.go:28-100`), plus an ephemeral session (`session.go:88-93`). Compile once, at `WhenReady`, on the main thread. `CompileContentRules` waits for WebKit's answer, which mygo delivers on the main thread (`cookies.go:270-290`, `internal/darwin/content_rules.go:62-74`); on the main thread mygo keeps handling events while it waits, but a goroutine's call deadlocks when the main thread waits for that goroutine (measured: no first frame) |
| Navigation | `Page.OnWillNavigate`: allow a URL equal to the base URL with only a fragment added. Open `http`, `https` and `mailto` with `Shell.OpenExternal` and `PreventDefault`. Prevent everything else (`window.go:1738-1741`, `events.go:32-42`). No `SetWindowOpenHandler`: without one, mygo opens `http` and `https` in the browser and denies the rest (`window.go:1549-1552`) |
| Context menu | `Page.OnContextMenu`: keep only the item with ID `ContextMenuCopy`, relabeled `common.copy`, when text is selected. Otherwise `PreventDefault` (`context_menu.go:25`, `context_menu.go:124`) |
| Developer tools | `DevToolsDisabled` (`window.go:38-46`), which also turns off WebView2's browser keys (`internal/windows/webview.go:205-212`) |
| Background | Two parts, both the detail's `Background` as `#rrggbb`. `WebView.SetBackgroundColor`, again when the theme changes (`webview.go:142-146`), fills the web view until the page paints. Not `Transparent`: the web view sits in a clip view under mygo's surface, which leaves a hole there (`internal/darwin/webview.go:15-24`), so a transparent page shows the window's backing, not the detail (measured on macOS: `#22201f` beside a `#18181b` detail in dark). The preview stylesheet's page background is transparent (DESIGN 15), and WebKitGTK paints a transparent page in the dark color scheme `#1e1e1e` whatever the web view's color (found in the Linux run, and in a probe with a green web view). So the app also ends the document's head with `<style>html { background: <color>; }</style>`, which the document's CSP allows (`style-src 'unsafe-inline'`), and loads the document again when the color changes |
| Zoom | `Page.SetZoomFactor(view.zoom)` (`window.go:1409`) |
| Appearance | On a change of `c.Theme().Dark`, render again with the new `theme` and load it |
| Missing web engine | When `NewWebView` fails, the Preview tab shows `preview.webview2_missing` on Windows and `preview.unavailable` elsewhere, centered, and the Markdown tab works |
| Accessible name | `a11y.preview` on the element. On Windows, Narrator finds the page's elements in WebView2's own window (`docs/ui/web-view.md:280-282`) |

### 8.2 Rendering

`preview_html` runs on a goroutine with `{markdown, theme, base_dir}` (DESIGN
3.3). Results are cached by `(item id, theme)`, at most 8. When the shown item
or theme changes, the view keeps the web view element unbuilt until the new
document is loaded (`Page.OnDidFinishLoad`), so the old document never shows
under the new title. A failure is not cached: the Preview tab shows
`preview.failed` centered, the Markdown tab works, and the next time the item
shows, after another item or page, it renders again (DESIGN 8).

### 8.3 Keyboard focus and shortcuts in the page

What mygo does. A click on the page gives it the keyboard and takes the focus
from the native UI (`ui/webview.go:149-167`). Tab stops at the web view, moves
through the page's elements, and past its last element returns to the next
native control (`ui/webview.go:169-180`, `docs/ui/web-view.md:122-125`). While
the page has the keyboard, keys go to the system web view and not to the
window's `c.Shortcut`. On macOS the web view is a sibling of mygo's surface,
not a child (`internal/darwin/webview.go:46-47`), so unhandled keys do not
reach it. On Windows, WebView2 hands keys only to the window's menu
accelerators (`internal/windows/webview.go:610-656`). mygo's page runtime sends
back only Escape, which closes popovers and nothing else
(`internal/bridge/bridge.js`, `bridge.go:327-328`, `flyout.go:272-276`).

mygo gives the keyboard back to the native UI only as it hides a web view
that has it (`internal/darwin/webview.go:90-93`, `internal/linux/webview.go:195-198`,
`internal/windows/webview_embed.go:278-281`). Focusing a native control
while the page has the keyboard leaves the keys with the page. mygo allows a
navigation the page did not ask for, as the history's back and forward,
without asking `OnWillNavigate` (`window.go:1886`).

What the app does. `previewJS(goos)` is the page's preload script, and its
calls go to the `Preview` service (section 2.3), which posts each to the UI
thread. The page is trusted because its base URL is `file:` or a scheme the
app handles (`window.go:1314-1326`), so the calls go through, and it is not `Untrusted`,
which would refuse them (`page.go:375-392`). The script keeps the page an
app's part, not a browser (owner decision, 2026-11-04):

- Keys. A `keydown` listener in the capture phase takes each key combination
  of section 9.2 that the page does not need: `preventDefault()`, then
  `Preview.Press("<action>")`, which runs the same `app` method a native
  shortcut runs. The page keeps ⌘C or Ctrl+C, ⌘A or Ctrl+A, arrows, Page Up
  and Page Down, Home, End, Space and Tab. Forwarded actions: Choose Files,
  Add Web Address, Go to Convert, Go to History, Settings, Switch Preview and
  Markdown, Copy Markdown, Save As, Show in folder, Search History, the three
  zoom keys, Back (Esc, ⌘[ or Alt+Left), and on Linux Close and Quit.
- Any keyboard layout. A key goes by the US key that types what it types, as
  mygo takes the native UI's keys (`platform.KeyForRune`,
  `internal/platform/surface.go:467`): an ASCII letter in `event.key` is
  `Key` and the letter, a digit `Digit` and the digit, and the eleven US
  punctuation characters (`- = , . / ; ' [ ] \` and the backquote) their US
  keys. A key typing none of them, as a Cyrillic letter, `é` or `&`, goes by
  `event.code`, its place, as mygo's `KeyByPosition` does with a shortcut
  modifier held (`internal/platform/surface.go:530-531`). So on AZERTY the
  page reads Ctrl+A at QWERTY's Q as Select All, Ctrl+, at QWERTY's M as
  Settings, Ctrl+- at the 6 as Zoom Out, and Ctrl+é and Ctrl+& as History
  and Convert by their places, as the native UI does. Found in the Linux run
  with a French keymap sent by a virtual keyboard: in the page Ctrl+A
  selected its text, Ctrl+- zoomed out and Ctrl+, opened Settings. Compared
  by `event.code` alone, as before, Ctrl+A there quit the app on Linux and
  Ctrl+Z closed its window (the Linux review).
- Mouse buttons. The back button (the page's button 3) runs Back on
  `mouseup`, and the forward button (4) does nothing; the script prevents
  both buttons' `mousedown`, `mouseup` and `auxclick`, so neither moves the
  page's history. WebKitGTK gives the page GDK's buttons 8 and 9 both as
  button 0 and moves no history for either (found in a WebKitGTK 2.54
  probe), so on Linux `toolkitReady` hooks `button-release-event` of every
  `WebKitWebView` with a GTK emission hook and runs Back for GDK's button 8
  (`toolkit_linux.go`).
- Drops. A drop on the page goes to the app, as one on the content area
  (section 10.1), and the page never opens it. While a drag that did not
  start in the page (`dragstart` and `dragend`) carries files, a URI list,
  `text/x-moz-url` or text, `dragover` asks for a copy and `drop` is
  prevented. Links: the lines of the first of `text/uri-list`,
  `text/x-moz-url` (every other line: a URL, then its title) and
  `text/plain` that has any, without blank lines and `#` comments, go to
  `Preview.Drop([], links)`, which adds the `http` and `https` ones
  (`droppedOnPreview`, `app.go`). Files: mygo reads their paths beside the
  page's drop and gives them to a trusted page as `mygo:file-drop`
  (`events.go:65`, `docs/architecture.md:833-849`), which goes to
  `Preview.Drop(paths, [])`. On Linux the native content takes a file
  dropped on the preview, and it is added once (found in the Linux run).
- The document is a Tab stop. The script sets `document.body.tabIndex = 0`,
  so Tab stops at a page with no element of its own to stop at, and the
  arrows then scroll it.
- Wheel zoom. On Windows the script also takes Ctrl with the wheel, which
  WebView2 would use to zoom the page apart from `view.zoom`
  (`internal/windows/webview.go:215`), and hands `wheelDeltaY` to
  `Preview.Wheel`. WebView2 gives the page a notch as 120 divided by the
  page's zoom, in whole pixels: at 2.0 a notch came as 60, and counting
  120 of them stepped on every other notch (found in the Windows run). So
  `app.wheeled` (`actions.go`) multiplies the delta by the zoom the web view
  has, steps Zoom In or Zoom Out for each notch, a little short of one to
  allow for the rounding, and starts the count again when the wheel turns
  the other way.

Focus actions from the page (owner decision, 2026-11-04). Add Web Address and Search
History focus a native field. Run from the page's keys, or from a menu item
chosen while the page has the keyboard, they first leave the web view out of
one frame (`previews.hold`, set in `runLater` and cleared by a posted
function, `actions.go`): mygo hides it and gives the keyboard back, the
field takes the focus, and the next frame shows the page again. Found in the
macOS and Linux runs: with the page clicked, ⌘L or Ctrl+L and then 5 typed 5
in the URL field, and ⌘F or Ctrl+F in the search field; before, the focus
stayed in the page and the 5 was lost.

On macOS a key the page leaves unhandled goes on to the menu bar. The
script's `preventDefault` marks it handled, so the action runs once, through
the script (section 18, claim 3).

## 9. Menus and shortcuts

### 9.1 macOS menu bar

`App.SetMenu` at `WhenReady`. Role menus take mygo's own titles and relabel
with the app's language (`menu.go:390-401`, `menu_text.go:34-66`).

| Menu | Items |
|---|---|
| App (`RoleAppMenu` with its own `Submenu`) | `menu.about.mygo` (shows the About panel with `ApplicationName` `Markitai Mygo`, the app version once, `settings.about.core` and the core's version as credits, and `settings.about.copyright`, as Native's; found in the macOS run: Version 2.3.2, or 版本 2.3.2, where it said Version 2.3.2 (2.3.2) before), separator, `menu.settings` ⌘,, separator, `RoleServices`, separator, `RoleHide`, `RoleHideOthers`, `RoleUnhide`, separator, `RoleQuit` |
| File (`RoleFileMenu`) | `convert.choose` ⌘O, `menu.add_url` ⌘L, separator, `RoleClose` (AppKit adds Close All after it), `detail.save_as` ⇧⌘S, `detail.show.macos` ⇧⌘R, separator, `detail.retry` ⌘R, `convert.remove` or `history.remove` ⌘⌫ |
| Edit (`RoleEditMenu`) | `RoleUndo`, `RoleRedo`, separator, `RoleCut`, `RoleCopy`, `RolePaste`, `RoleSelectAll`, separator, `detail.copy` ⇧⌘C, separator, `history.search` ⌘F |
| View (`RoleViewMenu`) | `nav.convert` ⌘1, `nav.history` ⌘2, separator, `menu.toggle_view` ⇧⌘M, separator, `menu.zoom_in` ⌘=, `menu.zoom_out` ⌘-, `menu.actual_size` ⌘0 |
| Window (`RoleWindowMenu`) | mygo's default |

Each item's `Click` calls the `app` method. Items are enabled and disabled each
frame with `SetEnabled` from the same state that enables the buttons, and the
remove item is relabeled with `SetLabel` on the History page. The items' places
are Markitai Native's (DESIGN 11). While the URL field or History's search
field has the focus, the remove item has no key equivalent (`SetAccelerator`
with none): AppKit gives a menu's key equivalents a key before the focused
view, so ⌘⌫ removed the selected row instead of deleting the field's text to
the start of the line (found in the macOS review). Paste reaches
`c.Shortcut(ui.Cmd, ui.KeyV)` when the focus takes no paste
(`docs/ui/input.md:167-172`).

Windows and Linux have no menu bar (`menu.go:944-950`). Every shortcut is a
`c.Shortcut` in the root view.

### 9.2 Shortcuts

`ui.Cmd` is ⌘ on macOS and Ctrl elsewhere (`docs/ui/input.md:164`). DESIGN 11's
actions:

| Action | macOS | Windows and Linux | mygo |
|---|---|---|---|
| Choose Files… | ⌘O | Ctrl+O | `ui.Cmd, ui.KeyO` |
| Add Web Address | ⌘L | Ctrl+L | `ui.Cmd, ui.KeyL`: go to Convert, focus the URL field through its `Handle` |
| Paste files or a URL (focus not in a text field) | ⌘V | Ctrl+V | `ui.Cmd, ui.KeyV`, section 10.1 |
| Go to Convert, History | ⌘1, ⌘2 | Ctrl+1, Ctrl+2 | `ui.Cmd`, `ui.Key1` or `ui.Key2` |
| Settings | ⌘, | Ctrl+, | `ui.Cmd, ui.KeyComma` |
| Switch Preview and Markdown | ⇧⌘M | Ctrl+Shift+M | `ui.Cmd\|ui.Shift, ui.KeyM` |
| Copy Markdown | ⇧⌘C | Ctrl+Shift+C | `ui.Cmd\|ui.Shift, ui.KeyC` |
| Save As… | ⇧⌘S | Ctrl+Shift+S | `ui.Cmd\|ui.Shift, ui.KeyS` |
| Show in folder | ⇧⌘R | Ctrl+Shift+R | `ui.Cmd\|ui.Shift, ui.KeyR` |
| Retry | ⌘R | Ctrl+R | `ui.Cmd, ui.KeyR` |
| Remove from List or History | ⌘⌫ | Delete | `Element.Shortcut` on the focused list. On macOS the menu item's ⌘⌫ too, except while a text field has the focus (section 9.1) |
| Search History | ⌘F | Ctrl+F | Go to History, focus the search field |
| Zoom in, out, actual size | ⌘=, ⌘-, ⌘0 | Ctrl+=, Ctrl+-, Ctrl+0 | `ui.Cmd` with `ui.KeyEqual` (with or without Shift), `ui.KeyMinus`, `ui.Key0`. Steps of 0.1 from 0.5 to 3.0. On Windows, Ctrl with the wheel over the preview steps too (section 8.3) |
| Back (stacked layout) | ⌘[ or Esc, the mouse's back button | Esc or Alt+Left, the mouse's back button | The router's keys (`ui/router.go:590-607`), plus `ui.KeyEscape`. Over the preview, the page's script (section 8.3) |
| Move in a list | ↑ ↓ | ↑ ↓ | The list's own |
| Close the window | ⌘W | Alt+F4 on Windows, Ctrl+W on Linux | `RoleClose`, the system, `ui.Cmd, ui.KeyW` |
| Quit | ⌘Q | Alt+F4 on Windows, Ctrl+Q on Linux | `RoleQuit`, the system, `ui.Cmd, ui.KeyQ` |

Linux follows the Windows column. The mygo build is a GTK 3 app with its own
drawing, not a GNOME app, so GNOME's Alt+1 and Alt+2 do not apply. mygo
matches a shortcut by the key that types what the key types, else, with a
shortcut modifier held, by its place (`internal/platform/surface.go:467`,
`internal/platform/surface.go:530-531`), so each works on any keyboard layout;
the preview's script follows the same rule (section 8.3). Shortcuts are
off while an alert shows, except Escape and Enter, which the alert takes.

## 10. Behavior per OS

DESIGN 7 applies. These rows replace its platform rows.

### 10.1 Adding sources

| Way | mygo |
|---|---|
| Drop on the window | The content area is the drop target on Convert and History: `ui.DataDragOver` drives the highlight, and `ui.DropData` takes `transfer.DropOptions{Formats: {transfer.FileList, transfer.URIList, "text/x-moz-url"}}` (`ui/api_gen.go:1130-1140`, `docs/ui/drag-and-drop.md:60-87`). Files become items. Without files, the `http` and `https` entries of `Data.URLs()` become URL items (`transfer/data.go:268-285`); without those, the first line of `text/x-moz-url`, read as UTF-16 or UTF-8, when it is one `http` or `https` URL. Chromium on Linux offers a dragged link only as `text/x-moz-url`, HTML and text, with no URI list (found in the Linux review). Each entry must start with `http://` or `https://`, as a paste's (DESIGN 7.1). macOS registers `public.url` for this (`internal/darwin/surface_drag.go:178-187`). A drop on History goes to Convert |
| Drop on the preview | The page's script hands links and files to the app, which adds them as a drop on the content area does (section 8.3) |
| `convert.choose` | `mygo.Dialog.Open(OpenDialogOptions{Parent: win, Title: convert.choose, Multiple: true})` (`dialog.go:128-159`) |
| Paste | `c.Services().ReadClipboardData(transfer.FileList, transfer.Text)` (`ui/services.go:41-51`): files first, else text that is one `http` or `https` URL with its scheme, as Native's (DESIGN 7.1). Other text, `example.com` or a sentence with a dot, adds nothing |
| Open with the app | macOS: `App.OnOpenFile` (Dock drops, Open With), which `CFBundleDocumentTypes` allows for any file (`internal/darwin/app.go:345-360`). Windows and Linux: `os.Args[1:]` at launch, and the `args` of `App.OnSecondInstance` (`single_instance.go:119-121`), keeping arguments that are existing paths. `OnOpenFile` is not used there, because it passes only extensions in `fileAssociations` (`deeplink.go:95-123`). The Linux launcher entry passes no files (section 14) |

### 10.2 Copy, Save As, Show

| Action | mygo |
|---|---|
| Copy Markdown | `c.WriteClipboard` (`ui/context.go:491-492`) |
| Save As… | `mygo.Dialog.Save(SaveDialogOptions{Parent: win, DefaultPath: <folder>/<name>.md, Filters: [{Markdown, [md]}]})` (`dialog.go:161-189`). The folder is the output folder, or its nearest folder that exists (`existingFolder`, `app.go`), and the name is the result folder's. The dialog asks before it replaces the file it names. A name without an extension gets `.md`, and that file is made new (`O_EXCL`): a file already there is left as it is and the save fails, since the dialog asked about another name. Then DESIGN 7.2's write and `.markitai` copy run on a goroutine. A failure shows the notice `error.save_failed` on the page where Save As… was chosen (section 6.7). Found in the Linux run, through the portal and GTK's own chooser: Save As… started in the output folder with the result's name and the Markdown filter, a name given without an extension saved `<name>.md`, and Change… started in the output folder |
| Show in folder | `mygo.Shell.ShowItemInFolder(path)` (`modules.go:45-48`). On Linux the app asks the file manager itself, through `org.freedesktop.FileManager1.ShowItems`, else opens the folder, so that the program starts in the user's environment (section 10.5, `shell_linux.go`) |
| Open configuration | `mygo.Shell.OpenPath(path)` (`modules.go:40`); on Linux the app's own launch in the user's environment (section 10.5). A failure shows `error.open_failed` |

### 10.3 Quit and closing the window

| OS | Rule |
|---|---|
| macOS | Closing the window hides it: `Window.OnClose` calls `PreventDefault` and `Hide` unless the app is quitting (`window.go:1661-1663`, `events.go:8-24`). Conversions keep running. `App.OnActivate` shows the window again when the Dock icon is clicked with no window visible (`app.go:494-499`). `App.OnWindowAllClosed` is set, so mygo does not quit (`app.go:473-478`). The fatal window (section 6.9) is the exception: it has no `OnClose`, so closing it quits |
| Windows | Closing the window quits. `Window.OnClose` covers the close button and Alt+F4 (`internal/windows/window.go:421-425`) |
| Linux | Closing the window quits. `Window.OnClose` covers the close button and Ctrl+W. Ctrl+Q runs the same path |

Every quit path ends in `App.OnBeforeQuit` or `Window.OnClose`
(`app.go:220-248`). ⌘Q, the Dock's Quit and logout reach them on macOS
(`internal/darwin/app.go:289`, `internal/darwin/app.go:304-330`). With any
item Converting, the handler calls `PreventDefault`, shows the window,
and sets `asking = quit`. DESIGN 7.3's rules then apply: the body follows the
queue each frame, the alert closes and the app quits when nothing converts any
more, and Quit deletes the claimed folders of Converting items, sets
`quitting`, and calls `App.Quit`. Windows logoff cannot be canceled
(`internal/windows/app.go:463-468`). The app then deletes those folders in
`App.OnQuit`.

### 10.4 Appearance

The theme follows the system (`modules.go:480-483`). There is no theme
setting. A change re-renders the preview (section 8.1).

On Linux mygo reads the system's appearance from the portal, but tells GTK
only when the app sets the theme's source (`internal/linux/system.go:440-462`),
so GTK's own widgets, as the rows' context menus, stayed light in a dark
system (found in the Linux review). `prepareToolkit` (`toolkit_linux.go`)
calls `Theme.SetSource(ThemeSystem)` before `App.Run`, and again after each
change mygo reports, on the main thread once GTK's notification has
returned: called inside the notification, which is where mygo reports the
change (`modules.go:504-525`), it never returned (found in the Linux run).
The file chooser is the portal's where the desktop has one, which follows
the desktop, else GTK's own, which follows the source the app sets (section
10.5). On macOS and Windows the app does not set the source.

### 10.5 Language

| Rule | mygo |
|---|---|
| Choice | DESIGN 7.5: zh-Hans when the first of `mygo.App.PreferredLanguages()` starts with `zh`, else English. On macOS that list is the global `AppleLanguages`, never the app's own domain (`internal/darwin/locale.go:44-84`). On Linux it is `LANGUAGE`, then the locale (`internal/linux/locale.go:19-34`), read before the app sets `LANGUAGE`. `MYGO_LOCALE`, when set, goes first, for development. `main` chooses first, before the folders, the toolkit and the single-instance lock (section 2.3) |
| App strings | `desktop/strings.json`, embedded through the copy in `internal/l10n/strings.json`, which a test keeps byte-equal |
| mygo's own text | `App.SetLanguages("en", "zh-Hans")`, then `App.SetLocale(<choice>)` (`locale.go:77-180`). Role menus, text-field context menus and mygo-drawn labels follow at once. `SetLocale` also stores the choice, on macOS in the app's defaults (`internal/darwin/locale.go:93-104`). Each launch chooses again from the global list |
| Windows system text | mygo sets the process's preferred UI languages to the app's tags (`internal/windows/locale.go:112-129`). Windows keeps its own Simplified Chinese text under `zh-CN`, not `zh-Hans`, so the file dialogs stayed in the system's language (found in the Windows review). At `WhenReady`, after mygo set them, the app sets them again with `zh-CN` before `zh-Hans` (`lang_windows.go`, `SetProcessPreferredUILanguages`). With `MYGO_LOCALE=zh-Hans` on a system shown in English, the open dialog's own text (组织, 新建文件夹, 打开, 取消) is then in Simplified Chinese |
| Linux system text | The file chooser is the desktop's, through the portal, as Native's, when the portal has one: before GTK starts, `prepareToolkit` asks `org.freedesktop.portal.Desktop` for its `FileChooser` interface's `version`, waiting up to 3 s, as the portal may have to start, and sets `GTK_USE_PORTAL=1` only when it answers (`toolkit_linux.go`). mygo's strings in the app's language give its title and accept button (`internal/linux/dialog.go:72-106`), and the portal shows the rest in the desktop's language. GTK told to use a portal without a file chooser waits for an answer that never comes: no dialog showed, and quitting then aborted in `gtk_file_chooser_native_portal_hide` (the Linux review, and the build before this fix in the Linux run). Without one the chooser is GTK's own, in the app's process, kept within its screen (section 14): in the Linux run, with a portal without a file chooser and with no portal, Choose Files… showed at 1132 × 572 on a 1280 × 720 screen, and quitting was clean. GTK's own words in the app's process, as those of that chooser, follow gettext's `LANGUAGE`, which `main` sets to `zh_CN` or `en` before GTK starts, as Native does (`lang_linux.go`). Found in the Linux run: with `LANGUAGE=zh_TW` and the in-process chooser forced, its places read 主目录, not 家目錄 |
| Programs the app opens (Linux) | They get the user's environment, not the app's. Before the app sets a variable for its toolkit (`LANGUAGE`, `GTK_USE_PORTAL`, `BAMF_DESKTOP_FILE_HINT`), and before `App.Run`, where mygo sets `WEBKIT_DISABLE_DMABUF_RENDERER=1` for WebKitGTK when it is unset (`internal/linux/app.go:112-116`), it records the user's value, or none (`appEnv`, `env.go`). Links, the configuration file and the file manager start through GIO with a `GAppLaunchContext` that gives back those values (`launchAsUser`, `shell_linux.go`); mygo's `Shell` launches with no launch context, in the app's environment (`internal/linux/system.go:179-185`). Found in the Linux run: a link opened from the preview reached its handler with none of the four set; before the fix it had `LANGUAGE=zh_CN`, `GTK_USE_PORTAL=1` and `WEBKIT_DISABLE_DMABUF_RENDERER=1` |
| macOS system text | AppKit's own text (open and save panels, the About panel, the items AppKit adds to menus) takes `AppleLanguages` when AppKit starts, and mygo's `SetLanguages` does nothing on macOS (`internal/darwin/locale.go:106-108`). So `main`, before `App.Run`, sets `AppleLanguages` = `[<choice>]` in the process's argument domain: `NSUserDefaults.standardUserDefaults`, its `volatileDomainForName:` `NSArgumentDomain` with the key added, then `setVolatileDomain:forName:`. It calls them through `github.com/ebitengine/purego/objc`, without cgo. Markitai Native's macOS `main` does the same (`localizeTheSystemMenus` in its `Main.swift`). mygo makes `NSApplication` only in `App.Run` (`app.go:105-110`, `internal/darwin/app.go:102-121`), after this. The argument domain lasts for the process and writes nothing. `CFBundleLocalizations` lists English and zh-Hans. Foundation keeps the languages it reads first, so this is the first thing `main` does after choosing. AppKit's items and the About panel then follow the app from the first launch (section 18, claim 4) |

### 10.6 Single instance

Section 2.1. A second launch's file arguments are added as items, and the
window is shown and focused (`Window.Show`, `Window.Focus`). Arguments that
come before the window is made wait with the first launch's.

On Windows mygo binds the lock's socket under `XDG_RUNTIME_DIR`, else the
temporary folder, and only at a path under 100 bytes (section 14).
`singleInstance` (`instance_windows.go`) makes the file arguments absolute,
moves to the data folder, and takes the lock with `XDG_RUNTIME_DIR=.`, which
it restores at once: the socket is then a relative name in the data folder,
which every launch of the user shares, whatever the temporary folder's
length. `PrepareEnvironment` keeps the data folder as the working directory
(section 2.3), where mygo removes the socket as the app quits. With a
`TEMP` of 125 characters, a second launch handed its file to the first and
exited, and the socket was gone after the first quit. When the data folder
cannot be made or entered, or takes no new file (`writable`, `app.go`), the
app takes mygo's lock as it is, under the temporary folder: mygo, unable to
bind its socket in such a folder, reports the lock taken, which would let
every launch run (`single_instance.go:105-109`). Found in the Windows run:
with a data folder that refused new files, a second launch handed its file
to the first and exited, through a socket in the temporary folder; before
the fix it opened a second window.

## 11. Accessibility

DESIGN 12 applies. mygo exposes its drawn UI to VoiceOver, Narrator, NVDA and
Orca (`docs/ui/accessibility.md`).

| DESIGN 12 rule | mygo |
|---|---|
| Labels | Icon-only buttons: `Label` = `Tooltip` = the action's copy key. URL field `Label(convert.url.label)`. Drop zone and drop bar `Role(RoleGroup)` and `Label(a11y.drop_zone)`. Conversions list `Role(RoleList)`, `Label(a11y.conversions)`. History list `Label(nav.history)`. View switch: `SegmentedBase` is a radio group, `Label(detail.view)` (`docs/ui/segmented.md`). Warnings box `Label(a11y.warnings)`, its trigger `Expanded`. Detail bar `ui.Toolbar` `Label(a11y.actions)`. Preview `Label(a11y.preview)`. Markdown text `Label(a11y.markdown_text)`. The pages' sidebar `Label(app.name.mygo)`, the Settings sidebar `Label(nav.settings)`: each is a tree that assistive technology names |
| Row names | Convert rows `Label("<title>, <subtitle>")`. History rows through `ListState.Label`, which a list's rows are read by, with "n of N" (`docs/ui/accessibility.md`, Lists and tables) |
| State changes | `c.Announce(a11y.done)` and `c.Announce(a11y.failed)` in the frame after the outcome, and for notices and Copied (`ui/context.go:499-503`) |
| Headings | Page titles `RoleHeading` level 1, the detail title and Settings group headings level 2 |
| Focus order | DESIGN 12's order, with mygo's stops. The pages' sidebar, then the Settings sidebar, each one Tab stop. On Convert with items, Clear Finished in the column header, then the drop bar's Choose Files…, the URL field, Add and the list. Then the split's divider, then the detail: the view switch, the actions, which the arrow keys reach in the detail bar's `ui.Toolbar`, the warnings disclosure and the page (section 8.3). `ui.Split`'s divider takes the focus between the panes (`ui/split.go:113`), and mygo has no way to leave it out, where Native's split views stop on no divider. A focused sidebar gives assistive technology its chosen item as the active descendant (`ui/sidebar.go:57`), which mygo offers no way to change; a sidebar with no item chosen, as the Settings sidebar on Convert, reads as the tree alone, and an arrow key chooses its first item. Empty Convert focuses Choose Files… (`AutoFocus`). A router page change moves the focus into the page and announces its title (`docs/ui/navigation.md`, The focus and screen readers). Each route sets `r.Title` to the page's name |
| Keyboard | Every action has a shortcut (section 9.2) or a Tab path. Row context menus repeat buttons and shortcuts |
| Contrast and color | Section 5.1. Status glyphs differ by shape |
| Reduced motion | DESIGN 12's rule, realized in section 7: list insertion and removal, and the motion of Native's system controls for the same changes. With the system's Reduce Motion nothing moves but the spinner |
| Text size | `Preferences().TextScale` scales `B` and the type roles (section 5.3). `view.zoom` scales the preview and the Markdown text |
| High contrast | Section 5.1 |
| Alerts | `RoleAlertDialog` named by its title, described by its message, focus on the default button (`ui/feedback.go:186-188`) |

## 12. Errors

DESIGN 9 applies, with `{settings}` = `nav.settings`. `action.open_settings`
goes to `/settings`. The error block, in the detail content:

| Part | mygo |
|---|---|
| Sentence | `Text`, `B`, the mapped sentence |
| Buttons | `ui.Row` gap 8: `ui.Button(detail.retry)`, then the code's second button |
| Details | `detail.details`, `B - 2`, `TextMuted`, then a box (section 5.4) holding the raw `message` in monospace, `Selectable()` |

A panic in a conversion goroutine is recovered and becomes Failed with
`internal_error`.

## 13. New copy keys

Add these to DESIGN 10 and `desktop/strings.json`.

| Key | en | zh-Hans | Where |
|---|---|---|---|
| `app.name.mygo` | Markitai Mygo | Markitai Mygo | Window title, About |
| `menu.about.mygo` | About Markitai Mygo | 关于 Markitai Mygo | macOS app menu |
| `detail.show.linux` | Show in Folder | 在文件夹中显示 | Linux, in place of `detail.show.gnome`: mygo opens whatever file manager the desktop has |
| `preview.unavailable` | The preview can't be shown on this computer. The Markdown tab still works. | 此电脑无法显示预览。Markdown 标签页仍可使用。 | macOS and Linux, when the web view cannot be made |
| `common.copy` | Copy | 复制 | The preview's context menu |
| `preview.failed` | The preview couldn't be shown. The Markdown tab still works. | 无法显示预览。Markdown 标签页仍可使用。 | Every version, when `preview_html` fails (DESIGN 8, section 8.2) |

`detail.show.*` per OS: `detail.show.macos`, `detail.show.windows`,
`detail.show.linux`.

## 14. mygo gaps and how the app handles them

| Gap | Source | Handling |
|---|---|---|
| No middle truncation. Text cuts only at its end (`SingleLine`, `Ellipsis`) | `ui/api_gen.go:3888-3897`, `ui/api_gen.go:5759-5760` | File names draw through `middleText`: a `Box` of the line's height whose `Draw` writes the name cut in its middle, found by a binary search over `Painter.MeasureText` (spike `view.go`). The row's `Label` carries the full name |
| `ui.Split` has no minimum or maximum, only a 40 DIP guard | `ui/api_gen.go:2021-2028`, `docs/ui/split.md` | After `ui.Split`, clamp `listWidth` to 260..420. The stacked breakpoint (section 6.10) keeps the detail at least 340 wide |
| `ui.Divider` inside a `Row` is vertical | `ui/widgets.go:38-46` | Horizontal hairlines inside a row are a `Box` with `Height(1)`, `Grow(1)`, `Background(rule)` |
| `ui.List` rows have no transitions | `docs/ui/transitions.md:167-169` | Convert uses a scroll of keyed rows with transitions. History uses `ui.List` without row motion |
| Transitions still fade under Reduce Motion | `ui/api_gen.go:6114-6117` | `c.DisableAnimations()` each frame (section 7) |
| `ui.Segmented` does not animate | `ui/toggle.go:150-169` | `ui.SegmentedBase` with a matched pill (section 7) |
| `ui.Sidebar` has no footer | `docs/ui/sidebar.md` | Two `ui.Sidebar`s on one `*string`. Each is one Tab stop. A sidebar whose items do not hold the current path shows none chosen (`ui/sidebar.go:177-200`) |
| The page keeps keys from `c.Shortcut` | Section 8.3 | `previewJS` forwards them |
| `ui/markdown`'s `FileImages` reads paths outside its folder | `ui/markdown/image.go:25-43` (spike report) | Not used. The preview is the web view, whose images `preview_html` confines (DESIGN 3.3). No user image is decoded in native UI |
| No destructive alert style, no alert motion | `ui/feedback.go:176-219` | Accepted |
| macOS system panels take the language when AppKit starts | `internal/darwin/locale.go:106-108` | The argument domain, set before `App.Run` (section 10.5) |
| Linux launcher entry passes no files without `fileAssociations` | `cmd/mygo/build.go:609-651` | Accepted. Command-line arguments still add files |
| `Tester.Texts` omits a `TextArea`'s content | `ui/headless.go:471-478` | Tests read the Markdown text with Select All and Copy (spike `TestDropToDone`) |
| `Tester` places web views but shows no page | `ui/headless.go:204-208`, `docs/ui/web-view.md:286-290` | The app loads pages through a small `previewPane` interface whose `View()` is the `ui.NativeWebView` to build. Tests pass a fake that records the HTML, base URL and zoom. Its view embeds a nil `ui.NativeWebView`, because the interface's one method names mygo's internal types (`internal/surface/surface.go:109-113`) and the Tester never calls it (`ui/headless.go:208`). `Tester.WebViewAt` then finds it where the Preview tab shows |
| On macOS `Page.LoadHTML` with a `file:` base loads no file the page refers to, though its documentation says relative URLs resolve against the base, and `Page.LoadFile` of a Markdown file downloads it on macOS 15 | `window.go:1345-1348`, `internal/darwin/window.go:1379-1388` | Serve the result folder at a scheme of the app's (section 8.1) |
| `Transparent` shows the window's backing, not the native UI's background, under the page | `webview.go:44-46`, `internal/darwin/webview.go:15-24` | `WebView.SetBackgroundColor` with the detail's color (section 8.1) |
| `CompileContentRules` from a goroutine waits for the main thread | `content_rules.go:81-100`, `cookies.go:270-290` | Called on the main thread at `WhenReady` (section 8.1) |
| WebKit keeps compiled content rules in the real home, by identifier | `internal/darwin/content_rules.go:40-61` | Accepted, listed in section 2.1 |
| An `Absolute` element paints over its later siblings | `ui/paint.go:360-372` | The segmented control's pill shares the chosen segment's grid cell and is built first (section 6.4) |
| A `ui.SidebarItem`'s glyph takes the accent | `ui/sidebar.go:180-200` | The item draws its glyph and the chosen bar itself in `Draw`, with no glyph given to mygo |
| A `Router` builds the page going away as the next one comes in, so a `ui.Handle` bound in both binds twice | `ui/router.go:535-560` | A page binds its handles only when it is the route shown (`view.go`: `a.leaving`) |
| A percentage of a height inside a `Scroll` does not resolve: the scroll lays its children out in an unbounded height | `ui/layout.go:577-584` | Empty Convert centers its zone with a column outside the scroll (section 6.2) |
| mygo saves a frame of which only a corner shows, restores it so, and makes the first window at its options' size whatever the work area | `window_state.go:113-151`, `window.go:61-206` | The app's `SetWindowStateStore` fits a saved frame to its work area, and the first size to the primary display's (sections 2.1 and 6.1) |
| AppKit gives a menu item's key equivalent the key before the focused view | AppKit's `performKeyEquivalent:` | Remove's ⌘⌫ is cleared while a text field has the focus (section 9.1) |
| mygo turns on WebView2's zoom control, which zooms the page on Ctrl with the wheel apart from `view.zoom` | `internal/windows/webview.go:215` | `previewJS` takes the wheel on Windows and hands it to `Preview.Wheel` (section 8.3) |
| WebView2 gives the page the wheel in the page's own pixels: a notch's `wheelDeltaY` is 120 divided by the page's zoom, in whole pixels (60 at 2.0) | WebView2 156 (found in the Windows run: notches counted as 120 each were lost from 1.1 on) | `app.wheeled` multiplies the delta by the web view's zoom before it counts notches (section 8.3) |
| On Windows, mygo names the zh-Hans UI language `zh-Hans`, under which Windows finds none of its own Simplified Chinese text | `internal/windows/locale.go:112-129` | The app sets the list again with `zh-CN` first (section 10.5) |
| On Windows, `PathDocuments` is `Documents` in the home folder, not the known folder `FOLDERID_Documents`, which OneDrive and policies redirect | `paths.go:114`, `paths.go:140-155` | `documentsFolder` (`documents_windows.go`) asks `SHGetKnownFolderPath` for it, without cgo. When `USERPROFILE` is not the account's profile folder, as in the tests' isolated home, it keeps mygo's path, inside that home |
| On Linux, GTK's own widgets take the dark appearance only when the app sets the theme's source, and setting it inside mygo's change notification never returns | `internal/linux/system.go:440-462`, `modules.go:504-525` | Section 10.4 |
| On Linux, GTK's own words follow gettext's environment, not `App.SetLocale` | `internal/linux/locale.go:57-70` | `LANGUAGE`, set before GTK starts (section 10.5) |
| On Linux, GTK 3's in-process file chooser opens at 1203 × 950 whatever the screen, lists the working directory, the app's data folder, among its places, and keeps its size in the user's dconf | `internal/linux/dialog.go:85` (`gtk_file_chooser_native_new` uses it when no portal answers) | `GTK_USE_PORTAL=1` when the portal has a file chooser: the portal's runs outside the app, at the desktop's size, with the desktop's places and style (found in the Linux run: 875 × 600, dark, no `dev.markitai.mygo`). Without one the in-process chooser remains, and `toolkitReady` gives each `GtkFileChooserDialog`, as it is realized, a largest size of its screen's work area less 48 on each side, with a `realize` emission hook (`toolkit_linux.go`; section 10.5). It still lists the data folder among its places |
| On Linux, GTK told to use the portal by `GTK_USE_PORTAL=1` waits for a portal file chooser that may not exist: no dialog shows, and the app aborts as it quits | GTK 3's `gtkfilechoosernativeportal.c` (`gtk_file_chooser_native_portal_hide: assertion failed`) | The variable is set only when the portal answers for its `FileChooser`, within 3 s (section 10.5). A dialog that fails is a cancel (section 6.7) |
| On Linux, a `DefaultPath` that is a bare name starts the chooser in the working directory, the data folder | `internal/linux/dialog.go:37-48` (`filepath.Dir` of a name is `.`) | Change… and Save As… start at absolute paths in the output folder (section 10.2) |
| On Linux, mygo registers the app with the portal by its executable's name, `markitai-mygo`, unless the desktop launched it from an entry, and the portal knows no such app | `internal/linux/system.go:543-551` (`desktopEntryID`), `internal/linux/portal.go:121-127` | Before `App.Run`, `nameToPortal` sets `BAMF_DESKTOP_FILE_HINT` to the installed `dev.markitai.mygo.desktop`, which mygo reads first (section 2.3). Found in the Linux run: the portal's `Registry.Register` got `dev.markitai.mygo`, and `markitai-mygo` before the fix |
| On Linux, mygo's `Shell` starts programs in the app's environment, with the variables set for its toolkit, `WEBKIT_DISABLE_DMABUF_RENDERER=1` among them, which mygo sets itself | `internal/linux/system.go:179-185`, `internal/linux/app.go:112-116` | The app launches them itself with the user's values (section 10.5) |
| On Linux, WebKitGTK gives the page the mouse's back and forward buttons, GDK's 8 and 9, both as button 0 | WebKitGTK 2.54 (a probe) | A GTK emission hook on the web view's `button-release-event` runs Back for button 8 (section 8.3) |
| mygo gives the keyboard back to the native UI only as it hides a web view that has it: focusing a native control leaves the keys with the page | `internal/darwin/webview.go:90-93`, `internal/linux/webview.go:195-198`, `internal/windows/webview_embed.go:278-281` | Add Web Address and Search History leave the web view out of one frame first (section 8.3) |
| mygo allows the page's history back and forward without `OnWillNavigate`. A link dropped on the page navigates it, which the navigation rule sends to the browser, and dropped files reach only `Window.OnFileDrop` (found in the Windows review) | `window.go:1886` (`nav.Unasked`), `docs/architecture.md:833-849` | `previewJS` takes the mouse's buttons and drops (section 8.3) |
| One `ListState` shown by two `ui.List`s in a frame panics, and a router builds the page going away with the new one | `ui/list.go:474`, `ui/router.go:535-560` | History's page going away builds no list while the window is not stacked (section 6.10). Convert's list is no `ui.List` |
| `ui.Split`'s divider is a Tab stop | `ui/split.go:113` | Accepted (section 11) |
| A focused `ui.Sidebar` gives assistive technology its chosen item, and none when no item is chosen | `ui/sidebar.go:57` | Accepted. Each sidebar has a `Label` (section 11) |
| mygo's Simplified Chinese app menu says 隐藏Markitai Mygo and 退出Markitai Mygo, with no space, beside 关于 Markitai Mygo and macOS's own menus | `locales/zh-Hans/zh-Hans.json:105`, `locales/zh-Hans/zh-Hans.json:107` | `internal/l10n/mygo/` changes `mygo.menuHide` and `mygo.menuQuitApp` in an app catalog that `l10n.ChangeMygoText` registers (`i18n/catalog.go:411-416`) |
| On macOS, `App.ShowAboutPanel` leaves out an option given empty, so AppKit adds the build number after the version: Version 2.3.2 (2.3.2) | `internal/darwin/app.go:452-469` | `showAbout` (`about_darwin.go`) calls `orderFrontStandardAboutPanelWithOptions:` itself with an empty `Version`, the core's version as credits and the copyright, as Native's panel shows |
| mygo's `MinWidth` and `MinHeight` are the content's on Linux, and the window's, title bar included, on macOS and Windows; on Windows `Window.Bounds` and `ContentBounds` round each size to whole DIPs, so their difference can be short of the frame | `internal/linux/window.go:390-404`, `internal/darwin/window.go:772-774`, `internal/windows/window.go:477-482`, `internal/windows/window.go:863` | `contentMinimum` adds the frame on macOS and Windows (section 6.10). Found in this round's macOS run: a titled 640 × 480 frame held 640 × 448 of content. On Windows, at 192 dpi, 1254 × 889 pixels before the fix, and 1280 × 959 with the frame taken from the rounded sizes |
| GTK writes `gvfs-metadata/` and `recently-used.xbel` under `$XDG_DATA_HOME` once a file dialog has shown | GTK 3's `GtkFileChooserNative` | Accepted, listed in section 2.1 |
| mygo build names the Linux desktop entry, the packages' paths and `install.sh`'s copy after the slug, which cannot hold a dot, and writes no `Comment[zh_CN]`, `Keywords`, `StartupNotify` or `StartupWMClass` | `cmd/mygo/config.go:304`, `cmd/mygo/config.go:348`, `cmd/mygo/build.go:609-651` | `linux_entry.py` finishes every copy after `mygo build` (section 15) |
| mygo build installs the Linux icon at 256 and 512 pixels only | `cmd/mygo/deb.go:248-262` | `linux_entry.py` adds Native's icons at 16, 32, 48, 64 and 128 pixels to the Debian package and the `PKGBUILD`, as Native installs 16 to 512 (section 15) |
| `Tester.DropFiles` drops only files | `ui/headless.go:674-681` | The link drop calls `app.dropped` with `transfer.URLData` (`convert_test.go`) |
| `Tester` has no automation of the real window | `plugins/automation` | The `e2e` build links mygo's automation plugin; `internal/e2e` drives it (section 16) |
| On Linux, the first web view of a shown window that is not active yet takes the keyboard from the native UI for good: the window gets no key until a click on the content | `internal/linux/webview.go:136-166` (`makeLayers` moves the content's widget into an overlay and gives the focus back only when the widget has it, which it does not before the window is active; removing the widget clears the window's focus), `window.go:575-577` (`NewWindow` shows the window unless `Hidden`), `internal/linux/window.go:740-758` (`Focus` only presents the window), `internal/linux/surface.go:926-931` (only a press gives the content the focus again) | The window is made with `Hidden`, gets its web view, and is shown after it (section 2.3). Found in the Linux runs: a probe typed through the compositor reached a text field without a web view or with this order, and nothing reached it with the web view made after the window showed |
| On Linux, a web view placed again after it was hidden keeps the size it had before: a taller frame shows GTK's window color in a band under the page | `internal/linux/webview.go:183-189` (`PlaceWebViews` sets the size request, then shows the hidden widget). The cause inside GTK 3 is not traced: in the probe, a change of size while the widget shows takes effect, and one made while it was hidden does not | On Linux, the preview's first frame after it comes back has a bottom margin of 1 DIP and asks for another frame, whose new size GTK applies (`preview.go`, section 8.1). Found in the Linux run (a band of 59 DIP under the page after switching items); a probe reproduced it, and with the margin it filled its frame (`.local` evidence, `probe-web/`) |
| On Linux, a transparent page in the dark color scheme paints `#1e1e1e`, not the web view's color | WebKitGTK; `internal/linux/window.go:840-844` sets the web view's color | The document's head gets the detail's background (section 8.1) |
| On Linux, mygo keeps web data by the display name and WebKitGTK's program name, not under `PathUserData` | `internal/linux/content_rules.go:84-89`, `internal/linux/cookies.go:92-103` | The app names itself and the program `dev.markitai.mygo` before `App.Run` (section 2), so all of it is under that name (section 2.1) |
| On Windows, when the temporary folder's path is long, a second launch opens a second window instead of handing its files to the first: the lock's socket goes in the temporary folder only while its path stays under 100 bytes, else in `/tmp`, which Windows lacks, and a lock that cannot listen lets every instance run | `single_instance.go:176-184` (the path), `single_instance.go:105-109` (any error but address-in-use returns true) | `singleInstance` takes the lock in the data folder by a relative name, and under the temporary folder only when the data folder takes no file (section 10.6). The e2e driver still gives each run a short temporary folder of its own (section 16) |

Upstream candidates. Gaps above that mygo itself could close, recorded here
only; nothing is filed from this repository:

- Give the keyboard back to the native UI when it focuses a control while a
  web view has it, not only when it hides the web view
  (`internal/darwin/webview.go:90-93`, `internal/linux/webview.go:195-198`,
  `internal/windows/webview_embed.go:278-281`).
- Ask `OnWillNavigate` about the history's back and forward too
  (`window.go:1886`).
- Report a single-instance lock it could not listen on as not taken, or fall
  back to another folder, instead of returning true for any error but
  address-in-use (`single_instance.go:105-109`).
- On Linux, start a dialog whose `DefaultPath` is a bare name in a folder
  other than the working directory (`internal/linux/dialog.go:37-48`).
- On Linux, try the program name's `.desktop` entry for the portal's app ID
  before the executable's name (`internal/linux/system.go:543-551`).
- On Linux, start programs from `Shell` with the user's environment, without
  `WEBKIT_DISABLE_DMABUF_RENDERER` that mygo set for itself
  (`internal/linux/system.go:179-185`, `internal/linux/app.go:112-116`).
- Make `MinWidth` and `MinHeight` the content's on every OS, as on Linux
  (`internal/darwin/window.go:772-774`, `internal/windows/window.go:477-482`),
  and on Windows round the frame, not each size, to DIPs
  (`internal/windows/window.go:863`).
- On Windows, let an app turn WebView2's own zoom off, or give it the wheel
  of Ctrl with the wheel in notches (`internal/windows/webview.go:215`).
- Let an app leave `ui.Split`'s divider out of the Tab order
  (`ui/split.go:113`), and choose what a focused `ui.Sidebar` with no item
  chosen gives assistive technology (`ui/sidebar.go:57`).
- Put a space before `{app}` in the zh-Hans `menuHide` and `menuQuitApp`, as
  macOS does (`locales/zh-Hans/zh-Hans.json:105`, `locales/zh-Hans/zh-Hans.json:107`).
- On macOS, pass an option given empty to AppKit's About panel, so an app can
  leave out the build number (`internal/darwin/app.go:452-469`).

## 15. Build

`python3.13 desktop/mygo/build.py [--check] [--no-core] [--platform GOOS/GOARCH]
[--e2e]` on macOS, Windows and Linux, Python 3.13 or newer, standard library
only:

1. Build the core: `cargo rustc -p markitai-app --release --locked --lib
   --crate-type cdylib`, honoring `CARGO_TARGET_DIR`. On macOS with
   `MACOSX_DEPLOYMENT_TARGET=13.0`, `SDKROOT` from `xcrun` when unset, and
   `-C link-arg=-Wl,-install_name,@rpath/libmarkitai_app.dylib`: without it the
   library's install name is its absolute path under `target/`, which would
   ship a build machine's home path. On Windows with `-Ctarget-feature=+crt-static`
   added to the flags, as Markitai Native's Windows build does, so the app
   needs no Visual C++ redistributable, in `target/mygo-windows` so other builds
   keep their flags. `--no-core` reuses the library built before.
2. Stage it in `desktop/mygo/resources/<goos>-<goarch>/` (ignored).
3. Copy `desktop/strings.json` to `internal/l10n/strings.json` and
   `desktop/icon.png` to `internal/brand/icon.png`. Check that `mygo.json`'s
   `version` equals the workspace version.
4. `go vet ./...`.
5. `go test -count=1 ./...` with `MARKITAI_APP_LIB` set to the staged
   library and an isolated environment: `HOME`, `USERPROFILE`,
   `MARKITAI_HOME`, `XDG_DATA_HOME`, `XDG_CONFIG_HOME`, `XDG_CACHE_HOME`,
   `XDG_STATE_HOME`, `APPDATA` and `LOCALAPPDATA` in a temporary folder,
   `TEMP` and `TMP` in a short folder there, `TERM=dumb`, `CGO_ENABLED=0`. `GOMODCACHE`, `GOCACHE` and `GOPATH` keep
   their real values so the module cache is not downloaded again.
6. `CGO_ENABLED=0 go tool mygo build -platform <goos>/<goarch> -skip-installer`,
   on macOS also `-skip-dmg`, into `desktop/mygo/build/<goos>-<goarch>/`.
7. On Linux, `linux_entry.py`: in the desktop file beside the app, the
   `.tar.gz`, `install.sh`, the Debian package and the `PKGBUILD`, the entry
   gets Native's `Comment[zh_CN]`, `Keywords` and `StartupNotify` and
   `StartupWMClass=dev.markitai.mygo`, and installs as
   `dev.markitai.mygo.desktop`. The Debian package's description gets
   Native's long description, the `PKGBUILD` the workspace's license and the
   archive's new SHA-256. The Debian package and the `PKGBUILD` also get
   Native's icons at 16, 32, 48, 64 and 128 pixels beside mygo's 256 and 512,
   so the package installs 16 to 512 as Native's does, with the Debian
   package's `md5sums` to match. The archives it writes again keep their
   members' times and the gzip header's, so the same input gives the same
   bytes (found on the Linux machine: two runs 2 s apart gave equal files,
   and different ones before). It fails when `mygo build` writes any of them
   differently, rather than guess.

`--check` runs steps 1, 2, 4 and 5 and `desktop/scripts/strings.py --check`,
and in place of step 3's copy fails when a copy differs. It builds no app and
never writes a tracked file. Without `--check`, `build.py` runs steps 1 to 3,
6 and 7; run `--check` for the vet and the tests. `--e2e` builds with
`GOFLAGS=-tags=e2e` and `MYGO_INSPECTOR=1` into `desktop/mygo/build/e2e/`:
`e2e.go` links mygo's automation plugin, which builds without the inspector
leave out. Never ship that build. The e2e build keeps the identifier
`dev.markitai.mygo`: its only data outside `HOME` is WebKit's content-rule
list of section 2.1, the same rules an installed app compiles.

Hosted verification: a `windows` job in `.github/workflows/desktop-mygo.yml`
on `windows-2025`: checkout, Python 3.13, `actions/setup-go` with
`go-version-file: desktop/mygo/go.mod`, the repository's Rust toolchain with
its dependencies cached for `target/mygo-windows`, `build.py --check`, then
`build.py --no-core` and `build.py --no-core --e2e`, the SHA-256 of the
executables and the DLL with the commit and the WebView2 Runtime version, then
the e2e driver of section 16 on the runner's desktop. The evidence uploads as
`markitai-mygo-windows-evidence` and `desktop/mygo/build/windows-amd64/` as
`markitai-mygo-windows-amd64`. The workflow runs on pushes, on any branch, that
change `desktop/mygo/`, `desktop/strings.json`, `desktop/icon.png`,
`desktop/scripts/strings.py`, the core's crates, `vendor/`, `Cargo.toml`,
`Cargo.lock` or the workflow, one run per ref at a time. The user guide is
`docs/desktop-mygo.md`.

## 16. Test plan

Headless view tests (`ui.NewTester`) over the real core library with
repository fixtures, in English and zh-Hans where copy is asserted. Desk
(dialogs, Shell) and preview are fakes that record calls. Each test sets the
isolated environment of section 15 again with `t.Setenv`.

| Test | Flow it covers |
|---|---|
| Empty Convert | Every DESIGN 6.2 string, Choose Files… focused, drop zone and URL field named |
| URL field | Invalid text shows the error until the text changes. `example.com` adds `https://example.com` |
| Drop to Done | A PDF fixture dropped converts. Row name, announcement, preview HTML loaded with the right base URL and theme, Markdown text equal to the result, Copy, Show |
| Drop a URL | A `transfer.URIList` drop adds a URL item. A folder drop shows `notice.folder` |
| Paste | Clipboard files add items. Clipboard text with one URL adds it. Paste in the URL field types text |
| Two at a time | Three files: two Converting, one Queued, then all Done |
| Duplicate | Adding a Queued source shows `notice.duplicate` |
| Failure | `ocr/fixtures/english.png` with AI and OCR off fails with `image_only`: sentence, Retry, Open Settings goes to Settings, and the claimed folder is gone. A missing file fails with `not_found`. `TestFailureTitles`: Vision with every reader stopped and OCR with damaged models say `error.text_recognition_error`, a broken document `error.conversion_error`, each without the other |
| Retry | Retry captures the current settings and queues |
| Remove and Clear Finished | Converting rows refuse removal. Selection moves to the next row |
| History | Done items appear newest first. Search filters, no-match text. Remove keeps files. Clear asks, Cancel keeps, Clear empties. 501 additions keep 500. A corrupt file is renamed and reported once. A missing Markdown file shows `detail.missing.*` |
| Settings | Switches write `settings.json`. Desc and Alt are disabled without AI. Change refuses an unwritable folder, and taking a folder clears that notice. Configuration choice, Open and Create, scrolled to in a window too short for the page |
| Quit | With an item Converting, the close path asks. Cancel keeps it. The alert follows the count and closes by itself when the conversion ends. Quit deletes the claimed folder |
| Shortcuts | Each section 9.2 action, from the native UI and through `Preview.Press`. `TestPreviewScript` checks which actions the script forwards on each OS, Quit on Linux only, and the wheel on Windows only. `TestFocusFromThePreview`: Add Web Address and Search History from the page leave the web view out of one frame, focus their field, and show the page again (section 8.3) |
| Stacked layout | At 800 wide: list fills, Enter pushes the detail, Esc and Back return with the selection kept, widening replaces the page. `TestWideningFromAStackedItem`: widening while a History item shows, is pushed or is popped keeps the app running with the list and the item (it panicked before). `TestStackedClickOnTheChosenRow`: a click on History's chosen row pushes its page |
| Motion | `TestMotion`: row removal slides over 150 ms, and is at once with Reduce Motion. `TestReduceMotionSettles`: with Reduce Motion no frame is pending after any change except the spinner. `TestMotionKeptToTheSpec`: a sidebar page change and a selection leave no frame pending, and a stacked push does (section 7) |
| Accessibility | Names of section 11 found with `Tester.Find`. Announcements with `Tester.Announcements` |
| Copy | Every visible string comes from `strings.json` in both languages: a test walks every screen and checks `Tester.Texts` against the catalog's values, allowing file names, paths, numbers and the core's text |
| Images | The preview's base URL is the result folder, at a scheme the app serves on macOS (section 8.1). Remote images are not loaded (the HTML from `preview_html` holds `img-remote`). The web view is filled with the detail's background |
| Library | `internal/core/ship_test.go` reads the staged core: on macOS its install name is `@rpath/libmarkitai_app.dylib` and it links only `/usr/lib` and `/System/Library`; on Windows it imports no `vcruntime`, `msvcp` or `api-ms-win-crt-` DLL; on Linux it needs only `lib*` sonames |
| Fixes of the first review | `fixes_test.go`: paste takes only `http` and `https` text, a Chromium link drop (`text/x-moz-url` in UTF-16), a notice stays on its page, a failed preview shows `preview.failed` and renders again, ⌘⌫'s menu key leaves text fields, the menu bar's order, Settings reads the configuration file once per visit. `windowstate_test.go`: frames fitted to work areas. `layout_test.go`: empty Convert fits and centers at 640 × 480 and 1200 × 780. `history_test.go`: History's Markdown leaves the file's front matter out, also from a file saved again with CRLF or a byte-order mark, and keeps a leading thematic break. A cleanup in `startWith` waits for conversions still running, so Windows can remove the test's folder |
| Fixes of the second review | `fixes_test.go`: `TestDropOnThePreview`, links and files the page hands over add items as a drop on the content area does, and other links nothing; `TestSaveAsAndFolderDialogs`, Change… and Save As… start in the output folder or its nearest folder that exists, a name without an extension gets `.md` and replaces nothing, and a failure's notice shows on the page Save As… began on. `layout_test.go`: `TestEmptyConvertFits`, the zone and Add inside a 640 × 480 content without scrolling. `windowstate_test.go`: `TestKeepFrame`. `internal/l10n`: `TestMygoTextChanged`, mygo's zh-Hans Hide and Quit items. `preview_script_test.go` runs `previewJS` in Node over a stand-in window, and skips without `node`: `TestPreviewKeysOnAnyLayout` (AZERTY and others), `TestPreviewMouseButtons`, `TestPreviewDrops`, `TestPreviewDocumentFocusable`, `TestPreviewWheelZoom` (on Windows only, Ctrl with the wheel goes to `Preview.Wheel` with its `wheelDeltaY`). `shortcuts_test.go`: `TestWheelZoom`, with the page's delta divided by its zoom as WebView2 gives it, a notch steps once at any zoom, half notches add up, and the first notch the other way steps back. On Linux `shell_linux_test.go`: `TestLaunchAsUser`, a program the app opens gets the user's values of the toolkit's variables. On Windows `lang_windows_test.go`: `TestDialogLanguages` |
| Design shots | `design_test.go`, only with `MARKITAI_MYGO_DESIGN_SHOTS` naming a folder: every screen and state of section 6, light and dark, English and zh-Hans, as PNGs from the `Tester` for review |

The real window. `internal/e2e` (build tag `e2e`) drives the `--e2e` build
through mygo's automation socket (`MYGO_AUTOMATION`). Each run of
`MARKITAI_MYGO_E2E_RUNS` (default `en-light,zh-Hans-dark,en-dark,zh-Hans-light`)
gets its own `HOME`, XDG folders, `USERPROFILE`, `APPDATA`, `LOCALAPPDATA` and
`MARKITAI_HOME` under `MARKITAI_MYGO_E2E_OUT/<run>/`, a short folder of its
own under the system's temporary folder, for the automation socket, whose
path must stay under about 100 bytes, and as `TEMP`,
where the single-instance lock falls back when the data folder takes no file
(section 10.6), `MYGO_LOCALE` and
`MARKITAI_MYGO_E2E_THEME` (read by `e2e.go`). The run launches the app with a PDF
fixture whose images are confined and a folder, waits for Done and
`notice.folder`, opens the Markdown
tab, starts a second instance with two more files (one Markdown file with a
table, task lists, a footnote, a remote image and raw HTML), adds a page from
a local server, searches History, turns OCR on and reads `settings.json`,
starts a page the server never answers, then quits, cancels, quits again and
checks that the converting item's folder is gone. `report.json` lists what the
app wrote outside its folders of section 2.1, the output folder and the
markitai home, as the shared files of section 2.1, and the run fails only if
a path names another version, or on Linux holds `Markitai Mygo/` or
`markitai-mygo/`, the names mygo and WebKitGTK use when the app does not name
itself (section 2.1). Quitting
is asked through the app menu on macOS, Ctrl+Q on Linux and `WM_CLOSE`
(`taskkill` without `/F`) on Windows. Each step keeps the window's own capture
and, with `MARKITAI_MYGO_E2E_WINDOW_SHOT` naming a program that takes a process
ID and a PNG path, the system's (`internal/e2e/windowshot.ps1` on Windows),
which shows the web view. With `MARKITAI_MYGO_E2E_PAGE_KEY` naming a program
that takes a process ID and `zoom-in` and sends that window the platform's
Zoom In keys as the keyboard does (`internal/e2e/pagekey.ps1` on Windows), it
gives the preview page the keyboard with the automation's `focus`, sends the
keys and checks that `view.zoom` rises exactly one step: only the page's
injected script can act on them there (section 8.3), and two steps would mean
the action ran twice. Without such a program, as on macOS, where posting keys
to another process needs the Accessibility permission, the e2e build itself
dispatches that keydown in the page when the driver writes `zoom-in` to the
file `MARKITAI_MYGO_E2E_PAGE_EVENT` names (`e2e.go`, `Page.Eval`). That checks
the script and its `mygo.call`, not the system's delivery of the key, and
`report.json` says which way ran. `report.json` holds the first-frame and
window-visible times, the time from the first Done to its preview, the memory
after the first conversion, the second instance's exit time, the rows and the files written.

## 17. Checks before calling a platform done

| Check | How |
|---|---|
| Build | `build.py --check` and `build.py` pass on the platform. Windows passes in the hosted job |
| States | DESIGN 16's States, Folders and History checks, run in the built app with an isolated `HOME` and `MARKITAI_HOME` |
| Preview | In the built app: a fixture with tables, task lists, footnotes and a relative image shows them, light and dark. A link opens the browser. A remote image stays a placeholder. No request leaves the machine. The context menu holds only Copy |
| Keys in the preview | With the page focused (clicked), every forwarded action of section 8.3 runs exactly once, on a US layout and on AZERTY, and ⌘C or Ctrl+C copies the page's selection. Add Web Address and Search History then take typing in their fields. Tab stops at a page with no link, and the arrows scroll it |
| The preview is the app's | Over the page, the mouse's back button returns from a stacked item to the list, and the forward button changes nothing. A link or a file dropped on the page is added once, and the page opens nothing. On Windows, Ctrl with one wheel notch zooms one step at any zoom, and the first notch the other way steps back |
| Window | At the minimum size the content area is 640 × 480 in the system's capture, and the empty Convert page shows its whole zone |
| Dialogs | Save As… starts in the output folder with the result's name and the Markdown filter, and a name without an extension saves `<name>.md`; Change… starts in the output folder. On Linux, with a portal without a file chooser and with none, each dialog shows within the screen, and quitting is clean. A program the app opens on Linux gets the user's environment |
| Quit | Every quit path of section 10.3 asks while converting. macOS: closing the window keeps conversions running, the Dock icon brings it back |
| Coexistence | With Markitai Web and Markitai Native running, Markitai Mygo starts its own instance, writes only its folders of section 2.1, and its History stays its own |
| Language | zh-Hans and English runs show no other language in mygo-drawn UI. macOS, with zh-Hant first in the system's languages: the first launch's open panel and About panel are in zh-Hans |
| Accessibility | Keyboard-only run through every flow. VoiceOver, Narrator and Orca read row names, announcements and the alert |
| Motion | Reduce Motion on: no movement and no fades except the spinner |
| Copy | DESIGN 16's Copy check |

## 18. Unverified

The claims this section listed before the first builds, and what the runs of
2026-11-04 and 2026-11-05 found (evidence in the ignored `.local/desktop-mygo-r1/`,
for the first review's fixes `.local/desktop-mygo-fix-r1/`, and for the
second's `.local/desktop-mygo-review-r2/` and `.local/desktop-mygo-fix-r2/`).
Where a run disproved a claim, the sections above now say what the app does.

| Claim | macOS | Linux | Windows |
|---|---|---|---|
| 1. Images inside the result folder load from a document that `LoadHTML` serves at a `file:` base URL (section 8.1) | False: `loadHTMLString:baseURL:` grants no read access (`naturalWidth` 0). The app serves the folder at a scheme of its own (sections 8.1 and 14) | True: the e2e runs show the image | True: on a 192-dpi display the real-window runs show the PDF fixture's image. mygo serves the document at its `file:` base URL through WebView2's resource requests (`internal/windows/webview.go:667-684`, `internal/windows/webview.go:1191-1194`). The hosted runner's runs, at 96 dpi, show it too |
| 2. A host-injected `PreloadScript` runs in a page whose CSP is `default-src 'none'`, and `mygo.call` reaches a bound method from it (section 8.3) | True: a keydown of Cmd+= that the e2e build dispatches in the page raises `view.zoom` one step. The system's delivery of the key is not checked (claim 3) | True with system keys: Ctrl+= from the compositor (Hyprland `send_shortcut`) raises `view.zoom` one step, once the window shows after its web view (section 14) | True with system keys: Ctrl+= sent with `keybd_event` to the preview, once the automation gave it the focus, raises `view.zoom` one step (`internal/e2e/pagekey.ps1`, 192 dpi). The hosted runner's runs agree |
| 3. On macOS, a key the preview script prevents does not also reach the menu bar's key equivalents (section 8.3) | True, two ways. First, a test library loaded into the app made `NSEvent`s from HID-source `CGEvent`s and posted them to `NSApp`'s own queue, so they took AppKit's whole path (key window, web view, menu key equivalents) without leaving the process and without the Accessibility permission. With the page focused by a click, ⌘= raised `view.zoom` one step, ⇧⌘M switched the view once, ⌘- and ⌘0 acted once, and no menu item took them. Second, in the second review, through the WindowServer: `CGEventPostToPid` to the app's own process ID only, checked first by its executable's path. With the page holding the keyboard, ⌘= raised `view.zoom` to 1.1, ⌘- and ⌘0 brought it back to 1, ⇧⌘= rose one step, and ⇧⌘M switched the view once; the trace showed the window taking each, and no menu item. ⌘1, ⌘2, ⌘,, ⌘L and ⌘F from the page each ran once | Not applicable | Not applicable |
| 4. AppKit's panels and About panel follow `AppleLanguages` set in the argument domain from Go, through purego, before `App.Run` (section 10.5) | True for the app's choice: set first in `main`, with `MYGO_LOCALE=zh-Hans` on a system whose first language is English, the items AppKit adds (Close All, AutoFill, Start Dictation, Emoji & Symbols) and the About panel are in zh-Hans; set after `App.PreferredLanguages`, as before this fix, they stayed English. A system whose first language is zh-Hant is untried: that needs changing the system's language. `TestAppleLanguages` reads the argument domain back | Not applicable | Not applicable |
| 5. Dropping a link from a browser delivers `transfer.URIList` (section 10.1) | Not claimed | False for Chromium: it offers `text/x-moz-url`, HTML and text. With the drop taking `text/x-moz-url`, a link dragged from Chromium (isolated profile, a local page, a virtual pointer) highlights the drop zone and adds a URL item | True for Edge: a dragged link adds a URL item, in the Windows review and again with the first fixes' drop (Edge 156, an isolated profile, a local page, a pointer moved with `SetCursorPos`) |
| 6. The page's mouse buttons and drops are the app's (section 8.3) | Buttons true: with in-process events over the page of a stacked item, the back button returned to the list and the forward button changed nothing; before the fix neither did anything. Drops on the page not tried in the real window | True: with the GTK hook, GDK's button 8 over the page returned to the list and 9 changed nothing (a virtual pointer); a file from a GTK drag source and a link from Chromium dropped on the page each added one item | True: with `mouse_event` over the page (192 dpi, the window in front), the back button (XBUTTON1) changed nothing in the wide layout and returned from a stacked item to the list, and the forward button changed nothing; before the fix the back button showed the earlier item's output folder as a file listing. A file dropped on the page from an OLE drag source (`CF_HDROP`) was added once, where before the fix it was dropped silently, and a link (`UniformResourceLocatorW`) added one URL item and opened no browser |
| 7. Add Web Address and Search History from the page take the keyboard (section 8.3) | True: ⌘L and ⌘F, then typing, reached the URL and search fields; before the fix the typing was lost | True: Ctrl+L and Ctrl+F from the compositor, then typing | True: Ctrl+L and Ctrl+F from the page (`keybd_event`, the window in front), then typing, reached the URL and search fields, also with a History item's page shown; before the fix the typing after Ctrl+L was lost, and after Ctrl+F from a History item's page |
| 8. The second review's window and Windows rules: the content's minimum (section 6.10), one zoom step per wheel notch (section 8.3), the dialogs' starts and filter (section 10.2), the lock's fallback (section 10.6) | Minimum true: the least frame mygo set held 640 × 448 of content before the fix and 640 × 480 with it (AppKit's own sizes, read in the process), and at that size the empty Convert page showed its whole zone. The rest not applicable | Dialogs true (section 10.2) | True: at its least size the window held 1280 × 961 pixels of content at 192 dpi (1254 × 889 before the fix), and the empty Convert page showed its whole zone. Each notch of Ctrl with the wheel over the page stepped once, from 1.0 up to 3.0 and down to 0.5, half notches added up, and the first notch back stepped back. Save As… started in the output folder with the item's name and the Markdown filter, and `123` saved `123.md`; Change… started in the output folder; under zh-Hans the dialogs' own words were Simplified Chinese. With a data folder that takes no file, a second launch handed its files to the first and quit, where before the fix it opened a second window |
