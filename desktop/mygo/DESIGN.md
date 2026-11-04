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
| Linux | binary and Debian package `markitai-mygo`, x86-64. GTK 3 and WebKitGTK 4.1 (`libgtk-3-0`, `libwebkit2gtk-4.1-0`, `cmd/mygo/deb.go:74-81`) |
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
  "icon": "icon.png",
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
  "linux": {"comment": "Convert documents and URLs to Markdown", "categories": ["Office", "Utility"], "command": "markitai-mygo"}
}
```

`icon.png` is a copy of `desktop/icon.png`. `linux.webview` stays unset,
which means true: a `ui.WebView` needs WebKitGTK (`cmd/mygo/deb.go:37-41`).
`macos.infoPlist` overrides the keys mygo writes (`cmd/mygo/bundle.go:307-309`).
`build.py` fails when `version` differs from the workspace version.

### 2.1 Coexistence with the other versions

The three versions run side by side on one machine. Markitai Web is
`dev.markitai.web`, Markitai Native is `dev.markitai.native`.

| Concern | Markitai Mygo |
|---|---|
| Data folder (`data_dir`, working directory) | macOS `~/Library/Application Support/dev.markitai.mygo`. Windows `%LOCALAPPDATA%\Markitai Mygo`. Linux `$XDG_DATA_HOME/dev.markitai.mygo` (`~/.local/share/...`) |
| History (DESIGN 4.4 shape) | `<data folder>/history.json` |
| Settings (DESIGN 4.5 keys except `window.*`) | macOS and Windows `<data folder>/settings.json`. Linux `$XDG_CONFIG_HOME/dev.markitai.mygo/settings.json`. Writes follow DESIGN 4.4's temporary-file-and-rename rule |
| Window frame | mygo's `WindowOptions.StateKey` `"main"`, saved in `window-state.json` under `PathUserData` (`window.go:192-199`, `window_state.go:24`). It is restored only where it shows on a connected display (`window_state.go:113-116`) |
| mygo's own folders | Before `App.Run`, `App.SetPath(PathUserData, <data folder>)` and `App.SetPath(PathCache, <cache folder>)` (`paths.go:72-80`). Without this, mygo names them after the display name (`paths.go:90-95`). The cache folder is macOS `~/Library/Caches/dev.markitai.mygo`, Windows `%LOCALAPPDATA%\Markitai Mygo\Cache`, Linux `$XDG_CACHE_HOME/dev.markitai.mygo` |
| WebView2 data (Windows) | `%LOCALAPPDATA%\Markitai Mygo\WebView2`, which mygo derives from the display name (`internal/windows/app.go:185-189`) |
| Single instance | `App.RequestSingleInstanceLock()`. A packaged app's key is `identifier + "\n" + name` (`single_instance.go:164-172`), so the three versions never share a lock |
| macOS defaults domain | `dev.markitai.mygo`, used only for `AppleLanguages` (section 10.5) |
| Output folders | The shared default output folder (DESIGN 4.3) is safe to share: `claim_output_folder` creates each conversion's folder with an atomic `create_dir` (DESIGN 3.3), so no two apps get one folder |

### 2.2 Code structure and data shapes

The spike's packages stay and grow:

| Package | Holds |
|---|---|
| `internal/core` | The purego binding of DESIGN 3.1 (spike `core.go`) |
| `internal/l10n` | `strings.json` lookup, plurals, formats of DESIGN 4.2 |
| `internal/model` | `Item`, `State` (`Queued`, `Converting`, `Done`, `Failed`), `Queue` (DESIGN 4.1, spike `queue.go`), and new `History` (DESIGN 4.4: add, remove, clear, 500 limit, atomic write, unreadable rename) and `Settings` (DESIGN 4.5) with their files |
| `main` | `main.go` (launch, section 2.3), `app.go` (state and actions), one file per screen, `preview.go` (section 8), `menu.go` (section 9.1) |

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

1. Compute the folders of section 2.1. Call `App.SetPath` for them.
2. `App.RequestSingleInstanceLock()`. A second launch hands its arguments to the
   first and exits (`single_instance.go:50-73`).
3. Choose the language. On macOS, put it in the argument domain now, before
   AppKit starts (section 10.5).
4. `core.Load(core.Find(resources))`. An ABI mismatch shows only the fatal
   screen (section 6.9).
5. `PrepareEnvironment(data folder)`, before `App.Run`. It blocks up to 10 s
   (DESIGN 3.3).
6. Load settings and history.
7. `mygo.BindAs("PreviewKeys", &previewKeys{})` (section 8.3, `ipc.go:105-110`).
8. `App.WhenReady`: make the window, its web view, and on macOS the menu bar.
   Add the files from `os.Args[1:]` (section 10.1).

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
| About | The last group of Settings. On macOS also the app menu's `menu.about.mygo`, which shows the standard About panel (`App.ShowAboutPanel`, `app.go:455-468`) |
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
| `Accent`, `AccentHover`, `AccentPressed`, `AccentText`, `Focus` | the ink of section 5.2 | the ink of section 5.2 | Primary button, switches, marks, chosen-segment pill, focus ring, drag highlight, the chosen row's leading bar |
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
`SurfaceHover` is `rowHover`, and `Focus` keeps the real accent, with
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
pill, the focus ring, the drag highlight, and the leading bar of a chosen row
or item. Text is never drawn in the accent.

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
| Window | `mygo.NewWindow(WindowOptions{Title: app.name.mygo, Width: 1200, Height: 780, MinWidth: 640, MinHeight: 480, StateKey: "main", Content: ui.View(a.view)})` (`window.go:61-206`). The system's own title bar (`TitleBarDefault`) on every OS | |
| Root | `ui.Row` filling the window, children stretched | |
| Sidebar | `ui.Themed` (section 5.1) 200 wide, `sidebarFill`, hairline `rule` at its right. A `ui.Sidebar` with `nav.convert` and `nav.history` grows, and a second `ui.Sidebar` with `nav.settings` sits at the bottom. Both bind the same `*string` page path (section 14) | Items as mygo draws them |
| Content | `router.View`, growing | |
| List and detail | `ui.Split(c, &listWidth, list, detail)` with the clamp of section 14 | List column `listFill`, detail `Background` |
| Column header | `ui.Row` 52 high, padding 0 12: page title (`RoleHeading`, level 1), `Spacer`, icon buttons | |
| Detail bar | `ui.Toolbar` 52 high, padding 0 16, `Label(a11y.actions)`: Back (stacked layout only), `Spacer`, view switch, `Spacer`, action buttons | |

### 6.2 Convert, empty

DESIGN 6.2's content, centered in the content area:

| Element | mygo |
|---|---|
| Drop zone | `ui.Column` with the section 5.4 frame, `Role(RoleGroup)`, `Label(a11y.drop_zone)`, the drop target of section 10.1 |
| Glyph | Add files, 48, `TextMuted` |
| Title | Empty-state title. Reads `convert.drop.release` while a drag is over the window |
| Body | `TextMuted`, centered, with the abbreviated output folder |
| Primary button | `ui.PrimaryButton(convert.choose).AutoFocus()` |
| `convert.or` divider | `ui.Row`: a `Box` 1 high growing in `rule`, the word in `TextMuted`, another such `Box` (section 14) |
| URL row | Field and `convert.url.add` button (section 6.3) |
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
| Body, Done | Preview: the web view element of section 8, growing. Markdown: `ui.TextArea` with `ReadOnly(true)`, `Font("monospace")`, `FontSize((B - 1) * zoom)`, word wrap, `Label(a11y.markdown_text)`, padding 16, growing. Both sit inside the content inset |
| Body, Converting | `detail.converting.note` in `TextMuted` |
| Body, Queued | `detail.queued`, `TextMuted` |
| Body, Failed | Section 12's error block |
| No selection | `detail.empty`, centered, `TextMuted` |
| Footer | `ui.Row` 28 high, padding 0 24, `B - 2`, `TextMuted`, `SingleLine()` |

History entries use the Done column (DESIGN 6.4), their status line followed
by ` · ` and the finish date of section 6.5. The Markdown is read from
`markdown_path` on a goroutine. A missing file shows `detail.missing.title`
(detail title style) and `detail.missing.body`, and only Remove from History
stays enabled.

### 6.5 History

| Element | mygo |
|---|---|
| Column header | `nav.history`, then an icon button Clear History (trash glyph, `Label` and `Tooltip` `history.clear`, disabled when History is empty) |
| Search | `ui.SearchField(c, &query)` with `Label(history.search)` and `Placeholder(history.search)`, full width, below the header (`docs/ui/search-field.md`). Filters on `Changed`: case-insensitive substring of title or source |
| List | `ui.List(c, &historyList, n, row)` growing, `Label(nav.history)`. `ListState.Key` is the entry id, `Label` the row name, `Selected` the selected index among the filtered entries (`docs/ui/list.md`). Rows are 44 high and drawn as section 6.3's rows |
| Row | Kind glyph, title, subtitle = finish date (`date.today`, `date.yesterday`, else the locale's medium date and short time) and ` · ` `state.warnings.*` when there are warnings, trailing Warning glyph |
| Context menu | DESIGN 6.5's items |
| Empty | `history.empty.title` (empty-state title) and `history.empty.body`, centered in the content area, without a split |
| No matches | `history.no_results` as the list's child, shown while it has no rows (`docs/ui/list.md`, Layout), padding 12, `TextMuted` |
| Detail | Section 6.4, Done column |

### 6.6 Settings

A page in the content area: `ui.Scroll` filling it, holding a column of at most
640, padding 24, gap 24. The page title `nav.settings` (page title style,
`RoleHeading` level 1) comes first. Each group is a heading (`RoleHeading`
level 2) above a box (section 5.4) whose rows are separated by `rule`
hairlines. Rows are at least 52 high: title and note on the left (`B`, then
`B - 2` `TextMuted`), the control on the right.

| Group | Row | mygo |
|---|---|---|
| `settings.conversion` | `settings.llm`, `settings.ocr`, `settings.screenshot`, `settings.desc`, `settings.alt`, each with its `.note` | `ui.Switch` with `Label` = the row title. `settings.desc` and `settings.alt` are `Disabled` while `settings.llm` is off |
| `settings.output` | `settings.output.folder`, note, then the abbreviated path (`Selectable()`) | `ui.Button(settings.output.change)` opens `mygo.Dialog.Open` with `Directory: true` (`dialog.go:34-35`). `ui.Button(settings.output.show)` calls `Shell.ShowItemInFolder` |
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
| Notice (folder refused, duplicate, history reset, save or open failed) | A `ui.Row` below the URL row, gap 6: Warning glyph 14 in `Warning`, the text in `Text`. `Role(RoleStatus)`, and `c.Announce` of its text when it appears. It clears after 8 s (`c.After`) or at the next add |
| Copied | The Copy Markdown button's glyph becomes `check` for 1.5 s, and `c.Announce(notice.copied)` |
| File pickers | `mygo.Dialog.Open` (multiple, files) and `mygo.Dialog.Save` with `Parent` set, which makes them sheets on macOS (`dialog.go:24-25`, `internal/darwin/dialog.go:19-28`). They block, so they run on a goroutine and post the answer back (`dialog.go:9-10`) |

The two alerts are drawn in the window, so their text follows the app's
language on every OS. mygo draws no destructive style. The button's words
carry the meaning.

### 6.8 Errors in the detail

See section 12.

### 6.9 Fatal

On an ABI mismatch the view draws only a centered column: `fatal.abi.title`
(page title style), `fatal.abi.body`, and `ui.PrimaryButton(quit.confirm)`,
which calls `App.Quit`. No sidebar and no shortcuts other than Quit.

### 6.10 Window sizes and narrow widths

| | Value |
|---|---|
| Default | 1200 × 780 |
| Minimum | 640 × 480 on every OS |
| Navigation | Never collapses. At the minimum width the sidebar keeps 200 and leaves 440 for the page |
| Stacked list and detail | When the content width (`c.Size()` minus 200, `ui/context.go:411`) is below 760, Convert and History stack |

Stacked layout:

- The list column fills the content area. Choosing a row with a click, Enter or
  a context menu's default pushes `/convert/item` or `/history/item`. Up and
  Down only move the selection.
- The detail page has a Back button at the leading end of its detail bar:
  chevron glyph and `common.back`, `Label(common.back)`.
- Back, Esc, ⌘[ on macOS and Alt+Left elsewhere (the router's own keys,
  `docs/ui/navigation.md:84-87`), and a mouse's back button return
  to the list with the row still selected.
- Leaving the stacked width while on an `/item` page replaces it with the list
  page. The selection is kept.
- Empty Convert, empty History and Settings never stack (DESIGN 14).

## 7. Motion

Each animated change, with the mygo call that makes it:

| Change | Element | mygo | Timing |
|---|---|---|---|
| Row enters (Convert) | The new row | `Transition(rowMotion)` with `rowMotion = ElementTransition{Duration: 150ms, Ease: EaseOut, Enter: &Motion{}, Exit: &Motion{}}`: it fades in while the rows below slide down (`docs/ui/transitions.md`, Coming and going) | 150 ms, ease out |
| Row leaves (Convert) | The removed row and those below | Same transition: the row fades out in place while those below slide up | 150 ms, ease out |
| Row state changes | Trailing glyph, subtitle | `ContentTransition{}`: the old glyph shrinks away as the new grows in, the text cross-fades (`ui/content.go:19-27`) | Spring, about 350 ms, no bounce |
| History rows | None | `ui.List` places its rows without transitions (`docs/ui/transitions.md:167-169`) | none |
| Hover and selection | Row fill | `row.AnimateColor("fill", fill, Animation{Duration: 120ms})` | 120 ms, ease out |
| Drop highlight | Drop zone or bar | `Transition(ElementTransition{Colors: true, Duration: 120ms})` fades the border to `Accent`. Dashed turns solid at once. The heading uses `ContentTransition{}` | 120 ms |
| Segmented switch | The chosen-segment pill | A `Box` built in the chosen segment with `Transition(ElementTransition{Animation: &ui.Snappy, Match: viewPill{}})`: it flies from the old segment to the new (`docs/ui/transitions.md`, Shared elements). The body swaps at once | `ui.Snappy` spring |
| Sidebar page change | Content | The router fades the new page in (`TransitionAuto`, `ui/router.go:646-665`) | 150 ms, ease out |
| Stacked push and pop | Content | The router slides the deeper page in from the right and back out to the right | 250 ms, ease out |
| Notice shows and hides | Notice row | `Transition(ElementTransition{Duration: 150ms, Enter: &Motion{Y: -4}, Exit: &Motion{}})` | 150 ms |
| Copy checkmark | Copy button glyph | `ContentTransition{}` from copy to check, and back after 1.5 s | Spring, about 350 ms |
| Warnings disclosure | Panel and chevron | `CollapsibleBase` grows the panel (`ui/collapsible.go:38`). The chevron's `Rotate` is `90 * Progress()` | 200 ms |
| Converting | Spinner | `ui.Spinner` | Continuous |
| Alerts | None | mygo draws `AlertDialog` at once (`ui/feedback.go:176-219`) | none |

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
	Transparent: true,
	Page: mygo.PageOptions{
		DevTools:      mygo.DevToolsDisabled,
		Session:       mygo.NewEphemeralSession(),
		ContentRules:  []*mygo.ContentRules{noNetwork},
		PreloadScript: previewKeysJS,
	},
})
```

The detail builds `ui.WebView(c, preview).Grow(1).Label(a11y.preview)` only
while a Done item's Preview tab shows. An element the view does not build
hides its web view and keeps its page (`ui/webview.go:26-27`).

| DESIGN 8 concern | mygo |
|---|---|
| Load | `Page.LoadHTML(html, baseURL)` with `baseURL` the `file://` URL of the result folder with a trailing slash (`window.go:1348-1351`). macOS calls `loadHTMLString:baseURL:` (`internal/darwin/window.go:1379-1388`), Linux `webkit_web_view_load_html` (`internal/linux/window.go:1100-1103`), Windows serves the document at the base URL (`internal/windows/webview.go:667-684`). Images inside the folder load on Windows **(unverified)** |
| Scripts | The page's own scripts never run: the document's CSP `default-src 'none'` blocks them, and `preview_html` escapes raw HTML (DESIGN 3.3). The only script is `previewKeysJS` (section 8.3), injected by the host, which the CSP does not govern **(unverified per OS)** |
| Network | The CSP, plus `noNetwork = CompileContentRules("preview-no-network", rules)` blocking `^https?://` and `^wss?://` (`content_rules.go:28-81`), plus an ephemeral session (`session.go:88-93`). Compile once, off the main thread, at launch |
| Navigation | `Page.OnWillNavigate`: allow a URL equal to the base URL with only a fragment added. Open `http`, `https` and `mailto` with `Shell.OpenExternal` and `PreventDefault`. Prevent everything else (`window.go:1738-1741`, `events.go:32-42`). No `SetWindowOpenHandler`: without one, mygo opens `http` and `https` in the browser and denies the rest (`window.go:1549-1552`) |
| Context menu | `Page.OnContextMenu`: keep only the item with ID `ContextMenuCopy`, relabeled `common.copy`, when text is selected. Otherwise `PreventDefault` (`context_menu.go:25`, `context_menu.go:124`) |
| Developer tools | `DevToolsDisabled` (`window.go:38-46`), which also turns off WebView2's browser keys (`internal/windows/webview.go:205-212`) |
| Background | `Transparent: true`, so the detail's `Background` shows under the page (`webview.go:44-46`) |
| Zoom | `Page.SetZoomFactor(view.zoom)` (`window.go:1409`) |
| Appearance | On a change of `c.Theme().Dark`, render again with the new `theme` and load it |
| Missing web engine | When `NewWebView` fails, the Preview tab shows `preview.webview2_missing` on Windows and `preview.unavailable` elsewhere, centered, and the Markdown tab works |
| Accessible name | `a11y.preview` on the element. On Windows, Narrator finds the page's elements in WebView2's own window (`docs/ui/web-view.md:280-282`) |

### 8.2 Rendering

`preview_html` runs on a goroutine with `{markdown, theme, base_dir}` (DESIGN
3.3). Results are cached by `(item id, theme)`, at most 8. When the shown item
or theme changes, the view keeps the web view element unbuilt until the new
document is loaded (`Page.OnDidFinishLoad`), so the old document never shows
under the new title.

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

What the app does. `previewKeysJS` listens to `keydown` in the capture phase.
For each key combination of section 9.2 that the page does not need, it calls
`preventDefault()` and `mygo.call("PreviewKeys.Press", "<action>")`.
`PreviewKeys.Press` posts the action to the UI thread, which runs the same
`app` method a native shortcut runs. The page is trusted because its base URL
is `file:` (`window.go:1314-1323`), so the call goes through, and it is not
`Untrusted`, which would refuse it (`page.go:375-392`). The page keeps ⌘C or
Ctrl+C, ⌘A or Ctrl+A, arrows, Page Up and Page Down, Home, End, Space and
Tab. Forwarded actions: Choose Files, Add Web Address, Go to Convert, Go to
History, Settings, Switch Preview and Markdown, Copy Markdown, Save As, Show in
folder, Search History, the three zoom keys, Back (Esc, ⌘[ or Alt+Left), and on
Linux Close and Quit. The script compares `event.code` with modifiers, so it
works on any keyboard layout. On macOS a key the page leaves unhandled goes
on to the menu bar. The script's `preventDefault` marks it handled, so the
action runs once, through the script **(unverified)**.

## 9. Menus and shortcuts

### 9.1 macOS menu bar

`App.SetMenu` at `WhenReady`. Role menus take mygo's own titles and relabel
with the app's language (`menu.go:390-401`, `menu_text.go:34-66`).

| Menu | Items |
|---|---|
| App (`RoleAppMenu` with its own `Submenu`) | `menu.about.mygo` (shows the About panel with `ApplicationName` `Markitai Mygo`, the app version and `settings.about.copyright`), separator, `menu.settings` ⌘,, separator, `RoleServices`, separator, `RoleHide`, `RoleHideOthers`, `RoleUnhide`, separator, `RoleQuit` |
| File (`RoleFileMenu`) | `convert.choose` ⌘O, `menu.add_url` ⌘L, separator, `detail.save_as` ⇧⌘S, `detail.show.macos` ⇧⌘R, separator, `RoleClose` |
| Edit (`RoleEditMenu`) | `RoleUndo`, `RoleRedo`, separator, `RoleCut`, `RoleCopy`, `detail.copy` ⇧⌘C, `RolePaste`, `RoleSelectAll`, separator, `history.search` ⌘F |
| View (`RoleViewMenu`) | `nav.convert` ⌘1, `nav.history` ⌘2, separator, `menu.toggle_view` ⇧⌘M, separator, `menu.actual_size` ⌘0, `menu.zoom_in` ⌘=, `menu.zoom_out` ⌘-, separator, `detail.retry` ⌘R, `convert.remove` or `history.remove` ⌘⌫ |
| Window (`RoleWindowMenu`) | mygo's default |

Each item's `Click` calls the `app` method. Items are enabled and disabled each
frame with `SetEnabled` from the same state that enables the buttons, and the
remove item is relabeled with `SetLabel` on the History page. Paste reaches
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
| Remove from List or History | ⌘⌫ | Delete | `Element.Shortcut` on the focused list |
| Search History | ⌘F | Ctrl+F | Go to History, focus the search field |
| Zoom in, out, actual size | ⌘=, ⌘-, ⌘0 | Ctrl+=, Ctrl+-, Ctrl+0 | `ui.Cmd` with `ui.KeyEqual` (with or without Shift), `ui.KeyMinus`, `ui.Key0`. Steps of 0.1 from 0.5 to 3.0 |
| Back (stacked layout) | ⌘[ or Esc | Esc or Alt+Left | The router's keys, plus `ui.KeyEscape` |
| Move in a list | ↑ ↓ | ↑ ↓ | The list's own |
| Close the window | ⌘W | Alt+F4 on Windows, Ctrl+W on Linux | `RoleClose`, the system, `ui.Cmd, ui.KeyW` |
| Quit | ⌘Q | Alt+F4 on Windows, Ctrl+Q on Linux | `RoleQuit`, the system, `ui.Cmd, ui.KeyQ` |

Linux follows the Windows column. The mygo build is a GTK 3 app with its own
drawing, not a GNOME app, so GNOME's Alt+1 and Alt+2 do not apply. Shortcuts are
off while an alert shows, except Escape and Enter, which the alert takes.

## 10. Behavior per OS

DESIGN 7 applies. These rows replace its platform rows.

### 10.1 Adding sources

| Way | mygo |
|---|---|
| Drop on the window | The content area is the drop target on Convert and History: `ui.DataDragOver` drives the highlight, and `ui.DropData` takes `transfer.DropOptions{Formats: {transfer.FileList, transfer.URIList}}` (`ui/api_gen.go:1130-1140`, `docs/ui/drag-and-drop.md:60-87`). Files become items. Without files, `http` and `https` entries of `Data.URLs()` become URL items (`transfer/data.go:268-285`). macOS registers `public.url` for this (`internal/darwin/surface_drag.go:178-187`). A drop on History goes to Convert |
| `convert.choose` | `mygo.Dialog.Open(OpenDialogOptions{Parent: win, Multiple: true})` (`dialog.go:128-159`) |
| Paste | `c.Services().ReadClipboardData(transfer.FileList, transfer.Text)` (`ui/services.go:41-51`): files first, else text that is one valid URL |
| Open with the app | macOS: `App.OnOpenFile` (Dock drops, Open With), which `CFBundleDocumentTypes` allows for any file (`internal/darwin/app.go:345-360`). Windows and Linux: `os.Args[1:]` at launch, and the `args` of `App.OnSecondInstance` (`single_instance.go:119-121`), keeping arguments that are existing paths. `OnOpenFile` is not used there, because it passes only extensions in `fileAssociations` (`deeplink.go:95-123`). The Linux launcher entry passes no files (section 14) |

### 10.2 Copy, Save As, Show

| Action | mygo |
|---|---|
| Copy Markdown | `c.WriteClipboard` (`ui/context.go:491-492`) |
| Save As… | `mygo.Dialog.Save(SaveDialogOptions{Parent: win, DefaultPath: <label>.md})` (`dialog.go:161-189`), then DESIGN 7.2's write and `.markitai` copy on a goroutine. A failure shows the notice `error.save_failed` |
| Show in folder | `mygo.Shell.ShowItemInFolder(path)` (`modules.go:45-48`) |
| Open configuration | `mygo.Shell.OpenPath(path)` (`modules.go:40`). A failure shows `error.open_failed` |

### 10.3 Quit and closing the window

| OS | Rule |
|---|---|
| macOS | Closing the window hides it: `Window.OnClose` calls `PreventDefault` and `Hide` unless the app is quitting (`window.go:1661-1663`, `events.go:8-24`). Conversions keep running. `App.OnActivate` shows the window again when the Dock icon is clicked with no window visible (`app.go:494-499`). `App.OnWindowAllClosed` is set, so mygo does not quit (`app.go:473-478`) |
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

The app never calls `Theme.SetSource`, so the theme follows the system
(`modules.go:480-483`). There is no theme setting. A change re-renders the preview (section 8.1).

### 10.5 Language

| Rule | mygo |
|---|---|
| Choice | DESIGN 7.5: zh-Hans when the first of `mygo.App.PreferredLanguages()` starts with `zh`, else English. On macOS that list is the global `AppleLanguages`, never the app's own domain (`internal/darwin/locale.go:44-84`). `MYGO_LOCALE`, when set, goes first, for development |
| App strings | `desktop/strings.json`, embedded through the copy in `internal/l10n/strings.json`, which a test keeps byte-equal |
| mygo's own text | `App.SetLanguages("en", "zh-Hans")`, then `App.SetLocale(<choice>)` (`locale.go:77-180`). Role menus, text-field context menus and mygo-drawn labels follow at once. `SetLocale` also stores the choice, on macOS in the app's defaults (`internal/darwin/locale.go:93-104`). Each launch chooses again from the global list |
| Windows system text | mygo sets the process's preferred UI languages (`internal/windows/locale.go:112-129`), so the system file pickers follow the app |
| Linux system text | The file pickers are labeled with mygo's strings in the app's language (`internal/linux/dialog.go:72-106`) |
| macOS system text | AppKit's own text (open and save panels, the About panel, the items AppKit adds to menus) takes `AppleLanguages` when AppKit starts, and mygo's `SetLanguages` does nothing on macOS (`internal/darwin/locale.go:106-108`). So `main`, before `App.Run`, sets `AppleLanguages` = `[<choice>]` in the process's argument domain: `NSUserDefaults.standardUserDefaults`, its `volatileDomainForName:` `NSArgumentDomain` with the key added, then `setVolatileDomain:forName:`. It calls them through `github.com/ebitengine/purego/objc`, without cgo. Markitai Native's macOS `main` does the same (`localizeTheSystemMenus` in its `Main.swift`). mygo makes `NSApplication` only in `App.Run` (`app.go:105-110`, `internal/darwin/app.go:102-121`), after this. The argument domain lasts for the process and writes nothing. `CFBundleLocalizations` lists English and zh-Hans. A zh-Hant user then sees zh-Hans in every panel from the first launch **(unverified in a mygo process)** |

### 10.6 Single instance

Section 2.1. A second launch's file arguments are added as items, and the
window is shown and focused (`Window.Show`, `Window.Focus`).

## 11. Accessibility

DESIGN 12 applies. mygo exposes its drawn UI to VoiceOver, Narrator, NVDA and
Orca (`docs/ui/accessibility.md`).

| DESIGN 12 rule | mygo |
|---|---|
| Labels | Icon-only buttons: `Label` = `Tooltip` = the action's copy key. URL field `Label(convert.url.label)`. Drop zone and drop bar `Role(RoleGroup)` and `Label(a11y.drop_zone)`. Conversions list `Role(RoleList)`, `Label(a11y.conversions)`. History list `Label(nav.history)`. View switch: `SegmentedBase` is a radio group, `Label(detail.view)` (`docs/ui/segmented.md`). Warnings box `Label(a11y.warnings)`, its trigger `Expanded`. Detail bar `ui.Toolbar` `Label(a11y.actions)`. Preview `Label(a11y.preview)`. Markdown text `Label(a11y.markdown_text)` |
| Row names | Convert rows `Label("<title>, <subtitle>")`. History rows through `ListState.Label`, which a list's rows are read by, with "n of N" (`docs/ui/accessibility.md`, Lists and tables) |
| State changes | `c.Announce(a11y.done)` and `c.Announce(a11y.failed)` in the frame after the outcome, and for notices and Copied (`ui/context.go:499-503`) |
| Headings | Page titles `RoleHeading` level 1, the detail title and Settings group headings level 2 |
| Focus order | DESIGN 12's order. The navigation is two Tab stops: the pages' sidebar, then the Settings sidebar. Empty Convert focuses Choose Files… (`AutoFocus`). A router page change moves the focus into the page and announces its title (`docs/ui/navigation.md`, The focus and screen readers). Each route sets `r.Title` to the page's name |
| Keyboard | Every action has a shortcut (section 9.2) or a Tab path. Row context menus repeat buttons and shortcuts |
| Contrast and color | Section 5.1. Status glyphs differ by shape |
| Reduced motion | Section 7. Its added motion replaces DESIGN 12's "no other custom animation" (owner decision 6) |
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
| The page keeps keys from `c.Shortcut` | Section 8.3 | `previewKeysJS` forwards them |
| `ui/markdown`'s `FileImages` reads paths outside its folder | `ui/markdown/image.go:25-43` (spike report) | Not used. The preview is the web view, whose images `preview_html` confines (DESIGN 3.3). No user image is decoded in native UI |
| No destructive alert style, no alert motion | `ui/feedback.go:176-219` | Accepted |
| macOS system panels take the language when AppKit starts | `internal/darwin/locale.go:106-108` | The argument domain, set before `App.Run` (section 10.5) |
| Linux launcher entry passes no files without `fileAssociations` | `cmd/mygo/build.go:609-651` | Accepted. Command-line arguments still add files |
| `Tester.Texts` omits a `TextArea`'s content | `ui/headless.go:471-478` | Tests read the Markdown text with Select All and Copy (spike `TestDropToDone`) |
| `Tester` places web views but shows no page | `ui/headless.go:204-208`, `docs/ui/web-view.md:286-290` | The app loads pages through a small `previewPane` interface whose `View()` is the `ui.NativeWebView` to build. Tests pass a fake that records the HTML, base URL and zoom. Its view embeds a nil `ui.NativeWebView`, because the interface's one method names mygo's internal types (`internal/surface/surface.go:109-113`) and the Tester never calls it (`ui/headless.go:208`). `Tester.WebViewAt` then finds it where the Preview tab shows |

## 15. Build

`python3 desktop/mygo/build.py [--check] [--no-core]` on macOS, Windows and
Linux, Python 3.13 standard library only, mirroring
`desktop/native/macos/build.py` and `desktop/native/linux/build.py`:

1. Build the core: `cargo rustc -p markitai-app --release --locked --lib
   --crate-type cdylib`, on macOS with `MACOSX_DEPLOYMENT_TARGET=13.0`.
   `--no-core` reuses `target/release`.
2. Stage it in `desktop/mygo/resources/<goos>-<goarch>/` (ignored).
3. Copy `desktop/strings.json` to `internal/l10n/strings.json`. Check that
   `mygo.json`'s `version` equals the workspace version.
4. `go vet ./...`.
5. `go test -count=1 ./...` with `MARKITAI_APP_LIB` set to the staged
   library and an isolated environment: `HOME`, `MARKITAI_HOME`,
   `XDG_DATA_HOME`, `XDG_CONFIG_HOME`, `XDG_CACHE_HOME`, `XDG_STATE_HOME`,
   `USERPROFILE`, `APPDATA` and `LOCALAPPDATA` in a temporary folder, `TERM=dumb`.
   `GOMODCACHE`, `GOCACHE` and `GOPATH` keep their real values so the module
   cache is not downloaded again.
6. `CGO_ENABLED=0 go tool mygo build -platform <goos>/<goarch> -skip-dmg
   -skip-installer`, into `desktop/mygo/build/`.

`--check` runs steps 1 to 5 and `desktop/scripts/strings.py --check`, and
builds nothing. It never writes a tracked file: in place of step 3's copy it
fails when the copy differs.

Hosted verification: a `windows` job in `.github/workflows/desktop-mygo.yml`
on `windows-2025`: checkout, the repository's Rust toolchain, `actions/setup-go`
with `go-version-file: desktop/mygo/go.mod`, `python desktop/mygo/build.py
--check`, then `python desktop/mygo/build.py --no-core`, and an upload of
`desktop/mygo/build/` as `markitai-mygo-windows-amd64` and the test log. The
workflow runs on changes under `desktop/mygo/`, `desktop/strings.json` and
`crates/`. The user guide is `docs/desktop-mygo.md`.

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
| Failure | `ocr/fixtures/english.png` with AI and OCR off fails with `image_only`: sentence, Retry, Open Settings goes to Settings, and the claimed folder is gone. A missing file fails with `not_found` |
| Retry | Retry captures the current settings and queues |
| Remove and Clear Finished | Converting rows refuse removal. Selection moves to the next row |
| History | Done items appear newest first. Search filters, no-match text. Remove keeps files. Clear asks, Cancel keeps, Clear empties. 501 additions keep 500. A corrupt file is renamed and reported once. A missing Markdown file shows `detail.missing.*` |
| Settings | Switches write `settings.json`. Desc and Alt are disabled without AI. Change refuses an unwritable folder. Configuration choice, Open and Create |
| Quit | With an item Converting, the close path asks. Cancel keeps it. The alert follows the count and closes by itself when the conversion ends. Quit deletes the claimed folder |
| Shortcuts | Each section 9.2 action, from the native UI and through `PreviewKeys.Press` |
| Stacked layout | At 800 wide: list fills, Enter pushes the detail, Esc and Back return with the selection kept, widening replaces the page |
| Motion | Row removal slides over 150 ms, and is at once with Reduce Motion (spike `TestRowMotionFollowsReduceMotion`). With Reduce Motion no frame is pending after any change except the spinner |
| Accessibility | Names of section 11 found with `Tester.Find`. Announcements with `Tester.Announcements` |
| Copy | Every visible string comes from `strings.json` in both languages: a test walks every screen and checks `Tester.Texts` against the catalog's values, allowing file names, paths, numbers and the core's text |
| Images | The preview's base URL is the result folder. Remote images are not loaded (the HTML from `preview_html` holds `img-remote`) |

## 17. Checks before calling a platform done

| Check | How |
|---|---|
| Build | `build.py --check` and `build.py` pass on the platform. Windows passes in the hosted job |
| States | DESIGN 16's States, Folders and History checks, run in the built app with an isolated `HOME` and `MARKITAI_HOME` |
| Preview | In the built app: a fixture with tables, task lists, footnotes and a relative image shows them, light and dark. A link opens the browser. A remote image stays a placeholder. No request leaves the machine. The context menu holds only Copy |
| Keys in the preview | With the page focused (clicked), every forwarded action of section 8.3 runs exactly once, and ⌘C or Ctrl+C copies the page's selection |
| Quit | Every quit path of section 10.3 asks while converting. macOS: closing the window keeps conversions running, the Dock icon brings it back |
| Coexistence | With Markitai Web and Markitai Native running, Markitai Mygo starts its own instance, writes only its folders of section 2.1, and its History stays its own |
| Language | zh-Hans and English runs show no other language in mygo-drawn UI. macOS, with zh-Hant first in the system's languages: the first launch's open panel and About panel are in zh-Hans |
| Accessibility | Keyboard-only run through every flow. VoiceOver, Narrator and Orca read row names, announcements and the alert |
| Motion | Reduce Motion on: no movement and no fades except the spinner |
| Copy | DESIGN 16's Copy check |

## 18. Unverified

- Images inside the result folder load in WebView2 from a document that
  `LoadHTML` serves at a `file:` base URL (section 8.1).
- Host-injected `PreloadScript`s run in a page whose CSP is `default-src
  'none'`, on all three engines, and `mygo.call` reaches a bound method from
  it (section 8.3).
- On macOS, a key the preview script prevents does not also reach the menu
  bar's key equivalents (section 8.3).
- AppKit's panels and About panel follow `AppleLanguages` set in the
  argument domain from Go, through purego, before `App.Run` (section 10.5).
- Dropping a link from a browser delivers `transfer.URIList` on Windows and
  Linux (section 10.1).
