# Markitai native apps: design specification

This file specifies the macOS, Windows and Linux (GNOME) apps. The three apps
share one information architecture, one set of flows, states and copy, and one
brand. Each app draws them with its own platform's controls. The design canvas
"Markitai 原生應用設計稿" shows every screen on every platform. Where the canvas
and this file differ, this file is right.

The apps call the markitai Rust core in process through `crates/markitai-app`.
They never start or wrap `markitai serve`. The only web content they show is the
Markdown preview, rendered by the core and displayed in the system web view.

## 1. Scope

| In v1 | Not in v1 |
|---|---|
| Convert local files and web pages (http and https) to Markdown | Folders and recursive batches |
| Two conversions at a time, each into its own output folder | Canceling a running conversion |
| Preview, Markdown text, copy, save a copy, reveal the file | Editing Markdown |
| History of finished conversions | Model, provider and key management (users edit the markitai configuration) |
| Five feature switches, output folder, configuration source | Presets, profiles, `--keep-base`, PDF side-by-side original |
| English and Simplified Chinese (zh-Hans) | Math, Mermaid and syntax highlighting in the preview |
| Light and dark appearance | Notifications, Dock or taskbar badges, auto-update |

## 2. Technology

| | macOS | Windows | Linux (GNOME) |
|---|---|---|---|
| UI | SwiftUI, AppKit only where SwiftUI has no control (`NSTextView`, `WKWebView`) | WinUI 3 in C# | GTK 4 and libadwaita in Rust |
| Minimum system | macOS 13 | Windows 10 1809 (build 17763). Mica on Windows 11, solid background on Windows 10 | GTK 4.18, libadwaita 1.7, WebKitGTK 6.0 (GNOME 48: Debian 13, Ubuntu 25.04, Fedora 42 and later) |
| Toolchain | SwiftPM and Command Line Tools. No Xcode project, no asset catalog (`actool` is not in Command Line Tools) | .NET 10, Windows App SDK 1.8 or later, `WindowsPackageType=None`, `WindowsAppSDKSelfContained=true`, `SelfContained=true` | Cargo, gtk-rs crates with features `v4_18` and `v1_7`, the `webkit6` crate |
| Core link | `libmarkitai_app.a` (staticlib) through a module map. Link the frameworks that `bindings/go/link_static_darwin_arm64.go` names | `markitai_app.dll` (cdylib) next to `Markitai.exe`, called with `LibraryImport` | `markitai-app` (rlib) as a path dependency |
| Extra packages | none | `CommunityToolkit.WinUI.Controls.SettingsControls`, `CommunityToolkit.WinUI.Controls.Segmented` | `serde`, `serde_json`, `uuid` |
| App ID | Bundle ID `dev.markitai.Markitai`, executable `Markitai` in `Markitai.app` | `Markitai.exe`, single-instance key `dev.markitai.Markitai` | `dev.markitai.Markitai` (GApplication, desktop file, icon), binary `markitai-app` |
| Source folder | `native/macos/` | `native/windows/` | `native/linux/`, its own Cargo workspace. The root `Cargo.toml` gains `exclude = ["native/linux"]` |

`dev.markitai.Markitai` replaces the unreleased mygo shell's `dev.markitai.desktop`.

macOS bundle. A script assembles `Markitai.app`: `Info.plist`, the icon made
with `sips` and `iconutil` from `desktop/icon.png`, and an ad hoc `codesign`. The
app is not sandboxed, because the core runs LibreOffice, Chrome and the
subscription CLIs. `Info.plist` sets `CFBundleIdentifier`,
`CFBundleShortVersionString` (the workspace version), `LSMinimumSystemVersion`
`13.0`, `NSDocumentsFolderUsageDescription`, `NSDesktopFolderUsageDescription`
and `NSDownloadsFolderUsageDescription` (copy keys `plist.*`), and
`CFBundleDocumentTypes` with one entry (`CFBundleTypeRole` `Viewer`,
`LSHandlerRank` `None`, `LSItemContentTypes` `public.data`) so files dropped on
the Dock icon reach the app.

Linux files. `native/linux/data/` holds `dev.markitai.Markitai.desktop`
(`Exec=markitai-app %F`, `Categories=Office;Utility;`), the metainfo file and the
hicolor icons. GSettings is not used, so the app runs without an installed schema.

## 3. The `markitai-app` crate

`crates/markitai-app` is a private interface for these three apps only. It is
not a public binding and is not documented in `docs/bindings.md`.

| Item | Value |
|---|---|
| `crate-type` | `rlib` only. The macOS build runs `cargo rustc -p markitai-app --release --lib --crate-type staticlib`, the Windows build `--crate-type cdylib`: a crate type declared in `Cargo.toml` would also be built, and fully linked, every time the Linux app builds it as a dependency |
| Dependencies | `markitai-core`, `serde_json`, `pulldown-cmark` 0.13 (already in `Cargo.lock`) with default features off and `html` on, `libc` on Unix |
| Header | `crates/markitai-app/include/markitai_app.h` |
| Stylesheet | `crates/markitai-app/src/preview.css`, read with `include_str!`, exactly the text in section 15 |
| `default-members` | not added |
| Panics | every entry point catches them and answers `internal_error`, like `markitai-ffi` |
| Request limit | 64 MiB per request, else `invalid_input` |

### 3.1 C declarations

```c
#ifndef MARKITAI_APP_H
#define MARKITAI_APP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MARKITAI_APP_ABI_VERSION 1u

/* Owns len bytes of UTF-8 JSON, without a trailing NUL. Release it exactly once
   with markitai_app_buffer_free. Do not copy this ownership handle. */
typedef struct MarkitaiAppBuffer {
    uint8_t *data;
    size_t len;
} MarkitaiAppBuffer;

uint32_t markitai_app_abi_version(void);

/* The markitai version, such as "2.3.2". Static UTF-8, never free or modify. */
const char *markitai_app_version(void);

/* Each request is UTF-8 JSON, borrowed until return, at most 64 MiB. Each
   response is {"ok":true,"result":{...}} or
   {"ok":false,"error":{"code":"...","message":"..."}}. */
MarkitaiAppBuffer markitai_app_prepare_environment(const uint8_t *request, size_t len);
MarkitaiAppBuffer markitai_app_convert_json(const uint8_t *request, size_t len);
MarkitaiAppBuffer markitai_app_preview_html(const uint8_t *request, size_t len);
MarkitaiAppBuffer markitai_app_claim_output_folder(const uint8_t *request, size_t len);
MarkitaiAppBuffer markitai_app_config_file(void);

/* Clears the struct. Null, or a struct already cleared, is harmless. */
void markitai_app_buffer_free(MarkitaiAppBuffer *buffer);

#ifdef __cplusplus
}
#endif
#endif
```

`markitai_app_abi_version` returns `1`. Each app compares it with its own
constant at launch. A mismatch shows only the `fatal.abi.*` message and a Quit
button.

### 3.2 Rust API for the Linux app

The Linux app uses these functions and never calls `markitai-core` directly. They
take and return the same JSON as the C functions, so all three apps share one
contract.

```rust
pub const ABI_VERSION: u32 = 1;
pub fn version() -> &'static str;
pub fn prepare_environment(request: &str) -> String;
pub fn convert_json(request: &str) -> String;
pub fn preview_html(request: &str) -> String;
pub fn claim_output_folder(request: &str) -> String;
pub fn config_file() -> String;
```

Each C function is a thin wrapper over the Rust function of the same name.

### 3.3 Calls

| Function | Request | Result | When |
|---|---|---|---|
| `prepare_environment` | `{"data_dir": "<app data folder>"}` | `{"login_environment": "imported" \| "inherited" \| "failed", "detail": null \| "<reason>"}` | Once, first, from `main`, before the UI framework starts. Blocks up to 10 s |
| `convert_json` | Exactly `markitai_convert_json`'s request (`docs/bindings.md`, Shared contract) | Exactly its envelope | Per conversion, on a worker thread. Blocks until done. Cannot be canceled |
| `preview_html` | `{"markdown": "...", "theme": "light" \| "dark", "base_dir": "<folder>" \| null}` | `{"html": "<!doctype html>..."}` | Each time a Preview is shown or the appearance changes, on a worker thread |
| `claim_output_folder` | `{"parent": "<output folder>", "source": "<path or URL>"}` | `{"path": "<new folder>", "name": "<its name>"}` | Right before each conversion |
| `config_file` | none | `{"path": "<file>", "exists": true \| false}` | Each time Settings opens |

`prepare_environment` does three things, in this order:

1. It creates `data_dir` and its parents.
2. It sets the process working directory to `data_dir`. The core then never
   reads a `markitai.json` or `.env` from whatever folder launched the app.
3. On macOS and Linux, when `TERM` is unset, it imports the login shell's
   environment the way `desktop/env_unix.go` does: `$SHELL -l -i -c` (or
   `/bin/zsh` on macOS, `/bin/sh` on Linux) prints `env -0` between two random
   markers, in its own process group, killed after 10 s (as long as VS Code waits for the same environment). Each variable is set in
   this process. Finder and desktop launchers otherwise give the app no
   Homebrew `PATH`, no exported API keys and no `MARKITAI_HOME`. On Windows, and
   when `TERM` is set, the result is `inherited`. A `failed` result is not shown
   to the user.

`preview_html` renders the Markdown with these rules:

| Rule | Detail |
|---|---|
| Parser | `pulldown-cmark` with `ENABLE_TABLES`, `ENABLE_STRIKETHROUGH`, `ENABLE_TASKLISTS`, `ENABLE_FOOTNOTES`, `ENABLE_YAML_STYLE_METADATA_BLOCKS` |
| Raw HTML | `Event::Html` and `Event::InlineHtml` become `Event::Text`, so the HTML shows escaped |
| Front matter | The YAML block is left out of the preview. The Markdown tab shows it |
| Images | `data:` and `file:` sources stay. A relative source resolves against `base_dir` into an absolute `file://` URL, if the result stays inside `base_dir`. Every other source (http, https, `//`, absolute paths, `..` outside `base_dir`, any `base_dir` of null) becomes `<span class="img-remote" title="<source>"><alt or source></span>` |
| Links | Kept as written. The app decides what a click does (section 8) |
| Base | No `<base>` element, so footnote links stay in the same document |
| Output | Deterministic: the same request gives the same bytes on every platform |

The document skeleton:

```html
<!doctype html>
<html data-theme="light">
<head>
<meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src file: data:; style-src 'unsafe-inline'">
<meta name="referrer" content="no-referrer">
<meta name="color-scheme" content="light">
<style>/* preview.css, section 15 */</style>
</head>
<body><main>
<!-- rendered Markdown -->
</main></body>
</html>
```

`style-src 'unsafe-inline'` is needed because `pulldown-cmark` writes table
alignment as `style` attributes. It carries no risk here because raw HTML is
escaped. `data-theme` and `color-scheme` take the request's `theme`.

`claim_output_folder` names and creates the conversion's folder:

| Step | Rule |
|---|---|
| Label for a file | The last path component without its last extension (`Quarterly Report Q3 2026.pdf` gives `Quarterly Report Q3 2026`). A name that is only an extension keeps its full name |
| Label for a URL | The host without a leading `www.`, then ` - ` and the last non-empty path segment, percent-decoded, without its extension: `https://example.com/blog/markdown-for-language-models` gives `example.com - markdown-for-language-models`. No path segment gives the host alone |
| Clean | `markitai_core::output_name::sanitize(label, "untitled")` |
| Create | `create_dir_all(parent)`, then `create_dir(parent/label)`. If it exists, try `label (2)`, `label (3)` up to `label (999)`. `create_dir` is atomic, so two conversions never get one folder |
| Errors | `output_unavailable` when the parent cannot be created or written, or 999 names are taken. The message is the OS error |

`config_file` returns `markitai_core::config::selected_path(None)` when it is
`Some`, else `markitai_core::config::home().join("config.json")`. `exists`
reports whether that file exists now.

Error codes the crate adds to the core's: `output_unavailable`. Malformed
requests give `invalid_input`, as `markitai-ffi` does.

## 4. Data model

### 4.1 Conversion item

An item is one source in the Convert list. The Convert list lives in memory only.
A relaunch starts with an empty list. Finished items are also in History.

```text
Item {
  id: UUID
  source: File(absolute path) | Url(string)
  title: string
  added_at: time
  options: Options        captured when the item is added, and again on Retry
  state: Queued
       | Converting { started_at, folder }
       | Done { folder, result }
       | Failed { code, message, folder_was_claimed }
}
Options { llm, ocr, screenshot, alt, desc: bool; config: "user" | "builtin" }
```

| State | Enters when | Leaves to | List row | Detail body |
|---|---|---|---|---|
| Queued | The item is added, or Retry is chosen | Converting | Kind glyph, title, `state.queued`, queued glyph | `detail.queued` |
| Converting | Fewer than two items are converting. The oldest Queued item starts first | Done or Failed | Kind glyph, title, `state.converting`, small indeterminate spinner | Spinner, `detail.converting` with elapsed `m:ss`, `detail.converting.note` |
| Done | The envelope is `ok` and `skip_reason` is null | Removed only | Kind glyph, title, `state.done` and, with warnings, ` · ` and `state.warnings.*`, warning glyph if warnings | Preview or Markdown, warnings, footer |
| Failed | The envelope is an error, `skip_reason` is set, or the folder claim failed | Queued on Retry | Kind glyph, title, `row.failed` with the mapped sentence, error glyph | Error block (section 9) |

Running an item:

1. Call `claim_output_folder` with the output folder and the source. On error,
   the item is Failed with code `output_unavailable`.
2. Build the request. `alt` and `desc` are sent only together with `llm`:
   ```json
   {"source": "/Users/example/Downloads/Quarterly Report Q3 2026.pdf",
    "options": {"output_dir": "/Users/example/Documents/Markitai/Quarterly Report Q3 2026",
                "llm": true, "ocr": false, "screenshot": false, "alt": false, "desc": false}}
   ```
   `config` is omitted when `options.config` is `user`. It is `{}` when it is `builtin`.
3. Call `convert_json` on a worker thread.
4. Map the envelope. `ok` with `skip_reason` null is Done. `ok` with any
   `skip_reason` is Failed with code `skipped`. An error envelope is Failed with
   its `code` and `message`; `markitai-app` answers `image_only` for an image
   with no text when neither AI nor OCR is on (the core's bindings say
   `conversion_error`).
5. On Failed, delete the claimed folder and everything in it. The app created
   that folder for this item a moment earlier, and nothing else writes there.
6. On Done, add a History entry (4.4) and announce `a11y.done`. On Failed,
   announce `a11y.failed`.

Rules:

| Topic | Rule |
|---|---|
| Title | A file's title is its file name. A URL's title is `result.frontmatter.title` once Done, if it is a non-empty string (trimmed, at most 200 characters). Before that, and as a fallback, the URL without its scheme |
| Duplicates | Adding a source that is already Queued or Converting adds nothing and shows `notice.duplicate`. A source that is Done or Failed converts again as a new item |
| Order | Newest first. Items added together keep the order they were given. Processing is oldest first |
| Selection | Single selection. The first item added to an empty list is selected. Otherwise selection moves only when the user moves it, or when the selected item is removed (the next row, else the previous) |
| Remove | Allowed for Queued, Done and Failed. Converting items cannot be removed (`convert.remove.busy` as the disabled item's tooltip). Removing never touches files or History |
| Clear Finished | Removes every Done and Failed item from the list |
| Retry | Failed items only. Captures the current settings, then Queued |
| Settings changes | Apply to items added later and to retries. Queued items keep what they captured |

### 4.2 Result fields the apps use

| Use | Field |
|---|---|
| Markdown shown, copied and saved | `llm_markdown`, else `markdown` |
| File revealed and saved | `llm_output_path`, else `output_path` |
| Duration | `duration`, seconds |
| Warnings | `warnings`, shown as the core wrote them (English in both languages) |
| AI ran | `usage.requests > 0` |
| Tokens | `usage.input_tokens + usage.output_tokens` |
| Cost | `complete` when every `usage.by_model` row has `"cost_status": "complete"`: show `cost_usd`. `partial` when some row is not complete and `cost_usd > 0`: show `footer.cost.at_least`. Otherwise `unknown`: show `footer.cost.unknown` |
| URL title | `frontmatter.title` |

Formats. Durations below 60 s use one decimal (`18.6 s`, `18.6 秒`), and longer
ones use `unit.minutes`. Token counts use the locale's digit grouping. Costs are
USD: four decimals below $1 (`$0.0412`), two from $1.

### 4.3 Output folder

| Platform | Default (`output.folder` is null) |
|---|---|
| macOS | `FileManager` `.documentDirectory` + `Markitai` |
| Windows | Known folder `FOLDERID_Documents` (follows OneDrive redirection) + `Markitai` |
| Linux | `glib::user_special_dir(UserDirectory::Documents)` + `Markitai`, else `$HOME/Documents/Markitai` |

Each conversion writes into its own folder under the output folder (3.3). A folder holds
`<source name>.md`, `<source name>.llm.md` when AI ran, and `.markitai/` with
`assets/` and `screenshots/`. Paths shown in the UI abbreviate the home folder as
`~` on macOS and Linux. Windows shows full paths.

### 4.4 History

| Platform | File |
|---|---|
| macOS | `~/Library/Application Support/dev.markitai.Markitai/history.json` |
| Windows | `%LOCALAPPDATA%\Markitai\history.json` |
| Linux | `$XDG_DATA_HOME/dev.markitai.Markitai/history.json` (`~/.local/share/...`) |

```json
{
  "version": 1,
  "entries": [
    {
      "id": "6f1c2a9e-3b4d-4c55-9a8e-2f0b7d1e4c3a",
      "source": "/Users/example/Downloads/Quarterly Report Q3 2026.pdf",
      "kind": "file",
      "title": "Quarterly Report Q3 2026.pdf",
      "folder": "/Users/example/Documents/Markitai/Quarterly Report Q3 2026",
      "markdown_path": "/Users/example/Documents/Markitai/Quarterly Report Q3 2026/Quarterly Report Q3 2026.pdf.llm.md",
      "finished_at": "2026-11-04T02:42:18Z",
      "duration": 18.6,
      "warnings": [
        "PDF page 7: some vector charts could not be isolated safely; those labels are retained without a chart image.",
        "PDF page 12: 3 text item(s) drawn outside the visible page area (CropBox) were left out; a viewer does not show them."
      ],
      "usage": {"requests": 6, "input_tokens": 19870, "output_tokens": 4440, "cost_usd": 0.0412, "cost": "complete"},
      "options": {"llm": true, "ocr": false, "screenshot": false, "alt": false, "desc": false, "config": "user"},
      "markitai_version": "2.3.2"
    }
  ]
}
```

| Field | Type | Rule |
|---|---|---|
| `version` | integer | `1` |
| `entries` | array | Newest first, at most 500 |
| `id` | string | UUID v4 of the item |
| `source` | string | Absolute path or URL, as converted |
| `kind` | `"file"` or `"url"` | |
| `title` | string | The item's title when Done |
| `folder` | string | The claimed folder |
| `markdown_path` | string | `llm_output_path`, else `output_path` |
| `finished_at` | string | UTC, RFC 3339, seconds |
| `duration` | number | Seconds |
| `warnings` | array of strings | As returned |
| `usage` | object or null | Null when `usage.requests` is 0. `cost` is `complete`, `partial` or `unknown` (4.2) |
| `options` | object | The captured options |
| `markitai_version` | string | `markitai_app_version()` at conversion time |

| Operation | Effect |
|---|---|
| Add | Insert at the front. Over 500, drop the oldest entries |
| Remove from History | Delete that entry from the list. Files stay |
| Clear History | `entries` becomes empty, after the `history.clear.*` confirmation. Files stay |
| Write | After every change: write `history.json.tmp` in the same folder, flush, rename over `history.json` |
| Unreadable file | Invalid JSON, or a `version` other than 1: rename it to `history-unreadable-<UTC yyyyMMddTHHmmssZ>.json`, start empty, show `history.reset` once |
| Missing Markdown | When the selected entry's `markdown_path` is gone, the detail shows `detail.missing.*` and only Remove from History is enabled |

Each app is single-instance (7.8), so one process writes the file.

### 4.5 Settings

| Key | Type | Default | Meaning |
|---|---|---|---|
| `conversion.llm` | bool | `false` | Enhance with AI |
| `conversion.ocr` | bool | `false` | OCR |
| `conversion.screenshot` | bool | `false` | Screenshots |
| `conversion.desc` | bool | `false` | Image descriptions. Sent only with `llm` |
| `conversion.alt` | bool | `false` | Alt text. Sent only with `llm` |
| `output.folder` | string or null | `null` | Absolute path. Null means the platform default (4.3) |
| `configuration.source` | `"user"` or `"builtin"` | `"user"` | `user` omits `config`. `builtin` sends `config: {}` |
| `view.zoom` | number, 0.5 to 3.0 | `1.0` | Preview zoom and Markdown text size, both tabs |
| `window.width`, `window.height` | number | `1200`, `780` | Windows and Linux only |
| `window.maximized` | bool | `false` | Windows and Linux only |

| Platform | Storage |
|---|---|
| macOS | `UserDefaults` (domain `dev.markitai.Markitai`) with the keys above. The window frame uses the scene's own restoration |
| Windows | `%LOCALAPPDATA%\Markitai\settings.json`, one flat JSON object with the keys above. Unpackaged apps have no `ApplicationData` |
| Linux | `$XDG_CONFIG_HOME/dev.markitai.Markitai/settings.json`, same shape |

Writes use the temporary-file-and-rename rule from 4.4. A missing or unreadable
file means defaults.

### 4.6 App folders

| Platform | Data folder (`data_dir`, working directory) | Other |
|---|---|---|
| macOS | `~/Library/Application Support/dev.markitai.Markitai` | none |
| Windows | `%LOCALAPPDATA%\Markitai` | `WebView2\` (user data folder), `Preview\` (preview documents) |
| Linux | `$XDG_DATA_HOME/dev.markitai.Markitai` | none |

## 5. Information architecture

| Place | macOS | Windows | GNOME |
|---|---|---|---|
| Convert | Sidebar item, first | `NavigationView` item, first | Sidebar row, first |
| History | Sidebar item, second | `NavigationView` item, second | Sidebar row, second |
| Settings | Settings window (`Settings` scene, app menu `menu.settings`, ⌘,) | `NavigationView` footer item `nav.settings`, shown as a page. Set `IsSettingsVisible="False"` and add the item to `FooterMenuItems`, so its label follows the app's language rather than the system's | `AdwPreferencesDialog` from the primary menu item `nav.preferences` |
| About | App menu `menu.about` (standard panel) and Settings > About | Settings > About | Primary menu `menu.about` (`AdwAboutDialog`) and Preferences > About |
| Start page | Convert, always | Convert, always | Convert, always |

Settings has the same four groups in the same order everywhere: Conversion,
Output, Configuration, About.

## 6. Screens

### 6.1 Main window regions

| Region | macOS | Windows | GNOME |
|---|---|---|---|
| Window | `Window` scene (one window, no ⌘N), unified toolbar | `Window` with `ExtendsContentIntoTitleBar`, tall title bar (48), `MicaBackdrop` | `AdwApplicationWindow` |
| Navigation | `NavigationSplitView` sidebar, `List` with `.sidebar` style, 200 wide | `NavigationView`, `PaneDisplayMode="Auto"`, `OpenPaneLength="200"`, back button visible only in stacked layout | Outer `AdwNavigationSplitView`, sidebar with its own `AdwHeaderBar` (title `app.name`, primary menu button), `GtkListBox` with `.navigation-sidebar` |
| List column | `HSplitView` left pane, 320 wide, resizable 260 to 420 | `Grid` column 320 wide | Inner `AdwNavigationSplitView` sidebar, `sidebar-width-fraction` 0.32, `min-sidebar-width` 260, `max-sidebar-width` 420 |
| Detail | `HSplitView` right pane | `Grid` remaining column | Inner content page |
| Column header | Toolbar title (`.navigationTitle`) over the list, actions to its right | Page title (Title style, 28) above the list column, actions in a `CommandBar` to its right | List pane `AdwHeaderBar` with the page name as title and actions at the end |
| Detail bar | Toolbar items over the detail: view switch centered (`.principal`), actions trailing | A 48 high row: view switch at the left, `CommandBar` (`DefaultLabelPosition="Right"`) at the right, the Windows place for a selector beside commands | Detail `AdwHeaderBar`: `AdwToggleGroup` as the centered title widget, actions packed at the end |

### 6.2 Convert, empty

When the Convert list is empty, the list column and the detail are replaced by
one large drop zone, centered in the area right of the navigation.

| Element | Content |
|---|---|
| Drop zone | Rounded rectangle, 1 px dashed border (macOS and GNOME 1.5 px), 560 wide, height fits content, padding 40. While files are dragged over the window, the border turns solid in the accent color and the label reads `convert.drop.release` |
| Glyph | "Add files" icon (13.6), 48 px, secondary color |
| Title | `convert.empty.title` |
| Body | `convert.empty.body` with the output folder path |
| Primary button | `convert.choose`, the platform's accent (prominent) button. Gets initial focus |
| Divider | `convert.or`, between hairlines |
| URL row | Text field labeled `convert.url.label` (label visually hidden, placeholder `convert.url.placeholder`), then a standard button `convert.url.add`. Enter adds |
| Error | `convert.url.invalid` below the field, in primary text with the error glyph, until the text changes |
| Notices | Below the URL row |

GNOME builds this as an `AdwStatusPage` inside the drop zone frame: icon, title
(`.title-1`), description, then a `.pill.suggested-action` button and the URL row.

### 6.3 Convert, with items

The list column holds, top to bottom:

| Element | Content |
|---|---|
| Compact drop bar | One row, 48 high, dashed border, radius per 13.4: "Add files" glyph, `convert.drop.bar`, and a standard button `convert.choose` at the right. Drag highlight as in 6.2 |
| URL row | Text field (hidden label `convert.url.label`, placeholder `convert.url.placeholder`, leading link glyph) and an `convert.url.add` button |
| Notice | Platform notice (6.7) |
| List | Rows as in 4.1. Accessible name `a11y.conversions` |

Column header action: `convert.clear_finished` (enabled when any item is Done or Failed).

List row anatomy, the same on every platform:

| Part | Content |
|---|---|
| Leading | Kind glyph, 16 px: file or web page (13.6) |
| Title | One line, middle truncation for file names, tail truncation for URLs |
| Subtitle | One line, tail truncation: the state text from 4.1 |
| Trailing | Queued glyph, spinner, warning glyph (Done with warnings) or error glyph. Nothing for Done without warnings |
| Accessible name | `<title>, <subtitle>` |
| Context menu | Done: `detail.copy`, `detail.save_as`, `detail.show.*`, separator, `convert.remove`. Failed: `detail.retry`, separator, `convert.remove`. Queued: `convert.remove`. Converting: `convert.remove` disabled |

### 6.4 Detail

| Part | Done | Converting | Queued | Failed | No selection |
|---|---|---|---|---|---|
| Detail bar | View switch `detail.preview` and `detail.markdown` (accessible name `detail.view`), actions `detail.copy`, `detail.save_as`, `detail.show.*` | empty | empty | empty (Retry is in the error block) | empty |
| Header | Title (detail title style), source line (kind glyph, abbreviated path or full URL, selectable), status line (done glyph, `state.done`) | Title, source, spinner and `detail.converting` | Title, source, queued glyph and `state.queued` | Title, source, error glyph and `state.failed` | none |
| Warnings | When `warnings` is not empty: a disclosure whose header is the warning glyph and `state.warnings.*`, with the platform's disclosure chevron. Expanded, it lists each warning as its own selectable line. Collapsed by default, state kept per item for the session | none | none | none | none |
| Body | Preview (web view) or Markdown text, filling the rest | `detail.converting.note` | `detail.queued` | Error block: mapped sentence (primary text, body size), action buttons, then `detail.details` with the raw `message` in a selectable monospace box | `detail.empty`, centered |
| Footer | `footer.duration`, then ` · ` `footer.tokens` and ` · ` cost when AI ran. Caption style, one line | none | none | none | none |

History entries use the Done column. Their status line is `state.done`, then
` · ` and the finish date as in History rows (6.5). The Markdown is read from
`markdown_path` off the main thread.

Platform controls for these parts:

| Part | macOS | Windows | GNOME |
|---|---|---|---|
| View switch | `Picker` with `.segmented` style | `Segmented` (CommunityToolkit) | `AdwToggleGroup` with two text toggles |
| Actions | Toolbar icon buttons with tooltips | `AppBarButton`s with icon and label | Flat icon `GtkButton`s with tooltips |
| Warnings box | Rounded box (radius 12, `.quaternary` fill) with a `DisclosureGroup` | `Expander` (header glyph and count) | `AdwExpanderRow` in a `.boxed-list` |
| Preview | `WKWebView` in `NSViewRepresentable` | `WebView2` | `webkit6::WebView` |
| Markdown text | `NSTextView`: not editable, selectable, monospaced, word wrap, insets 20 | `TextBox`: `IsReadOnly`, `TextWrapping="Wrap"`, `AcceptsReturn`, Cascadia Mono, padding 24 | `GtkTextView`: not editable, `monospace`, `wrap-mode` word-char, margins 24 |
| Error block | `VStack` in the detail | `StackPanel` in the detail | `AdwStatusPage`-like box: `.title-4` sentence, buttons, `.card` with the raw message |
| Footer | Text in a bottom bar, `.secondary` | `TextBlock` with `CaptionTextBlockStyle` | `AdwToolbarView` bottom bar with a `.caption` label |

### 6.5 History

| Element | Content |
|---|---|
| Column header | `nav.history` title, action `history.clear` (disabled when empty) |
| Search | Field labeled `history.search` at the top of the list column. Filters by title and source, case-insensitive substring, as you type. macOS `NSSearchField` through `NSViewRepresentable`, Windows `AutoSuggestBox`, GNOME `GtkSearchEntry` |
| Rows | Kind glyph, title, subtitle = finish date (`date.today`, `date.yesterday` or the platform's medium date and short time) and, with warnings, ` · ` `state.warnings.*`. Warning glyph trailing |
| Context menu | `detail.copy`, `detail.save_as`, `detail.show.*`, separator, `history.remove` |
| Empty | `history.empty.title`, `history.empty.body`, centered in the list and detail area |
| No matches | `history.no_results` in the list column |
| Detail | 6.4, Done column |

### 6.6 Settings

| Group | Row | macOS (`Form`, `.formStyle(.grouped)`, window 520 wide) | Windows (page, `SettingsCard`, content max width 1000) | GNOME (`AdwPreferencesDialog`, one page, `search-enabled` false) |
|---|---|---|---|---|
| Conversion | `settings.llm` + note | `Toggle` with title and caption | `SettingsCard` + `ToggleSwitch` with empty `OnContent` and `OffContent` (the default "On" and "Off" follow the system language, not the app's) | `AdwSwitchRow` (title, subtitle) |
| | `settings.ocr` + note | same | same | same |
| | `settings.screenshot` + note | same | same | same |
| | `settings.desc` + note | same, disabled while `settings.llm` is off | same | same |
| | `settings.alt` + note | same | same | same |
| Output | `settings.output.folder`, note, path | `LabeledContent` with path, `settings.output.change`, `settings.output.show` | `SettingsCard`, description = path, two buttons | `AdwActionRow` (subtitle = path) with a `settings.output.change` button and a folder icon button (tooltip `detail.show.gnome`) |
| Configuration | Choice `settings.config.user` / `settings.config.builtin` with notes | `Picker` with `.radioGroup` style | `SettingsCard` holding two `RadioButton`s | Two `AdwActionRow`s with grouped `GtkCheckButton` prefixes |
| | `settings.config.open` or `settings.config.create` | Button | Clickable `SettingsCard` (`IsClickEnabled`, chevron action icon) | `AdwButtonRow` |
| About | App icon, `app.name`, `settings.about.app` + version, `settings.about.core` + version, `settings.about.copyright` | Rows in a section | `SettingsExpander` (expanded) | `AdwActionRow`s with `.property` style |

The Configuration note under `settings.config.user` is the selected file's path,
or `settings.config.user.missing` when it does not exist. `settings.config.create`
writes `{}` and a newline to that path (creating folders), then opens it.
`settings.config.open` opens the file in its default app. Choosing a folder that
cannot be written (a test file cannot be created and deleted) shows
`settings.output.unwritable` and keeps the old folder.

### 6.7 Dialogs and notices

| Need | macOS | Windows | GNOME |
|---|---|---|---|
| Quit with running conversions | `NSAlert` (sheet on the main window, app-modal when it is closed). `quit.title`, `quit.body.*` + `quit.body.queued` when items are Queued. Buttons `quit.confirm`, `common.cancel` (default) | `ContentDialog`, same copy, `DefaultButton="Close"` | `AdwAlertDialog`, same copy, `quit.confirm` with `ADW_RESPONSE_DESTRUCTIVE`, default `common.cancel` |
| Clear History | `.confirmationDialog`, `history.clear.title.*`, `history.clear.body`, `history.clear.confirm` (destructive role), `common.cancel` | `ContentDialog`, same | `AdwAlertDialog`, same |
| Notice (folder refused, duplicate, history reset) | Inline line below the URL row: warning glyph and text, cleared after 8 s or the next add | `InfoBar` (`Severity="Warning"`, closable) at the top of the list column, cleared after 8 s or the next add | `AdwToast`, 5 s |
| Copied | The Copy button shows a checkmark glyph for 1.5 s | The Copy button shows `Accept` and `notice.copied` for 1.5 s | `AdwToast` `notice.copied`, 2 s |
| Fatal (ABI mismatch) | Window with only `fatal.abi.title`, `fatal.abi.body`, `quit.confirm` | same | same |

## 7. Behavior

### 7.1 Adding sources

| Way | Rule |
|---|---|
| Drop on the window | Accepted on the Convert and History pages (History switches to Convert). Files become items. Web links (a URL drag) become URL items |
| `convert.choose` | The platform's open dialog, multiple selection, files only, no type filter |
| URL field | Trim. With no scheme and a dot in the text, prefix `https://`. Accept only `http` and `https` URLs with a host. Clear the field after adding |
| Paste (7.6 shortcut) | When focus is not in a text field: file references on the clipboard become items, else text that is one valid URL becomes a URL item |
| Open with the app | Files opened through the Dock icon (macOS), command-line arguments (Windows) or `GApplication::open` (Linux) are added |
| Folders | Refused before any core call, with `notice.folder` or `notice.folders`. Exception: a directory whose name ends in `.numbers` (any case) is a Numbers document and is accepted |
| Type filter | None. The core reports `unsupported` |

### 7.2 Copy, Save As, Show

| Action | Rule |
|---|---|
| Copy Markdown | Puts the shown Markdown on the clipboard as plain text |
| Save As… | Save dialog, suggested name = the folder's label + `.md`. Writes the shown Markdown. If the text contains `.markitai/`, it also copies the conversion's `.markitai` folder next to the saved file, adding files that are not there and never replacing a file |
| Show in Finder, File Explorer, Files | macOS `NSWorkspace.activateFileViewerSelecting`, Windows `explorer.exe /select,"<file>"`, GNOME `GtkFileLauncher::open_containing_folder` |

### 7.3 Quit and closing the window

| Platform | Rule |
|---|---|
| macOS | Closing the window does not quit. Conversions keep running and the Dock icon reopens the same window. ⌘Q with any item Converting returns `.terminateLater` and shows the quit dialog |
| Windows | Closing the window quits. With any item Converting, `AppWindow.Closing` is canceled and the quit dialog shows. Quit closes the app |
| GNOME | Closing the window quits. `close-request` returns stop when any item is Converting and shows the quit dialog. Ctrl+Q does the same |

Queued items exist only while two items are Converting, so the dialog's
condition is "any item Converting".

After the user confirms Quit, the app deletes the folders it claimed for items
still Converting (their conversions end with the process), then exits. A
conversion that finishes between the dialog and the quit is kept and added to
History as usual.

### 7.4 Appearance

The apps follow the system's light or dark setting and have no theme setting.
On a change, the Preview is rendered again with the new `theme`.

### 7.5 Language

| Rule | Detail |
|---|---|
| Choice | If the first preferred UI language starts with `zh`, the app uses zh-Hans. Otherwise English |
| Source | macOS `Locale.preferredLanguages`, Windows `Windows.System.UserProfile.GlobalizationPreferences.Languages`, Linux `glib::language_names()` |
| Strings | `native/strings.json`, generated from section 10 by `native/scripts/strings.py`, which also fails when the two differ. Each app embeds the JSON and looks keys up at run time |
| Shape | `{"<key>": {"en": "...", "zh-Hans": "..."}}`. Placeholders are `{name}`. Plural pairs end in `.one` and `.other`, chosen by count == 1 |
| Not translated | The core's warnings and raw error messages, file names, URLs |

### 7.6 Single instance

| Platform | Rule |
|---|---|
| macOS | One process by default. One `Window` scene |
| Windows | `AppInstance.FindOrRegisterForKey("dev.markitai.Markitai")`. A second launch redirects its activation (and file arguments) to the first and exits |
| GNOME | `GApplication` uniqueness with `HANDLES_OPEN` |

## 8. Preview web view

| Concern | macOS `WKWebView` | Windows `WebView2` | GNOME WebKitGTK 6 |
|---|---|---|---|
| Load | `loadHTMLString(html, baseURL:)` with the file URL of `base_dir` (trailing slash), which grants read access to that folder | Write the document to `%LOCALAPPDATA%\Markitai\Preview\current.html`, then `Navigate` to its file URI | `load_html(html, Some(<file URI of base_dir>))` |
| Scripts | `WKWebpagePreferences.allowsContentJavaScript = false` | `CoreWebView2.Settings.IsScriptEnabled = false` | `WebKitSettings` `enable-javascript` false |
| Network | CSP, plus a `WKContentRuleList` that blocks `^https?://` and `^wss?://`. `websiteDataStore` `.nonPersistent()` | CSP, plus `AddWebResourceRequestedFilter("*", All)` answering every non-`file:` request with 403 | CSP, plus an ephemeral `WebKitNetworkSession` |
| Navigation | Allow the initial load and same-document fragment links. Open `http`, `https` and `mailto` links with `NSWorkspace.open` and cancel. Cancel all else | `NavigationStarting` and `NewWindowRequested` with the same rule, opening with `Launcher.LaunchUriAsync` | `decide-policy` with the same rule, opening with `GtkUriLauncher` |
| Context menu | Only Copy | Only Copy (`ContextMenuRequested`) | Only Copy (`context-menu`) |
| Developer tools | Off | `AreDevToolsEnabled = false`, `AreBrowserAcceleratorKeysEnabled = false`, `IsStatusBarEnabled = false` | `enable-developer-extras` false |
| Background | Transparent (`drawsBackground` false), so the pane's own background shows | `DefaultBackgroundColor` transparent | `set_background_color` transparent |
| Zoom | `pageZoom = view.zoom` | `ZoomFactor = view.zoom` | `zoom_level = view.zoom` |
| Missing runtime | n/a | When `CoreWebView2` fails to start, the Preview tab shows `preview.webview2_missing` and the Markdown tab works | n/a |
| Accessible name | `a11y.preview` | `a11y.preview` | `a11y.preview` |

## 9. Errors

The Failed detail shows the mapped sentence, then the buttons, then the raw
message under `detail.details`. `detail.retry` is always the first button.

| Code | Sentence key | Second button |
|---|---|---|
| `not_found` | `error.not_found` | `convert.remove` |
| `is_directory` | `error.is_directory` | `convert.remove` |
| `no_model_configured` | `error.no_model_configured` | `action.open_settings` |
| `invalid_input` | `error.invalid_input` | none |
| `unsupported` | `error.unsupported` | `convert.remove` |
| `conversion_error` | `error.conversion_error` | none |
| `fetch_error` | `error.fetch_error` | none |
| `config_error` | `error.config_error` | `settings.config.open` |
| `io_error` | `error.io_error` | `action.open_settings` |
| `output_unavailable` (app) | `error.output_unavailable` | `action.open_settings` |
| `image_only` (`markitai-app`: an image with no text and neither AI nor OCR on) | `error.image_only` | `action.open_settings` |
| `skipped` (app, from another `skip_reason`) | `error.skipped` | none |
| `invalid_json`, `internal_error` | `error.internal` | none |
| any other code | `error.unknown` | none |

`{settings}` in copy is `nav.settings` on macOS and Windows and
`nav.preferences` on GNOME. `action.open_settings` opens the Settings window,
the Settings page or the Preferences dialog.

## 10. Copy

Every user-visible string. The table is the source of `native/strings.json`.

| Key | en | zh-Hans |
|---|---|---|
| `app.name` | Markitai | Markitai |
| `nav.convert` | Convert | 转换 |
| `nav.history` | History | 历史记录 |
| `nav.settings` | Settings | 设置 |
| `nav.preferences` | Preferences | 首选项 |
| `menu.settings` | Settings… | 设置… |
| `menu.about` | About Markitai | 关于 Markitai |
| `menu.main` | Main Menu | 主菜单 |
| `menu.add_url` | Add Web Address | 添加网址 |
| `menu.toggle_view` | Switch Preview and Markdown | 切换预览与 Markdown |
| `menu.zoom_in` | Zoom In | 放大 |
| `menu.zoom_out` | Zoom Out | 缩小 |
| `menu.actual_size` | Actual Size | 实际大小 |
| `convert.empty.title` | Drop files to convert | 拖入文件即可转换 |
| `convert.empty.body` | PDF, Word, PowerPoint, Excel, images, web pages and more become Markdown in {folder}. | PDF、Word、PowerPoint、Excel、图片、网页等都会转成 Markdown，保存到 {folder}。 |
| `convert.choose` | Choose Files… | 选择文件… |
| `convert.or` | or | 或 |
| `convert.url.label` | Web address | 网址 |
| `convert.url.placeholder` | https://example.com/article | https://example.com/article |
| `convert.url.add` | Add | 添加 |
| `convert.url.invalid` | Enter a web address that starts with http:// or https://. | 请输入以 http:// 或 https:// 开头的网址。 |
| `convert.drop.bar` | Drop files here | 将文件拖到这里 |
| `convert.drop.release` | Release to add | 松开即可添加 |
| `convert.clear_finished` | Clear Finished | 清除已完成 |
| `convert.remove` | Remove from List | 从列表中移除 |
| `convert.remove.busy` | A conversion can't be stopped once it starts. | 转换开始后无法停止。 |
| `notice.folder` | “{name}” is a folder. Add the files inside it instead. | “{name}”是文件夹。请添加其中的文件。 |
| `notice.folders` | {count} folders weren't added. Add the files inside them instead. | {count} 个文件夹未添加。请添加其中的文件。 |
| `notice.duplicate` | “{name}” is already waiting or converting. | “{name}”已在等待或转换中。 |
| `notice.copied` | Copied | 已复制 |
| `state.queued` | Queued | 排队中 |
| `state.converting` | Converting… | 正在转换… |
| `state.done` | Done in {duration} | 已完成，用时 {duration} |
| `state.failed` | Failed | 失败 |
| `state.warnings.one` | 1 warning | 1 条警告 |
| `state.warnings.other` | {count} warnings | {count} 条警告 |
| `row.failed` | Failed · {reason} | 失败 · {reason} |
| `detail.empty` | Select a conversion to see its Markdown. | 选择一个转换以查看 Markdown。 |
| `detail.queued` | Waiting for a free slot. Markitai converts two files at a time. | 正在等待空位。Markitai 每次同时转换两个文件。 |
| `detail.converting` | Converting… {elapsed} | 正在转换… {elapsed} |
| `detail.converting.note` | A conversion can't be stopped once it starts. Quitting now loses its result. | 转换开始后无法停止。现在退出会丢失结果。 |
| `detail.view` | View | 视图 |
| `detail.preview` | Preview | 预览 |
| `detail.markdown` | Markdown | Markdown |
| `detail.copy` | Copy Markdown | 复制 Markdown |
| `detail.save_as` | Save As… | 另存为… |
| `detail.show.macos` | Show in Finder | 在访达中显示 |
| `detail.show.windows` | Show in File Explorer | 在文件资源管理器中显示 |
| `detail.show.gnome` | Show in Files | 在“文件”中显示 |
| `detail.more` | More Actions | 更多操作 |
| `detail.retry` | Retry | 重试 |
| `detail.details` | Details | 详细信息 |
| `detail.missing.title` | The Markdown file is missing | 找不到 Markdown 文件 |
| `detail.missing.body` | It is no longer at {path}. It may have been moved or deleted. | 它已不在 {path}，可能已被移动或删除。 |
| `footer.duration` | Converted in {duration} | 转换用时 {duration} |
| `footer.tokens` | {count} tokens | {count} 个 token |
| `footer.cost.at_least` | at least {cost} | 至少 {cost} |
| `footer.cost.unknown` | cost unknown | 费用未知 |
| `unit.seconds` | {value} s | {value} 秒 |
| `unit.minutes` | {minutes} min {seconds} s | {minutes} 分 {seconds} 秒 |
| `date.today` | Today, {time} | 今天 {time} |
| `date.yesterday` | Yesterday, {time} | 昨天 {time} |
| `history.search` | Search History | 搜索历史记录 |
| `history.empty.title` | No conversions yet | 还没有转换记录 |
| `history.empty.body` | Finished conversions appear here. Removing an entry keeps its files. | 完成的转换会显示在这里。移除记录不会删除文件。 |
| `history.no_results` | No results for “{query}” | 没有与“{query}”匹配的结果 |
| `history.remove` | Remove from History | 从历史记录中移除 |
| `history.clear` | Clear History… | 清除历史记录… |
| `history.clear.title.one` | Clear the 1 entry? | 清除这 1 条记录？ |
| `history.clear.title.other` | Clear all {count} entries? | 清除全部 {count} 条记录？ |
| `history.clear.body` | Your converted files stay in the output folder. | 转换后的文件仍保留在输出文件夹中。 |
| `history.clear.confirm` | Clear | 清除 |
| `history.reset` | History couldn't be read, so Markitai started a new one. The old file is kept as {name}. | 无法读取历史记录，已重新开始。旧文件保留为 {name}。 |
| `settings.conversion` | Conversion | 转换 |
| `settings.llm` | Enhance with AI | AI 增强 |
| `settings.llm.note` | A model from your markitai configuration tidies the Markdown. Providers may charge for requests. | 由 markitai 配置中的模型整理 Markdown。服务商可能按请求收费。 |
| `settings.ocr` | OCR | OCR 文字识别 |
| `settings.ocr.note` | Reads scans and images on this computer with the OCR settings in your markitai configuration. The first use downloads its models. | 按 markitai 配置中的 OCR 设置在本机识别扫描件和图片。首次使用会下载模型。 |
| `settings.screenshot` | Screenshots | 页面截图 |
| `settings.screenshot.note` | Saves an image of each page. Office files need LibreOffice. Web pages need Chrome or Chromium. | 保存每一页的图像。Office 文件需要 LibreOffice，网页需要 Chrome 或 Chromium。 |
| `settings.desc` | Image descriptions | 图片描述 |
| `settings.desc.note` | A vision model from your markitai configuration describes each image. Needs Enhance with AI. | 由 markitai 配置中的视觉模型描述每张图片。需要开启 AI 增强。 |
| `settings.alt` | Alt text | 替代文本 |
| `settings.alt.note` | A vision model from your markitai configuration writes alt text for each image. Needs Enhance with AI. | 由 markitai 配置中的视觉模型为每张图片写替代文本。需要开启 AI 增强。 |
| `settings.output` | Output | 输出 |
| `settings.output.folder` | Output folder | 输出文件夹 |
| `settings.output.note` | Each conversion gets its own folder here. | 每次转换都会在这里新建一个文件夹。 |
| `settings.output.change` | Change… | 更改… |
| `settings.output.show` | Show | 显示 |
| `settings.output.unwritable` | Markitai can't write to that folder. Choose another one. | Markitai 无法写入该文件夹，请另选一个。 |
| `settings.config` | Configuration | 配置 |
| `settings.config.user` | Use my markitai configuration | 使用我的 markitai 配置 |
| `settings.config.user.missing` | No configuration file yet. Built-in defaults apply until you create one. | 还没有配置文件。创建之前使用内置默认值。 |
| `settings.config.builtin` | Use built-in defaults | 使用内置默认值 |
| `settings.config.builtin.note` | Ignores configuration files. MODEL and provider keys in your environment still apply. | 忽略配置文件。环境中的 MODEL 和服务商密钥仍然有效。 |
| `settings.config.open` | Open Configuration File | 打开配置文件 |
| `settings.config.create` | Create Configuration File | 创建配置文件 |
| `settings.about` | About | 关于 |
| `settings.about.app` | App version | 应用版本 |
| `settings.about.core` | markitai version | markitai 版本 |
| `settings.about.copyright` | Copyright © 2026 Ynewtime | Copyright © 2026 Ynewtime |
| `quit.title` | Quit anyway? | 仍要退出吗？ |
| `quit.body.one` | 1 conversion is still running. If you quit now, it stops and its result is lost. | 还有 1 个转换正在进行。现在退出会中止它，结果将丢失。 |
| `quit.body.other` | {count} conversions are still running. If you quit now, they stop and their results are lost. | 还有 {count} 个转换正在进行。现在退出会中止它们，结果将丢失。 |
| `quit.body.queued` | Files waiting in the queue are not converted. | 排队中的文件也不会被转换。 |
| `quit.confirm` | Quit | 退出 |
| `common.cancel` | Cancel | 取消 |
| `common.back` | Back | 返回 |
| `action.open_settings` | Open {settings} | 打开{settings} |
| `error.not_found` | The file is no longer there. It may have been moved or deleted. | 文件已不在原处，可能已被移动或删除。 |
| `error.is_directory` | This is a folder. Add the files inside it instead. | 这是文件夹。请添加其中的文件。 |
| `error.no_model_configured` | Enhance with AI needs a model. Add one to your markitai configuration, or turn off Enhance with AI. | AI 增强需要模型。请在 markitai 配置中添加模型，或关闭 AI 增强。 |
| `error.invalid_input` | Markitai couldn't read this input. | Markitai 无法读取此输入。 |
| `error.unsupported` | Markitai can't convert this kind of file. | Markitai 无法转换此类文件。 |
| `error.conversion_error` | The document couldn't be converted. Check that it opens in its own app. | 无法转换此文档。请确认它能在原应用中正常打开。 |
| `error.fetch_error` | Couldn't download this page. Check the address and your internet connection. | 无法下载此网页。请检查网址和网络连接。 |
| `error.config_error` | Your markitai configuration has a problem. | markitai 配置有问题。 |
| `error.io_error` | Markitai couldn't read or write a file. Check that the output folder exists, can be written and has free space. | Markitai 无法读写文件。请确认输出文件夹存在、可写入且空间充足。 |
| `error.output_unavailable` | Markitai couldn't create a folder in {folder}. Choose another output folder in {settings}. | 无法在 {folder} 中创建文件夹。请在{settings}中另选输出文件夹。 |
| `error.image_only` | This image has no text layer. Turn on OCR or Enhance with AI in {settings} to read its text. | 这张图片没有文字层。请在{settings}中开启 OCR 或 AI 增强以识别其中的文字。 |
| `error.skipped` | Markitai skipped this file ({reason}). | Markitai 跳过了此文件（{reason}）。 |
| `error.internal` | Markitai hit an internal error. Retry, and report the details if it happens again. | Markitai 遇到内部错误。请重试；如再次出现，请报告详细信息。 |
| `error.unknown` | The conversion failed. | 转换失败。 |
| `error.save_failed` | The file couldn't be saved. | 无法保存文件。 |
| `error.open_failed` | Markitai couldn't open {name}. | Markitai 无法打开 {name}。 |
| `preview.webview2_missing` | The preview needs the Microsoft Edge WebView2 Runtime. The Markdown tab still works. | 预览需要 Microsoft Edge WebView2 运行时。Markdown 标签页仍可使用。 |
| `fatal.abi.title` | This copy of Markitai is incomplete | 这份 Markitai 不完整 |
| `fatal.abi.body` | Its conversion library doesn't match the app. Reinstall Markitai. | 它的转换库与应用不匹配。请重新安装 Markitai。 |
| `a11y.drop_zone` | Drop files or web links here to convert them. | 将文件或网页链接拖到这里进行转换。 |
| `a11y.conversions` | Conversions | 转换列表 |
| `a11y.preview` | Markdown preview | Markdown 预览 |
| `a11y.markdown_text` | Markdown text | Markdown 文本 |
| `a11y.warnings` | Warnings | 警告 |
| `a11y.done` | {title} converted | {title} 已转换 |
| `a11y.failed` | {title} failed | {title} 转换失败 |
| `plist.documents` | Markitai saves converted Markdown in your Documents folder. | Markitai 会把转换后的 Markdown 保存到“文稿”文件夹。 |
| `plist.desktop` | Markitai reads the files you add from your Desktop. | Markitai 会读取你从桌面添加的文件。 |
| `plist.downloads` | Markitai reads the files you add from your Downloads folder. | Markitai 会读取你从“下载”文件夹添加的文件。 |

## 11. Keyboard shortcuts

| Action | macOS | Windows | GNOME |
|---|---|---|---|
| Choose Files… | ⌘O | Ctrl+O | Ctrl+O |
| Add Web Address (focus the URL field, switch to Convert) | ⌘L | Ctrl+L | Ctrl+L |
| Add files or a URL from the clipboard (focus not in a text field) | ⌘V | Ctrl+V | Ctrl+V |
| Go to Convert | ⌘1 | Ctrl+1 | Alt+1 |
| Go to History | ⌘2 | Ctrl+2 | Alt+2 |
| Settings, Preferences | ⌘, | Ctrl+, | Ctrl+, |
| Switch Preview and Markdown | ⇧⌘M | Ctrl+Shift+M | Ctrl+Shift+M |
| Copy Markdown | ⇧⌘C | Ctrl+Shift+C | Ctrl+Shift+C |
| Save As… | ⇧⌘S | Ctrl+Shift+S | Ctrl+Shift+S |
| Show in Finder, File Explorer, Files | ⇧⌘R | Ctrl+Shift+R | Ctrl+Shift+R |
| Retry | ⌘R | Ctrl+R | Ctrl+R |
| Remove from List, Remove from History | ⌘⌫ | Delete | Delete |
| Search History | ⌘F | Ctrl+F | Ctrl+F |
| Zoom in, out, actual size | ⌘=, ⌘-, ⌘0 | Ctrl+=, Ctrl+-, Ctrl+0 | Ctrl+plus, Ctrl+minus, Ctrl+0 |
| Back to the list (stacked layout) | ⌘[ or Esc | Esc or Alt+Left | Esc or Alt+Left |
| Move in a list | ↑ ↓ | ↑ ↓ | ↑ ↓ |
| Close the window | ⌘W | Alt+F4 | Ctrl+W |
| Quit | ⌘Q | Alt+F4 (closing quits) | Ctrl+Q |

macOS puts these in the menu bar: File (Choose Files…, Add Web Address, Save
As…, Show in Finder), Edit (Copy Markdown, Paste), View (Convert, History,
Switch Preview and Markdown, zoom items), and the item commands (Retry, Remove
from List). Windows sets `KeyboardAccelerator`s, so tooltips show them. GNOME
registers app actions with `set_accels_for_action`.

## 12. Accessibility

| Topic | Rule |
|---|---|
| Labels | Every icon-only button has an accessible name equal to its tooltip (the copy key of its action). The URL field is labeled `convert.url.label`. The drop zone's name is `a11y.drop_zone`. Lists are named `a11y.conversions` and `nav.history`. The view switch is a group named `detail.view`. The warnings box is named `a11y.warnings` |
| Row names | `<title>, <subtitle>`, such as "Quarterly Report Q3 2026.pdf, Done in 18.6 s · 2 warnings" |
| State changes | Done and Failed are announced politely: macOS `NSAccessibility.post` with `.announcementRequested`, Windows `AutomationPeer.RaiseNotificationEvent`, GNOME `gtk::Accessible::announce` |
| Focus order | Navigation, then the drop bar's Choose Files… button, the URL field, Add, the list, then the detail: view switch, actions, warnings disclosure, content, footer. Empty Convert focuses Choose Files… first |
| Keyboard | Every action has a keyboard path (section 11 or Tab). No action is drag-only |
| Contrast | Text 4.5:1 or more against its background. Status colors appear only on glyphs, always next to text that says the state. Informative text (errors, warnings) is primary text, never secondary |
| Color | States differ by glyph shape as well as color |
| Reduced motion | List insertion and removal fade over 150 ms, and not at all with the platform's reduce-motion setting. No other custom animation. The platforms' spinners are kept |
| Text size | Native text follows the platform. `view.zoom` scales the preview and the Markdown text |
| High contrast | Windows high-contrast themes use system colors: skip the accent override there. GNOME and macOS keep their own contrast modes |

## 13. Visual tokens

The apps take three things from the brand: the ink accent, the app icon, and the
preview's typography and neutrals. Everything else is each platform's own:
window and sidebar materials, controls, fonts, focus rings and selection.

### 13.1 Ink, the brand ramp

Values from the workbench's `crates/markitai-cli/src/server/web/src/styles/app.css`
(`--accent` `#18181b` light and `#fafafa` dark, zinc neutrals).

| Step | 50 | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 950 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Hex | `#fafafa` | `#f4f4f5` | `#e4e4e7` | `#d4d4d8` | `#a1a1aa` | `#71717a` | `#52525b` | `#3f3f46` | `#27272a` | `#18181b` | `#09090b` |

### 13.2 Accent per platform

| Platform | Light: fill, text on fill | Dark: fill, text on fill | How |
|---|---|---|---|
| macOS | 900, white | 500, white | `.tint(Brand.tint)` on each scene's root, with `Brand.tint` an `NSColor(name:dynamicProvider:)`. Never draw text in the tint (no `.borderless` or `.link` buttons in tint), because 500 on a dark window is below 4.5:1. List and sidebar selection stay as macOS draws them |
| Windows | 900 (`SystemAccentColorDark1`), white | 100 (`SystemAccentColorLight2`), black | Override in `App.xaml` after `XamlControlsResources`: `SystemAccentColorLight3` 50, `Light2` 100, `Light1` 300, `SystemAccentColor` 500, `Dark1` 900, `Dark2` 950, `Dark3` `#000000` |
| GNOME | 900, white | 500, white | `style.css`: `:root { --accent-bg-color: #18181b; --accent-fg-color: #ffffff; --accent-color: #18181b; }`. `style-dark.css`: `:root { --accent-bg-color: #71717a; --accent-fg-color: #ffffff; --accent-color: #d4d4d8; }` |

Where the accent shows: the primary button, switches, radio and check marks,
the Windows navigation and list selection pill, and the drag highlight.

### 13.3 Status colors

| Status | macOS | Windows | GNOME |
|---|---|---|---|
| Done (glyph) | `.green` | `SystemFillColorSuccessBrush` | `--success-color` |
| Warning (glyph) | `.orange` | `SystemFillColorCautionBrush` | `--warning-color` |
| Failed (glyph) | `.red` | `SystemFillColorCriticalBrush` | `--error-color` |
| Secondary text | `.secondary` | `TextFillColorSecondaryBrush` | `.dimmed` |

### 13.4 Spacing and radii

Spacing scale: 4, 8, 12, 16, 20, 24, 32, 48.

| Use | macOS | Windows | GNOME |
|---|---|---|---|
| Pane content inset | 20 | 24 | 24 (12 in the list pane) |
| Between related controls | 8 | 8 | 8 |
| Between groups | 20 | 24 | 24 |
| List row height | 44 | 60 | 58 |
| Drop zone and boxes (warnings, raw error) radius | 12 | 8 (`OverlayCornerRadius`) | 12 |
| Buttons and fields radius | system | 4 (`ControlCornerRadius`) | system |
| Window radius | system | system | system |

### 13.5 Type

| Role | macOS (SF Pro) | Windows (Segoe UI Variable) | GNOME (Adwaita Sans) |
|---|---|---|---|
| Empty-state title | `.title` semibold | Subtitle (20, semibold) | `.title-1` |
| Page title | toolbar title | Title (28, semibold) | header bar title |
| Detail title | `.title3` semibold | Subtitle (20, semibold) | `.title-3` |
| Body, row title | `.body` (13) | Body (14) | default (11 pt) |
| Row subtitle, notes, footer | `.subheadline` (11) | Caption (12) | `.caption` |
| Monospace | `.system(size: 12, design: .monospaced)` | Cascadia Mono 13, then Consolas | `monospace` (Adwaita Mono) 10 pt |

### 13.6 Icons

One meaning per row. Each platform uses its own glyph set.

| Meaning | SF Symbols | Segoe Fluent Icons | Adwaita symbolic |
|---|---|---|---|
| Convert (navigation) | `doc.text` | `Page` U+E7C3 | `x-office-document-symbolic` |
| History (navigation) | `clock.arrow.circlepath` | `History` U+E81C | `document-open-recent-symbolic` |
| Settings | `gearshape` | `Setting` U+E713 | `open-menu-symbolic` (primary menu) |
| Add files (drop zone, drop bar) | `arrow.down.doc` | `OpenFile` U+E8E5 | `document-open-symbolic` |
| Web address | `link` | `Link` U+E71B | `insert-link-symbolic` |
| Source: file | `doc` | `Document` U+E8A5 | `text-x-generic-symbolic` |
| Source: web page | `globe` | `Globe` U+E774 | `web-browser-symbolic` |
| Queued | `hourglass` | `Clock` U+E917 | `content-loading-symbolic` |
| Converting | `ProgressView` | `ProgressRing` | `AdwSpinner` |
| Done | `checkmark.circle.fill` | `CompletedSolid` U+EC61 | `object-select-symbolic` |
| Warning | `exclamationmark.triangle.fill` | `Warning` U+E7BA | `dialog-warning-symbolic` |
| Failed | `xmark.octagon.fill` | `ErrorBadge` U+EA39 | `dialog-error-symbolic` |
| Copy Markdown | `doc.on.doc` | `Copy` U+E8C8 | `edit-copy-symbolic` |
| Save As… | `square.and.arrow.down` | `SaveAs` U+E792 | `document-save-as-symbolic` |
| Show in folder | `folder` | `FolderOpen` U+E838 | `folder-open-symbolic` |
| Retry | `arrow.clockwise` | `Refresh` U+E72C | `view-refresh-symbolic` |
| Clear Finished | `xmark.circle` | `Clear` U+E894 | `edit-clear-all-symbolic` |
| Clear History | `trash` | `Delete` U+E74D | `user-trash-symbolic` |
| Search | `magnifyingglass` | `Search` U+E721 | `system-search-symbolic` |

The app icon is `desktop/icon.png`, the ink rounded square with the white M
stroke. Each platform builds its icon files from it.

## 14. Window sizes and narrow widths

| | macOS | Windows | GNOME |
|---|---|---|---|
| Default | 1200 × 780 | 1200 × 780 | 1200 × 780 |
| Minimum | 640 × 480 | 640 × 480 | 360 × 480 |
| Navigation collapses | `NavigationSplitView` hides the sidebar when narrow. The toolbar's sidebar button shows it | `PaneDisplayMode="Auto"`: expanded from 1008, compact (48 px icons) from 641, minimal below | `AdwBreakpoint` `max-width: 640sp` collapses the outer split view |
| Stacked list and detail | When the area right of the navigation is narrower than 760, the list fills it and choosing a row shows the detail in its place, with a back button | Same rule, using the `NavigationView` back button | `AdwBreakpoint` `max-width: 960sp` collapses the inner split view |

In the stacked layout, Back, Esc and the back shortcut return to the list with the
row still selected. Empty Convert, empty History and Settings never stack.

## 15. Preview stylesheet

`crates/markitai-app/src/preview.css`, the exact text placed in the `<style>`
element of every preview document:

```css
:root {
  color-scheme: light;
  --text-1: #18181b;
  --text-2: #52525b;
  --text-3: #6b6b73;
  --rule: #e4e4e7;
  --rule-strong: #d4d4d8;
  --code-bg: #f4f4f5;
  --ink: #18181b;
  --selection-bg: #18181b;
  --selection-fg: #fafafa;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI Variable Text", "Segoe UI", "Adwaita Sans", Cantarell, system-ui, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei UI", "Noto Sans CJK SC", sans-serif;
  --mono: ui-monospace, "SF Mono", Menlo, "Cascadia Mono", Consolas, "Adwaita Mono", "DejaVu Sans Mono", "PingFang SC", "Microsoft YaHei UI", "Noto Sans Mono CJK SC", monospace;
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --text-1: #fafafa;
  --text-2: #a1a1aa;
  --text-3: #a1a1aa;
  --rule: #3f3f46;
  --rule-strong: #52525b;
  --code-bg: #27272a;
  --ink: #fafafa;
  --selection-bg: #fafafa;
  --selection-fg: #18181b;
}
* { box-sizing: border-box; }
html { background: transparent; -webkit-text-size-adjust: 100%; }
body {
  margin: 0;
  padding: 24px 24px 48px;
  font: 14px/1.75 var(--sans);
  color: var(--text-1);
  overflow-wrap: break-word;
}
main { max-width: 760px; margin: 0 auto; }
::selection { background: var(--selection-bg); color: var(--selection-fg); }
main > :first-child { margin-top: 0; }
h1, h2, h3, h4, h5, h6 { color: var(--text-1); font-weight: 600; }
h1 { margin: 0 0 16px; font-size: 24px; line-height: 1.25; letter-spacing: -0.02em; }
h2 { margin: 32px 0 12px; font-size: 18px; line-height: 1.35; letter-spacing: -0.01em; }
h3 { margin: 24px 0 8px; font-size: 15px; line-height: 1.4; }
h4, h5, h6 { margin: 20px 0 8px; font-size: 14px; line-height: 1.5; }
p { margin: 0 0 16px; color: var(--text-2); }
a { color: var(--text-1); text-decoration: underline; text-underline-offset: 2px; }
strong { color: var(--text-1); font-weight: 600; }
del { color: var(--text-3); }
ul, ol { margin: 12px 0 16px; padding-left: 22px; color: var(--text-2); }
ul { list-style: none; padding-left: 20px; }
ul > li { position: relative; }
ul > li::before { content: "-"; position: absolute; left: -18px; color: var(--text-3); }
ul > li:has(> input[type="checkbox"])::before { content: none; }
li { margin-bottom: 6px; line-height: 1.7; }
li > p { margin-bottom: 6px; }
input[type="checkbox"] { margin: 0 8px 0 -20px; vertical-align: -2px; accent-color: var(--ink); }
blockquote { margin: 16px 0; padding: 2px 0 2px 16px; border-left: 2px solid var(--rule-strong); }
blockquote p { margin: 0 0 8px; color: var(--text-3); }
blockquote p:last-child { margin-bottom: 0; }
code { padding: 1px 5px; border: 1px solid var(--rule); border-radius: 6px; background: var(--code-bg); font: 12.5px/1.5 var(--mono); }
pre { margin: 16px 0; padding: 14px 16px; overflow-x: auto; border: 1px solid var(--rule); border-radius: 8px; background: var(--code-bg); }
pre code { padding: 0; border: 0; background: none; font-size: 12.5px; line-height: 1.65; }
table { display: block; max-width: 100%; overflow-x: auto; margin: 0 0 16px; border-collapse: collapse; }
th { padding: 0 16px 8px 0; border-bottom: 1px solid var(--ink); font-weight: 600; text-align: left; color: var(--text-1); white-space: nowrap; }
td { padding: 9px 16px 9px 0; border-bottom: 1px solid var(--rule); color: var(--text-2); vertical-align: top; }
img { max-width: 100%; height: auto; border: 1px solid var(--rule); border-radius: 8px; }
hr { margin: 24px 0; border: 0; border-top: 1px solid var(--rule); }
.img-remote { display: inline-block; padding: 1px 8px; border: 1px dashed var(--rule-strong); border-radius: 6px; font-size: 12.5px; color: var(--text-3); }
.footnote-reference { font-size: 11px; line-height: 0; }
.footnote-reference a { text-decoration: none; }
.footnote-definition { display: flex; gap: 8px; margin: 8px 0; font-size: 13px; line-height: 1.6; color: var(--text-3); }
.footnote-definition:first-of-type { margin-top: 32px; padding-top: 16px; border-top: 1px solid var(--rule); }
.footnote-definition p { margin: 0; color: var(--text-3); }
.footnote-definition-label { flex: none; font-size: 11px; }
```

## 16. Checks before calling a platform done

| Check | How |
|---|---|
| Preview parity | The same `preview_html` request gives byte-identical HTML in all three apps (it is one Rust function). Test it in `markitai-app` with fixtures for tables, task lists, footnotes, raw HTML, relative and remote images, and both themes |
| States | Each of Queued, Converting, Done (with and without warnings, with and without AI usage) and Failed (every row of section 9) is reachable with repository fixtures and an isolated `HOME` and `MARKITAI_HOME` |
| Folders | Two sources with one label give `label` and `label (2)`. A failed conversion leaves no folder |
| History | 501 additions keep 500. Remove and Clear leave the files. A corrupt file is renamed and reported once |
| Quit | With one item Converting, every quit path shows the quit dialog. Cancel keeps it running |
| Copy | Every visible string comes from `native/strings.json`, in both languages |
| Accessibility | Keyboard-only run through every flow. VoiceOver, Narrator and Orca read row names and announcements |
