# Markitai Mygo

Markitai Mygo is a desktop version of Markitai with one app for macOS, Windows
and Linux. It converts local files and web pages to Markdown, two at a time,
and shows the result as a preview or as Markdown text. It keeps a history of
finished conversions. [Desktop versions](desktop.md) compares it with Markitai
Web and Markitai Native.

The app is one Go program on [mygo](https://github.com/egoist/mygo). mygo's
`ui` package draws the whole interface on the GPU, so the app looks the same on
every system. The app calls the markitai Rust core in process: it loads the C
ABI of the private crate `crates/markitai-app` with purego, without cgo. It does
not start `markitai serve` or show the web workbench. The only web content it
shows is the Markdown preview: the core renders it, and the system web view
displays it with scripts and network access off.

Markitai Mygo has the information architecture, flows, states, copy deck and
brand of Markitai Native. [`desktop/DESIGN.md`](../desktop/DESIGN.md) is the
product specification, and [`desktop/mygo/DESIGN.md`](../desktop/mygo/DESIGN.md)
says how this app meets it. Where the app differs from those files, the app has
the bug.

| | macOS | Windows | Linux |
|---|---|---|---|
| Minimum system | macOS 13 | Windows 10, x86-64. The preview needs the Microsoft Edge WebView2 Runtime | x86-64 with GTK 3. The preview needs WebKitGTK 4.1 |
| App | `Markitai Mygo.app`, bundle ID `dev.markitai.mygo` | `Markitai Mygo.exe` with `markitai_app.dll` beside it | `markitai-mygo`, with the desktop file `dev.markitai.mygo.desktop` |
| Settings | `~/Library/Application Support/dev.markitai.mygo/settings.json` | `%LOCALAPPDATA%\Markitai Mygo\settings.json` | `$XDG_CONFIG_HOME/dev.markitai.mygo/settings.json` |
| History | `~/Library/Application Support/dev.markitai.mygo/history.json` | `%LOCALAPPDATA%\Markitai Mygo\history.json` | `$XDG_DATA_HOME/dev.markitai.mygo/history.json` |

The window's size and position are kept in `window-state.json` beside the
history. On macOS, WebKit also keeps the preview's compiled blocking rules in
`~/Library/WebKit/dev.markitai.mygo/`. On Windows, the preview's web view data
is in `%LOCALAPPDATA%\Markitai Mygo\WebView2`. On Linux, everything the app
and its web view keep is in folders named `dev.markitai.mygo`: the blocking
rules and a cookie store the preview does not use in
`$XDG_CONFIG_HOME/dev.markitai.mygo/`, and WebKitGTK's own data in
`$XDG_DATA_HOME/dev.markitai.mygo/` and `$XDG_CACHE_HOME/dev.markitai.mygo/`.
As in any GTK app, once a file dialog has shown, GTK also updates the shared
`recently-used.xbel` and `gvfs-metadata/` in `$XDG_DATA_HOME`.

## What the app shares with the CLI

The app reads the same markitai configuration as the CLI, including models,
providers and keys. To change those, edit the configuration. The app does not
manage it. In Settings, the configuration source can be set to the built-in
defaults instead.

Each conversion writes into its own folder under the output folder. The default
output folder is `Documents/Markitai`, shared with Markitai Native. On Windows,
`Documents` is the system's Documents folder wherever it is, as when OneDrive
keeps it. A folder
holds `<source name>.md`, or `<source name>.llm.md` instead when AI ran, and
`.markitai/` with assets and screenshots. With `llm.keep_base` set in the
configuration, the folder holds both Markdown files.

## Building from source

Each platform builds on its own system. Every build compiles the core from the
checkout.

```sh
python3.13 desktop/mygo/build.py            # the core, then the app in desktop/mygo/build/<goos>-<goarch>/
python3.13 desktop/mygo/build.py --check    # the core, go vet and go test, isolated; builds no app
python3.13 desktop/mygo/build.py --no-core  # reuse the core built before
```

The build needs cargo, Go 1.27.3 and Python 3.13 or newer. mygo comes in as a
Go tool of `desktop/mygo/go.mod`, so it needs no separate installation.

- macOS: `Markitai Mygo.app`, with an ad hoc signature and not sandboxed,
  because the core runs LibreOffice, Chrome and the subscription CLIs.
- Windows: `Markitai Mygo.exe` and `markitai_app.dll`, which links the C
  runtime statically. There is no installer. The `Desktop Mygo` workflow builds
  the app, runs the tests and drives the app's window on a hosted runner.
- Linux: the `markitai-mygo` executable with its desktop file, a Debian
  package, a `.tar.gz` of the app, `install.sh` beside it, and a `PKGBUILD`.
  `sh install.sh` installs the app from the `.tar.gz` for the user in
  `~/.local`, and `sh install.sh --uninstall` removes it. Every package
  installs the desktop file as `dev.markitai.mygo.desktop`, the name of the
  app's window, so launchers and docks match them.

## Copy and strings

The copy for both languages lives in section 10 of `desktop/DESIGN.md`.
`desktop/scripts/strings.py` generates `desktop/strings.json` from it, and the
app embeds a copy that its tests keep equal. A system language starting with
`zh` gets Simplified Chinese. Every other language gets English. On macOS and
Windows, the system's open and save dialogs and, on macOS, the About panel and
the menu items the system adds follow the app's language. On Linux, as in
Markitai Native, the file dialogs are the desktop's when its portal has them:
their title follows the app, and the rest the desktop's language and style.
Without them, GTK's own dialogs show, within the screen and in the app's
language. The programs the app opens on Linux, as the browser for a link or
the editor for the configuration file, start with your own environment, not
the language and toolkit settings the app sets for itself.

## Limits

- No folders or recursive batches, and no canceling a running conversion.
- No editing Markdown. Math, Mermaid and syntax highlighting do not render in
  the preview.
- No model, provider or key management, presets, profiles or `--keep-base`
  switch.
- No notifications, badges or auto-update. Builds are unsigned.
- On Linux, the launcher entry opens the app without files. Add files by
  dropping them, with Choose Files, or as command-line arguments.
