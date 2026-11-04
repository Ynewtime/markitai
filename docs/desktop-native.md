# Markitai Native

Markitai Native is a desktop version of Markitai with an app for macOS, Windows
and Linux (GNOME). Each app converts local files and web pages to Markdown, two
at a time, and shows the result as a preview or as Markdown text. Each app keeps
a history of finished conversions. [Desktop versions](desktop.md) compares it
with Markitai Web and Markitai Mygo.

The apps call the markitai Rust core in process through the private crate
`crates/markitai-app`. They do not start `markitai serve` or show the web
workbench. The only web content they show is the Markdown preview: the core
renders it, and the system web view displays it with scripts and network access
off.

All three apps share one information architecture, one set of flows and states,
one copy deck and one brand. Each app draws them with its own platform's
controls. [`desktop/DESIGN.md`](../desktop/DESIGN.md) is the specification.
Where an app differs from that file, the app has the bug.

| | macOS | Windows | Linux (GNOME) |
|---|---|---|---|
| UI | SwiftUI, with AppKit where SwiftUI has no control | WinUI 3 in C# (.NET 10) | GTK 4 and libadwaita in Rust |
| Minimum system | macOS 13 | Windows 10 1809. Mica on Windows 11 | GTK 4.18, libadwaita 1.7, WebKitGTK 6.0 (GNOME 48) |
| App | `Markitai Native.app`, bundle ID `dev.markitai.native` | `Markitai Native.exe` | `markitai-native`, app ID `dev.markitai.native` |
| Source | `desktop/native/macos/` | `desktop/native/windows/` | `desktop/native/linux/`, a Cargo workspace of its own |
| Settings | `UserDefaults` domain `dev.markitai.native` | `%LOCALAPPDATA%\Markitai Native\settings.json` | `$XDG_CONFIG_HOME/dev.markitai.native/settings.json` |
| History | `~/Library/Application Support/dev.markitai.native/history.json` | `%LOCALAPPDATA%\Markitai Native\history.json` | `$XDG_DATA_HOME/dev.markitai.native/history.json` |

## What the apps share with the CLI

The apps read the same markitai configuration as the CLI, including models,
providers and keys. To change those, edit the configuration. The apps do not
manage it. In Settings, the configuration source can be set to the built-in
defaults instead.

Each conversion writes into its own folder under the output folder. The default
output folder is `Documents/Markitai`. A folder holds `<source name>.md`, or
`<source name>.llm.md` instead when AI ran, and `.markitai/` with assets and
screenshots. With `llm.keep_base` set in the configuration, the folder holds
both Markdown files.

## Building from source

Each platform builds on its own system. Every build compiles the core from the
checkout.

```sh
python3.13 desktop/native/macos/build.py            # desktop/native/macos/build/Markitai Native.app
python3.13 desktop/native/macos/build.py --check    # also the tests, isolated
```

macOS needs cargo, the Command Line Tools' Swift, `sips`, `iconutil` and
`codesign`. Xcode is not used. The app has an ad hoc signature and is not
sandboxed, because the core runs LibreOffice, Chrome and the subscription CLIs.

```sh
python3.13 desktop/native/windows/build.py          # prints the app folder
python3.13 desktop/native/windows/build.py --check  # also Markitai.Core.Tests
```

Windows needs Rust with the `x86_64-pc-windows-msvc` target, the .NET 10 SDK
and Python 3.13. The app folder is self-contained: `Markitai Native.exe`,
`markitai_app.dll` with a static C runtime, and the Windows App SDK. The preview
needs the Microsoft Edge WebView2 Runtime. Without it, the Markdown tab still
works. `desktop/native/windows/smoke.ps1` drives the
built app through UI Automation. The `Desktop Native` workflow runs the build,
the tests and that script on a hosted runner.

```sh
python3 desktop/native/linux/build.py               # desktop/native/linux/build/, an install prefix
python3 desktop/native/linux/build.py --check       # strings, tests and clippy
```

Linux needs cargo and the GTK 4, libadwaita and WebKitGTK 6.0 development
packages. The staged tree holds `bin/markitai-native`, the desktop file, the
icons and the metainfo. Copy it to `/usr/local` or `~/.local`, or run
`build/bin/markitai-native` in place.

## Copy and strings

The copy for both languages lives in section 10 of `desktop/DESIGN.md`.
`desktop/scripts/strings.py` generates `desktop/strings.json` from it, and fails
when the two differ. Each app embeds that file. A UI language starting with
`zh` gets Simplified Chinese. Every other language gets English.

## Limits

- No folders or recursive batches, and no canceling a running conversion.
- No editing Markdown. Math, Mermaid and syntax highlighting do not render in
  the preview.
- No model, provider or key management, presets, profiles or `--keep-base`
  switch.
- No notifications, badges or auto-update. Builds are unsigned.
