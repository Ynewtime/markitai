# Markitai Web

Markitai Web, a desktop app, opens the [browser workbench](web-ui.md) in a window
of its own: open Markitai Web and convert, with no terminal and no browser tab. It
runs the `markitai` program bundled inside it as a server for that window
([running under another app](serve.md#running-under-another-app)), so it
converts exactly as the CLI of the same version does. It runs on macOS 13 or
later on Apple silicon, on Linux on x86-64 ([what it needs](#linux)), and on
Windows 10 or later on x86-64 ([what it needs](#windows)).

No release publishes Markitai Web yet; build it from a checkout with Go 1.27.3 or
later, [on a Mac](#building-from-source), [on Linux](#linux) or
[on Windows](#windows).

## What it shares with the CLI

- The app uses the same `MARKITAI_HOME` as `markitai` in your terminal
  (`~/.markitai` unless your login shell sets another): configuration, keys,
  models, installed browsers and the workbench history are one.
- On macOS and Linux it starts its server with your login shell's
  environment (the shell `SHELL` names, csh and tcsh included), so programs
  on your `PATH`,
  exported keys and a `MARKITAI_HOME` set in your startup files work as they
  do in a terminal. It waits up to 10 seconds for the startup files; if the
  shell fails or takes longer, the server starts with the app's own
  environment and the first line of its log says why. Jobs the startup files
  leave running in the background end with that shell, as when a terminal
  closes. [On Windows](#windows) it uses the environment the app starts
  with. The server's working directory is `MARKITAI_HOME`.
- One server runs per `MARKITAI_HOME`. While `markitai serve` runs with the
  same home, the app shows that its server stopped, with the reason from its
  log; stop the other server and choose Try Again. While the app runs,
  `markitai serve` with that home refuses to start in the same way.
- The server listens on `127.0.0.1` on a port of its own, with a token only
  the app holds: the first free one from 23600 to 23699, below the ports the
  system hands to other programs' connections, and never 3600. The app keeps
  the port across launches, so the workbench's language, theme and other
  per-browser preferences stay; if another program has taken it, the app
  picks a new port and they reset once.

## How it differs from a browser

- Downloads and exports ask where to save.
- Export PDF saves an A4 PDF, laid out as the workbench prints, where you
  choose. There is no print dialog and no printing to a printer. Only the
  workbench starts it: a link in converted content cannot.
- Links to other sites open in your default browser, and `mailto:` links in
  your mail app, when you click them; a page cannot open them by itself. The
  window shows only the workbench and the app's own pages: a link that would
  take it elsewhere opens in the browser instead.
- The workbench has no Access token page and no update entry: the token is
  the app's, and a newer app brings a newer `markitai`.
- Closing the window quits the app. With no unfinished conversion (running,
  queued, or a file still uploading) it quits at once. Otherwise it asks, also
  when the server cannot say how many are unfinished: Quit When Done lets
  running conversions finish and cancels queued ones (the window says so, with
  Stop Now and Quit), Stop and Quit stops them all now, Cancel keeps working.
  Quitting again meanwhile only brings the window forward. A conversion that
  started as you quit, too late to be counted, is not stopped either: after 5
  seconds the window shows that it is finishing, as with Quit When Done.
- With conversions unfinished, logging out, restarting or shutting down the
  Mac is canceled while the app asks, or finishes them, as above; once it has
  quit, try again.
- If the app crashes or is force quit, its server notices, finishes its
  running conversions and exits by itself. A launch in the meantime shows that
  the server stopped; Try Again works once the old one is gone.
- On macOS the server's log is `~/Library/Logs/Markitai Web/serve.log` (Show Log
  on the error page; [on Linux](#linux) and [on Windows](#windows) it is
  elsewhere): one file per launch
  of the app, Try Again included. The previous launch's is
  `serve.previous.log`.

## Building from source

On a Mac you need Go 1.27.3 or later and the `markitai` release archive for
`aarch64-apple-darwin` of the checkout's version: the GitHub release's, or one
built from the checkout with `python3 scripts/release_build.py cli --target
aarch64-apple-darwin --output DIR`. Then, in the checkout:

```sh
python3 scripts/desktop_web_build.py stage --archive markitai-v<version>-aarch64-apple-darwin.tar.gz --sums SHA256SUMS
python3 scripts/desktop_web_build.py build --platform darwin/arm64
open "desktop/web/build/darwin-arm64/Markitai Web.app"
```

`stage` refuses an archive whose name or checksum does not match the
checkout's version (`--sums` is the release's `SHA256SUMS`; leave it out for
an archive you built). `build` writes `Markitai Web.app` and the disk image
`markitai-web-<version>-arm64.dmg` to `desktop/web/build/darwin-arm64/`. Copy
the app to Applications, or open the disk image and drag it there. The app
carries the licences of `markitai` and of the Go modules it links in
`Contents/Resources/licenses/`.

## Unsigned builds

The app is signed ad hoc, not with an Apple Developer ID, and is not
notarized. A copy you built opens normally. A copy that came through a web
browser, AirDrop or a chat app carries the download "quarantine" mark, and
macOS reports that it cannot verify the app. Once you are sure where it came
from, remove the mark from the disk image before opening it, or from the app
after copying it:

```sh
xattr -d com.apple.quarantine markitai-web-<version>-arm64.dmg
xattr -dr com.apple.quarantine "/Applications/Markitai Web.app"
```

As with [the CLI](quickstart.md#unsigned-builds-on-macos), this exempts only
that file; there is no need to turn Gatekeeper off.

## Linux

The Linux app is for x86-64 and needs glibc 2.28 or later, GTK 3 and
WebKitGTK 4.1 (`libwebkit2gtk-4.1-0` on Debian and Ubuntu, `webkit2gtk-4.1`
on Arch). It has been tested under Hyprland on Wayland. Build it on Linux from
the `markitai` release archive for `x86_64-unknown-linux-gnu` of the
checkout's version (the GitHub release's, or one `release_build.py cli
--target x86_64-unknown-linux-gnu` builds in the manylinux_2_28 image), then
install it for your user:

```sh
python3 scripts/desktop_web_build.py stage --archive markitai-v<version>-x86_64-unknown-linux-gnu.tar.gz --sums SHA256SUMS
python3 scripts/desktop_web_build.py build --platform linux/amd64
sh desktop/web/build/linux-amd64/install.sh
```

`build` writes to `desktop/web/build/linux-amd64/` the app (`markitai-web`,
with `bin/markitai` and `licenses/` beside it), the archive
`markitai-web-<version>-linux-amd64.tar.gz`, the Debian package
`markitai-web_<version>_amd64.deb` (it installs in
`/opt/markitai-web`), a `PKGBUILD` and `install.sh`. `install.sh` puts the
archive beside it in `~/.local/markitai-web.app` with an entry in the
applications menu; run it again to update, or with `--uninstall` to remove the
app and leave its settings and data.

On Linux the server's log is `~/.config/Markitai Web/logs/serve.log` (under
`XDG_CONFIG_HOME`), beside the app's settings; the window's web data is in
`~/.local/share/markitai-web` and `~/.cache/markitai-web`. There is no
application menu: closing the window is how the app quits.

## Windows

The Windows app is for x86-64 and needs Windows 10 or later with the
Microsoft Edge WebView2 Runtime, which Windows 11 includes. It has been
tested on Windows Server 2025. Build it on Windows, with Python 3, from the
`markitai` release archive for `x86_64-pc-windows-msvc` of the checkout's
version (the GitHub release's, or one `release_build.py cli --target
x86_64-pc-windows-msvc` builds), then run the installer:

```sh
python scripts/desktop_web_build.py stage --archive markitai-v<version>-x86_64-pc-windows-msvc.zip --sums SHA256SUMS
python scripts/desktop_web_build.py build --platform windows/amd64
desktop\web\build\windows-amd64\markitai-web-setup-<version>-amd64.exe
```

`build` writes to `desktop\web\build\windows-amd64\` the app (`Markitai Web.exe`,
with `bin\markitai.exe` and `licenses\` beside it) and the installer
`markitai-web-setup-<version>-amd64.exe`. The first build downloads NSIS,
which makes the installer, into `%LOCALAPPDATA%\mygo`, unless NSIS is
installed. The installer needs no administrator rights: it installs the app
for your user in `%LOCALAPPDATA%\Programs\Markitai Web`, with a Start menu
shortcut and an entry in Settings > Apps that uninstalls it and leaves its
settings and data. To update, run a newer installer; it asks to close a
running app, closes its window as you would, and stops what still runs 10
seconds later, conversions included.

Signing out, restarting or shutting down Windows does not wait for the app:
unfinished conversions stop at once and the workbench shows them as
interrupted when the app next starts.

The installer and the app are not signed. SmartScreen may stop an installer
that came through a browser with "Windows protected your PC": once you are
sure where it came from, choose More info, then Run anyway.

The app starts its server with the environment Windows gives the programs it
starts: your user and system variables, as a new terminal has them.
`MARKITAI_HOME` is `%USERPROFILE%\.markitai` unless one of them sets
another. The server's log is `%APPDATA%\Markitai Web\logs\serve.log`, beside the
app's settings in `%APPDATA%\Markitai Web`; the window's web data is in
`%LOCALAPPDATA%\Markitai Web\WebView2`.

## Limits

- macOS on Apple silicon, and Linux and Windows on x86-64, only.
- No automatic updates: install a newer app. A `markitai` you installed
  separately is updated on its own and does not change the app's;
  `markitai update` leaves the app's server running as it is.
