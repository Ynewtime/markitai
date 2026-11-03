# Desktop app

Markitai for macOS opens the [browser workbench](web-ui.md) in a window of its
own: double-click Markitai and convert, with no terminal and no browser tab. It
runs the `markitai` program bundled inside it as a server for that window
([running under another app](serve.md#running-under-another-app)), so it
converts exactly as the CLI of the same version does. It needs an Apple
silicon Mac with macOS 12 or later.

No release publishes the app yet; [build it](#building-from-source) from a
checkout.

## What it shares with the CLI

- The app uses the same `MARKITAI_HOME` as `markitai` in your terminal
  (`~/.markitai` unless your login shell sets another): configuration, keys,
  models, installed browsers and the workbench history are one.
- It starts its server with your login shell's environment (the shell
  `SHELL` names, csh and tcsh included), so programs on your `PATH`,
  exported keys and a `MARKITAI_HOME` set in your startup files work as they
  do in a terminal. It waits up to 10 seconds for the startup files; if the
  shell fails or takes longer, the server starts with the app's own
  environment and the first line of its log says why. The server's working
  directory is `MARKITAI_HOME`.
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
  choose. There is no print dialog and no printing to a printer.
- Links to other sites open in your default browser, and `mailto:` links in
  your mail app, when you click them; a page cannot open them by itself. The
  window shows only the workbench and the app's own pages: a link that would
  take it elsewhere opens in the browser instead.
- The workbench has no Access token page and no update entry: the token is
  the app's, and a newer app brings a newer `markitai`.
- Closing the window quits the app. With no unfinished conversion (running
  or queued) it quits at once. Otherwise it asks, also when the server cannot
  say how many are unfinished: Quit When Done lets running conversions finish
  and cancels queued ones (the window says so, with Stop Now and Quit), Stop
  and Quit stops them all now, Cancel keeps working. Quitting again meanwhile
  only brings the window forward.
- If the app crashes or is force quit, its server notices, finishes its
  running conversions and exits by itself. A launch in the meantime shows that
  the server stopped; Try Again works once the old one is gone.
- The server's log is `~/Library/Logs/Markitai/serve.log` (Show Log on the
  error page): one file per launch of the app, Try Again included. The
  previous launch's is `serve.previous.log`.

## Building from source

You need Go 1.27.3 or later and the `markitai` release archive for
`aarch64-apple-darwin` of the checkout's version: the GitHub release's, or one
built from the checkout with `python3 scripts/release_build.py cli --target
aarch64-apple-darwin --output DIR`. Then, in the checkout:

```sh
python3 scripts/desktop_build.py stage --archive markitai-v<version>-aarch64-apple-darwin.tar.gz --sums SHA256SUMS
python3 scripts/desktop_build.py build --platform darwin/arm64
open desktop/build/darwin-arm64/Markitai.app
```

`stage` refuses an archive whose name or checksum does not match the
checkout's version (`--sums` is the release's `SHA256SUMS`; leave it out for
an archive you built). `build` writes `Markitai.app` and the disk image
`markitai-desktop-<version>-arm64.dmg` to `desktop/build/darwin-arm64/`. Copy
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
xattr -d com.apple.quarantine markitai-desktop-<version>-arm64.dmg
xattr -dr com.apple.quarantine /Applications/Markitai.app
```

As with [the CLI](quickstart.md#unsigned-builds-on-macos), this exempts only
that file; there is no need to turn Gatekeeper off.

## Limits

- macOS on Apple silicon only. The app's code builds for Windows and Linux,
  but nothing packages it there yet.
- No automatic updates: install a newer app. A `markitai` you installed
  separately is updated on its own and does not change the app's.
