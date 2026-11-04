# Development

Read [AGENTS.md](../AGENTS.md) and inspect `git status --short --branch` before work.
Between releases the version carries a `-dev` suffix (`Cargo.toml`); a release is
[one commit and a tag](ci.md#the-release-commit), and publishing requires explicit authorization.
Package manifests set the floors: Rust 1.92 (workspace `rust-version`, the
floor of the portable PDF renderer hayro), Python 3.10 (`bindings/python/pyproject.toml`),
Node.js 18 (`bindings/node/package.json`) and Go 1.23 (`bindings/go/go.mod`).
The CI workflow pins build toolchains; consult `.github/workflows/native.yml`
for the versions used by a particular run.
The separate minimum-Rust CI lane checks core, CLI and FFI with Rust 1.92; it
does not run all tests or validate minimum Python/Node/Go runtimes. User-facing build and
installation steps are in the [quick start](quickstart.md) and
[bindings](bindings.md#installation).

`python scripts/check.py` runs the source gate with `MARKITAI_HOME` set to
`.local/test-home` and `HOME` and `USERPROFILE` set to `.local/test-user-home`
on all hosts; `scripts/check.sh` delegates to it on Unix. It runs the first
three commands below and the `scripts/test_*.py` helper tests. The Rust tests
run with [cargo-nextest](https://nexte.st) when it is installed (each test in a
process of its own, every test binary at once), otherwise with `cargo test`;
`MARKITAI_CHECK_RUNNER=cargo` or `nextest` chooses. Formatting and the helper
tests (several files at once, a file of over 16 tests in parts) run beside the Rust tests and clippy, and their
output follows them. The helper tests include the
workbench tests, which import its npm packages: when Node is installed and
`crates/markitai-cli/src/server/web/node_modules` is missing or differs from
`package-lock.json`, the gate first runs `npm ci` there (it says so) and skips
the helper tests if npm is missing or the install fails. Helper tests start
child processes through `scripts/bounded_run.py`: each is stopped, with all it
started, after a minute or two, and a test running over three minutes ends the run with
every thread's stack, so a hang fails instead of holding CI. Every step runs even
after one fails; the last line names the failed ones, and the exit code is non-zero. `CARGO_HOME` and
`RUSTUP_HOME` default to the original home's `.cargo` and `.rustup`, and Python
or Node reached through a home-relative shim (mise, asdf) is replaced on `PATH`
by the directory of the interpreter it resolves to. Other tools must be on
`PATH` by an absolute directory. However they are started, the Rust tests never
see the developer's own `copilot`, `claude` or `codex` (hosted CI has none):
each test process replaces a `PATH` directory holding one by links to its other
programs, or leaves it out on Windows. `MARKITAI_CHECK_SKIP="name other"` passes each
space-separated name to the Rust tests as `--skip name` (e.g. Vision OCR in a macOS VM).

For branch work, `python scripts/check.py --quick` checks only what the files
changed since the branch left `origin/main` (committed or not; `--base REF`
compares with another branch) can affect: with nextest, the tests of the
changed crates and of every crate depending on them (all crates when
`Cargo.toml`, `Cargo.lock` or `vendor/` changed; none for workbench sources or
scripts alone), clippy and formatting when a crate changed, and all helper tests. Run the full gate before a merge or
a release. The CLI's integration tests build as two binaries: `update` and
`integration` (`crates/markitai-cli/tests/integration.rs` lists the other test files).
Build the optimized CLI separately:

```sh
cargo fmt --all --check
cargo test --workspace --locked --no-fail-fast
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo build --release --locked -p markitai-cli
```

The binding suites run against built artifacts; see
[bindings](bindings.md). [Native CI](ci.md) builds, installs and tests every
package from a clean checkout.

## Intel macOS from Apple silicon

With `rustup target add x86_64-apple-darwin`, the same checks run for x86-64
macOS; the test executables run under Rosetta 2:

```sh
cargo clippy --workspace --all-targets --locked --target x86_64-apple-darwin -- -D warnings
cargo test --workspace --locked --no-fail-fast --target x86_64-apple-darwin
cargo build --release --locked --target x86_64-apple-darwin -p markitai-cli
```

Rosetta is not an Intel Mac. Under it, Vision text recognition fails without an
error and CoreGraphics' rejection of a malformed PDF crashes the process
intermittently; the affected tests check the explicit Rosetta error or skip
that one call, and say so on stderr. Rosetta exposes no AVX, AVX2, FMA or BMI,
so the AVX2 paths that dependencies select on most Intel Macs are not run.

Follow [AGENTS.md](../AGENTS.md) for test isolation, credentials, validation and
Git rules. A source check does not establish installed-package compatibility.

The original repository is not a build dependency. Any referenced differential
test runner must accept its location explicitly and record the reference SHA.

See [native CI](ci.md) for the clean-checkout package driver and its actual
installation checks. Configuring a runner is separate from executing it.

## Markitai Web

`desktop/web/` is [Markitai Web](desktop-web.md), a desktop app: a Go module
(Go 1.27.3 or later, `desktop/web/go.mod`) on
[mygo](https://github.com/egoist/mygo), a pure-Go toolkit for native windows,
which comes in as a Go tool (`go tool mygo`). It packages a release CLI
archive; its version is the workspace's, which `release_build.py
prepare-version` writes to `desktop/web/mygo.json` and `release_build.py
version` compares.

```sh
python3 scripts/desktop_web_build.py vet
python3 scripts/desktop_web_build.py stage --archive markitai-v<version>-<target>.tar.gz|.zip [--sums SHA256SUMS]
python3 scripts/desktop_web_build.py build --platform darwin/arm64|linux/amd64|windows/amd64 [--e2e]
python3 scripts/desktop_web_e2e.py ROUND [--only NAME,...]
```

`vet` checks gofmt and runs `go vet` for darwin, linux and windows, and the Go
tests on this system (compiling them for the others); the tests stand in for
`markitai serve` with the test binary itself, so they need no markitai.
`stage` replaces the Git-ignored `desktop/web/resources/` with the archive's
`markitai` and the licences: the archive's, Go's, and those of every module
`go list -deps` links (a module without one fails). Each platform is built on
its own system: `build` makes `desktop/web/build/darwin-arm64/Markitai Web.app`
and `markitai-web-<version>-arm64.dmg` on an Apple silicon Mac, or the app,
its `.tar.gz`, `.deb`, `PKGBUILD` and `install.sh` in
`desktop/web/build/linux-amd64/` on x86-64 Linux
([Linux](desktop-web.md#linux)), from the `x86_64-unknown-linux-gnu` archive
the manylinux_2_28 image builds, or `Markitai Web.exe` and the installer
`markitai-web-setup-<version>-amd64.exe` in `desktop/web/build/windows-amd64/`
on x86-64 Windows ([Windows](desktop-web.md#windows)). No
artifact may be named like a CLI archive (`markitai-v…`). `--e2e` builds with
the `e2e` tag, mygo's automation plugin and the web inspector, without
packages or installer, into `desktop/web/build/e2e/`, from a generated project
(`desktop/web/build/e2e-project/`) whose identifier is the app's
(`dev.markitai.web`) plus `.e2e`: on macOS web data is kept by identifier
outside `HOME`, and an e2e run must not share an installed app's.
`desktop_web_e2e.py` drives that build over the automation socket through the
scenarios its docstring lists, each with its own
`HOME` (with the XDG directories on Linux, and `USERPROFILE`, `APPDATA`,
`LOCALAPPDATA` and `TEMP` on Windows), `MARKITAI_HOME` and app data, and
keeps the evidence in `.local/desktop-web-e2e/ROUND/`; the bundled `markitai` must
have `serve --supervised`. On Linux it runs in a Hyprland session and quits by
closing the window with `hyprctl`, as there is no application menu; on
Windows it runs in a desktop session and closes the window with `WM_CLOSE`
(signals become window messages and `TerminateProcess`, the automation socket
is Winsock's `AF_UNIX`). The native Save panel and quit question are answered
by the e2e build's environment on both, and a link it would open in the
browser is recorded instead, so check them by hand once. Building the e2e
project without the symlink privilege links its resources with a junction.
`scripts/test_desktop_web_build.py` runs with the helper tests.

`.github/workflows/desktop-web.yml` builds the Windows app on a hosted
`windows-2025` runner on a push to `desktop/web/**`, `desktop/icon.png` or
`scripts/desktop_web_*.py`, or by hand: it builds the CLI archive (cached by
the tree it is built from), the e2e build and the installer, runs a full
`desktop_web_e2e.py` round, installs, launches, closes and uninstalls the app,
and keeps the evidence and the installer for 7 days.

## Markitai Native

`desktop/native/` holds [Markitai Native](desktop-native.md): the apps for macOS
(SwiftUI), Windows (WinUI 3) and Linux (GTK 4 and libadwaita), specified by
`desktop/DESIGN.md`. They call the core in process through `crates/markitai-app`,
a private crate: a C ABI for the macOS and Windows apps, a Rust API for the
Linux app. `desktop/native/linux` is a Cargo workspace of its own, excluded from
the root one. The three versions share the app icon `desktop/icon.png`;
Markitai Native and Markitai Mygo share the design and `desktop/strings.json`
([desktop versions](desktop.md)).

```sh
python3.13 desktop/scripts/strings.py [--check]
python3.13 desktop/native/macos/build.py [--check] [--no-core]
python3.13 desktop/native/windows/build.py [--check] [--state DIR]
python3 desktop/native/linux/build.py [--check]
```

`strings.py` writes `desktop/strings.json` from section 10 of the design, and
`--check` fails when the two differ. Each build's `--check` runs that app's
tests with an isolated home. `desktop/native/windows/smoke.ps1` drives the built
Windows app through UI Automation, and `.github/workflows/desktop-native.yml`
runs the Windows build, tests and smoke on a hosted `windows-2025` runner on a
push to the files that build reads, or by hand.

## Markitai Mygo

`desktop/mygo/` holds [Markitai Mygo](desktop-mygo.md), one app for macOS,
Windows and Linux: a Go module (`markitai.local/mygo`, Go 1.27.3) on mygo's
`ui` package, with `go tool mygo` from its `go.mod`, specified by
`desktop/DESIGN.md` and `desktop/mygo/DESIGN.md`. It loads the core's C ABI
(`crates/markitai-app` as a cdylib) with purego, so every build is
`CGO_ENABLED=0`. The core library is staged in the Git-ignored
`desktop/mygo/resources/<goos>-<goarch>/`, which `mygo build` puts beside the
executable or in the bundle's resources.

```sh
python3.13 desktop/mygo/build.py [--check] [--no-core] [--platform GOOS/GOARCH] [--e2e]
```

`--check` builds and stages the core, runs `strings.py --check`, fails when
the module's embedded copies of `desktop/strings.json` and `desktop/icon.png`
differ from them or `mygo.json`'s version from the workspace's, and runs
`go vet` and `go test` with `HOME`, `MARKITAI_HOME`, the XDG folders,
`USERPROFILE`, `APPDATA`, `LOCALAPPDATA`, `TEMP` and `TMP` in a temporary
folder. The preview script's tests run it in Node and skip when `node` is not
on the `PATH`. It builds no app and writes no tracked file. Without `--check`,
`build.py` refreshes those copies and builds the app into
`desktop/mygo/build/<goos>-<goarch>/`. On Linux, `desktop/mygo/linux_entry.py`
then gives every copy of the desktop entry (beside the app, in the `.tar.gz`,
`install.sh`, the Debian package and the `PKGBUILD`) Markitai Native's keys and
the name `dev.markitai.mygo.desktop`, which `mygo build` cannot, adds Markitai
Native's 16 to 128 pixel icons to the Debian package and the `PKGBUILD`, keeps
the archives' times so that the same build gives the same bytes, and fails when
`mygo build` writes them differently. The
Windows core links the C runtime statically and builds in
`target/mygo-windows`; the macOS core's install name is
`@rpath/libmarkitai_app.dylib`.

`--e2e` builds with the `e2e` tag, which links mygo's automation plugin and
keeps the web inspector, into `desktop/mygo/build/e2e/`; never ship it. The
driver is a Go test behind the same tag:

```sh
cd desktop/mygo
MARKITAI_MYGO_APP=<e2e executable> MARKITAI_MYGO_E2E_OUT=<evidence folder> \
  go test -tags e2e -count=1 -v ./internal/e2e
```

It launches the app over the automation socket once per run of
`MARKITAI_MYGO_E2E_RUNS` (default `en-light,zh-Hans-dark,en-dark,zh-Hans-light`),
each with its own `HOME`, XDG folders, Windows user folders and `MARKITAI_HOME`
under the evidence folder and a `TEMP` of its own under a short path, and converts fixtures and pages from a local server
through Convert, a second instance, History, Settings and quitting while a
conversion runs. It fails when the app writes in another version's folders,
or on Linux under `Markitai Mygo` or `markitai-mygo`, the names mygo and
WebKitGTK use when the app does not name itself. It keeps the window's own
captures, `report.json` and the app's log per run. With `MARKITAI_MYGO_E2E_WINDOW_SHOT` naming a program that
takes a process ID and a PNG path (`internal/e2e/windowshot.ps1` on Windows), it
also keeps what the system composites, web view included. With
`MARKITAI_MYGO_E2E_PAGE_KEY` naming a program that takes a process ID and
`zoom-in` and sends that window Ctrl+= (`internal/e2e/pagekey.ps1` on Windows),
it gives the preview page the keyboard, presses Zoom In and checks that the
page's script zooms exactly one step. Without it, the e2e build dispatches that
keydown in the page itself, which checks the page's script but not the system's
delivery of the key. Quitting is asked through the app menu on
macOS, Ctrl+Q on Linux and `WM_CLOSE` on Windows.

`.github/workflows/desktop-mygo.yml` runs `--check`, both builds and the e2e
driver on a hosted `windows-2025` runner on a push to the files those read, or
by hand, and keeps the evidence and the app folder for 7 days.

## Website

`website/` is markitai.dev (VitePress, English at the root, Chinese under
`zh/`). It documents markitai 2.x, the code on `main`; the Python markitai 1.x
line lives on the `python` branch. With Bun 1.4:

```sh
cd website
bun install --frozen-lockfile
bun run docs:build    # output in website/.vitepress/dist
```

The changelog pages show the 2.x notes of the repository's `CHANGELOG.md` /
`CHANGELOG.zh.md` first, then the Python 1.x notes in
`website/changelog/python-1.x*.md`, released versions only: the build leaves a
section off the site until its `vX.Y.Z` tag exists. `bun scripts/changelog.mjs
--check` fails on an untagged section in `website/changelog/`, on one below a
released section in `CHANGELOG*.md` (only the top may wait for its tag), and
when English and Chinese publish different versions. The installers served as `https://markitai.dev/setup.sh` and
`/setup.ps1` are `scripts/install/setup.sh` and `setup.ps1`:
`website/scripts/setup-scripts.mjs` copies them into `website/public/`
(git-ignored) on every build, so the site has no copy of its own. The contract
tests in `website/tests` compare the pages with the CLI built from this
checkout (`--help`, `config list`/`config get`, presets, the models
`init --yes` picks):

```sh
cargo build --locked -p markitai-cli
MARKITAI_BIN=target/debug/markitai uv run --no-project --with 'pytest==9.1.1' pytest website/tests
```

The website workflows are described in [native CI](ci.md#website).

The workbench screenshots on the home page and the serve guide are
`website/images/workbench[.zh][.dark].{png,webp}`, referenced by relative path
so the build gives each a content-hashed URL under `/assets/` (a new shot is
never hidden by a cached old one; keep them out of `public/`). Retake all eight
from a clean commit with `cargo build --release --locked -p markitai-cli` and
`uv run scripts/website_shots.py`; its docstring lists the inputs it fetches
and what it needs installed.

`uv run scripts/workbench_upload_e2e.py` drives the workbench's folder upload
in Chromium against a `markitai serve` debug build: no job before Convert, one
upload, and a refused submission that says why. `--nginx` also runs it through
nginx and checks that a refused 9 MB upload reaches the page as the server's 403
with its reason, not a 502. `--ui DIR` tests unbuilt sources from a
`node build.mjs` output; its docstring has the other options.

`uv run scripts/workbench_address_e2e.py` checks the provider form's API address
list (Z.ai, MiniMax, OpenAI at 1280 and 390 px: the entries, arrows, Enter,
Escape, typed text, taps) the same way; `--browser webkit` runs it in Safari's
engine and `--shots DIR` saves the open lists.

A ```` ```mermaid ```` block becomes SVG while the site builds
(`website/.vitepress/mermaid.ts`, Pele); a diagram Pele cannot draw fails the
build. `website/NOTICE` credits the site's build dependencies, separately from
the packages' `NOTICE`; a contract test fails when a `package.json` dependency
is missing from it.
