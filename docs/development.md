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
`Cargo.toml`, `Cargo.lock` or `vendor/` changed; the CLI's when a workflow
changed; none for workbench sources or
scripts alone), clippy and formatting when a crate changed, and all helper tests. Run the full gate before a merge or
a release. The CLI's integration tests build as two binaries: `update` and
`integration` (`crates/markitai-cli/tests/integration.rs` lists the other test files).
One of those files, `tests/workflows.rs`, reads `.github/workflows/` as GitHub
does before it runs anything: each file is YAML, and the jobs a job needs, the
inputs it gives a workflow it calls, and the steps, inputs, matrix entries and
needed jobs' outputs its expressions name are there. It does not check what an
expression computes, or an action's own inputs.
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
python3 scripts/desktop_web_install.py run --package PACKAGE [--evidence DIR]
python3 scripts/desktop_web_install.py convert --pid PID --origin http://127.0.0.1:PORT
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
the manylinux_2_28 image builds (after `mygo build`, the script installs the
menu entry of every package as `dev.markitai.web.desktop`, with
`Comment[zh_CN]`, `Keywords` and `StartupNotify` and the window's class, and
fails where mygo writes them otherwise; it keeps the Debian package's members
GNU tars, as dpkg reads no pax header), or `Markitai Web.exe` and the installer
`markitai-web-setup-<version>-amd64.exe` in `desktop/web/build/windows-amd64/`
on x86-64 Windows ([Windows](desktop-web.md#windows)). No
artifact may be named like a CLI archive (`markitai-v…`). `--e2e` builds with
the `e2e` tag, mygo's automation plugin and the web inspector, without
packages or installer, into `desktop/web/build/e2e/`, from a generated project
(`desktop/web/build/e2e-project/`) whose identifier is the app's
(`dev.markitai.web`) plus `.e2e`: on macOS web data is kept by identifier
outside `HOME`, on Linux under the app's name, which is that identifier
there, and an e2e run must not share an installed app's.
`desktop_web_e2e.py` drives that build over the automation socket through the
scenarios its docstring lists, each with its own
`HOME` (with the XDG directories on Linux, and `USERPROFILE`, `APPDATA`,
`LOCALAPPDATA` and `TEMP` on Windows), `MARKITAI_HOME` and app data, and
keeps the evidence in `.local/desktop-web-e2e/ROUND/`; the bundled `markitai` must
have `serve --supervised`. On Linux it runs in a Hyprland session and quits by
closing the window with `hyprctl`, as there is no application menu, also
where a quit must ask (SIGTERM there is the session's end, which asks
nothing); on
Windows it runs in a desktop session and closes the window with `WM_CLOSE`
(signals become window messages and `TerminateProcess`, the automation socket
is Winsock's `AF_UNIX`). The native Save panel and quit question are answered
by the e2e build's environment on both, and a link it would open in the
browser is recorded instead, so check them by hand once. Building the e2e
project without the symlink privilege links its resources with a junction.
A release build has no automation socket, so an installed app is checked
through its server: `desktop_web_install.py convert` reads the token from the
environment of the `markitai serve` the app started (PID; the app gives the
token to no one else), sends a small HTML page to `POST /api/jobs` at the
address `serve.log` names, as the workbench does, and fails unless the
finished item's Markdown is the page's.
`desktop_web_install.py run` is the whole check of a release build's package
on macOS and Linux: it installs the package (and does not start when the app
is installed or runs already), starts the app with a `HOME` and
`MARKITAI_HOME` of its own, and fails unless the bundled `markitai serve`
listens, the app has a window on the screen, that server converts the page,
and the app quits with 0 leaving its server drained; then it removes the app.
It opens the app's window, so run it where that disturbs no one. On macOS the
package is the disk image, whose `Markitai Web.app` it copies to
`/Applications`; the image was never downloaded, so Gatekeeper, which would
stop an app that is not signed for other Macs, is not part of the check. On
Linux the package is the `.deb`, which `apt-get` installs with what it depends
on and no recommended package and removes afterwards (as root, through `sudo`
when the check is not root; what it depends on stays), or the `.tar.gz`,
which the `install.sh` beside it installs in the check's `HOME` and
`install.sh --uninstall` removes, leaving the app's settings. There the app
runs on the X display `DISPLAY` names, where the check finds its window and
closes it as a window manager does for a user.
`scripts/test_desktop_web_build.py` and `scripts/test_desktop_web_install.py`
run with the helper tests.

`.github/workflows/desktop-web.yml` builds the app on hosted runners, in a job
for each system its `platforms` input names: `windows-amd64` (`windows-2025`),
`darwin-arm64` (`macos-15`) and `linux-amd64` (`ubuntu-24.04`, its archive
built in the manylinux_2_28 image as `release.yml` builds it). A push to
`desktop/web/**`, `desktop/icon.png`, `scripts/desktop_web_*.py` or the built
workbench (`crates/markitai-cli/src/server/web/dist/**`) runs `windows-amd64`
alone; a run by hand takes the list (`gh workflow run desktop-web.yml -f
platforms='["darwin-arm64"]'`). Each job builds the CLI archive, or takes it
from the cache, where it is kept under `desktop_web_build.py archive-key`: the
files an archive is built from (the Rust workspace and what it compiles in,
the attribution and guides the archive carries, the scripts that build it), so
a change to the app, another guide, a test of the scripts or a workflow keeps
the archive, whose build commit is then an earlier one. The job then builds the
app with its package, installs and launches the
app, converts a file through the server it started, closes and removes it, and
keeps the evidence and the package for 7 days: on Windows in the workflow's own
step, which calls `desktop_web_install.py convert`, on macOS with
`desktop_web_install.py run`, and on Linux with `run` on Xvfb's display, once
for the `.deb` and once for the `.tar.gz` with `install.sh`. The Windows job
also builds the e2e build and runs a full `desktop_web_e2e.py` round; the macOS
and Linux jobs do not. A dry run of `release.yml` ([releases](ci.md#releases))
calls the workflow for the three systems with `archive: artifact`: each job
then takes its archive from that run's `cli` job instead of building or
restoring one, and makes no e2e build.

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
