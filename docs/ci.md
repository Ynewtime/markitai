# Native CI and package acceptance

`scripts/check.py` runs the development gate without a Unix-shell dependency.
The workflow in `.github/workflows/native.yml` selects native host jobs. It runs
only when started by hand (`gh workflow run native.yml -f platforms='["windows-2025"]'`
or the Actions page). The `platforms` input explicitly selects runners to
control resource use: (`windows-2025`, `windows-11-arm`, `ubuntu-24.04`,
`macos-15`, `macos-15-intel`; default Windows x86-64 and Linux), and
`minimum_rust` adds the declared-minimum toolchain check. Each requested
platform runs independent `checks` and `packages` lanes in parallel, with
`fail-fast: false` and a 120-minute timeout per lane. A matrix entry
describes intended coverage; it is not evidence that the job has run or that its
artifacts work. Only a completed job and its retained logs establish that host's
result.

Runner contents change over time. Record the actual compiler host and version,
and retain the workflow's `Set up job` runner-image details. GitHub documents
those fields in its [hosted-runner guide](https://docs.github.com/en/actions/concepts/runners/github-hosted-runners);
the [official runner-images repository](https://github.com/actions/runner-images)
describes the images and installed software. A moving runner label does not pin
the installed SDK or codec version.

The workflow pins Rust 1.99.1, Python 3.13, Node.js 24 and Go 1.27.3. Its
optional Rust 1.92 lane runs `cargo check --locked` for core, CLI and FFI on
Ubuntu; it does not execute the complete test or binding suites on that compiler.

Results apply to their recorded commit and target. No Release, tag or publishing
step is part of these drivers.

The workflow caches third-party Cargo dependencies separately by runner and
validation lane; compiler and dependency changes affect cache keys. First-party
and vendored build artifacts rebuild from source. Windows builds one optimized CLI and copies it to the three executable names;
all installed entry points remain tested. Package stages report elapsed
time and log size every 30 seconds, preserve full logs, and add a timing summary
to Actions. Build stages have a 75-minute limit, other stages 15 minutes, within
a shared 110-minute package budget. Silent linking alone is not a failure.
Actual cache benefit depends on the runner and cache availability; compare real
stage times before claiming a speedup.

## Checks

`.github/workflows/check.yml` runs `scripts/check.py` with cargo-nextest on the
six release platforms on every push to `main`, on pull requests (without the
Intel Mac), by hand (`gh workflow run check.yml -f platforms='["windows-2025"]'`)
and as the release workflow's `test` job. Runs on `main` save a Cargo cache per
platform (`target/` and Cargo's downloads), which every run restores, so only
the workspace crates build: third-party crates are keyed by their Cargo.lock
entries and the manifests (a release commit's version bump keeps the key), and
the vendored crates are reused only when `vendor/` is the tree the cache was
built from (`scripts/ci_cache.py`). `main` restores only a cache with its own
dependencies, so a saved cache never carries dropped crates; a branch or tag
falls back to the newest one. Only a passing check saves; `-f save_cache=true`
also saves on a branch, for that branch alone (to time a warm run before a merge).

## Running package acceptance

From a clean committed checkout, with Rust, Node/npm, Python with venv/pip,
maturin 1.15.0 (the version `bindings/python/pyproject.toml` pins; the driver
refuses another), and Go/cgo available:

```text
python scripts/ci_packages.py --expected-host <rustc-host-triple> --output .local/ci-artifacts
```

The output directory must not already exist. The script creates private state
under `.local`, isolates `HOME`, `MARKITAI_HOME` and temporary directories, and
preserves explicit `CARGO_HOME`/`RUSTUP_HOME` for toolchain discovery. It uses an
empty npm user configuration and removes Python/Node module-path overrides so
imports resolve to the installed packages.
`CARGO_BUILD_TARGET` is rejected: a foreign artifact cannot validate the current
Python/Node process. The target directory must be the repository's `target`,
because the Go source binding currently names `target/release` in its linker
directives. Build commands use `--locked`; dependency downloads are still allowed.

The script builds release libraries and CLI without an unused Python prebuild,
executes an extracted CLI archive,
packs and installs Node into a private consumer directory, and builds and
installs Python into a fresh venv. It then runs the actual binding tests.
Windows invokes npm's JavaScript CLI through `node.exe`, avoiding batch-file
execution and shell quoting. Unix hosts also run Go's source-package race tests;
Windows Go/cgo import-library distribution remains an explicitly recorded gap;
a Windows CLI, Node or Python result does not establish Windows Go/cgo release
installation support.

Installed CLI probes check version/alias behavior, a Unicode Markdown round
trip, real MCP initialization and tool listing, and `doctor --json`. The doctor
probe records a valid exit 0 or 1 and its configuration readiness separately
from archive correctness; it does not run `--fix` or install optional components.
These probes do not establish full OCR/Office quality or complete feature
coverage.

The evidence records the source revision and hashes every tracked and nonignored
source file before work, after the workspace build, after the wheel build, and
at completion. It rejects an initially dirty checkout and any observed source
change. File content hashes catch changes that leave Git's status text, size or
mtime unchanged. Source symlinks must target regular files inside the repository;
both link text and target contents are checked. These are boundary snapshots,
not a filesystem monitor that can detect an edit reverted between snapshots.

## Archives for another target

On a Unix host, `scripts/package_cli_target.py --target <triple> --output <new directory>`
cross-builds only the release CLI for one non-host Unix target (for example
`x86_64-apple-darwin` on Apple silicon) and writes the same
`-single-binary.tar.gz` as the native run, checked by the same member
inventory, hashes and attribution bytes. A Unix host target is still handled by
`ci_packages.py`; this standalone script does not build bindings.

An Apple target builds for macOS 11.0, the minimum the arm64 build records,
unless `MACOSX_DEPLOYMENT_TARGET` is set (Rust's x86-64 default, 10.12, would name
systems nothing was tested on). The minimum the executable records and whether
it carries a code signature are read back into the record. When the host can
execute the target (x86-64 macOS under Rosetta 2), the extracted CLI/aliases run
version/help, Unicode Markdown, MCP protocol and doctor probes; otherwise the
record says `executed.ran=false` and its status is `built-not-executed`, not
`passed`. Running under Rosetta is not evidence from Intel hardware. Each step
is stopped with its process tree after 75 minutes (build) or 15 minutes
(other steps) and the run fails.

### Native Windows CLI

For a native Windows CLI-only ZIP, use the same script on the matching Windows
Rust host:

```text
python scripts/package_cli_target.py --target <native-windows-msvc-triple> --output .local/windows-cli
```

The only Windows targets are `x86_64-pc-windows-msvc` and
`aarch64-pc-windows-msvc`, and the requested target must equal `rustc -vV`'s host.
An ARM64 host running an x64 executable does not count as an ARM64 package.
Cross-host Windows packaging, other Windows targets and Unix targets on a
Windows host are refused. The Windows output is
`markitai-<version>-<target>.zip`; a native Windows run must execute the installed
probes to pass. Both paths require a new output directory, a clean committed
checkout and unchanged source snapshots, and use isolated home/state/tmp.

Windows CLI builds use a dedicated target directory with `+crt-static`; this
flag is not applied to Python/Node/FFI dynamic libraries or host proc macros.
The packagers inspect imports of all three CLI EXEs and reject dynamically
linked MSVC runtime DLLs, then execute the extracted entries. This verifies the
CLI runtime linkage, not the absence of all system dependencies or optional
applications on a clean machine.

The packagers record format and instruction set separately from executed
probes. Windows PE32+ console EXEs must have the expected COFF Machine
(`0x8664` for x64 or `0xaa64` for ARM64); DLLs, PE32, unknown machines or the wrong
OS format are rejected. Reading a valid executable header establishes identity,
not runtime compatibility. The native job's `evidence.json` or the standalone
`record.json` retains the actual commands, probe results and limitations; no
other host, full binding suite or release completion is implied.

## Package inventory and attribution

Unix single-binary tar archives contain one `markitai` executable and relative
`mkai`/`markitai-mcp` symlinks. Legacy ZIPs contain separate CLI entries and have
a different total size. Windows ZIPs contain three regular EXEs; the MCP name
selects stdio mode directly. Packagers reject missing, extra, duplicate or
redirected entries and run the installed aliases. Report complete archive size
separately from a single executable.

CLI, FFI and language packages carry project and upstream attribution,
including `licenses/hayro/`, `licenses/paddleocr/`, `licenses/libwebp/`, `licenses/rav1d/`, pricing and
Tailwind palette (`licenses/tailwind/`) notices. CLI archives also include `vendor/web/` attribution and the explicit
public Markdown whitelist in `scripts/cli_documentation.py`; release wheels, which serve the same workbench,
carry the same `vendor/web/` attribution under `.dist-info/licenses/` (`release_build.py licenses` compares the
sets). Models and browser
runtimes are not included. The drivers verify source, archive and installed
notice bytes rather than trusting a filename alone.

Python packaging preserves maturin's original wheel and creates a separate
supplement with notices under `.dist-info/licenses`, rebuilding RECORD hashes
without replacing Python/native bytes or METADATA. Node staging adds attribution
to its package file list and verifies the actual archive and installed addon.
A raw local `maturin build` or `npm pack` does not perform these supplements.

Static Go is a separate native-host procedure in `scripts/package_go_static.py`
for macOS ARM64 and Linux x86-64 glibc; see [bindings](bindings.md#static-go-package).
Its source/binary manifest, notice collection, race tests and relocated consumer
checks are distinct from ordinary package acceptance. There is no Windows
static-Go delivery in this workflow.

`dependency-licenses.json`/`licenses.json` record mechanical attribution and
any unresolved texts; they are not a legal review. None of these drivers signs,
publishes, proves another architecture works, or tests every OS version.

## Website

Two workflows serve markitai.dev from `website/` (see
[development](development.md#website)); neither builds or publishes packages.

- `deploy-website.yml` builds the site and deploys it to GitHub Pages on a push
  to `main` that changes `website/**`, `scripts/install/**` (served as
  `/setup.sh` and `/setup.ps1`), `CHANGELOG.md`, `CHANGELOG.zh.md` or the
  workflow, and when started by hand on `main` (a run on another branch skips
  both jobs). The changelog pages publish a section once its `vX.Y.Z` tag
  exists, and a release tag starts no deploy, so the release workflow's
  `website` job starts this one on `main` after a tag release (see
  [the release commit](#the-release-commit)).
- `website-check.yml` runs on pushes to `main` and pull requests that change
  `website/**`, `scripts/install/**`, `CHANGELOG.md`, `CHANGELOG.zh.md`, the CLI
  or core crates, `Cargo.lock` or the workflow, and by hand. It fails on a
  release-notes section that would be lost (an untagged one in
  `website/changelog/`, or below a released one in `CHANGELOG*.md`; the top of
  those may wait for its tag) or that only one language publishes, builds the
  CLI (debug) and runs `website/tests` against it, and builds the site for pull
  requests and manual runs.

## Releases

Which number a release raises is decided by [versioning](versioning.md).
`.github/workflows/release.yml` builds, tests and publishes a release in one
run on GitHub-hosted runners. Pushing a `v2.*` tag publishes; starting it by
hand (`gh workflow run release.yml`) is a dry run that builds and tests
everything and publishes nothing (`-f dry_run=false` publishes only when run on
a tag). It uses no stored secrets, and its builds no caches (the `test` job
restores the [check](#checks) cache saved by runs on `main`). Each step runs a subcommand of
`scripts/release_build.py`, which also runs locally:

| Job | What it does |
|---|---|
| `prepare` | Checks the tag names the Cargo version and that `bindings/python/pyproject.toml` and `server.json` agree, that a published tag is on `main`, and writes the release notes: the tag's `## [X.Y.Z]` section of `CHANGELOG.md` with a link to the full changelog, and the same from `CHANGELOG.zh.md` (a release without either fails). A section over 100,000 characters is represented by its lead, the summary between the heading and its first `###`; without one, the run fails here, dry runs too (GitHub refuses a Release body over 125,000 characters). |
| `test` | [`check.yml`](#checks) on the six release platforms; a dry run leaves out the Intel Mac unless started with `-f intel_mac_tests=true` (`main`'s check runs it, and `smoke-intel` runs the x86_64 macOS artifacts there). A tag run always runs all six. |
| `cli` | `markitai-v<version>-<target>.tar.gz` (`.zip` on Windows) for `aarch64-apple-darwin`, `x86_64-apple-darwin`, `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`, `x86_64-pc-windows-msvc` and `aarch64-pc-windows-msvc`, built natively, then extracted and run (version, aliases, Markdown round trip, MCP tools, `doctor --json`). The `x86_64-apple-darwin` one is cross-compiled on the arm64 macOS runners (same release profile and macOS 11 minimum, about twice as fast) and run by `smoke-intel`. |
| `wheel` | One abi3 wheel per platform (`cp310-abi3`, `macosx_11_0`, `manylinux_2_28`, `win_amd64`, `win_arm64`) with the CLI archives' license set, installed into a new venv: its three commands, the Python API and `bindings/python/tests`. The `x86_64-apple-darwin` wheel is cross-compiled like its archive and installed by `smoke-intel`. |
| `smoke-intel` | On an Intel Mac, the same install-and-run checks for the cross-compiled `x86_64-apple-darwin` archive and wheel; it builds nothing. |
| `sdist` | The source distribution, then a wheel pip builds from it alone. |
| `github-release` | A check that each target's archive and wheel carry the same license files, `SHA256SUMS` over every asset, build-provenance attestations, and the Release with the notes (also its `RELEASE-NOTES.md` and `RELEASE-NOTES.zh.md` assets, which `markitai update` and the workbench read), created as a draft with all assets and then published. |
| `pypi` | The six wheels, not the sdist, by trusted publishing from the `pypi` environment (the action also uploads PEP 740 attestations). On a platform without a wheel pip has nothing to install (PyPI no longer carries 1.x) rather than building 2.x from source; the sdist is on the GitHub Release for those who do. |
| `mcp-registry` | `server.json` (`io.github.Ynewtime/markitai`, pointing at the PyPI release) through `mcp-publisher` with GitHub OIDC, once PyPI lists the version. |
| `website` | Starts `deploy-website.yml` on `main` (with `actions: write` only) once `github-release` and `pypi` have succeeded, so markitai.dev's changelog pages show the new section only when the archives and wheels its install commands fetch are out. It does not wait for `mcp-registry`. |

The archives keep the package inventory above, at the archive root. macOS builds
target macOS 11.0. Linux archives and wheels are built in the pinned
`manylinux_2_28` image (`scripts/release_container.sh`), so they need glibc 2.28
or later; the build refuses an executable that asks for more, and the image also
installs the wheel with Python 3.10. musl (Alpine) is not a release target.
Windows CLIs carry the static CRT. Nothing is signed or notarized.

maturin's sdist omits the crates the workspace patches in from `vendor/`
(`[patch.crates-io]`), so `release_build.py sdist` adds them and the
attribution files to it.

If a tag run fails, fix the cause and use **Re-run failed jobs** on that same
run, within the artifacts' 14 days: the re-run reuses the built artifacts, a
published Release with the same `SHA256SUMS` passes, and PyPI skips files it
already has. Do not start a new run (`gh workflow run release.yml --ref vX.Y.Z
-f dry_run=false`) after anything was published: it rebuilds every artifact,
the archives and wheels are not byte-reproducible, and `github-release` then
stops with "already published with other assets", so PyPI and the MCP Registry
are never reached. A failure before `github-release` published nothing; a fix
that needs a new commit needs a new release commit and tag.

A user checks a download with `sha256sum -c SHA256SUMS --ignore-missing` and
its origin with `gh attestation verify <file> --repo Ynewtime/markitai`.

### The release commit

A release is one commit on `main` that sets the version, then the tag on it.
On macOS or Linux (the help snapshot test is Unix-only):

1. `python3 scripts/release_build.py prepare-version X.Y.Z` writes the version
   into `Cargo.toml`, `Cargo.lock` (`cargo update --workspace`; `--locked`
   builds fail on a stale lock), `bindings/python/pyproject.toml`,
   `bindings/node/package.json` and `server.json`, regenerates
   `crates/markitai-cli/tests/fixtures/help_en.txt` (it prints the version),
   and lists every tracked line that still names the old version. Tests,
   `AGENTS.md`, the changelogs, `website/` and the private workbench package are
   exempt.
2. Correct the listed lines, chiefly the guides the archives ship (README,
   `docs/index.md`, `docs/quickstart.md`, `docs/cli.md`), until the command
   finishes clean (`git checkout -- .` undoes a trial run).
3. Move `.local/changelog-unreleased.md` and `.local/changelog-unreleased.zh.md`
   into `CHANGELOG.md` and `CHANGELOG.zh.md` as `## [X.Y.Z] - YYYY-MM-DD`. When
   the English section exceeds 100,000 characters, write a short summary
   between the heading and its first `###`: the Release shows it with a link to
   the full section. `python3 scripts/release_build.py notes --version X.Y.Z
   --output notes.md` shows the Release text and fails where `prepare` would.
4. `python3 scripts/release_build.py version --tag vX.Y.Z` and
   `python3 scripts/check.py`, commit, push `main`, run a dry run on it
   (`gh workflow run release.yml --ref main`), then push an annotated
   `vX.Y.Z` tag on that commit.
5. The tag run's `website` job deploys markitai.dev, which puts the new
   section on its changelog pages (the push in step 4 built them before the
   tag existed). If that job fails, re-run it, or run
   `gh workflow run deploy-website.yml --ref main` by hand.

### Setup scripts

`scripts/install/setup.sh` (macOS, Linux) and `scripts/install/setup.ps1`
(Windows PowerShell 5.1 and PowerShell 7, x64 and ARM64) install the release
archive for this machine: the latest release, or `MARKITAI_VERSION`. They read
the asset names from the release's `SHA256SUMS`, verify the archive against it,
and install `markitai`, `mkai` and `markitai-mcp` to `~/.local/bin` (notices and
guides in `~/.local/share/markitai/current`) or to
`%LOCALAPPDATA%\Programs\markitai`, which setup.ps1 adds to the user Path. The
new files are staged in the install folder first; only then is a Python
markitai 1.x that uv installed uninstalled and the new release renamed into
place, and a failed Windows update puts the previous release back. A receipt
(`~/.local/share/markitai/current/receipt`, or `receipt` in the Windows install
folder) lists what they installed: updates and uninstall remove only those
names, setup.ps1 refuses a folder without its receipt that already holds one of
the release's names, and removes the user Path entry only when it added it. A
run without `MARKITAI_INSTALL_DIR` reuses the last install's folder (setup.ps1
records a custom one in `%LOCALAPPDATA%\markitai\install-location`). With a
terminal they offer `markitai doctor --fix` (OCR models, a headless Chromium
when no browser is found) when `doctor --json` marks a check `fixable`, on
every run (no answer is kept; components already there need no question);
`MARKITAI_INSTALL_OPTIONAL=1` runs it without one, `0` never. `--uninstall` (setup.ps1: `-Uninstall`; both: `MARKITAI_UNINSTALL=1`)
removes what they installed and keeps `~/.markitai`. setup.ps1 runs in a script
block of its own, so `irm | iex` leaves the user's session as it was.
`MARKITAI_DOWNLOAD_URL` points them at another releases base, which
`scripts/test_setup_scripts.py` uses with a local fake release.
