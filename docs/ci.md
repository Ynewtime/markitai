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

The workflow pins Rust 1.99.0, Python 3.13, Node.js 24 and Go 1.27.1. Its
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

## Running package acceptance

From a clean committed checkout, with Rust, Node/npm, Python with venv/pip,
`maturin>=1.9,<2`, and Go/cgo available:

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
including `licenses/hayro/`, `licenses/paddleocr/`, pricing and Codex catalog
notices. CLI archives also include `vendor/web/` attribution and the explicit
public Markdown whitelist in `scripts/cli_documentation.py`. Models and browser
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
