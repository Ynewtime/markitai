# Native CI and package acceptance

`scripts/check.py` runs the development gate without a Unix-shell dependency.
The workflow in `.github/workflows/native.yml` selects native host jobs. It runs
only when started by hand (`gh workflow run native.yml -f platforms='["windows-2025"]'`
or the Actions page), because the repository is private and hosted macOS minutes
are billed at ten times Linux minutes and Windows at twice: the `platforms`
input lists the runners (`windows-2025`, `windows-11-arm`, `ubuntu-24.04`,
`macos-15`, `macos-15-intel`; default Windows x86-64 and Linux), and
`minimum_rust` adds the declared-minimum toolchain check. A matrix entry
describes intended coverage; it is not evidence that the job has run or that its
artifacts work. Only a completed job and its retained logs establish that host's
result.

Runner contents change over time. Record the actual compiler host and version,
and retain the workflow's `Set up job` runner-image details. GitHub documents
those fields in its [hosted-runner guide](https://docs.github.com/en/actions/concepts/runners/github-hosted-runners);
the [official runner-images repository](https://github.com/actions/runner-images)
describes the images and installed software. A moving runner label does not pin
the installed SDK or codec version.

The procedures below define package acceptance; they do not report a completed
R49 release round. Development packages remain `1.3.0-dev` until Windows support
and final release acceptance are complete. Use each host's retained evidence for
its actual build, installation and test coverage.

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

The script builds the release workspace, executes an extracted CLI archive,
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
record says `executed.ran=false`. Running under Rosetta is not evidence from
Intel hardware.

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

The packagers record format and instruction set separately from executed
probes. Windows PE32+ console EXEs must have the expected COFF Machine
(`0x8664` for x64 or `0xaa64` for ARM64); DLLs, PE32, unknown machines or the wrong
OS format are rejected. Reading a valid executable header establishes identity,
not runtime compatibility. The native job's `evidence.json` or the standalone
`record.json` retains the actual commands, probe results and limitations; no
other host, full binding suite or release completion is implied.

## Package license files and provenance

CLI archives and C-ABI artifacts carry the repository's LICENSE and NOTICE.
CLI archives also carry the embedded Markdown renderer/sanitizer licenses and
provenance under `vendor/web/`.
Package attribution includes the five original notice/provenance files under
`licenses/hayro/` and the four under `licenses/paddleocr/`, with source bytes
verified in the archive and extracted installation. The PaddleOCR/RapidOCR
license texts are checked against their recorded provenance. These notices do
not include ONNX model files; model installation and offline use are described
in [local OCR](ocr.md).

The legacy ZIP retains both `markitai` and `mkai` executable files; report its
size separately from one runnable executable. On Unix it also carries a relative
`markitai-mcp` symlink. A separate `-single-binary.tar.gz` contains exactly one
regular executable, `markitai`, with relative `mkai` and `markitai-mcp` symlinks.
It includes the same project, web, pricing, upstream and Codex catalog attribution
as the ZIP. Its complete member inventory, executable hash, symlink targets and
license bytes are checked during private extraction; the extracted `mkai --version`
and `markitai-mcp --help` are executed. The MCP alias must select the subcommand,
not the main CLI help. Windows ZIPs contain three regular, directly executable
files: `markitai.exe`, `mkai.exe` and `markitai-mcp.exe`. `mkai.exe` is checked
against its separately built executable; its bytes need not equal `markitai.exe`.
`markitai-mcp.exe` is a byte-for-byte copy of the main CLI whose executable name
selects MCP directly, without a `.cmd` forwarder. Exact inventory, executable
and notice bytes are checked before extraction, including rejection of missing,
extra, duplicate or redirected entries; the installed aliases must also run.
Report the Windows ZIP's complete size, including all three executable entries,
separately from the Unix single-binary tar. A bare binary or a measurement-only
tar without these notices is not the complete distribution archive described
here.

Node staging explicitly includes the package attribution, including LICENSE,
NOTICE and the hayro/PaddleOCR notices, in its `files` list, then checks their
bytes inside the actual `.tgz` and after installation. Installed native
Node bytes must match the staged library.

The Python project does not yet declare all license files in its native package
metadata. The script therefore keeps maturin's original wheel, creates a separate
wheel with the same package attribution in `.dist-info/licenses`, including
LICENSE, NOTICE and the hayro/PaddleOCR notices, and rebuilds its RECORD
hashes and sizes. Existing Python/native bytes and METADATA are preserved. The
evidence labels this supplement and records both wheel hashes; the original is
never overwritten. The installed wheel's license files and native extension are
checked against those exact bytes. This is an explicit packaging step, not a
claim that the current pyproject has complete PEP 639 declarations. A later
packaging change should move this declaration into the normal build metadata.

`dependency-licenses.json` records Cargo's license metadata. It is an inventory,
not a completed redistribution review. The script does not sign or publish
packages, prove compatibility on another host, test every OS version, or claim
universal binaries. Helper unit tests use tiny authored archives and temporary
files; passing them does not substitute for actual installed-package acceptance.

The first actual macOS arm64 execution and the separate Rosetta limitation are
recorded in [round twenty](validation/media-cli-round20.md); the x86-64 macOS
release build, its full test run under Rosetta 2 and a universal-binary size
check are in [macOS x86-64 under Rosetta](validation/macos-x86_64-rosetta.md).
No remote matrix run is implied by those local results.

The compiled Codex capability catalog is Apache-2.0 data. `licenses/codex/`
contains the complete original license/copyright, modification description,
selected catalog and exact provenance. `scripts/codex_attribution.py` requires
the packaged catalog to equal the compiled catalog, and rejects changed hashes,
redirected source paths or missing files. The same verified bytes flow into CLI
ZIP/tar, C-ABI delivery, the wheel supplement, Node package and Go static archive.
The Go packager records its separate `codex_attribution` inventory alongside the
existing pricing and dependency notices. These helpers do not download runtime
executables, establish subscription entitlement or complete a legal review.

Static Go archives also retain the same nine hayro/PaddleOCR notice files and
record their identities in `licenses.json`. The static delivery script remains
limited to its native macOS ARM64 and Linux x86-64 glibc targets; this attribution
step does not add a Windows Go/cgo release or installed-package result.
