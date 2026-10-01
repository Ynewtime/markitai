# Native CI and package acceptance

`scripts/check.py` runs the development gate without a Unix-shell dependency.
The workflow in `.github/workflows/native.yml` selects native host jobs. A matrix
entry describes intended coverage; it is not evidence that the job has run or
that its artifacts work. Only a completed job and its retained logs establish
that host's result.

Runner contents change over time. Record the actual compiler host and version,
and retain the workflow's `Set up job` runner-image details. GitHub documents
those fields in its [hosted-runner guide](https://docs.github.com/en/actions/concepts/runners/github-hosted-runners);
the [official runner-images repository](https://github.com/actions/runner-images)
describes the images and installed software. A moving runner label does not pin
the installed SDK or codec version.

## Running package acceptance

From a clean committed checkout, with Rust, Node/npm, Python with venv/pip,
`maturin>=1.9,<2`, and Go/cgo available:

```text
python scripts/ci_packages.py --expected-host <rustc-host-triple> --output .local/ci-artifacts
```

The output directory must not already exist. The script creates private state
under `.local`, sets `MARKITAI_HOME`, and leaves HOME unchanged. It removes
Python/Node module-path overrides so imports resolve to the installed packages.
`CARGO_BUILD_TARGET` is rejected: a foreign artifact cannot validate the current
Python/Node process. The target directory must be the repository's `target`,
because the Go source binding currently names `target/release` in its linker
directives. Build commands use `--locked`; dependency downloads are still allowed.

The script builds the release workspace, executes an extracted CLI archive,
packs and installs Node into a private consumer directory, and builds and
installs Python into a fresh venv. It then runs the actual binding tests.
Windows invokes npm's JavaScript CLI through `node.exe`, avoiding batch-file
execution and shell quoting. Unix hosts also run Go's source-package race tests;
Windows Go/cgo import-library distribution remains an explicitly recorded gap.

The evidence records the source revision and hashes every tracked and nonignored
source file before work, after the workspace build, after the wheel build, and
at completion. It rejects an initially dirty checkout and any observed source
change. File content hashes catch changes that leave Git's status text, size or
mtime unchanged. Source symlinks must target regular files inside the repository;
both link text and target contents are checked. These are boundary snapshots,
not a filesystem monitor that can detect an edit reverted between snapshots.

## Archives for another target

`scripts/package_cli_target.py --target <triple> --output <new directory>`
cross-builds only the release CLI for one non-host target (for example
`x86_64-apple-darwin` on Apple silicon) and writes the same
`-single-binary.tar.gz` as the native run, checked by the same member
inventory, hashes and attribution bytes. It records the executable's
instruction set read from its Mach-O or ELF header. When the host can execute
the target (x86-64 macOS under Rosetta 2), the archived `markitai`, `mkai` and
`markitai-mcp` run (version, MCP alias selection, a Markdown round trip and
`doctor --json`); otherwise the record says it did not run. Bindings are not
built, a host target or a Windows target is refused, and the same clean-checkout
and source-snapshot rules apply. Running under Rosetta is not evidence from
Intel hardware.

## Package license files and provenance

CLI archives and C-ABI artifacts carry the repository's LICENSE and NOTICE.
CLI archives also carry the embedded Markdown renderer/sanitizer licenses and
provenance under `vendor/web/`.
The legacy ZIP retains both `markitai` and `mkai` executable files; report its
size separately from one runnable executable. On Unix it also carries a relative
`markitai-mcp` symlink. A separate `-single-binary.tar.gz` contains exactly one
regular executable, `markitai`, with relative `mkai` and `markitai-mcp` symlinks.
It includes the same project, web, pricing, upstream and Codex catalog attribution
as the ZIP. Its complete member inventory, executable hash, symlink targets and
license bytes are checked during private extraction; the extracted `mkai --version`
and `markitai-mcp --help` are executed. The MCP alias must select the subcommand,
not the main CLI help. Windows ZIPs instead contain a small `markitai-mcp.cmd`
forwarder to the existing executable; byte validation is not Windows execution
proof. A bare binary or a measurement-only tar without these notices is not the
complete distribution archive described here.

Node staging explicitly includes LICENSE and NOTICE in the package's `files` list, then checks
their bytes inside the actual `.tgz` and after installation. Installed native
Node bytes must match the staged library.

The Python project does not yet declare all license files in its native package
metadata. The script therefore keeps maturin's original wheel, creates a separate
wheel with LICENSE and NOTICE in `.dist-info/licenses`, and rebuilds its RECORD
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
