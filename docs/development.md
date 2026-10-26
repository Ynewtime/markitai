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
tests (several files at once) run beside the Rust tests and clippy, and their
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
`PATH` by an absolute directory. `MARKITAI_CHECK_SKIP="name other"` passes each
space-separated name to the Rust tests as `--skip name` (e.g. Vision OCR in a macOS VM).

For branch work, `python scripts/check.py --quick` checks only what the files
changed since the branch left `origin/main` (committed or not; `--base REF`
compares with another branch) can affect: the tests and clippy of the changed
crates and of every crate depending on them (all crates when `Cargo.toml`,
`Cargo.lock` or `vendor/` changed; none for workbench sources or scripts
alone), formatting, and all helper tests. Run the full gate before a merge or
a release. The CLI's integration tests build as one binary
(`crates/markitai-cli/tests/integration.rs` lists the CLI's test files).
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
