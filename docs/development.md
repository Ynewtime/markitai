# Development and recovery

Start at `docs/CONTROL.md`; inspect `git status --short --branch` before work.
The current delivery target is `main` for local user verification. Keep
`1.3.0-dev`; this round does not create a release, tag or publication.
Package manifests set the floors: Rust 1.92 (workspace `rust-version`, the
floor of the portable PDF renderer hayro), Python 3.10 (`bindings/python/pyproject.toml`),
Node.js 18 (`bindings/node/package.json`) and Go 1.23 (`bindings/go/go.mod`).
Native checkpoints use Rust 1.99.0, Python 3.13, Node.js 24 and Go 1.27.1.
The separate minimum-Rust CI lane checks core, CLI and FFI with Rust 1.92; it
does not run all tests or validate minimum Python/Node/Go runtimes. User-facing build and
installation steps are in the [quick start](quickstart.md) and
[bindings](bindings.md#installation).

`python scripts/check.py` runs the source gate with `MARKITAI_HOME` set to
`.local/test-home` on all hosts; `scripts/check.sh` delegates to it on Unix. It
runs the first three commands below and the `scripts/test_*.py` helper tests.
It isolates Markitai state, but inherits `HOME`; for local verification provide
a separate `HOME` as well, preserving explicit toolchain paths as needed.
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
so the AVX2 paths that dependencies select on most Intel Macs are not run. The
[Rosetta record](validation/macos-x86_64-rosetta.md) has the measurements.

Tests must set both `HOME` and `MARKITAI_HOME` to private directories under
`.local/` or a test-owned temporary directory. Use repository fixtures, authored
samples and loopback mock servers. Do not read real `~/.markitai`, personal
folders or credentials; do not send user identity information to external
services. Optional live-provider checks require separately supplied test
credentials and an explicit scope. Never print or commit credentials.

Before a checkpoint, inspect the diff, run relevant checks, then stage explicit
paths. A commit marks a recoverable state, not a claim of feature completeness.
Use `git log` and `git show` to recover an individual file into a new file for
comparison. Never discard working tree changes to recover an earlier version.

The original repository is not a build dependency. Any referenced differential
test runner must accept its location explicitly and record the reference SHA.

See [native CI](ci.md) for the clean-checkout package driver and its actual
installation checks. Configuring a runner is separate from executing it.
