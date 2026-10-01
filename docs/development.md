# Development and recovery

Start at `docs/CONTROL.md`; inspect `git status --short --branch` before work.
Package manifests set the floors: Rust 1.89 (workspace `rust-version`, due to
resolved dependencies), Python 3.10 (`bindings/python/pyproject.toml`),
Node.js 18 (`bindings/node/package.json`) and Go 1.23 (`bindings/go/go.mod`).
Checkpoints are verified with current toolchains (Rust 1.98.1, Python 3.13,
Node.js 24, Go 1.27), not with the minimum versions. User-facing build and
installation steps are in the [quick start](quickstart.md) and
[bindings](bindings.md#installation).

`python scripts/check.py` runs the source gate with `MARKITAI_HOME` set to
`.local/test-home` on all hosts; `scripts/check.sh` delegates to it on Unix. It
runs the first three commands below and the `scripts/test_*.py` helper tests.
Build the optimized CLI separately:

```sh
cargo fmt --all --check
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo build --release -p markitai-cli
```

The binding suites run against built artifacts; see
[bindings](bindings.md). [Native CI](ci.md) builds, installs and tests every
package from a clean checkout.

## Intel macOS from Apple silicon

With `rustup target add x86_64-apple-darwin`, the same checks run for x86-64
macOS; the test executables run under Rosetta 2:

```sh
cargo clippy --workspace --all-targets --locked --target x86_64-apple-darwin -- -D warnings
cargo test --workspace --locked --target x86_64-apple-darwin
cargo build --release --locked --target x86_64-apple-darwin -p markitai-cli
```

Rosetta is not an Intel Mac. Under it, Vision text recognition fails without an
error and CoreGraphics' rejection of a malformed PDF crashes the process
intermittently; the affected tests check the explicit Rosetta error or skip
that one call, and say so on stderr. Rosetta exposes no AVX, AVX2, FMA or BMI,
so the AVX2 paths that dependencies select on most Intel Macs are not run. The
[Rosetta record](validation/macos-x86_64-rosetta.md) has the measurements.

Tests must set `MARKITAI_HOME` to a private directory under `.local/` or a
temporary directory. Real-provider tests are opt-in. If needed, explicitly
import environment variables from `~/.markitai/.env` into an isolated test
process; never source it as shell code, print its values, copy caches, or write
back. Offline fixtures and localhost mock servers are the default.

Before a checkpoint, inspect the diff, run relevant checks, then stage explicit
paths. A commit marks a recoverable state, not a claim of feature completeness.
Use `git log` and `git show` to recover an individual file into a new file for
comparison. Never discard working tree changes to recover an earlier version.

The original repository is not a build dependency. Any referenced differential
test runner must accept its location explicitly and record the reference SHA.

See [native CI](ci.md) for the clean-checkout package driver and its actual
installation checks. Configuring a runner is separate from executing it.
