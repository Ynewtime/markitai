# Contributing to Markitai

## Development setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11–3.14.

```bash
git clone https://github.com/Ynewtime/markitai.git
cd markitai
uv sync --all-extras          # install workspace + all optional extras
uv run pre-commit install     # ruff on commit
uv run pre-commit install --hook-type pre-push   # pyright + tests on push
```

This is a uv workspace: the published package lives in `packages/markitai`,
the docs site in `website/` (VitePress + pnpm), install scripts in `scripts/`.

## Everyday commands

```bash
uv run pytest -q                          # full suite (parallel, excludes slow/network)
uv run pytest -m "slow or network"        # opt-in slow/network tests
uv run ruff check && uv run ruff format   # lint + format
uv run pyright packages/markitai/src     # type check (0 errors required)
uv run bandit -c pyproject.toml -r packages/markitai/src -q   # security lint
uv run markitai <file>                    # run the CLI from source
```

Test markers: `slow`, `network`, `parity` (see `pyproject.toml`). CI runs the
default selection on Linux/macOS/Windows × Python 3.11–3.14.

## Conversion-quality benchmark

`packages/markitai/benchmarks/` holds a dev-only harness that scores the
HTML→Markdown pipeline against the defuddle ground-truth corpus
(`tests/defuddle_fixtures/`) with a marker-style heuristic scorer
(rapidfuzz fuzzy block alignment, 0–100 per fixture). Unlike the parity
tests, it measures continuous quality drift rather than pass/fail:

```bash
uv run python packages/markitai/benchmarks/webextract_quality.py   # full corpus
```

It prints per-fixture scores with deltas vs the committed
`benchmarks/results/baseline.json` and writes `benchmarks/results/latest.json`
(gitignored). Run it before/after extraction changes; regenerate the baseline
deliberately with `--update-baseline` when a quality change is intentional.
The full-corpus run is manual/CI-cron only — a fast smoke test
(`tests/unit/test_webextract_quality_benchmark.py`) covers the scorer math.

## Conventions

- Match surrounding code style; ruff (`E,W,F,I,B,C4,UP,ARG,SIM`) and pyright
  (basic mode) run in CI and must be clean.
- Logging uses loguru with `{}`-style formatting (`logger.info("x={}", x)`),
  never printf-style `%s`.
- Google-style docstrings; English comments.
- Every bug fix ships with a regression test.

## Releasing

Releases are automated with
[release-please](https://github.com/googleapis/release-please)
(config in `.github/release-please-config.json`):

1. Use [Conventional Commits](https://www.conventionalcommits.org/) on `main` —
   they drive versioning and the changelog: `fix:` bumps patch, `feat:` bumps
   minor, `feat!:`/`BREAKING CHANGE:` bumps major; `chore:`, `docs:`, `ci:`
   etc. don't trigger a release.

   **Reserve `feat:`/`fix:` for the published package** (`packages/markitai/src`
   or its `pyproject.toml` deps) — that is the only code shipped in the wheel.
   Changes to the installer scripts (`scripts/`), docs/website, CI, or tests are
   **not** in the package, so type them `chore:`/`docs:`/`ci:`/`test:` so they
   don't bump the package version. As a safety net, `release-please.yml` only
   runs on `packages/markitai/src/**` and `pyproject.toml` changes, so a docs- or
   installer-only push won't open or churn a release PR even if mis-typed.
2. `release-please.yml` keeps a release PR open that bumps `__version__` in
   `packages/markitai/src/markitai/__init__.py` — still the **single source
   of truth** (hatch reads it at build time; keep the
   `# x-release-please-version` annotation on that line) — and prepends the
   generated section to `CHANGELOG.md`.
3. To release, merge the release PR. release-please creates the `vX.Y.Z` tag
   and GitHub release, then dispatches `publish.yml` on that tag, which runs
   tests, verifies the tag matches the built version, and publishes to PyPI
   via trusted publishing.
