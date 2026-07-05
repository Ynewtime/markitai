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

Releases are **manual** and driven by pushing a version tag. There is no
auto-generated release PR — you cut a release only when you intend to.

1. Bump `__version__` in `packages/markitai/src/markitai/__init__.py`. This is
   the **single source of truth** — hatch reads it at build time.
2. Add a `## [X.Y.Z] - YYYY-MM-DD` section to `CHANGELOG.md` (this becomes the
   GitHub Release notes verbatim).
3. Commit and push to `main` (e.g. `chore(release): v X.Y.Z`).
4. Tag and push:

   ```bash
   git tag vX.Y.Z && git push origin vX.Y.Z
   ```

Pushing the `vX.Y.Z` tag runs `.github/workflows/publish.yml`, which: runs the
test suite, builds the package, verifies the tag matches `__version__` (fails
loudly if you forgot to bump), publishes to PyPI via trusted publishing, and
creates the GitHub Release with the matching `CHANGELOG.md` section as notes.

To re-publish an existing tag (e.g. after a transient failure), run the
**Release** workflow manually from the Actions tab with the tag as input.
Commit-type discipline no longer affects releases, but keep using
[Conventional Commits](https://www.conventionalcommits.org/) — they keep the
history readable and make writing the changelog easier.
