# Contributing to Markitai

## Development setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11–3.13.

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
uv run pyright packages/markitai/src packages/markitai/tests   # type check (0 errors required, same scope as CI)
uv run lint-imports                       # architecture layering contracts (0 broken required)
uv run bandit -c pyproject.toml -r packages/markitai/src -q   # security lint
uv run markitai <file>                    # run the CLI from source
```

Test markers: `slow`, `network`, `parity` (see `pyproject.toml`). CI runs the
default selection and an isolated built-wheel install smoke test on
Linux/macOS/Windows × Python 3.11–3.13.

Architecture rule: modules below `markitai.cli` must never import it (and the
other layering contracts in `[tool.importlinter]`, root `pyproject.toml`).
The contracts' `ignore_imports` allowlist only shrinks — fix the dependency
direction instead of adding an exemption. When code below the CLI needs the
user (a prompt, a notice), go through `markitai.ports`.

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

## Document conversion snapshot guardrail

`packages/markitai/benchmarks/docs_snapshot.py` freezes markitai's default
(no LLM, no OCR, no screenshot) conversion output for a small, stable fixture
set — one PDF/DOCX/PPTX/XLSX fixture per pure-Python converter path under
`tests/fixtures/` — and fails when a later change shifts it. Unlike the
webextract quality benchmark (a continuous fuzzy score, because HTML
extraction has no single "correct" answer), document conversion for a fixed
input is deterministic, so this is a plain normalize-then-exact-match
snapshot against `benchmarks/docs_snapshots/expected/*.md`. Normalization
strips environment noise (timestamps, version strings, absolute paths) so
the comparison survives running on a different machine or day; see the
module docstring for the exact patterns.

```bash
uv run python packages/markitai/benchmarks/docs_snapshot.py            # compare
uv run python packages/markitai/benchmarks/docs_snapshot.py --update   # regenerate
```

`tests/unit/test_docs_snapshot_guardrail.py` runs the same comparison as a
fast (~seconds), network-free pytest check, so it is part of the default
suite and therefore CI with no extra workflow wiring. When a change
intentionally shifts conversion output, regenerate deliberately with
`--update` and review the diff like any other code change — the same
discipline `webextract_quality.py --update-baseline` uses for its baseline.
Legacy `.doc`/`.ppt`/`.xls` fixtures (LibreOffice/MS Office CLI conversion)
and OCR/screenshot paths are deliberately out of scope: their output can
vary by installed tool version across CI runners, a bad fit for an
exact-match snapshot.

## Syncing the defuddle fixture corpus

`scripts/sync_defuddle_fixtures.sh /path/to/defuddle` refreshes the defuddle
parity corpus in `tests/defuddle_fixtures/` from a local clone of the upstream
[defuddle](https://github.com/kepano/defuddle) repo — it copies that repo's
`tests/fixtures/*.html` and `tests/expected/*.md` and records the source commit
in `VERSION`. Run it only when deliberately refreshing the corpus against a
newer defuddle; both the parity tests and the quality benchmark read these
fixtures, so a resync can shift scores. The corpus was last synced 2026-03-23
(see `tests/defuddle_fixtures/VERSION`). The script also rewrites the pin in
`src/markitai/webextract/PORT_MANIFEST.md` (the upstream→port module map);
a unit test keeps the two pins equal, and the weekly
`.github/workflows/defuddle-watch.yml` opens an issue when upstream cuts a
release ahead of the pin.

## olmOCR-bench feasibility harness (not CI)

`scripts/olmocr_bench_subset.py` is an opt-in, network-using script that
scores markitai's `--ocr` output against a subset of AI2's
[olmOCR-bench](https://huggingface.co/datasets/allenai/olmOCR-bench) dataset
(present/absent/order rule types only — table/math rules need the official
`olmocr[bench]` toolkit; see the script's module docstring for the full
feasibility writeup, licensing, and manual steps for an authoritative run).
It downloads a small PDF slice directly over HTTPS (no new dependency) and
scores markitai's own conversion:

```bash
uv run python scripts/olmocr_bench_subset.py --split old_scans --limit 5
```

Never run in CI or the default test suite; `tests/unit/test_olmocr_bench_subset.py`
covers the scoring math offline.

## LLM enhancement A/B evaluation

`packages/markitai/benchmarks/llm_ab_eval.py` measures what markitai's LLM
enhancement (`llm=True` / `--llm`) actually buys: a blind, position-debiased
A/B judge compares base vs. enhanced conversions of the same documents (see
the module docstring for the full methodology — position-swap x2 debiasing,
per-format aggregation, jsonl resume/checkpointing, and an optional Batches
API path at 50% of list price on OpenAI and Anthropic). It costs real money
once you supply `--judge-model` with live credentials — run `--dry-run`
first for a cost estimate; nothing in this repository ever calls a real
judge model:

```bash
uv run python packages/markitai/benchmarks/llm_ab_eval.py \
  --docs report.pdf memo.docx --output /tmp/ab.jsonl --dry-run
```

`tests/unit/test_llm_ab_eval.py` covers the harness with a stubbed judge.

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
   the **single source of truth** for the published package — hatch reads it
   at build time. Also bump the workspace `version` in the root
   `pyproject.toml` (unpublished, but shell prompts like starship read it).
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
