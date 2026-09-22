# Contributing to Markitai

## Development setup

You need [uv](https://docs.astral.sh/uv/) and Python 3.11–3.13.

```bash
git clone https://github.com/Ynewtime/markitai.git
cd markitai
uv sync --all-extras          # install workspace + all optional extras
uv run pre-commit install     # ruff on commit
uv run pre-commit install --hook-type pre-push   # pyright + tests on push
```

The repo is a uv workspace. The published package lives in `packages/markitai`,
the docs site in `website/` (VitePress + bun), and the install scripts in
`scripts/`. Dated audit and experiment records go in
[`process/`](process/README.md).

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

Test markers are `slow`, `network` and `parity` (see `pyproject.toml`). CI
runs the default selection plus an isolated built-wheel install smoke test on
Linux/macOS/Windows × Python 3.11–3.13.

Architecture rule: modules below `markitai.cli` must never import it. The
other layering contracts sit in `[tool.importlinter]` in the root
`pyproject.toml`. The contracts' `ignore_imports` allowlist only shrinks, so
fix the dependency direction rather than adding an exemption. When code below
the CLI needs the user (a prompt, a notice), go through `markitai.ports`.

## Conversion-quality benchmark

`packages/markitai/benchmarks/` holds a dev-only harness that scores the
HTML→Markdown pipeline against the defuddle ground-truth corpus
(`tests/defuddle_fixtures/`) with a marker-style heuristic scorer
(rapidfuzz fuzzy block alignment, 0–100 per fixture). Where the parity tests
give a pass/fail, this one measures continuous quality drift:

```bash
uv run python packages/markitai/benchmarks/webextract_quality.py   # full corpus
```

It prints per-fixture scores with deltas vs the committed
`benchmarks/results/baseline.json` and writes `benchmarks/results/latest.json`
(gitignored). Run it before and after extraction changes. When a quality change
is intentional, and only then, regenerate the baseline with
`--update-baseline`. The full-corpus run is manual or CI-cron only; a fast
smoke test (`tests/unit/test_webextract_quality_benchmark.py`) covers the
scorer math.

## Document conversion snapshot guardrail

`packages/markitai/benchmarks/docs_snapshot.py` freezes markitai's default
(no LLM, no OCR, no screenshot) conversion output for a small, stable fixture
set, one PDF/DOCX/PPTX/XLSX fixture per pure-Python converter path under
`tests/fixtures/`, and fails when a later change shifts it. The webextract
quality benchmark has to use a continuous fuzzy score, because HTML extraction
has no single "correct" answer. Document conversion for a fixed input is
deterministic, so this one is a plain normalize-then-exact-match snapshot
against `benchmarks/docs_snapshots/expected/*.md`. Normalization strips
environment noise (timestamps, version strings, absolute paths) so the
comparison survives a different machine or a different day; the module
docstring has the exact patterns.

```bash
uv run python packages/markitai/benchmarks/docs_snapshot.py            # compare
uv run python packages/markitai/benchmarks/docs_snapshot.py --update   # regenerate
```

`tests/unit/test_docs_snapshot_guardrail.py` runs the same comparison as a
fast (~seconds), network-free pytest check, so it is part of the default
suite and therefore of CI, with no extra workflow wiring. When a change
intentionally shifts conversion output, regenerate with `--update` and review
the diff like any other code change, the same way you would treat
`webextract_quality.py --update-baseline`. Legacy `.doc`/`.ppt`/`.xls`
fixtures (LibreOffice/MS Office CLI conversion) and the OCR/screenshot paths
are left out on purpose: their output varies with the installed tool version
across CI runners, which makes them a bad fit for an exact-match snapshot.

## Syncing the defuddle fixture corpus

`scripts/sync_defuddle_fixtures.sh /path/to/defuddle` refreshes the defuddle
parity corpus in `tests/defuddle_fixtures/` from a local clone of the upstream
[defuddle](https://github.com/kepano/defuddle) repo. It copies that repo's
`tests/fixtures/*.html` and `tests/expected/*.md` and records the source commit
in `VERSION`. Run it only when you mean to move the corpus to a newer
defuddle; both the parity tests and the quality benchmark read these fixtures,
so a resync can shift scores. `tests/defuddle_fixtures/VERSION` records the
commit and date of the last sync. The script also rewrites the pin in
`src/markitai/webextract/PORT_MANIFEST.md` (the upstream→port module map).
A unit test keeps the two pins equal, and the weekly
`.github/workflows/defuddle-watch.yml` opens an issue when upstream cuts a
release ahead of the pin.

## olmOCR-bench feasibility harness (not CI)

`scripts/olmocr_bench_subset.py` is an opt-in, network-using script that
scores markitai's `--ocr` output against a subset of AI2's
[olmOCR-bench](https://huggingface.co/datasets/allenai/olmOCR-bench) dataset.
It handles the present/absent/order rule types only; the table/math rules need
the official `olmocr[bench]` toolkit. The script's module docstring has the
full feasibility writeup, the licensing, and the manual steps for an
authoritative run. It downloads a small PDF slice directly over HTTPS (no new
dependency) and scores markitai's own conversion:

```bash
uv run python scripts/olmocr_bench_subset.py --split old_scans --limit 5
```

It never runs in CI or in the default test suite;
`tests/unit/test_olmocr_bench_subset.py` covers the scoring math offline.

## LLM enhancement A/B evaluation

`packages/markitai/benchmarks/llm_ab_eval.py` measures what markitai's LLM
enhancement (`llm=True` / `--llm`) actually buys: a blind, position-debiased
A/B judge compares base vs. enhanced conversions of the same documents. The
module docstring carries the methodology (debiasing, aggregation,
checkpointing and the optional Batches API path). It costs real money once you
supply `--judge-model` with live credentials, so run `--dry-run` first for a
cost estimate. Nothing in this repository ever calls a real judge model:

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

Releases are **manual**: pushing a version tag drives them. There is no
auto-generated release PR, so you cut a release only when you mean to.

1. Bump `__version__` in `packages/markitai/src/markitai/__init__.py`. That is
   the **single source of truth** for the published package; hatch reads it
   at build time. Also bump the workspace `version` in the root
   `pyproject.toml` (unpublished, but shell prompts like starship read it).
2. Add a `## [X.Y.Z] - YYYY-MM-DD` section to `CHANGELOG.md` (this becomes the
   GitHub Release notes verbatim) and mirror it in `CHANGELOG.zh.md`. The docs
   build copies both files in, so the website publishes whatever you write.
3. Commit and push to `main` (e.g. `chore(release): v X.Y.Z`).
4. Tag and push:

   ```bash
   git tag vX.Y.Z && git push origin vX.Y.Z
   ```

Pushing the `vX.Y.Z` tag runs `.github/workflows/publish.yml`. It runs the
test suite, builds the package, verifies the tag matches `__version__` (and
fails loudly if you forgot to bump), publishes to PyPI via trusted publishing,
and creates the GitHub Release with the matching `CHANGELOG.md` section as
notes.

Once PyPI shows the new version, publish the MCP Registry entry: bump `version`
in both places in the root `server.json`, then `mcp-publisher validate`,
`login github` and `publish`. The registry reads that exact version's README
from PyPI and looks for the `mcp-name:` marker in it.

To re-publish an existing tag (e.g. after a transient failure), run the
**Release** workflow manually from the Actions tab with the tag as input.
Commit type does not affect what gets released, but keep using
[Conventional Commits](https://www.conventionalcommits.org/): they keep the
history readable and make the changelog easier to write.

`skills/markitai-release/SKILL.md` carries the full checklist, including the
preflight gates and the end-to-end release check.
