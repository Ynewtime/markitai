---
name: markitai-dev
description: "Develop in the markitai monorepo. Use when changing markitai source code, running its test, lint, or typecheck gates, measuring HTML-to-Markdown extraction quality against the defuddle corpus, or syncing that fixture corpus."
metadata:
  internal: true
---

# Working in the markitai codebase

A uv workspace: the published package lives in `packages/markitai` (source in `packages/markitai/src/markitai/`), the VitePress docs site in `website/` (pnpm), install scripts in `scripts/`. Python 3.11–3.13.

```bash
uv sync --all-extras                              # install workspace + extras
uv run pre-commit install                         # ruff on commit
uv run pre-commit install --hook-type pre-push    # pyright + tests on push
```

## Gates — a change is done only when all four pass

```bash
uv run pytest -q                          # default selection: parallel, excludes slow/network
uv run ruff check && uv run ruff format   # lint + format (rules: E,W,F,I,B,C4,UP,ARG,SIM)
uv run pyright packages/markitai/src      # 0 errors required
uv run lint-imports                       # architecture layering contracts, 0 broken required
uv run bandit -c pyproject.toml -r packages/markitai/src -q   # security lint
```

Opt-in markers: `uv run pytest -m "slow or network"`; `parity` marks defuddle-parity tests. CI runs the default selection plus an isolated built-wheel install smoke test on Linux/macOS/Windows × Python 3.11–3.13 — platform-only failures are real failures.

Run the CLI from source with `uv run markitai <input>`.

## Conventions

- Match surrounding code style; ruff and pyright must stay clean.
- Logging is loguru with `{}` formatting: `logger.info("x={}", x)` — never printf-style `%s`.
- Google-style docstrings; English comments.
- Every bug fix ships with a regression test.
- Conventional Commits keep history and changelog writing easy (releases themselves are tag-driven — see the `markitai-release` skill).

## Extraction-quality benchmark

`packages/markitai/benchmarks/` scores the HTML→Markdown pipeline against the defuddle ground-truth corpus in `tests/defuddle_fixtures/` (rapidfuzz block alignment, 0–100 per fixture). It measures continuous quality drift, complementing the pass/fail parity tests. Run it before **and** after any extraction change:

```bash
uv run python packages/markitai/benchmarks/webextract_quality.py
```

It prints per-fixture deltas against the committed `benchmarks/results/baseline.json` and writes `benchmarks/results/latest.json` (gitignored). A quality change that is intentional gets a deliberate `--update-baseline`; an unintentional delta is a regression to fix. The full-corpus run is manual/CI-cron only; `tests/unit/test_webextract_quality_benchmark.py` smoke-tests the scorer math.

## Syncing the defuddle fixture corpus

`scripts/sync_defuddle_fixtures.sh /path/to/defuddle-clone` copies upstream defuddle's `tests/fixtures/*.html` + `tests/expected/*.md` into `tests/defuddle_fixtures/` and records the source commit in `VERSION`. Both the parity tests and the benchmark read this corpus, so resync only as a deliberate act — expect scores to shift, and re-baseline afterwards.
