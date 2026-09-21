# Performance and output audits

Run these from the repository root with the project Python environment. They use
synthetic or checked-in public fixtures, and they keep timing separate from
output validation. Write results outside the tracked source tree.

## Fixed comparison

`BEFORE_SOURCE` must contain a `markitai/` source package from the version being
compared against. Preserve that directory: switching it mid-run changes the
baseline. For an unmodified commit, extract it with `git archive`; a dirty
source snapshot must also retain its patch and revision. Use fresh result
directories.

```bash
export LITELLM_LOCAL_MODEL_COST_MAP=True
BEFORE_SOURCE=/absolute/path/to/frozen/source
RESULTS=/tmp/markitai-performance
.venv/bin/python scripts/benchmarks/audit_paired.py --before-source "$BEFORE_SOURCE" --output "$RESULTS/files"
.venv/bin/python scripts/benchmarks/audit_urls.py --before-source "$BEFORE_SOURCE" --output "$RESULTS/urls" --repeats 3
.venv/bin/python scripts/benchmarks/audit_quality.py --before-source "$BEFORE_SOURCE" --inputs "$RESULTS/files/before/inputs" --output "$RESULTS/quality"
.venv/bin/python scripts/benchmarks/score_performance.py --before "$RESULTS/files/before" --after "$RESULTS/files/after" --urls "$RESULTS/urls" --quality "$RESULTS/quality/quality.json" --output "$RESULTS/score.json"
```

Run the timing commands sequentially, with no builds or tests alongside them.
`audit_paired.py` alternates source order for 49 fixed cases and runs each in
three fresh processes; its worker records cold and warm conversion details, CPU
time, peak RSS, loaded dependencies and content hashes.

`score_performance.py` builds the composite from cold CLI times in three
equally weighted groups — five loopback HTTP pages, four Office/PDF formats, and
a batch of 30 TXT/HTML/DOCX files — and reports the score against the 10×
research target it was set up to test. Loopback transport excludes internet
latency and remote service caching. The weights, the target and the input set
are what make one run comparable to another; changing any of them produces a
different measurement, which needs its own record rather than a rewritten one.

`audit_quality.py` compares 40 converter and API snapshots, covering Markdown,
metadata, image bytes and dimensions, frontmatter and output assets. Only the
clock, the output directory and explicit engine-provenance transitions are
normalized. Missing quality evidence, or changed outputs, fails the scoring
gate.

Recorded runs live in [`results/`](results), one dated file each, with the
baseline identity and that run's own limitations:
[`2026-09-14.json`](results/2026-09-14.json) is the latest. Its headline figure
is published in the [performance guide](https://markitai.dev/guide/performance).

## Defuddle and scaling

Build a separate Defuddle checkout with `bun install --frozen-lockfile` and
`bun run build`, following that checkout's package scripts, then run:

```bash
.venv/bin/python scripts/benchmarks/compare_defuddle.py --defuddle-root /absolute/path/to/defuddle --output /tmp/markitai-defuddle-comparison --iterations 3
```

This compares the shared web corpus and records text, metadata, extraction
acceptance and browser-fallback decisions. Inspect the non-timing fields as well
as speed. `compare_cli.py` covers file/URL/stdin behavior; `audit_title_scaling.py`
and `audit_spreadsheet_scaling.py` exercise large documents; `excel_fixtures.py`
supplies the shared workbook edge cases `tests/unit/test_xlsx_plain_tables.py`
also loads. Run any of them with `--help` for arguments.

Backends that were prototyped here and not adopted are recorded in
[`process/rejected-backends.md`](../../process/rejected-backends.md), so they
do not get re-proposed.

## Release end-to-end validation

`scripts/e2e_release_check.sh` builds and installs a wheel into a temporary
home, then exercises the public CLI commands, local HTTP, formats,
batch/resume, the Python API, serve/MCP, browsers and optional services. It also
calls real models and remote providers using `ENV_FILE`; those checks send
synthetic fixtures and can incur provider charges. The HTML report distinguishes
passed, failed and skipped checks, and missing credentials do not count as a
successful remote check. A Fake-IP DNS resolver does not skip the remote
extraction step: the installed product has to perform its own public DNS
verification.

```bash
WORKDIR=/tmp/markitai-release-check bash scripts/e2e_release_check.sh
```

`scripts/check_wheel.py` and `scripts/check_licenses.py` are separate offline
checks of the built artifact and of dependency licences; they do not replace the
end-to-end run.
