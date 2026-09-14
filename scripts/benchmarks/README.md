# Performance and output audits

Run from the repository root with the project Python environment. These tools
use synthetic or checked-in public fixtures and keep measurements separate from
complete-output validation. Write results outside the tracked source tree.

## Fixed comparison

`BEFORE_SOURCE` must contain a `markitai/` source package from the version being
compared. Preserve that directory: switching it mid-run changes the baseline.
For an unmodified commit it can be extracted with `git archive`; a dirty source
snapshot must also retain its patch and revision. Use fresh result directories.

```bash
export LITELLM_LOCAL_MODEL_COST_MAP=True
BEFORE_SOURCE=/absolute/path/to/frozen/source
RESULTS=/tmp/markitai-performance
.venv/bin/python scripts/benchmarks/audit_paired.py --before-source "$BEFORE_SOURCE" --output "$RESULTS/files"
.venv/bin/python scripts/benchmarks/audit_urls.py --before-source "$BEFORE_SOURCE" --output "$RESULTS/urls" --repeats 3
.venv/bin/python scripts/benchmarks/audit_quality.py --before-source "$BEFORE_SOURCE" --inputs "$RESULTS/files/before/inputs" --output "$RESULTS/quality"
.venv/bin/python scripts/benchmarks/score_performance.py --before "$RESULTS/files/before" --after "$RESULTS/files/after" --urls "$RESULTS/urls" --quality "$RESULTS/quality/quality.json" --output "$RESULTS/score.json"
```

Run timing commands sequentially, without builds or tests running alongside
them. `audit_paired.py` alternates source order for 49 fixed cases and runs each
in three fresh processes. Its worker records cold and warm conversion details,
CPU time, peak RSS, loaded dependencies and content hashes. The score uses cold
CLI times: five loopback HTTP pages, four Office/PDF formats, and a batch of 30
TXT/HTML/DOCX files. Each group has one third of the geometric-mean weight.
Loopback transport excludes internet latency and remote service caching.

`audit_quality.py` compares 40 converter/API snapshots, including Markdown,
metadata, image bytes and dimensions, frontmatter and output assets. Only the
clock/output directory and explicit engine-provenance transitions are
normalized. Missing quality evidence or changed outputs fails the scoring gate.
The tool reports whether the original 10× research target was reached; the
recorded September 2026 result is approximately 6×; its [recorded score](results/2026-09-14.json) includes baseline identity and limitations. Do not change the threshold,
weights or input set to present a different result as the same measurement.

## Defuddle and scaling

Build a separate Defuddle checkout with `bun install --frozen-lockfile` and
`bun run build`, following that checkout's package scripts, then run:

```bash
.venv/bin/python scripts/benchmarks/compare_defuddle.py --defuddle-root /absolute/path/to/defuddle --output /tmp/markitai-defuddle-comparison --iterations 3
```

This compares the shared web corpus and records text, metadata, extraction
acceptance and browser-fallback decisions. Inspect non-timing fields as well as
speed. `compare_cli.py` covers file/URL/stdin behavior; `audit_title_scaling.py`
and `audit_spreadsheet_scaling.py` exercise large documents. Run `--help` for
arguments. `excel_fixtures.py` supplies 21 shared workbook edge cases to the
regression tests. Rejected Calamine and PDF model experiments are not production
backends and are intentionally absent from the maintained tool set.

## Release end-to-end validation

`scripts/e2e_release_check.sh` builds and installs a wheel into a temporary home,
then exercises public CLI commands, local HTTP, formats, batch/resume, API,
serve/MCP, browsers and optional services. It also calls real models and remote
providers using `ENV_FILE`; those checks send synthetic fixtures and can incur
provider charges. The HTML report distinguishes passed, failed and skipped
checks. Missing credentials do not count as a successful remote check. Fake-IP
DNS no longer skips the remote extraction step: the installed product must
perform its public DNS verification.

```bash
WORKDIR=/tmp/markitai-release-check bash scripts/e2e_release_check.sh
```

`scripts/check_wheel.py` and `scripts/check_licenses.py` provide separate offline
artifact and dependency-license checks; they do not replace end-to-end testing.
