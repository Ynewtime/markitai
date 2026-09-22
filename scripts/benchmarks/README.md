# Performance and output audits

Run these from the repository root with the project Python environment. They
use synthetic or checked-in public fixtures, and they keep timing separate from
output validation. Write results somewhere outside the tracked source tree.

## Fixed comparison

`BEFORE_SOURCE` must contain a `markitai/` source package from the version you
are comparing against. Leave that directory alone once you start: switching it
mid-run changes the baseline. For an unmodified commit, extract it with
`git archive`; a dirty source snapshot must also keep its patch and revision.
Use fresh result directories.

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
equally weighted groups (five loopback HTTP pages, four Office/PDF formats, and
a batch of 30 TXT/HTML/DOCX files) and reports the score against the 10×
research target it was set up to test. Loopback transport leaves out internet
latency and remote service caching. If you change the weights, the target or
the input set, you have a different measurement, and it needs its own record.

`audit_quality.py` compares 40 converter and API snapshots, covering Markdown,
metadata, image bytes and dimensions, frontmatter and output assets. It
normalizes only the clock, the output directory and explicit engine-provenance
transitions. Missing quality evidence, or changed outputs, fails the scoring
gate.

Recorded runs live in [`results/`](results), one dated file each, with its
baseline identity and limitations. The latest,
[`2026-09-14.json`](results/2026-09-14.json), is what the
[performance guide](https://markitai.dev/guide/performance) quotes.

## Defuddle and scaling

Build a separate Defuddle checkout with `bun install --frozen-lockfile` and
`bun run build`, following that checkout's package scripts, then run:

```bash
.venv/bin/python scripts/benchmarks/compare_defuddle.py --defuddle-root /absolute/path/to/defuddle --output /tmp/markitai-defuddle-comparison --iterations 3
```

This compares the shared web corpus and records text, metadata, extraction
acceptance and browser-fallback decisions. Look at the non-timing fields, not
just the speed. `compare_cli.py` covers file/URL/stdin behavior;
`audit_title_scaling.py` and `audit_spreadsheet_scaling.py` exercise large
documents; `excel_fixtures.py` supplies the shared workbook edge cases that
`tests/unit/test_xlsx_plain_tables.py` also loads. Run any of them with
`--help` for arguments. Backends we prototyped here and did not adopt are
listed in [`process/rejected-backends.md`](../../process/rejected-backends.md).

## Release end-to-end validation

`scripts/e2e_release_check.sh` builds and installs a wheel into a temporary
home, then exercises the public CLI commands, local HTTP, formats,
batch/resume, the Python API, serve/MCP, browsers and optional services. It also
calls real models and remote providers using `ENV_FILE`; those checks send
synthetic fixtures and can cost you provider charges. The HTML report separates
passed, failed and skipped checks, and missing credentials do not count as a
successful remote check. A Fake-IP DNS resolver does not skip the remote
extraction step: the installed product has to do its own public DNS
verification.

```bash
WORKDIR=/tmp/markitai-release-check bash scripts/e2e_release_check.sh
```

`scripts/check_wheel.py` and `scripts/check_licenses.py` are separate offline
checks, of the built artifact and of dependency licences. They do not replace
the end-to-end run.
