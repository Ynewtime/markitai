# Conversion performance

URL extraction runs local-first: short articles and CJK prose usually finish in
the static strategy without starting a browser. Local file conversion loads only
the backend a format needs, with dedicated paths for plain Office documents and
fallback readers for rich content, equations and ambiguous spreadsheet values.
The [Fetch Policy](./fetch-policy) guide has the full strategy cascade.

## Recorded results (14 September 2026)

An audit on macOS with Python 3.12.14 measured a **6.01×** composite speedup
against a frozen pre-optimization source snapshot:

| Equally weighted workload | Speedup |
| --- | ---: |
| Five URL pages over loopback HTTP | 5.83× |
| DOCX, XLSX, PPTX and PDF cold CLI runs | 5.61× |
| 30-file TXT/HTML/DOCX batch | 6.64× |

The composite is the geometric mean of the three groups, over 49 fixed cases
with three fresh processes each, and with model enrichment, remote fetching and
the cache all off. The audit also compared output: 40 converter and API
snapshots, 23 complex table cases and 208 web fixtures came out identical.

These are cold CLI aggregates; don't read them as a figure for one document
or machine. URL timings run over loopback HTTP, excluding internet latency and
any remote service's cache. The frozen baseline already contained the Fake-IP
and CJK fixes, so this does not compare against a released version.

[`2026-09-14.json`](https://github.com/Ynewtime/markitai/blob/main/scripts/benchmarks/results/2026-09-14.json)
holds the full record; the
[benchmark guide](https://github.com/Ynewtime/markitai/blob/main/scripts/benchmarks/README.md)
has the commands to reproduce it.

## Understanding a slow run

`-v` prints the selected strategy and the reason for each fallback. `--no-cache`
turns off markitai's own cache and cannot reach a remote service's, so timing an
already-warm remote reader against a fresh local CLI process compares different
work. Measure local conversion alone with `--preset minimal --no-remote-fetch`;
measure the workflow you actually run with your usual enhancement and OCR
options left on.
