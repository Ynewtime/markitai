# Conversion performance

Two things keep a typical run cheap. URL extraction is local-first: short
articles and CJK prose usually finish in the static strategy without starting a
browser — the [Fetch Policy](./fetch-policy) guide has the full cascade, and the
public-DNS check that gates remote readers behind a Fake-IP proxy. Local file
conversion loads only the backend a format needs, with dedicated paths for plain
Office documents and compatible fallback readers for rich content, equations and
ambiguous spreadsheet values.

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
the cache all off. The same audit compared output rather than only speed: 40
converter and API snapshots, 23 complex table cases and 208 web fixtures came
out identical.

Conditions and scope: these are cold CLI aggregates, not a figure for one
particular document or machine. URL timings run over loopback HTTP, so they
exclude internet latency and any remote service's cache. The frozen baseline
already contained the Fake-IP and CJK fixes, so it is not a comparison against a
released version.

The full record — baseline identity, per-case speedups and the measurement's own
limitations — is
[`scripts/benchmarks/results/2026-09-14.json`](https://github.com/Ynewtime/markitai/blob/main/scripts/benchmarks/results/2026-09-14.json).
Commands for reproducing it are in the
[benchmark guide](https://github.com/Ynewtime/markitai/blob/main/scripts/benchmarks/README.md).

## Understanding a slow run

`-v` prints the selected strategy and the reason for each fallback. `--no-cache`
turns off markitai's own cache and cannot reach a remote service's, so timing an
already-warm remote reader against a fresh local CLI process compares different
work. Measure local conversion alone with `--preset minimal --no-remote-fetch`;
measure the workflow you actually run with your usual enhancement and OCR
options left on.
