# Conversion performance

Complete short articles and CJK prose can finish through local static extraction
without an unnecessary browser launch. Public URLs behind Fake-IP proxies can
use an explicitly selected remote reader after the public DNS checks described
in [Fetch Policy](./fetch-policy).

Local file conversion loads the backend it needs. Plain Office documents take
dedicated paths; rich content, equations and ambiguous spreadsheet values retain
compatible fallback readers. Images, metadata, error handling and configured
adapters remain part of the output contract. PDF model settings are unchanged.

## Recorded results

A September 2026 audit on macOS with Python 3.12 measured approximately **6.01×**
composite speedup against a frozen pre-optimization source snapshot:

| Equally weighted workload | Speedup |
| --- | ---: |
| Five URL pages over loopback HTTP | 5.83× |
| DOCX, XLSX, PPTX and PDF cold CLI runs | 5.61× |
| 30-file TXT/HTML/DOCX batch | 6.64× |

The metric is the geometric mean of the three groups, with three fresh-process
runs per case. These are local conversion results with model enrichment and
remote fetching disabled. Internet delays, remote cache hits, LLM responses,
OCR and arbitrary document sizes can produce different results.

The audit compares full output and assets, not speed alone: 40 converter/API
snapshots, 23 additional complex table cases and 208 web fixtures were checked.
The broader test suite also covers rich DOCX fallbacks and runtime adapters.

For reproducible commands and quality gates, see the repository's
[benchmark guide](https://github.com/Ynewtime/markitai/blob/main/scripts/benchmarks/README.md).

## Understanding a slow run

Use `-v` to inspect the selected strategy and fallback reasons. `--no-cache`
turns off Markitai's cache; it cannot disable a remote service's cache. Comparing
an already-running remote reader to a new local CLI process therefore measures
different work. Use `--preset minimal --no-remote-fetch` when measuring local
conversion alone. Keep your normal enhancement and OCR options when measuring
the complete workflow you actually use.
