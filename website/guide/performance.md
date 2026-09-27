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

## PDF extraction, LLM batches and the web workspace (25 September 2026)

PDF text goes through a layout model, one of the heaviest steps in a
conversion. The CLI, `markitai serve` and the MCP server extract the pages of
a PDF with 12 or more pages, and of PDFs converted at the same time, in worker
processes that each run the model on one thread. The output is identical to
extracting the document in one piece. A page then takes about a third of the
CPU time, and each worker holds about 450 MB. `MARKITAI_PDF_WORKERS` sets the
number of workers (by default the CPU count minus two, at most 12, and no
more than a quarter of the available memory); `0` or `1` keeps extraction
in-process. A Python script opts in with
[`enable_worker_processes()`](./python-api#parallel-pdf-extraction).

In an LLM batch a converted file hands its conversion slot to the next file
while it waits on the model, so conversion and enhancement overlap.

Cold CLI runs against 1.1.0 on an 18-core Mac, median of three interleaved
runs, cache off:

| Workload | 1.1.0 | 1.2.0 | Speedup |
| --- | ---: | ---: | ---: |
| One 300-page text PDF | 16.6 s | 4.0 s | 4.1× |
| 40 PDFs (5 pages each) | 12.1 s | 2.9 s | 4.1× |
| 240 mixed files (PDF, DOCX, PPTX, XLSX, HTML, CSV) | 12.9 s | 3.9 s | 3.3× |
| 30 documents with `--llm --alt`, 0.5 s model latency | 7.0 s | 4.7 s | 1.5× |

`markitai serve` runs the same pipeline. A job now takes up to 1000 items,
so 240 mixed files go in one job (1.1.0 needed five): 12.8 s → 4.0 s. Thirty
documents with the `standard` preset: 6.2 s → 3.7 s. During a 300-page PDF
the API answered in 295 ms at the 95th percentile before and 3 ms now.

Single small files, OCR and URL batches take the same time as before. The
speedup grows with the number of cores and pages; a small PDF converted on
its own stays in-process, where starting workers would cost more than it
saves.

## Understanding a slow run

`-v` prints the selected strategy and the reason for each fallback. `--no-cache`
turns off markitai's own cache and cannot reach a remote service's, so timing an
already-warm remote reader against a fresh local CLI process compares different
work. Measure local conversion alone with `--preset minimal --no-remote-fetch`;
measure the workflow you actually run with your usual enhancement and OCR
options left on.
