# CLI run reports

The CLI writes version `"1.0"` reports for single files, single URLs, directories
and URL lists. Directory reports also include URLs discovered in `.urls` files.
Reports summarize a finished run; separate [recovery state](state-storage.md)
supports batch resume on Unix and Windows. Optional [history archives](history.md) retain independent
documents and assets. Public core and
Node/Python/Go conversion calls do not publish CLI reports.

## Selection and lifecycle

| `output.report` | Single file or URL | Directory or URL list |
|---|---|---|
| Omitted or `null` | Off | On |
| `true` | On | On |
| `false` | Off | Off |

```sh
markitai note.txt -o output/ --config-json '{"output":{"report":true}}'
markitai https://example.com/article -o output/ --config-json '{"output":{"report":true}}'
markitai documents/ -o output/ --json
markitai pages.urls -o output/ --config-json '{"output":{"report":false}}'
```

A single-item report requires a completed conversion with an output directory;
failed or skipped single items produce no new report. Batches report processed
items even when some or all fail. Stdout-only conversion, dry runs and empty
directory discovery without recoverable state publishes no report. Empty or invalid URL lists fail without
publishing one. Report selection is independent of the batch recovery journal.

Publication occurs before the CLI's single stdout JSON envelope. A publication
error preserves completed output files and their JSON items, sets an envelope
error and exits nonzero: 10 if batch conversion already failed, otherwise 1.
Diagnostics go to stderr. Verbose mode displays the published or
existing report path unless quiet mode is active.

## Paths and identity

Reports live under the selected output directory:

```text
<output>/.markitai/reports/markitai.<six-hex-hash>.report.json
<output>/.markitai/reports/markitai.<six-hex-hash>.v2.report.json
```

For an explicit `-o chosen.md`, the parent directory contains the report. Item
`output` points at the actual finalized document, including LLM-enhanced output;
newly observed paths retain relative spelling where supplied. Recovered outputs
and saved URL provenance use the checkpoint's anchored paths. Directory options contain resolved absolute
`input_dir` and `output_dir`.

The filename hash matches the reference's first six MD5 hex digits over resolved
paths and selected options, serialized with Python's sorted, spaced, ASCII JSON
conventions. It is a compatibility filename, not a content fingerprint or an
ownership guarantee.

| Report mode | Input/output paths used for hashing | Selected options |
|---|---|---|
| Single file | File / output directory | `llm,ocr,screenshot,alt,desc` |
| Directory | Input directory / output directory | Five flags above, `scan_max_depth`, nonempty trimmed `glob_patterns` |
| Single URL | Output directory / output directory | `llm` |
| URL list | Output directory / output directory | `llm,alt,desc` |

The URL and URL-list path are deliberately absent from their report identities.
Model, profile, pure mode, cache and conflict policy also do not enter these
hashes. Repeated invocations can therefore share a base report name.

## Four serialized projections

Reports use ordered fields independently of stdout and bindings JSON. All have
`version,generated_at,log_file,summary,llm_usage`; unused logging is null.
Mode-specific fields retain their reference ordering, including nested maps.

| Mode | Shape and counting rules |
|---|---|
| Single file | One completed entry in `documents`, keyed by basename; no `options` or `url_sources`. Item `llm_usage` contains only `cost_usd,input_tokens,output_tokens`. |
| Single URL | `options` contains `llm,cache,alt,desc,fetch_strategy`; zero documents and one URL grouped under `url_sources.cli`. The entry has duration, asset counts, per-model usage and cache details. |
| URL list | No `options` or `documents`; entries are grouped under `url_sources["unknown.urls"]`, preserving a reference quirk. Completed entries contain `status,output,error,fetch_strategy,images,screenshots`; failures contain `status,error`; skipped entries contain `status:"skipped",error:"Output exists"`. No per-item duration, usage or cache fields are added. |
| Directory | Adds `started_at,updated_at,options,documents,url_sources`. Entries retain status, output/error, timing, asset counts and usage. Successful skips count as completed; failed items also count as pending. Summary includes URL-source count, URL cache hits and summed `processing_time`. |

Document keys and source groups are sorted; URLs within a group retain task
input order. Directory document keys are relative to the input directory, and
URL groups retain the source-list path. A named URL key is `URL + " " + raw_name`:
`name` and `name.md` remain distinct. Exact URL/name pairs are deduplicated before
processing, including across lists within a directory.

`fetch_strategy` records the strategy that produced the document, including in
pure mode. The core retains this internally without adding a field to public
conversion JSON or pure frontmatter.

Cache fields intentionally differ by interface:

| Location | Meaning |
|---|---|
| Single-URL report `cache_hit` | Fetch OR LLM hit; `cache_details` contains both flags |
| Directory entry `cache_hit` and summary `url_cache_hits` | LLM hits only |
| URL-list report entries | No cache fields |
| CLI stdout `cache_hit` / `llm_cache_hit` | LLM hit; `fetch_cache_hit` remains separate |

Top-level `llm_usage` contains `models,requests,input_tokens,output_tokens,cost_usd`.
Models are sorted; request/token totals come from their records, while total cost
comes from item usage. Both directory and URL-list aggregate model records
initialize and sum `cached_input_tokens`. URL-list aggregation includes completed
items only. Reports use original usage values, not the rounded stdout projection;
they cannot recover usage that the conversion runtime did not return.

Timestamps use local-offset ISO strings. Durations are strings: one decimal plus
`s` below a minute, then `MM:SS` or `HH:MM:SS` using truncated integer seconds.
Batch duration measures wall time; directory `processing_time` sums item times
and can exceed wall time when work runs concurrently.

## Publication and intentional differences

`output.on_conflict` also governs reports. `rename` creates another report;
`overwrite` replaces the base report; `skip` preserves existing bytes. Conversion
results remain separate from this report conflict decision.

The native writer stages and syncs a same-directory temporary file before atomic
publication; on a verified local macOS APFS/HFS volume that is `fsync` plus an
ordering barrier rather than a full cache flush. The report's name is not
synchronized, so, as before, a report is not guaranteed durable when the CLI exits,
but one that survives a crash has complete bytes. Rename and skip use no-clobber publication, including when another
process claims a name concurrently. Output symlink policy applies to report
paths. This is not a transaction with document output or provider requests, and
does not supply recovery after interruption.

Two reference defects are deliberately corrected: report `skip` no longer
overwrites an existing report, and explicit enhanced output records the actual
final file instead of a stale intermediate path. Naming is also bounded: the
reference shared helper increments `.vN` indefinitely; the native helper tries
through `.v9999`, then UUID names. Directory naming follows the reference's
timestamp fallback after `.v9999`, adding a UUID on timestamp collision. Native
fallback attempts are capped at 64, then publication fails explicitly.

Reports omit document bodies and serialize only whitelisted options, never a
complete provider configuration. They can still contain plaintext URLs, custom
names, local paths, model names and errors. Hash names provide no confidentiality.
Tests use temporary input/output and `MARKITAI_HOME`; reports live in the output
tree independently of the global state directory.

## Latest attempt diagnostics

A report with observed model work adds an optional `terminal_diagnostics` object.
Its `documents` and `urls` maps use the existing report identity: relative file
keys and raw URL plus any supplied name. Each value is the shared
`diagnostics.last_attempt` shape: `operation`, `status`, string/null `error`, and
`usage` with request/token/cost totals and per-model records. File keys are sorted;
URL entries retain report encounter order. Empty maps remain present inside the
section; the whole section is omitted when there are no observations.

This section is independent of the existing `llm_usage` aggregate and item
schemas. Known failed calls appear here without changing success totals, single
item report eligibility, URL-list sparse entries or output ownership. A failed
single conversion still produces no saved report; stdout and optional history
can carry its observation. Report publication failures do not reclassify finished
conversions or charge their recorded work again.

Native resumed reports retain saved observations for unprocessed entries and
replace them with an observed new attempt, including clearing an old value when
that attempt has no known usage. They never add old failed work to the new
attempt. A known request with zero tokens is retained; absent observations remain
unknown. This is neither lifetime spend nor a crash-safe billing ledger. Existing
legacy reports and minimal states have no backfilled measurements.
