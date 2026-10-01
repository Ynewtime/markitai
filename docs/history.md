# CLI history archives

`--record-history` copies the current invocation's final Markdown documents and
supported asset trees into an independent archive. Archives live under
`MARKITAI_HOME/serve/jobs/<12-hex-id>/`; without that override the home is
`~/.markitai`. This location is independent of `cache.global_dir`.

```sh
MARKITAI_HOME="$PWD/.local/test-home" markitai document.docx -o out/ --record-history
MARKITAI_HOME="$PWD/.local/test-home" markitai inputs/ -o out/ --resume --record-history
```

The flag pair `--record-history` / `--no-record-history` takes precedence over
the environment and configuration, with the last flag winning. A nonempty,
trimmed `MARKITAI_RECORD_HISTORY` overrides `history.record`: `1`, `true`, `yes`
and `on` enable it, ignoring case; other nonempty values disable it. Empty or
whitespace-only values leave configuration unchanged. The default is disabled.

Stdout Markdown, dry runs, empty work and invalid single-file inputs create no
archive. JSON disk-output mode can record history without adding JSON fields or
printing a success message. Ordinary mode prints the saved path to stderr unless
quiet. The native `serve` command reads completed archives in place and exposes
results, assets and downloads; see [service history](serve.md). Native bindings
do not invoke the CLI archive writer.

## Which outcomes are saved

One invocation produces at most one job. Directory items appear in input order,
files before URLs; URL-list items retain completion order. Resume records only
items actually processed this time, so resuming an entirely completed run adds
no job. A controlled batch interruption or fatal recovery-store error suppresses
history. An interruption observed during staging discards the stage before
publication; a completed publication is the boundary after which the archive
already exists.

Conversion failures remain `error` items. Skips remain `done` with `skipped:true`
and their reason. An existing base output may be copied for an `exists` skip;
an image-only skip has no output. Missing or uncopyable individual outputs leave
their item intact with null output fields. A failure to copy assets, serialize
metadata or publish emits a warning and leaves conversion exit codes unchanged.
This can leave a successful conversion without a history archive.

The archive contains `meta.json` and `out/`. Metadata uses the reference job
schema: `job_id`, `created_at`, `finished_at`, `status`, `options`,
`dir_size_bytes`, `items`. Job status is `done` even when all conversions failed.
Options are exactly `preset`, `llm`, `ocr`, `origin:"cli"`. Size counts copied
outputs and assets after rewriting, excluding metadata.

Each item stores `item_id`, `name`, `kind`, `status`, `error`, `output`,
`output_name`, `duration_ms`, `finished_at`, `cost_usd`, `llm_enhanced`,
`operation`, `skipped`, `skip_reason`, `retryable`, `warnings`. Names are source
basenames for single files, relative paths for directory files and bare URLs for
URLs. Durations use nonnegative milliseconds with ties-to-even rounding. Local
RFC3339 timestamps have millisecond precision, with one finish timestamp shared
by the job and its items. Warnings are deduplicated in encounter order. Only URL
items are retryable because original input files are not archived.

## Numbers package inputs

A modern `.numbers` directory is eligible as one local document, including when
its conversion fails. The reader owns the package content limits; directory
metadata length is not treated as its byte size. Single-package archives use the
package basename, while a package found in a directory batch uses its relative
path. History copies the generated Markdown and referenced assets, never the
package's internal source files. Deleting the source package does not remove the
archived output. CLI file history remains nonretryable without its original input.

## Attempt diagnostics

New CLI archives add optional item `diagnostics.last_attempt` only when core
reports observed model work. It contains operation `convert`, status `done` or
`error`, the existing string/null error and full request/token/cost/per-model
usage. A response reporting one request and zero tokens remains observable;
missing or unreadable usage produces no new field. Existing item `cost_usd`,
output authority, skip rules and archive eligibility retain their prior meaning.

A resumed CLI archive contains only this invocation's processed items and their
latest observations. Earlier archives are independent snapshots and are not
rewritten or summed into the retry. REST reads the same additive shape; older
items without it remain unchanged. This observation is not a lifetime ledger,
provider pricing guarantee or proof that hard-killed work was unbilled.

## Independent files and references

Final documents are flattened into `out/`. Name reservations use full Unicode
default case folding (Unicode 16 through `caseless` 0.2.2), including
multi-character folds such as `ß` → `ss`.
Conflicts add ` (2)`, ` (3)` before the suffix, treating `.llm.md` as one suffix.
`output_name` projects that enhanced suffix back to `.md`; `llm_enhanced` requires
an actual copied `.llm.md` output and an item that was not skipped.

Asset discovery starts at the output's parent and stays within the planned
output directory. File items may ascend by their relative source depth; URLs
remain at their output parent. The nearest `.markitai/` or `assets/` directory
stops the search, including a `.markitai/` containing only recovery metadata.
Only `.markitai/assets`, `.markitai/screenshots` and `assets` are copied. Original
inputs, reports, state files and ownership receipts are excluded.

Assets with the same destination and bytes are shared. Different contents get
`-2`, `-3` before the final extension. Unicode case aliases are reserved across
directories and files. Hashes narrow duplicate candidates; exact byte comparison
proves reuse. Each document uses only its own asset root's relocation map.
Identical contents can also reuse a previously renamed conflicting variant;
this avoids an extra copy that the reference writer can create.

Reference rewriting shares the native Markdown parser and looks up each original
destination once. It supports links, definitions, wikilinks, HTML `a[href]`,
`img[src/srcset]`, `source[src/srcset]`, `video[src/poster]`, `audio[src]` and
`track[src]`, [CSS resources in HTML styles](css-resources.md), plus exact frontmatter path scalars, including percent-encoded
spellings. All supported attributes in a tag are processed together; duplicate
attribute names use their first occurrence, including a valueless attribute.
Image candidate URLs are separated from their descriptors after decoding HTML
entities, so commas within data URLs or filenames do not become list separators.
Literal code
and unrelated prose stay unchanged. Parent-relative references are relocated when
flattening. URI query/fragment suffixes retain their spelling, while literal
`%`, `#` and `?` characters in replacement filenames are encoded as file data.
These behaviors deliberately improve on the reference writer's
global text replacement and retained `../` paths; byte parity with those defects
is not a target.

Additional element/attribute families, including custom lazy-load attributes,
are not parsed by the relocation helper. Arbitrary Markdown or model output
using these forms can retain a stale reference when an asset collision changes
a name. External stylesheet contents and dynamically computed CSS URLs are also
outside destination rewriting; these remain tracked fidelity gaps.

The newer media syntax is covered by source tests and
[round-eleven native release acceptance](validation/html-media-round11.md);
the frozen round-ten paired reference evidence predates this extension.
CSS resources have [round-twelve release and browser acceptance](validation/css-round12.md)
on authored cases; the reference history writer uses global text replacement.
The candidate boundaries follow the [HTML srcset algorithm](https://html.spec.whatwg.org/multipage/images.html#parse-a-srcset-attribute).
New filename data is encoded before HTML escaping so the
[URL parser](https://url.spec.whatwg.org/#concept-basic-url-parser) cannot discard
filename whitespace or treat a backslash as a separator. This is destination
rewriting, not browser resource selection or full malformed-HTML recovery.

## Publication and resource boundaries

New directories use mode `0700` and files `0600` on Unix. History metadata paths
reject symbolic links regardless of the document-output policy; selected asset
leaves and traversed entries must be regular files or directories. Permitted
output ancestor aliases still resolve within the canonical output boundary.
The archive is not a filesystem snapshot: concurrent source changes are detected
during copy and can prevent the affected output or whole job from being saved.

Limits are 256 MiB per copied file, 1 GiB total copied content, 100,000 copied
files/inspected entries, asset depth 64, 100,000 items, 64 MiB per Markdown document
requiring rewriting and 16 MiB of serialized metadata. Copying streams through
64 KiB buffers. Name counters and content indexes avoid repeated whole-directory
scans. Rewriting shares a lookup across documents at the same root depth and
releases that lookup before processing the next depth.

Work happens in a private `.tmp-*` sibling; metadata is written last. A stable
`.publish.lock` serializes only final naming, metadata and directory publication,
with a five-second wait limit. Every file and directory of the stage is then
synchronized and ordered before the stage is renamed to its unique ID, and the
parent's durable synchronization follows on Unix: a job that survives a crash is
complete, and a published job is durable when publication returns. On a verified
local macOS APFS/HFS volume this is per-object `fsync`, one `F_BARRIERFSYNC` and
one `F_FULLFSYNC` (a full cache flush per file and directory before); elsewhere
each object keeps `File::sync_all`. Ordinary
errors remove the stage. A hard process kill may leave an unpublished `.tmp-*`
directory, which readers must ignore. Published jobs and the stable lock are not
automatically pruned. This is cooperative local-process protection, not a defense
against a privileged actor replacing ancestors during system calls.

Current source tests exercise the native recorder and isolated real CLI processes.
The [round-ten release audit](validation/history-round10.md) passes five authored
archive contract pairs, four strictly after declared identity/time normalization.
The failed-notebook pair retains its existing parser diagnostic difference.
Native service history consumption is implemented and covered by actual HTTP
tests. Broader paired coverage and cross-platform execution still need their own
acceptance evidence. The older filename adaptation below has its own focused
source tests; executed results belong in the current validation record.

## Older enhanced histories

The service reads completed reference histories from the existing job directory;
there is no import command or required directory move. Some old recorders saved
the actual enhanced filename, such as `page.html.llm.md`, in both `output` and
`output_name`. The latter is supposed to identify the base Markdown filename.
On loading an unindexed legacy item, the service normalizes that field in memory
to `page.html.md` while keeping the actual `output` and all files unchanged.
When `llm_enhanced` is missing, the enhanced output suffix supplies the legacy
default. An explicitly saved false value is retained.

Native `native_bases` entries take precedence over filename inference. In
particular, a literal source name `notes.llm` can legitimately have
`notes.llm.md` as its base; the loader does not strip that identity or mark it
enhanced merely because of the suffix. Result preview, retry and item deletion
use the same output-family resolver. Inconsistent saved identities return 409;
mutations also reject Markdown families claimed by another item, including
case-folded collisions. These checks do not prevent safe direct file downloads
or whole-job archive downloads for inspecting ambiguous metadata.

For valid legacy jobs without a pending native recovery transaction, startup,
history listing, preview and download do not rewrite `meta.json`, create native
indexes or rename output files. Stopping the native service therefore leaves
the same original history available to the reference service. This is read-time
compatibility, not a destructive schema migration or permission to run both
implementations as simultaneous writers.

Explicit retry and deletion retain their ordinary persistence semantics. URL
items can refetch their original URL; web items with a retained upload can
reconvert it. CLI file archives without uploads remain nonretryable. A successful
plain retry writes the correct base filename, removes its stale enhanced sibling
and persists the resolved native identity. Deleting an item removes its correct
Markdown pair while preserving assets still claimed by siblings.

A failed retry of a previous successful item preserves its output fields
and existing result bytes through the service's staged transaction and recovery
path. The job's latest options and completion timestamp can still be updated by
normal finalization; explicit retry is not a promise that the whole metadata
file stays byte-identical. No automatic undo of a successful retry or deletion
is added. Focused tests cover authored legacy reads, restart, one loopback URL
retry, failure preservation, native-name ambiguity and shared-asset deletion.
