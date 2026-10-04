# Run reports, resume state and history

Historical design and validation record. Implementation-stage statements below
describe their original checkpoints. Current behavior and crash/platform
boundaries are documented in [recovery](../state-storage.md),
[history](../history.md) and [output ownership](../output-ownership.md).

This design follows a read-only audit of reference revision
`ba374322f884b0e720b45466cc1196f4574a3da5`, against native baseline `658cf00`.
Round-six artifacts use `e0cf110fdf89724555b8afb4a23a53dfa5e89055` and predate the
report implementation. These are historical design baselines, not the revision
of the newly validated implementation. Reference anchors below are relative to
`packages/markitai/src/markitai/` in the reference repository. The initial audit
did not execute its code or inspect real user state; subsequent report comparison
uses isolated CLI processes and loopback fixtures.

## Scope and architecture

Implement the existing CLI's complete file, URL, directory and URL-list report
behavior; directory/URL-list resume; and optional history export. Keep these as
three projections of a typed run model, with separate schemas and lifecycles.
The stdout JSON envelope is neither a report nor a recovery checkpoint.

The implemented report model retains raw source identity, relative file key, raw
URL custom name, source list, actual output, status, start/end time, duration,
image/screenshot counts, usage, independent fetch/LLM cache hits, warnings and
skip reason. Workers return typed terminal records; the coordinator serializes
the report once. Recovery additionally retains claimed base targets and
persists claim/start/terminal events before allowing dependent work.

Keep persistence in the CLI initially. Public `ConvertOptions`, conversion JSON
and Node/Python/Go calls must not gain report/history side effects. This work does
not implement the REST job server, provider batch submission, or an interactive
workspace. History uses the compatible job export format without requiring them.

## Controls and lifecycle

| Control | Behavior | Status |
|---|---|---|
| `output.report = null` | Off for one file/URL; on for directory/URL-list batches | Implemented |
| `output.report = true/false` | Explicit report selection; batch state remains independent | Implemented on Unix |
| `--resume` | Load and merge directory/URL-list state; unavailable/corrupt state starts fresh with a diagnostic | Implemented on Unix; unsupported for single-item and non-Unix resume |
| `--record-history` / `--no-record-history` | Override environment/configuration; preserve paired-flag precedence | Enabled history rejected; disabling supported |
| `MARKITAI_RECORD_HISTORY` | Trimmed nonempty value overrides config; `1,true,yes,on` enable, other nonempty values disable | Precedence implemented; enabled history rejected |
| `history.record` | Default false; applies without a CLI/environment override | Enabled history rejected |
| Stdout conversion | No single-item report side effects | Implemented; history remains unsupported |
| Dry run / empty discovery | No report; no recovery/history files are currently written | Implemented |
| Conversion failures | Exit 1 for a failed single item, 10 for ordinary batch conversion failures; fatal state errors exit 1, signals 130/143; otherwise 0 | Implemented |
| History write failure | Warning; preserve the conversion exit status | Planned |
| Report write failure | Preserve completed output and JSON items; surface error, exit 1 or retain batch failure 10 | Implemented |
| State write failure | Stop dispatch and preserve observed outcomes; do not publish a partial success report; retain the original interruption exit | Implemented on Unix |

The planned history stage will publish after normal success or partial failure
using only this invocation's outcomes. The reference interruption path flushes
resume state but does not publish history; do not mistake an interrupt for an ordinary failed run.
Quiet mode and JSON stdout must remain compatible while diagnostics use stderr.

Sources: `cli/main.py:950,1429`, `runs/report.py:209`,
`cli/processors/file.py:279`, `cli/processors/batch.py:974,1466`,
`cli/processors/url.py:905,1459`, `runs/history.py:267`.

## Identity and physical paths

Reports belong to the output directory. The state paths below are implemented
by the store and are created by Unix batch dispatch:

```text
<output>/.markitai/reports/markitai.<hash>.report.json
<output>/.markitai/reports/markitai.<hash>.v2.report.json
<output>/.markitai/states/markitai.<hash>.state.json
<output>/.markitai/states/markitai.<hash>.state.jsonl
```

Use the reference six-hex MD5 task identity. Hash the UTF-8 representation of
Python `json.dumps({"input": resolved_input, "output": resolved_output,
"options": selected}, sort_keys=True)`: recursively sorted keys, default spaces
after separators, and ASCII escaping. Compact serde JSON is not interchangeable.
Absent options and explicit false differ. This is a compatibility filename,
not a cryptographic ownership check.

| Caller | Hash inputs |
|---|---|
| Directory state/report | Input/output directories; five flags `llm,ocr,screenshot,alt,desc`, scan depth and nonempty normalized globs; preserve supported legacy flag aliases |
| URL-list state | List path and output directory; five flags |
| Single-file report | File path and output directory; five flags |
| Single-URL report | Output directory for both path arguments; `llm` only |
| URL-list report | Output directory for both path arguments; `llm,alt,desc` |

Do not unify the URL-list report and state hashes. Source URL, content/mtime,
model, prompt, profile, pure mode, strategy, cache and conflict policy are absent
from some or all hash identities. Completed-state reuse does not validate them.
Check the stored input/output scope before trusting any saved target.

The reference directory report helper tries `.v2` through `.v9999`, then a
timestamp. Its shared helper for single files, single URLs and URL lists instead
increments `.vN` without a bound. Native publication deliberately bounds both:
after `.v9999`, directories try a timestamp then timestamp-plus-UUID on collision;
the other modes use UUID names. At most 64 fallback candidates are attempted.
Native report `skip` preserves existing bytes, correcting the reference's
overwrite behavior. These differences are explicit acceptance targets.

State always has one stable base and sidecar. Write checkpoints by atomic
replacement, never version renaming.

File keys are relative to input. URL state keys are the bare URL, or
`url + " " + raw_custom_name`; preserve raw spelling before adding `.md` to an
output filename. `name` and `name.md` are distinct identities. Deduplicate exact
URL/name pairs across directory lists, retaining different names as separate work.
Named URL records also retain their actual URL and source list.

Sources: `batch.py:632,674,682`, `utils/cli_helpers.py:131,178`,
`cli/processors/url.py:914,1161,1463`, `cli/processors/file.py:299`,
`urls.py:122,155`, `constants.py:414`.

## Report projection

Match the serialized version `"1.0"` report, including the transformations in
`order_report`, rather than merely serializing its builder's intermediate map.
Common fields are `version,generated_at,log_file`; absent logging is null.
Timestamps use local-offset ISO format. Directory reports additionally carry
`started_at,updated_at,options`; single-file and URL-list reports omit options.

| Mode | Required differences |
|---|---|
| Single file | Successful output only; one completed document, basename key; item usage is flattened to input/output tokens and cost |
| Single URL | One completed URL and zero documents; top-level feature/cache/strategy controls; per-model item usage and separate cache flags before transformation |
| URL list | URL counts, no documents; success, failure and `skipped`/`Output exists` entry shapes differ; do not invent missing per-item duration/usage/cache fields |
| Directory | Document and URL total/completed/failed/pending counts, URL cache hits/source count, wall duration and summed processing time; document keys relative to input |

Transform flat `urls` into `url_sources[source_file]` groups containing
`total,completed,failed,urls`. Remove `source_file` from grouped entries. Preserve
sorted group/document keys and within-group URL insertion order. A single URL
uses `cli`; current URL-list reports with absent source provenance use
`unknown.urls`, an observed format quirk that must remain explicit.

When URL cache flags are present, report `cache_hit` is fetch OR LLM, with
`cache_details:{fetch,llm}`. Do not synthesize details from absent flags; directory
reports retain the narrower reference state-shaped entries. Their `cache_hit`
and summary `url_cache_hits` are LLM-only. The existing CLI stdout meaning of
`cache_hit` remains LLM-only. Keep the flags separate internally.

Format numeric durations as one-decimal seconds below a minute, then `MM:SS` or
`HH:MM:SS` using truncated integer seconds. Preserve null durations. Keep numeric
values in run state; report formatting must not mutate the model.

Top-level usage is `{models,requests,input_tokens,output_tokens,cost_usd}`; sort
models, sum requests/tokens from their records and cost from item totals.
Both directory and URL-list aggregate model records initialize and accumulate
`cached_input_tokens`; URL-list aggregation includes completed items only.
Directory skips count as completed, while URL-list skips retain their separate
status and do not increment completed counts. Directory pending counts include
failed work. Planned minimal resume state cannot recover past usage, warnings or
skip reasons, so a resumed report must not fabricate those values.

Sources: `runs/report.py:19,31,58,82,164`, `json_order.py:161,242,268,329`,
`batch.py:338,1288,1330,1635`, `cli/processors/url.py:1193,1370`.

## Recovery journal and scheduler

Base state contains `version,options,documents,urls`. A file entry always has
`status`; a URL entry additionally has `source_file`. Completed entries retain
`output` when available; failed entries retain `error`; unfinished entries retain
their reserved base `.md` `target`. Newly written output/target paths are absolute.
Read legacy detailed states and cwd-relative strings without silently changing
their meaning; validate their resolved write scope before reuse.

State transitions are `pending -> in_progress -> completed|failed`. Acquire the
work slot before marking in progress. A successful skip is completed. Reloaded
in-progress entries become failed and are eligible for retry.

1. Bound and parse the base; retain the reference 10 MiB base ceiling. Missing or
   corrupt base cannot be recovered solely from JSONL.
2. Replay legacy lines shaped `{type:"file"|"url",key,data}` in order. Update
   known base keys only. Absent output/error/target fields keep prior values;
   absent status resets to pending in the reference. Skip malformed JSON syntax;
   a semantic or decoding failure stops the remaining legacy replay and retains
   the already-applied prefix, with a diagnostic. Bound journal/line/entry reads.
3. Keep completed entries even if outputs disappeared. Rediscover input, merge
   new keys and preserve old entries. Directory pending entries can refer to
   deleted files; URL scheduling uses current list entries. Preserve the legacy
   bare-URL adoption rule: move it to at most one named entry only when no unnamed
   entry still claims that identity.
4. Reserve completed output names before new claims. Retry a failed/interrupted
   item at its recorded, validated owned target without `.v2`; absent a target,
   use normal conflict policy. Another item in the run always gets a distinct
   name when producing a new output under every policy. Several items may skip
   the same preexisting file without claiming new names. A merely pending entry with a
   target does not acquire the failed-item overwrite privilege.
5. Publish the full merged base before any newly discovered work. Persist claims
   before expensive conversion, then append dirty status records. Compact by
   atomically replacing the base before removing the sidecar, using the replay
   fence below so a crash between these operations cannot regress saved state.
6. Flush on normal completion and controlled cancellation/error. Handle Ctrl-C
   explicitly and stop scheduling; scoped threads alone do not supply this
   lifecycle. Flush errors during abort must retain the original error.

Respect `batch.state_flush_interval_seconds` (default 10 seconds; the
reference treats zero as five). A single coordinator can serialize mutation
without the reference's per-save worker lock. Add exclusive cross-process run
ownership so competing invocations cannot rewrite one checkpoint concurrently.
Write through same-directory temporary files and sync at documented durability
boundaries. Do not promise exactly-once paid requests: process kill between a
provider response and a durable completion can repeat work, and no transaction
spans provider billing and the filesystem.

### Native replay fence and staged implementation

The internal store implements this native state extension. Keep the reference `"1.0"` envelope
and entry fields, and add namespaced checkpoint metadata containing a generation,
applied sequence and validated run scope. Native journal events retain
`type,key,data` plus the same generation and their sequence. Every mutation,
including buffered completion, receives a sequence before entering a snapshot.
On compaction, snapshot all changes through sequence N, sync the temporary base,
atomically replace it with applied sequence N, and sync its directory where
supported before removing/truncating the old journal and syncing that change.
Replay ignores other generations and events at or below the base's applied
sequence. A fresh run uses a new generation. Claims are acknowledged only after
their events and any preceding buffered changes are flushed and synced.

Without this fence, a base containing a newly completed item can be overwritten
by an old journal's `in_progress` event after a crash between base replacement
and journal deletion. Atomic file replacement alone does not prevent that window.
Legacy untagged state is read with its reference rules, then upgraded under the
exclusive lock before new work; a native tagged base must not replay leftover
untagged events. The Python reader ignores added keys but does not implement this
fence, so native crash guarantees do not apply to reopening these files in Python.
No state filename/discovery change is intended.

The storage implementation has authored legacy fixtures, bounded reads, scope
checks, an OS-backed process lock and deterministic crash/failure tests around
claim sync, base rename and journal cleanup. It keeps the lock inode stable.
The Unix scheduler now adds durable admissions, periodic flushes, recovered versus
observed outcomes, member ownership and explicit signal handling. A first signal
drains active blocking core work and suppresses report publication; it cannot
cancel an already submitted provider request. Separate member leases cover
cooperating writers across different hashes and overlapping roots. Corrupt state
is preserved before replacement, while foreign scope remains a hard error.

Sources: `batch.py:51,193,208,220,314,406,444,1048,1137,1214,1263`,
`cli/processors/batch.py:118,147,1005,1140,1210,1428`,
`cli/processors/url.py:1186,1257,1270`, `security.py:130,192`,
`utils/output.py:125`, `config.py:530`, `constants.py:18`.

### Output ownership in the scheduler

Preserve independent CLI processes writing unrelated files in the same output
directory. A whole-parent lock held across conversion or network/model requests
would unnecessarily make those invocations fail as busy. Ownership follows
the actual Markdown member paths instead, using stable OS locks and deterministic
all-or-release acquisition. The parent identity must be physical and respect the
original symlink policy; case/Unicode aliases need filesystem-level validation.
Stem-only identity is insufficient: `x` and `x.llm` can both address `x.llm.md`.
Do not remove a held lock inode or infer ownership from a PID file.

A failed/interrupted retry needs publication evidence bound to the complete
scope, native generation, typed raw item key, actual member path, file object and
bytes. Merely reading a matching digest/path from edited state is insufficient.
Persist prepared-write evidence before atomic publication so a kill between
rename and receipt finalization can be recovered. A prepared receipt can remain
the evidence when its staged file identity matches the published file, avoiding
a mandatory second receipt write just to rename its phase. The bounded protocol
and its limits are described in [output ownership](../output-ownership.md).

Observe publication inside the core write path: the existing model failure path
can save base Markdown and then return an error, so recording only successful
CLI outcomes misses ownership. Integrate all cooperating CLI disk writers while
keeping existing binding entrypoints and serialized JSON unchanged. Explicit
fresh overwrite is a separate authority from implicit failed-item retry. Tests
must cover competing hashes/overlapping roots, member collisions, crash windows,
changed files and independent-output concurrency. No portable pre-rename check
eliminates a hostile noncooperating writer's race; keep that limitation explicit.

## Optional history export

The native implementation follows this design. See [history](../history.md) for
the current interfaces, resource limits and remaining validation boundaries.

Publish one job under `config::home()/serve/jobs/<12-hex-uuid-prefix>/`, honoring
`MARKITAI_HOME` and independent of `cache.global_dir`. Never use the reference's
import-time real-home constant. No processed outcomes means no job; resumed
completed entries are not new history items.

Create a hidden sibling `.tmp-<uuid>/out`, copy output/assets, write `meta.json`
last, then rename the staged directory to its unique ID. Clean staging on error;
an individually missing/uncopyable output can remain an item with null output.
Global publication failure is a warning, not a conversion failure.

Metadata fields are `job_id,created_at,finished_at,status:"done",options,
dir_size_bytes,items`. CLI options are `{preset,llm,ocr,origin:"cli"}`. Size is
measured before writing metadata. Each item carries `item_id,name,kind,status,
error,output,output_name,duration_ms,finished_at,cost_usd,llm_enhanced,
operation:"convert",skipped,skip_reason,retryable,warnings`.

Map failures to `error`, completions/skips to `done`; round duration to
nonnegative milliseconds and deduplicate warnings in encounter order. URL items
are retryable; file items are not because original inputs are not copied.
`output_name` is the base `.md` name even for `.llm.md`; enhanced requires that
actual enhanced output and no skip. Do not invent success for future provider
batch-pending work; that feature remains separately unsupported.

Flatten outputs into `out/` with case-insensitive reservations and ` (2)` suffixes
before the complete `.llm.md` suffix. Copy `.markitai/assets`,
`.markitai/screenshots` and visible `assets`, with bounded source-root ascent;
exclude reports/state and original uploads. Deduplicate identical assets and
rename differing collisions with `-2`, `-3`. Rewrite each document using its own
asset map, including percent-encoded paths, wikilinks and frontmatter. Protect
literal code and exact target boundaries rather than replacing arbitrary text.

Sources: `runs/history.py:50,77,84,95,125,166,192,218,267,301`,
`cli/main.py:1429`, `cli/processors/batch.py:63,97,1475`.

## Reference defects and deliberate safety boundaries

Compatibility does not make these reference limitations new guarantees:

- Hashes omit important settings and have only six hex digits. Preserve filenames,
  but validate stored run scope and output ownership; do not infer ownership from
  a hash match. Keep completed-output reuse behavior documented.
- Directory resume may append new-key events without first saving merged keys,
  which replay then ignores after a hard kill. The native coordinator saves
  merged keys first, matching the safer existing URL-list path.
- Reference locks are process-local and JSONL is not explicitly bounded. Planned
  native ownership/limits need their own tests, separately from schema parity.
- An edited state's arbitrary absolute target must not authorize an overwrite
  outside validated output scope or through forbidden symlinks.
- Existing report `skip` can overwrite. Native publication now preserves that
  report, distinct from document processing; this has dedicated acceptance tests.
  Do not claim exact reference behavior for this branch.
- Reference shared report naming increments versions indefinitely. Native naming
  bounds version/fallback attempts as described above; exhausted candidates
  produce a normal publication error rather than an unbounded loop.
- Reference explicit enhanced-output reports can retain a stale intermediate
  path. Native reports use the actual finalized output; subprocess acceptance
  checks that the reported file exists.
- Minimal state loses old usage, warnings and skip reasons. Preserve absence
  rather than inventing recovered totals. URL-list grouping/hash quirks remain
  compatibility targets, not reasons to redesign all serialized output.

Reports, recovery and planned history metadata can contain plaintext URLs, custom
names, local paths and errors; history additionally copies document contents and
assets. Reports do not serialize document bodies or complete provider settings.
Hash filenames provide no confidentiality. Keep global history opt-in; test only
explicit temporary roots, disable real providers and never copy existing real
caches/history as fixtures. Bound future asset copies and apply the native
output containment/symlink policy before every publication.

## Delivery order and acceptance gates

1. **Typed terminal model and atomic report writer — implemented:** preserve raw
   named-URL identity and reservations; golden-test Python-compatible hash bytes,
   including Unicode and spaces. Keep stdout/binding JSON unchanged.
2. **Four report projections — implemented:** single-file, single-URL, URL-list
   and mixed directory coverage using temporary files, loopback HTTP and mock
   models. Tests inspect final JSON, defaults, cache flags, usage, timing, actual
   output, conflicts and write failures. The final source gate passes 296 Rust
   executions / 273 distinct tests plus 23 Python harness tests. Development R3
   matches four successful reference report schemas, values and recursive key
   order, with empty model usage. Nonzero usage has native mock/unit coverage;
   neither it nor multiple-model ordering has reference differential evidence
   from this run. R4 subsequently validated the same four pairs using the clean
   release binary; host packages also passed their scoped checks. See the
   [release evidence](../validation/reports-round7.md). This does not establish
   differential coverage of all failure, mixed-input and model-usage branches;
   this historical run predates resume integration.
3. **Recovery storage and scheduling — implemented:** the internal
   codec/store passes 57 shared tests and 13 authored reference snapshot/replay
   pairs; see [storage validation](../validation/state-round8.md). Real-process acceptance tests now cover
   failure-only retries, already-completed runs, mixed file/URL collisions,
   custom-name identities, legacy adoption, changed cwd, corrupt/oversized and
   truncated journals, claim ownership, interruption, merge-before-work and
   cross-process exclusion. Report-disabled batches must still be resumable.
4. **History export — planned:** test all dispatch modes, precedence, stdout/dry-run/quiet,
   partial failures, no new items after completed resume, exact item metadata,
   staged publication failures, asset deduplication/renaming and reference
   rewriting with encoded spaces, wikilinks, frontmatter and literal code.

Current CLI integration uses `app.rs`'s `Task`, `parse_urls`,
`reserve_batch_names`, `convert_item`, `outcome` and `finish_report`. `report.rs`
owns the typed terminal records and ordered projections; `report_store.rs` owns
hashing and atomic report publication. Failure timing and original usage survive
until projection. `batch_run.rs` extends the existing naming plan with physical
member leases, durable dispatch and recovery; it applies independent file/URL
concurrency caps with two queues. Its stable checkpoint replacement has a different lifecycle
from versioned report publication.

Reference test anchors, relative to `packages/markitai/tests/unit/`:

| Contract | Tests |
|---|---|
| Reports and formatting | `test_cli_helpers.py:109,147,196,217,260`; `test_json_order.py:73,168,186,218,268,317`; `test_batch.py:871` |
| Claims, cancellation, cwd, named URLs | `test_batch_resume_state.py:36,93,153,210,251,310,384,542,572,611,680,703`; `test_batch.py:1058,1116,1154,1172` |
| Minimal state, journal and concurrency | `test_batch_processor.py:1508,1663,2033,2063,2101,2132,2196`; `test_batch_thread_safety.py:11,33,64,105` |
| History lifecycle | `cli/test_record_history.py:71,154,170,182,192,213,245,278,307,449` |
| History schema and asset collisions | `runs/test_history.py:40,123,150,171,191,216,224,242,272,288,326,372,407` |

Each stage needs fresh source/artifact provenance and its own passing tests before
the corresponding unsupported guard is removed. Report source tests, four
clean-release differential cases and installed host checks have passed. The
[state storage](../state-storage.md) implementation is validated separately from
CLI scheduling. Recovery dispatch and history have no runtime validation claim.
