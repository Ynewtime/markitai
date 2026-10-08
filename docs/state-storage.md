# Recovery state storage

The CLI connects this store to directory and URL-list dispatch. Every batch
persists merged work and flushes each admitted target before a worker can make
provider requests; `--resume` retains completed entries and retries unfinished
work. Reports and optional [history archives](history.md) remain independently
selectable; history is implemented separately from recovery state.
Admission may reserve multiple targets in one journal flush. `in_progress` means
the target is durably admitted; it does not prove that a worker or provider has
started. Dispatch still obeys the separate file and URL concurrency limits.
On a controlled interruption or fatal coordinator error, admitted but unsent
items restore their previous status, target, output, error and attempt diagnostics.
After an abrupt process kill, unfinished reservations follow ordinary resume.
The native conversion API and language bindings do not acquire recovery state.
See [output ownership](output-ownership.md) for the separate publication protocol.

## Files and run scope

A directory or URL-list run uses stable files below its output directory:

```text
.markitai/states/markitai.<six-hex-hash>.state.json
.markitai/states/markitai.<six-hex-hash>.state.jsonl
.markitai/states/markitai.<six-hex-hash>.state.lock
```

Directory state identity uses resolved input/output directories and the selected
feature, scan-depth and glob options. URL-list state uses the list path, output
directory and five feature flags; it is distinct from the URL-list report hash.
The shared Python-compatible MD5 serializer supplies the filename. Six hex digits
are not proof of ownership: the loader validates saved scope and destinations.

The store holds an OS exclusive lock on the open lock file for its lifetime.
It never removes that file to release the lock. Only this sidecar is locked:
a Windows lock is mandatory, and the base and journal stay readable. This excludes cooperating
writers of one checkpoint, including processes that start a fresh run. It does
not reserve document output names across different hashes or overlapping roots.
Separate [member leases and receipts](output-ownership.md) protect document names across those run scopes.

Original output path spelling is retained privately for symlink-policy checks,
while the serialized scope contains resolved identities. On Windows a resolved
path takes the final spelling of its longest existing prefix (as Python's
`realpath` there), so a short (8.3) name or another case is the same scope.
Document keys are relative paths with `/` between names on every platform, as
batch reports spell them; a key saved with `\` (a Windows reference state)
loads as the same item. Native checkpoint
publication anchors scope options, URL provenance and saved destinations to
absolute paths before durable encoding, so changing cwd cannot reinterpret them.
New journal mutations also store validated absolute destinations. The codec interprets
old relative destinations against the process cwd. It checks resolved containment,
mirrored parents and the metadata boundary independently of serialized path
spelling. It does not turn an edited path into permission to overwrite a document.
Each codec operation (decode, encode, event preparation or application) is pure
and observes each ancestor's `lstat`/`readlink` once, reusing it for every saved
path in that operation; the rules equal the core symlink policy and the report
path resolver, and each new operation observes the filesystem again. Recording
one event prepares and applies it under a single such observation (a compaction
between the two ends it, and the application observes afresh). The store's own
checks of a step (output spelling and its resolution, the states directory, base
and journal) likewise share one observation of their common ancestors; every
step, including the recheck before the base is renamed, observes again, and no
observation outlives the call that made it.
The existing core path policy is reused; hostile parent-directory replacement
races are not closed by these path checks.

## Legacy codec and ordered data

Legacy snapshots retain the `"1.0"` envelope with `options`, `documents` and
`urls`. Documents serialize by key; URLs preserve encounter order. State options
put the reference's preferred fields first and retain remaining top-level option
order. These use local ordered maps without changing global binding JSON order.

A completed entry keeps optional output; a failed entry keeps optional error;
unfinished entries keep their reserved target. URL records retain `source_file`
and named entries retain the underlying URL. Missing URL provenance and explicit
null stay distinct when serialized. Detailed legacy measurements are retained
as optional in-memory observations; missing values are not fabricated as newly
measured usage or time. Legacy encoding remains minimal; native checkpoints may
add the explicitly observed attempt diagnostics described below.

Merging preserves stored entries and adds discoveries. Legacy bare-URL adoption
moves one old entry to the first eligible named entry in explicit discovery
order, only when the current list has no unnamed claimant. Raw names such as
`name` and `name.md` remain separate identities. Merged keys must be checkpointed
before the store will accept their first event.

Legacy JSONL lines update only known file/URL keys. Missing status means pending;
missing output/error/target preserves its old value, while explicit null clears
it. Malformed JSON syntax is skipped with a bounded diagnostic. Semantic errors
or invalid UTF-8 stop further replay and retain the valid prefix. Only after all
replay does an in-progress entry become failed for retry. Missing/corrupt base
state cannot be reconstructed solely from a sidecar.

## Preserving a legacy pair before takeover

Before replacing a valid untagged legacy base, `begin` preserves the original
base and any journal in a private directory beside the current state:

```text
markitai.<hash>.state.json.legacy.<new-generation>/
  markitai.<hash>.state.json
  markitai.<hash>.state.jsonl   # only if the original existed
  manifest.json
```

This applies to both `--resume` and a fresh run replacing an existing legacy
checkpoint. A native tagged checkpoint does not trigger another legacy backup.
The original file bytes are copied, including unknown fields, whitespace and
journal lines skipped during replay. `manifest.json` records `version: 1`,
`kind: "legacy-recovery-pair"`, the new generation, and a base/journal descriptor
with `name`, byte count and SHA-256. A null journal descriptor means it did not
exist; an empty journal is preserved as a real zero-byte file.

Copies stream within the existing per-base/per-journal limits. The source must
be a regular file without a symlink leaf, even when general output symlinks are
allowed. The copy is checked against source identity and a second bounded digest
before publication. On Unix the backup directory is 0700 and files are 0600;
Windows has no mode bits and the backup inherits the states directory's ACL.
Files and the temporary directory are synced before the complete directory is
renamed into place; its parent is then synced (on Windows, where no directory
can be flushed, its manifest is flushed after the rename instead). Every file in
the staged directory is closed before that rename, which Windows requires. Only after that acknowledgment may
the current base be replaced and the old journal removed. Copy, verification or
backup-sync failure leaves the current base/journal unchanged. A complete backup
may remain after a later error, including failure to sync its final parent;
unfinished staging is removed. Existing corrupt-state quarantine is separate.

The non-quiet CLI reports the local backup path on stderr without dumping state
contents; stdout JSON keeps its existing schema. A missing state for the exact
paths/options can start a fresh run and is reported on non-quiet stderr. No other
hash is automatically selected. A loaded state whose saved options differ from
this run's (concurrency aside) is refused before any state change: the error
names each differing option with its saved and current value, also under `-q`. Foreign scope and Numbers-package child records
remain errors before native takeover.

Rollback here means recovering the old **state pair**, not undoing a run. Stop
all writers first. Preserve the current native base/journal and any new output,
verify the complete backup manifest and file hashes, then restore both original
basenames together (including journal absence). An older reader may need the
original cwd for old relative paths. Keep the backup itself intact. No automatic
rollback command deletes or overwrites the only native record.

New outputs, receipts and paid provider requests are not undone by restoring old
state. Work completed after the backup can be requested again, and the Python
writer does not enforce native receipts or fences. Do not run Python and native
writers concurrently against one output directory. Hostile directory replacement
or a writer ignoring the native lock is outside this guarantee. This feature
adds a reversible state-format boundary, not exactly-once execution.

## Native generation and sequence

A native snapshot adds `_markitai` metadata containing a UUID generation, applied
sequence and full run scope. Native events keep `type,key,data` and add their
generation and sequence. A fresh run gets a new generation; resumed native state
retains the loaded generation. Legacy state is upgraded before new events.

`record` validates and buffers one known-item mutation and returns its sequence.
That return value is not a durable claim. `flush` appends pending events and syncs
the journal and its directory before returning the durable sequence. The scheduler waits for this acknowledgment before starting work that relies on
recovery, including provider requests. A dropped writer does not pretend to flush.
The admission window retains at most the greater of 16 and the configured worker
count for the discovered task classes. A failed admission flush dispatches none
of that window. Non-skip claims first require a checked and synchronized metadata
namespace; a failed namespace fence likewise sends none of its new admission.
Class-aware admission counts queued reservations and refills available file/URL
capacity within the same total retained bound. Workers can release a conversion slot once exact output bytes
are prepared; completion is recorded only after the separate
[group publication protocol](grouped-publication.md) succeeds. Failed preparation
or publication remains a failed result, including its observed model usage.

Compaction writes the complete snapshot through sequence N to a same-directory
temporary file, syncs it, replaces the base, syncs the directory, then removes
the old journal and syncs that directory change. Replay ignores other generations
and events already included through N. Remaining events must advance in order;
unknown native keys, gaps, duplicates and reversal stop replay with a diagnostic.
A native tagged base does not replay leftover untagged legacy events.

This fence prevents a stale `in_progress` event from undoing a completed snapshot
if the process dies after base replacement but before journal removal. The Python
reader can ignore added fields but does not understand the fence; it does not
inherit the native crash guarantees. No protocol can promise exactly-once paid
requests across a process kill between a provider response and durable completion.

### Ordering and durability fences

The fences mean what they mean for [output ownership](output-ownership.md#ordering-and-durability-fences):
an ordering fence keeps every later write to the same device from reaching stable
storage first; a durability fence returns once what it covers is on stable storage.

| Step | Fence | Windows |
|---|---|---|
| Created `.markitai/states` chain and the parent that received its first entry | ordering | existence checked |
| `begin`, or a compaction forced by journal capacity: snapshot bytes, then the base name before the old journal's removal | ordering, ordering | snapshot flushed; flushed again after its rename |
| `flush` of a new or replaced journal: its bytes before its directory entry | ordering | journal flushed |
| `flush` acknowledgement (journal, directory, everything ordered before it) | durability | journal flushed |
| `flush` with nothing pending after an ordered checkpoint | durability | directory existence checked; the checkpoint was already flushed |
| Final compaction: snapshot bytes; base name before the journal's removal; removal | ordering, ordering, durability | snapshot flushed before and after its rename |
| Legacy-pair backup; corrupt-state quarantine | durability, as before | each copy flushed after its rename; the backup's manifest after the directory rename |

An ordered checkpoint does not advance the acknowledged sequence: the next `flush`
completes a durability fence even when no event is pending. The scheduler flushes
admission before any dispatch and compacts durably at the end, so each
acknowledgement it relies on is as durable as before. Until that fence, an
operating-system crash or power loss can leave the previous base with its journal
(every acknowledged event; after `begin`, nothing has been dispatched yet), the
new base with the superseded journal (whose events the generation and sequence
fence ignores), or the new base alone; only unacknowledged events can be lost,
as before. A base
or journal name never refers to incomplete bytes, and a journal is never removed
while the base it supersedes remains. The created directory chain can only be lost
while nothing inside it can be on stable storage; a chain that reaches another
volume than the checkpoint keeps an immediate durability fence. A process kill
needs no fence: renames are atomic in the operating system's cache.

On a verified local macOS APFS or HFS volume each object is synchronized with
`fsync`, and each fence is one `F_BARRIERFSYNC` (ordering) or `F_FULLFSYNC`
(durability) per volume, through the output claims' synchronization group.
Elsewhere, including Linux, each object still receives `File::sync_all` when it is
staged; there, the only change is that a journal removed under an ordered
checkpoint has its directory synchronized by the next flush instead of at once.
Windows has no barrier and no directory flush: each file is flushed with
`FlushFileBuffers` when it is staged, and a renamed base is flushed again after
its rename. NTFS writes its log of renames, creations and removals in order, so
that flush also commits the journal removal and directory creation that precede
it on the volume; directory staging only checks existence. A journal removal
has no file of its own to flush: until the next flush on the volume, a crash can
bring the superseded journal back beside the new base, which the generation and
sequence fence already ignores. The same steps make the same fence calls on
every platform.
Counted with an interposed `fsync`/`fcntl` library on release builds (macOS 27,
APFS), a fresh 24- or 32-file directory run now spends 3 full-cache flushes and
6 barriers on recovery state (9 full flushes before): one flush per admission
flush (two in these runs) and one for the final compaction.

## Atomic Numbers packages

Directory discovery records a `.numbers` directory package as one file entry.
Its children are never new work items, even if a glob excludes the package or its
contents are invalid. Resume skips its completed entry and retries its failed
entry using the normal output-ownership checks; no state schema change is needed.

An older scanner may have saved entries strictly inside such a package. Before
receipt adoption, checkpoint upgrade or worker dispatch, resume checks saved file
keys and URL `source_file` paths, including completed entries. A strict package
ancestor causes an explicit error. The package's own key is allowed. Absolute and
legacy cwd-relative URL provenance follow the codec's existing path rules; both
spelling and resolved paths are checked. The stable checkpoint lock may be opened
or created to perform this preflight, but rejection preserves existing checkpoint,
journal, receipt and output bytes. It neither deletes saved child entries nor
silently schedules them. Preserve that state and use a fresh output directory
without `--resume` to convert the package as one document.

## Native attempt observations

Native item entries and journal events may carry `diagnostics.last_attempt` with
`operation`, `status`, string/null `error` and full `usage`. The shared validator
requires an observed request, token count or model record, unsigned counters,
finite nonnegative costs and a consistent done/error shape. A present malformed
native observation fails semantic validation before an entry is changed; ordinary
corrupt-state and journal-prefix handling still apply. Error messages do not echo
the rejected object. Missing or null means no observation. Untagged legacy input
continues to ignore this unknown extension and keeps its existing minimal wire
shape.

The scheduler writes null when admitting a new attempt. A terminal event replaces
that value with this attempt's diagnostic or null; it never sums retries. A job
prepared but not sent restores its complete prior status and observation. Native
checkpoint encoding and compaction retain valid observations, and report recovery
uses them independently of the old success-usage aggregates. CLI resume work uses
operation `convert`, matching the public conversion entry point.

A synced completion preserves this observation under the existing generation
fence; it does not close the window between a provider response and that sync.
Unstarted work, hard kills and unparseable provider responses cannot supply known
usage merely because a task or output path exists.

## Limits and failures

Default limits are 64 MiB per base, 64 MiB per journal, 1 MiB per event line and
100,000 combined document/URL entries; the base is written as compact JSON, so
the entry limit is reachable with per-item usage diagnostics. Files are bounded
before unrestricted reads; serialization uses a bounded writer. Journal capacity
triggers compaction, and an oversized snapshot fails instead of becoming
unreadable on the next run. Because entries grow as items finish, a batch begins
only if its state still fits when every unfinished entry gains 512 bytes (its
output path and usage); otherwise it stops before any conversion with `Recovery
state projected base bytes limit exceeded`. Split such a batch.
Diagnostics do not dump raw state lines, provider configuration or document data.

New native base and journal files use private permissions on Unix (Windows
files inherit the states directory's ACL). Before
replacing a corrupt base, the store preserves both base and sidecar in
unique same-directory quarantine files, syncs them and retains private file
permissions. Oversized quarantine copies fail while leaving originals in place.
Foreign scope is an error, not corruption to overwrite as a fresh run. I/O failure
poisons the writer; callers must reopen and recover rather than append after an
uncertain partial write. Temporary staging is cleaned up when publication fails.

Directory synchronization is `fsync` on Unix and replaced on Windows by the
post-rename file flushes above. Validation must record the exact source,
platform and filesystem: cross-target checks alone are
not native execution, and successful recovery tests do not prove arbitrary
power-loss durability or behavior on every filesystem.
