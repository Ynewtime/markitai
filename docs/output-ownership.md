# Output ownership and recovery

The CLI coordinates Markdown writers through filesystem member locks. Native
batch recovery additionally records evidence of the exact file object and bytes
published by an item. A saved output path, the six-character state hash, matching
frontmatter, or a content digest alone cannot authorize implicit replacement.

This document describes the implementation contract. Source, process and release
checks are separately scoped in [round-nine validation](validation/recovery-round9.md).

## Where the protocol applies

Single-file and single-URL CLI conversions acquire output member locks but do not
create a recovery checkpoint or publication receipt. Directory and URL-list runs
also maintain native checkpoints and receipts. Disabling reports does not disable
batch recovery storage. Stdout-only conversions and dry runs do not acquire
output claims; empty discovery without a stored run creates no persistence shell.

The existing core conversion functions and Node.js, Python and Go bindings do not
join this CLI protocol or create its metadata. The CLI passes an explicit native
publication callback into core; rendered Markdown, asset rewriting and result
paths still come from core. This includes a base document written before an LLM
failure is returned. There is no process-global callback or new binding JSON field.

Native claims require native file identities and locking, which Unix and
Windows provide (see [platform primitives](#platform-primitives)). A platform
without them fails the claim before provider work; there is no weaker,
pathname-only fallback. A directory or URL-list run also installs controlled
interruption before its first request, and fails there with an explicit error
where that is not available. Host conversion APIs retain their existing
publication path. Windows builds are type-checked at each integration; release
validation on Windows hosts is recorded separately and still in progress.

### Platform primitives

All of these live in one module (`markitai_core::platform`); the protocol code
above it has no platform conditions.

| Primitive | Unix | Windows |
|---|---|---|
| File identity | `st_dev` and `st_ino` | volume serial number and 128-bit file ID (`FILE_ID_INFO`; the 32-bit serial and 64-bit index where a file system lacks it) |
| Hard-link count | `st_nlink` | `FILE_STANDARD_INFO.NumberOfLinks` |
| Change detection | size, modification time, identity and `st_ctime` | size, modification time, identity and `FILE_BASIC_INFO.ChangeTime` |
| Private entry | mode grants nothing to group or others (`0600`/`0700` on creation) | owner SID is this process's user, or its token's default owner (an elevated administrator's files belong to `BUILTIN\Administrators`); new entries inherit the parent's ACL |
| Same owner (receipts, provider batches) | user ID | owner SID |
| Open without following | `O_NOFOLLOW` with `O_NONBLOCK` | `FILE_FLAG_OPEN_REPARSE_POINT`; a symbolic link or junction is refused; another reparse point (a cloud placeholder) is reopened normally and must be the same file |
| Directory identity | `O_DIRECTORY` descriptor | handle opened with `FILE_FLAG_BACKUP_SEMANTICS` |
| Directory synchronization | `fsync` of the directory | none exists: only the directory's existence is checked |
| Name of a renamed file | the parent directory's synchronization | `FlushFileBuffers` of the renamed file after the rename |
| Locks | `flock` (advisory) | `LockFileEx` (mandatory): only sidecar lock files are ever locked, never a file another process reads |
| Blocked rename | not retried | access denied, sharing or lock violations retried up to five times, 50 ms longer each, as the reference writer did |
| Path spelling | `realpath` | final path name without `\\?\` when the plain spelling names the same file, so a short (8.3) name or a different case spells the same path; missing components keep their spelling |

Case or Unicode aliases that the file system folds to one file share one
identity, hence one lock and one reservation key, on either platform. A Windows
junction counts as a symbolic link for `output.allow_symlinks`; Windows has no
root-owned system links, so nothing on the path is exempt.

## Locks follow actual filesystem names

For each actual output parent, stable lock files live at:

```text
<parent>/.markitai/ownership/members/<document member filename>
```

A family normally reserves `name.md` and `name.llm.md`, even when a conversion
ultimately writes only one of them. Thus `name` and `name.llm` overlap at
`name.llm.md`. The claim validates the exact members core may publish. Explicit
filenames and `llm.keep_base` still determine which member receives enhanced text.

The original parent spelling is checked against `output.allow_symlinks` before
physical resolution. Locks use the member filename itself. When case or Unicode
spellings refer to the same lock file on the actual filesystem, its file
identity also gives the same in-process reservation key. Lowercasing strings is
not used as a substitute for filesystem identity. Permitted parent aliases reach
the same physical metadata directory.

Each active item holds nonblocking OS locks for its family. Unrelated families
can proceed concurrently; there is no long-lived lock for the whole output
directory. Lock files remain after release and are never unlinked as a cleanup
step. Their presence alone does not indicate an active process. Publication
rechecks the held lock paths, directory identities and requested member.

Every check observes the path afresh. Within one check, the symlink policy and
the resolution of the same parent share a single walk of its components, and a
parent already walked by that check (for example, for the other member of a
family) is not walked again. The shared walk is used only for an absolute path
spelled canonically whose every component is an ordinary entry or absent; a
symbolic link anywhere (permitted, root-owned or not), `..`, a redundant
separator or an unexpected error repeats the separate policy and resolution
steps unchanged. No observation outlives its check: the recheck after directory
creation and every later protocol step walk again, so a link substituted in
between is still rejected.

Ordinary rename policy can try another version when a candidate is occupied or
busy. An owned retry must use its verified family. A skipped existing output does
not become owned by the skipped item and does not create a receipt. Completed
entries retain name reservations even when their output files have disappeared.
Those planning reservations do not themselves grant write authority.

## Durable prepared receipts

Native receipts are stored separately from locks:

```text
<parent>/.markitai/ownership/records/<full-sha256>.json
```

The locator hashes the full owner and exact family. The bounded receipt body is
also validated against them. The owner includes checkpoint generation, mode,
resolved input/output scope, typed file-or-URL identity and the raw item key. A
named URL retains its raw name in this key; it is not reconstructed from a
sanitized output filename.

Each member records its exact target and, when a write is prepared:

- The private temporary basename and staged regular-file proof: file identity
  (`device` and `inode`, holding the Windows volume serial number and file ID
  there), byte length and SHA-256 of the actual bytes.
- The expected prior object, if any, with the same proof and an explicit
  `explicit_overwrite` or `native_owned` authority.

An explicitly permitted symlink replacement records the symlink object's identity
and link-text bytes. It never follows the link merely to authorize publication.
An implicit retry cannot replace an unrelated symlink.

Publication performs these steps while the item holds its member locks:

1. Validate the owner, receipt and current family, then capture the expected
   destination. An ordinary new write requires absence; a native retry requires
   matching evidence; explicit overwrite supplies separate authority.
2. Write the rendered document to a temporary file in the output parent and sync
   it. Install the prepared receipt by atomic replacement: an ordering fence puts
   the staged document and receipt bytes before the receipt name, and a second
   one puts the receipt name before the output target is touched.
3. Recheck the staged object and expected destination. Publish with atomic
   no-clobber semantics for an absent destination, or replacement for the exact
   authorized prior object. Complete a durability fence on the output parent's
   volume before acknowledging success.

There is no required second receipt commit after document publication. Renaming
the staged file preserves its file identity, so the prepared record can prove
that publication happened even if the process stops before returning or
recording task completion. Preparing a later attempt replaces this evidence only
after retaining the currently authorized prior proof. Single-file and single-URL
conversions have no receipt; their staged bytes are ordered before the rename and
the same durability fence precedes success.

## Ordering and durability fences

An ordering fence guarantees that what it covers reaches stable storage before
any later write to the same device. A durability fence guarantees that it is on
stable storage when the call returns. Every success acknowledgement is preceded
by a durability fence; every other phase boundary uses an ordering fence:

| Point | Fence | Windows |
|---|---|---|
| Created output, `.markitai`, `ownership` and `members` directories of a claim | ordering | existence checked; committed by the next file flush on the volume |
| Run-wide output ancestors and each namespace window (see [grouped publication](grouped-publication.md)) | ordering | as above |
| Staged document and receipt bytes, before the receipt name | ordering | each file flushed when staged |
| Installed receipt name, before the document rename | ordering | the receipt flushed after its rename |
| Document rename, before success is reported | durability | the document flushed after its rename |
| Receipt copied for an adopted URL owner, before the checkpoint names it | durability | the receipt flushed |
| Records directory created by this call; directory created after a concurrent removal | durability (unchanged immediate synchronization) | existence checked |

Recovery state outside the publication protocol has its own
[fence table](state-storage.md#ordering-and-durability-fences). Writers that never
synchronize the published name (the core's content-addressed assets and immediate
documents, reports) order the staged bytes before the rename with the same kind
of ordering fence: such a file was not guaranteed durable on return before
either, and a name that survives a crash still refers to complete bytes.

On a verified local macOS APFS or HFS volume, each object is synchronized with
`fsync`, then a retained descriptor on that volume issues one `F_BARRIERFSYNC`
(ordering) or `F_FULLFSYNC` (durability). The [fcntl manual](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/fcntl.2.html)
describes this two-phase use of one barrier after per-descriptor `fsync`, and that
`F_FULLFSYNC` persists everything previously `fsync`ed on the same device. A file
system that rejects the barrier operation gets the full flush instead. Elsewhere,
including Linux, every object still receives `File::sync_all` when it is staged,
so both kinds remain per-object durable synchronization, as before.

Windows has neither a barrier nor a directory flush. Each staged file is flushed
with `FlushFileBuffers` (a full flush) when it is staged, so both fence kinds are
per-object durable there too. NTFS journals every rename, creation and removal
and writes its log in order; flushing a file after the rename that named it
commits that rename and every earlier logged change on the volume, which is what
the directory synchronization provides on Unix. A whole directory renamed into
place (a history job, a legacy backup, a serve job) is committed by flushing the
file inside it that the rename published. Volumes other than NTFS are not
verified; every one of them still gets the full per-file flush, the strongest
operation Windows offers, so no file-system check changes what is done.

Claim directories, receipts and staged documents share the output parent's file
system (the claim refuses metadata on another device), so the acknowledgement's
full flush on that volume also persists every earlier ordered write of the
publication. After an operating-system crash or power loss before the
acknowledgement, a document visible at its final name therefore has its complete
bytes and its receipt; a receipt name has complete bytes; the publication may also
be absent entirely, which recovery already handles. Acknowledged publications are
durable, as before. A process kill needs no fence: the renames are atomic in the
operating system's cache.

The barrier relies on the storage stack honoring it; IOKit reports this as the
media's `Barrier` storage feature, which Apple documents as guaranteed for Apple
SSDs. On an APFS disk image whose media lacks that feature, a barrier took as long
as a full flush (8.4 ms versus 8.7 ms, against 0.4 ms versus 3.5 ms on the internal
SSD), consistent with a fallback to a full flush rather than a silent no-op; the
APFS implementation is not public. The guarantee remains conditional on file
system and hardware flush behavior, which these measurements cannot prove.

For fresh explicit overwrite, the change-detection window starts when publication
preparation captures the destination, after conversion. Native retries are also
verified before expensive work and again at publication. Neither check provides
an atomic compare-and-replace against an uncooperative writer.

## Recovery decisions

| Observed member | Recovery behavior |
|---|---|
| Matches the prepared identity, length and digest | Recognize the native publication and allow an owned retry. |
| Matches the recorded authorized prior object | Retain the original authorization after a pre-rename interruption. |
| Absent | Recreate using no-clobber publication, subject to owner/receipt validation. |
| Same bytes but another file identity, or changed bytes on the same identity | Preserve the object and fail the implicit retry. |
| Existing object without matching evidence | Do not infer ownership from the path or content. |
| Malformed, foreign or conflicting native receipt | Fail without replacing the output or silently discarding evidence. |

Only failed entries, including interrupted entries normalized to failed after
journal replay, receive native target-reuse treatment. Pending entries use normal
conflict policy. Legacy checkpoints have no native receipt authority; their
unfinished items use configured rename, skip or explicit overwrite rather than
silently overwriting a saved target. A fresh generation cannot inherit an older
generation's implicit authority.

Legacy unfinished targets are cleared before a fresh native checkpoint is saved.
This prevents an interruption during that upgrade from promoting an unproved
legacy path into a native retry target on the next run.

One interruption window deliberately fails safely: a fresh explicit-overwrite
run can record its target and begin conversion before any publication receipt
exists. If it is killed there, an existing target has no native evidence and
resume rejects implicit replacement. The user can start a fresh run with explicit
overwrite. An admission-time durable authorization record for this window is not
implemented; recovery is not seamless at every possible kill point.

Completed entries retain their completed state without rereading or regenerating
their outputs. Receipt shape can disambiguate a result such as `x.llm.md`: it may
be the enhanced member of `x` or the base member of literal stem `x.llm`. This
lookup checks at most two candidate receipts and grants no publication authority.
Without unambiguous evidence, planning conservatively reserves both families.

When a native failed bare URL is adopted into its first discovered named key,
the CLI verifies the old family and durably copies its receipt under the new full
owner before saving the merged checkpoint. The old receipt remains, so a crash
before checkpoint replacement still permits recovery with the old key. An
existing destination receipt must contain identical member evidence. The copy
does not transfer authority across a generation, scope or underlying URL.
Completed-key adoption does not migrate overwrite authority or revalidate edited
completed documents. Retained URL entries keep their original source-list
provenance and mirrored output parent.

## State, interruptions and limits

The coordinator saves merged keys before dispatch. It obtains a claim and flushes
`in_progress` with its selected base target before releasing work to a converter.
Receipt persistence occurs at actual document publication. Buffered terminal
states use the configured flush interval; final compaction records known results.
A storage failure stops new work and is reported separately from conversion
failures. It does not turn a partial invocation into a normal finished report.

The first SIGINT or SIGTERM stops new dispatch, flushes known state and drains
active synchronous conversions. Shutdown time depends on their configured
timeouts and retries; this does not forcibly cancel an in-flight request. A second
signal exits immediately without promising another flush. Receipts preserve
publication evidence independently of terminal state acknowledgement.

This is not an exactly-once conversion or billing protocol. A crash can require
another provider call. Base and enhanced members are published separately, so a
failed operation can leave a valid partial family. Asset files use the separate
content-addressed writer and are not covered by document receipts.

Receipts are limited to 64 KiB and read with a bounded buffer. File proofs are
computed with bounded streaming reads. Ownership directories and receipt files
must be private (as each platform defines it above), ordinary filesystem objects
on the output filesystem; metadata symlinks and junctions are rejected. Normal failures remove only the staged object still
matching the recorded proof. Crash-orphan staging and receipt garbage collection
are not automatic.

Receipt contents include plaintext paths and raw URL keys, which can contain
query credentials. Private permissions are not encryption. Diagnostics do not
echo receipt bodies or owner keys. Development and tests must use isolated output
directories and `MARKITAI_HOME`; ownership metadata never requires reading the
user's real configuration home.

The guarantee covers cooperating CLI writers and detected external edits or
replacements. Bindings and other programs are external writers. A hostile writer
racing verification and rename, hostile parent replacement, or a coherently forged
checkpoint plus receipt is outside this guarantee. Filesystem locking and sync
behavior must also be supported by the underlying filesystem. The protocol is
not a security sandbox or a claim that pathname validation authenticates files.
