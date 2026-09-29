# Bounded group publication

Directory and URL-list workers prepare exact final Markdown bytes while retaining
the native output claim. Preparation is provisional: it does not emit success,
complete recovery state, or write the image metadata index. Assets and screenshots
can already exist, as in the immediate conversion path. The core's owned prepared
result retains paid usage even when enhancement or publication fails. Single-file
conversion and language bindings retain their immediate publication path.

The coordinator groups at most 16 documents and 64 MiB of rendered Markdown.
Each document has one combined receipt for its base and optional enhanced member.
Empty and oversized plans publish through the immediate path without converting
or requesting model work again. A full group, end of work, interruption, fatal
coordinator error or a 100 ms age observed by the coordinator triggers draining.
That age is a scheduling target, not a wall-clock completion guarantee.

Three ordered durability phases precede acknowledgement:

1. Sync every staged document and receipt, including their temporary directory
   entries, then complete the media fence.
2. Install all receipts and sync their directories, then complete the media fence.
3. Install all document members and sync their directories, then complete the
   media fence. Only now finalize image indexes and terminal results.

Every phase retains the same claims. The scheduler independently owns each claim
through terminal recording, including when a group operation fails. Failures may
leave some installed receipts or document members; they do not roll back the
whole group or report provisional successes. Durable receipts authorize only
exact prior/prepared proofs during a later retry. Modified foreign files and
substituted temporary names remain protected by the existing ownership checks.

On a verified local macOS APFS/HFS volume, each object receives ordinary `fsync`,
followed by `F_FULLFSYNC` through a retained descriptor on that volume for each
phase. The inference is that the full device cache flush covers the preceding
completed object flushes. Apple's [storage session](https://developer.apple.com/videos/play/wwdc2019/419/)
describes the distinction between OS-cache synchronization and flushing a drive's
cache; its [fcntl manual](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/fcntl.2.html)
describes `F_FULLFSYNC`. This uses a full flush, not `F_BARRIERFSYNC` ordering.
Other filesystems/platforms retain per-object `File::sync_all`. Each phase holds
at most 32 volume descriptors; additional volumes use the per-object path.
An error fails the group, without a weaker success fallback.

The guarantee remains conditional on filesystem and hardware flush behavior.
Injected fence failures and real process-kill/reopen tests exercise ordering and
ownership; they cannot simulate physical loss of drive power. Output-ancestor creation fences are retained. Controlled claim metadata
initialization uses the separately checked admission protocol below. The 64 MiB limit bounds staged
Markdown, not total process memory or active conversion memory.

Admission persistence is separately batched before dispatch; see
[recovery state](state-storage.md). Worker concurrency remains configurable,
including values above 16. Publication groups remain bounded independently.
Performance gains require fresh equivalent-output measurements; the old directory
regression is not closed by source-level syscall accounting.

## Claim namespace admission

Before non-skip admission, the coordinator prepares at most16 output parents per
subwindow. It opens the parent plus the controlled `.markitai/ownership/members`
and `records` chain, retaining directory identities and descriptors. Newly created
and already existing directories receive the same synchronization: a concurrent
initializer may have created an entry without completing its own fence. Each
held directory is synchronized before the volume fence; the proof is checked
before and after that fence and again around actual member-claim acquisition.
There is no persistent initialization marker or process-wide proof cache.

The extra descriptor bound is96 per subwindow (80 directories plus at most16
volume anchors), or81 on one volume. This is an additional bound, not a bound on
all process descriptors. The same per-object synchronization fallback applies off
verified local macOS volumes. Name reservation now uses the same durable output-ancestor creation as claim
acquisition before its case-sensitivity probe. Its previous create_dir_all could
pre-create nested output paths without their parent-entry fences, causing later
claim acquisition to see only already existing paths. Each newly created output
directory and its containing parent receive host synchronization, and before any
name probe, claim or dispatch one media fence per verified local volume covers
all output parents of the run (per-object durable synchronization elsewhere).
Staging follows every creation, so a directory's synchronization covers entries
made later in the same pass. Existing directories on each parent's chain below
the parents' common prefix are staged as well: after a crash before an earlier
commit their existence is not durability. Directories above that prefix (the
output root established by recovery state, and its ancestors) are not touched
here. Staging opens resolved paths with `O_DIRECTORY|O_NOFOLLOW`. A staging,
commit, policy or identity failure admits no work, and claim acquisition keeps
its immediate creator for any directory still missing. For the 24-file corpus
this replaces 24 of 45 full-cache flushes with one. The skip policy retains its
original path.

Ordinary parent preparation failures fail only the affected items. A failed
namespace fence stops new dispatch; earlier unsent reservations are restored.
Several namespace subwindows may feed one durable admission journal flush, so a
configured worker count above16 remains supported. Namespace preparation and
claim acquisition never grant permission to dispatch before that journal flush.

Admission fills available file and URL capacity before buffering excess tasks.
Queued reservations count toward their class and the total retained window;
completed publication frees room to refill either class even when the other
class still has queued work. Thus a backlog of earlier-sorted files does not
prevent later URLs from using their configured capacity. Reporting order and
per-class concurrency limits remain unchanged.
