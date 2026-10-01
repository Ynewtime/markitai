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

Three phases precede acknowledgement; the first two need only ordering, the last
is the durability fence of [output ownership](output-ownership.md#ordering-and-durability-fences):

1. Sync every staged document and receipt, including their temporary directory
   entries, then complete an ordering fence: no receipt can reach stable storage
   before them.
2. Install all receipts and sync their directories, then complete an ordering
   fence: no document name can reach stable storage before its receipt.
3. Install all document members and sync their directories, then complete the
   durability fence. Only now finalize image indexes and terminal results.

Every phase retains the same claims. The scheduler independently owns each claim
through terminal recording, including when a group operation fails. Failures may
leave some installed receipts or document members; they do not roll back the
whole group or report provisional successes. Installed receipts authorize only
exact prior/prepared proofs during a later retry. Modified foreign files and
substituted temporary names remain protected by the existing ownership checks.

On a verified local macOS APFS/HFS volume, each object receives ordinary `fsync`,
followed, through a retained descriptor on that volume, by `F_BARRIERFSYNC` for the
first two phases and `F_FULLFSYNC` for the third. Apple's [storage session](https://developer.apple.com/videos/play/wwdc2019/419/)
describes the distinction between OS-cache synchronization and flushing a drive's
cache; its [fcntl manual](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/fcntl.2.html)
describes ordering phases with one barrier after per-descriptor `fsync`, and that
`F_FULLFSYNC` persists everything previously `fsync`ed on the device. A document's
receipt, staged files and claim metadata share its output parent's file system,
so the third phase's flush on each volume also persists what the barriers
ordered there. Other filesystems/platforms retain per-object `File::sync_all` in
every phase. Each phase holds at most 32 volume descriptors; additional volumes use
the per-object path. An error fails the group, without a weaker success fallback.

The guarantee remains conditional on filesystem and hardware flush and barrier
behavior. Injected fence failures, recorded fence sequences and real
process-kill/reopen tests exercise ordering and ownership; they cannot simulate
physical loss of drive power. Controlled claim metadata initialization uses the
separately checked admission protocol below. The 64 MiB limit bounds staged
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
held directory is synchronized before the volume's ordering fence; the proof is
checked before and after that fence and again around actual member-claim
acquisition. There is no persistent initialization marker or process-wide proof
cache. The fence orders the namespace before every lock, receipt and document
created in it; durability follows from the admission journal's full flush on the
output root's device and, on any other volume, at the latest from the first
document acknowledged there. A crash before then can lose only directories in
which no receipt or document can have reached stable storage; a retry's claim
acquisition recreates them.

The extra descriptor bound is96 per subwindow (80 directories plus at most16
volume anchors), or81 on one volume. This is an additional bound, not a bound on
all process descriptors. The same per-object synchronization fallback applies off
verified local macOS volumes. Name reservation now uses the same fenced output-ancestor creation as claim
acquisition before its case-sensitivity probe. Its previous create_dir_all could
pre-create nested output paths without their parent-entry fences, causing later
claim acquisition to see only already existing paths. Each newly created output
directory and its containing parent receive host synchronization, and before any
name probe, claim or dispatch one ordering fence per verified local volume covers
all output parents of the run (per-object durable synchronization elsewhere);
their durability follows as for the namespace above.
Staging follows every creation, so a directory's synchronization covers entries
made later in the same pass. Existing directories on each parent's chain below
the parents' common prefix are staged as well: after a crash before an earlier
commit their existence is not durability. Directories above that prefix (the
output root established by recovery state, and its ancestors) are not touched
here. Staging opens resolved paths with `O_DIRECTORY|O_NOFOLLOW`. A staging,
commit, policy or identity failure admits no work, and claim acquisition keeps
its immediate creator for any directory still missing. For the 24-file corpus
this replaced 24 of 45 full-cache flushes with one. The skip policy retains its
original path.

A single conversion's claim acquisition groups its own directory creation the
same way: the missing output chain and the `.markitai`, `ownership` and `members`
metadata directories are created first, each metadata level checked as before (a
real directory, private where required, on the output's filesystem) before
anything is created inside it; then every created directory and its parent are
staged and one ordering fence per volume completes before any lock file exists;
the publication's durability fence on the same file system persists them.
Directories this call created are staged even when a later check rejects the
claim. Creating nothing stages nothing and issues no fence, exactly as the
immediate path synchronized only directories it created. The immediate path
still follows and covers a directory removed and recreated in between. As
before, a directory another process created concurrently is not fenced by this
call, and a failed commit leaves the created directories unfenced for a retry
that finds them existing.

Counted with an interposed `fsync`/`fcntl` library on release builds (macOS 27,
APFS): a fresh single-file conversion now issues one full-cache flush, its
acknowledgement, plus two barriers (three flushes before); the 24-file corpus at
one job issues 11 flushes and 7 barriers (18 flushes before), of which the
publication protocol accounts for 2 flushes and 7 barriers (9 flushes before) and
recovery state storage for the other 9. In two runs the 32-file mixed corpus
issued 15 and 17 flushes with 7 and 10 barriers (22 and 27 flushes before).
Directory counts follow the number of publication groups and admission windows,
which depends on conversion timing: each group now issues two barriers and one
flush instead of three flushes, and each namespace window and the run's ancestor
preparation one barrier instead of one flush.

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
