# Batch output publication

Directory and URL-list conversions can prepare several documents before
publishing their Markdown. Preparation alone is not success: output ownership,
file publication and recovery recording must finish first. Single-file calls and
language bindings use their immediate publication path.

The coordinator groups at most 16 documents and 64 MiB of prepared Markdown.
A full group, end of work or interruption drains pending publication; a short
age target also prevents low-volume work waiting indefinitely. This is a
scheduling target, not a wall-clock completion guarantee. Conversion concurrency
remains independent of the publication group size.

## Claim namespace admission

Output names are reserved before work starts. The coordinator retains and
rechecks the parent directory's identity and the private ownership namespace,
so redirecting a directory alias cannot redirect an already acquired claim.
New directories and coordination records are synchronized before dependent
publication can be acknowledged.

A group retains each document's claim through publication. Assets can already
exist before the document is acknowledged. A failed group can leave some
receipts or files installed; it does not roll back the entire group or report
unfinished items as successful. A later resume verifies exact ownership and
bytes before using an earlier prepared result. Modified foreign files do not
become replaceable merely because their names match.

## Durability and recovery

Publication orders prepared documents, receipts and final names before reporting
success. On supported local macOS filesystems it groups ordering barriers and
full flushes; other platforms retain their native per-object synchronization.
The guarantee depends on filesystem and hardware behavior. Process-kill tests
are not proof against arbitrary power loss, and the Markdown group bound is not
a limit on total conversion memory.

See [output ownership](output-ownership.md) for replacement and link policies,
[recovery state](state-storage.md) for `--resume`, and [output](output.md) for
files and asset naming. Publication does not send a second model request merely
to flush a prepared result.
