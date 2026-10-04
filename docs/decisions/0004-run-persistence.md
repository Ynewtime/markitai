# ADR 0004: separate reports, recovery and history

Status: accepted.

Reports, resumable batch state and optional history archives serve different
purposes. Keep them as separate projections of typed run outcomes, with separate
schemas and lifecycles. The stdout JSON envelope is another interface, not a
recovery checkpoint. Language bindings must not acquire CLI persistence side
effects.

- Reports describe observed outcomes, usage and output paths. Workers return typed
  records; the coordinator publishes the report. Reporting can be disabled
  independently of batch recovery.
- Recovery retains input identity, claimed output targets and ordered state
  transitions. Flush admissions before dependent work can issue provider
  requests. Hold a stable OS lock to exclude cooperating writers; use separate
  member leases and receipts for output ownership across run scopes.
- History is opt-in and copies generated documents and referenced assets into an
  independent archive. Publish a complete private stage atomically; a history
  failure warns without changing an otherwise successful conversion's exit code.

Preserve supported legacy serialization and naming for interoperability, while
validating saved scope and destinations before reuse. A short compatibility hash
is not ownership proof. Missing historical usage is unknown, not a newly measured
zero. Reports, state and archives may retain paths, URLs and content; they are not
encrypted merely because filenames contain hashes.

Current contracts and platform/crash boundaries are maintained in [Reports](../reports.md),
[Recovery](../state-storage.md), [History](../history.md) and
[Output ownership](../output-ownership.md).
