# Provider Batch jobs

Provider Batch sends prepared requests to a provider's asynchronous queue. It is
separate from local directory concurrency, REST conversion jobs and MCP
`batch_convert`. Those interfaces continue to execute ordinary live requests.

```text
markitai input-directory --llm --llm-batch -o output-directory
markitai --llm-batch-collect batch_ID -o output-directory -c config.json
```

Collection needs the original output directory and matching configured model and
endpoint. It does not need the original input directory. Credentials are resolved
again from the current configuration and remain in memory. Inline configuration
and keys are never inserted into a suggested collection command or saved job.

OpenAI's Batch API and Anthropic's Message Batches are implemented; see
[supported scope](#initial-supported-scope) and [Anthropic](#anthropic-message-batches).

## Resuming an unfinished job

`markitai --llm-batch --resume -o output-directory` selects the unique unfinished
native job. INPUT is optional and, when supplied, must match the recorded resolved
scope; it may have been deleted. Frozen plans and request bytes are reused without
source discovery or reconversion. Each plan records how its protected markers were
made, so a job submitted with markitai 2.0.0 and collected by a later version checks
and restores its answers against the markers its requests carried. No pending job returns an explicit error instead
of silently starting a fresh submission. Rejected/completed jobs are not selected.

Prepared work rebuilds missing JSONL only from saved plans. An uploaded file is
reused. The creating phase is durably saved before the one paid create attempt.
Every retained output receipt and base hash is verified before paid work and again
after upload, with member leases held through creation. Modified outputs cannot
become new publication authority merely by matching a hash.

Creating/uncertain work only searches remote batches with bounded pagination and
an exact file/nonce/endpoint match, then verifies the unique result with a fresh GET.
No match, multiple matches or incomplete search remains unresolved and returns 2;
none authorizes another create. An explicit `--llm-batch-collect ID` may bind an
uncertain local attempt only after the same strict remote identity proof. Storage
corruption, ownership conflict or lock contention never becomes that fallback.
A new dollar budget does not block read-only reconciliation/collection; new paid
Batch creation still cannot enforce a continuation dollar budget.

## Initial supported scope

The current implementation accepts one effective OpenAI or Anthropic deployment
and structured text documents that fit one existing document request. Repeated identical entries
may share that identity; different endpoints, accounts, models, token limits or
fallback groups are rejected. A custom OpenAI-compatible endpoint is useful for
local protocol tests, but does not establish that the remote service implements
OpenAI's Batch API or uses its prices.

Preparation shares the live document prompts, protected source literals, typed
metadata validation, cache identity and output profiles. Valid cache hits publish
without a new provider request. Successful base Markdown remains beside enhanced
Markdown. The slide-marker preference is saved at preparation time and applied
after restoring protected markers; changing it during collection does not alter
that job's output. Long documents, pure mode, OCR, screenshots and image analysis are
currently rejected rather than silently sent through a paid live fallback.
Ordinary `--resume` refuses an input scope with an
unresolved native provider job.
The CLI currently permits one unfinished provider job per output directory.
A preparation lock enforces that boundary before any base file is published,
including submissions from different input directories with overwrite enabled.

`--llm-batch-timeout` is the current process's wait, with a minimum of 60 seconds
and a default of 3,600 seconds. It does not change the provider's processing window
and does not cancel the job. Waiting polls at most once per five seconds. An
interrupt or status-read failure retains the job and returns collection guidance.
Collection can be called again while the provider is still processing.

JSON submission uses the existing version 1.0 envelope, pending items, and a
`batch` object containing the provider ID, status and collection command.
JSON collection remains a usage error, consistent with the reference CLI.
Pending jobs return 2; completed jobs return 0; an item failure under the fail
policy returns 10; storage or provider protocol errors return 1.

For `llm.on_failure=fallback`, a failed provider response keeps the verified base
Markdown with a warning and its observed usage. No automatic live request is
made. This differs from the reference's automatic per-item live fallback. Output
ownership conflicts and edited files are errors and cannot be downgraded into
successful fallback.

## Anthropic Message Batches

An `anthropic/<model>` deployment submits to Message Batches
(`POST /v1/messages/batches`) with its `x-api-key` and `anthropic-version:
2026-10-21`. The lifecycle above is the same, with these differences:

- There is no file upload. The upload step only validates the saved
  `requests.jsonl` (each row exactly `custom_id` and `params`, one model, a
  positive `max_tokens`, nothing API version 2026-10-21 refuses: `temperature`,
  `top_p`, `top_k`, a `document` block in an `assistant` message or cited text
  in a `user` message) and records its digest; the single create call sends
  those rows inline after checking that they still match the digest.
- Each request is the live Messages request: the system prompt, the document
  and the same structured mode (forced tool for Claude Haiku 5.5, JSON schema
  for the other reviewed Claude models), with `max_tokens` from the
  deployment or 8,192.
- Custom IDs are `<submission nonce>-<n>`. A Message Batch keeps no metadata,
  so these IDs are the identity proof: reconciling an uncertain creation lists
  batches of the same size and opens the results of ended ones (at most four),
  binding only the one whose first result names one of this submission's IDs.
  A same-size batch that is still processing leaves the search unresolved, and
  `--llm-batch-collect ID` for an uncertain attempt waits until that batch has
  ended. Nothing in this search can lead to a second create.
- Status: `in_progress` and `canceling` are pending; an ended batch is
  completed, cancelled when cancellation began, or expired when no request ran
  before the 24-hour deadline. Results are read from
  `/v1/messages/batches/{id}/results` built from the validated ID, never from
  the batch's `results_url`, and matched by custom ID.
- `errored`, `canceled` and `expired` results carry no message and no usage
  (Anthropic does not bill them); they are item failures handled by
  `llm.on_failure` like OpenAI error rows. Succeeded messages are priced from
  the catalog's batch rates (half the list price).

Anthropic's own limits are 100,000 requests or 256 MB per batch; markitai's
transport limits below are lower and apply first.

## Batches Python markitai left pending

A batch that Python markitai 1.2.x submitted and had not collected when it was
replaced by this build is collected here: `markitai --llm-batch-collect ID -o
output-directory -c config.json`, or `markitai --llm-batch --resume -o
output-directory` when the output directory holds exactly one such batch
(with several, `--resume` lists them and each is collected by ID). A normal
`--llm-batch` run into that directory names the pending Python batch and its
collection command.

- Python's state in `OUTPUT/.markitai/batch-<ID>/` (`state.json`,
  `requests.jsonl`, `inputs/`) is only read, never changed or deleted. It becomes
  a native job in `.markitai/provider-batches/` that records the digest of the
  Python state it came from and a copy of Python's request rows, and continues
  from the submitted phase: status, results, publication and `llm.on_failure`
  work as for a job this build submitted. Python's batch ID is used as it is.
- The configured model must be the one the Python batch used (OpenAI or
  Anthropic, same model name); the credentials and endpoint come from the
  current configuration. A batch Python already collected is reported as such;
  one that ended without results (failed, expired, cancelled) is an error that
  asks for a new run.
- Each base Markdown Python wrote is adopted as it is at import (a native
  receipt is recorded for the same bytes), so an edit made after the import is
  preserved and reported like any other changed base.
- Answers are read as Python sent the requests: its tool names and its
  `__MARKITAI_*__` placeholders for images and page or slide markers, which are
  recomputed from Python's saved input with Python's rules and checked against
  the saved request. Each image, page and slide placeholder must come back
  exactly once; an answer that drops or repeats one is an item failure (no
  automatic live re-run, unlike Python). The enhanced Markdown then follows this
  build's output rules: the base frontmatter plus the answer's description and
  tags, page markers and output profile from the current configuration.
- Image analyses of a Python `--alt` or `--desc` run are applied as Python
  would: the caption becomes the alt text of the image with that file name in
  the enhanced Markdown, and with `--desc` the description goes to the
  `images.json` beside the image. A failed image analysis keeps the original alt
  text with a warning. Image analyses of documents Python enhanced live (too
  long for one request) are recorded for usage but not applied, because that
  Markdown is not this job's output.
- Usage is priced at batch rates like native results.

## Durable evidence and output ownership

Each attempt has a unique private directory below
`OUTPUT/.markitai/provider-batches/attempt-UUID`. The directory holds bounded,
versioned state, frozen plans, the exact request JSONL and response evidence.
An index maps provider IDs to local attempts. The index locates a job; its contents
alone grant no authority over any output.

The lifecycle persists each boundary before depending on it:

1. Save the credential-free plans and request bytes.
2. Upload an anonymous immutable snapshot of those validated bytes; save its
   uploaded file ID and content digest.
3. Save the creating phase before issuing the single non-idempotent create POST.
4. Save the returned job ID before waiting or returning collection guidance.
5. Save each raw result and its observed usage before semantic validation or
   output publication. Derive totals from those records on every replay.
6. Publish through the existing output-family leases and native receipts; then
   record completion. A published receipt can recover a crash between rename
   and job finalization without claiming unrelated files.

A disconnected or malformed create response can leave the provider outcome
unknown. Such an attempt retains its uploaded ID, nonce and plans and blocks
automatic resubmission. The nonce is reconciliation evidence, not a provider
idempotency guarantee. Explicit provider rejection is recorded separately.

Both success and error result files are downloaded, including completed partial
work from expired or cancelled jobs. Results match `custom_id`, never line order.
Unknown or duplicate IDs stop collection. A later interrupted or malformed file
does not erase complete rows already associated with an input. Replaying the
same result retains its first quote; conflicting response identities are errors.
Until every item is finalized, collection revalidates the complete result files.
Saving all expected rows before a malformed tail cannot turn a failed download
into a successful replay; repeated reads do not create new model requests.

Base and enhanced paths are relative to the saved physical output root. Collection
verifies the base digest and existing native output receipt before publication.
An edited base or enhanced file is preserved and reported as a conflict. Locks
remain held through verification and publication. File hashes in job state are
evidence, not substitutes for the native ownership checks.

## Usage and costs

Per-result provider token counters are recorded before structured decoding,
including observable paid error responses. Missing results do not create guessed
token counters or synthetic charges. Batch and later live attempts, when such a
fallback is added, must use their own billing classes; a document aggregate is
never discounted after the fact.

The reviewed native tariff catalog supplies published-price estimates. Unknown
models, custom endpoint prices or unsupported counters retain explicit unknown
coverage; numeric `cost_usd` remains the known subtotal. A saved quote and its
snapshot survive later collection with a newer executable.

Positive continuation dollar budgets are rejected for provider Batch submission: after cloud
submission the local process cannot stop individual queued requests at its own
document-spend threshold. Collection is still permitted with such a budget because
it reads an existing cloud job without starting new model work. Request preparation
and storage limits still apply.

## Bounds and validation limits

Transport limits include 50,000 requests, 200,000,000 input bytes, 8 MiB per JSONL
line, 1 MiB per control response and 256 MiB combined result bytes. The CLI store
adds its own lower document-count, per-blob and total-disk bounds. Limits are
checked before parsing or allocation where the wire format permits.

HTTP redirects and automatic retries are disabled. Downloads use validated IDs
under the configured API base, never arbitrary provider-returned URLs. Errors do
not echo credentials or raw response content. Test suppliers are local loopback
servers using fake keys and isolated state. Those tests do not establish live
provider acceptance, quotas, invoice totals or 24-hour service reliability.
