# Persistent LLM cache

The native runtime can reuse successful non-pure local and URL text processing
across invocations and processes. A hit returns the saved typed Markdown and generated description/tags without an
HTTP request, retry delay, request-budget charge or new token/cost usage. The
normal output pipeline still applies current metadata, profiles and output paths.
Pure enhancement and standalone caption/description analysis do not use
this cache. Non-pure page/browser visual processing uses a separate batch namespace. Fetched response caching is a separate layer. There is no completed-answer process-memory cache. Identical active typed requests may share one response within the same caller-owned runtime, as described below.

## Configuration and CLI behavior

| Setting | Default | Effect |
|---|---|---|
| `cache.enabled` | `true` | Enables reads and writes for eligible document enhancement |
| `cache.no_cache` | `false` | Bypasses reads but saves a successful fresh answer |
| `cache.no_cache_patterns` | `[]` | Matching source contexts bypass reads but still refresh answers |
| `cache.global_dir` | `~/.markitai` | Directory containing `cache.db` |
| `cache.max_size_bytes` | `536870912` | Maximum sum of serialized cached-answer bytes |

The default state path honors `MARKITAI_HOME`. An explicitly configured custom
path keeps its ordinary meaning. Development and tests must select a private
state directory; they do not need to read an existing user's cache or credentials.
Disabled caching never opens or creates the database during conversion. Cache
inspection remains available when caching is disabled.

`--no-cache` sets `cache.no_cache`; `--cache` clears it without overriding
`cache.enabled`. The final occurrence wins. `--no-cache-for` supplies the existing
comma-separated pattern list. Patterns match the full source context or its
basename; Windows separators are normalized and `**/` can match zero directories.
The original context is used only for matching and is not persisted. These flags affect eligible LLM document calls and the independent
[static HTML/text fetch cache](fetch.md), which uses URL-aware pattern matching.

Persistent LLM entries have no TTL. `cache.fetch_ttl_seconds` is a separate URL
fetch setting and does not expire document answers. The reference processor did
not forward every public enabled/capacity setting, and its memory lookup could
precede bypass controls. Native conversion honors these controls consistently.

## Cache identity

The `native-document-v1` key namespace prevents legacy Markdown-only entries from
being treated as typed document metadata. Existing `native-markdown-v1` rows are
preserved and remain inspectable; there is no destructive migration. The key includes:

- A SHA-256 digest of the complete protected chunk, including edits in its middle.
  Protection tokens are deterministic for the whole source; changing source text
  containing protected literals can therefore also invalidate unchanged chunks.
- Resolved system/user prompt templates, JSON schema instructions, prompt category,
  applicable mode rules and the chunk/protection format version.
- The deduplicated, sorted set of model identifiers whose configured weight is
  positive. Automatic model selection uses the same model names as routing.

Prompt templates are digested before substituting timestamps or source labels.
An ordinary file rename, pool reordering, duplicate deployment, API-key rotation
or endpoint change does not invalidate an otherwise matching entry. Changing the
content, resolved prompt text or eligible model set does. Disabled models do not
participate. This follows the reference's model-pool scope rather than the model
chosen by one weighted request; an answer from any member can satisfy the pool.
Provider credentials and endpoints are not stored in cache identity or metadata.

A configured-model lookup inside the enhancement function runs before that
function resolves credentials, reads dotenv or constructs an HTTP client, so an existing answer remains usable when that
configuration's API key is temporarily unavailable. Automatic model selection
still needs the environment to identify its pool. Prompt files must remain
readable because their actual text determines whether an answer is reusable.
The normal CLI may already have read dotenv during configuration loading; this
lookup ordering does not promise that the complete CLI invocation avoids it.

## Visual batch identity and retry

`native-vision-v1` keeps visual entries separate from typed text and legacy
Markdown. Its length-framed digest includes the protected text for that batch,
resolved prompt templates and protocol marker, eligible positive-weight model
identities, and every ordered frame's number, MIME and full encoded bytes.
Changing image pixels, image order, prompt rules or first/cleaner response mode
invalidates the corresponding entry. Filenames alone never identify pixels.
Image payloads, credentials and provider endpoints are not saved in keys or rows.

Each batch has its own protection scope, so an unchanged batch can be reused
when another batch changes. A changed source-owned image reference or other
body text also correctly changes that batch's identity. First-batch entries
contain Markdown and validated description/tags; cleaner entries contain Markdown
and null metadata. Reads revalidate the schema, protected markers and visual
content guards. Rejected, truncated, refused or partial answers are not admitted.
A legacy entry cannot satisfy a visual lookup.

Successful batches may be cached even when another batch makes the complete
conversion fail. A retry sends only misses, while the final enhanced body still
requires every batch to succeed. A wholly cached visual document has zero new
usage; any new paid response makes the complete conversion's cache-hit indicator
false. The same bypass, capacity, isolation and SQLite initialization rules apply.
The reference's name-based visual cache identity is intentionally strengthened
with actual bytes to prevent stale OCR after image changes.

## Storage and failure behavior

SQLite is bundled into the native artifact. It uses WAL, normal synchronization
and a 30-second busy timeout. First-use WAL/schema initialization also retries
BUSY/LOCKED errors against a 30-second deadline, dropping the failed connection
before retrying. SQLite can bypass its busy handler while upgrading a rollback
journal read lock, so a busy timeout alone does not cover simultaneous first
writers. Only idempotent initialization is retried; row transactions are not
replayed, and corruption or other I/O failures remain explicit cache warnings.
The table retains the reference fields `key`,
`value`, `model`, `created_at`, `accessed_at` and `size_bytes`; typed document `value` is a JSON object containing `cleaned_markdown`,
`description` and `tags`; legacy entries remain JSON strings. `model` records
the pool fingerprint. The database
is created only when there is a successful answer to save.

Writes admit the replacement and evict the least recently accessed entries in
one immediate transaction. Ties use row insertion order. A result larger than the
configured capacity removes a stale entry for its own key but does not evict
other answers. An insert failure rolls back both replacement and eviction.
Failed, malformed, structurally damaged and token-truncated answers are not
stored. A validated boilerplate chunk may have an empty body and still carry
metadata; the merged whole document must remain valid and nonempty.
Stored answers also have a 100 MiB entry bound. Separate processes can safely
write the same database. Active typed text chunks and first visual batches can
merge within one shared `LlmRuntime`; independent runtimes and separate processes
can still send duplicate requests on simultaneous misses.

Capacity counts serialized answer bytes, matching the reference accounting; it
is not a hard limit on SQLite metadata, free pages or WAL file size. Reducing the
capacity takes effect on subsequent admitted writes. Clearing entries does not
promise to shrink the physical database file immediately.

Unavailable, malformed or unwritable caches produce a fixed warning and leave a
successful enhancement intact. Errors do not expose the database path, SQL,
credentials or cached content. Reads update `accessed_at`; if the database cannot
accept that update, the call degrades to a cache miss and may contact the model.
Typed stored values are schema-checked and their protected markers checked
against the current chunk before replay. Invalid stored values are ignored with
a warning; they never count as successful cache hits. Diagnostics describe cache unavailability rather than implying a free
provider request.


Each successful chunk is committed separately, including when a sibling chunk
later fails. The caller still receives a document-level failure and publishes no
partial enhanced result. A subsequent attempt requests only missing chunks.
`llm_cache_hit` is true only for an item fully reused from disk or active shared
requests, with no new paid LLM usage;
a cache-backed document followed by image-analysis requests is not a full hit.

## Active request sharing

Sharing is enabled only when the ordinary cache is enabled and the source does
not match `no_cache` or `no_cache_patterns`. Those refresh settings still force a
new request and save its successful result. No new option is introduced.

The active key is separate from disk keys: it additionally hashes complete
rendered prompts, image bytes and resolved endpoint/credential/routing policy,
using a runtime-local salt. It is neither logged nor persisted. Disk rows retain
the established model/prompt/content identity and can still be read before
credential resolution. A waiter validates the semantic answer against its own
source, receives zero newly paid usage, and never inherits the owner's budget or
error. A failed owner wakes waiters to retry independently. Cache writes occur
before a successful result is shared; write failures remain warnings.

Sharing is bounded to 128 active identities, 16 MiB per serialized answer and
64 MiB of retained serialized answers, with ordinary routing as the overflow
fallback. These are payload bounds, not a heap/RSS limit. Completed entries are
removed immediately. See [runtime accounting](llm.md) for lifecycle and scope
limits; persistent cache capacity and SQLite size are unchanged.

## Inspection

`cache stats --json` retains the existing `cache`, `enabled` and `fetch_cache`
envelope. An absent LLM database yields `cache: null` without creating a database.
An unreadable one yields a sanitized `cache.error` and the CLI reports failure.
Normal details include entry count, logical bytes/MiB, configured capacity/MiB
and the database path. Verbose details add model-group statistics and recent
entries, including timestamps and short content previews. The recent-entry limit
is capped at 1,000; counts and model groups use one consistent read snapshot.
Previews are intentional user-requested content display, not error diagnostics.

The storage clear operation deletes LLM table entries transactionally and returns
the deleted count; an absent database returns zero without creating it. The CLI
checks both LLM and fetch stores before clearing either one. With
`--include-spa-domains`, it also preflights the learned-domain store before any
deletion. Statistics include the independent fetch store.
A later deletion failure can still leave one store cleared; the CLI reports the
completed portion and exits unsuccessfully. Two WAL databases are not one atomic
transaction, and partial clearing is never reported as complete success. The
optional routing store is also independently transactional.

A cache hit reaches the existing CLI `cache_hit` and `llm_cache_hit` fields through
internal conversion state. It does not add fields to the JSON conversion protocol
used by Node.js, Python or Go bindings.

## Learned-domain management

`cache spa-domains` lists learned browser-routing authorities. JSON output keeps
the reference fields `domain`, `learned_at`, `hits`, `last_hit` and `expired`,
ordered by descending hits with a deterministic domain tie-break. Timestamps are
UTC RFC 3339 strings. `cache spa-domains --clear --json` returns the removed count
as `{"cleared": N}`; an absent store yields zero without creating a directory or
database. Ordinary listing is read-only and neither increments hits nor prunes
expired records.

`cache clear --include-spa-domains` includes this independent store in the
existing preflight and clear sequence. Separate stores cannot provide a single
atomic clear transaction; a later failure reports the already completed portion.
Neither `cache.enabled=false`, `--no-cache`, nor bypass patterns disable learned
routing. These controls govern cached document/page content, while the learned
store only records browser-routing experience. Its private native SQLite file
does not import or rewrite the reference's learned-domain JSON file. See
[fetching](fetch.md#learned-browser-routing) for identity, expiry and capacity.
