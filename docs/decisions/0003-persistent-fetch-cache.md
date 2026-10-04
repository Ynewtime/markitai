# Persistent fetch cache

Historical design and round-five validation record for the first static
HTML/text cache stage. Current behavior is documented in [fetch](../fetch.md).
The source audit used reference commit
`ba374322f884b0e720b45466cc1196f4574a3da5` and clean native source
`5280fd3c928066a31cffda260ef5b2270cef758a`, recorded in
[`artifacts-round4.json`](../validation/artifacts-round4.json).
Fetch-cache statistics and clearing are now integrated. Unsupported SPA-domain
management remains guarded before any cache deletion.

## First implementation boundary

Cache the extracted Markdown and page metadata produced by the native `auto` and
`static` HTML, text and Markdown paths. A cached page must enter the same output,
profile and optional LLM pipeline as a newly fetched page. Generated output
frontmatter, output filenames and LLM answers do not belong in this cache.

PDF, Office and other downloaded document formats continue to convert normally
without fetch caching in this first stage. They can carry in-memory asset bytes
which the reference fetch table cannot restore. Cache admission must therefore
require both an eligible response kind and no owned assets. Do not cache only
their Markdown and leave image references pointing to missing files.

Remote extraction strategies, browser rendering, screenshots and SPA-domain
learning are outside this stage. Existing remote fetch behavior continues; a
cache hit must not make an unsupported strategy appear implemented. This stage
does not add URL LLM caching, which is separate from page fetching.

## Identity and strategy provenance

The reference hashes the exact supplied URL string, including its query and
fragment. Its key is the first 32 hexadecimal characters of SHA-256 over
`"2\0" + scope`, where scope is either `url` or `url + "\0" + strategy`.
Only an explicitly selected non-`auto` CLI strategy adds the strategy suffix.
A strategy inherited from configuration does not count as explicit.

Use the same input and scope rules with a native extraction namespace, initially
`native-fetch-v1`, instead of the reference version `2`. A native key must not
replay Markdown produced by a different extraction engine. Bump the namespace
when parser changes make old extracted results incompatible. (Amended
2026-10-02: the namespace is `native-fetch-<package version>-r<revision>`, so every
release also starts a new namespace; the quality round after R48 had changed
extraction without bumping `v1`, and a cached X post replayed the old reader.) The original URL
is the identity; redirects, canonical links and display redaction do not replace
it. Hashing a redacted URL could merge requests with different credentials.

The CLI currently folds `-s` into configuration and loses whether the user
supplied it. Carry this provenance through an internal fetch context, separate
from public configuration defaults and the bindings JSON protocol. Deriving it
from `fetch.strategy != "auto"` would change configuration-only behavior.
In the first stage, default/configured `auto` or `static` uses the unscoped key;
explicit `-s static` uses a separate key; explicit `-s auto` stays unscoped.

## Reuse and HTTP validation

`cache.fetch_ttl_seconds` defaults to 86,400. It is a reuse window for pages
without HTTP validators, measured from `created_at`. A row is expired when
`now - created_at >= TTL`; a zero TTL always requires a fresh unvalidated fetch.
Reading a row changes `accessed_at`, not its creation time. Expired rows remain
available for statistics and eventual replacement or LRU eviction.

| Stored state | Next fetch behavior |
|---|---|
| No validator, younger than TTL | Return cached page and update access time |
| No validator, expired or TTL zero | Fetch normally and replace after successful extraction |
| ETag or Last-Modified present | Make a conditional request on every invocation, regardless of TTL |
| Conditional response 304 | Return cached page, mark a fetch hit and update access time |
| Conditional response with fresh usable content | Extract and replace; this is not a fetch hit |
| Conditional request fails | Try the normal fetch path; do not silently serve an old page |

Send the available `If-None-Match` and `If-Modified-Since` headers. Preserve the
reference static content negotiation:
`Accept: text/markdown, text/html;q=0.9, */*;q=0.5`. Handle 304 before the common
success/body reader, which currently rejects it. An unsolicited 304 without a
usable cached page cannot produce a successful empty document.

Capture validators and the effective URL before consuming the response. Decode
fresh responses through the same extraction function with or without validators,
so redirect-relative links, character encoding and metadata remain consistent.
The reference 304 path only touches access time; it does not restart the TTL or
replace the stored validators.

Fresh and revalidated content need the same admission checks. Blank output,
failed extraction and recognized HTML challenge pages must not replace a good
entry. Keep HTML challenge detection separate from literal mentions of CAPTCHA
in plain text or legitimate technical documents.

The reference has a narrower stale fallback than a general offline cache: when
a conditional 200 becomes a challenge page or JavaScript shell, it tries the
full strategy chain; if that also fails, it may return the old page with
`metadata.stale=true` and `metadata.fetch_warning`. Those run-specific fields do
not overwrite the saved entry. Policy refusals still fail. Full AUTO quality
classification and browser/remote fallback are later work; this stage must
document that difference and fail explicitly where the necessary fallback is
unavailable, while retaining the previous good cache entry.

## Controls and storage

Use the existing configuration without adding user-facing settings:

| Setting | Effect |
|---|---|
| `cache.enabled=false` | No conversion-time reads, writes or database creation |
| `cache.no_cache=true` | Bypass cache reads and conditional requests; still save fresh success |
| Matching `cache.no_cache_patterns` | The same read bypass and write refresh |
| `cache.global_dir` | Parent of `fetch_cache.db`, resolved with `config::state_path` |
| `cache.max_size_bytes` | Logical content capacity of this database; default 512 MiB |

`--cache` clears the read bypass without overriding `enabled=false`.
`--no-cache` and `--cache` retain last-occurrence precedence. The existing empty
versus whitespace-only `--no-cache-for` behavior remains unchanged.

URL patterns need a separate matcher from document paths. Match the full URL,
the URL without its scheme, the authority/hostname and the final path segment.
Scheme and hostname compare without case sensitivity; paths and filenames keep
their case. Trim patterns and support the reference's optional leading `**/`
variant. Test ports, queries and trailing slashes; do not apply the document
matcher's stage-colon stripping to a URL.

Retain the reference SQLite table shape:

```sql
CREATE TABLE fetch_cache (
    key TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    content TEXT NOT NULL,
    strategy_used TEXT NOT NULL,
    title TEXT,
    final_url TEXT,
    metadata TEXT,
    created_at INTEGER NOT NULL,
    accessed_at INTEGER NOT NULL,
    size_bytes INTEGER NOT NULL,
    etag TEXT,
    last_modified TEXT,
    screenshot_path TEXT,
    static_content TEXT,
    browser_content TEXT
);
CREATE INDEX idx_fetch_accessed ON fetch_cache(accessed_at);
CREATE INDEX idx_fetch_url ON fetch_cache(url);
```

`content` is plain Markdown; `metadata` is JSON. The first stage leaves browser
and screenshot columns empty. Existing older tables can lack validator or
multi-source columns; add missing columns during a write-capable schema setup.
Inspection should not create or migrate a database. Legacy reference rows may
remain visible to statistics and clearing, but the native namespace excludes
them from conversion reads.

Use bundled SQLite, WAL, normal synchronization and a 30-second busy timeout.
Capacity admission, replacement and eviction share one immediate transaction.
Evict by `accessed_at`, then row insertion order. An oversized replacement removes
its own stale entry but does not evict unrelated entries; failed insertion rolls
back replacement and eviction. Count UTF-8 content bytes, as the reference does.
This excludes metadata, SQLite overhead and WAL size, and each of the LLM and
fetch databases has its own capacity. It is not a total disk-size guarantee.
Bound cache row allocation as well as network response bodies.

Open the cache lazily and create it only to save an admissible result. A broken
or unwritable cache should produce a fixed, sanitized warning while preserving
a successful network result. This is an intentional correction to the reference
fetch path, where uncaught cache-operation errors can fail a successful fetch.
Concurrent misses may make duplicate network requests; request coalescing is
not required in this stage.

## Core and CLI integration

Introduce an internal fetch outcome carrying the `Document` and a boolean hit
state. Carry the latter into a serde-skipped `ConversionOutput` field with a
getter, following the existing LLM-cache integration. Do not store the hit flag
in page metadata, frontmatter or persistent rows, and do not add it to the
bindings conversion JSON. Cache warnings follow the normal warning path.

Populate the existing CLI `fetch_cache_hit` field for direct reuse and 304 reuse.
Keep `cache_hit` and `llm_cache_hit` as LLM-cache indicators. Fetching a cached
page must not claim that its subsequent LLM request was cached or free.
`fetch_strategy` remains the strategy that produced the page.

The reference has inconsistent propagation here: normal single-URL history
outcomes omit fetch/LLM hit state even though report outcomes include it; batch
URL outcomes reuse one flag for both cache categories. Preserve the public
field names and correct the meanings instead of reproducing these defects.

`cache stats --json` keeps `{cache, enabled, fetch_cache}`. A missing fetch store
returns `null` without creating it; an unreadable store returns a sanitized
`fetch_cache.error` and a nonzero CLI exit. Fetch details are `count`,
`size_bytes`, `size_mb`, `max_size_mb`, and `db_path`; expired rows count too.
The reference has no fetch-specific verbose entries or model groups.

Combined `cache clear` should clear both existing stores after the existing
confirmation, or with `-y`. Preflight both stores before deleting rows. If a
later failure still leaves only one store cleared, report the completed and
failed portions and exit nonzero; do not report complete success or promise
atomicity across two WAL databases. The reference catches clearing errors and
can still exit successfully; that behavior should not be copied. SPA-domain
requests remain unsupported and must be rejected before either store changes.
Remove the current existing-fetch-store guard only when these operations work.

## Privacy and isolation

Fetch caches are plaintext page storage. The reference stores original and
final URLs as well as metadata and content; URL query strings can contain
credentials. A hashed key does not make the database secret-free. Document this
behavior and keep cache errors and routine diagnostics free of raw URLs, SQL,
headers and page content. Any future opt-out for credentialed URLs is a separate
documented compatibility choice, not an assumption about the reference.

Default paths must honor `MARKITAI_HOME`; explicitly configured custom paths
retain their meaning. Tests must use temporary configuration and state paths,
with loopback HTTP servers and no provider credentials. Existing fetch tests
that call `config::defaults()` need explicit cache isolation before caching is
enabled. Do not read the user's cache to develop migrations or test hits.

Current local static fetching and remote extraction have different permission
rules. The first stage must not accidentally apply remote-service consent rules
to direct local fetching, or use a cached page to bypass an unsupported strategy.
Future anonymous/public-network service modes must disable trusted-session cache
reuse or establish an equivalent authority boundary; the reference disables
fetch-cache reads and writes in that mode.

## Acceptance tests and deferred work

The matrix below guided the implementation. Storage, loopback HTTP and CLI/API
checks now cover the first stage; actual commands and counts are recorded in
CONTROL. Later-stage behavior is not implied by those checks.

- Independent processes: first fetch writes; the next unvalidated fetch makes
  no request and returns the same page metadata, with fresh output paths.
- TTL boundaries, zero TTL, creation versus access time, and expired-row stats.
- ETag and Last-Modified, both conditional headers, 304 reuse, fresh 200
  replacement, redirect-relative links, and failure without a blanket stale hit.
- Disabled caching, bypass-refresh-reuse, paired CLI flags, URL pattern case and
  basename behavior, and explicit versus configured strategy scope.
- Empty output, extraction failure and challenge responses preserve the last
  good entry; plain documents discussing CAPTCHA are not rejected as challenges.
- Downloaded asset-bearing documents bypass caching without losing images.
- LRU replacement, oversized admission, rollback, concurrent writers, malformed
  rows, bounded allocations and unwritable state with visible warnings.
- Missing/legacy/corrupt databases in stats, combined-clear cancellation,
  partial-failure reporting, and unsupported SPA preflight.
- Accurate CLI fetch/LLM hit fields in single and batch modes, warning/quiet
  behavior, and an unchanged bindings JSON shape.
- Isolated default/custom state paths and no access to real user credentials.

Later stages cover remote strategy caching and policy changes, complete AUTO
quality checks and stale fallback, owned assets and binary responses, browser
multi-source content and screenshot lifetimes, SPA domains, anonymous service
authority, and request coalescing. None is implied by completing this first
stage.

## Source anchors

Reference paths below are relative to `/Users/example-user/work/markitai` at the pinned
revision; line numbers identify the audited snapshot, with symbols supplied for
later lookup. Native paths are relative to this repository.

| Contract | Reference file and anchor |
|---|---|
| Defaults and TTL meaning | `packages/markitai/src/markitai/config.py:569`, `CacheConfig` |
| URL matching and key | `packages/markitai/src/markitai/fetch_cache.py:48`, `_is_stale`; `:69`, `url_matches_cache_patterns`; `:244`, `_compute_hash` |
| Schema, rows and accounting | `packages/markitai/src/markitai/fetch_cache.py:185`, `_init_db`; `:264`, `_get_unlocked`; `:398`, `_get_with_validators_unlocked`; `:482`, `_set_with_validators_unlocked` |
| Transactional capacity | `packages/markitai/src/markitai/utils/sqlite_cache.py:27`, `cache_write` |
| Conditional requests | `packages/markitai/src/markitai/fetch.py:703`, `_resolve_cache_and_check`; `packages/markitai/src/markitai/fetch_strategies/static.py:334`, `fetch_with_static_conditional` |
| Bypass, authority, scope and writes | `packages/markitai/src/markitai/fetch.py:854`, `fetch_url`; `:893`, `cache_strategy`; `:989`, cache write |
| Explicit CLI provenance | `packages/markitai/src/markitai/cli/main.py:1044` |
| Stats and clear | `packages/markitai/src/markitai/fetch_cache.py:364`; `packages/markitai/src/markitai/cli/commands/cache.py:279` |
| Hit-state propagation defects | `packages/markitai/src/markitai/cli/processors/url.py:974`; `packages/markitai/src/markitai/cli/processors/batch.py:97` |
| Behavioral tests | `packages/markitai/tests/unit/test_fetch_cache_policy.py`; `test_fetch_cache_lru.py`; `test_fetch_cache_module.py` |

Native integration points are `crates/markitai-core/src/fetch.rs` (`fetch`,
`fetch_static`, `body`), `lib.rs` (`convert`), `types.rs` (`ConversionOutput`),
`config.rs` (`state_path`), `llm_cache.rs` (`stats`), and
`crates/markitai-cli/src/app.rs` (`outcome`, `cache_command`).
