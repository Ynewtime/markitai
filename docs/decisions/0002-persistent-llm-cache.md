# Persistent document LLM cache

Historical design and round-four validation record. Contract reviewed against
reference `ba374322f884b0e720b45466cc1196f4574a3da5`. Current cache behavior is
documented in [cache](../cache.md).

Repeated document enhancement should reuse a successful result across processes
before spending a model request. The initial implementation covers non-pure
local documents; pure enhancement explicitly bypasses caching. Image, URL,
process-memory and fetch caches require their own contracts.

## Compatibility requirements

- The public cache default is enabled, with a 512 MiB maximum and a `cache.db`
  database in the configured global directory. Resolve the default through the
  native state-path function so an isolated `MARKITAI_HOME` remains effective.
- Persistent LLM entries have no TTL; the existing fetch TTL is unrelated.
- `no_cache` and matching `no_cache_patterns` bypass reads while still refreshing
  successful results. `--cache` clears `no_cache`; it does not force `enabled`.
- The key covers complete document content, resolved prompt templates and rules,
  and a deduplicated, sorted set of positive-weight model names. Credentials,
  endpoints, deployment order and ordinary local filenames do not change that
  model scope. Runtime timestamps must not invalidate otherwise identical work.
- Query configured-model cache entries before creating an HTTP client or requiring
  credentials. A hit consumes no request budget and returns zero new model usage.
- Never cache failures, empty answers or truncated responses. Cache damage or
  unavailability must not discard an otherwise successful conversion.

Use a native cache-version namespace to avoid reading incompatible legacy
structured responses as Markdown. Bundled SQLite is the selected storage engine:
WAL and transactional admission/eviction provide a bounded shared cache while
keeping the CLI independent of a system SQLite installation. Record total binary growth with release validation; isolating SQLite
from other added code requires a separate controlled build comparison. A single oversized result
must be rejected without evicting existing useful entries.

The reference processor currently omits some public enabled/size wiring, and its
memory lookup can precede the persistent bypass check. These are observed
implementation defects, not intended behavior to reproduce. Respect the public
configuration and document these corrections.

## Integration and validation

An internal enhancement outcome carries cache-hit state into the existing
CLI `cache_hit` and `llm_cache_hit` fields without adding fields to the bindings'
JSON contract. CLI bypass controls, statistics and clearing use the same persistent store;
unsupported guards remain for the other cache categories.

The implementation uses `rusqlite 0.40.2` with default features disabled and its
`bundled` feature, pinned with `libsqlite3-sys 0.38.2` in Cargo.lock. See the
[upstream SQLite binding documentation](https://docs.rs/rusqlite/0.40.2/rusqlite/).
The runtime contract and remaining cache categories are described in
[cache.md](../cache.md). Existing URL-fetch stores and SPA-domain clearing still
receive explicit errors before a combined clear changes any entries.

Local mock-server and storage checks demonstrate cross-process reuse with zero second
request; no new usage on a hit; bypass-refresh-reuse; no database when disabled;
content/prompt/model invalidation and filename-independent hits; concurrent
writes; capacity eviction; and successful conversion with a damaged or unwritable
cache. All state stays in test-owned directories.

Cost budgets follow separately. Accurate provider or versioned-price accounting,
including usage from attempts preceding terminal failure, is needed before the
current cost-budget rejection can be removed. Caching does not establish pricing
or authentication compatibility.
