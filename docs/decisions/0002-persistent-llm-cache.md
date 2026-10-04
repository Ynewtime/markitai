# ADR 0002: persistent model-result cache

Status: accepted.

Repeated enhancement should reuse valid results across processes before spending
another model request. Store typed answers in bundled SQLite, with WAL and
transactional admission/eviction, so the CLI needs no system SQLite installation.
Text and visual answers use separate versioned namespaces; legacy values cannot
silently satisfy a newer response contract.

Cache identity covers content, resolved prompts and the eligible model pool.
Runtime timestamps and ordinary filename changes do not invalidate an answer.
Persistent identity excludes credentials and endpoints; active request sharing
uses a separate runtime-local identity that includes routing and credentials.

A hit carries no new model usage. Invalid, failed and truncated answers are not
admitted. Cache failures warn without discarding an otherwise successful
conversion. Read bypass still refreshes successful answers; disabling the cache
prevents both reads and writes. Capacity is a logical payload limit, not a bound
on SQLite's total disk usage.

Current settings, supported modes, privacy and failure behavior are maintained in
[Cache](../cache.md). Fetch caching is a [separate decision](0003-persistent-fetch-cache.md).
