# ADR 0003: extracted-page cache

Status: accepted.

Cache extracted Markdown and page metadata before output naming, profiles and
optional model enhancement. Replaying a page therefore uses the normal output
pipeline and does not imply a model-result cache hit. Responses with owned assets
are ineligible: storing only their Markdown would leave broken asset references.

Identity hashes the original URL, explicit strategy provenance and a versioned
native extraction namespace. Query strings and fragments remain significant;
redirects and redaction do not change identity. Extraction changes must advance
the namespace so older Markdown is not replayed as a new reader's result.

Pages without validators use a configured TTL. Pages with ETag or Last-Modified
are conditionally requested on every invocation; a 304 reuses the saved page.
Failed refreshes preserve the previous row but do not silently return stale
content. Disabled caching prevents reads and writes, while read bypass still
refreshes admissible results.

Use a separate bundled SQLite database with bounded reads and transactional LRU
eviction. Clearing multiple stores is not one atomic transaction: report any
partial failure. Cache errors must not expose content, SQL or credentials.
The database itself is plaintext and can contain credential-bearing URLs; hashed
keys provide no confidentiality or separation between callers' authority.

Current admission rules, controls and HTTP limitations are maintained in
[URL fetching](../fetch.md#stored-results-and-identity); management commands are
in [Cache](../cache.md#inspection).
