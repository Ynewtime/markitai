# Fetch Policy Engine

Markitai uses a sophisticated Fetch Policy Engine to determine the best strategy for retrieving URL content. This engine is inspired by modern web scraping architectures and is designed to be resilient, fast, and user-friendly.

## Strategy Selection Logic

The engine follows a policy-driven approach to select the order of fetching strategies:

1.  **Explicit Strategy**: If you provide an explicit strategy (e.g., `--playwright` or `--jina`), the engine will use only that strategy.
2.  **Domain Profiles**: You can configure specific settings for individual domains, such as custom selectors to wait for or extra wait times.
3.  **Adaptive Fallback**: In `auto` mode (default), the engine intelligently orders strategies based on the domain and previous success history.

### Default Order (Standard Domains)
For most websites, Markitai prioritizes speed:
`Static (HTTP) -> Playwright (Browser) -> Cloudflare -> Jina`

### SPA/Heavy-JS Order
For domains known to require JavaScript (like `x.com`, `instagram.com`, or domains that have failed static fetching before):
`Playwright (Browser) -> Cloudflare -> Jina -> Static`

## Configuration

You can tune the fetch policy in your `markitai.json`:

```json
{
  "fetch": {
    "policy": {
      "enabled": true,
      "max_strategy_hops": 4
    },
    "domain_profiles": {
      "x.com": {
        "wait_for_selector": "[data-testid=tweetText]",
        "wait_for": "domcontentloaded",
        "extra_wait_ms": 1200
      }
    },
    "playwright": {
      "session_mode": "domain_persistent",
      "session_ttl_seconds": 600
    }
  }
}
```

### Options

- `policy.enabled`: Enable or disable the intelligent strategy ordering.
- `policy.max_strategy_hops`: Maximum number of strategies to attempt before giving up.
- `domain_profiles`: A map of domain-specific overrides.
- `playwright.session_mode`: 
    - `isolated`: (Default) New browser context for every request.
    - `domain_persistent`: Reuses browser contexts for the same domain, significantly speeding up multiple requests to the same site.

## Static HTTP Adapters

The Static strategy uses **httpx** by default. For sites with TLS fingerprint detection, optionally enable **curl-cffi**:

- **httpx** (default): Built-in, fast and reliable.
- **curl-cffi** (optional): Uses `curl-impersonate` to mimic Chrome TLS/HTTP signatures. Install via `uv pip install markitai[extra-fetch]` and set `MARKITAI_STATIC_HTTP=curl_cffi`.

If curl-cffi is requested but not installed, Markitai silently falls back to httpx.

---

## Native Extraction Architecture

After HTML is fetched, Markitai runs it through the `webextract` pipeline — a four-layer architecture that converts raw HTML into clean Markdown without any external API dependency.

### The Four Layers

```
fetch          # Raw HTML from network
    ↓
resolve        # Site-specific structured extraction (resolver layer)
    ↓
render         # Semantic content rendered to canonical HTML fragment
    ↓
assess         # Quality profile gate (reject noise-polluted results)
```

### Layer 1 — Fetch

`fetch.py` retrieves the raw HTML using the strategy chain (static → playwright → cloudflare → jina). The output is an HTML string passed into `extract_web_content(html, url)`.

### Layer 2 — Resolve

`webextract/resolver.py` contains `resolve_page(html, url)`, which:

1. Looks up a site-specific extractor from the registry using the URL.
2. Calls the extractor's `resolve()` method if it exists.
3. Validates that the result is a `ResolvedPage` (not final Markdown — the pipeline always handles HTML→Markdown conversion).
4. Returns `None` if no extractor matches, falling back to the generic pipeline.

`ResolvedPage` carries `content_html`, optional `metadata_overrides`, and an optional `semantic` field holding a `ConversationThread`.

The async variant, `resolve_page_async(html, url, policy=...)`, additionally runs any registered enrichers that are permitted by the `EnrichmentPolicy`.

**Resolver ownership rule**: site-specific extraction logic lives in extractor `resolve()` methods and enrichers, not in the fetch layer or the generic pipeline.

### Layer 3 — Render

`webextract/render.py` converts a `SemanticExtraction` (containing a `ConversationThread`) into a canonical HTML fragment via `render_semantic_content()`. This intermediate HTML is then passed through the shared Markdown renderer (`render_markdown()`), so both the semantic path and the generic path produce Markdown through the same final step.

`render_semantic_content()` is the **only** place where threaded content may define its presentation structure. Downstream code must not re-implement this rendering logic.

### Layer 4 — Assess

`webextract/quality.py` exposes `assess_native_markdown(markdown, profile=...)`, which returns a `QualityAssessment(accepted, score, reasons)`. The pipeline uses `accepted` to decide whether to use the native Markdown or fall back to an external strategy.

Each profile encodes domain-specific heuristics:

| Profile | Applicable pages | Key rejection signals |
|---------|-----------------|----------------------|
| `generic_article` | Generic articles (default) | Fewer than 10 chars or 3 words of plain text |
| `social_post` | X/Twitter posts | Recommendation / trending sidebar leakage |
| `conversation_thread` | X threads, forum threads | Recommendation noise or fewer than 10 words |
| `discussion_issue` | GitHub Issues, discussions | Assignees / Labels sidebar leakage |

`score` and `reasons` are internal diagnostics. They are never written to user-facing YAML frontmatter.

### Semantic Thread Model

All site-specific extractors share the same semantic data structures from `webextract/semantics.py`, keeping the render layer independent of any particular site's HTML structure:

```python
ConversationThread(
    title="Post by @handle",
    main_item=ConversationItem(
        id="...",
        author_handle="@handle",
        text="...",
        timestamp="2025-01-01T00:00:00Z",
    ),
    items=[
        ConversationItem(id="...", parent_id="<id of main_item>", ...),
    ],
)
```

`parent_id` on a `ConversationItem` expresses the reply-to relationship. Renderers may choose to flatten the tree for Markdown readability.

The shared semantic model is also used across GitHub Issues (`github_thread.py`), Reddit posts (`reddit_post.py`), Hacker News threads (`hackernews_thread.py`), and YouTube pages (`youtube_page.py`).

### Enrichers

Enrichers are optional async components that run after the sync resolver to improve extraction quality using network sources (e.g. oEmbed APIs). They are controlled by `EnrichmentPolicy` and only reached via `resolve_page_async()`.

Design rules for enrichers:
- **Never hard-fail**: catch all exceptions internally and return `None` to signal "not better than sync result".
- **Never bypass the policy**: check `policy.allow_network` and `policy.allow_async` before making requests.
- **Name yourself**: set `enricher.name` so it is recorded in diagnostics.

```python
class MyEnricher:
    name = "my_enricher"

    async def enrich(self, url: str, sync_result: ResolvedPage | None) -> ResolvedPage | None:
        if "example.com" not in url:
            return None
        try:
            # fetch supplementary data from an API
            ...
            return ResolvedPage(content_html=..., metadata_overrides={...})
        except Exception:
            return None

    def should_run(self, url: str, policy: EnrichmentPolicy) -> bool:
        return "example.com" in url and policy.allow_network
```

---

## Frontmatter Generation and Migration

### `build_source_frontmatter()` — preferred API

`webextract.frontmatter.build_source_frontmatter(result: ExtractedWebContent) -> dict` produces the user-facing YAML frontmatter from a typed extraction result. It exports:

- Page metadata: `title`, `author`, `site`, `published`, `description`, `canonical_url`
- Extraction facts: `word_count`, `content_profile`

It deliberately excludes internal quality diagnostics (`score`, `accepted`, `reasons`).

### `coerce_source_frontmatter()` — compatibility wrapper

`webextract.coerce_source_frontmatter(metadata)` is a compatibility wrapper that accepts any metadata object (dataclass, dict, or duck-typed) and returns a serializable dict. It was the original API before `ExtractedWebContent` grew typed `info` and `quality` fields.

**Migration policy**:

1. `coerce_source_frontmatter()` remains available and functional during rollout — no callers break.
2. Internal callers (pipeline, CLI processors) migrate to `build_source_frontmatter()` one by one.
3. A deprecation warning is added only after all internal callers have migrated.
4. External callers are notified of the deprecation via the warning before removal.

New code should always call `build_source_frontmatter()` directly.
