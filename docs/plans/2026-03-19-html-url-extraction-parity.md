# HTML/URL Extraction Parity Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Bring Markitai's native HTML/URL extraction quality closer to Defuddle, starting with X/Twitter and then closing broader pipeline gaps across Playwright, static HTML, and URL processing.

**Architecture:** Treat the current gap as an extraction-contract problem, not just a fetch-strategy problem. The first step is to upgrade Markitai from a root-only extractor model to a structured extraction model that can override content, metadata, and diagnostics. Then rebuild high-value site extractors such as X/Twitter on top of that contract, add optional async enrichers where DOM-only extraction is inherently insufficient, and tighten acceptance gates and regression coverage so weak native output no longer silently ships.

**Tech Stack:** Python 3.13, BeautifulSoup4, MarkItDown, Playwright, Pydantic/dataclasses, pytest, golden fixtures derived from `~/dev/defuddle`

---

## Context Summary

Defuddle is ahead of Markitai in four important ways:

1. **Extractor contract**: Defuddle extractors return structured content and metadata variables, not just a root DOM node.
2. **Async enrichers**: X/Twitter and YouTube can use async fallbacks when DOM extraction is incomplete.
3. **DOM preprocessing**: Defuddle resolves streamed content, shadow roots, and mobile styles before scoring.
4. **Regression harness**: Defuddle ships fixture and server tests for real site outputs; Markitai mostly tests extractor selection and generic quality.

Markitai has already ported much of Defuddle's removal and standardization logic, but the remaining deficit is mostly in the **edges of the pipeline**: extractor outputs, metadata overrides, markdown fidelity, fallback orchestration, and acceptance tests.

## Implementation Strategy

Work in this order:

1. Freeze current behavior with parity tests.
2. Introduce a structured extraction contract without breaking current generic extraction.
3. Rebuild X/Twitter on the new contract.
4. Add optional async enrichers for cases where DOM-only extraction is not enough.
5. Backfill generic pipeline features Defuddle already has and Markitai still lacks.
6. Expand extractor coverage and acceptance gates.

---

### Task 1: Build a Defuddle-Parity Regression Harness

**Files:**
- Create: `packages/markitai/tests/fixtures/web/x_status_2030105637204676808.playwright.html`
- Create: `packages/markitai/tests/fixtures/web/x_status_2030105637204676808.expected.md`
- Create: `packages/markitai/tests/unit/webextract/test_x_tweet_parity.py`
- Create: `packages/markitai/tests/unit/webextract/test_source_frontmatter_parity.py`
- Modify: `packages/markitai/tests/unit/test_fetch_playwright.py`

**Step 1: Write the failing tests**

```python
def test_x_tweet_native_extraction_matches_expected_fixture() -> None:
    html = fixture_path.read_text(encoding="utf-8")
    result = extract_web_content(html, "https://x.com/ixiaowenz/status/2030105637204676808")
    assert result.markdown.strip() == expected_markdown.strip()


def test_x_tweet_frontmatter_prefers_extractor_metadata() -> None:
    html = fixture_path.read_text(encoding="utf-8")
    result = extract_web_content(html, "https://x.com/ixiaowenz/status/2030105637204676808")
    fm = coerce_source_frontmatter(result.metadata)
    assert fm["title"] == "Post by @ixiaowenz"
    assert fm["author"] == "@ixiaowenz"
    assert fm["site"] == "X (Twitter)"
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/markitai/tests/unit/webextract/test_x_tweet_parity.py -q`

Expected: failure due to current generic metadata and quote/media leakage.

**Step 3: Add one fetch-layer acceptance test**

```python
def test_playwright_rejects_native_x_output_when_quality_is_tweet_incomplete() -> None:
    assert _is_content_incomplete(bad_tweet_markdown) is True
```

**Step 4: Run test to verify it fails**

Run: `uv run pytest packages/markitai/tests/unit/test_fetch_playwright.py -k x_output -q`

Expected: failure because current acceptance gate is too weak.

**Step 5: Commit**

```bash
git add packages/markitai/tests/fixtures/web packages/markitai/tests/unit/webextract packages/markitai/tests/unit/test_fetch_playwright.py
git commit -m "test: add native extraction parity fixtures for x status pages"
```

---

### Task 2: Introduce a Structured Site Extractor Contract

**Files:**
- Modify: `packages/markitai/src/markitai/webextract/extractors/base.py`
- Modify: `packages/markitai/src/markitai/webextract/types.py`
- Modify: `packages/markitai/src/markitai/webextract/pipeline.py`
- Modify: `packages/markitai/src/markitai/webextract/__init__.py`
- Test: `packages/markitai/tests/unit/webextract/test_source_frontmatter_parity.py`

**Step 1: Write the failing tests**

```python
def test_pipeline_uses_structured_extractor_content_over_root_selection() -> None:
    extractor = FakeStructuredExtractor(...)
    result = _extract_with_extractor(html, url, extractor)
    assert result.markdown == "expected structured markdown"


def test_source_frontmatter_includes_extractor_overrides() -> None:
    metadata = WebMetadata(title="generic")
    extraction = ExtractedWebContent(..., metadata=metadata, extractor_metadata={"title": "override"})
    assert coerce_source_frontmatter(extraction.metadata)["title"] == "override"
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/markitai/tests/unit/webextract/test_source_frontmatter_parity.py -q`

Expected: failure because the current extractor protocol only supports `extract_root()`.

**Step 3: Write minimal implementation**

Add a structured result type:

```python
@dataclass(slots=True)
class SiteExtractionResult:
    root: Tag | None = None
    clean_html: str | None = None
    markdown: str | None = None
    metadata_overrides: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
```

Update the extractor protocol to support:

```python
def extract(self, soup: BeautifulSoup, url: str) -> SiteExtractionResult | None: ...
```

Pipeline rules:
- If extractor returns `markdown`, skip generic HTML→Markdown.
- If extractor returns `clean_html`, skip root scoring/removals for that page.
- Merge `metadata_overrides` over generic metadata.
- Carry extractor diagnostics into `ExtractedWebContent.diagnostics`.

**Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/markitai/tests/unit/webextract/test_source_frontmatter_parity.py -q`

Expected: PASS

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/webextract/extractors/base.py packages/markitai/src/markitai/webextract/types.py packages/markitai/src/markitai/webextract/pipeline.py packages/markitai/src/markitai/webextract/__init__.py packages/markitai/tests/unit/webextract/test_source_frontmatter_parity.py
git commit -m "refactor: add structured webextract site extractor contract"
```

---

### Task 3: Rebuild the X/Twitter Extractor on the New Contract

**Files:**
- Modify: `packages/markitai/src/markitai/webextract/extractors/x_tweet.py`
- Create: `packages/markitai/src/markitai/webextract/extractors/x_common.py`
- Test: `packages/markitai/tests/unit/webextract/test_x_tweet_parity.py`

**Step 1: Write the failing tests**

```python
def test_x_tweet_extractor_only_keeps_main_tweet_and_allowed_thread() -> None:
    result = extract_web_content(html, x_url)
    assert "Discover more" not in result.markdown
    assert "Quote" not in result.markdown


def test_x_tweet_extractor_outputs_clean_author_header() -> None:
    result = extract_web_content(html, x_url)
    assert result.markdown.startswith("**Xiaowen** @ixiaowenz")
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/markitai/tests/unit/webextract/test_x_tweet_parity.py -q`

Expected: failure due to current root-wrapper approach.

**Step 3: Write minimal implementation**

Port the key structure from Defuddle's `twitter.ts`, but adapt to Markitai:
- identify `main_tweet` and `thread_tweets`
- stop before "Discover more" / post-recommendation sections
- extract user full name, handle, timestamp, permalink, tweet text, media, and quoted tweet separately
- render a clean extractor-owned HTML fragment
- set metadata overrides:

```python
{
    "title": f"Post by {handle}",
    "author": handle,
    "site": "X (Twitter)",
    "description": truncated_main_tweet_text,
}
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/markitai/tests/unit/webextract/test_x_tweet_parity.py -q`

Expected: PASS

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/webextract/extractors/x_tweet.py packages/markitai/src/markitai/webextract/extractors/x_common.py packages/markitai/tests/unit/webextract/test_x_tweet_parity.py
git commit -m "feat: rebuild x tweet extraction with structured site output"
```

---

### Task 4: Add Optional Async Site Enrichers for DOM-Insufficient Pages

**Files:**
- Create: `packages/markitai/src/markitai/webextract/async_extractors/base.py`
- Create: `packages/markitai/src/markitai/webextract/async_extractors/x_oembed.py`
- Modify: `packages/markitai/src/markitai/webextract/extractors/registry.py`
- Modify: `packages/markitai/src/markitai/fetch_playwright.py`
- Modify: `packages/markitai/src/markitai/fetch.py`
- Modify: `packages/markitai/src/markitai/config.py`
- Test: `packages/markitai/tests/unit/test_fetch_playwright.py`
- Test: `packages/markitai/tests/unit/webextract/test_x_async_fallback.py`

**Step 1: Write the failing tests**

```python
async def test_playwright_uses_x_oembed_enricher_when_native_tweet_quality_is_low() -> None:
    result = await fetch_with_playwright(x_url, ...)
    assert result.metadata["source_frontmatter"]["title"] == "Post by @ixiaowenz"


async def test_x_oembed_enricher_is_optional_and_gracefully_skips_on_http_failure() -> None:
    result = await enrich_x_status(...)
    assert result is None
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/markitai/tests/unit/webextract/test_x_async_fallback.py packages/markitai/tests/unit/test_fetch_playwright.py -q`

Expected: failure because Markitai has no async extractor layer.

**Step 3: Write minimal implementation**

Create an optional async extractor contract:

```python
class AsyncSiteExtractor(Protocol):
    def matches_url(self, url: str) -> bool: ...
    async def extract(self, url: str, html: str | None = None) -> SiteExtractionResult | None: ...
```

Implement `x_oembed.py` with this order:
- try FxTwitter-compatible API if configured
- fall back to `publish.twitter.com/oembed`
- never hard-fail the fetch strategy; just return `None`

Config flags:
- `fetch.async_extractors.enabled`
- `fetch.async_extractors.x_oembed.enabled`
- `fetch.async_extractors.x_oembed.timeout`

Use this only when:
- URL matches X/Twitter status
- native Playwright/static extraction passes "page loaded" checks but fails "tweet quality" checks

**Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/markitai/tests/unit/webextract/test_x_async_fallback.py packages/markitai/tests/unit/test_fetch_playwright.py -q`

Expected: PASS

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/webextract/async_extractors packages/markitai/src/markitai/fetch_playwright.py packages/markitai/src/markitai/fetch.py packages/markitai/src/markitai/config.py packages/markitai/tests/unit/webextract/test_x_async_fallback.py packages/markitai/tests/unit/test_fetch_playwright.py
git commit -m "feat: add optional async site enrichers for x status extraction"
```

---

### Task 5: Add DOM Preprocessing Missing from Markitai's Native Pipeline

**Files:**
- Create: `packages/markitai/src/markitai/webextract/preprocess.py`
- Modify: `packages/markitai/src/markitai/webextract/dom.py`
- Modify: `packages/markitai/src/markitai/webextract/pipeline.py`
- Test: `packages/markitai/tests/unit/webextract/test_preprocess.py`

**Step 1: Write the failing tests**

```python
def test_preprocess_resolves_streamed_content_placeholders() -> None:
    result = extract_web_content(streaming_html, url)
    assert "real body text" in result.markdown


def test_preprocess_flattens_declarative_shadow_content() -> None:
    result = extract_web_content(shadow_html, url)
    assert "shadow text" in result.markdown
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/markitai/tests/unit/webextract/test_preprocess.py -q`

Expected: failure because `parse_html()` only builds BeautifulSoup.

**Step 3: Write minimal implementation**

Add preprocessing functions inspired by Defuddle:
- `resolve_streamed_content(html: str) -> str`
- `flatten_declarative_shadow_dom(html: str) -> str`
- `normalize_wbr(html: str) -> str`
- `apply_mobile_style_hints(html: str) -> str` (limited, heuristic-only; no browser CSSOM emulation)

Rules:
- keep this deterministic and dependency-free
- preprocess raw HTML before `BeautifulSoup`
- log diagnostics when a transform changes the document

**Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/markitai/tests/unit/webextract/test_preprocess.py -q`

Expected: PASS

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/webextract/preprocess.py packages/markitai/src/markitai/webextract/dom.py packages/markitai/src/markitai/webextract/pipeline.py packages/markitai/tests/unit/webextract/test_preprocess.py
git commit -m "feat: add native html preprocessing for streamed and shadow content"
```

---

### Task 6: Improve Metadata and Frontmatter Parity

**Files:**
- Modify: `packages/markitai/src/markitai/webextract/types.py`
- Modify: `packages/markitai/src/markitai/webextract/metadata.py`
- Modify: `packages/markitai/src/markitai/webextract/__init__.py`
- Modify: `packages/markitai/src/markitai/cli/processors/url.py`
- Test: `packages/markitai/tests/unit/webextract/test_metadata.py`

**Step 1: Write the failing tests**

```python
def test_frontmatter_includes_word_count_domain_and_extractor_type() -> None:
    result = extract_web_content(html, url)
    fm = coerce_source_frontmatter(result.metadata)
    assert fm["word_count"] == 241
    assert fm["domain"] == "x.com"
    assert fm["extractor_type"] == "x_tweet"
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/markitai/tests/unit/webextract/test_metadata.py -q`

Expected: failure because `WebMetadata` is too small.

**Step 3: Write minimal implementation**

Extend `WebMetadata` to include:
- `domain`
- `language`
- `word_count`
- `extractor_type`
- `schema_org_type`
- `image`
- `favicon`

Rules:
- generic metadata extractor fills what it can
- structured extractors can override
- `coerce_source_frontmatter()` exports only stable user-facing fields

**Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/markitai/tests/unit/webextract/test_metadata.py -q`

Expected: PASS

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/webextract/types.py packages/markitai/src/markitai/webextract/metadata.py packages/markitai/src/markitai/webextract/__init__.py packages/markitai/src/markitai/cli/processors/url.py packages/markitai/tests/unit/webextract/test_metadata.py
git commit -m "feat: expand native extraction metadata and frontmatter parity"
```

---

### Task 7: Add a Native Markdown Fidelity Layer Above MarkItDown

**Files:**
- Create: `packages/markitai/src/markitai/webextract/markdown.py`
- Modify: `packages/markitai/src/markitai/webextract/pipeline.py`
- Modify: `packages/markitai/src/markitai/webextract/standardize.py`
- Test: `packages/markitai/tests/unit/webextract/test_markdown_fidelity.py`

**Step 1: Write the failing tests**

```python
def test_embed_iframe_to_x_status_markdown_reference() -> None:
    markdown = html_fragment_to_markdown(embed_html)
    assert "![](https://x.com/i/status/123)" in markdown


def test_figure_caption_is_preserved_under_image() -> None:
    markdown = html_fragment_to_markdown(figure_html)
    assert "![Alt](https://example.com/image.jpg)" in markdown
    assert "Figure caption" in markdown
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/markitai/tests/unit/webextract/test_markdown_fidelity.py -q`

Expected: failure because generic `MarkItDown` loses site-specific semantics.

**Step 3: Write minimal implementation**

Wrap MarkItDown output with focused post-processing:
- normalize embedded X/Twitter/Youtube iframes into canonical links
- preserve figure captions
- choose best image URL from `srcset`/`picture`
- keep callout and footnote markup stable
- collapse tweet-specific header wrappers into clean text blocks

Keep this layer narrow; do not reimplement a full markdown engine.

**Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/markitai/tests/unit/webextract/test_markdown_fidelity.py -q`

Expected: PASS

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/webextract/markdown.py packages/markitai/src/markitai/webextract/pipeline.py packages/markitai/src/markitai/webextract/standardize.py packages/markitai/tests/unit/webextract/test_markdown_fidelity.py
git commit -m "feat: add native markdown fidelity layer for structured html"
```

---

### Task 8: Replace the Minimal Native Acceptance Gate with Quality Profiles

**Files:**
- Create: `packages/markitai/src/markitai/webextract/quality.py`
- Modify: `packages/markitai/src/markitai/webextract/__init__.py`
- Modify: `packages/markitai/src/markitai/fetch.py`
- Modify: `packages/markitai/src/markitai/fetch_playwright.py`
- Test: `packages/markitai/tests/unit/webextract/test_quality_profiles.py`

**Step 1: Write the failing tests**

```python
def test_x_status_markdown_with_quote_card_fails_quality_profile() -> None:
    assert is_native_markdown_acceptable(bad_x_markdown, url=x_url) is False


def test_article_markdown_with_clean_body_passes_quality_profile() -> None:
    assert is_native_markdown_acceptable(clean_article_markdown, url=article_url) is True
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/markitai/tests/unit/webextract/test_quality_profiles.py -q`

Expected: failure because the current gate is length-only.

**Step 3: Write minimal implementation**

Introduce domain-aware quality profiles:
- generic article
- social post
- issue/discussion
- conversation/transcript

Signal examples:
- title length and cleanliness
- duplicate author/header ratio
- quote-card leakage
- CTA/login/promo density
- markdown/table/image balance
- extractor diagnostics confidence

Use these profiles in:
- `fetch_playwright.py`
- native static HTML fetch path
- strategy fallback decisions

**Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/markitai/tests/unit/webextract/test_quality_profiles.py -q`

Expected: PASS

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/webextract/quality.py packages/markitai/src/markitai/webextract/__init__.py packages/markitai/src/markitai/fetch.py packages/markitai/src/markitai/fetch_playwright.py packages/markitai/tests/unit/webextract/test_quality_profiles.py
git commit -m "feat: add domain-aware native extraction quality profiles"
```

---

### Task 9: Expand Extractor Coverage Based on Defuddle Priority

**Files:**
- Modify: `packages/markitai/src/markitai/webextract/extractors/registry.py`
- Create: `packages/markitai/src/markitai/webextract/extractors/github_thread.py`
- Create: `packages/markitai/src/markitai/webextract/extractors/reddit_post.py`
- Create: `packages/markitai/src/markitai/webextract/extractors/hackernews_thread.py`
- Create: `packages/markitai/src/markitai/webextract/extractors/youtube_page.py`
- Test: `packages/markitai/tests/unit/webextract/test_registry.py`
- Test: `packages/markitai/tests/unit/webextract/test_github_thread.py`
- Test: `packages/markitai/tests/unit/webextract/test_reddit_post.py`

**Step 1: Write the failing tests**

```python
def test_registry_uses_reddit_post_extractor() -> None:
    assert find_extractor(reddit_url).name == "reddit_post"


def test_github_thread_extractor_keeps_issue_body_and_comments() -> None:
    result = extract_web_content(html, github_url)
    assert "## Comments" in result.markdown
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/markitai/tests/unit/webextract/test_registry.py packages/markitai/tests/unit/webextract/test_github_thread.py packages/markitai/tests/unit/webextract/test_reddit_post.py -q`

Expected: failure because these extractors do not exist.

**Step 3: Write minimal implementation**

Priority order:
1. GitHub issue/PR thread parity
2. Reddit post + extractor-owned comments
3. Hacker News comments
4. YouTube page metadata plus optional transcript variable hook

Do not build all async fallbacks at once; keep extractor coverage incremental.

**Step 4: Run tests to verify they pass**

Run the same pytest command as Step 2.

Expected: PASS

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/webextract/extractors packages/markitai/tests/unit/webextract
git commit -m "feat: expand native extractor coverage for discussions and threads"
```

---

### Task 10: Establish Fixture-Based Parity CI and Rollout Guardrails

**Files:**
- Create: `packages/markitai/tests/integration/test_defuddle_parity.py`
- Create: `packages/markitai/tests/fixtures/web/README.md`
- Modify: `pyproject.toml`
- Modify: `docs/guide/fetch-policy.md`
- Modify: `docs/architecture.md`

**Step 1: Write the failing tests**

```python
@pytest.mark.integration
def test_x_status_fixture_has_no_regression() -> None:
    assert run_native_fixture_case("x_status_2030105637204676808") == expected_output
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/markitai/tests/integration/test_defuddle_parity.py -q`

Expected: failure until fixture harness and outputs are wired in.

**Step 3: Write minimal implementation**

Add:
- fixture loader
- normalized markdown diff helper
- optional snapshot refresh script
- CI target for a small parity suite
- docs describing what "native parity" means and how to update fixtures

**Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/markitai/tests/integration/test_defuddle_parity.py -q`

Expected: PASS

**Step 5: Commit**

```bash
git add packages/markitai/tests/integration/test_defuddle_parity.py packages/markitai/tests/fixtures/web/README.md pyproject.toml docs/guide/fetch-policy.md docs/architecture.md
git commit -m "test: add fixture-based parity regression suite for native extraction"
```

---

## Rollout Notes

- Ship Tasks 1-3 first. That closes the most visible X/Twitter quality gap without introducing new network dependencies.
- Task 4 should be behind config and optional at runtime. It must degrade gracefully and never block `--playwright`.
- Tasks 5-8 strengthen the generic pipeline for all HTML/URL strategies, not just Playwright.
- Task 9 should be driven by fixture evidence, not by broad extractor porting for its own sake.
- Task 10 is mandatory before declaring "native parity" achieved.

## Success Criteria

- The X/Twitter example in the user report produces:
  - clean `Post by @handle`-style title
  - clean author/site metadata
  - no trailing quote-card leakage
  - stable thread/reply behavior by config
- Playwright/static native extraction can reject weak native output and fall back deterministically.
- `coerce_source_frontmatter()` preserves native extractor metadata at parity with Defuddle's useful top-level fields.
- Markitai has fixture-based regression coverage for at least X, GitHub thread, Reddit post, and one generic article.

## Risks

- Over-generalizing extractor contracts can bloat the pipeline. Keep the structured extractor API narrow.
- Async enrichers can create surprising network behavior. Keep them explicit, optional, and observable in diagnostics.
- Trying to match Defuddle's entire markdown engine in one pass is a trap. Prefer a thin fidelity layer over a rewrite.

## Recommended Execution Order

1. Task 1
2. Task 2
3. Task 3
4. Task 8
5. Task 4
6. Task 5
7. Task 6
8. Task 7
9. Task 9
10. Task 10

Plan complete and saved to `docs/plans/2026-03-19-html-url-extraction-parity.md`. Two execution options:

**1. Subagent-Driven (this session)** - I implement task-by-task in this session with review checkpoints.

**2. Parallel Session (separate)** - Open a fresh session and execute the plan end-to-end from the document.
