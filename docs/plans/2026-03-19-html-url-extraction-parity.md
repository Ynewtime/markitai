# HTML/URL Extraction Parity Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Bring Markitai's native HTML/URL extraction quality close to Defuddle on high-value sites without copying Defuddle's architecture blindly or scattering site-specific logic across fetch strategies.

**Architecture:** Split the problem into four layers: `fetch` acquires page sources, `resolve` turns those sources into a unified extraction model, `render` converts canonical content into Markdown, and `assess` decides whether the result is good enough or should fall back. Site extractors and async enrichers participate only in the `resolve` layer; they never emit final Markdown directly. Threaded/conversational sites share a common semantic model so X, GitHub, Reddit, and Hacker News do not each reinvent their own HTML assembly rules.

**Tech Stack:** Python 3.13, BeautifulSoup4, MarkItDown, Playwright, dataclasses/Pydantic, pytest, fixtures derived from `~/dev/defuddle`

---

## Context Summary

The current gap with Defuddle is not mainly "Playwright vs Defuddle API". It is that Markitai still spreads extraction responsibilities across the wrong seams:

1. `fetch_playwright.py` accepts weak native output too early.
2. `webextract` site extractors only choose a root node, instead of returning a structured resolution.
3. `coerce_source_frontmatter()` conflates user-facing metadata with extraction diagnostics.
4. Threaded sites are handled as raw DOM cleanup instead of conversation reconstruction.
5. Async fallback opportunities exist, but there is no parser-level orchestration like Defuddle's `parseAsync()`.

The plan below fixes those seams first, then ports site-specific capabilities into the cleaner architecture.

## Guiding Decisions

1. **No extractor-owned Markdown**
   Extractors and enrichers may return canonical HTML fragments or semantic blocks, but final Markdown always comes from a shared renderer.

2. **Separate page semantics from extraction facts**
   `WebMetadata` stays about the page. Word count, extractor name, async fallback usage, and confidence belong in extraction info and quality assessment objects.

3. **Async enrichers are resolver-level and policy-aware**
   They are optional augmenters, not fetcher-specific hacks.

4. **Parity tests climb a ladder**
   First semantic assertions, then canonical HTML snapshots, then a small number of Markdown goldens.

5. **Conversation pages share a model**
   X, GitHub issues/PRs, Reddit posts, Hacker News threads, and future chat transcripts should reuse one thread/message abstraction.

---

### Task 1: Freeze Semantic Parity Expectations Before Changing Architecture

**Files:**
- Create: `packages/markitai/tests/fixtures/web/x_status_2030105637204676808.playwright.html`
- Create: `packages/markitai/tests/fixtures/web/x_status_2030105637204676808.expected.json`
- Create: `packages/markitai/tests/fixtures/web/github_issue_thread.expected.json`
- Create: `packages/markitai/tests/unit/webextract/test_semantic_parity.py`
- Create: `packages/markitai/tests/unit/webextract/test_thread_policy.py`

**Step 1: Write the failing tests**

```python
def test_x_status_semantics_match_expected_fixture() -> None:
    result = extract_web_content(html, x_url)
    assert result.metadata.title == "Post by @ixiaowenz"
    assert result.info.profile == "social_post"
    assert result.semantic.thread is not None
    assert result.semantic.thread.main_item.author_handle == "@ixiaowenz"
    assert "Discover more" not in result.markdown
    assert "Quote" not in result.markdown


def test_thread_policy_defaults_do_not_include_unrelated_replies() -> None:
    policy = get_thread_policy(x_url)
    assert policy.include_main_item is True
    assert policy.include_author_thread is True
    assert policy.include_third_party_replies is False
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/markitai/tests/unit/webextract/test_semantic_parity.py packages/markitai/tests/unit/webextract/test_thread_policy.py -q`

Expected: failure because the current extraction result has no semantic model, no policy object, and no strong X semantics.

**Step 3: Write minimal implementation**

Do not implement the whole extractor yet. Only add enough placeholder test scaffolding and fixture helpers so later tasks can extend behavior without rewriting test intent.

**Step 4: Run tests to verify they still fail for the right reason**

Run the same pytest command as Step 2.

Expected: failure moves from missing fixtures/helpers to missing extraction model and policy behavior.

**Step 5: Commit**

```bash
git add packages/markitai/tests/fixtures/web packages/markitai/tests/unit/webextract/test_semantic_parity.py packages/markitai/tests/unit/webextract/test_thread_policy.py
git commit -m "test: freeze semantic parity expectations for threaded pages"
```

---

### Task 2: Introduce a Unified Extraction Result Model and Frontmatter Builder

**Files:**
- Modify: `packages/markitai/src/markitai/webextract/types.py`
- Create: `packages/markitai/src/markitai/webextract/frontmatter.py`
- Modify: `packages/markitai/src/markitai/webextract/__init__.py`
- Modify: `packages/markitai/src/markitai/fetch_types.py`
- Create: `packages/markitai/tests/unit/webextract/test_extraction_types.py`
- Create: `packages/markitai/tests/unit/webextract/test_frontmatter_builder.py`

**Step 1: Write the failing tests**

```python
def test_frontmatter_builder_exports_metadata_but_not_internal_diagnostics() -> None:
    result = ExtractedWebContent(
        metadata=WebMetadata(title="Post by @ixiaowenz", author="@ixiaowenz", site="X (Twitter)"),
        info=ExtractionInfo(word_count=241, extractor_name="x_tweet"),
        quality=QualityAssessment(profile="social_post", accepted=True, score=0.95),
        ...
    )
    fm = build_source_frontmatter(result)
    assert fm["title"] == "Post by @ixiaowenz"
    assert fm["word_count"] == 241
    assert "score" not in fm
    assert "accepted" not in fm


def test_fetch_result_can_carry_resolved_page_without_losing_metadata() -> None:
    fetch_result = FetchResult(...)
    assert fetch_result.resolved_page is not None
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/markitai/tests/unit/webextract/test_extraction_types.py packages/markitai/tests/unit/webextract/test_frontmatter_builder.py -q`

Expected: failure because `ExtractedWebContent` has only `metadata` plus a flat diagnostics dict, and `FetchResult` has no first-class resolved extraction field.

**Step 3: Write minimal implementation**

Introduce typed models:

```python
@dataclass(slots=True)
class ExtractionInfo:
    extractor_name: str
    profile: str
    word_count: int
    async_enricher: str | None = None
    source_kind: str = "html"


@dataclass(slots=True)
class QualityAssessment:
    profile: str
    accepted: bool
    score: float
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ExtractedWebContent:
    clean_html: str
    markdown: str
    metadata: WebMetadata
    info: ExtractionInfo
    quality: QualityAssessment
    semantic: SemanticExtraction | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
```

Add `build_source_frontmatter(result: ExtractedWebContent) -> dict[str, Any]` in `frontmatter.py`. Keep `coerce_source_frontmatter()` as a compatibility wrapper that delegates to the new builder when given an extraction result.

**Step 4: Run tests to verify they pass**

Run the same pytest command as Step 2.

Expected: PASS

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/webextract/types.py packages/markitai/src/markitai/webextract/frontmatter.py packages/markitai/src/markitai/webextract/__init__.py packages/markitai/src/markitai/fetch_types.py packages/markitai/tests/unit/webextract/test_extraction_types.py packages/markitai/tests/unit/webextract/test_frontmatter_builder.py
git commit -m "refactor: add typed extraction result and frontmatter builder"
```

---

### Task 3: Add a Resolver Layer Above Root Selection

**Files:**
- Modify: `packages/markitai/src/markitai/webextract/extractors/base.py`
- Modify: `packages/markitai/src/markitai/webextract/extractors/registry.py`
- Create: `packages/markitai/src/markitai/webextract/resolver.py`
- Modify: `packages/markitai/src/markitai/webextract/pipeline.py`
- Create: `packages/markitai/tests/unit/webextract/test_resolver.py`

**Step 1: Write the failing tests**

```python
def test_resolver_prefers_structured_content_html_over_generic_root_selection() -> None:
    result = resolve_page(html, url, resolver=fake_structured_resolver)
    assert result.clean_html == "<article><p>Structured</p></article>"


def test_resolver_does_not_allow_extractors_to_return_final_markdown() -> None:
    with pytest.raises(TypeError):
        resolve_page(html, url, resolver=fake_markdown_returning_resolver)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/markitai/tests/unit/webextract/test_resolver.py -q`

Expected: failure because extractors currently expose only `extract_root()` and there is no resolver orchestration layer.

**Step 3: Write minimal implementation**

Define a structured resolver contract:

```python
@dataclass(slots=True)
class ResolverOutput:
    content_root: Tag | None = None
    content_html: str | None = None
    metadata_overrides: dict[str, Any] = field(default_factory=dict)
    semantic: SemanticExtraction | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
```

Rules:
- Resolver output may contain `content_root` or `content_html`, not final Markdown.
- Generic pipeline remains the fallback when resolver returns `None`.
- Metadata overrides merge over generic metadata.
- Resolver orchestration lives in `resolver.py`, not in fetchers.

**Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/markitai/tests/unit/webextract/test_resolver.py -q`

Expected: PASS

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/webextract/extractors/base.py packages/markitai/src/markitai/webextract/extractors/registry.py packages/markitai/src/markitai/webextract/resolver.py packages/markitai/src/markitai/webextract/pipeline.py packages/markitai/tests/unit/webextract/test_resolver.py
git commit -m "refactor: add resolver layer for structured site extraction"
```

---

### Task 4: Introduce a Shared Conversation/Thread Semantic Model

**Files:**
- Create: `packages/markitai/src/markitai/webextract/semantics.py`
- Create: `packages/markitai/src/markitai/webextract/render.py`
- Modify: `packages/markitai/src/markitai/webextract/types.py`
- Create: `packages/markitai/tests/unit/webextract/test_semantics.py`
- Create: `packages/markitai/tests/unit/webextract/test_render_thread.py`

**Step 1: Write the failing tests**

```python
def test_render_thread_emits_clean_main_item_then_followups() -> None:
    thread = ConversationThread(
        title="Post by @ixiaowenz",
        main_item=ConversationItem(...),
        followups=[ConversationItem(...)],
    )
    html = render_semantic_content(SemanticExtraction(thread=thread))
    assert "<article" in html
    assert "conversation-item" in html


def test_render_thread_preserves_quote_and_media_as_structured_children() -> None:
    html = render_semantic_content(semantic_with_quote_and_image)
    assert "quoted-item" in html
    assert "<img" in html
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/markitai/tests/unit/webextract/test_semantics.py packages/markitai/tests/unit/webextract/test_render_thread.py -q`

Expected: failure because Markitai has no reusable conversation/thread semantic model.

**Step 3: Write minimal implementation**

Add:
- `ConversationThread`
- `ConversationItem`
- `EmbeddedQuote`
- `MediaAttachment`
- `SemanticExtraction`

Implement `render_semantic_content()` that turns these types into canonical HTML fragments. This is the only place where threaded extractors get to define presentation structure.

**Step 4: Run tests to verify they pass**

Run the same pytest command as Step 2.

Expected: PASS

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/webextract/semantics.py packages/markitai/src/markitai/webextract/render.py packages/markitai/src/markitai/webextract/types.py packages/markitai/tests/unit/webextract/test_semantics.py packages/markitai/tests/unit/webextract/test_render_thread.py
git commit -m "feat: add shared semantic model for threaded extractions"
```

---

### Task 5: Rebuild X/Twitter Extraction on Top of the Shared Model

**Files:**
- Modify: `packages/markitai/src/markitai/webextract/extractors/x_tweet.py`
- Create: `packages/markitai/src/markitai/webextract/extractors/x_common.py`
- Modify: `packages/markitai/src/markitai/webextract/pipeline.py`
- Modify: `packages/markitai/tests/unit/webextract/test_semantic_parity.py`

**Step 1: Write the failing tests**

```python
def test_x_resolver_returns_thread_semantic_model() -> None:
    result = extract_web_content(html, x_url)
    assert result.semantic.thread is not None
    assert result.semantic.thread.main_item.author_name == "Xiaowen"
    assert result.semantic.thread.main_item.author_handle == "@ixiaowenz"


def test_x_output_excludes_recommendation_sections_and_quote_card_leakage() -> None:
    result = extract_web_content(html, x_url)
    assert "Discover more" not in result.markdown
    assert "\nQuote\n" not in result.markdown
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/markitai/tests/unit/webextract/test_semantic_parity.py -k x_ -q`

Expected: failure because the current extractor only wraps a root article and generic cleanup.

**Step 3: Write minimal implementation**

Port the useful ideas from Defuddle's X extractor, but through Markitai's shared semantics:
- identify conversation timeline and stop before recommendation sections
- extract main item and allowed followups according to thread policy
- parse author, handle, timestamp, text, media, and quoted item
- build `ConversationThread`
- render that semantic model through `render_semantic_content()`
- set metadata overrides for `title`, `author`, `site`, and `description`

Do not implement async fallback here.

**Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/markitai/tests/unit/webextract/test_semantic_parity.py -k x_ -q`

Expected: PASS

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/webextract/extractors/x_tweet.py packages/markitai/src/markitai/webextract/extractors/x_common.py packages/markitai/src/markitai/webextract/pipeline.py packages/markitai/tests/unit/webextract/test_semantic_parity.py
git commit -m "feat: rebuild x extraction on shared thread semantics"
```

---

### Task 6: Validate the Abstraction With a Second Threaded Site Before Adding Async

**Files:**
- Create: `packages/markitai/src/markitai/webextract/extractors/github_thread.py`
- Modify: `packages/markitai/src/markitai/webextract/extractors/registry.py`
- Create: `packages/markitai/tests/unit/webextract/test_github_thread.py`
- Modify: `packages/markitai/tests/unit/webextract/test_semantic_parity.py`

**Step 1: Write the failing tests**

```python
def test_github_thread_uses_shared_thread_semantics() -> None:
    result = extract_web_content(html, github_url)
    assert result.semantic.thread is not None
    assert result.metadata.site == "GitHub"
    assert "## Comments" in result.markdown


def test_github_thread_keeps_issue_body_without_sidebar_noise() -> None:
    result = extract_web_content(html, github_url)
    assert "Assignees" not in result.markdown
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/markitai/tests/unit/webextract/test_github_thread.py -q`

Expected: failure because the current GitHub handling does not reconstruct a thread model.

**Step 3: Write minimal implementation**

Implement GitHub issue/PR thread extraction using the same semantic types as X. This task proves that the abstraction is not X-specific before adding more complexity.

**Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/markitai/tests/unit/webextract/test_github_thread.py -q`

Expected: PASS

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/webextract/extractors/github_thread.py packages/markitai/src/markitai/webextract/extractors/registry.py packages/markitai/tests/unit/webextract/test_github_thread.py packages/markitai/tests/unit/webextract/test_semantic_parity.py
git commit -m "feat: validate shared thread extraction with github discussions"
```

---

### Task 7: Replace Length-Only Acceptance With Typed Quality Profiles

**Files:**
- Create: `packages/markitai/src/markitai/webextract/quality.py`
- Modify: `packages/markitai/src/markitai/webextract/__init__.py`
- Modify: `packages/markitai/src/markitai/webextract/pipeline.py`
- Modify: `packages/markitai/src/markitai/fetch.py`
- Modify: `packages/markitai/src/markitai/fetch_playwright.py`
- Create: `packages/markitai/tests/unit/webextract/test_quality_profiles.py`

**Step 1: Write the failing tests**

```python
def test_social_post_with_quote_card_leakage_fails_quality() -> None:
    assessment = assess_native_markdown(bad_x_markdown, profile="social_post")
    assert assessment.accepted is False
    assert "quote_card_leakage" in assessment.reasons


def test_clean_thread_markdown_passes_quality() -> None:
    assessment = assess_native_markdown(clean_markdown, profile="conversation_thread")
    assert assessment.accepted is True
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/markitai/tests/unit/webextract/test_quality_profiles.py -q`

Expected: failure because `is_native_markdown_acceptable()` is currently only length-based.

**Step 3: Write minimal implementation**

Create a typed `QualityAssessment` generator with profiles such as:
- `generic_article`
- `social_post`
- `conversation_thread`
- `discussion_issue`

Wire it into:
- native webextract acceptance
- `fetch_playwright.py`
- static HTML fallback decisions

Keep the first pass narrow and deterministic.

**Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/markitai/tests/unit/webextract/test_quality_profiles.py -q`

Expected: PASS

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/webextract/quality.py packages/markitai/src/markitai/webextract/__init__.py packages/markitai/src/markitai/webextract/pipeline.py packages/markitai/src/markitai/fetch.py packages/markitai/src/markitai/fetch_playwright.py packages/markitai/tests/unit/webextract/test_quality_profiles.py
git commit -m "feat: add typed native extraction quality profiles"
```

---

### Task 8: Add Policy-Aware Async Enrichers at the Resolver Layer

**Files:**
- Create: `packages/markitai/src/markitai/webextract/async_extractors/base.py`
- Create: `packages/markitai/src/markitai/webextract/async_extractors/x_oembed.py`
- Modify: `packages/markitai/src/markitai/webextract/resolver.py`
- Modify: `packages/markitai/src/markitai/webextract/extractors/registry.py`
- Modify: `packages/markitai/src/markitai/config.py`
- Create: `packages/markitai/tests/unit/webextract/test_async_resolver.py`

**Step 1: Write the failing tests**

```python
async def test_resolver_prefers_async_for_x_when_policy_allows_and_sync_quality_is_low() -> None:
    result = await resolve_page_async(x_html, x_url, policy=policy)
    assert result.info.async_enricher == "x_oembed"
    assert result.metadata.title == "Post by @ixiaowenz"


async def test_async_enricher_is_skipped_when_policy_disallows_network() -> None:
    result = await resolve_page_async(x_html, x_url, policy=local_only_policy)
    assert result.info.async_enricher is None
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/markitai/tests/unit/webextract/test_async_resolver.py -q`

Expected: failure because Markitai has no parser-level async extraction orchestration.

**Step 3: Write minimal implementation**

Add async resolver support with explicit policy checks:
- async enrichers are optional
- they never hard-fail the page
- they respect config and fetch policy
- preferred async is allowed to run before sync only when the site profile says it is strictly better

Mirror Defuddle's orchestration semantics, not its exact API surface.

**Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/markitai/tests/unit/webextract/test_async_resolver.py -q`

Expected: PASS

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/webextract/async_extractors packages/markitai/src/markitai/webextract/resolver.py packages/markitai/src/markitai/webextract/extractors/registry.py packages/markitai/src/markitai/config.py packages/markitai/tests/unit/webextract/test_async_resolver.py
git commit -m "feat: add policy-aware async enrichers to resolver pipeline"
```

---

### Task 9: Add Raw HTML Preprocess and Browser DOM Normalize as Separate Phases

**Files:**
- Create: `packages/markitai/src/markitai/webextract/preprocess.py`
- Modify: `packages/markitai/src/markitai/webextract/dom.py`
- Modify: `packages/markitai/src/markitai/fetch_playwright.py`
- Modify: `packages/markitai/src/markitai/webextract/pipeline.py`
- Create: `packages/markitai/tests/unit/webextract/test_preprocess.py`
- Create: `packages/markitai/tests/unit/test_fetch_playwright_dom_normalize.py`

**Step 1: Write the failing tests**

```python
def test_preprocess_resolves_streamed_content_placeholders() -> None:
    result = extract_web_content(streaming_html, url)
    assert "real body text" in result.markdown


async def test_playwright_normalizes_open_shadow_dom_before_extraction() -> None:
    result = await fetch_with_playwright(url, ...)
    assert "shadow text" in result.content
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/markitai/tests/unit/webextract/test_preprocess.py packages/markitai/tests/unit/test_fetch_playwright_dom_normalize.py -q`

Expected: failure because Markitai currently treats HTML preprocessing and browser DOM cleanup as the same concern.

**Step 3: Write minimal implementation**

Keep the phases separate:
- raw HTML preprocess: streamed content, declarative shadow DOM, `wbr` normalization
- browser DOM normalize: live DOM flattening and cleanup before `page.content()`

Do not implement CSS/mobile emulation heuristics.

**Step 4: Run tests to verify they pass**

Run the same pytest command as Step 2.

Expected: PASS

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/webextract/preprocess.py packages/markitai/src/markitai/webextract/dom.py packages/markitai/src/markitai/fetch_playwright.py packages/markitai/src/markitai/webextract/pipeline.py packages/markitai/tests/unit/webextract/test_preprocess.py packages/markitai/tests/unit/test_fetch_playwright_dom_normalize.py
git commit -m "feat: split raw html preprocess from browser dom normalization"
```

---

### Task 10: Add a Narrow Markdown Fidelity Layer Above the Shared Renderer

**Files:**
- Create: `packages/markitai/src/markitai/webextract/markdown.py`
- Modify: `packages/markitai/src/markitai/webextract/render.py`
- Modify: `packages/markitai/src/markitai/webextract/pipeline.py`
- Create: `packages/markitai/tests/unit/webextract/test_markdown_fidelity.py`

**Step 1: Write the failing tests**

```python
def test_figure_caption_is_preserved_under_image() -> None:
    markdown = render_markdown(canonical_html)
    assert "![Alt](https://example.com/image.jpg)" in markdown
    assert "Figure caption" in markdown


def test_embed_iframe_is_reduced_to_canonical_link() -> None:
    markdown = render_markdown(embed_html)
    assert "https://x.com/i/status/123" in markdown
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/markitai/tests/unit/webextract/test_markdown_fidelity.py -q`

Expected: failure because current HTML-to-Markdown conversion relies almost entirely on raw MarkItDown defaults.

**Step 3: Write minimal implementation**

Add only a thin fidelity layer:
- preserve captions
- choose best image URL from `srcset`
- canonicalize social/video embeds to links
- stabilize thread header rendering emitted by `render.py`

Do not rewrite Markdown rendering from scratch.

**Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/markitai/tests/unit/webextract/test_markdown_fidelity.py -q`

Expected: PASS

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/webextract/markdown.py packages/markitai/src/markitai/webextract/render.py packages/markitai/src/markitai/webextract/pipeline.py packages/markitai/tests/unit/webextract/test_markdown_fidelity.py
git commit -m "feat: add narrow markdown fidelity layer for canonical content"
```

---

### Task 11: Expand Site Coverage Only After the Shared Abstractions Hold

**Files:**
- Create: `packages/markitai/src/markitai/webextract/extractors/reddit_post.py`
- Create: `packages/markitai/src/markitai/webextract/extractors/hackernews_thread.py`
- Create: `packages/markitai/src/markitai/webextract/extractors/youtube_page.py`
- Modify: `packages/markitai/src/markitai/webextract/extractors/registry.py`
- Create: `packages/markitai/tests/unit/webextract/test_reddit_post.py`
- Create: `packages/markitai/tests/unit/webextract/test_hackernews_thread.py`
- Create: `packages/markitai/tests/unit/webextract/test_youtube_page.py`

**Step 1: Write the failing tests**

```python
def test_reddit_post_reuses_thread_semantics_for_post_and_comments() -> None:
    result = extract_web_content(html, reddit_url)
    assert result.semantic.thread is not None


def test_youtube_page_sets_video_metadata_even_without_transcript() -> None:
    result = extract_web_content(html, youtube_url)
    assert result.metadata.site == "YouTube"
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/markitai/tests/unit/webextract/test_reddit_post.py packages/markitai/tests/unit/webextract/test_hackernews_thread.py packages/markitai/tests/unit/webextract/test_youtube_page.py -q`

Expected: failure because these extractors do not yet exist on the new contract.

**Step 3: Write minimal implementation**

Add sites in this order:
1. Reddit
2. Hacker News
3. YouTube page metadata, then optional transcript variable hook later

If a site does not fit the shared thread model, stop and add a new semantic type before continuing.

**Step 4: Run tests to verify they pass**

Run the same pytest command as Step 2.

Expected: PASS

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/webextract/extractors packages/markitai/tests/unit/webextract
git commit -m "feat: expand native extraction coverage on shared abstractions"
```

---

### Task 12: Add Parity CI, Diagnostics, and Benchmark Guardrails

**Files:**
- Create: `packages/markitai/tests/integration/test_defuddle_parity.py`
- Create: `packages/markitai/tests/fixtures/web/README.md`
- Create: `packages/markitai/tests/integration/test_webextract_benchmarks.py`
- Modify: `packages/markitai/src/markitai/batch.py`
- Modify: `packages/markitai/src/markitai/fetch.py`
- Modify: `pyproject.toml`
- Modify: `docs/guide/fetch-policy.md`
- Modify: `docs/architecture.md`

**Step 1: Write the failing tests**

```python
def test_parity_fixture_matches_semantic_expectations() -> None:
    report = run_native_fixture_case("x_status_2030105637204676808")
    assert report.semantic_ok is True
    assert report.html_snapshot_ok is True


def test_native_extraction_benchmark_does_not_regress_beyond_budget() -> None:
    stats = run_fixture_benchmark("x_status_2030105637204676808")
    assert stats.total_ms < 1000
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/markitai/tests/integration/test_defuddle_parity.py packages/markitai/tests/integration/test_webextract_benchmarks.py -q`

Expected: failure because there is no fixture runner, no semantic parity report, and no benchmark guardrail.

**Step 3: Write minimal implementation**

Add:
- semantic parity runner
- canonical HTML snapshot helper
- a very small Markdown golden suite for stable cases only
- benchmark helper and budget docs
- extraction diagnostics aggregation in reports:
  - resolver chosen
  - async enricher used
  - quality rejection reasons
  - fallback path

**Step 4: Run tests to verify they pass**

Run the same pytest command as Step 2.

Expected: PASS

**Step 5: Commit**

```bash
git add packages/markitai/tests/integration/test_defuddle_parity.py packages/markitai/tests/integration/test_webextract_benchmarks.py packages/markitai/tests/fixtures/web/README.md packages/markitai/src/markitai/batch.py packages/markitai/src/markitai/fetch.py pyproject.toml docs/guide/fetch-policy.md docs/architecture.md
git commit -m "test: add parity, diagnostics, and benchmark guardrails for native extraction"
```

---

## Rollout Notes

- Ship Tasks 1-5 first. That gives Markitai a clean architecture plus the highest-value X improvement without external-network dependence.
- Task 6 is the abstraction proof point. Do not add Reddit/HN/YouTube before a second threaded site works cleanly.
- Task 7 should land before Task 8 so async enrichers are only used behind an explicit quality decision.
- Task 8 must remain optional, observable, and policy-aware.
- Tasks 9-10 strengthen the generic pipeline after the main seams are correct.
- Task 12 is mandatory before claiming parity.

## Success Criteria

- X/Twitter status pages produce clean title, author, site metadata, and stable thread behavior under an explicit policy.
- Native extraction results are typed as metadata, extraction info, quality assessment, and optional semantic content instead of one flat diagnostics blob.
- Fetch strategies no longer own site-specific extraction logic.
- Async enrichers are resolver-level, optional, and transparent in diagnostics.
- At least two threaded sites share the same semantic model successfully before extractor coverage expands.
- Parity is measured first semantically, then structurally, then textually.

## Risks

- If `WebMetadata` keeps absorbing diagnostics, the design will rot immediately.
- If extractors are allowed to emit final Markdown, renderer drift will explode across sites.
- If async enrichers are added before quality profiles, they will mask sync extraction bugs instead of surfacing them.
- If parity is defined only as exact Markdown equality, the suite will become too brittle to guide refactors.

## Recommended Execution Order

1. Task 1
2. Task 2
3. Task 3
4. Task 4
5. Task 5
6. Task 6
7. Task 7
8. Task 8
9. Task 9
10. Task 10
11. Task 11
12. Task 12

Plan complete and saved to `docs/plans/2026-03-19-html-url-extraction-parity.md`. Two execution options:

**1. Subagent-Driven (this session)** - I implement task-by-task in this session with review checkpoints.

**2. Parallel Session (separate)** - Open a fresh session and execute the plan end-to-end from the document.
