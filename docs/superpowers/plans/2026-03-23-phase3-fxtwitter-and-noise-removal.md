# Phase 3: FxTwitter API + Content Pattern Removal + Site DOM Cleanup

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Solve the Twitter/X quality problem with FxTwitter API integration, and improve general extraction quality by porting defuddle's P0 content pattern removal and Twitter-specific DOM cleanup selectors.

**Architecture:** FxTwitter is an x.com-specific fetch enrichment inside `_dispatch_strategy()`, not a global fetch strategy. It returns structured data converted to `ConversationThread` semantics, reusing `render_semantic_content()`. Content pattern removal adds hero header and trailing thin section detection. Site DOM cleanup adds Twitter-specific selectors to Playwright's noise removal.

**Tech Stack:** Python, httpx (async HTTP), pytest, BeautifulSoup

**Spec:** `docs/superpowers/specs/2026-03-23-webextract-quality-speed-optimization-design.md` (Modules 2.1 P0, 2.4, 3.1)

**Scope note:** CSS @media mobile styles (Module 2.2) and React SSR (Module 2.3) are deferred to Phase 4 — they require new dependencies (tinycss2) and more complex integration.

---

## File Structure

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `packages/markitai/src/markitai/fetch_fxtwitter.py` | FxTwitter API client + response parsing + ConversationThread building |
| Modify | `packages/markitai/src/markitai/fetch.py:1743-1775` | Insert FxTwitter pre-playwright attempt in `_dispatch_strategy()` |
| Modify | `packages/markitai/src/markitai/webextract/removals/content_patterns.py` | Add hero header + trailing thin section removal |
| Modify | `packages/markitai/src/markitai/constants.py:172-194` | Add `SITE_NOISE_SELECTORS` dict |
| Modify | `packages/markitai/src/markitai/fetch_playwright.py` | Inject site-specific noise selectors in DOM cleanup |
| Create | `packages/markitai/tests/unit/test_fetch_fxtwitter.py` | FxTwitter tests with mocked API responses |
| Modify | `packages/markitai/tests/unit/webextract/test_content_patterns.py` | Tests for new content patterns |

---

### Task 1: Create FxTwitter API client

**Files:**
- Create: `packages/markitai/src/markitai/fetch_fxtwitter.py`
- Create: `packages/markitai/tests/unit/test_fetch_fxtwitter.py`

- [ ] **Step 1: Write failing tests**

Create `packages/markitai/tests/unit/test_fetch_fxtwitter.py`:

```python
"""Tests for FxTwitter API fetch enrichment."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from markitai.fetch_fxtwitter import (
    _build_conversation_thread,
    _extract_twitter_url_parts,
    fetch_with_fxtwitter,
)


class TestExtractTwitterUrlParts:
    def test_x_com_status_url(self) -> None:
        user, tweet_id = _extract_twitter_url_parts(
            "https://x.com/zty0826/status/2035899567837978794"
        )
        assert user == "zty0826"
        assert tweet_id == "2035899567837978794"

    def test_twitter_com_status_url(self) -> None:
        user, tweet_id = _extract_twitter_url_parts(
            "https://twitter.com/elonmusk/status/123456789"
        )
        assert user == "elonmusk"
        assert tweet_id == "123456789"

    def test_non_status_url_returns_none(self) -> None:
        result = _extract_twitter_url_parts("https://x.com/zty0826")
        assert result is None

    def test_non_twitter_url_returns_none(self) -> None:
        result = _extract_twitter_url_parts("https://github.com/user/repo")
        assert result is None


MOCK_FXTWITTER_RESPONSE = {
    "code": 200,
    "tweet": {
        "text": "This is a test tweet with some content.",
        "author": {
            "name": "Test User",
            "screen_name": "testuser",
        },
        "created_at": "2026-03-23T10:00:00.000Z",
        "media": {
            "all": [
                {
                    "type": "photo",
                    "url": "https://pbs.twimg.com/media/test.jpg",
                    "width": 800,
                    "height": 600,
                }
            ]
        },
    },
}


class TestBuildConversationThread:
    def test_basic_tweet(self) -> None:
        thread = _build_conversation_thread(
            MOCK_FXTWITTER_RESPONSE["tweet"],
            tweet_id="123",
            url="https://x.com/testuser/status/123",
        )
        assert thread.main_item.author_name == "Test User"
        assert thread.main_item.author_handle == "@testuser"
        assert "test tweet" in thread.main_item.text
        assert thread.main_item.id == "123"
        assert len(thread.main_item.media) == 1
        assert thread.main_item.media[0].url == "https://pbs.twimg.com/media/test.jpg"

    def test_tweet_with_quoted_tweet(self) -> None:
        data = {
            **MOCK_FXTWITTER_RESPONSE["tweet"],
            "quote": {
                "text": "Original quoted tweet.",
                "author": {
                    "name": "Quoted Author",
                    "screen_name": "quoteduser",
                },
            },
        }
        thread = _build_conversation_thread(
            data, tweet_id="456", url="https://x.com/testuser/status/456"
        )
        assert thread.main_item.quoted_item is not None
        assert thread.main_item.quoted_item.author_handle == "@quoteduser"
        assert "Original quoted tweet" in thread.main_item.quoted_item.text


@pytest.mark.asyncio
class TestFetchWithFxtwitter:
    async def test_successful_fetch(self) -> None:
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_FXTWITTER_RESPONSE
        mock_response.raise_for_status = lambda: None

        with patch("markitai.fetch_fxtwitter.httpx.AsyncClient") as MockClient:
            client_instance = AsyncMock()
            client_instance.get.return_value = mock_response
            client_instance.__aenter__ = AsyncMock(return_value=client_instance)
            client_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = client_instance

            result = await fetch_with_fxtwitter(
                "https://x.com/testuser/status/123"
            )
            assert result is not None
            assert result.strategy_used == "fxtwitter"
            assert "Test User" in result.content
            assert "test tweet" in result.content

    async def test_non_twitter_url_returns_none(self) -> None:
        result = await fetch_with_fxtwitter("https://github.com/user/repo")
        assert result is None

    async def test_api_failure_returns_none(self) -> None:
        with patch("markitai.fetch_fxtwitter.httpx.AsyncClient") as MockClient:
            client_instance = AsyncMock()
            client_instance.get.side_effect = Exception("Connection timeout")
            client_instance.__aenter__ = AsyncMock(return_value=client_instance)
            client_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = client_instance

            result = await fetch_with_fxtwitter(
                "https://x.com/testuser/status/123"
            )
            assert result is None
```

- [ ] **Step 2: Run tests to verify ImportError**

```bash
cd packages/markitai && uv run pytest tests/unit/test_fetch_fxtwitter.py -v
```

- [ ] **Step 3: Implement `fetch_fxtwitter.py`**

Create `packages/markitai/src/markitai/fetch_fxtwitter.py`:

```python
"""FxTwitter API fetch enrichment for X/Twitter status URLs.

Provides a pre-playwright attempt for x.com and twitter.com tweet pages
using the FxTwitter community API. Returns ``None`` on any failure to
allow transparent fallback to Playwright.
"""
from __future__ import annotations

import re
from typing import Any

import httpx
from loguru import logger

from markitai.fetch_types import FetchResult
from markitai.webextract.render import render_semantic_content
from markitai.webextract.semantics import (
    ConversationItem,
    ConversationThread,
    EmbeddedQuote,
    MediaAttachment,
)
from markitai.webextract.types import SemanticExtraction

_TWITTER_STATUS_RE = re.compile(
    r"https?://(?:x|twitter)\.com/([a-zA-Z0-9_]{1,15})/status/(\d+)"
)

_FXTWITTER_API_BASE = "https://api.fxtwitter.com"
_FXTWITTER_TIMEOUT = 10  # seconds
_USER_AGENT = "Mozilla/5.0 (compatible; MarkitAI/1.0)"


def _extract_twitter_url_parts(url: str) -> tuple[str, str] | None:
    """Extract (username, tweet_id) from a Twitter/X status URL.

    Returns ``None`` if the URL is not a valid tweet URL.
    """
    m = _TWITTER_STATUS_RE.search(url)
    if m:
        return m.group(1), m.group(2)
    return None


def _build_conversation_thread(
    tweet_data: dict[str, Any],
    *,
    tweet_id: str,
    url: str,
) -> ConversationThread:
    """Build a ``ConversationThread`` from FxTwitter API response data."""
    author = tweet_data.get("author", {})
    author_name = author.get("name", "")
    author_handle = f"@{author.get('screen_name', '')}"

    # Extract text — prefer raw_text if available
    raw_text = tweet_data.get("raw_text", {})
    text = raw_text.get("text", "") if isinstance(raw_text, dict) else ""
    if not text:
        text = tweet_data.get("text", "")

    # Extract media
    media_items: list[MediaAttachment] = []
    media_data = tweet_data.get("media", {})
    if isinstance(media_data, dict):
        for item in media_data.get("all", []):
            media_items.append(
                MediaAttachment(
                    url=item.get("url", ""),
                    alt=item.get("altText", ""),
                    media_type=item.get("type", "image"),
                )
            )

    # Extract quoted tweet
    quoted_item: EmbeddedQuote | None = None
    quote_data = tweet_data.get("quote")
    if isinstance(quote_data, dict):
        quote_author = quote_data.get("author", {})
        quoted_item = EmbeddedQuote(
            author_name=quote_author.get("name"),
            author_handle=f"@{quote_author.get('screen_name', '')}",
            text=quote_data.get("text", ""),
            url=quote_data.get("url"),
        )

    main_item = ConversationItem(
        id=tweet_id,
        author_name=author_name,
        author_handle=author_handle,
        text=text,
        timestamp=tweet_data.get("created_at"),
        media=media_items,
        quoted_item=quoted_item,
    )

    title = f"Post by {author_handle}" if author_handle else f"Post by {author_name}"

    return ConversationThread(
        title=title,
        main_item=main_item,
    )


async def fetch_with_fxtwitter(url: str) -> FetchResult | None:
    """Attempt to fetch a tweet via the FxTwitter API.

    Returns ``None`` on any failure (timeout, API error, non-matching URL)
    to allow transparent fallback to other strategies.
    """
    parts = _extract_twitter_url_parts(url)
    if parts is None:
        return None

    username, tweet_id = parts
    api_url = f"{_FXTWITTER_API_BASE}/{username}/status/{tweet_id}"

    try:
        async with httpx.AsyncClient(timeout=_FXTWITTER_TIMEOUT) as client:
            response = await client.get(
                api_url,
                headers={"User-Agent": _USER_AGENT},
            )
            response.raise_for_status()
            data = response.json()
    except Exception as e:
        logger.debug(f"FxTwitter API failed for {url}: {e}")
        return None

    tweet_data = data.get("tweet")
    if not isinstance(tweet_data, dict):
        logger.debug(f"FxTwitter: no tweet data in response for {url}")
        return None

    thread = _build_conversation_thread(
        tweet_data, tweet_id=tweet_id, url=url
    )

    # Render to HTML via shared semantic renderer
    semantic = SemanticExtraction(thread=thread)
    content_html = render_semantic_content(semantic)

    # Convert HTML to markdown via webextract pipeline
    from markitai.webextract.markdown import html_to_markdown, postprocess_markdown

    markdown = html_to_markdown(content_html)
    markdown = postprocess_markdown(markdown)

    metadata: dict[str, Any] = {
        "author": thread.main_item.author_name,
        "author_handle": thread.main_item.author_handle,
        "site": "X (Twitter)",
        "source_frontmatter": {
            "title": thread.title,
            "author": thread.main_item.author_handle,
            "site": "X (Twitter)",
        },
    }

    return FetchResult(
        content=markdown,
        strategy_used="fxtwitter",
        title=thread.title,
        url=url,
        final_url=url,
        metadata=metadata,
    )
```

- [ ] **Step 4: Run tests**

```bash
cd packages/markitai && uv run pytest tests/unit/test_fetch_fxtwitter.py -v
```

- [ ] **Step 5: Run full test suite for regression**

```bash
cd packages/markitai && uv run pytest tests/unit/ -q --ignore=tests/unit/test_fetch_playwright.py
```

- [ ] **Step 6: Commit**

```bash
git add packages/markitai/src/markitai/fetch_fxtwitter.py packages/markitai/tests/unit/test_fetch_fxtwitter.py
git commit -m "feat(fetch): add FxTwitter API client for x.com tweet enrichment"
```

---

### Task 2: Integrate FxTwitter into dispatch strategy

**Files:**
- Modify: `packages/markitai/src/markitai/fetch.py:1743-1775`
- Modify: `packages/markitai/tests/unit/test_fetch_fxtwitter.py`

- [ ] **Step 1: Write integration test**

Add to `test_fetch_fxtwitter.py`:

```python
@pytest.mark.asyncio
async def test_dispatch_strategy_tries_fxtwitter_before_playwright() -> None:
    """_dispatch_strategy should attempt FxTwitter before Playwright for x.com URLs."""
    from unittest.mock import AsyncMock, patch

    from markitai.fetch_types import FetchResult

    mock_fxtwitter_result = FetchResult(
        content="# FxTwitter content",
        strategy_used="fxtwitter",
        title="Test tweet",
        url="https://x.com/user/status/123",
    )

    with patch(
        "markitai.fetch.fetch_with_fxtwitter",
        new_callable=AsyncMock,
        return_value=mock_fxtwitter_result,
    ) as mock_fx:
        from markitai.config import FetchConfig
        from markitai.fetch import _dispatch_strategy
        from markitai.fetch_types import FetchStrategy

        result, _ = await _dispatch_strategy(
            url="https://x.com/user/status/123",
            strategy=FetchStrategy.PLAYWRIGHT,
            config=FetchConfig(),
            explicit_strategy=False,
            screenshot_kwargs={},
            screenshot_config=None,
            screenshot_dir=None,
            renderer=None,
        )
        mock_fx.assert_called_once()
        assert result.strategy_used == "fxtwitter"
```

- [ ] **Step 2: Run test — should fail**

```bash
cd packages/markitai && uv run pytest tests/unit/test_fetch_fxtwitter.py::test_dispatch_strategy_tries_fxtwitter_before_playwright -v
```

- [ ] **Step 3: Integrate into `_dispatch_strategy`**

Read `packages/markitai/src/markitai/fetch.py`, find `_dispatch_strategy` (around line 1726). Find the `if strategy == FetchStrategy.PLAYWRIGHT:` block. At the BEGINNING of that block (before the `pw_result = await fetch_with_playwright(...)` call), add:

```python
        # FxTwitter enrichment: try API before launching browser
        if not explicit_strategy:
            from markitai.fetch_fxtwitter import fetch_with_fxtwitter

            fxtwitter_result = await fetch_with_fxtwitter(url)
            if fxtwitter_result is not None:
                logger.debug("[Fetch] FxTwitter succeeded for {}", url)
                return fxtwitter_result, None
```

The `not explicit_strategy` guard ensures that `--playwright` explicit flag bypasses FxTwitter.

- [ ] **Step 4: Run tests**

```bash
cd packages/markitai && uv run pytest tests/unit/test_fetch_fxtwitter.py -v
```

- [ ] **Step 5: Commit**

```bash
git add packages/markitai/src/markitai/fetch.py packages/markitai/tests/unit/test_fetch_fxtwitter.py
git commit -m "feat(fetch): integrate FxTwitter pre-playwright enrichment in dispatch strategy"
```

---

### Task 3: Add hero header and trailing thin section removal

**Files:**
- Modify: `packages/markitai/src/markitai/webextract/removals/content_patterns.py`
- Test: `packages/markitai/tests/unit/webextract/test_content_patterns.py` (existing or create)

- [ ] **Step 1: Write failing tests**

Find existing test file or create `tests/unit/webextract/test_content_patterns.py`:

```python
"""Tests for content pattern removal."""
from __future__ import annotations

from bs4 import BeautifulSoup, Tag

from markitai.webextract.removals.content_patterns import remove_content_patterns


def _make_root(html: str) -> Tag:
    soup = BeautifulSoup(html, "html.parser")
    return soup.find("div") or soup


class TestHeroHeaderRemoval:
    def test_removes_hero_header_with_time(self) -> None:
        html = """<div>
        <header><h1>Article Title</h1><time>March 23, 2026</time></header>
        <p>This is the actual article content with enough words to matter.</p>
        </div>"""
        root = _make_root(html)
        removed = remove_content_patterns(root)
        text = root.get_text(strip=True)
        assert "actual article content" in text
        assert removed > 0

    def test_preserves_substantial_header(self) -> None:
        """Headers with >30 words should not be removed."""
        words = " ".join(f"word{i}" for i in range(35))
        html = f"""<div>
        <header><h1>Title</h1><p>{words}</p></header>
        <p>Body content.</p>
        </div>"""
        root = _make_root(html)
        remove_content_patterns(root)
        text = root.get_text(strip=True)
        assert "word0" in text


class TestTrailingThinSectionRemoval:
    def test_removes_trailing_cta(self) -> None:
        html = """<div>
        <p>Main article content with enough words to be meaningful.</p>
        <div><h3>Subscribe</h3><p>Get updates</p></div>
        </div>"""
        root = _make_root(html)
        removed = remove_content_patterns(root)
        text = root.get_text(strip=True)
        assert "Main article" in text
        assert removed > 0

    def test_preserves_substantial_trailing_section(self) -> None:
        """Trailing sections with >25 words should not be removed."""
        words = " ".join(f"word{i}" for i in range(30))
        html = f"""<div>
        <p>Main content.</p>
        <div><h3>Conclusion</h3><p>{words}</p></div>
        </div>"""
        root = _make_root(html)
        remove_content_patterns(root)
        text = root.get_text(strip=True)
        assert "word0" in text


class TestBoilerplateCascadeDelete:
    def test_cascade_deletes_siblings_after_boilerplate(self) -> None:
        html = """<div>
        <p>Real content here.</p>
        <p>This article originally appeared in The Times.</p>
        <p>Related stories you might like.</p>
        <p>More junk after boilerplate.</p>
        </div>"""
        root = _make_root(html)
        remove_content_patterns(root)
        text = root.get_text(strip=True)
        assert "Real content" in text
        assert "Related stories" not in text
        assert "More junk" not in text
```

- [ ] **Step 2: Run tests — should fail**

```bash
cd packages/markitai && uv run pytest tests/unit/webextract/test_content_patterns.py -v
```

- [ ] **Step 3: Implement new patterns**

Read `packages/markitai/src/markitai/webextract/removals/content_patterns.py` fully. Add the following functions and integrate them into `remove_content_patterns()`:

**Hero header removal** — find header/div at start of content with H1/H2 + `<time>`, total < 30 words:

```python
def _remove_hero_headers(root: Tag) -> int:
    """Remove hero headers (title + date blocks) at the start of content."""
    removed = 0
    # Check first 3 direct children
    for child in list(root.children)[:5]:
        if not isinstance(child, Tag):
            continue
        # Must contain a heading
        if not child.find(["h1", "h2"]):
            continue
        # Must be short (metadata, not content)
        text = child.get_text(strip=True)
        if len(text.split()) > 30:
            continue
        # Must contain time/date indicator
        if child.find("time") or child.find(class_=lambda c: c and "date" in str(c).lower()):
            child.decompose()
            removed += 1
            break  # Only remove the first hero header
    return removed
```

**Trailing thin section removal** — scan backward from end, remove blocks < 25 words with heading:

```python
def _remove_trailing_thin_sections(root: Tag) -> int:
    """Remove thin trailing sections (CTAs, newsletter prompts)."""
    removed = 0
    children = [c for c in root.children if isinstance(c, Tag)]
    # Scan backward
    for child in reversed(children):
        text = child.get_text(strip=True)
        word_count = len(text.split())
        if word_count > 25:
            break  # Stop at first substantial block
        # Must have a heading to look like a section
        if child.find(["h2", "h3", "h4", "h5", "h6"]):
            child.decompose()
            removed += 1
        else:
            break  # Non-heading thin block — stop scanning
    return removed
```

**Boilerplate cascade** — after matching boilerplate, remove all following siblings:

Modify existing boilerplate removal to cascade-delete siblings after the match.

Add calls to these at the end of `remove_content_patterns()`:

```python
    removed += _remove_hero_headers(root)
    removed += _remove_trailing_thin_sections(root)
```

- [ ] **Step 4: Run tests**

```bash
cd packages/markitai && uv run pytest tests/unit/webextract/test_content_patterns.py -v
cd packages/markitai && uv run pytest tests/unit/webextract/ -q
```

- [ ] **Step 5: Commit**

```bash
git add packages/markitai/src/markitai/webextract/removals/content_patterns.py packages/markitai/tests/unit/webextract/test_content_patterns.py
git commit -m "feat(removals): add hero header and trailing thin section removal"
```

---

### Task 4: Add Twitter-specific DOM cleanup selectors

**Files:**
- Modify: `packages/markitai/src/markitai/constants.py`
- Modify: `packages/markitai/src/markitai/fetch_playwright.py`
- Test: `packages/markitai/tests/unit/test_playwright_domain_profiles.py`

- [ ] **Step 1: Write failing test**

Add to `test_playwright_domain_profiles.py`:

```python
def test_site_noise_selectors_exist_for_x_com() -> None:
    """x.com should have site-specific noise selectors."""
    from markitai.constants import SITE_NOISE_SELECTORS

    assert "x.com" in SITE_NOISE_SELECTORS
    selectors = SITE_NOISE_SELECTORS["x.com"]
    assert '[data-testid="sidebarColumn"]' in selectors
    assert '[data-testid="bottomBar"]' in selectors
```

- [ ] **Step 2: Run test — should fail**

```bash
cd packages/markitai && uv run pytest tests/unit/test_playwright_domain_profiles.py::test_site_noise_selectors_exist_for_x_com -v
```

- [ ] **Step 3: Add `SITE_NOISE_SELECTORS` to constants.py**

At end of `packages/markitai/src/markitai/constants.py` (after `DOM_NOISE_ATTRIBUTES`):

```python
# Site-specific noise selectors injected into Playwright DOM cleanup
# based on the URL domain. These target site chrome that the generic
# selectors cannot match.
SITE_NOISE_SELECTORS: dict[str, tuple[str, ...]] = {
    "x.com": (
        '[data-testid="sidebarColumn"]',
        '[data-testid="DMDrawer"]',
        '[data-testid="sheetDialog"]',
        '[data-testid="bottomBar"]',
        '[data-testid="placementTracking"]',
        '[aria-label="Sign up"]',
        '[aria-label="Footer"]',
    ),
    "twitter.com": (
        '[data-testid="sidebarColumn"]',
        '[data-testid="DMDrawer"]',
        '[data-testid="sheetDialog"]',
        '[data-testid="bottomBar"]',
        '[data-testid="placementTracking"]',
        '[aria-label="Sign up"]',
        '[aria-label="Footer"]',
    ),
}
```

- [ ] **Step 4: Inject into Playwright DOM cleanup**

Read `packages/markitai/src/markitai/fetch_playwright.py`, find `_build_dom_cleanup_script()`. Modify it to accept an optional `url` parameter, and inject site-specific selectors:

```python
def _build_dom_cleanup_script(url: str | None = None) -> str:
```

Add site selector injection inside the function:

```python
    # Inject site-specific noise selectors
    site_selectors_js = ""
    if url:
        from urllib.parse import urlparse
        from markitai.constants import SITE_NOISE_SELECTORS
        domain = urlparse(url).netloc.lower()
        site_sels = SITE_NOISE_SELECTORS.get(domain, ())
        if site_sels:
            site_selectors_js = ", ".join(json.dumps(s) for s in site_sels)
```

Include site selectors in the generated JavaScript alongside the generic ones.

Then update the call site in `fetch()` to pass `url`:

```python
cleanup_script = _build_dom_cleanup_script(url=url)
```

- [ ] **Step 5: Run tests**

```bash
cd packages/markitai && uv run pytest tests/unit/test_playwright_domain_profiles.py -v
cd packages/markitai && uv run pytest tests/unit/test_fetch_playwright.py -q
```

- [ ] **Step 6: Commit**

```bash
git add packages/markitai/src/markitai/constants.py packages/markitai/src/markitai/fetch_playwright.py packages/markitai/tests/unit/test_playwright_domain_profiles.py
git commit -m "feat(playwright): add site-specific DOM cleanup selectors for x.com"
```

---

### Task 5: Verify quality improvement

**Files:** No new files — run existing tests.

- [ ] **Step 1: Run defuddle parity tests**

```bash
cd packages/markitai && uv run pytest tests/integration/test_defuddle_parity_quality.py --tb=no -q 2>&1 | tail -5
```

Compare with Phase 2 baseline (323 passed / 9 failed).

- [ ] **Step 2: Run benchmarks**

```bash
cd packages/markitai && uv run pytest tests/integration/test_webextract_benchmarks.py -v -s
```

- [ ] **Step 3: Run full unit test suite**

```bash
cd packages/markitai && uv run pytest tests/unit/webextract/ tests/unit/test_fetch_fxtwitter.py tests/unit/test_playwright_domain_profiles.py -q
```

All tests must pass.

---

## What's Next

After Phase 3 lands:
- **Phase 4**: CSS @media mobile styles (Module 2.2) + React SSR (Module 2.3) + Markdown engine P1-P2 (math, footnotes, callouts) + Pipeline enhancement (Module 5)
