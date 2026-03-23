# Phase 2: Markdown Engine P0 + Playwright Speed Optimization

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve Markdown output quality by porting defuddle's code-block language detection, and speed up Playwright fetching with domain-specific profiles that skip unnecessary waits and scrolling.

**Architecture:** Create `WebExtractMarkdownConverter` extending MarkItDown's `_CustomMarkdownify` with custom `convert_pre()` for code block language detection. Register it via direct `register_converter()` call in `_create_markitdown()`. For Playwright, extend `DomainProfileConfig` with `skip_auto_scroll` and propagate `reject_resource_patterns`, then add built-in profiles for x.com and github.com.

**Tech Stack:** Python, markdownify, MarkItDown, Playwright, pytest

**Spec:** `docs/superpowers/specs/2026-03-23-webextract-quality-speed-optimization-design.md` (Modules 1 P0 & 4)

---

## File Structure

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `packages/markitai/src/markitai/converter/webextract_html_converter.py` | Custom markdownify converter + HtmlConverter with code-block language detection |
| Modify | `packages/markitai/src/markitai/webextract/pipeline.py:458-466` | Register custom converter in `_create_markitdown()` |
| Modify | `packages/markitai/src/markitai/config.py:507-532` | Add `skip_auto_scroll` + `reject_resource_patterns` to `DomainProfileConfig` |
| Create | `packages/markitai/src/markitai/domain_profiles.py` | Built-in domain profiles (x.com, twitter.com, github.com) |
| Modify | `packages/markitai/src/markitai/fetch.py:831-849` | Propagate new fields + merge built-in profiles |
| Modify | `packages/markitai/src/markitai/fetch_playwright.py:500-507` | Check `skip_auto_scroll` before auto-scrolling |
| Create | `packages/markitai/tests/unit/webextract/test_webextract_markdown_converter.py` | Tests for code-block language detection |
| Create | `packages/markitai/tests/unit/test_playwright_domain_profiles.py` | Tests for domain profile resolution and skip_auto_scroll |

**Scope note**: This phase covers code-block language detection (Module 1 P0 item 1) and all of Module 4 (Playwright speed). The remaining Module 1 P0 items (table enhancement, list nesting) are deferred to Phase 4 alongside P1-P2 items, as they share the same `WebExtractMarkdownConverter` extension point.

---

## Part A: Playwright Speed Optimization (Module 4)

### Task 1: Add `skip_auto_scroll` and `reject_resource_patterns` to DomainProfileConfig

**Files:**
- Modify: `packages/markitai/src/markitai/config.py:507-532`
- Create: `packages/markitai/tests/unit/test_playwright_domain_profiles.py`

- [ ] **Step 1: Write failing test**

Create a new test file:

```python
# tests/unit/test_playwright_domain_profiles.py
"""Tests for Playwright domain profile configuration and resolution."""

from __future__ import annotations

from markitai.config import DomainProfileConfig


def test_domain_profile_skip_auto_scroll_default_false() -> None:
    """skip_auto_scroll defaults to False."""
    profile = DomainProfileConfig()
    assert profile.skip_auto_scroll is False


def test_domain_profile_skip_auto_scroll_can_be_set() -> None:
    """skip_auto_scroll can be explicitly enabled."""
    profile = DomainProfileConfig(skip_auto_scroll=True)
    assert profile.skip_auto_scroll is True


def test_domain_profile_reject_resource_patterns_default_none() -> None:
    """reject_resource_patterns defaults to None."""
    profile = DomainProfileConfig()
    assert profile.reject_resource_patterns is None


def test_domain_profile_reject_resource_patterns_can_be_set() -> None:
    """reject_resource_patterns can be set to a list."""
    profile = DomainProfileConfig(reject_resource_patterns=["**/ads/**"])
    assert profile.reject_resource_patterns == ["**/ads/**"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd packages/markitai && uv run pytest tests/unit/test_playwright_domain_profiles.py -v
```

Expected: FAIL — `DomainProfileConfig` has no `skip_auto_scroll` or `reject_resource_patterns` field.

- [ ] **Step 3: Add both fields to `DomainProfileConfig`**

In `packages/markitai/src/markitai/config.py`, inside the `DomainProfileConfig` class (around line 530), add:

```python
    skip_auto_scroll: bool = Field(
        default=False,
        description="Skip auto-scrolling for single-content pages (tweets, issues, docs).",
    )
    reject_resource_patterns: list[str] | None = Field(
        default=None,
        description="URL patterns to block during Playwright navigation (e.g. '**/analytics/**').",
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd packages/markitai && uv run pytest tests/unit/test_playwright_domain_profiles.py -v
```

- [ ] **Step 5: Commit**

```bash
git add packages/markitai/src/markitai/config.py packages/markitai/tests/unit/test_playwright_domain_profiles.py
git commit -m "feat(config): add skip_auto_scroll and reject_resource_patterns to DomainProfileConfig"
```

---

### Task 2: Propagate new fields through profile resolution

**Files:**
- Modify: `packages/markitai/src/markitai/fetch.py:831-849`
- Test: `packages/markitai/tests/unit/test_playwright_domain_profiles.py`

- [ ] **Step 1: Write failing test**

Add to `test_playwright_domain_profiles.py`:

```python
from markitai.config import DomainProfileConfig, FetchConfig


def test_profile_overrides_propagate_skip_auto_scroll() -> None:
    """skip_auto_scroll from domain profile must be propagated to fetch kwargs."""
    from markitai.fetch import _resolve_playwright_profile_overrides

    profiles = {
        "x.com": DomainProfileConfig(
            skip_auto_scroll=True,
            wait_for_selector='[data-testid="tweet"]',
            extra_wait_ms=500,
        ),
    }
    overrides = _resolve_playwright_profile_overrides("https://x.com/user/status/123", profiles)
    assert overrides.get("skip_auto_scroll") is True
    assert overrides.get("wait_for_selector") == '[data-testid="tweet"]'
    assert overrides.get("extra_wait_ms") == 500


def test_profile_overrides_propagate_reject_resource_patterns() -> None:
    """reject_resource_patterns from domain profile must be propagated."""
    from markitai.fetch import _resolve_playwright_profile_overrides

    profiles = {
        "x.com": DomainProfileConfig(
            reject_resource_patterns=["**/analytics/**", "**/*.mp4"],
        ),
    }
    overrides = _resolve_playwright_profile_overrides("https://x.com/user/status/123", profiles)
    assert overrides.get("reject_resource_patterns") == ["**/analytics/**", "**/*.mp4"]


def test_profile_overrides_no_match_returns_empty() -> None:
    """Non-matching domain returns empty overrides."""
    from markitai.fetch import _resolve_playwright_profile_overrides

    profiles = {
        "x.com": DomainProfileConfig(skip_auto_scroll=True),
    }
    overrides = _resolve_playwright_profile_overrides("https://github.com/repo", profiles)
    assert overrides == {}
```

- [ ] **Step 2: Run tests to verify failures**

```bash
cd packages/markitai && uv run pytest tests/unit/test_playwright_domain_profiles.py -v -k "propagate"
```

Expected: FAIL — `skip_auto_scroll` and `reject_resource_patterns` not in overrides.

- [ ] **Step 3: Update `_resolve_playwright_profile_overrides` in fetch.py**

Read `packages/markitai/src/markitai/fetch.py` and find `_resolve_playwright_profile_overrides` (around line 831). Add propagation for the new fields. After the existing fields, add:

```python
    if profile.skip_auto_scroll:
        out["skip_auto_scroll"] = profile.skip_auto_scroll
    if profile.reject_resource_patterns:
        out["reject_resource_patterns"] = profile.reject_resource_patterns
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd packages/markitai && uv run pytest tests/unit/test_playwright_domain_profiles.py -v
```

- [ ] **Step 5: Commit**

```bash
git add packages/markitai/src/markitai/fetch.py packages/markitai/tests/unit/test_playwright_domain_profiles.py
git commit -m "feat(fetch): propagate skip_auto_scroll and reject_resource_patterns from domain profiles"
```

---

### Task 3: Implement skip_auto_scroll in Playwright fetch

**Files:**
- Modify: `packages/markitai/src/markitai/fetch_playwright.py:424-507`
- Test: `packages/markitai/tests/unit/test_playwright_domain_profiles.py`

- [ ] **Step 1: Write test**

Add to `test_playwright_domain_profiles.py`. Since `PlaywrightRenderer.fetch()` is a complex async method with browser dependencies, we test the parameter propagation path: `_get_playwright_fetch_kwargs` already spreads profile overrides into kwargs via `kwargs.update(profile_overrides)` at line 153, so if `skip_auto_scroll` is in overrides, it will be passed to `fetch()`. We verify the fetch method accepts this parameter:

```python
def test_fetch_method_accepts_skip_auto_scroll() -> None:
    """PlaywrightRenderer.fetch() must accept skip_auto_scroll parameter."""
    import inspect

    from markitai.fetch_playwright import PlaywrightRenderer

    sig = inspect.signature(PlaywrightRenderer.fetch)
    assert "skip_auto_scroll" in sig.parameters, (
        "PlaywrightRenderer.fetch must accept skip_auto_scroll parameter"
    )
    # Default should be False
    assert sig.parameters["skip_auto_scroll"].default is False


def test_get_playwright_fetch_kwargs_includes_skip_auto_scroll() -> None:
    """_get_playwright_fetch_kwargs must pass skip_auto_scroll from domain profile."""
    from markitai.config import FetchConfig
    from markitai.fetch import _get_playwright_fetch_kwargs

    # Use a URL that matches a built-in profile with skip_auto_scroll=True
    kwargs = _get_playwright_fetch_kwargs("https://x.com/user/status/123", FetchConfig())
    assert kwargs.get("skip_auto_scroll") is True
```

- [ ] **Step 2: Run test to verify failure**

```bash
cd packages/markitai && uv run pytest tests/unit/test_playwright_domain_profiles.py::test_skip_auto_scroll_skips_scroll_script -v
```

- [ ] **Step 3: Add `skip_auto_scroll` parameter to `PlaywrightRenderer.fetch()`**

In `packages/markitai/src/markitai/fetch_playwright.py`, read the `fetch` method signature (around line 424). Add `skip_auto_scroll: bool = False` parameter. Then modify the auto-scroll block (around line 500-507):

```python
            # Auto-scroll to trigger lazy-loaded content
            if not skip_auto_scroll:
                try:
                    scroll_script = _build_auto_scroll_script()
                    await page.evaluate(scroll_script)
                    await asyncio.sleep(DEFAULT_PLAYWRIGHT_POST_SCROLL_DELAY_MS / 1000)
                except Exception as e:
                    logger.debug(f"Auto-scroll failed (non-critical): {e}")
```

Also modify the wait logic (around line 487-498) to use `extra_wait_ms` as a stabilization buffer after `wait_for_selector` (not either/or):

```python
            # Precise element waiting (preferred) with optional stabilization
            if wait_for_selector:
                try:
                    await page.wait_for_selector(
                        wait_for_selector, timeout=min(timeout, 10000)
                    )
                except Exception as e:
                    logger.debug(
                        f"wait_for_selector '{wait_for_selector}' timed out: {e}"
                    )
                # Short stabilization wait after selector found
                if extra_wait_ms > 0:
                    await asyncio.sleep(extra_wait_ms / 1000)
            elif extra_wait_ms > 0:
                await asyncio.sleep(extra_wait_ms / 1000)
```

- [ ] **Step 4: Run all tests**

```bash
cd packages/markitai && uv run pytest tests/unit/test_playwright_domain_profiles.py -v
cd packages/markitai && uv run pytest tests/unit/test_fetch_playwright.py -v -q
```

- [ ] **Step 5: Commit**

```bash
git add packages/markitai/src/markitai/fetch_playwright.py packages/markitai/tests/unit/test_playwright_domain_profiles.py
git commit -m "feat(playwright): implement skip_auto_scroll and smart wait strategy"
```

---

### Task 4: Add built-in domain profiles

**Files:**
- Modify: `packages/markitai/src/markitai/constants.py`
- Modify: `packages/markitai/src/markitai/fetch.py`
- Test: `packages/markitai/tests/unit/test_playwright_domain_profiles.py`

- [ ] **Step 1: Write failing test**

```python
def test_builtin_profiles_applied_for_x_com() -> None:
    """x.com should get built-in profile with skip_auto_scroll and wait_for_selector."""
    from markitai.fetch import _resolve_playwright_profile_overrides
    from markitai.domain_profiles import BUILTIN_DOMAIN_PROFILES

    assert "x.com" in BUILTIN_DOMAIN_PROFILES
    assert BUILTIN_DOMAIN_PROFILES["x.com"].skip_auto_scroll is True

    # With empty user profiles, built-in should kick in
    overrides = _resolve_playwright_profile_overrides(
        "https://x.com/user/status/123",
        {},  # no user-configured profiles
    )
    assert overrides.get("skip_auto_scroll") is True
    assert overrides.get("wait_for_selector") == '[data-testid="tweet"]'


def test_user_profile_overrides_builtin() -> None:
    """User-configured profile takes precedence over built-in."""
    from markitai.fetch import _resolve_playwright_profile_overrides

    user_profiles = {
        "x.com": DomainProfileConfig(extra_wait_ms=2000),
    }
    overrides = _resolve_playwright_profile_overrides(
        "https://x.com/user/status/123",
        user_profiles,
    )
    # User's extra_wait_ms overrides built-in's 500
    assert overrides.get("extra_wait_ms") == 2000
    # User didn't set skip_auto_scroll, so it defaults to False (not inherited from built-in)
    assert "skip_auto_scroll" not in overrides
```

- [ ] **Step 2: Run test to verify failure**

```bash
cd packages/markitai && uv run pytest tests/unit/test_playwright_domain_profiles.py::test_builtin_profiles_applied_for_x_com -v
```

- [ ] **Step 3: Create `domain_profiles.py` with built-in profiles**

Create a new module `packages/markitai/src/markitai/domain_profiles.py` to avoid circular imports between `constants.py` ↔ `config.py`:

```python
"""Built-in domain profiles for common sites.

Separated into its own module to avoid circular imports between
``constants.py`` (imported by ``config.py``) and ``config.py``
(needed to construct ``DomainProfileConfig`` instances).
"""

from __future__ import annotations

from markitai.config import DomainProfileConfig

_X_COM_PROFILE = DomainProfileConfig(
    wait_for_selector='[data-testid="tweet"]',
    wait_for="domcontentloaded",
    extra_wait_ms=500,
    skip_auto_scroll=True,
    reject_resource_patterns=[
        "**/analytics/**",
        "**/ads/**",
        "**/tracking/**",
        "**/*.mp4",
    ],
)

BUILTIN_DOMAIN_PROFILES: dict[str, DomainProfileConfig] = {
    "x.com": _X_COM_PROFILE,
    "twitter.com": _X_COM_PROFILE,
    "github.com": DomainProfileConfig(
        wait_for_selector=".markdown-body",
        wait_for="domcontentloaded",
        extra_wait_ms=300,
        skip_auto_scroll=True,
    ),
}
```

- [ ] **Step 4: Merge built-in profiles in `_resolve_playwright_profile_overrides`**

In `fetch.py`, modify `_resolve_playwright_profile_overrides` to import from the new module and merge built-in profiles with user profiles (user takes precedence):

```python
def _resolve_playwright_profile_overrides(
    url: str, domain_profiles: dict[str, Any]
) -> dict[str, Any]:
    """Resolve domain-specific Playwright overrides from config."""
    from urllib.parse import urlparse

    from markitai.domain_profiles import BUILTIN_DOMAIN_PROFILES

    domain = urlparse(url).netloc.lower()

    # User config takes precedence over built-in
    profile = domain_profiles.get(domain)
    if not profile:
        profile = BUILTIN_DOMAIN_PROFILES.get(domain)
    if not profile:
        return {}

    out: dict[str, Any] = {}
    if profile.wait_for_selector:
        out["wait_for_selector"] = profile.wait_for_selector
    if profile.wait_for:
        out["wait_for"] = profile.wait_for
    if profile.extra_wait_ms is not None:
        out["extra_wait_ms"] = profile.extra_wait_ms
    if profile.skip_auto_scroll:
        out["skip_auto_scroll"] = profile.skip_auto_scroll
    if profile.reject_resource_patterns:
        out["reject_resource_patterns"] = profile.reject_resource_patterns
    return out
```

- [ ] **Step 5: Run all tests**

```bash
cd packages/markitai && uv run pytest tests/unit/test_playwright_domain_profiles.py -v
cd packages/markitai && uv run pytest tests/unit/test_fetch_playwright.py -q
```

- [ ] **Step 6: Commit**

```bash
git add packages/markitai/src/markitai/constants.py packages/markitai/src/markitai/fetch.py packages/markitai/tests/unit/test_playwright_domain_profiles.py
git commit -m "feat(playwright): add built-in domain profiles for x.com, twitter.com, github.com"
```

---

## Part B: Markdown Engine P0 — Code Block Language Detection (Module 1)

### Task 5: Create WebExtractMarkdownConverter with code-block language detection

**Files:**
- Create: `packages/markitai/src/markitai/converter/webextract_html_converter.py`
- Create: `packages/markitai/tests/unit/webextract/test_webextract_markdown_converter.py`

- [ ] **Step 1: Write failing tests for code-block language detection**

```python
# tests/unit/webextract/test_webextract_markdown_converter.py
"""Tests for WebExtractMarkdownConverter code-block language detection."""

from __future__ import annotations

import pytest

from markitai.converter.webextract_html_converter import WebExtractMarkdownConverter


def _convert(html: str) -> str:
    """Helper: convert an HTML fragment to Markdown via our custom converter."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    converter = WebExtractMarkdownConverter()
    return converter.convert_soup(soup).strip()


class TestCodeBlockLanguageDetection:
    """Code blocks should have language identifiers extracted from class names."""

    def test_language_class_prefix(self) -> None:
        html = '<pre><code class="language-python">print("hello")</code></pre>'
        md = _convert(html)
        assert "```python" in md
        assert 'print("hello")' in md

    def test_lang_class_prefix(self) -> None:
        html = '<pre><code class="lang-javascript">const x = 1;</code></pre>'
        md = _convert(html)
        assert "```javascript" in md

    def test_highlight_class_prefix(self) -> None:
        html = '<pre><code class="highlight-ruby">puts "hi"</code></pre>'
        md = _convert(html)
        assert "```ruby" in md

    def test_data_lang_attribute(self) -> None:
        html = '<pre><code data-lang="rust">fn main() {}</code></pre>'
        md = _convert(html)
        assert "```rust" in md

    def test_prism_class_on_pre(self) -> None:
        """Prism.js puts language class on <pre>, not <code>."""
        html = '<pre class="language-typescript"><code>let x: number = 1;</code></pre>'
        md = _convert(html)
        assert "```typescript" in md

    def test_no_language_produces_plain_code_block(self) -> None:
        html = "<pre><code>plain code</code></pre>"
        md = _convert(html)
        assert "```" in md
        assert "plain code" in md

    def test_syntax_highlighter_class(self) -> None:
        """SyntaxHighlighter uses brush: prefix."""
        html = '<pre class="brush: java">public class Foo {}</pre>'
        md = _convert(html)
        assert "```java" in md

    def test_multiple_classes_picks_language(self) -> None:
        html = '<pre><code class="hljs language-go">func main() {}</code></pre>'
        md = _convert(html)
        assert "```go" in md

    def test_unknown_language_class_still_detected(self) -> None:
        """Even non-standard languages should be extracted if pattern matches."""
        html = '<pre><code class="language-solidity">pragma solidity ^0.8.0;</code></pre>'
        md = _convert(html)
        assert "```solidity" in md
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd packages/markitai && uv run pytest tests/unit/webextract/test_webextract_markdown_converter.py -v
```

Expected: ImportError — `webextract_html_converter` does not exist yet.

- [ ] **Step 3: Implement `WebExtractMarkdownConverter`**

Create `packages/markitai/src/markitai/converter/webextract_html_converter.py`:

```python
"""Custom Markdown converter with enhanced code-block language detection.

Extends MarkItDown's ``_CustomMarkdownify`` with rules ported from
defuddle's ``elements/code.ts``.  Registered as a higher-priority
HTML converter so the webextract pipeline benefits from these rules.
"""

from __future__ import annotations

import re
from typing import Any

from markitdown.converters._markdownify import _CustomMarkdownify

# ---------------------------------------------------------------------------
# Language detection patterns (ported from defuddle elements/code.ts)
# ---------------------------------------------------------------------------

_LANG_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^language-(\w+)$"),        # language-python
    re.compile(r"^lang-(\w+)$"),            # lang-python
    re.compile(r"^highlight-(\w+)$"),       # highlight-python
    re.compile(r"^(\w+)-code$"),            # python-code
    re.compile(r"^code-(\w+)$"),            # code-python
    re.compile(r"^syntax-(\w+)$"),          # syntax-python
    re.compile(r"^code-snippet__(\w+)$"),   # code-snippet__python
    re.compile(r"^(\w+)-snippet$"),         # python-snippet
]

# Fallback pattern for space-separated class lists
_LANG_FALLBACK_RE = re.compile(
    r"(?:^|\s)(?:language|lang|brush|syntax)[:\s-](\w+)(?:\s|$)", re.IGNORECASE
)


def _detect_language(el: Any) -> str | None:
    """Detect programming language from element attributes.

    Checks ``data-lang``, then class names on the element and its
    parent ``<pre>`` (Prism.js style), using patterns ported from
    defuddle's code.ts.

    Args:
        el: BeautifulSoup Tag to inspect (typically ``<code>`` or ``<pre>``).

    Returns:
        Detected language string, or ``None``.
    """
    # 1. Check data-lang attribute
    data_lang = el.get("data-lang")
    if data_lang:
        return str(data_lang).strip()

    # 2. Check class names on this element + parent <pre>
    candidates = [el]
    parent = el.parent
    if parent and parent.name == "pre":
        candidates.append(parent)
    elif el.name == "pre":
        # Check child <code> too
        code = el.find("code")
        if code:
            candidates.insert(0, code)

    for candidate in candidates:
        classes = candidate.get("class", [])
        if isinstance(classes, str):
            classes = classes.split()

        for cls in classes:
            # Try each pattern
            for pattern in _LANG_PATTERNS:
                m = pattern.match(cls)
                if m:
                    return m.group(1).lower()

        # Fallback: check full class string for "brush: java" etc.
        class_str = " ".join(classes) if isinstance(classes, list) else str(classes)
        m = _LANG_FALLBACK_RE.search(class_str)
        if m:
            return m.group(1).lower()

    return None


class WebExtractMarkdownConverter(_CustomMarkdownify):
    """Markdownify converter with enhanced rules for web content.

    Extends MarkItDown's ``_CustomMarkdownify`` with:
    - Code-block language detection from CSS classes / data attributes
    """

    def convert_pre(self, el: Any, text: str, parent_tags: set) -> str:
        """Convert ``<pre>`` to fenced code block with language detection.

        Detects language from class names (``language-python``,
        ``lang-js``, ``highlight-ruby``, ``brush: java``, etc.)
        and ``data-lang`` attributes.

        Args:
            el: The ``<pre>`` BeautifulSoup Tag.
            text: Pre-converted inner text (from markdownify).
            parent_tags: Set of ancestor tag names.
        """
        # Find the <code> child if present
        code_el = el.find("code") if el.name == "pre" else el
        if code_el is None:
            code_el = el

        language = _detect_language(code_el) or ""

        # Extract raw text content (preserving whitespace)
        code_text = code_el.get_text() if code_el else el.get_text()

        # Strip single leading/trailing newline (common in <pre><code>)
        if code_text.startswith("\n"):
            code_text = code_text[1:]
        if code_text.endswith("\n"):
            code_text = code_text[:-1]

        return f"\n\n```{language}\n{code_text}\n```\n\n"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd packages/markitai && uv run pytest tests/unit/webextract/test_webextract_markdown_converter.py -v
```

- [ ] **Step 5: Commit**

```bash
git add packages/markitai/src/markitai/converter/webextract_html_converter.py packages/markitai/tests/unit/webextract/test_webextract_markdown_converter.py
git commit -m "feat(markdown): add WebExtractMarkdownConverter with code-block language detection"
```

---

### Task 6: Register custom converter in pipeline

**Files:**
- Modify: `packages/markitai/src/markitai/webextract/pipeline.py:458-466`
- Test: `packages/markitai/tests/unit/webextract/test_pipeline.py`

- [ ] **Step 1: Write failing test**

Add to `test_pipeline.py`:

```python
def test_pipeline_detects_code_block_language() -> None:
    """Pipeline must produce language-tagged code blocks."""
    html = """<html><body><article>
    <p>Here is an article with enough content to pass extraction threshold checks easily.</p>
    <pre><code class="language-python">
def hello():
    print("world")
    </code></pre>
    <p>More content after the code block for word count.</p>
    </article></body></html>"""
    result = extract_web_content(html, "https://example.com")
    assert "```python" in result.markdown
    assert "def hello():" in result.markdown
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd packages/markitai && uv run pytest tests/unit/webextract/test_pipeline.py -v -k "test_pipeline_detects_code_block"
```

Expected: FAIL — current pipeline uses default markdownify which doesn't detect languages.

- [ ] **Step 3: Add `WebExtractHtmlConverter` class to the converter module**

In `packages/markitai/src/markitai/converter/webextract_html_converter.py`, add at the end of the file (after `WebExtractMarkdownConverter`):

```python
from markitdown._base_converter import DocumentConverterResult
from markitdown.converters._html_converter import HtmlConverter


class WebExtractHtmlConverter(HtmlConverter):
    """HtmlConverter that uses our custom markdownify converter.

    Registered at higher priority than the built-in ``HtmlConverter``
    so code-block language detection and other enhanced rules apply.
    """

    def convert(self, file_stream, stream_info, **kwargs):
        from bs4 import BeautifulSoup

        encoding = "utf-8" if stream_info.charset is None else stream_info.charset
        soup = BeautifulSoup(file_stream, "html.parser", from_encoding=encoding)

        for script in soup(["script", "style"]):
            script.extract()

        body = soup.find("body")
        if body:
            webpage_text = WebExtractMarkdownConverter(**kwargs).convert_soup(body)
        else:
            webpage_text = WebExtractMarkdownConverter(**kwargs).convert_soup(soup)

        return DocumentConverterResult(
            markdown=webpage_text.strip(),
            title=None if soup.title is None else soup.title.string,
        )
```

- [ ] **Step 4: Register it in `_create_markitdown()` in pipeline.py**

In `packages/markitai/src/markitai/webextract/pipeline.py`, modify `_create_markitdown()` (around line 458):

```python
def _create_markitdown() -> object:
    """Create a MarkItDown instance with WebExtract's custom converter.

    Registers ``WebExtractHtmlConverter`` at higher priority than the
    built-in ``HtmlConverter`` so code-block language detection and
    other enhanced rules are applied.
    """
    from markitdown import MarkItDown

    from markitai.converter.webextract_html_converter import WebExtractHtmlConverter

    md = MarkItDown()
    md.register_converter(WebExtractHtmlConverter(), priority=-1)
    return md
```

- [ ] **Step 4: Run tests**

```bash
cd packages/markitai && uv run pytest tests/unit/webextract/test_pipeline.py -v -k "test_pipeline_detects_code_block"
cd packages/markitai && uv run pytest tests/unit/webextract/ -q
```

- [ ] **Step 5: Commit**

```bash
git add packages/markitai/src/markitai/webextract/pipeline.py
git commit -m "feat(pipeline): register WebExtractMarkdownConverter for code-block language detection"
```

---

### Task 7: Verify defuddle parity improvement

**Files:** No new files — run existing tests.

- [ ] **Step 1: Run defuddle parity tests to measure improvement**

```bash
cd packages/markitai && uv run pytest tests/integration/test_defuddle_parity_quality.py -v --tb=short 2>&1 | tail -15
```

Compare pass/fail counts with Phase 1 baseline (303 passed, 29 failed).

- [ ] **Step 2: Run defuddle benchmarks to verify no performance regression**

```bash
cd packages/markitai && uv run pytest tests/integration/test_webextract_benchmarks.py -v -s
```

- [ ] **Step 3: Run full test suite**

```bash
cd packages/markitai && uv run pytest tests/unit/webextract/ -q
```

All tests must pass.

---

## What's Next

After Phase 2 lands:
- **Phase 3 plan**: Noise removal (Module 2) + FxTwitter API (Module 3.1)
- **Phase 4 plan**: Markdown engine P1-P2 (math, footnotes, callouts, highlights) + Pipeline enhancement (Module 5)
