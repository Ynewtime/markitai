# Phase 4B: CSS @media Mobile Styles + React SSR + Retry Enhancement

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix remaining parity failures by adding CSS @media mobile style pruning (fixes tailwind-hidden), React SSR streaming resolution (fixes substack-app), and hidden-content retry enhancement.

**Architecture:** CSS @media pruning runs as a document-level step BEFORE content scoring in `_extract_generic()`. React SSR resolution runs in `preprocess.py` as a string-level operation before BeautifulSoup parsing. Both are independent modules with clear interfaces.

**Tech Stack:** Python, tinycss2, BeautifulSoup, pytest

**Spec:** `docs/superpowers/specs/2026-03-23-webextract-quality-speed-optimization-design.md` (Modules 2.2, 2.3, 5.3)

---

## File Structure

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `packages/markitai/src/markitai/webextract/mobile_styles.py` | CSS @media analysis + document-level pruning |
| Modify | `packages/markitai/src/markitai/webextract/pipeline.py` | Call mobile style pruning before scoring |
| Modify | `packages/markitai/src/markitai/webextract/preprocess.py` | Add React SSR streaming resolution |
| Modify | `packages/markitai/pyproject.toml` | Add tinycss2 dependency |
| Create | `packages/markitai/tests/unit/webextract/test_mobile_styles.py` | Tests for CSS @media pruning |
| Modify | `packages/markitai/tests/unit/webextract/test_preprocess.py` | Tests for React SSR |

---

### Task 1: Add tinycss2 dependency

**Files:**
- Modify: `packages/markitai/pyproject.toml`

- [ ] **Step 1: Add tinycss2 to dependencies**

In `packages/markitai/pyproject.toml`, add `tinycss2` to the dependencies list:

```
"tinycss2>=1.2",
```

- [ ] **Step 2: Install**

```bash
uv sync
uv run python -c "import tinycss2; print(tinycss2.__version__)"
```

- [ ] **Step 3: Commit**

```bash
git add packages/markitai/pyproject.toml uv.lock
git commit -m "deps: add tinycss2 for CSS @media mobile style analysis"
```

---

### Task 2: CSS @media mobile style pruning

**Files:**
- Create: `packages/markitai/src/markitai/webextract/mobile_styles.py`
- Create: `packages/markitai/tests/unit/webextract/test_mobile_styles.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for CSS @media mobile style pruning."""
from __future__ import annotations

from bs4 import BeautifulSoup

from markitai.webextract.mobile_styles import apply_mobile_style_pruning


def test_removes_sidebar_hidden_on_mobile() -> None:
    """Elements hidden via @media max-width should be removed."""
    html = """<html><head><style>
    @media (max-width: 768px) { .sidebar { display: none; } }
    </style></head><body>
    <article><p>Main content.</p></article>
    <div class="sidebar"><p>Sidebar noise.</p></div>
    </body></html>"""
    soup = BeautifulSoup(html, "html.parser")
    removed = apply_mobile_style_pruning(soup)
    assert removed > 0
    assert "Sidebar noise" not in soup.get_text()
    assert "Main content" in soup.get_text()


def test_preserves_elements_not_hidden() -> None:
    """Elements NOT hidden in mobile styles should be preserved."""
    html = """<html><head><style>
    @media (max-width: 768px) { .nav { display: none; } }
    </style></head><body>
    <article><p>Content.</p></article>
    <div class="footer"><p>Footer.</p></div>
    </body></html>"""
    soup = BeautifulSoup(html, "html.parser")
    apply_mobile_style_pruning(soup)
    assert "Footer" in soup.get_text()


def test_no_style_tags_returns_zero() -> None:
    """HTML without style tags should return 0 removals."""
    html = "<html><body><p>Just text.</p></body></html>"
    soup = BeautifulSoup(html, "html.parser")
    assert apply_mobile_style_pruning(soup) == 0


def test_ignores_large_breakpoint_media_queries() -> None:
    """@media with max-width > 768px should be ignored (desktop, not mobile)."""
    html = """<html><head><style>
    @media (max-width: 1200px) { .wide-sidebar { display: none; } }
    </style></head><body>
    <div class="wide-sidebar"><p>Should stay.</p></div>
    </body></html>"""
    soup = BeautifulSoup(html, "html.parser")
    apply_mobile_style_pruning(soup)
    assert "Should stay" in soup.get_text()
```

- [ ] **Step 2: Run tests — should fail (module not found)**

- [ ] **Step 3: Implement mobile_styles.py**

```python
"""CSS @media mobile style analysis and pruning.

Analyzes ``<style>`` tags for ``@media (max-width: ≤768px)`` rules
that hide elements (``display: none``). Applies those selectors to
remove mobile-hidden sidebar/nav elements before content scoring.
"""
from __future__ import annotations

import re

from bs4 import BeautifulSoup, Tag
from loguru import logger


def apply_mobile_style_pruning(soup: BeautifulSoup) -> int:
    """Remove elements hidden via CSS @media mobile breakpoints.

    Parses ``<style>`` tags, finds ``@media (max-width: ≤768px)``
    rules with ``display: none``, and removes matching elements.

    Args:
        soup: Full document BeautifulSoup (mutated in place).

    Returns:
        Number of elements removed.
    """
    try:
        import tinycss2
    except ImportError:
        logger.debug("tinycss2 not installed, skipping mobile style pruning")
        return 0

    hidden_selectors: list[str] = []

    for style_tag in soup.find_all("style"):
        css_text = style_tag.string
        if not css_text:
            continue
        hidden_selectors.extend(_extract_mobile_hidden_selectors(css_text))

    if not hidden_selectors:
        return 0

    removed = 0
    for selector in hidden_selectors:
        try:
            for el in soup.select(selector):
                if isinstance(el, Tag):
                    el.decompose()
                    removed += 1
        except Exception:  # noqa: BLE001
            continue

    if removed > 0:
        logger.debug(
            "[MobileStyles] Removed {} elements via {} mobile-hidden selectors",
            removed,
            len(hidden_selectors),
        )

    return removed


_MAX_WIDTH_RE = re.compile(r"max-width\s*:\s*(\d+(?:\.\d+)?)\s*(px|em|rem)", re.IGNORECASE)
_DISPLAY_NONE_RE = re.compile(r"display\s*:\s*none", re.IGNORECASE)
_MOBILE_MAX_WIDTH_PX = 768


def _extract_mobile_hidden_selectors(css_text: str) -> list[str]:
    """Extract selectors hidden at mobile breakpoints from CSS text."""
    import tinycss2

    selectors: list[str] = []
    rules = tinycss2.parse_stylesheet(css_text, skip_comments=True)

    for rule in rules:
        if rule.type != "at-rule" or rule.lower_at_keyword != "media":
            continue

        # Check if this is a mobile breakpoint (max-width <= 768px)
        prelude_str = tinycss2.serialize(rule.prelude)
        match = _MAX_WIDTH_RE.search(prelude_str)
        if not match:
            continue

        value = float(match.group(1))
        unit = match.group(2).lower()

        # Convert em/rem to approximate px (assume 16px base)
        if unit in ("em", "rem"):
            value *= 16

        if value > _MOBILE_MAX_WIDTH_PX:
            continue

        # Parse the content of the @media block
        if not rule.content:
            continue

        content_rules = tinycss2.parse_rule_list(rule.content, skip_comments=True)
        for content_rule in content_rules:
            if content_rule.type != "qualified-rule":
                continue

            # Check if any declaration is display: none
            declarations_str = tinycss2.serialize(content_rule.content)
            if not _DISPLAY_NONE_RE.search(declarations_str):
                continue

            # Extract the selector
            selector_str = tinycss2.serialize(content_rule.prelude).strip()
            if selector_str:
                selectors.append(selector_str)

    return selectors
```

- [ ] **Step 4: Run tests**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(webextract): add CSS @media mobile style pruning"
```

---

### Task 3: Integrate mobile style pruning into pipeline

**Files:**
- Modify: `packages/markitai/src/markitai/webextract/pipeline.py`

- [ ] **Step 1: Write test**

```python
def test_mobile_hidden_sidebar_removed_before_scoring() -> None:
    """Mobile-hidden elements should be removed before content scoring."""
    html = """<html><head><style>
    @media (max-width: 600px) { .sidebar { display: none; } }
    </style></head><body>
    <article><p>Real article content with enough words for extraction.</p></article>
    <div class="sidebar"><nav><a href="/a">Link A</a><a href="/b">Link B</a></nav></div>
    </body></html>"""
    result = extract_web_content(html, "https://example.com")
    assert "Real article" in result.markdown
    assert "Link A" not in result.markdown
```

- [ ] **Step 2: Modify `_extract_generic` in pipeline.py**

Add mobile style pruning call BEFORE `_pick_root`. In `_extract_generic`:

```python
    ctx = _ExtractionContext(html, url)

    # Prune mobile-hidden elements before scoring
    from markitai.webextract.mobile_styles import apply_mobile_style_pruning
    apply_mobile_style_pruning(ctx.original_soup)

    extractor = find_extractor(url)
    root = _pick_root(ctx.original_soup, extractor)
```

- [ ] **Step 3: Run tests**

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(pipeline): integrate mobile style pruning before content scoring"
```

---

### Task 4: React SSR streaming resolution

**Files:**
- Modify: `packages/markitai/src/markitai/webextract/preprocess.py`
- Modify: `packages/markitai/tests/unit/webextract/test_preprocess.py`

- [ ] **Step 1: Write failing test**

```python
def test_react_ssr_boundary_resolution() -> None:
    """React SSR $RC boundaries should be resolved to actual content."""
    from markitai.webextract.preprocess import preprocess_html

    html = '''<html><body>
    <!--$?--><template id="B:0"></template>Loading...<!--/$-->
    <div hidden id="S:0"><p>Server-rendered content here.</p></div>
    <script>$RC("B:0","S:0")</script>
    </body></html>'''
    result = preprocess_html(html)
    assert "Server-rendered content here" in result
    assert "Loading..." not in result or "Server-rendered" in result
```

- [ ] **Step 2: Implement React SSR resolution**

Add to `preprocess.py`:

```python
_RC_CALL_RE = re.compile(r'\$RC\s*\(\s*"(B:\d+)"\s*,\s*"(S:\d+)"\s*\)')

def _resolve_react_ssr_boundaries(html: str) -> str:
    """Replace React SSR streaming boundaries with actual content.

    React Streaming SSR uses $RC("B:X","S:X") calls to replace
    placeholder template boundaries with server-rendered content
    stored in hidden divs.
    """
    matches = list(_RC_CALL_RE.finditer(html))
    if not matches:
        return html

    for match in matches:
        boundary_id = match.group(1)  # e.g. "B:0"
        content_id = match.group(2)   # e.g. "S:0"

        # Find the hidden div with the content
        content_pattern = re.compile(
            rf'<div[^>]+id="{re.escape(content_id)}"[^>]*>(.*?)</div>',
            re.DOTALL | re.IGNORECASE,
        )
        content_match = content_pattern.search(html)
        if not content_match:
            continue

        content_html = content_match.group(1)

        # Replace the boundary placeholder with actual content
        boundary_pattern = re.compile(
            rf'<!--\$\?--><template id="{re.escape(boundary_id)}"></template>.*?<!--/\$-->',
            re.DOTALL,
        )
        html = boundary_pattern.sub(content_html, html)

        # Remove the hidden source div
        html = content_pattern.sub("", html)

    return html
```

Add `_resolve_react_ssr_boundaries` to the `preprocess_html` function's pipeline.

- [ ] **Step 3: Run tests**

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(preprocess): add React SSR streaming boundary resolution"
```

---

### Task 5: Verify parity improvement

- [ ] **Step 1: Run parity tests**

```bash
cd packages/markitai && uv run pytest tests/integration/test_defuddle_parity_quality.py --tb=no -q
```

Target: fix `tailwind-hidden-blog-index` (CSS @media) and `substack-app` (React SSR).

- [ ] **Step 2: Run full test suite**

```bash
cd packages/markitai && uv run pytest tests/unit/webextract/ -q
```
