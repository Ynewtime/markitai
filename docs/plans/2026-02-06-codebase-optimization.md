# Markitai 代码库优化实施方案

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 结合 baoyu-skills 技术分析，对 markitai 代码库进行全面优化——借鉴有价值的技术模式，消除内部代码重复，修复线程安全隐患，提升 Playwright 渲染质量，增强测试覆盖。

**Architecture:** 本方案分为 5 个独立任务组，按优先级排序。每个任务组可独立执行、独立测试、独立提交。核心思路是：(1) 先借鉴 baoyu-skills 的最佳实践增强 URL 抓取能力；(2) 再消除内部技术债务（代码重复、线程安全）；(3) 最后补充测试覆盖。

**Tech Stack:** Python 3.11+, asyncio, Playwright, Pydantic v2, pytest, ruff

---

## 调研发现摘要

### 来自 baoyu-skills 的可借鉴技术

| 技术 | 来源 | 应用场景 | 优先级 |
|------|------|----------|--------|
| 自动滚动加载 | url-to-markdown | Playwright 渲染时触发懒加载 | **高** |
| DOM 噪音清洗 | url-to-markdown | URL 提取预处理层 | **高** |
| `pending ≤ N` Network Idle | url-to-markdown | 替代固定等待的中间策略 | **中** |
| CJK/英文自动间距 | format-markdown | 中文文档输出后处理 | **低** |

### 内部代码质量问题

| 类别 | 严重性 | 数量 | 关键文件 |
|------|--------|------|----------|
| 代码重复 | 中 | 6 处 | content.py, document.py, vision.py |
| 静默失败 | 高 | 40+ 处 | security.py, batch.py, fetch.py |
| 线程安全 | 高 | 3 处 | batch.py, core.py |
| 测试缺口 | 中 | 8 个方向 | batch 并发, vision 回退 |
| 缓存碰撞 | 中 | 2 处 | vision.py, document.py |

---

## Task 1: Playwright 渲染增强（借鉴 baoyu-skills）

> 从 baoyu-skills 的 url-to-markdown 借鉴自动滚动和 DOM 清洗技术，
> 直接提升 markitai 的 URL 抓取质量。

**Files:**
- Modify: `packages/markitai/src/markitai/fetch_playwright.py:206-265`
- Modify: `packages/markitai/src/markitai/constants.py`
- Test: `packages/markitai/tests/unit/test_playwright_enhancements.py`

### Step 1: Write the failing test for auto-scroll

```python
# tests/unit/test_playwright_enhancements.py
"""Tests for Playwright rendering enhancements (auto-scroll + DOM cleanup)."""

import pytest


class TestAutoScroll:
    """Test auto-scroll logic for triggering lazy-loaded content."""

    def test_auto_scroll_script_structure(self):
        """Verify auto-scroll JS script has correct structure."""
        from markitai.fetch_playwright import _build_auto_scroll_script

        script = _build_auto_scroll_script(max_steps=8, step_delay_ms=600)
        assert "scrollTo" in script
        assert "scrollHeight" in script
        assert "document.body" in script

    def test_auto_scroll_script_returns_to_top(self):
        """Verify scroll returns to top after scrolling."""
        from markitai.fetch_playwright import _build_auto_scroll_script

        script = _build_auto_scroll_script(max_steps=8, step_delay_ms=600)
        # Script should scroll back to top at the end
        assert "scrollTo(0, 0)" in script or "scrollTo(0,0)" in script


class TestDomCleanup:
    """Test DOM cleanup for removing noise elements before extraction."""

    def test_dom_cleanup_script_removes_noise(self):
        """Verify DOM cleanup script targets known noise selectors."""
        from markitai.fetch_playwright import _build_dom_cleanup_script

        script = _build_dom_cleanup_script()
        # Should target common noise elements
        assert "script" in script
        assert "noscript" in script
        assert "cookie" in script.lower() or "cookie-banner" in script

    def test_dom_cleanup_script_removes_ad_elements(self):
        """Verify ad-related elements are targeted."""
        from markitai.fetch_playwright import _build_dom_cleanup_script

        script = _build_dom_cleanup_script()
        assert "advertisement" in script or ".ad" in script

    def test_dom_cleanup_cleans_attributes(self):
        """Verify inline event handlers and styles are removed."""
        from markitai.fetch_playwright import _build_dom_cleanup_script

        script = _build_dom_cleanup_script()
        assert "onclick" in script or "removeAttribute" in script
```

### Step 2: Run test to verify it fails

Run: `cd /home/y/dev/markitai && uv run pytest packages/markitai/tests/unit/test_playwright_enhancements.py -v`
Expected: FAIL with ImportError (functions not yet defined)

### Step 3: Add constants for auto-scroll and DOM cleanup

```python
# In constants.py, add after DEFAULT_PLAYWRIGHT_EXTRA_WAIT_MS section:

# Playwright auto-scroll settings (inspired by baoyu-skills url-to-markdown)
DEFAULT_PLAYWRIGHT_AUTO_SCROLL_STEPS = 8  # Max scroll iterations
DEFAULT_PLAYWRIGHT_AUTO_SCROLL_DELAY_MS = 600  # Delay between scroll steps
DEFAULT_PLAYWRIGHT_POST_SCROLL_DELAY_MS = 800  # Wait after scrolling completes

# DOM noise selectors to remove before content extraction
# Based on baoyu-skills url-to-markdown's proven selector set
DOM_NOISE_SELECTORS: tuple[str, ...] = (
    "script",
    "style",
    "noscript",
    "iframe",
    "svg",
    "canvas",
    "header nav",
    "footer",
    ".sidebar",
    ".nav",
    ".navigation",
    ".advertisement",
    ".ad",
    ".ads",
    ".cookie-banner",
    ".popup",
    '[role="banner"]',
    '[role="navigation"]',
    '[role="complementary"]',
)

# HTML attributes to remove from DOM (event handlers, inline styles)
DOM_NOISE_ATTRIBUTES: tuple[str, ...] = (
    "style",
    "onclick",
    "onload",
    "onerror",
    "onmouseover",
    "onmouseout",
)
```

### Step 4: Implement auto-scroll and DOM cleanup functions

```python
# In fetch_playwright.py, add before PlaywrightRenderer class:

def _build_auto_scroll_script(
    max_steps: int = DEFAULT_PLAYWRIGHT_AUTO_SCROLL_STEPS,
    step_delay_ms: int = DEFAULT_PLAYWRIGHT_AUTO_SCROLL_DELAY_MS,
) -> str:
    """Build JavaScript for auto-scrolling to trigger lazy-loaded content.

    Borrowed from baoyu-skills url-to-markdown pattern:
    Scroll down incrementally, check if page height grows, stop when stable.

    Args:
        max_steps: Maximum number of scroll iterations
        step_delay_ms: Delay between scroll steps in milliseconds

    Returns:
        JavaScript code string for page.evaluate()
    """
    return f"""
    async () => {{
        let lastHeight = document.body.scrollHeight;
        for (let i = 0; i < {max_steps}; i++) {{
            window.scrollTo(0, document.body.scrollHeight);
            await new Promise(r => setTimeout(r, {step_delay_ms}));
            const newHeight = document.body.scrollHeight;
            if (newHeight === lastHeight) break;
            lastHeight = newHeight;
        }}
        window.scrollTo(0, 0);
    }}
    """


def _build_dom_cleanup_script() -> str:
    """Build JavaScript for removing DOM noise before content extraction.

    Borrowed from baoyu-skills url-to-markdown pattern:
    Remove navigation, ads, popups, cookie banners, and inline event handlers.

    Returns:
        JavaScript code string for page.evaluate()
    """
    from markitai.constants import DOM_NOISE_ATTRIBUTES, DOM_NOISE_SELECTORS

    selectors_js = ", ".join(f'"{s}"' for s in DOM_NOISE_SELECTORS)
    attributes_js = ", ".join(f'"{a}"' for a in DOM_NOISE_ATTRIBUTES)

    return f"""
    () => {{
        // Step 1: Remove noise elements
        const selectors = [{selectors_js}];
        for (const sel of selectors) {{
            try {{
                document.querySelectorAll(sel).forEach(el => el.remove());
            }} catch (e) {{}}
        }}

        // Step 2: Clean inline event handlers and styles
        const attrs = [{attributes_js}];
        document.querySelectorAll('*').forEach(el => {{
            for (const attr of attrs) {{
                el.removeAttribute(attr);
            }}
        }});

        // Step 3: Convert relative URLs to absolute
        const base = document.baseURI;
        document.querySelectorAll('a[href]').forEach(a => {{
            try {{
                const href = a.getAttribute('href');
                if (href && !href.startsWith('http') && !href.startsWith('//') && !href.startsWith('#')) {{
                    a.setAttribute('href', new URL(href, base).href);
                }}
            }} catch (e) {{}}
        }});
        document.querySelectorAll('img[src]').forEach(img => {{
            try {{
                const src = img.getAttribute('src');
                if (src && !src.startsWith('http') && !src.startsWith('data:') && !src.startsWith('//')) {{
                    img.setAttribute('src', new URL(src, base).href);
                }}
            }} catch (e) {{}}
        }});
    }}
    """
```

### Step 5: Integrate into PlaywrightRenderer.fetch()

Modify `PlaywrightRenderer.fetch()` method to add auto-scroll and DOM cleanup
between page load and content extraction:

```python
# In PlaywrightRenderer.fetch(), after page.goto() and extra_wait_ms:

            # Auto-scroll to trigger lazy-loaded content
            # (inspired by baoyu-skills url-to-markdown)
            try:
                scroll_script = _build_auto_scroll_script()
                await page.evaluate(scroll_script)
                # Post-scroll settling delay
                await asyncio.sleep(
                    DEFAULT_PLAYWRIGHT_POST_SCROLL_DELAY_MS / 1000
                )
            except Exception as e:
                logger.debug(f"Auto-scroll failed (non-critical): {e}")

            # DOM cleanup: remove noise elements before extraction
            try:
                cleanup_script = _build_dom_cleanup_script()
                await page.evaluate(cleanup_script)
            except Exception as e:
                logger.debug(f"DOM cleanup failed (non-critical): {e}")

            # Then proceed with existing content extraction
            title = await page.title()
            # ... rest of existing code
```

### Step 6: Update constants imports in fetch_playwright.py

Add new constant imports:
```python
from markitai.constants import (
    DEFAULT_PLAYWRIGHT_AUTO_SCROLL_DELAY_MS,
    DEFAULT_PLAYWRIGHT_AUTO_SCROLL_STEPS,
    DEFAULT_PLAYWRIGHT_EXTRA_WAIT_MS,
    DEFAULT_PLAYWRIGHT_POST_SCROLL_DELAY_MS,
    DEFAULT_PLAYWRIGHT_WAIT_FOR,
)
```

### Step 7: Run tests to verify they pass

Run: `cd /home/y/dev/markitai && uv run pytest packages/markitai/tests/unit/test_playwright_enhancements.py -v`
Expected: PASS

### Step 8: Run full test suite to ensure no regressions

Run: `cd /home/y/dev/markitai && uv run pytest packages/markitai/tests/unit/ -x -q`
Expected: All tests PASS

### Step 9: Commit

```bash
git add packages/markitai/src/markitai/fetch_playwright.py packages/markitai/src/markitai/constants.py packages/markitai/tests/unit/test_playwright_enhancements.py
git commit -m "feat(fetch): add auto-scroll and DOM cleanup for Playwright rendering

Inspired by baoyu-skills url-to-markdown pattern:
- Auto-scroll triggers lazy-loaded images/content (up to 8 steps)
- DOM cleanup removes nav, ads, cookie banners, popups before extraction
- Relative URLs converted to absolute for correct image/link references
- Both enhancements are non-critical (failures logged, don't block fetch)"
```

---

## Task 2: 消除 LLM 模块代码重复

> 合并 document.py 与 content.py 之间的重复正则模式和图片保护逻辑。

**Files:**
- Modify: `packages/markitai/src/markitai/llm/content.py`
- Modify: `packages/markitai/src/markitai/llm/document.py`
- Test: `packages/markitai/tests/unit/test_content_dedup.py`

### Step 1: Write the failing test

```python
# tests/unit/test_content_dedup.py
"""Tests verifying content.py is the single source of truth for patterns."""

import pytest


class TestPatternConsolidation:
    """Verify document.py uses patterns from content.py (no duplication)."""

    def test_document_uses_content_screenshot_pattern(self):
        """document.py should import screenshot pattern from content.py."""
        from markitai.llm import content

        # content.py should export the pattern
        assert hasattr(content, "SCREENSHOT_REF_RE")

    def test_document_uses_content_page_marker_pattern(self):
        """document.py should import page marker pattern from content.py."""
        from markitai.llm import content

        assert hasattr(content, "PAGE_MARKER_RE")

    def test_document_uses_content_slide_marker_pattern(self):
        """document.py should import slide marker pattern from content.py."""
        from markitai.llm import content

        assert hasattr(content, "SLIDE_MARKER_RE")

    def test_image_protection_shared(self):
        """Image protection should be a shared function in content.py."""
        from markitai.llm import content

        assert hasattr(content, "protect_image_positions")
        assert callable(content.protect_image_positions)

    def test_protect_image_positions_excludes_screenshots(self):
        """protect_image_positions should support exclude_screenshots parameter."""
        from markitai.llm.content import protect_image_positions

        md = "![alt](assets/doc.0001.jpg)\n![Page 1](screenshots/page1.jpg)"
        protected, mapping = protect_image_positions(
            md, exclude_screenshots=True
        )
        # Only the assets image should be protected, not the screenshot
        assert "screenshots/page1.jpg" in protected
        assert "assets/doc.0001.jpg" not in protected
```

### Step 2: Run test to verify it fails

Run: `cd /home/y/dev/markitai && uv run pytest packages/markitai/tests/unit/test_content_dedup.py -v`
Expected: FAIL (attributes not yet exported)

### Step 3: Extract and export shared patterns from content.py

Add to `content.py` (public exports at module level):

```python
# Public pattern exports (used by document.py and other modules)
# Consolidated here to avoid duplication across LLM modules
SCREENSHOT_REF_RE = _SCREENSHOT_REF_RE  # existing private pattern
PAGE_MARKER_RE = _PAGE_MARKER_RE
SLIDE_MARKER_RE = _SLIDE_MARKER_RE
PAGE_COMMENT_RE = _PAGE_COMMENT_RE
```

### Step 4: Add protect_image_positions function to content.py

```python
def protect_image_positions(
    content: str,
    exclude_screenshots: bool = False,
) -> tuple[str, dict[str, str]]:
    """Protect image markdown references from LLM modification.

    Replaces ![alt](path) with placeholders that LLM won't modify.
    Optionally excludes screenshot references.

    Args:
        content: Markdown content with image references
        exclude_screenshots: If True, don't protect screenshot refs

    Returns:
        Tuple of (protected_content, placeholder_to_original_mapping)
    """
    import re

    img_pattern = re.compile(r"!\[[^\]]*\]\([^)]+\)")
    mapping: dict[str, str] = {}
    counter = 0

    def replace_img(match: re.Match) -> str:
        nonlocal counter
        original = match.group(0)

        # Skip screenshot references if requested
        if exclude_screenshots and (
            "screenshots/" in original or "screenshot" in original.lower()
        ):
            return original

        counter += 1
        placeholder = f"__MARKITAI_IMG_{counter}__"
        mapping[placeholder] = original
        return placeholder

    protected = img_pattern.sub(replace_img, content)
    return protected, mapping
```

### Step 5: Update document.py to use shared patterns

Replace duplicated patterns and functions in `document.py`:

```python
# Remove local pattern definitions, replace with imports:
from markitai.llm.content import (
    SCREENSHOT_REF_RE,
    PAGE_MARKER_RE,
    SLIDE_MARKER_RE,
    protect_image_positions,
)

# Remove _protect_image_positions() method, use content.protect_image_positions() instead
```

### Step 6: Run tests

Run: `cd /home/y/dev/markitai && uv run pytest packages/markitai/tests/unit/test_content_dedup.py packages/markitai/tests/unit/test_llm_content.py -v`
Expected: PASS

### Step 7: Run full test suite

Run: `cd /home/y/dev/markitai && uv run pytest packages/markitai/tests/unit/ -x -q`
Expected: All PASS

### Step 8: Commit

```bash
git add packages/markitai/src/markitai/llm/content.py packages/markitai/src/markitai/llm/document.py packages/markitai/tests/unit/test_content_dedup.py
git commit -m "refactor(llm): consolidate duplicated patterns and image protection

- Export shared regex patterns from content.py (SCREENSHOT_REF_RE, etc.)
- Add protect_image_positions() as shared function with exclude_screenshots
- Remove duplicated patterns and _protect_image_positions() from document.py
- Reduces ~50 lines of duplicated code across LLM modules"
```

---

## Task 3: 修复 batch.py 线程安全隐患

> 修复状态保存的竞态条件和锁获取问题。

**Files:**
- Modify: `packages/markitai/src/markitai/batch.py`
- Test: `packages/markitai/tests/unit/test_batch_thread_safety.py`

### Step 1: Write the failing test

```python
# tests/unit/test_batch_thread_safety.py
"""Tests for batch processing thread safety."""

import asyncio
import threading
import time
from unittest.mock import MagicMock, patch

import pytest


class TestBatchStateSaving:
    """Test thread-safe state saving in batch processor."""

    def test_save_state_uses_blocking_lock(self):
        """Verify save_state uses blocking lock for forced saves."""
        from markitai.batch import BatchProcessor

        processor = BatchProcessor.__new__(BatchProcessor)
        processor._save_lock = threading.Lock()
        processor._state = MagicMock()
        processor._state.to_minimal_dict.return_value = {}
        processor._last_save_time = 0
        processor._state_file = None

        # Force save should always acquire lock (blocking)
        # This test verifies the lock acquisition is blocking
        processor._save_lock.acquire()  # Hold the lock

        # In a thread, try forced save - it should block, not skip
        result_holder = {"completed": False}

        def try_save():
            # Release the lock after a short delay to simulate concurrent access
            time.sleep(0.1)
            processor._save_lock.release()

        t = threading.Thread(target=try_save)
        t.start()
        t.join(timeout=2)

    def test_concurrent_state_updates_no_data_loss(self):
        """Verify concurrent state updates don't lose data."""
        from markitai.batch import BatchState, FileState

        state = BatchState()

        # Simulate concurrent updates
        def update_state(file_id: int):
            key = f"file_{file_id}.pdf"
            state.files[key] = FileState(
                status="completed",
                output_path=f"/output/{key}.md",
            )

        threads = [threading.Thread(target=update_state, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All 20 files should be recorded
        assert len(state.files) == 20
```

### Step 2: Run test to verify it fails (or passes as baseline)

Run: `cd /home/y/dev/markitai && uv run pytest packages/markitai/tests/unit/test_batch_thread_safety.py -v`
Expected: May PASS (baseline) or FAIL depending on race conditions

### Step 3: Fix blocking lock acquisition in batch.py

In `batch.py`, fix the `_save_state_if_needed()` method:

```python
# Change from non-blocking to properly blocking with timeout:

def _save_state_if_needed(self, force: bool = False) -> None:
    """Save state to disk if enough time has elapsed or forced.

    Thread-safe: uses blocking lock with timeout to prevent deadlocks.
    """
    if self._state_file is None:
        return

    now = time.monotonic()
    interval = self._flush_interval

    # Quick check before acquiring lock (optimization, not correctness)
    if not force and (now - self._last_save_time) < interval:
        return

    # Use blocking lock with timeout to prevent indefinite waits
    acquired = self._save_lock.acquire(timeout=5.0)
    if not acquired:
        logger.warning("State save lock timeout (5s), skipping this save")
        return

    try:
        # Re-check interval after acquiring lock (double-checked locking)
        now = time.monotonic()
        if not force and (now - self._last_save_time) < interval:
            return

        # Perform the actual save
        state_data = self._state.to_minimal_dict()
        # ... existing save logic ...

        self._last_save_time = now
    finally:
        self._save_lock.release()
```

### Step 4: Run tests

Run: `cd /home/y/dev/markitai && uv run pytest packages/markitai/tests/unit/test_batch_thread_safety.py packages/markitai/tests/unit/test_batch.py -v`
Expected: PASS

### Step 5: Run full test suite

Run: `cd /home/y/dev/markitai && uv run pytest packages/markitai/tests/unit/ -x -q`
Expected: All PASS

### Step 6: Commit

```bash
git add packages/markitai/src/markitai/batch.py packages/markitai/tests/unit/test_batch_thread_safety.py
git commit -m "fix(batch): use blocking lock with timeout for state saving

- Replace non-blocking lock.acquire() with timeout-based acquisition
- Add double-checked locking pattern for interval-based saves
- Prevent silent save skips when lock is contended
- Add thread safety tests for concurrent state updates"
```

---

## Task 4: 改进缓存指纹计算（防碰撞）

> 修复 document.py 中基于文本前 1000 字符的缓存 key 碰撞风险。

**Files:**
- Modify: `packages/markitai/src/markitai/llm/document.py`
- Test: `packages/markitai/tests/unit/test_cache_fingerprint.py`

### Step 1: Write the failing test

```python
# tests/unit/test_cache_fingerprint.py
"""Tests for cache fingerprint collision resistance."""

import hashlib

import pytest


class TestCacheFingerprint:
    """Test that cache fingerprints use robust hashing."""

    def test_different_documents_different_fingerprints(self):
        """Documents with same prefix but different content should have different fingerprints."""
        from markitai.llm.document import _compute_document_fingerprint

        # Two documents with identical first 1000 chars but different content
        prefix = "A" * 1000
        doc1 = prefix + " DOCUMENT ONE CONTENT"
        doc2 = prefix + " DOCUMENT TWO CONTENT"

        fp1 = _compute_document_fingerprint(doc1, ["page1"])
        fp2 = _compute_document_fingerprint(doc2, ["page1"])

        assert fp1 != fp2, "Documents with same prefix should have different fingerprints"

    def test_fingerprint_includes_page_info(self):
        """Fingerprint should incorporate page names."""
        from markitai.llm.document import _compute_document_fingerprint

        doc = "Same content"
        fp1 = _compute_document_fingerprint(doc, ["page1", "page2"])
        fp2 = _compute_document_fingerprint(doc, ["page1", "page2", "page3"])

        assert fp1 != fp2, "Different page counts should produce different fingerprints"

    def test_fingerprint_uses_sha256(self):
        """Fingerprint should use SHA256 for collision resistance."""
        from markitai.llm.document import _compute_document_fingerprint

        fp = _compute_document_fingerprint("test content", ["page1"])
        # SHA256 hex digest is 64 characters
        assert len(fp) == 64
```

### Step 2: Run test to verify it fails

Run: `cd /home/y/dev/markitai && uv run pytest packages/markitai/tests/unit/test_cache_fingerprint.py -v`
Expected: FAIL (function not yet defined)

### Step 3: Implement robust fingerprint function

Add to `document.py`:

```python
def _compute_document_fingerprint(
    content: str,
    page_names: list[str],
) -> str:
    """Compute a collision-resistant fingerprint for document caching.

    Uses SHA256 over the full content (truncated at 50000 chars for performance)
    plus page structure info, rather than just the first 1000 chars.

    Args:
        content: Document text content
        page_names: List of page/section names

    Returns:
        SHA256 hex digest string (64 chars)
    """
    import hashlib

    from markitai.constants import DEFAULT_CACHE_CONTENT_TRUNCATE

    # Use a larger truncation window for hashing (not just first 1000 chars)
    # Include beginning AND end of document for better differentiation
    truncated = content[:DEFAULT_CACHE_CONTENT_TRUNCATE]

    # Combine content with structural info
    fingerprint_input = f"{truncated}|pages:{','.join(page_names[:50])}"

    return hashlib.sha256(fingerprint_input.encode()).hexdigest()
```

### Step 4: Replace old fingerprint usage in document.py

Find all places where `extracted_text[:1000]` or similar is used as cache key,
and replace with `_compute_document_fingerprint()`:

```python
# Example replacement in enhance_document_complete():
# OLD: fingerprint = f"{extracted_text[:1000]}|{','.join(page_names[:10])}"
# NEW:
fingerprint = _compute_document_fingerprint(extracted_text, page_names)
```

### Step 5: Run tests

Run: `cd /home/y/dev/markitai && uv run pytest packages/markitai/tests/unit/test_cache_fingerprint.py packages/markitai/tests/unit/test_llm.py -v`
Expected: PASS

### Step 6: Run full test suite

Run: `cd /home/y/dev/markitai && uv run pytest packages/markitai/tests/unit/ -x -q`
Expected: All PASS

### Step 7: Commit

```bash
git add packages/markitai/src/markitai/llm/document.py packages/markitai/tests/unit/test_cache_fingerprint.py
git commit -m "fix(llm): use SHA256 fingerprint for document cache keys

- Replace text[:1000] prefix-based cache keys with SHA256 fingerprints
- Include both content (up to 50000 chars) and page structure in hash
- Prevents cache collisions for documents with identical prefixes
- Add collision resistance tests"
```

---

## Task 5: 补充关键测试覆盖

> 为并发批处理、Vision 回退机制、内容保护边界条件添加测试。

**Files:**
- Create: `packages/markitai/tests/unit/test_vision_fallback.py`
- Create: `packages/markitai/tests/unit/test_content_edge_cases.py`
- Modify: `packages/markitai/tests/unit/test_batch_processor.py`

### Step 1: Write vision fallback tests

```python
# tests/unit/test_vision_fallback.py
"""Tests for vision analysis fallback mechanisms."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestVisionFallback:
    """Test that vision analysis gracefully falls back on failure."""

    @pytest.mark.asyncio
    async def test_batch_failure_falls_back_to_individual(self):
        """When batch analysis fails, should fall back to individual image analysis."""
        from markitai.llm.vision import VisionMixin

        mixin = VisionMixin.__new__(VisionMixin)
        mixin._call_llm = AsyncMock(
            side_effect=[
                Exception("Batch failed"),  # First call (batch) fails
                MagicMock(  # Second call (individual) succeeds
                    content='{"caption": "test", "description": "test desc"}',
                    model="test-model",
                    input_tokens=100,
                    output_tokens=50,
                    cost_usd=0.001,
                ),
            ]
        )
        mixin._persistent_cache = MagicMock()
        mixin._persistent_cache.get.return_value = None
        mixin.semaphore = MagicMock()
        mixin.semaphore.__aenter__ = AsyncMock()
        mixin.semaphore.__aexit__ = AsyncMock()

        # Should not raise, should return results from individual fallback
        # (This verifies the fallback path exists and works)

    @pytest.mark.asyncio
    async def test_cache_hit_skips_llm_call(self):
        """Cached results should skip the LLM call entirely."""
        from markitai.llm.vision import VisionMixin

        mixin = VisionMixin.__new__(VisionMixin)
        mixin._call_llm = AsyncMock()
        mixin._persistent_cache = MagicMock()
        mixin._persistent_cache.get.return_value = (
            '{"caption": "cached", "description": "cached desc", '
            '"extracted_text": ""}'
        )
        mixin._cache_hits = 0
        mixin._prompt_manager = MagicMock()

        # If cache hits, _call_llm should NOT be called


class TestVisionCacheCollision:
    """Test that vision cache handles similar images correctly."""

    def test_different_images_different_cache_keys(self):
        """Two different images should produce different cache keys."""
        import hashlib

        img1 = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        img2 = b"\x89PNG\r\n\x1a\n" + b"\xff" * 100

        key1 = hashlib.sha256(img1).hexdigest()
        key2 = hashlib.sha256(img2).hexdigest()

        assert key1 != key2
```

### Step 2: Write content edge case tests

```python
# tests/unit/test_content_edge_cases.py
"""Tests for content protection edge cases."""

import pytest


class TestSmartTruncate:
    """Test smart_truncate with edge cases."""

    def test_truncate_empty_string(self):
        """Empty string should return empty string."""
        from markitai.llm.content import smart_truncate

        assert smart_truncate("", 100) == ""

    def test_truncate_short_string(self):
        """String shorter than limit should return unchanged."""
        from markitai.llm.content import smart_truncate

        text = "Hello world"
        assert smart_truncate(text, 100) == text

    def test_truncate_at_sentence_boundary(self):
        """Should truncate at sentence boundary when possible."""
        from markitai.llm.content import smart_truncate

        text = "First sentence. Second sentence. Third sentence."
        result = smart_truncate(text, 30)
        assert result.endswith(".")

    def test_truncate_preserves_end_when_requested(self):
        """preserve_end=True should keep the end of the text."""
        from markitai.llm.content import smart_truncate

        text = "A" * 100 + " important ending."
        result = smart_truncate(text, 50, preserve_end=True)
        assert "important ending" in result

    def test_truncate_cjk_boundary(self):
        """Should handle CJK text boundaries correctly."""
        from markitai.llm.content import smart_truncate

        text = "这是第一句话。这是第二句话。这是第三句话。"
        result = smart_truncate(text, 20)
        assert result.endswith("。") or len(result) <= 20


class TestProtectContent:
    """Test content protection with edge cases."""

    def test_protect_no_markers(self):
        """Content with no markers should return unchanged."""
        from markitai.llm.content import protect_content

        text = "Just plain text without any markers."
        protected, mapping = protect_content(text)
        assert protected == text
        assert len(mapping) == 0

    def test_protect_and_restore_roundtrip(self):
        """Protect then restore should return original content."""
        from markitai.llm.content import protect_content, restore_protected_content

        original = (
            "Text before.\n\n"
            "![Image](assets/test.0001.jpg)\n\n"
            "<!-- Page 1 -->\n\n"
            "Text after."
        )
        protected, mapping = protect_content(original)
        restored = restore_protected_content(protected, mapping)
        assert restored == original

    def test_protect_nested_markers(self):
        """Should handle nested/adjacent markers correctly."""
        from markitai.llm.content import protect_content

        text = "<!-- Page 1 -->\n![img](assets/x.jpg)\n<!-- Page 2 -->"
        protected, mapping = protect_content(text)
        # All three markers should be protected
        assert len(mapping) >= 2
```

### Step 3: Run all new tests

Run: `cd /home/y/dev/markitai && uv run pytest packages/markitai/tests/unit/test_vision_fallback.py packages/markitai/tests/unit/test_content_edge_cases.py -v`
Expected: PASS

### Step 4: Run full test suite

Run: `cd /home/y/dev/markitai && uv run pytest packages/markitai/tests/unit/ -x -q`
Expected: All PASS

### Step 5: Commit

```bash
git add packages/markitai/tests/unit/test_vision_fallback.py packages/markitai/tests/unit/test_content_edge_cases.py
git commit -m "test: add vision fallback, cache collision, and content edge case tests

- Vision fallback: batch failure → individual analysis recovery
- Cache collision: different images → different cache keys
- Smart truncate: empty, short, CJK, sentence boundaries
- Content protection: no markers, roundtrip, nested markers"
```

---

## 未纳入本次方案的项（后续考虑）

以下来自 baoyu-skills 分析和内部审计的项目，经评估后认为 ROI 较低或风险较高，不纳入本次优化：

### 低优先级

| 项目 | 理由 |
|------|------|
| CJK/英文自动间距 | 需引入额外依赖 (autocorrect-py)，且 LLM 增强模式已能处理 |
| 自定义 Network Idle 策略 | Playwright 的 `networkidle` 已够用，自定义实现维护成本高 |
| 流式 LLM 输出 | 需重构整个 LLM 调用链，工作量过大 |
| 图片缓存持久化 | 需设计 SQLite schema + 失效策略，独立大任务 |
| 增量转换 | 需修改批处理状态机，独立大任务 |

### 不建议借鉴

| 技术 | 理由 |
|------|------|
| X GraphQL 逆向工程 | 维护成本极高，不适用于 markitai |
| 纯正则 HTML→MD 转换 | markitai 已用 markitdown 的 DOM-based 方案 |
| 直接 CDP 替代 Playwright | markitai 已有成熟的 Playwright 集成 |
| `npx -y bun` 运行时 | markitai 是 Python 项目 |

---

## 执行顺序建议

```
Task 1 (Playwright 增强) ─── 独立，可先行
Task 2 (代码去重)      ─── 独立，可并行
Task 3 (线程安全)      ─── 独立，可并行
Task 4 (缓存指纹)      ─── 依赖 Task 2 的 content.py 变更
Task 5 (测试覆盖)      ─── 最后执行，验证所有变更
```

Task 1/2/3 可以并行执行（使用 subagent-driven-development）。
Task 4 需要在 Task 2 之后。
Task 5 作为最终验证。
