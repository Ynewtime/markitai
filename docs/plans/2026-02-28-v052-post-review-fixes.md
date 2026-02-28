# v0.5.2 Post-Review Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all 4 Important issues and 5 Suggestions identified in the v0.5.2+ code review.

**Architecture:** Pure refactoring — no new features. Extract helpers to reduce duplication, fix lifecycle/naming bugs, add missing test coverage. All changes are internal, no public API changes.

**Tech Stack:** Python 3.10+, pytest, asyncio, httpx, Pydantic

---

### Task 1: Lazy-init `_cf_br_semaphore` (Important #1)

**Files:**
- Modify: `packages/markitai/src/markitai/fetch.py:699-701`
- Test: `packages/markitai/tests/unit/test_fetch.py`

**Step 1: Write the failing test**

```python
# In test_fetch.py — new test class
class TestCfBrSemaphore:
    """Tests for CF BR semaphore lazy initialization."""

    def test_get_cf_semaphore_returns_semaphore(self):
        """get_cf_semaphore returns an asyncio.Semaphore."""
        from markitai.fetch import get_cf_semaphore

        sem = get_cf_semaphore()
        assert isinstance(sem, asyncio.Semaphore)

    def test_get_cf_semaphore_returns_same_instance(self):
        """get_cf_semaphore returns the same instance on repeated calls."""
        from markitai.fetch import get_cf_semaphore

        sem1 = get_cf_semaphore()
        sem2 = get_cf_semaphore()
        assert sem1 is sem2
```

**Step 2: Run test to verify it fails**

Run: `cd packages/markitai && python -m pytest tests/unit/test_fetch.py::TestCfBrSemaphore -v`
Expected: FAIL with "cannot import name 'get_cf_semaphore'"

**Step 3: Implement lazy semaphore**

Replace in `fetch.py`:

```python
# OLD (line 701):
_cf_br_semaphore = asyncio.Semaphore(2)

# NEW:
_cf_br_semaphore: asyncio.Semaphore | None = None


def get_cf_semaphore() -> asyncio.Semaphore:
    """Get or create the CF BR rate-limiting semaphore.

    Lazily initialized to avoid binding to a wrong event loop at import time.
    CF Free plan allows 2 concurrent browser instances.
    """
    global _cf_br_semaphore
    if _cf_br_semaphore is None:
        _cf_br_semaphore = asyncio.Semaphore(2)
    return _cf_br_semaphore
```

Then replace `_cf_br_semaphore` usage at line ~1773:

```python
# OLD:
async with _cf_br_semaphore:

# NEW:
async with get_cf_semaphore():
```

**Step 4: Run test to verify it passes**

Run: `cd packages/markitai && python -m pytest tests/unit/test_fetch.py::TestCfBrSemaphore -v`
Expected: PASS

**Step 5: Run full test suite to verify no regressions**

Run: `cd packages/markitai && python -m pytest tests/unit/test_fetch.py -v`
Expected: All PASS

---

### Task 2: `hasattr` → `isinstance` for `convert_async` dispatch (Important #2)

**Files:**
- Modify: `packages/markitai/src/markitai/workflow/core.py:239-244`

**Step 1: Write the failing test**

```python
# In tests/unit/test_workflow_core.py — new test
class TestConvertAsyncDispatch:
    """Tests for async converter dispatch in workflow."""

    @pytest.mark.asyncio
    async def test_cloudflare_converter_uses_convert_async(self):
        """CloudflareConverter dispatches via convert_async."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from markitai.converter.cloudflare import CloudflareConverter

        converter = MagicMock(spec=CloudflareConverter)
        converter.convert_async = AsyncMock(return_value=MagicMock())

        # Verify isinstance check works
        assert isinstance(converter, CloudflareConverter)
```

Actually, this is a straightforward refactor that doesn't need a new test — the existing tests (if any) for workflow/core.py already exercise the dispatch path. This is a 1-line logic change.

**Step 1: Apply the fix**

In `workflow/core.py`, replace:

```python
# OLD (line 239-244):
# Check if converter supports async (e.g. CloudflareConverter)
if hasattr(ctx.converter, "convert_async"):
    ctx.conversion_result = await ctx.converter.convert_async(

# NEW:
from markitai.converter.cloudflare import CloudflareConverter

# CloudflareConverter has a native async API — dispatch directly
if isinstance(ctx.converter, CloudflareConverter):
    ctx.conversion_result = await ctx.converter.convert_async(
```

Note: The import should be inside the function or use `TYPE_CHECKING` to avoid circular imports. Check if `CloudflareConverter` can be imported at module level — if it causes a circular import, use a local import.

**Step 2: Run full test suite**

Run: `cd packages/markitai && python -m pytest tests/unit/test_workflow_core.py tests/unit/test_converter_cloudflare.py -v`
Expected: All PASS

---

### Task 3: Fix `_resolve_api_key` naming for account_id (Important #3)

**Files:**
- Modify: `packages/markitai/src/markitai/config.py:405-406`
- Test: existing `tests/unit/test_config.py` should still pass

**Step 1: Apply the fix**

```python
# OLD (line 405-406):
    if self.account_id:
        return _resolve_api_key(self.account_id, strict=strict)

# NEW:
    if self.account_id:
        return resolve_env_value(self.account_id, strict=strict)
```

`resolve_env_value` is already a public function in the same module (line 85). Since the caller already checks `if self.account_id:` (truthiness), the `_resolve_api_key` None guard is redundant. Direct call to `resolve_env_value` is both semantically correct and functionally equivalent.

**Step 2: Run tests**

Run: `cd packages/markitai && python -m pytest tests/unit/test_config.py -v`
Expected: All PASS

---

### Task 4: Move httpx client outside retry loop (Important #4)

**Files:**
- Modify: `packages/markitai/src/markitai/fetch.py:1773-1800`

**Step 1: Apply the fix**

```python
# OLD (lines 1773-1796):
        async with _cf_br_semaphore:
            for attempt in range(max_retries):
                async with httpx.AsyncClient(
                    timeout=max(timeout / 1000 + 10, 60.0),
                    proxy=proxy_config,
                ) as client:
                    response = await client.post(...)
                    if response.status_code == 429:
                        ...
                        continue

# NEW:
        async with get_cf_semaphore():
            async with httpx.AsyncClient(
                timeout=max(timeout / 1000 + 10, 60.0),
                proxy=proxy_config,
            ) as client:
                for attempt in range(max_retries):
                    response = await client.post(...)
                    if response.status_code == 429:
                        ...
                        continue
```

Note: This task also applies the `get_cf_semaphore()` from Task 1. The indentation of everything inside the retry loop shifts left by one level.

**Step 2: Run tests**

Run: `cd packages/markitai && python -m pytest tests/unit/test_fetch.py -k cloudflare -v`
Expected: All PASS

---

### Task 5: Extract Playwright advanced kwargs helper (Suggestion #5)

**Files:**
- Modify: `packages/markitai/src/markitai/fetch.py` (4 call sites)
- Test: `packages/markitai/tests/unit/test_fetch.py`

**Step 1: Write the test**

```python
class TestPlaywrightAdvancedKwargs:
    """Tests for _get_playwright_advanced_kwargs helper."""

    def test_returns_empty_when_all_none(self):
        from markitai.config import PlaywrightConfig
        from markitai.fetch import _get_playwright_advanced_kwargs

        pw = PlaywrightConfig()
        result = _get_playwright_advanced_kwargs(pw)
        assert result == {}

    def test_returns_set_values_only(self):
        from markitai.config import PlaywrightConfig
        from markitai.fetch import _get_playwright_advanced_kwargs

        pw = PlaywrightConfig(
            wait_for_selector="#main",
            user_agent="TestBot/1.0",
        )
        result = _get_playwright_advanced_kwargs(pw)
        assert result == {
            "wait_for_selector": "#main",
            "user_agent": "TestBot/1.0",
        }

    def test_returns_all_values_when_set(self):
        from markitai.config import PlaywrightConfig
        from markitai.fetch import _get_playwright_advanced_kwargs

        pw = PlaywrightConfig(
            wait_for_selector="div.content",
            cookies=[{"name": "sid", "value": "abc", "domain": ".example.com"}],
            reject_resource_patterns=["**/*.css"],
            extra_http_headers={"Accept-Language": "zh-CN"},
            user_agent="Bot/2.0",
            http_credentials={"username": "u", "password": "p"},
        )
        result = _get_playwright_advanced_kwargs(pw)
        assert len(result) == 6
        assert result["wait_for_selector"] == "div.content"
```

**Step 2: Run test to verify it fails**

Run: `cd packages/markitai && python -m pytest tests/unit/test_fetch.py::TestPlaywrightAdvancedKwargs -v`
Expected: FAIL

**Step 3: Implement the helper**

Add near the top of `fetch.py` (after imports/before functions):

```python
def _get_playwright_advanced_kwargs(pw: PlaywrightConfig) -> dict[str, Any]:
    """Extract advanced Playwright kwargs from config, omitting None values."""
    return {
        k: v
        for k, v in {
            "wait_for_selector": pw.wait_for_selector,
            "cookies": pw.cookies,
            "reject_resource_patterns": pw.reject_resource_patterns,
            "extra_http_headers": pw.extra_http_headers,
            "user_agent": pw.user_agent,
            "http_credentials": pw.http_credentials,
        }.items()
        if v is not None
    }
```

Then replace the 4 call sites (lines ~2091-2101, ~2197-2204, ~2340-2349, ~2623-2636) with:

```python
# OLD (6 lines of getattr):
wait_for_selector=getattr(config.playwright, "wait_for_selector", None),
cookies=getattr(config.playwright, "cookies", None),
reject_resource_patterns=getattr(config.playwright, "reject_resource_patterns", None),
extra_http_headers=getattr(config.playwright, "extra_http_headers", None),
user_agent=getattr(config.playwright, "user_agent", None),
http_credentials=getattr(config.playwright, "http_credentials", None),

# NEW (1 line):
**_get_playwright_advanced_kwargs(config.playwright),
```

**Step 4: Run tests**

Run: `cd packages/markitai && python -m pytest tests/unit/test_fetch.py tests/unit/test_fetch_playwright.py -v`
Expected: All PASS

---

### Task 6: Move `aiofiles` import to top level in cloudflare.py (Suggestion #6)

**Files:**
- Modify: `packages/markitai/src/markitai/converter/cloudflare.py:10,151`

**Step 1: Apply the fix**

Move `import aiofiles` from inside `convert_async()` (line 151) to the top-level imports (after line 14):

```python
# At top of file, add after existing imports:
import aiofiles

# Remove the lazy import at line 151:
# (delete) import aiofiles
```

**Step 2: Run tests**

Run: `cd packages/markitai && python -m pytest tests/unit/test_converter_cloudflare.py -v`
Expected: All PASS

---

### Task 7: Extract title regex helper (Suggestion #7)

**Files:**
- Modify: `packages/markitai/src/markitai/fetch.py` (3 call sites: lines ~1615, ~1814, ~2526)
- Test: `packages/markitai/tests/unit/test_fetch.py`

**Step 1: Write the test**

```python
class TestExtractMarkdownTitle:
    """Tests for _extract_markdown_title helper."""

    def test_extracts_h1_title(self):
        from markitai.fetch import _extract_markdown_title

        assert _extract_markdown_title("# Hello World\n\nContent") == "Hello World"

    def test_returns_none_for_no_heading(self):
        from markitai.fetch import _extract_markdown_title

        assert _extract_markdown_title("No heading here") is None

    def test_returns_none_for_empty_string(self):
        from markitai.fetch import _extract_markdown_title

        assert _extract_markdown_title("") is None

    def test_extracts_first_h1_only(self):
        from markitai.fetch import _extract_markdown_title

        content = "Some text\n# First Title\n## Sub\n# Second"
        assert _extract_markdown_title(content) == "First Title"
```

**Step 2: Run test to verify it fails**

Run: `cd packages/markitai && python -m pytest tests/unit/test_fetch.py::TestExtractMarkdownTitle -v`
Expected: FAIL

**Step 3: Implement the helper**

```python
def _extract_markdown_title(content: str) -> str | None:
    """Extract the first H1 title from markdown content."""
    match = re.match(r"^#\s+(.+)$", content, re.MULTILINE)
    return match.group(1) if match else None
```

Then replace 3 call sites:

```python
# OLD (3 occurrences):
title_match = re.match(r"^#\s+(.+)$", markdown_content, re.MULTILINE)
if title_match:
    title = title_match.group(1)

# NEW:
title = _extract_markdown_title(markdown_content)
```

**Step 4: Run tests**

Run: `cd packages/markitai && python -m pytest tests/unit/test_fetch.py -v`
Expected: All PASS

---

### Task 8: Add 429 retry test for `fetch_with_cloudflare` (Suggestion #8)

**Files:**
- Modify: `packages/markitai/tests/unit/test_fetch.py`

**Step 1: Write the tests**

```python
# In TestCloudflareStrategy class (or new class TestCloudflareBRRetry)

@pytest.mark.asyncio
async def test_fetch_with_cloudflare_429_retry_succeeds(self):
    """CF BR retries on 429 and succeeds on next attempt."""
    from markitai.fetch import fetch_with_cloudflare

    # First response: 429 rate limited
    mock_429 = MagicMock()
    mock_429.status_code = 429
    mock_429.headers = {"Retry-After": "1"}

    # Second response: 200 success
    mock_200 = MagicMock()
    mock_200.status_code = 200
    mock_200.json.return_value = {
        "success": True,
        "result": "# Retry OK\n\nContent after retry.",
        "errors": [],
        "messages": [],
    }
    mock_200.headers = {"X-Browser-Ms-Used": "2000"}
    mock_200.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=[mock_429, mock_200])

    with (
        patch("markitai.fetch._detect_proxy", return_value=""),
        patch("httpx.AsyncClient") as mock_client_class,
        patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = mock_client
        mock_ctx.__aexit__.return_value = None
        mock_client_class.return_value = mock_ctx

        result = await fetch_with_cloudflare(
            url="https://example.com",
            api_token="test-token",
            account_id="test-account",
        )

        assert result.content == "# Retry OK\n\nContent after retry."
        assert result.strategy_used == "cloudflare"
        mock_sleep.assert_called_once()  # Slept between retries

@pytest.mark.asyncio
async def test_fetch_with_cloudflare_429_exhausted(self):
    """CF BR raises FetchError after exhausting all retries on 429."""
    from markitai.fetch import FetchError, fetch_with_cloudflare

    mock_429 = MagicMock()
    mock_429.status_code = 429
    mock_429.headers = {"Retry-After": "1"}

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_429)

    with (
        patch("markitai.fetch._detect_proxy", return_value=""),
        patch("httpx.AsyncClient") as mock_client_class,
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = mock_client
        mock_ctx.__aexit__.return_value = None
        mock_client_class.return_value = mock_ctx

        with pytest.raises(FetchError, match="rate limit exceeded"):
            await fetch_with_cloudflare(
                url="https://example.com",
                api_token="test-token",
                account_id="test-account",
            )
```

**Step 2: Run tests**

Run: `cd packages/markitai && python -m pytest tests/unit/test_fetch.py -k "429" -v`
Expected: All PASS (after Task 1 and Task 4 changes are applied)

---

### Task 9: Add changelog note for `_fetch_multi_source` behavior change (Suggestion #9)

**Files:**
- Modify: `packages/markitai/src/markitai/fetch.py` (~line 2445)

**Step 1: Add clarifying comment**

```python
# Before the raise FetchError block, enhance the existing comment:

# OLD:
        # All strategies produced invalid content → fail
        raise FetchError(

# NEW:
        # All strategies produced invalid content → fail loudly.
        # Changed in v0.5.3: previously returned degraded browser content
        # with a warning; now raises FetchError after exhausting Jina fallback.
        raise FetchError(
```

**Step 2: Verify**

Run: `cd packages/markitai && python -m pytest tests/unit/test_fetch.py -v`
Expected: All PASS (no behavior change, just comment)

---

## Execution Order

Tasks 1–4 are independent Important fixes.
Tasks 5–7 are independent refactoring (helpers).
Task 8 (tests) depends on Tasks 1 + 4 (semaphore + retry loop changes).
Task 9 is standalone.

Recommended sequence: 1 → 4 → 2 → 3 → 5 → 6 → 7 → 8 → 9

After all tasks, run full suite: `cd packages/markitai && python -m pytest tests/unit/ -v`
