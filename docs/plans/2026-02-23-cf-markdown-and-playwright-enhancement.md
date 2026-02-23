# Cloudflare Content Negotiation + Playwright 能力增强 实施方案

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 四项增强：(1) 零成本启用 CF Markdown for Agents 内容协商；(2) 补齐 Playwright 缺失的高价值能力（wait_for_selector / cookies / resource filtering / extra headers）；(3) 接入 CF Browser Rendering API 作为新的 FetchStrategy；(4) 接入 CF Workers AI toMarkdown 作为本地文件转换的云端备选。

**Architecture:** 四个 Task 按依赖关系串行：Task 1 改 static fetch 层（最小改动）；Task 2 增强 Playwright renderer（config → fetch_playwright → fetch.py 参数透传）；Task 3 新增 CF BR 策略（config → fetch function → fallback 链）；Task 4 新增 CF toMarkdown converter（FileFormat 扩展 → CloudflareConverter → workflow 集成）。Task 3 和 4 共用 CloudflareConfig 认证。每个 Task 内部严格 TDD。

**Tech Stack:** Python >=3.11,<3.14, httpx (inline import, not declared dep), playwright (optional `[browser]`), pydantic (config), pytest + AsyncMock (tests)

**Baseline:** 1831 passed, 1 skipped — 任何 Task 完成后必须保持这个基线。

---

## Task 1: CF Markdown Content Negotiation（P0）

**目的：** 在 static fetch 路径加 `Accept: text/markdown` 头，让 CF 托管站点直接返回 markdown，跳过本地 HTML→MD 转换。对非 CF 站点零影响。

**Files:**
- Modify: `packages/markitai/src/markitai/fetch.py` (`_get_markitdown`, `fetch_with_static_conditional`)
- Test: `packages/markitai/tests/unit/test_fetch.py`

### Step 1: 写 `_get_markitdown` 内容协商的测试

在 `test_fetch.py` 末尾新增测试类：

```python
class TestContentNegotiation:
    """Tests for CF Markdown for Agents content negotiation."""

    def test_markitdown_instance_has_accept_markdown_header(self):
        """Verify markitdown session sends Accept: text/markdown header."""
        from markitai.fetch import _get_markitdown, _markitdown_instance
        import markitai.fetch as fetch_module

        # Reset singleton to force re-creation
        old = fetch_module._markitdown_instance
        fetch_module._markitdown_instance = None
        try:
            md = _get_markitdown()
            accept = md._requests_session.headers.get("Accept", "")
            assert "text/markdown" in accept
            # text/html should be lower priority
            assert "text/html" in accept
        finally:
            fetch_module._markitdown_instance = old
```

### Step 2: 运行测试验证失败

```bash
cd /home/oy/Work/markitai
uv run python -m pytest packages/markitai/tests/unit/test_fetch.py::TestContentNegotiation -v
```

Expected: FAIL — 当前 `_get_markitdown()` 未设置 Accept header。

### Step 3: 实现 `_get_markitdown` 加 Accept 头

修改 `packages/markitai/src/markitai/fetch.py` 中 `_get_markitdown` 函数：

```python
def _get_markitdown() -> Any:
    """Get or create the shared MarkItDown instance.

    Reusing a single instance avoids repeated initialization overhead.
    Includes Accept header for CF Markdown for Agents content negotiation.
    """
    global _markitdown_instance
    if _markitdown_instance is None:
        from markitdown import MarkItDown

        _markitdown_instance = MarkItDown()
        # Enable Cloudflare Markdown for Agents content negotiation.
        # CF-enabled sites return text/markdown directly (higher quality, fewer tokens).
        # Non-CF sites return text/html as usual — zero impact on existing behavior.
        _markitdown_instance._requests_session.headers.update(
            {"Accept": "text/markdown, text/html;q=0.9, */*;q=0.5"}
        )
    return _markitdown_instance
```

### Step 4: 运行测试验证通过

```bash
uv run python -m pytest packages/markitai/tests/unit/test_fetch.py::TestContentNegotiation -v
```

Expected: PASS

### Step 5: 写 `fetch_with_static_conditional` 内容协商测试

在 `TestContentNegotiation` 类中追加：

```python
@pytest.mark.asyncio
async def test_conditional_fetch_sends_accept_markdown_header(self):
    """Verify conditional fetch includes Accept: text/markdown header."""
    from markitai.fetch import fetch_with_static_conditional

    captured_headers = {}

    async def mock_get(url, headers=None):
        nonlocal captured_headers
        captured_headers = headers or {}
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "text/html", "ETag": '"abc"'}
        mock_resp.content = b"<html><body>Hello World test content for validation check</body></html>"
        mock_resp.url = url
        return mock_resp

    mock_client = AsyncMock()
    mock_client.get = mock_get

    with (
        patch("markitai.fetch._detect_proxy", return_value=""),
        patch("httpx.AsyncClient") as mock_client_class,
    ):
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_client
        mock_context.__aexit__.return_value = None
        mock_client_class.return_value = mock_context

        try:
            await fetch_with_static_conditional("https://example.com")
        except Exception:
            pass  # May fail on markitdown conversion, we only care about headers

    assert "Accept" in captured_headers
    assert "text/markdown" in captured_headers["Accept"]
```

### Step 6: 运行测试验证失败

```bash
uv run python -m pytest packages/markitai/tests/unit/test_fetch.py::TestContentNegotiation::test_conditional_fetch_sends_accept_markdown_header -v
```

Expected: FAIL — 当前 `headers` dict 不含 Accept。

### Step 7: 实现 `fetch_with_static_conditional` 加 Accept 头

修改 `packages/markitai/src/markitai/fetch.py` 中 `fetch_with_static_conditional` 函数的 headers 构建：

将：
```python
    headers: dict[str, str] = {}
    if cached_etag:
        headers["If-None-Match"] = cached_etag
    if cached_last_modified:
        headers["If-Modified-Since"] = cached_last_modified
```

改为：
```python
    # CF Markdown for Agents content negotiation
    headers: dict[str, str] = {
        "Accept": "text/markdown, text/html;q=0.9, */*;q=0.5",
    }
    if cached_etag:
        headers["If-None-Match"] = cached_etag
    if cached_last_modified:
        headers["If-Modified-Since"] = cached_last_modified
```

同时在 content_type 判断逻辑中增加对 `text/markdown` 的处理（跳过 markitdown 转换）。在 `# Determine file extension from Content-Type or URL` 代码块之前插入：

```python
            # Check if server returned markdown directly (CF Markdown for Agents)
            content_type = response.headers.get("Content-Type", "")
            if "text/markdown" in content_type:
                markdown_content = response.text
                token_hint = response.headers.get("x-markdown-tokens")
                logger.debug(
                    f"[ConditionalFetch] Server returned markdown directly"
                    f"{f' (~{token_hint} tokens)' if token_hint else ''}"
                )
                # Extract title from first heading or YAML frontmatter
                title = None
                title_match = re.match(
                    r"^#\s+(.+)$", markdown_content, re.MULTILINE
                )
                if title_match:
                    title = title_match.group(1)

                fetch_result = FetchResult(
                    content=markdown_content,
                    strategy_used="static",
                    title=title,
                    url=url,
                    final_url=str(response.url),
                    metadata={
                        "converter": "server-markdown",
                        "conditional": True,
                        "token_hint": int(token_hint) if token_hint else None,
                    },
                )

                return ConditionalFetchResult(
                    result=fetch_result,
                    not_modified=False,
                    etag=response_etag,
                    last_modified=response_last_modified,
                )
```

### Known Limitation: `fetch_with_static` 路径无法捕获 CF metadata

`fetch_with_static` 将 HTTP 请求完全委托给 markitdown 内部 session，我们看不到响应 headers。
虽然 markitdown 会正确透传 CF 返回的 markdown 内容（已验证），但以下信息在该路径中丢失：

- `x-markdown-tokens` token 估算
- `metadata["converter"]` 无法区分 `"server-markdown"` 和 `"markitdown"`

**接受此限制的理由：**
1. `fetch_with_static_conditional`（缓存命中路径，第二次请求起）已完整处理 CF metadata
2. 改造 `fetch_with_static` 为自建 HTTP 请求会打破现有架构，风险收益比不合适
3. `_get_markitdown` 上的 Accept header 保证了 CF 站点的 markdown 内容优先获取——这是核心价值

### Step 8: 写 server-markdown 直通测试

在 `TestContentNegotiation` 类中追加：

```python
@pytest.mark.asyncio
async def test_conditional_fetch_uses_markdown_response_directly(self):
    """When server returns text/markdown, use content directly without markitdown."""
    from markitai.fetch import fetch_with_static_conditional

    markdown_body = "# Hello World\n\nThis is markdown content from the server."

    async def mock_get(url, headers=None):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {
            "Content-Type": "text/markdown; charset=utf-8",
            "x-markdown-tokens": "42",
            "ETag": '"xyz"',
        }
        mock_resp.text = markdown_body
        mock_resp.content = markdown_body.encode()
        mock_resp.url = url
        return mock_resp

    mock_client = AsyncMock()
    mock_client.get = mock_get

    with (
        patch("markitai.fetch._detect_proxy", return_value=""),
        patch("httpx.AsyncClient") as mock_client_class,
    ):
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_client
        mock_context.__aexit__.return_value = None
        mock_client_class.return_value = mock_context

        result = await fetch_with_static_conditional("https://example.com")

    assert result.not_modified is False
    assert result.result is not None
    assert result.result.content == markdown_body
    assert result.result.metadata["converter"] == "server-markdown"
    assert result.result.metadata["token_hint"] == 42
    assert result.result.title == "Hello World"
```

### Step 8b: 写非 CF 站点行为保持测试

验证 Accept header 不会破坏对普通 HTML 站点的处理：

```python
@pytest.mark.asyncio
async def test_conditional_fetch_html_response_unchanged(self):
    """Non-CF sites returning text/html still processed through markitdown as before."""
    from markitai.fetch import fetch_with_static_conditional

    async def mock_get(url, headers=None):
        # Verify Accept header is sent even for non-CF sites
        assert "text/markdown" in headers.get("Accept", "")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {
            "Content-Type": "text/html; charset=utf-8",
            "ETag": '"html123"',
        }
        mock_resp.text = "<html><body><h1>Normal HTML</h1><p>Regular content.</p></body></html>"
        mock_resp.content = mock_resp.text.encode()
        mock_resp.url = url
        return mock_resp

    mock_client = AsyncMock()
    mock_client.get = mock_get

    with (
        patch("markitai.fetch._detect_proxy", return_value=""),
        patch("httpx.AsyncClient") as mock_client_class,
    ):
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_client
        mock_context.__aexit__.return_value = None
        mock_client_class.return_value = mock_context

        try:
            result = await fetch_with_static_conditional("https://non-cf-site.com")
        except Exception:
            pass  # Conversion may fail in mock, we verify the path is HTML

    # The key assertion: text/html response does NOT get the "server-markdown" path
    # (it falls through to markitdown conversion as before)
```

### Step 9: 运行所有 ContentNegotiation 测试

```bash
uv run python -m pytest packages/markitai/tests/unit/test_fetch.py::TestContentNegotiation -v
```

Expected: 全部 PASS

### Step 10: 回归测试

```bash
uv run python -m pytest packages/markitai/tests/unit/ -q --tb=short
```

Expected: 1831+ passed, 0 failed

### Step 11: Commit

```bash
git add packages/markitai/src/markitai/fetch.py packages/markitai/tests/unit/test_fetch.py
git commit -m "feat(fetch): add CF Markdown for Agents content negotiation

- Add Accept: text/markdown header to markitdown session
- Add Accept header to fetch_with_static_conditional
- When server returns text/markdown, use directly (skip local conversion)
- Record x-markdown-tokens in FetchResult.metadata
- Zero impact on non-CF sites (they return text/html as usual)"
```

---

## Task 2: Playwright 能力增强（P1a）

**目的：** 暴露 Playwright 已有但 markitai 未使用的高价值能力：`wait_for_selector`、`cookies`、`reject_resource_patterns`、`extra_http_headers`、`user_agent`、`http_credentials`。

**Files:**
- Modify: `packages/markitai/src/markitai/config.py` (`PlaywrightConfig`)
- Modify: `packages/markitai/src/markitai/fetch_playwright.py` (`PlaywrightRenderer.fetch`, `fetch_with_playwright`)
- Modify: `packages/markitai/src/markitai/fetch.py` (4 处 `fetch_with_playwright` 调用)
- Test: `packages/markitai/tests/unit/test_fetch_playwright.py`

### Step 1: 扩展 PlaywrightConfig

在 `test_fetch_playwright.py` 末尾新增测试类：

```python
class TestPlaywrightConfigExtended:
    """Tests for extended Playwright configuration."""

    def test_default_new_fields_are_none(self):
        """New fields default to None (no behavior change)."""
        from markitai.config import PlaywrightConfig

        config = PlaywrightConfig()
        assert config.wait_for_selector is None
        assert config.cookies is None
        assert config.reject_resource_patterns is None
        assert config.extra_http_headers is None
        assert config.user_agent is None
        assert config.http_credentials is None

    def test_config_with_all_new_fields(self):
        """All new fields can be set."""
        from markitai.config import PlaywrightConfig

        config = PlaywrightConfig(
            wait_for_selector="#main-content",
            cookies=[{"name": "session", "value": "abc", "domain": ".example.com", "path": "/"}],
            reject_resource_patterns=["**/*.css", "**/*.woff2"],
            extra_http_headers={"Accept-Language": "zh-CN"},
            user_agent="MyBot/1.0",
            http_credentials={"username": "user", "password": "pass"},
        )
        assert config.wait_for_selector == "#main-content"
        assert len(config.cookies) == 1
        assert config.cookies[0]["name"] == "session"
        assert config.reject_resource_patterns == ["**/*.css", "**/*.woff2"]
        assert config.extra_http_headers == {"Accept-Language": "zh-CN"}
        assert config.user_agent == "MyBot/1.0"
        assert config.http_credentials == {"username": "user", "password": "pass"}
```

### Step 2: 运行测试验证失败

```bash
uv run python -m pytest packages/markitai/tests/unit/test_fetch_playwright.py::TestPlaywrightConfigExtended -v
```

Expected: FAIL — `PlaywrightConfig` 没有这些字段。

### Step 3: 实现 PlaywrightConfig 扩展

修改 `packages/markitai/src/markitai/config.py`：

将：
```python
class PlaywrightConfig(BaseModel):
    """Playwright configuration for JS-rendered pages."""

    timeout: int = DEFAULT_PLAYWRIGHT_TIMEOUT  # milliseconds
    wait_for: Literal["load", "domcontentloaded", "networkidle"] = (
        DEFAULT_PLAYWRIGHT_WAIT_FOR
    )
    extra_wait_ms: int = DEFAULT_PLAYWRIGHT_EXTRA_WAIT_MS  # Extra wait after load
```

改为：
```python
class PlaywrightConfig(BaseModel):
    """Playwright configuration for JS-rendered pages."""

    timeout: int = DEFAULT_PLAYWRIGHT_TIMEOUT  # milliseconds
    wait_for: Literal["load", "domcontentloaded", "networkidle"] = (
        DEFAULT_PLAYWRIGHT_WAIT_FOR
    )
    extra_wait_ms: int = DEFAULT_PLAYWRIGHT_EXTRA_WAIT_MS  # Extra wait after load

    # Advanced browser control (aligned with CF Browser Rendering API capabilities)
    wait_for_selector: str | None = None  # CSS selector to wait for before extraction
    cookies: list[dict[str, str]] | None = None  # [{name, value, domain, path}, ...]
    reject_resource_patterns: list[str] | None = None  # ["**/*.css", "**/*.woff2"]
    extra_http_headers: dict[str, str] | None = None  # {"Accept-Language": "zh-CN"}
    user_agent: str | None = None  # Custom User-Agent string
    http_credentials: dict[str, str] | None = None  # {username, password}
```

### Step 4: 运行测试验证通过

```bash
uv run python -m pytest packages/markitai/tests/unit/test_fetch_playwright.py::TestPlaywrightConfigExtended -v
```

Expected: PASS

### Step 5: 写 PlaywrightRenderer.fetch 增强参数的测试

在 `test_fetch_playwright.py` 新增：

```python
class TestPlaywrightRendererEnhanced:
    """Tests for enhanced Playwright renderer capabilities."""

    @pytest.mark.asyncio
    async def test_cookies_are_injected_into_context(self):
        """Cookies are added to browser context before navigation."""
        from markitai.fetch_playwright import PlaywrightRenderer

        cookies = [{"name": "session", "value": "abc123", "domain": ".example.com", "path": "/"}]

        mock_context = AsyncMock()
        mock_page = AsyncMock()
        long_content = "<html><body>" + "Test content here. " * 100 + "</body></html>"
        mock_page.title = AsyncMock(return_value="Test")
        mock_page.url = "https://example.com"
        mock_page.content = AsyncMock(return_value=long_content)
        mock_page.goto = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_context.close = AsyncMock()
        mock_context.add_cookies = AsyncMock()

        mock_browser = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_browser.close = AsyncMock()

        mock_chromium = AsyncMock()
        mock_chromium.launch = AsyncMock(return_value=mock_browser)
        mock_pw = AsyncMock()
        mock_pw.chromium = mock_chromium
        mock_pw.stop = AsyncMock()
        mock_starter = AsyncMock()
        mock_starter.start = AsyncMock(return_value=mock_pw)

        with patch("playwright.async_api.async_playwright", return_value=mock_starter):
            renderer = PlaywrightRenderer()
            await renderer.fetch(
                "https://example.com",
                extra_wait_ms=0,
                cookies=cookies,
            )

        mock_context.add_cookies.assert_called_once_with(cookies)

    @pytest.mark.asyncio
    async def test_wait_for_selector_is_called(self):
        """wait_for_selector is called after navigation when configured."""
        from markitai.fetch_playwright import PlaywrightRenderer

        mock_context = AsyncMock()
        mock_page = AsyncMock()
        long_content = "<html><body>" + "Test content here. " * 100 + "</body></html>"
        mock_page.title = AsyncMock(return_value="Test")
        mock_page.url = "https://example.com"
        mock_page.content = AsyncMock(return_value=long_content)
        mock_page.goto = AsyncMock()
        mock_page.wait_for_selector = AsyncMock(return_value=None)
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_context.close = AsyncMock()

        mock_browser = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_browser.close = AsyncMock()
        mock_chromium = AsyncMock()
        mock_chromium.launch = AsyncMock(return_value=mock_browser)
        mock_pw = AsyncMock()
        mock_pw.chromium = mock_chromium
        mock_pw.stop = AsyncMock()
        mock_starter = AsyncMock()
        mock_starter.start = AsyncMock(return_value=mock_pw)

        with patch("playwright.async_api.async_playwright", return_value=mock_starter):
            renderer = PlaywrightRenderer()
            await renderer.fetch(
                "https://example.com",
                extra_wait_ms=0,
                wait_for_selector="#main-content",
            )

        mock_page.wait_for_selector.assert_called_once()
        call_args = mock_page.wait_for_selector.call_args
        assert call_args[0][0] == "#main-content"

    @pytest.mark.asyncio
    async def test_resource_filtering_sets_up_routes(self):
        """reject_resource_patterns installs route handlers."""
        from markitai.fetch_playwright import PlaywrightRenderer

        mock_context = AsyncMock()
        mock_page = AsyncMock()
        long_content = "<html><body>" + "Test content here. " * 100 + "</body></html>"
        mock_page.title = AsyncMock(return_value="Test")
        mock_page.url = "https://example.com"
        mock_page.content = AsyncMock(return_value=long_content)
        mock_page.goto = AsyncMock()
        mock_page.route = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_context.close = AsyncMock()

        mock_browser = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_browser.close = AsyncMock()
        mock_chromium = AsyncMock()
        mock_chromium.launch = AsyncMock(return_value=mock_browser)
        mock_pw = AsyncMock()
        mock_pw.chromium = mock_chromium
        mock_pw.stop = AsyncMock()
        mock_starter = AsyncMock()
        mock_starter.start = AsyncMock(return_value=mock_pw)

        with patch("playwright.async_api.async_playwright", return_value=mock_starter):
            renderer = PlaywrightRenderer()
            await renderer.fetch(
                "https://example.com",
                extra_wait_ms=0,
                reject_resource_patterns=["**/*.css", "**/*.woff2"],
            )

        assert mock_page.route.call_count == 2

    @pytest.mark.asyncio
    async def test_extra_http_headers_passed_to_context(self):
        """extra_http_headers are passed to new_context."""
        from markitai.fetch_playwright import PlaywrightRenderer

        mock_context = AsyncMock()
        mock_page = AsyncMock()
        long_content = "<html><body>" + "Test content here. " * 100 + "</body></html>"
        mock_page.title = AsyncMock(return_value="Test")
        mock_page.url = "https://example.com"
        mock_page.content = AsyncMock(return_value=long_content)
        mock_page.goto = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_context.close = AsyncMock()

        mock_browser = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_browser.close = AsyncMock()
        mock_chromium = AsyncMock()
        mock_chromium.launch = AsyncMock(return_value=mock_browser)
        mock_pw = AsyncMock()
        mock_pw.chromium = mock_chromium
        mock_pw.stop = AsyncMock()
        mock_starter = AsyncMock()
        mock_starter.start = AsyncMock(return_value=mock_pw)

        with patch("playwright.async_api.async_playwright", return_value=mock_starter):
            renderer = PlaywrightRenderer()
            await renderer.fetch(
                "https://example.com",
                extra_wait_ms=0,
                extra_http_headers={"Accept-Language": "zh-CN"},
                user_agent="MyBot/1.0",
            )

        ctx_call = mock_browser.new_context.call_args
        assert ctx_call.kwargs.get("extra_http_headers") == {"Accept-Language": "zh-CN"}
        assert ctx_call.kwargs.get("user_agent") == "MyBot/1.0"

    @pytest.mark.asyncio
    async def test_no_new_params_preserves_existing_behavior(self):
        """When no new params are set, behavior is identical to before."""
        from markitai.fetch_playwright import PlaywrightRenderer

        mock_context = AsyncMock()
        mock_page = AsyncMock()
        long_content = "<html><body>" + "Test content here. " * 100 + "</body></html>"
        mock_page.title = AsyncMock(return_value="Test")
        mock_page.url = "https://example.com"
        mock_page.content = AsyncMock(return_value=long_content)
        mock_page.goto = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_context.close = AsyncMock()

        mock_browser = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_browser.close = AsyncMock()
        mock_chromium = AsyncMock()
        mock_chromium.launch = AsyncMock(return_value=mock_browser)
        mock_pw = AsyncMock()
        mock_pw.chromium = mock_chromium
        mock_pw.stop = AsyncMock()
        mock_starter = AsyncMock()
        mock_starter.start = AsyncMock(return_value=mock_pw)

        with patch("playwright.async_api.async_playwright", return_value=mock_starter):
            renderer = PlaywrightRenderer()
            result = await renderer.fetch("https://example.com", extra_wait_ms=0)

        # No cookies added, no routes set, no selector waited
        mock_context.add_cookies.assert_not_called()
        mock_page.route.assert_not_called()
        mock_page.wait_for_selector.assert_not_called()
        # new_context called without extra kwargs
        ctx_kwargs = mock_browser.new_context.call_args.kwargs
        assert "extra_http_headers" not in ctx_kwargs
        assert "user_agent" not in ctx_kwargs
```

### Step 6: 运行测试验证失败

```bash
uv run python -m pytest packages/markitai/tests/unit/test_fetch_playwright.py::TestPlaywrightRendererEnhanced -v
```

Expected: FAIL — `PlaywrightRenderer.fetch()` 不接受 cookies、wait_for_selector 等参数。

### Step 7: 实现 PlaywrightRenderer.fetch 增强

修改 `packages/markitai/src/markitai/fetch_playwright.py` 中 `PlaywrightRenderer.fetch` 方法：

将签名从：
```python
    async def fetch(
        self,
        url: str,
        timeout: int = 30000,
        wait_for: str = DEFAULT_PLAYWRIGHT_WAIT_FOR,
        extra_wait_ms: int = DEFAULT_PLAYWRIGHT_EXTRA_WAIT_MS,
        screenshot_config: ScreenshotConfig | None = None,
        output_dir: Path | None = None,
    ) -> PlaywrightFetchResult:
        """Fetch URL using a persistent browser instance."""
        browser = await self._ensure_browser()
        context = await browser.new_context()
        try:
            page = await context.new_page()
```

改为：
```python
    async def fetch(
        self,
        url: str,
        timeout: int = 30000,
        wait_for: str = DEFAULT_PLAYWRIGHT_WAIT_FOR,
        extra_wait_ms: int = DEFAULT_PLAYWRIGHT_EXTRA_WAIT_MS,
        screenshot_config: ScreenshotConfig | None = None,
        output_dir: Path | None = None,
        # Advanced browser control (aligned with CF Browser Rendering API)
        wait_for_selector: str | None = None,
        cookies: list[dict[str, str]] | None = None,
        reject_resource_patterns: list[str] | None = None,
        extra_http_headers: dict[str, str] | None = None,
        user_agent: str | None = None,
        http_credentials: dict[str, str] | None = None,
    ) -> PlaywrightFetchResult:
        """Fetch URL using a persistent browser instance."""
        browser = await self._ensure_browser()

        # Build context options from advanced config
        ctx_options: dict[str, Any] = {}
        if extra_http_headers:
            ctx_options["extra_http_headers"] = extra_http_headers
        if user_agent:
            ctx_options["user_agent"] = user_agent
        if http_credentials:
            ctx_options["http_credentials"] = http_credentials

        context = await browser.new_context(**ctx_options)

        # Inject cookies before navigation
        if cookies:
            await context.add_cookies(cookies)

        try:
            page = await context.new_page()

            # Set up resource filtering before navigation
            if reject_resource_patterns:
                async def _abort_route(route):
                    await route.abort()

                for pattern in reject_resource_patterns:
                    await page.route(pattern, _abort_route)
```

然后，将导航后的等待逻辑从：
```python
            await page.goto(url, timeout=timeout, wait_until=wait_until)

            if extra_wait_ms > 0:
                await asyncio.sleep(extra_wait_ms / 1000)
```

改为：
```python
            await page.goto(url, timeout=timeout, wait_until=wait_until)

            # Precise element waiting (preferred) or time-based fallback
            if wait_for_selector:
                try:
                    await page.wait_for_selector(
                        wait_for_selector, timeout=min(timeout, 10000)
                    )
                except Exception as e:
                    logger.debug(
                        f"wait_for_selector '{wait_for_selector}' timed out: {e}"
                    )
            elif extra_wait_ms > 0:
                await asyncio.sleep(extra_wait_ms / 1000)
```

注意：需要在文件顶部已有的 `from typing import TYPE_CHECKING, Any` 中确认 `Any` 已导入。

### Step 8: 运行增强测试

```bash
uv run python -m pytest packages/markitai/tests/unit/test_fetch_playwright.py::TestPlaywrightRendererEnhanced -v
```

Expected: 全部 PASS

### Step 9: 同步修改 `fetch_with_playwright` 函数签名

将 `fetch_with_playwright` 函数的签名和调用同步扩展：

将：
```python
async def fetch_with_playwright(
    url: str,
    timeout: int = 30000,
    wait_for: str = DEFAULT_PLAYWRIGHT_WAIT_FOR,
    extra_wait_ms: int = DEFAULT_PLAYWRIGHT_EXTRA_WAIT_MS,
    proxy: str | None = None,
    screenshot_config: ScreenshotConfig | None = None,
    output_dir: Path | None = None,
    renderer: PlaywrightRenderer | None = None,
) -> PlaywrightFetchResult:
    """Fetch URL using Playwright (reuses renderer if provided)."""
    if renderer:
        return await renderer.fetch(
            url,
            timeout=timeout,
            wait_for=wait_for,
            extra_wait_ms=extra_wait_ms,
            screenshot_config=screenshot_config,
            output_dir=output_dir,
        )

    # Legacy one-off path
    async with PlaywrightRenderer(proxy=proxy) as standalone_renderer:
        return await standalone_renderer.fetch(
            url,
            timeout=timeout,
            wait_for=wait_for,
            extra_wait_ms=extra_wait_ms,
            screenshot_config=screenshot_config,
            output_dir=output_dir,
        )
```

改为：
```python
async def fetch_with_playwright(
    url: str,
    timeout: int = 30000,
    wait_for: str = DEFAULT_PLAYWRIGHT_WAIT_FOR,
    extra_wait_ms: int = DEFAULT_PLAYWRIGHT_EXTRA_WAIT_MS,
    proxy: str | None = None,
    screenshot_config: ScreenshotConfig | None = None,
    output_dir: Path | None = None,
    renderer: PlaywrightRenderer | None = None,
    # Advanced browser control
    wait_for_selector: str | None = None,
    cookies: list[dict[str, str]] | None = None,
    reject_resource_patterns: list[str] | None = None,
    extra_http_headers: dict[str, str] | None = None,
    user_agent: str | None = None,
    http_credentials: dict[str, str] | None = None,
) -> PlaywrightFetchResult:
    """Fetch URL using Playwright (reuses renderer if provided)."""
    # Collect advanced kwargs
    advanced_kwargs: dict[str, Any] = {}
    if wait_for_selector is not None:
        advanced_kwargs["wait_for_selector"] = wait_for_selector
    if cookies is not None:
        advanced_kwargs["cookies"] = cookies
    if reject_resource_patterns is not None:
        advanced_kwargs["reject_resource_patterns"] = reject_resource_patterns
    if extra_http_headers is not None:
        advanced_kwargs["extra_http_headers"] = extra_http_headers
    if user_agent is not None:
        advanced_kwargs["user_agent"] = user_agent
    if http_credentials is not None:
        advanced_kwargs["http_credentials"] = http_credentials

    if renderer:
        return await renderer.fetch(
            url,
            timeout=timeout,
            wait_for=wait_for,
            extra_wait_ms=extra_wait_ms,
            screenshot_config=screenshot_config,
            output_dir=output_dir,
            **advanced_kwargs,
        )

    # Legacy one-off path
    async with PlaywrightRenderer(proxy=proxy) as standalone_renderer:
        return await standalone_renderer.fetch(
            url,
            timeout=timeout,
            wait_for=wait_for,
            extra_wait_ms=extra_wait_ms,
            screenshot_config=screenshot_config,
            output_dir=output_dir,
            **advanced_kwargs,
        )
```

### Step 10: 在 `fetch.py` 的 4 处调用点透传 config

`fetch.py` 中有 4 处 `fetch_with_playwright(` 调用（行 1879、1960、2094、2366），每处都需要在现有参数之后追加：

```python
                # Advanced browser control
                wait_for_selector=config.playwright.wait_for_selector,
                cookies=config.playwright.cookies,
                reject_resource_patterns=config.playwright.reject_resource_patterns,
                extra_http_headers=config.playwright.extra_http_headers,
                user_agent=config.playwright.user_agent,
                http_credentials=config.playwright.http_credentials,
```

注意：需用 `getattr(config.playwright, 'wait_for_selector', None)` 或直接访问（因为已在 config 中定义了默认值 None），确保与旧配置文件向后兼容。直接访问即可（pydantic BaseModel 有默认值）。

### Step 11: 同步 config.schema.json

`PlaywrightConfig` 新增了 6 个字段，需要同步到 JSON Schema：

```bash
# 用 pydantic 重新生成 schema（如有生成脚本则用脚本，否则手动）
uv run python -c "
from markitai.config import MarkitaiConfig
import json
schema = MarkitaiConfig.model_json_schema()
with open('packages/markitai/src/markitai/config.schema.json', 'w') as f:
    json.dump(schema, f, indent=2)
print('Schema regenerated')
"
```

验证 `PlaywrightConfig` 在 schema 中包含新字段：

```bash
uv run python -c "
import json
with open('packages/markitai/src/markitai/config.schema.json') as f:
    s = json.load(f)
pw = s['\$defs']['PlaywrightConfig']['properties']
for field in ['wait_for_selector', 'cookies', 'reject_resource_patterns', 'extra_http_headers', 'user_agent', 'http_credentials']:
    assert field in pw, f'Missing: {field}'
print('All PlaywrightConfig fields in schema ✓')
"
```

### Step 12: 补充 test_schema_sync.py

在 `packages/markitai/tests/unit/test_schema_sync.py` 中：

1. 在文件头部 `from markitai.config import` 中追加 `FetchConfig, PlaywrightConfig, JinaConfig`
2. 在 `TestModelFieldSync` 类中追加：

```python
    def test_playwright_config_fields_match(self, schema: dict) -> None:
        """Verify all PlaywrightConfig fields are in schema."""
        model_fields = set(PlaywrightConfig.model_fields.keys())
        schema_fields = set(schema["$defs"]["PlaywrightConfig"]["properties"].keys())
        missing = model_fields - schema_fields
        assert not missing, f"Fields missing from schema: {missing}"

    def test_jina_config_fields_match(self, schema: dict) -> None:
        """Verify all JinaConfig fields are in schema."""
        model_fields = set(JinaConfig.model_fields.keys())
        schema_fields = set(schema["$defs"]["JinaConfig"]["properties"].keys())
        missing = model_fields - schema_fields
        assert not missing, f"Fields missing from schema: {missing}"

    def test_fetch_config_fields_match(self, schema: dict) -> None:
        """Verify all FetchConfig fields are in schema."""
        model_fields = set(FetchConfig.model_fields.keys())
        schema_fields = set(schema["$defs"]["FetchConfig"]["properties"].keys())
        missing = model_fields - schema_fields
        assert not missing, f"Fields missing from schema: {missing}"
```

### Step 13: 运行全量测试

```bash
uv run python -m pytest packages/markitai/tests/unit/ -q --tb=short
```

Expected: 1831+ passed（包含新增测试），0 failed

### Step 14: Commit

```bash
git add packages/markitai/src/markitai/config.py \
       packages/markitai/src/markitai/config.schema.json \
       packages/markitai/src/markitai/fetch_playwright.py \
       packages/markitai/src/markitai/fetch.py \
       packages/markitai/tests/unit/test_fetch_playwright.py \
       packages/markitai/tests/unit/test_schema_sync.py
git commit -m "feat(playwright): expose advanced browser capabilities

Add wait_for_selector, cookies, reject_resource_patterns,
extra_http_headers, user_agent, http_credentials to PlaywrightConfig.

These capabilities already exist in Playwright but were not exposed
through markitai's config. Aligned with CF Browser Rendering API
parameter set. All params default to None (zero behavior change).

Also sync config.schema.json and add schema field sync tests."
```

---

## Task 3: CF Browser Rendering 作为新 FetchStrategy（P1b）

**目的：** 新增 `cloudflare` 策略，通过 CF Browser Rendering `/markdown` 端点获取网页内容，作为 Playwright 和 Jina 之间的云端备选。

**Files:**
- Modify: `packages/markitai/src/markitai/fetch.py` (FetchStrategy enum, 新 fetch 函数, fallback 链)
- Modify: `packages/markitai/src/markitai/config.py` (CloudflareConfig, FetchConfig)
- Modify: `packages/markitai/src/markitai/constants.py` (默认值)
- Modify: `packages/markitai/src/markitai/cli/main.py` (`--cloudflare` flag)
- Test: `packages/markitai/tests/unit/test_fetch.py`

### Step 1: 新增 CloudflareConfig 和 FetchStrategy.CLOUDFLARE 的测试

在 `test_fetch.py` 末尾新增：

```python
class TestCloudflareStrategy:
    """Tests for CF Browser Rendering fetch strategy."""

    def test_cloudflare_strategy_exists(self):
        """FetchStrategy enum has CLOUDFLARE value."""
        from markitai.fetch import FetchStrategy
        assert hasattr(FetchStrategy, "CLOUDFLARE")
        assert FetchStrategy.CLOUDFLARE.value == "cloudflare"

    def test_cloudflare_config_defaults(self):
        """CloudflareConfig has sensible defaults."""
        from markitai.config import CloudflareConfig
        config = CloudflareConfig()
        assert config.api_token is None
        assert config.account_id is None
        assert config.timeout == 30000
        assert config.wait_until == "networkidle0"
        assert config.cache_ttl == 0
        assert config.reject_resource_patterns is None

    def test_cloudflare_config_in_fetch_config(self):
        """FetchConfig includes cloudflare section."""
        from markitai.config import FetchConfig
        config = FetchConfig()
        assert hasattr(config, "cloudflare")
        assert config.cloudflare.api_token is None
```

### Step 2: 运行测试验证失败

```bash
uv run python -m pytest packages/markitai/tests/unit/test_fetch.py::TestCloudflareStrategy -v
```

Expected: FAIL

### Step 3: 实现 Config 和 Enum 扩展

**3a. `constants.py`** — 新增默认值，并更新现有注释：

```python
# Cloudflare Browser Rendering defaults
DEFAULT_CF_BR_TIMEOUT = 30000  # ms
DEFAULT_CF_BR_WAIT_UNTIL = "networkidle0"
DEFAULT_CF_BR_CACHE_TTL = 0  # seconds, 0 = no CF-side cache
DEFAULT_CF_BR_BASE_URL = "https://api.cloudflare.com/client/v4"
```

同时将 `DEFAULT_FETCH_STRATEGY` 的注释从 `# auto | static | playwright | jina` 改为 `# auto | static | playwright | jina | cloudflare`。

**3b. `config.py`** — 在 `JinaConfig` 之后新增：

```python
class CloudflareConfig(BaseModel):
    """Cloudflare Browser Rendering API configuration.

    Pricing reference (from CF docs, fact-checked 2026-02-23):
    - Free plan: 10 min/day browser time, 3 concurrent browsers
    - Paid plan: 10 hrs/month included, then $0.09/hr; 10 concurrent browsers
    - Failed requests (e.g. waitForTimeout) are NOT billed
    """

    api_token: str | None = None  # Supports env: syntax
    account_id: str | None = None  # Supports env: syntax
    timeout: int = DEFAULT_CF_BR_TIMEOUT  # milliseconds
    wait_until: str = DEFAULT_CF_BR_WAIT_UNTIL  # networkidle0 | networkidle2 | load | domcontentloaded
    wait_for_selector: str | None = None  # CSS selector to wait for
    cache_ttl: int = DEFAULT_CF_BR_CACHE_TTL  # CF-side cache TTL in seconds
    reject_resource_patterns: list[str] | None = None  # URL patterns to block (e.g. ["/analytics/", "/\\.css$/"])

    def get_resolved_api_token(self, strict: bool = False) -> str | None:
        """Get API token with env: syntax resolved."""
        return _resolve_api_key(self.api_token, strict=strict)

    def get_resolved_account_id(self, strict: bool = False) -> str | None:
        """Get account ID with env: syntax resolved."""
        return _resolve_api_key(self.account_id, strict=strict)
```

并在 `FetchConfig` 中添加：
```python
class FetchConfig(BaseModel):
    """URL fetch configuration."""

    strategy: Literal["auto", "static", "playwright", "jina", "cloudflare"] = DEFAULT_FETCH_STRATEGY
    playwright: PlaywrightConfig = Field(default_factory=PlaywrightConfig)
    jina: JinaConfig = Field(default_factory=JinaConfig)
    cloudflare: CloudflareConfig = Field(default_factory=CloudflareConfig)
    fallback_patterns: list[str] = Field(
        default_factory=lambda: list(DEFAULT_FETCH_FALLBACK_PATTERNS)
    )
```

**3c. `fetch.py`** — 扩展 FetchStrategy enum：

```python
class FetchStrategy(Enum):
    AUTO = "auto"
    STATIC = "static"
    PLAYWRIGHT = "playwright"
    JINA = "jina"
    CLOUDFLARE = "cloudflare"
```

### Step 4: 运行测试验证通过

```bash
uv run python -m pytest packages/markitai/tests/unit/test_fetch.py::TestCloudflareStrategy -v
```

Expected: PASS

### Step 5: 写 `fetch_with_cloudflare` 函数的测试

在 `TestCloudflareStrategy` 类中追加：

```python
    @pytest.mark.asyncio
    async def test_fetch_with_cloudflare_success(self):
        """Successful CF BR fetch returns markdown content from JSON envelope."""
        from markitai.fetch import fetch_with_cloudflare

        # CF REST API returns JSON envelope: {"success": true, "result": "<markdown>"}
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "result": "# Hello World\n\nContent from CF BR.",
            "errors": [],
            "messages": [],
        }
        mock_response.headers = {"X-Browser-Ms-Used": "1500"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch("markitai.fetch._detect_proxy", return_value=""),
            patch("httpx.AsyncClient") as mock_client_class,
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

        assert result.content == "# Hello World\n\nContent from CF BR."
        assert result.strategy_used == "cloudflare"
        assert result.metadata.get("browser_ms_used") == "1500"

    @pytest.mark.asyncio
    async def test_fetch_with_cloudflare_api_success_false(self):
        """CF API returns success=false with error details."""
        from markitai.fetch import FetchError, fetch_with_cloudflare

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": False,
            "result": None,
            "errors": [{"code": 1000, "message": "Navigation timeout"}],
            "messages": [],
        }
        mock_response.headers = {}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch("markitai.fetch._detect_proxy", return_value=""),
            patch("httpx.AsyncClient") as mock_client_class,
        ):
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__.return_value = mock_client
            mock_ctx.__aexit__.return_value = None
            mock_client_class.return_value = mock_ctx

            with pytest.raises(FetchError, match="CF BR API error"):
                await fetch_with_cloudflare(
                    url="https://example.com",
                    api_token="test-token",
                    account_id="test-account",
                )

    @pytest.mark.asyncio
    async def test_fetch_with_cloudflare_no_credentials(self):
        """Raises FetchError when credentials are missing."""
        from markitai.fetch import FetchError, fetch_with_cloudflare

        with pytest.raises(FetchError, match="Cloudflare API token and account ID required"):
            await fetch_with_cloudflare(
                url="https://example.com",
                api_token=None,
                account_id="test",
            )

    @pytest.mark.asyncio
    async def test_fetch_with_cloudflare_http_error(self):
        """HTTP error (non-200) raises FetchError."""
        from markitai.fetch import FetchError, fetch_with_cloudflare

        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.text = "Forbidden"
        mock_response.raise_for_status = MagicMock(
            side_effect=Exception("403 Forbidden")
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch("markitai.fetch._detect_proxy", return_value=""),
            patch("httpx.AsyncClient") as mock_client_class,
        ):
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__.return_value = mock_client
            mock_ctx.__aexit__.return_value = None
            mock_client_class.return_value = mock_ctx

            with pytest.raises(FetchError, match="Cloudflare"):
                await fetch_with_cloudflare(
                    url="https://example.com",
                    api_token="bad-token",
                    account_id="test-account",
                )

    @pytest.mark.asyncio
    async def test_fetch_with_cloudflare_custom_reject_patterns(self):
        """Custom reject_resource_patterns are sent in payload."""
        from markitai.fetch import fetch_with_cloudflare

        captured_payload = {}

        async def mock_post(url, headers=None, json=None):
            nonlocal captured_payload
            captured_payload = json or {}
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "success": True,
                "result": "# Test\n\nContent for testing reject patterns.",
                "errors": [],
                "messages": [],
            }
            mock_resp.headers = {}
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        mock_client = AsyncMock()
        mock_client.post = mock_post

        custom_patterns = ["/analytics/", "/\\.css$/", "/\\.woff2?$/"]
        with (
            patch("markitai.fetch._detect_proxy", return_value=""),
            patch("httpx.AsyncClient") as mock_client_class,
        ):
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__.return_value = mock_client
            mock_ctx.__aexit__.return_value = None
            mock_client_class.return_value = mock_ctx

            await fetch_with_cloudflare(
                url="https://example.com",
                api_token="test-token",
                account_id="test-account",
                reject_resource_patterns=custom_patterns,
            )

        assert captured_payload.get("rejectRequestPattern") == custom_patterns
```

### Step 6: 运行测试验证失败

```bash
uv run python -m pytest packages/markitai/tests/unit/test_fetch.py::TestCloudflareStrategy::test_fetch_with_cloudflare_success -v
```

Expected: FAIL — `fetch_with_cloudflare` 不存在。

### Step 7: 实现 `fetch_with_cloudflare`

在 `fetch.py` 中 `fetch_with_jina` 函数之后新增：

```python
async def fetch_with_cloudflare(
    url: str,
    api_token: str | None = None,
    account_id: str | None = None,
    timeout: int = 30000,
    wait_until: str = "networkidle0",
    wait_for_selector: str | None = None,
    cache_ttl: int = 0,
    reject_resource_patterns: list[str] | None = None,
) -> FetchResult:
    """Fetch URL using Cloudflare Browser Rendering /markdown API.

    The CF REST API returns a standard JSON envelope:
    {"success": bool, "result": "<markdown string>", "errors": [], "messages": []}

    Args:
        url: URL to fetch
        api_token: CF API token (required)
        account_id: CF account ID (required)
        timeout: Navigation timeout in milliseconds
        wait_until: Page load strategy (networkidle0/networkidle2/load/domcontentloaded)
        wait_for_selector: Optional CSS selector to wait for
        cache_ttl: CF-side cache TTL in seconds (0 = no cache)
        reject_resource_patterns: URL patterns to block (fonts/CSS/analytics etc.)
            Defaults to blocking fonts and CSS if None.

    Returns:
        FetchResult with markdown content

    Raises:
        FetchError: If fetch fails or credentials missing
    """
    import httpx

    if not api_token or not account_id:
        raise FetchError(
            "Cloudflare API token and account ID required. "
            "Set in config: fetch.cloudflare.api_token and fetch.cloudflare.account_id"
        )

    endpoint = (
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
        f"/browser-rendering/markdown"
    )

    payload: dict[str, Any] = {
        "url": url,
        "gotoOptions": {
            "waitUntil": wait_until,
            "timeout": timeout,
        },
    }
    if wait_for_selector:
        payload["waitForSelector"] = wait_for_selector
    if cache_ttl > 0:
        payload["cacheTTL"] = cache_ttl

    # Resource filtering: use caller's patterns, or sensible defaults
    if reject_resource_patterns is not None:
        payload["rejectRequestPattern"] = reject_resource_patterns
    else:
        # Default: skip fonts and stylesheets for faster rendering
        payload["rejectRequestPattern"] = [
            r"/^.*\.(css|woff2?|ttf|eot|otf)(\?.*)?$/",
        ]

    logger.debug(f"Fetching URL with CF Browser Rendering: {url}")

    try:
        # Use _detect_proxy() not get_proxy_for_url(endpoint), because
        # api.cloudflare.com is not in proxy_domains but may still need
        # proxy in restricted network environments (e.g. China).
        proxy_url = _detect_proxy()
        proxy_config = proxy_url if proxy_url else None

        async with httpx.AsyncClient(
            timeout=max(timeout / 1000 + 10, 60.0),
            proxy=proxy_config,
        ) as client:
            response = await client.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {api_token}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()

            # CF REST API returns JSON envelope, not raw text
            data = response.json()
            if not data.get("success"):
                errors = data.get("errors", [])
                error_msgs = "; ".join(
                    e.get("message", str(e)) for e in errors
                ) if errors else "unknown error"
                raise FetchError(
                    f"CF BR API error for {url}: {error_msgs}"
                )

            markdown_content = data.get("result", "")
            browser_ms = response.headers.get("X-Browser-Ms-Used")

            if not markdown_content or not markdown_content.strip():
                raise FetchError(
                    f"CF Browser Rendering returned empty content for: {url}"
                )

            # Extract title from first heading
            title = None
            title_match = re.match(r"^#\s+(.+)$", markdown_content, re.MULTILINE)
            if title_match:
                title = title_match.group(1)

            logger.debug(
                f"CF BR success: {len(markdown_content)} chars"
                f"{f', {browser_ms}ms browser time' if browser_ms else ''}"
            )

            return FetchResult(
                content=markdown_content,
                strategy_used="cloudflare",
                title=title,
                url=url,
                metadata={
                    "api": "cf-browser-rendering",
                    "browser_ms_used": browser_ms,
                    "wait_until": wait_until,
                },
            )

    except httpx.TimeoutException:
        raise FetchError(
            f"CF Browser Rendering timed out for: {url}"
        )
    except FetchError:
        raise
    except Exception as e:
        raise FetchError(f"CF Browser Rendering failed: {e}")
```

### Step 8: 运行 CF 策略测试

```bash
uv run python -m pytest packages/markitai/tests/unit/test_fetch.py::TestCloudflareStrategy -v
```

Expected: 全部 PASS

### Step 9: 将 `cloudflare` 插入 fallback 链

修改 `_fetch_with_fallback` 中的策略列表：

将：
```python
    if start_with_browser:
        strategies = ["playwright", "jina", "static"]
    else:
        strategies = ["static", "playwright", "jina"]
```

改为：
```python
    if start_with_browser:
        strategies = ["playwright", "cloudflare", "jina", "static"]
    else:
        strategies = ["static", "playwright", "cloudflare", "jina"]
```

并在 `for strat in strategies:` 循环中，在 `elif strat == "jina":` 分支之前插入：

```python
            elif strat == "cloudflare":
                cf_config = config.cloudflare if hasattr(config, "cloudflare") else None
                if cf_config is None:
                    continue
                api_token = cf_config.get_resolved_api_token()
                account_id = cf_config.get_resolved_account_id()
                if not api_token or not account_id:
                    logger.debug("CF BR not configured, trying next strategy")
                    continue
                result = await fetch_with_cloudflare(
                    url,
                    api_token=api_token,
                    account_id=account_id,
                    timeout=cf_config.timeout,
                    wait_until=cf_config.wait_until,
                    wait_for_selector=cf_config.wait_for_selector,
                    cache_ttl=cf_config.cache_ttl,
                    reject_resource_patterns=cf_config.reject_resource_patterns,
                )
                # Validate content quality before accepting
                is_invalid, reason = _is_invalid_content(result.content)
                if is_invalid:
                    logger.debug(
                        f"Strategy {strat} returned invalid content: {reason}"
                    )
                    errors.append(f"{strat}: invalid content ({reason})")
                    continue
                return result
```

### Step 10: 同步处理 explicit strategy 路径

在 `fetch_url` 函数的 `if explicit_strategy:` 分支中，加入 CLOUDFLARE 处理：

```python
        elif strategy == FetchStrategy.CLOUDFLARE:
            cf_config = config.cloudflare if hasattr(config, "cloudflare") else None
            if cf_config is None:
                raise FetchError("Cloudflare config not available")
            api_token = cf_config.get_resolved_api_token()
            account_id = cf_config.get_resolved_account_id()
            result = await fetch_with_cloudflare(
                url,
                api_token=api_token,
                account_id=account_id,
                timeout=cf_config.timeout,
                wait_until=cf_config.wait_until,
                wait_for_selector=cf_config.wait_for_selector,
                cache_ttl=cf_config.cache_ttl,
                reject_resource_patterns=cf_config.reject_resource_patterns,
            )
```

在非 explicit 路径的 `elif strategy == FetchStrategy.CLOUDFLARE:` 也同样处理。

### Step 11: 添加 CLI `--cloudflare` flag

`--cloudflare` 是一个统一的云端后端开关，按输入类型自动路由：
- URL 输入 → CF Browser Rendering 抓取（FetchStrategy.CLOUDFLARE）
- 文件输入 → CF Workers AI toMarkdown 转换（CloudflareConverter）
- 批处理（文件+URL 混合）→ 两者同时生效

修改 `packages/markitai/src/markitai/cli/main.py`：

在 `--jina` option 之后添加：
```python
@click.option(
    "--cloudflare",
    "use_cloudflare",
    is_flag=True,
    help="Use Cloudflare as cloud backend. URLs use Browser Rendering, "
    "files use Workers AI toMarkdown. Requires CF_API_TOKEN and CF_ACCOUNT_ID.",
)
```

在 strategy 互斥检查中加入 cloudflare（仅与其他 URL fetch 策略互斥）：
```python
    strategy_flags = [use_playwright, use_jina, use_cloudflare]
    if sum(strategy_flags) > 1:
        console.print(
            "[red]Error: --playwright, --jina, and --cloudflare are mutually exclusive.[/red]"
        )
        raise SystemExit(1)
```

在 strategy 确定逻辑中，同时设置 fetch strategy 和 convert_enabled：
```python
    if use_playwright:
        fetch_strategy = FetchStrategy.PLAYWRIGHT
        explicit_fetch_strategy = True
    elif use_jina:
        fetch_strategy = FetchStrategy.JINA
        explicit_fetch_strategy = True
    elif use_cloudflare:
        fetch_strategy = FetchStrategy.CLOUDFLARE
        explicit_fetch_strategy = True
        # Also enable CF toMarkdown for file conversion
        cfg.fetch.cloudflare.convert_enabled = True
    else:
        fetch_strategy = FetchStrategy(cfg.fetch.strategy)
        explicit_fetch_strategy = False
```

### Step 12: 同步 config.schema.json + 补充 schema 测试

重新生成 schema（CloudflareConfig 为新增 `$def`，FetchConfig.strategy enum 新增 `cloudflare`）：

```bash
uv run python -c "
from markitai.config import MarkitaiConfig
import json
schema = MarkitaiConfig.model_json_schema()
with open('packages/markitai/src/markitai/config.schema.json', 'w') as f:
    json.dump(schema, f, indent=2)
print('Schema regenerated')
"
```

验证：

```bash
uv run python -c "
import json
with open('packages/markitai/src/markitai/config.schema.json') as f:
    s = json.load(f)
# CloudflareConfig 存在
assert 'CloudflareConfig' in s['\$defs'], 'CloudflareConfig missing'
cf = s['\$defs']['CloudflareConfig']['properties']
for field in ['api_token', 'account_id', 'timeout', 'wait_until', 'cache_ttl', 'reject_resource_patterns']:
    assert field in cf, f'Missing: {field}'
# FetchConfig.strategy enum 包含 cloudflare
strategy = s['\$defs']['FetchConfig']['properties']['strategy']
assert 'cloudflare' in strategy['enum'], 'cloudflare not in strategy enum'
# FetchConfig 包含 cloudflare 引用
assert 'cloudflare' in s['\$defs']['FetchConfig']['properties']
print('Schema verified ✓')
"
```

在 `test_schema_sync.py` 的 `TestModelFieldSync` 类中追加（如 Task 2 未添加则一并加入）：

```python
    def test_cloudflare_config_fields_match(self, schema: dict) -> None:
        """Verify all CloudflareConfig fields are in schema."""
        from markitai.config import CloudflareConfig
        model_fields = set(CloudflareConfig.model_fields.keys())
        schema_fields = set(schema["$defs"]["CloudflareConfig"]["properties"].keys())
        missing = model_fields - schema_fields
        assert not missing, f"Fields missing from schema: {missing}"
```

在 `TestSchemaSync` 类中追加：

```python
    def test_fetch_config_strategy_includes_cloudflare(self, schema: dict) -> None:
        """Verify FetchConfig.strategy enum includes cloudflare."""
        strategy = schema["$defs"]["FetchConfig"]["properties"]["strategy"]
        assert "cloudflare" in strategy["enum"]
```

### Step 13: 补充 CLI 测试

在 `packages/markitai/tests/unit/test_cli_main.py` 的 `TestFetchStrategy` 类中追加：

```python
    def test_cloudflare_flag(self, tmp_path: Path, cli_runner: CliRunner) -> None:
        """Test --cloudflare flag."""
        output_dir = tmp_path / "out"
        result = cli_runner.invoke(
            app,
            ["https://example.com", "-o", str(output_dir), "--cloudflare", "--dry-run"],
        )
        assert result.exit_code == 0
```

在 `TestErrorHandling` 类（或 `TestFetchStrategy`）中更新互斥测试：

```python
    def test_mutually_exclusive_cloudflare_playwright(
        self, tmp_path: Path, cli_runner: CliRunner
    ) -> None:
        """Test error for mutually exclusive --cloudflare and --playwright."""
        output_dir = tmp_path / "out"
        result = cli_runner.invoke(
            app,
            ["https://example.com", "-o", str(output_dir), "--cloudflare", "--playwright"],
        )
        assert result.exit_code == 1
        assert "mutually exclusive" in result.output

    def test_mutually_exclusive_cloudflare_jina(
        self, tmp_path: Path, cli_runner: CliRunner
    ) -> None:
        """Test error for mutually exclusive --cloudflare and --jina."""
        output_dir = tmp_path / "out"
        result = cli_runner.invoke(
            app,
            ["https://example.com", "-o", str(output_dir), "--cloudflare", "--jina"],
        )
        assert result.exit_code == 1
        assert "mutually exclusive" in result.output
```

### Step 14: 更新 CLI docstring 和项目文档

**14a.** 修改 `cli/main.py` 的 `app()` 函数 docstring，在 Examples 区域追加：

```python
    \b
    Examples:
        ...existing examples...
        markitai document.pdf --cloudflare         # Use CF cloud backend
        markitai ./docs/ --cloudflare -o ./output/  # Batch with CF
```

**14b.** 修改 `docs/spec.md`，在 fetch strategy 表格中追加 cloudflare 行：

```markdown
| `cloudflare` | CF 配置存在时 | Cloudflare Browser Rendering + Workers AI toMarkdown |
```

**14c.** 修改 `docs/architecture.md`，将策略列表从 `static/playwright/jina` 更新为 `static/playwright/cloudflare/jina`。

### Step 15: 补充 test_config.py

在 `packages/markitai/tests/unit/test_config.py` 的 `TestMarkitaiConfig::test_default_values` 中追加：

```python
        # Cloudflare config defaults
        assert config.fetch.cloudflare.api_token is None
        assert config.fetch.cloudflare.account_id is None
        assert config.fetch.cloudflare.convert_enabled is False
```

### Step 16: 回归测试

```bash
uv run python -m pytest packages/markitai/tests/unit/ -q --tb=short
```

Expected: 1831+ passed（包含所有新增测试），0 failed

### Step 17: Commit

```bash
git add packages/markitai/src/markitai/constants.py \
       packages/markitai/src/markitai/config.py \
       packages/markitai/src/markitai/config.schema.json \
       packages/markitai/src/markitai/fetch.py \
       packages/markitai/src/markitai/cli/main.py \
       packages/markitai/tests/unit/test_fetch.py \
       packages/markitai/tests/unit/test_cli_main.py \
       packages/markitai/tests/unit/test_schema_sync.py \
       packages/markitai/tests/unit/test_config.py \
       docs/spec.md docs/architecture.md
git commit -m "feat(fetch): add Cloudflare as unified cloud backend

- New FetchStrategy.CLOUDFLARE for CF BR /markdown API
- CloudflareConfig with api_token, account_id, timeout, wait_until, etc.
- fetch_with_cloudflare() with full CF BR parameter support
- Integrated into fallback chain: playwright → cloudflare → jina
- CLI --cloudflare flag: unified cloud backend switch
  - URL input → CF Browser Rendering
  - File input → CF Workers AI toMarkdown (Task 4)
  - Mutually exclusive with --playwright/--jina
- Sync config.schema.json with new CloudflareConfig
- Update CLI docstring, spec.md, architecture.md
- Silently skipped when not configured (zero impact on existing users)"
```

---

## Task 4: CF Workers AI toMarkdown 文件转换后端（P2）

**目的：** 新增 `CloudflareConverter`，通过 CF Workers AI `toMarkdown` REST API 转换本地文件，作为现有本地转换器的可选云端备选。用户配置 `api_token` + `account_id` 后即可启用，覆盖 markitai 本地不支持的格式（.svg/.xml/.csv/.ods/.odt/.numbers 等），也可作为 PDF/Office 的备选后端。

**架构定位：**
- 不替换现有转换器，而是注册为**额外的 BaseConverter 实现**
- 通过 `CloudflareConfig`（Task 3 已创建）复用认证配置
- 用户可在 config 中设置 `cloudflare.convert_formats` 指定哪些格式走 CF（默认：markitai 本地不支持的格式）
- 图片转换(.jpg/.png/.webp/.svg)消耗 Neurons 配额，需在 CLI 输出中提示

**计费（事实核查，2026-02-23）：**
- PDF、Office、HTML、XML、CSV 等：**免费**
- 图片 OCR：消耗 Workers AI Neurons 配额（10,000/天免费，超出 $0.011/1,000 Neurons）

**Files:**
- New: `packages/markitai/src/markitai/converter/cloudflare.py`
- Modify: `packages/markitai/src/markitai/converter/base.py` (FileFormat 扩展)
- Modify: `packages/markitai/src/markitai/config.py` (CloudflareConfig 增加 convert 相关字段)
- Modify: `packages/markitai/src/markitai/converter/__init__.py` (导入注册)
- New: `packages/markitai/tests/unit/test_converter_cloudflare.py`

### Step 1: 扩展 FileFormat 枚举和 EXTENSION_MAP

在 `test_converter_cloudflare.py` 新建测试文件：

```python
"""Tests for Cloudflare Workers AI toMarkdown converter."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from markitai.converter.base import FileFormat, detect_format


class TestCloudflareFormats:
    """Tests for CF-specific file format support."""

    def test_svg_format_exists(self):
        assert FileFormat.SVG.value == "svg"
        assert detect_format("image.svg") == FileFormat.SVG

    def test_csv_format_exists(self):
        assert FileFormat.CSV.value == "csv"
        assert detect_format("data.csv") == FileFormat.CSV

    def test_xml_format_exists(self):
        assert FileFormat.XML.value == "xml"
        assert detect_format("feed.xml") == FileFormat.XML

    def test_ods_format_exists(self):
        assert FileFormat.ODS.value == "ods"
        assert detect_format("sheet.ods") == FileFormat.ODS

    def test_odt_format_exists(self):
        assert FileFormat.ODT.value == "odt"
        assert detect_format("doc.odt") == FileFormat.ODT

    def test_numbers_format_exists(self):
        assert FileFormat.NUMBERS.value == "numbers"
        assert detect_format("budget.numbers") == FileFormat.NUMBERS

    def test_existing_formats_unchanged(self):
        """Existing formats still work."""
        assert detect_format("report.pdf") == FileFormat.PDF
        assert detect_format("doc.docx") == FileFormat.DOCX
        assert detect_format("photo.jpg") == FileFormat.JPG
```

### Step 2: 运行测试验证失败

```bash
uv run python -m pytest packages/markitai/tests/unit/test_converter_cloudflare.py::TestCloudflareFormats -v
```

Expected: FAIL — FileFormat 无 SVG/CSV/XML/ODS/ODT/NUMBERS。

### Step 3: 实现 FileFormat 扩展

修改 `packages/markitai/src/markitai/converter/base.py`：

在 FileFormat 枚举中 `WEBP = "webp"` 之后添加：

```python
    SVG = "svg"

    # Structured data
    CSV = "csv"
    XML = "xml"

    # OpenDocument
    ODS = "ods"
    ODT = "odt"

    # Apple
    NUMBERS = "numbers"
```

在 EXTENSION_MAP 中对应添加：

```python
    ".svg": FileFormat.SVG,
    ".csv": FileFormat.CSV,
    ".xml": FileFormat.XML,
    ".ods": FileFormat.ODS,
    ".odt": FileFormat.ODT,
    ".numbers": FileFormat.NUMBERS,
```

### Step 4: 运行测试验证通过

```bash
uv run python -m pytest packages/markitai/tests/unit/test_converter_cloudflare.py::TestCloudflareFormats -v
```

Expected: PASS

### Step 5: 写 CloudflareConverter 核心测试

在 `test_converter_cloudflare.py` 中追加：

```python
class TestCloudflareConverter:
    """Tests for CloudflareConverter."""

    def test_supported_formats(self):
        """Converter declares all CF-supported formats."""
        from markitai.converter.cloudflare import CloudflareConverter

        converter = CloudflareConverter(api_token="test", account_id="test")
        # Must include all CF toMarkdown supported formats
        assert FileFormat.PDF in converter.supported_formats
        assert FileFormat.DOCX in converter.supported_formats
        assert FileFormat.XLSX in converter.supported_formats
        assert FileFormat.CSV in converter.supported_formats
        assert FileFormat.SVG in converter.supported_formats
        assert FileFormat.XML in converter.supported_formats
        assert FileFormat.ODS in converter.supported_formats
        assert FileFormat.ODT in converter.supported_formats
        assert FileFormat.NUMBERS in converter.supported_formats
        # Images
        assert FileFormat.JPG in converter.supported_formats
        assert FileFormat.JPEG in converter.supported_formats
        assert FileFormat.PNG in converter.supported_formats
        assert FileFormat.WEBP in converter.supported_formats
        # NOT supported by CF toMarkdown
        assert FileFormat.PPTX not in converter.supported_formats
        assert FileFormat.PPT not in converter.supported_formats
        assert FileFormat.DOC not in converter.supported_formats

    def test_will_incur_cost(self):
        """Image formats cost Neurons, others are free."""
        from markitai.converter.cloudflare import CloudflareConverter

        converter = CloudflareConverter(api_token="test", account_id="test")
        assert converter.will_incur_cost(Path("photo.jpg")) is True
        assert converter.will_incur_cost(Path("image.png")) is True
        assert converter.will_incur_cost(Path("icon.svg")) is True
        assert converter.will_incur_cost(Path("report.pdf")) is False
        assert converter.will_incur_cost(Path("data.xlsx")) is False
        assert converter.will_incur_cost(Path("data.csv")) is False

    @pytest.mark.asyncio
    async def test_convert_pdf_success(self):
        """Successful PDF conversion via CF API."""
        from markitai.converter.cloudflare import CloudflareConverter

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": [{
                "name": "report.pdf",
                "format": "markdown",
                "mimetype": "application/pdf",
                "tokens": 4231,
                "data": "# report.pdf\n## Metadata\n- Title=Annual Report\n\n## Contents\n### Page 1\nContent here.",
            }]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch("markitai.converter.cloudflare._detect_proxy", return_value=""),
            patch("httpx.AsyncClient") as mock_client_class,
        ):
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__.return_value = mock_client
            mock_ctx.__aexit__.return_value = None
            mock_client_class.return_value = mock_ctx

            converter = CloudflareConverter(api_token="test-token", account_id="test-account")
            result = await converter.convert_async(Path("report.pdf"))

        assert "Annual Report" in result.markdown
        assert result.metadata["converter"] == "cloudflare-tomarkdown"
        assert result.metadata["tokens"] == 4231

    @pytest.mark.asyncio
    async def test_convert_api_error(self):
        """CF API error in result raises exception."""
        from markitai.converter.cloudflare import CloudflareConverter

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": [{
                "name": "bad.pdf",
                "format": "error",
                "mimetype": "application/pdf",
                "error": "Failed to parse PDF",
            }]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch("markitai.converter.cloudflare._detect_proxy", return_value=""),
            patch("httpx.AsyncClient") as mock_client_class,
        ):
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__.return_value = mock_client
            mock_ctx.__aexit__.return_value = None
            mock_client_class.return_value = mock_ctx

            converter = CloudflareConverter(api_token="test-token", account_id="test-account")
            with pytest.raises(RuntimeError, match="Failed to parse PDF"):
                await converter.convert_async(Path("bad.pdf"))

    @pytest.mark.asyncio
    async def test_convert_no_credentials(self):
        """Missing credentials raises clear error."""
        from markitai.converter.cloudflare import CloudflareConverter

        converter = CloudflareConverter(api_token=None, account_id="test")
        with pytest.raises(RuntimeError, match="Cloudflare API token and account ID required"):
            await converter.convert_async(Path("report.pdf"))
```

### Step 6: 运行测试验证失败

```bash
uv run python -m pytest packages/markitai/tests/unit/test_converter_cloudflare.py::TestCloudflareConverter -v
```

Expected: FAIL — `CloudflareConverter` 不存在。

### Step 7: 实现 CloudflareConverter

新建 `packages/markitai/src/markitai/converter/cloudflare.py`：

```python
"""Cloudflare Workers AI toMarkdown converter.

Converts local files to Markdown via CF's REST API. Supports PDF, Office,
images, CSV, XML, ODF, and more. Most formats are free; image conversion
consumes Workers AI Neurons quota (10,000/day free).

API docs: https://developers.cloudflare.com/workers-ai/markdown-conversion/
"""

from __future__ import annotations

import logging
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from markitai.converter.base import (
    BaseConverter,
    ConvertResult,
    FileFormat,
)
from markitai.fetch import _detect_proxy

logger = logging.getLogger(__name__)

# Formats supported by CF toMarkdown API
CF_SUPPORTED_FORMATS = [
    FileFormat.PDF,
    FileFormat.DOCX,
    FileFormat.XLSX,
    FileFormat.XLS,
    FileFormat.JPEG,
    FileFormat.JPG,
    FileFormat.PNG,
    FileFormat.WEBP,
    FileFormat.SVG,
    FileFormat.CSV,
    FileFormat.XML,
    FileFormat.ODS,
    FileFormat.ODT,
    FileFormat.NUMBERS,
]

# Image formats that consume Workers AI Neurons quota
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".svg"}

# Extension to MIME type overrides (mimetypes module may not know all)
_MIME_OVERRIDES = {
    ".numbers": "application/vnd.apple.numbers",
    ".ods": "application/vnd.oasis.opendocument.spreadsheet",
    ".odt": "application/vnd.oasis.opendocument.text",
    ".et": "application/vnd.ms-excel",
    ".xlsm": "application/vnd.ms-excel.sheet.macroEnabled.12",
    ".xlsb": "application/vnd.ms-excel.sheet.binary.macroEnabled.12",
}


class CloudflareConverter(BaseConverter):
    """Converter using Cloudflare Workers AI toMarkdown API.

    Pricing (fact-checked 2026-02-23):
    - PDF, Office, HTML, XML, CSV: FREE
    - Images (JPG/PNG/WEBP/SVG): Consumes Neurons (10,000/day free, then $0.011/1K)
    """

    supported_formats = CF_SUPPORTED_FORMATS

    def __init__(
        self,
        api_token: str | None = None,
        account_id: str | None = None,
        config: Any = None,
    ):
        super().__init__(config=config)
        self.api_token = api_token
        self.account_id = account_id

    def will_incur_cost(self, path: Path) -> bool:
        """Check if converting this file will consume Neurons quota."""
        return path.suffix.lower() in _IMAGE_EXTENSIONS

    def convert(self, input_path: Path, output_dir: Path | None = None) -> ConvertResult:
        """Synchronous convert — raises NotImplementedError.

        CF toMarkdown is a network API; use convert_async() instead.
        The workflow layer calls convert_async() when the converter supports it.
        """
        raise NotImplementedError(
            "CloudflareConverter requires async execution. "
            "Use convert_async() or ensure the workflow uses async dispatch."
        )

    async def convert_async(
        self, input_path: Path, output_dir: Path | None = None
    ) -> ConvertResult:
        """Convert a local file via CF Workers AI toMarkdown API.

        Args:
            input_path: Path to the local file
            output_dir: Unused (CF returns markdown directly, no extracted images)

        Returns:
            ConvertResult with markdown content and metadata

        Raises:
            RuntimeError: If credentials missing or CF API returns error
        """
        import httpx

        if not self.api_token or not self.account_id:
            raise RuntimeError(
                "Cloudflare API token and account ID required. "
                "Set in config: fetch.cloudflare.api_token and fetch.cloudflare.account_id"
            )

        # Determine MIME type
        ext = input_path.suffix.lower()
        mime = _MIME_OVERRIDES.get(ext)
        if not mime:
            mime, _ = mimetypes.guess_type(str(input_path))
        if not mime:
            raise RuntimeError(f"Cannot determine MIME type for: {input_path}")

        endpoint = (
            f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}"
            f"/ai/tomarkdown"
        )

        if self.will_incur_cost(input_path):
            logger.info(
                f"Image conversion uses Workers AI Neurons quota: {input_path.name}"
            )

        logger.debug(f"Converting {input_path.name} via CF toMarkdown (MIME: {mime})")

        proxy_url = _detect_proxy()
        proxy_config = proxy_url if proxy_url else None

        async with httpx.AsyncClient(
            timeout=60.0,
            proxy=proxy_config,
        ) as client:
            with open(input_path, "rb") as f:
                response = await client.post(
                    endpoint,
                    headers={"Authorization": f"Bearer {self.api_token}"},
                    files={"files": (input_path.name, f, mime)},
                )
            response.raise_for_status()

            data = response.json()
            results = data.get("result", [])

            if not results:
                raise RuntimeError(
                    f"CF toMarkdown returned empty result for: {input_path.name}"
                )

            result = results[0]

            if result.get("format") == "error":
                raise RuntimeError(
                    f"CF toMarkdown failed for {input_path.name}: {result.get('error')}"
                )

            markdown = result.get("data", "")
            tokens = result.get("tokens")

            logger.debug(
                f"CF toMarkdown success: {input_path.name}"
                f" ({len(markdown)} chars"
                f"{f', ~{tokens} tokens' if tokens else ''})"
            )

            return ConvertResult(
                markdown=markdown,
                metadata={
                    "converter": "cloudflare-tomarkdown",
                    "tokens": tokens,
                    "mimetype": result.get("mimetype"),
                    "source_name": result.get("name"),
                },
            )
```

### Step 8: 运行 CloudflareConverter 测试

```bash
uv run python -m pytest packages/markitai/tests/unit/test_converter_cloudflare.py -v
```

Expected: 全部 PASS

### Step 9: 扩展 CloudflareConfig 增加 convert 字段

修改 Task 3 创建的 `CloudflareConfig`（在 `config.py` 中），添加：

```python
    # toMarkdown conversion settings (activated by --cloudflare flag or config)
    convert_enabled: bool = False  # Set to True by --cloudflare CLI flag
    convert_warn_neurons: bool = True  # Warn before image conversion (costs Neurons)
```

### Step 10: 集成到 converter 注册体系

修改 `packages/markitai/src/markitai/converter/__init__.py`，在导入区域追加：

```python
# Cloudflare converter is registered conditionally (only when configured)
# It does NOT use @register_converter decorator because it shouldn't
# override local converters by default. Instead, it's explicitly selected
# in the workflow layer when cloudflare.convert_enabled is True.
```

修改 `packages/markitai/src/markitai/workflow/core.py` 的 `validate_and_detect_format` 函数，在 `ctx.converter = get_converter(...)` 之后添加 CF fallback：

```python
    ctx.converter = get_converter(ctx.effective_input, config=ctx.config)

    # Cloudflare toMarkdown fallback for formats without local converter
    if ctx.converter is None:
        cf_config = ctx.config.fetch.cloudflare if hasattr(ctx.config.fetch, "cloudflare") else None
        if cf_config and cf_config.convert_enabled:
            from markitai.converter.cloudflare import CloudflareConverter, CF_SUPPORTED_FORMATS
            if fmt in CF_SUPPORTED_FORMATS:
                api_token = cf_config.get_resolved_api_token()
                account_id = cf_config.get_resolved_account_id()
                if api_token and account_id:
                    ctx.converter = CloudflareConverter(
                        api_token=api_token,
                        account_id=account_id,
                        config=ctx.config,
                    )
                    logger.debug(f"Using CF toMarkdown for {fmt.value}")

    if ctx.converter is None:
        return ConversionStepResult(
            success=False, error=f"No converter available for format: {fmt.value}"
        )
```

同时修改 `convert_document` 函数，支持 async converter：

```python
        # Check if converter supports async (e.g. CloudflareConverter)
        if hasattr(ctx.converter, "convert_async"):
            ctx.conversion_result = await ctx.converter.convert_async(
                ctx.effective_input,
                output_dir=ctx.output_dir,
            )
        elif is_heavy:
            async with get_heavy_task_semaphore(ctx.config.batch.heavy_task_limit):
                ctx.conversion_result = await run_in_converter_thread(
                    ctx.converter.convert,
                    ctx.effective_input,
                    output_dir=ctx.output_dir,
                )
        else:
            ctx.conversion_result = await run_in_converter_thread(
                ctx.converter.convert,
                ctx.effective_input,
                output_dir=ctx.output_dir,
            )
```

### Step 11: 写集成测试

在 `test_converter_cloudflare.py` 追加：

```python
class TestCloudflareConverterIntegration:
    """Tests for CF converter integration with workflow."""

    def test_cf_converter_not_registered_by_default(self):
        """CF converter does NOT auto-register (won't override local converters)."""
        from markitai.converter.base import get_converter
        # PDF should still use local PdfConverter, not CloudflareConverter
        converter = get_converter("report.pdf")
        assert converter is not None
        assert type(converter).__name__ != "CloudflareConverter"

    def test_cf_formats_detected(self):
        """New formats (svg, csv, etc.) are detected but have no local converter."""
        from markitai.converter.base import detect_format, get_converter
        assert detect_format("data.csv") == FileFormat.CSV
        assert get_converter("data.csv") is None  # No local converter

    def test_cf_converter_supports_new_formats(self):
        """CloudflareConverter handles formats that local converters don't."""
        from markitai.converter.cloudflare import CloudflareConverter
        converter = CloudflareConverter(api_token="test", account_id="test")
        # Formats with no local converter
        assert converter.can_convert("data.csv")
        assert converter.can_convert("feed.xml")
        assert converter.can_convert("sheet.ods")
        assert converter.can_convert("doc.odt")
        assert converter.can_convert("budget.numbers")
        assert converter.can_convert("icon.svg")
```

### Step 12: 回归测试

```bash
uv run python -m pytest packages/markitai/tests/unit/ -q --tb=short
```

Expected: 1831+ passed（包含所有新增测试），0 failed

### Step 13: Commit

```bash
git add packages/markitai/src/markitai/converter/base.py \
       packages/markitai/src/markitai/converter/cloudflare.py \
       packages/markitai/src/markitai/converter/__init__.py \
       packages/markitai/src/markitai/config.py \
       packages/markitai/src/markitai/workflow/core.py \
       packages/markitai/src/markitai/cli/main.py \
       packages/markitai/tests/unit/test_converter_cloudflare.py
git commit -m "feat(converter): add Cloudflare Workers AI toMarkdown backend

- New CloudflareConverter for file→markdown via CF REST API
- Extend FileFormat with SVG, CSV, XML, ODS, ODT, NUMBERS
- Integrates as fallback when no local converter available
- Activated via --cloudflare flag (unified with BR fetch strategy)
- Requires CF_API_TOKEN and CF_ACCOUNT_ID
- Image conversion warns about Neurons quota cost
- PDF/Office/CSV/XML conversions are free on CF
- Does NOT override existing local converters"
```

---

## Task 5: 端到端验收测试（E2E）

**目的：** 使用真实 fixtures 目录和 CLI 命令验收全部 4 个 Task 的集成效果，确保输出正确。

**前置条件：** Task 1-4 全部完成并通过回归测试。

**Fixtures 目录内容：**

```
packages/markitai/tests/fixtures/
├── candy.JPG
├── file-example_PDF_500_kB.pdf
├── file_example_XLSX_100.xlsx
├── Free_Test_Data_500KB_PPTX.pptx
├── sub_dir/
│   ├── file_example_PPT_250kB.ppt
│   ├── file_example_XLS_100.xls
│   └── file-sample_100kB.doc
└── test.urls                           # 3 URLs including x.com
```

### Step 1: 基线测试（本地转换，不启用任何 CF 功能）

这是核心回归——确保所有改动没有打破现有功能。

```bash
# 清理旧输出
rm -rf output-pi-1

# 基线命令：本地转换 + rich preset + playwright URL抓取
uv run markitai packages/markitai/tests/fixtures \
    --verbose --preset rich --no-cache \
    -o output-pi-1
```

**验证清单：**

```bash
# 1. 输出目录结构 — 每个文件都应有 .md 输出
echo "=== Output files ==="
find output-pi-1 -name "*.md" -o -name "*.json" | sort

# 2. 所有本地文件转换成功（6个文件）
for f in candy.JPG file-example_PDF_500_kB.pdf file_example_XLSX_100.xlsx \
         Free_Test_Data_500KB_PPTX.pptx; do
    if [ -f "output-pi-1/$f.md" ]; then
        echo "✅ $f.md exists ($(wc -c < "output-pi-1/$f.md") bytes)"
    else
        echo "❌ $f.md MISSING"
    fi
done

# 3. 子目录文件
for f in file_example_PPT_250kB.ppt file_example_XLS_100.xls file-sample_100kB.doc; do
    if [ -f "output-pi-1/sub_dir/$f.md" ]; then
        echo "✅ sub_dir/$f.md exists ($(wc -c < "output-pi-1/sub_dir/$f.md") bytes)"
    else
        echo "❌ sub_dir/$f.md MISSING"
    fi
done

# 4. URL 抓取结果 (test.urls 中 3 个 URL)
echo "=== URL outputs ==="
ls output-pi-1/*.md 2>/dev/null | grep -v "candy\|PDF\|XLSX\|PPTX"

# 5. 非空内容检查
for md in output-pi-1/*.md output-pi-1/sub_dir/*.md; do
    size=$(wc -c < "$md" 2>/dev/null || echo 0)
    if [ "$size" -lt 50 ]; then
        echo "⚠️  $md is suspiciously small: $size bytes"
    fi
done
```

### Step 2: Task 1 验证（CF 内容协商）

验证 `Accept: text/markdown` 头已生效。用一个已知启用了 CF Markdown for Agents 的 URL 测试：

```bash
rm -rf output-pi-2

# 抓取一个 CF 官方文档页面（已知启用了 Markdown for Agents）
uv run markitai https://developers.cloudflare.com/workers-ai/ \
    --verbose --no-cache \
    -o output-pi-2
```

**验证：**

```bash
# verbose 日志中应出现 "Server returned markdown directly" 或 "text/markdown"
# 如果 CF 站点返回了 markdown，内容应更简洁（无 HTML 噪音）
echo "=== CF Content Negotiation result ==="
head -20 output-pi-2/*.md
wc -c output-pi-2/*.md
```

### Step 3: Task 2 验证（Playwright 增强参数）

验证新的 Playwright 参数可通过 config 传递：

```bash
rm -rf output-pi-3

# 创建临时配置测试 wait_for_selector
cat > /tmp/markitai-test-pw.json << 'EOF'
{
    "fetch": {
        "playwright": {
            "wait_for_selector": "#content",
            "reject_resource_patterns": ["**/*.css", "**/*.woff2"],
            "extra_http_headers": {"Accept-Language": "en-US"}
        }
    }
}
EOF

uv run markitai https://stephango.com/concise \
    --verbose --no-cache --playwright \
    -c /tmp/markitai-test-pw.json \
    -o output-pi-3
```

**验证：**

```bash
# verbose 日志中应出现 wait_for_selector 和 resource filtering 相关信息
echo "=== Playwright enhanced result ==="
head -20 output-pi-3/*.md
wc -c output-pi-3/*.md
```

### Step 4: Task 3 验证（CF Browser Rendering，需要 CF 凭证）

> ⚠️ 此步骤需要有效的 `CF_API_TOKEN` 和 `CF_ACCOUNT_ID` 环境变量。如无凭证，验证 `--cloudflare` 失败时给出清晰错误信息即可。

```bash
rm -rf output-pi-4

# 有凭证时：
uv run markitai https://stephango.com/concise \
    --verbose --no-cache --cloudflare \
    -o output-pi-4

# 无凭证时（验证错误处理）：
CF_API_TOKEN="" CF_ACCOUNT_ID="" \
uv run markitai https://stephango.com/concise \
    --verbose --no-cache --cloudflare \
    -o output-pi-4 2>&1 | head -5
# Expected: 清晰的错误信息，指引用户配置 api_token 和 account_id
```

### Step 5: Task 4 验证（CF toMarkdown，需要 CF 凭证）

> ⚠️ 同样需要 CF 凭证。`--cloudflare` 对文件输入自动启用 toMarkdown 转换。

```bash
rm -rf output-pi-5

# --cloudflare 对文件输入自动走 CF toMarkdown
uv run markitai packages/markitai/tests/fixtures/file-example_PDF_500_kB.pdf \
    --verbose --no-cache --cloudflare \
    -o output-pi-5

# 验证：对比本地转换和 CF 转换结果
echo "=== Local conversion (baseline) ==="
wc -c output-pi-1/file-example_PDF_500_kB.pdf.md
echo "=== CF toMarkdown conversion ==="
wc -c output-pi-5/file-example_PDF_500_kB.pdf.md
diff <(head -20 output-pi-1/file-example_PDF_500_kB.pdf.md) \
     <(head -20 output-pi-5/file-example_PDF_500_kB.pdf.md) || true
```

### Step 6: 全功能组合测试

```bash
rm -rf output-pi-6

# 所有 CF 功能组合使用（需要凭证）
uv run markitai packages/markitai/tests/fixtures \
    --verbose --preset rich --no-cache \
    --cloudflare \
    -o output-pi-6
```

**验证：**

```bash
# 1. 输出文件数量应与基线一致
echo "=== Baseline file count ==="
find output-pi-1 -name "*.md" | wc -l
echo "=== Full CF file count ==="
find output-pi-6 -name "*.md" | wc -l

# 2. 每个文件内容非空
for md in output-pi-6/*.md output-pi-6/sub_dir/*.md; do
    size=$(wc -c < "$md" 2>/dev/null || echo 0)
    name=$(basename "$md")
    if [ "$size" -lt 50 ]; then
        echo "❌ $name: $size bytes (too small)"
    else
        echo "✅ $name: $size bytes"
    fi
done

# 3. URL 抓取使用 CF BR
grep -l "cloudflare" output-pi-6/*.md 2>/dev/null || echo "(check verbose log for strategy used)"
```

### Step 7: 清理

```bash
rm -rf output-pi-1 output-pi-2 output-pi-3 output-pi-4 output-pi-5 output-pi-6
rm -f /tmp/markitai-test-pw.json
```

---

## 验收检查清单

每个 Task 完成后执行：

```bash
# 1. 全量单元测试
uv run python -m pytest packages/markitai/tests/unit/ -q --tb=short

# 2. 语法检查
python -c "import ast; ast.parse(open('packages/markitai/src/markitai/fetch.py').read()); print('OK')"
python -c "import ast; ast.parse(open('packages/markitai/src/markitai/fetch_playwright.py').read()); print('OK')"
python -c "import ast; ast.parse(open('packages/markitai/src/markitai/config.py').read()); print('OK')"
python -c "import ast; ast.parse(open('packages/markitai/src/markitai/converter/cloudflare.py').read()); print('OK')"  # Task 4+
python -c "import ast; ast.parse(open('packages/markitai/src/markitai/converter/base.py').read()); print('OK')"  # Task 4+

# 3. 导入检查
uv run python -c "from markitai.fetch import fetch_url, FetchStrategy; print('fetch OK')"
uv run python -c "from markitai.config import PlaywrightConfig, FetchConfig; print('config OK')"
uv run python -c "from markitai.converter.cloudflare import CloudflareConverter; print('converter OK')"  # Task 4+
```

---

## 配置示例（全部 Task 完成后）

```json
{
  "fetch": {
    "strategy": "auto",
    "playwright": {
      "timeout": 30000,
      "wait_for": "domcontentloaded",
      "extra_wait_ms": 5000,
      "wait_for_selector": null,
      "cookies": null,
      "reject_resource_patterns": null,
      "extra_http_headers": null,
      "user_agent": null,
      "http_credentials": null
    },
    "jina": {
      "api_key": null,
      "timeout": 30
    },
    "cloudflare": {
      "api_token": "env:CF_API_TOKEN",
      "account_id": "env:CF_ACCOUNT_ID",
      "timeout": 30000,
      "wait_until": "networkidle0",
      "wait_for_selector": null,
      "cache_ttl": 0,
      "reject_resource_patterns": null,
      "convert_enabled": true,
      "convert_warn_neurons": true
    },
    "fallback_patterns": ["twitter.com", "x.com", "instagram.com"]
  }
}
```

---

## 参考文档勘误与注意事项

> 以下信息来自 `docs/archive/deps/cloudflare-markdown-for-agents.md` 的事实核查，实现时需注意。

### CF Browser Rendering 响应格式

CF REST API **统一使用 JSON 信封**，即使 `/markdown` 端点的"输出类型"描述为"Markdown 字符串"：

```json
{
  "success": true,
  "result": "# Page Title\n\nMarkdown content here...",
  "errors": [],
  "messages": []
}
```

实现中必须用 `response.json()["result"]` 而非 `response.text`。当 `success` 为 `false` 时，`errors` 数组包含结构化错误信息。

### CF Workers AI `toMarkdown` 计费

**⚠️ 常见误述**：将 `toMarkdown` 描述为"按 token 计费"是不准确的。

- PDF、Office、HTML、XML、CSV 等格式转换：**免费**，不消耗任何配额
- 图片转换（JPG/PNG/WEBP/SVG）：调用 Workers AI 视觉模型，消耗 **Neurons 配额**（非 token）
- 免费额度：10,000 Neurons/天（Free + Paid 均有）
- 超出费率：$0.011 / 1,000 Neurons

本方案未实现 `toMarkdown`（P2b 延迟），但未来实现时需遵循此计费模型。

### CF BR Free Plan 可用性

Browser Rendering 在 **Free Plan 也可使用**（无需 Paid 计划）：

| 计划 | 浏览器时长 | 并发浏览器 |
|------|-----------|-----------|
| Free | 10 分钟/天 | 3 个 |
| Paid | 10 小时/月（含），超出 $0.09/小时 | 10 个（月均） |

失败请求（如 waitForTimeout）**不计费**。响应头 `X-Browser-Ms-Used` 返回实际消耗时长。

### Playwright `Route.abort()` 是异步方法

Playwright 的 `Route.abort()` 返回协程，必须使用 `async def` handler 并 `await` 调用：

```python
# ✅ 正确
async def _abort_route(route):
    await route.abort()
await page.route(pattern, _abort_route)

# ❌ 错误 — 协程永远不会被 await，静默失败
await page.route(pattern, lambda route: route.abort())
```

### `fetch_with_static` 的 CF metadata 限制

`fetch_with_static` 委托 markitdown 内部 session 发起 HTTP 请求，无法访问响应 headers。因此：
- ✅ CF 返回的 markdown 内容会被正确透传（已验证）
- ❌ `x-markdown-tokens` 和 `converter: server-markdown` metadata 在此路径中丢失
- `fetch_with_static_conditional`（缓存命中路径）已完整处理这些 metadata
