# Dependency Simplification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Simplify dependency installation by replacing agent-browser with Playwright Python, reducing Node.js requirement and improving user experience

**Architecture:** Replace external CLI tool (agent-browser) with native Python library (Playwright). Use lazy loading for optional features. Provide fallback to Jina Reader API for users who don't want browser automation.

**Tech Stack:** playwright (Python), httpx (async HTTP), markitdown

---

## Background Analysis

### Current Pain Points

1. **agent-browser requires Node.js** (~100MB download)
2. **agent-browser requires Chromium** (~150MB download)
3. **Windows shim issues** - npm/pnpm CMD files depend on `/bin/sh`
4. **Version lock** - Must use 0.7.6 due to daemon bugs in 0.8.x
5. **Complex installation logic** - `Invoke-AgentBrowser` workarounds in setup scripts

### Current Dependency Structure

```
markitai
├── Core (required)
│   ├── pymupdf4llm     (PDF processing)
│   ├── markitdown      (Document conversion)
│   ├── litellm         (LLM abstraction)
│   ├── instructor      (Structured output)
│   ├── rapidocr        (OCR)
│   ├── click/loguru/rich (CLI)
│   ├── Pillow/opencv-python (Image)
│   └── pydantic        (Config)
├── Optional (extras)
│   ├── claude-agent-sdk (Claude CLI integration)
│   └── github-copilot-sdk (Copilot integration)
└── External (Node.js)
    └── agent-browser   (Browser automation) ← TO BE REPLACED
```

### Proposed Solution: Playwright Python

| Aspect | agent-browser | Playwright Python |
|--------|---------------|-------------------|
| Runtime | Node.js + Chromium | Pure Python + Chromium |
| Install | `npm -g` + PATH issues | `pip install` + `playwright install` |
| Size | ~150MB Chromium | ~150MB Chromium (same) |
| Windows | CMD shim bugs | Native support |
| API | CLI subprocess | Native async Python |
| Maintenance | External | Microsoft-backed |

### Migration Strategy

**Phase 1 (MVP):** Add Playwright as optional dependency, coexist with agent-browser
**Phase 2:** Make Playwright the default, deprecate agent-browser
**Phase 3:** Remove agent-browser support entirely

---

## Task List

### Task 1: Add Playwright to optional dependencies

**Files:**
- Modify: `packages/markitai/pyproject.toml:51-54`

**Step 1: Update pyproject.toml**

```toml
[project.optional-dependencies]
claude-agent = ["claude-agent-sdk>=0.1.0"]
copilot = ["github-copilot-sdk>=0.1.0"]
browser = ["playwright>=1.50.0"]  # NEW
all = ["claude-agent-sdk>=0.1.0", "github-copilot-sdk>=0.1.0", "playwright>=1.50.0"]  # UPDATED
```

**Step 2: Run uv sync**

```bash
cd packages/markitai && uv sync --all-extras
```

**Step 3: Commit**

```bash
git add packages/markitai/pyproject.toml
git commit -m "feat: add playwright as optional browser dependency"
```

---

### Task 2: Create Playwright fetch backend

**Files:**
- Create: `packages/markitai/src/markitai/fetch_playwright.py`
- Test: `packages/markitai/tests/unit/test_fetch_playwright.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_fetch_playwright.py
"""Tests for Playwright-based fetch backend."""

import pytest


class TestPlaywrightAvailable:
    """Tests for playwright availability check."""

    def test_is_playwright_available_not_installed(self, monkeypatch):
        """Test returns False when playwright not installed."""
        monkeypatch.setattr(
            "markitai.fetch_playwright.find_spec", lambda x: None
        )
        from markitai.fetch_playwright import is_playwright_available
        assert is_playwright_available() is False

    def test_is_playwright_available_installed(self, monkeypatch):
        """Test returns True when playwright is installed."""
        # Mock find_spec to return a non-None value
        class MockSpec:
            pass
        monkeypatch.setattr(
            "markitai.fetch_playwright.find_spec", lambda x: MockSpec()
        )
        from markitai.fetch_playwright import is_playwright_available
        assert is_playwright_available() is True
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_fetch_playwright.py -v
```

Expected: FAIL (module not found)

**Step 3: Write minimal implementation**

```python
# src/markitai/fetch_playwright.py
"""Playwright-based URL fetch backend.

This module provides browser automation using Playwright Python as an
alternative to agent-browser, eliminating the Node.js dependency.

Features:
- Pure Python implementation (no external CLI)
- Native async support
- Cross-platform (Windows/Linux/macOS)
- Automatic proxy detection
- Screenshot capture support

Usage:
    from markitai.fetch_playwright import fetch_with_playwright, is_playwright_available

    if is_playwright_available():
        result = await fetch_with_playwright(url, config)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from markitai.config import FetchConfig, ScreenshotConfig


def is_playwright_available() -> bool:
    """Check if playwright is installed.

    Returns:
        True if playwright can be imported
    """
    return find_spec("playwright") is not None


def is_playwright_browser_installed() -> bool:
    """Check if playwright browser is installed.

    Returns:
        True if at least one browser is available
    """
    if not is_playwright_available():
        return False

    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            # Check if chromium is available
            try:
                browser = p.chromium.launch(headless=True)
                browser.close()
                return True
            except Exception:
                return False
    except Exception:
        return False


@dataclass
class PlaywrightFetchResult:
    """Result from Playwright fetch."""
    content: str
    title: str | None = None
    final_url: str | None = None
    screenshot_path: Path | None = None
    metadata: dict[str, Any] | None = None


async def fetch_with_playwright(
    url: str,
    timeout: int = 30000,
    wait_for: str = "domcontentloaded",
    extra_wait_ms: int = 3000,
    proxy: str | None = None,
    screenshot_config: "ScreenshotConfig | None" = None,
    output_dir: Path | None = None,
) -> PlaywrightFetchResult:
    """Fetch URL using Playwright headless browser.

    Args:
        url: URL to fetch
        timeout: Page load timeout in milliseconds
        wait_for: Wait condition (load, domcontentloaded, networkidle)
        extra_wait_ms: Extra wait after load state
        proxy: Proxy URL (e.g., http://127.0.0.1:7890)
        screenshot_config: Screenshot settings
        output_dir: Directory for screenshots

    Returns:
        PlaywrightFetchResult with markdown content

    Raises:
        ImportError: If playwright is not installed
        RuntimeError: If browser is not installed
    """
    if not is_playwright_available():
        raise ImportError(
            "playwright is not installed. "
            "Install with: pip install playwright && playwright install chromium"
        )

    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        # Configure browser launch options
        launch_options: dict[str, Any] = {
            "headless": True,
        }

        if proxy:
            launch_options["proxy"] = {"server": proxy}

        browser = await p.chromium.launch(**launch_options)

        try:
            page = await browser.new_page()

            # Navigate to URL
            await page.goto(url, timeout=timeout, wait_until=wait_for)

            # Extra wait for JS rendering
            if extra_wait_ms > 0:
                await asyncio.sleep(extra_wait_ms / 1000)

            # Get page info
            title = await page.title()
            final_url = page.url

            # Get page content as markdown
            # Use page.content() and convert to markdown
            html_content = await page.content()
            markdown_content = _html_to_markdown(html_content)

            # Capture screenshot if requested
            screenshot_path = None
            if screenshot_config and screenshot_config.enabled and output_dir:
                screenshot_path = await _capture_screenshot(
                    page, screenshot_config, output_dir, url
                )

            return PlaywrightFetchResult(
                content=markdown_content,
                title=title,
                final_url=final_url,
                screenshot_path=screenshot_path,
                metadata={"renderer": "playwright", "wait_for": wait_for},
            )
        finally:
            await browser.close()


def _html_to_markdown(html: str) -> str:
    """Convert HTML to markdown using markitdown.

    Args:
        html: HTML content

    Returns:
        Markdown content
    """
    try:
        from markitdown import MarkItDown
        md = MarkItDown()
        # MarkItDown can convert HTML directly
        result = md.convert_html(html)
        return result.text_content if result else ""
    except Exception as e:
        logger.warning(f"HTML to markdown conversion failed: {e}")
        # Fallback: strip HTML tags
        import re
        text = re.sub(r"<[^>]+>", "", html)
        return text


async def _capture_screenshot(
    page: Any,
    config: "ScreenshotConfig",
    output_dir: Path,
    url: str,
) -> Path | None:
    """Capture page screenshot.

    Args:
        page: Playwright page object
        config: Screenshot configuration
        output_dir: Output directory
        url: Original URL (for filename)

    Returns:
        Path to screenshot file, or None on failure
    """
    try:
        import hashlib
        from datetime import datetime

        # Generate filename
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}_{url_hash}.png"

        output_dir.mkdir(parents=True, exist_ok=True)
        screenshot_path = output_dir / filename

        await page.screenshot(
            path=str(screenshot_path),
            full_page=config.full_page if hasattr(config, "full_page") else True,
            type="png",
        )

        logger.debug(f"Screenshot saved: {screenshot_path}")
        return screenshot_path
    except Exception as e:
        logger.warning(f"Screenshot capture failed: {e}")
        return None
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/test_fetch_playwright.py -v
```

**Step 5: Commit**

```bash
git add src/markitai/fetch_playwright.py tests/unit/test_fetch_playwright.py
git commit -m "feat: add Playwright-based fetch backend"
```

---

### Task 3: Integrate Playwright into fetch.py

**Files:**
- Modify: `packages/markitai/src/markitai/fetch.py`
- Modify: `packages/markitai/src/markitai/config.py`

**Step 1: Add new fetch strategy**

In `fetch.py`, add `PLAYWRIGHT` to `FetchStrategy` enum:

```python
class FetchStrategy(Enum):
    """URL fetch strategy."""
    AUTO = "auto"
    STATIC = "static"
    BROWSER = "browser"  # agent-browser (legacy)
    PLAYWRIGHT = "playwright"  # NEW: Playwright Python
    JINA = "jina"
```

**Step 2: Add Playwright fetch path**

In `fetch_url()` function, add handler for `PLAYWRIGHT` strategy:

```python
elif strategy == FetchStrategy.PLAYWRIGHT:
    from markitai.fetch_playwright import (
        fetch_with_playwright,
        is_playwright_available,
    )

    if not is_playwright_available():
        raise FetchError(
            "playwright is not installed. "
            "Install with: pip install playwright && playwright install chromium"
        )

    pw_result = await fetch_with_playwright(
        url,
        timeout=config.agent_browser_timeout,
        wait_for=config.agent_browser_wait_for,
        extra_wait_ms=config.agent_browser_extra_wait_ms,
        proxy=_detect_proxy() if config.auto_proxy else None,
        screenshot_config=screenshot_config,
        output_dir=output_dir,
    )

    result = FetchResult(
        content=pw_result.content,
        strategy_used="playwright",
        title=pw_result.title,
        url=url,
        final_url=pw_result.final_url,
        metadata=pw_result.metadata or {},
        screenshot_path=pw_result.screenshot_path,
    )
```

**Step 3: Update auto strategy priority**

In auto strategy logic, check Playwright before agent-browser:

```python
# Priority: static → playwright → agent-browser → jina
if is_playwright_available():
    # Use Playwright (preferred)
    result = await fetch_with_playwright(...)
elif is_agent_browser_available():
    # Fallback to agent-browser
    result = await fetch_with_browser(...)
elif jina_api_key:
    # Fallback to Jina
    result = await fetch_with_jina(...)
```

**Step 4: Commit**

```bash
git add src/markitai/fetch.py src/markitai/config.py
git commit -m "feat: integrate Playwright into fetch strategies"
```

---

### Task 4: Update CLI for Playwright

**Files:**
- Modify: `packages/markitai/src/markitai/cli/main.py`
- Modify: `packages/markitai/src/markitai/cli/commands/deps.py`

**Step 1: Add --playwright flag**

In CLI options, add:

```python
@click.option(
    "--playwright",
    is_flag=True,
    help="Use Playwright for browser rendering (recommended over --agent-browser)",
)
```

**Step 2: Update deps check command**

In `deps.py`, add Playwright check:

```python
def check_playwright():
    """Check Playwright installation status."""
    from markitai.fetch_playwright import (
        is_playwright_available,
        is_playwright_browser_installed,
    )

    if not is_playwright_available():
        return "not installed", "pip install playwright"

    if not is_playwright_browser_installed():
        return "browser not installed", "playwright install chromium"

    return "ready", None
```

**Step 3: Commit**

```bash
git add src/markitai/cli/main.py src/markitai/cli/commands/deps.py
git commit -m "feat: add --playwright CLI option"
```

---

### Task 5: Update setup scripts

**Files:**
- Modify: `scripts/lib.ps1`
- Modify: `scripts/lib.sh`
- Modify: `scripts/setup.ps1`
- Modify: `scripts/setup.sh`

**Step 1: Add Playwright installation option**

In setup scripts, add new option for browser automation:

```
Browser automation options:
  1. Playwright Python (recommended) - Pure Python, no Node.js needed
  2. agent-browser (legacy) - Requires Node.js
  3. Skip - Use Jina API or static fetching only
```

**Step 2: Add Install-Playwright function (PowerShell)**

```powershell
function Install-Playwright {
    Write-Info "Installing Playwright..."
    Write-Info "  Purpose: Browser automation for JavaScript-rendered pages"
    Write-Info "  Size: ~150MB (Chromium browser)"

    # Install via pip
    $cmdParts = $script:PYTHON_CMD -split " "
    $exe = $cmdParts[0]
    $baseArgs = if ($cmdParts.Length -gt 1) { $cmdParts[1..($cmdParts.Length-1)] } else { @() }

    $pipArgs = $baseArgs + @("-m", "pip", "install", "playwright")
    & $exe @pipArgs

    if ($LASTEXITCODE -eq 0) {
        # Install Chromium browser
        $playwrightArgs = $baseArgs + @("-m", "playwright", "install", "chromium")
        & $exe @playwrightArgs

        if ($LASTEXITCODE -eq 0) {
            Write-Success "Playwright installed successfully"
            Track-Install -Component "Playwright" -Status "installed"
            return $true
        }
    }

    Write-Error2 "Playwright installation failed"
    Track-Install -Component "Playwright" -Status "failed"
    return $false
}
```

**Step 3: Add install_playwright function (Bash)**

```bash
install_playwright() {
    print_info "Installing Playwright..."
    print_info "  Purpose: Browser automation for JavaScript-rendered pages"
    print_info "  Size: ~150MB (Chromium browser)"

    # Install via pip
    $PYTHON_CMD -m pip install playwright

    if [ $? -eq 0 ]; then
        # Install Chromium browser
        $PYTHON_CMD -m playwright install chromium

        if [ $? -eq 0 ]; then
            print_success "Playwright installed successfully"
            return 0
        fi
    fi

    print_error "Playwright installation failed"
    return 1
}
```

**Step 4: Commit**

```bash
git add scripts/lib.ps1 scripts/lib.sh scripts/setup.ps1 scripts/setup.sh
git commit -m "feat: add Playwright installation to setup scripts"
```

---

### Task 6: Update documentation

**Files:**
- Modify: `website/guide/getting-started.md`
- Modify: `website/zh/guide/getting-started.md`
- Modify: `website/guide/configuration.md`
- Modify: `website/zh/guide/configuration.md`

**Step 1: Update getting-started guide**

Add new section for browser automation:

```markdown
## Browser Automation (Optional)

For JavaScript-rendered pages (Twitter/X, SPAs, etc.), you need browser automation.

### Option 1: Playwright (Recommended)

Pure Python solution, no Node.js required:

\`\`\`bash
pip install playwright
playwright install chromium
\`\`\`

### Option 2: Jina Reader API

Cloud-based, no local installation needed:

\`\`\`bash
export JINA_API_KEY=your_api_key
\`\`\`

Free tier: 100 RPM, 10M tokens.

### Option 3: agent-browser (Legacy)

Requires Node.js 18+:

\`\`\`bash
pnpm add -g agent-browser@0.7.6
agent-browser install
\`\`\`
```

**Step 2: Update configuration guide**

Document new `--playwright` flag:

```markdown
## URL Fetch Strategies

| Strategy | Flag | Requirement | Best For |
|----------|------|-------------|----------|
| static | (default) | None | Most websites |
| playwright | `--playwright` | `playwright` package | JS-rendered pages |
| browser | `--agent-browser` | Node.js + agent-browser | Legacy support |
| jina | `--jina` | `JINA_API_KEY` | Cloud-based, no local deps |
| auto | `--auto` | Varies | Automatic fallback |
```

**Step 3: Commit**

```bash
git add website/
git commit -m "docs: update browser automation options"
```

---

### Task 7: Add deprecation warning for agent-browser

**Files:**
- Modify: `packages/markitai/src/markitai/fetch.py`
- Modify: `packages/markitai/src/markitai/cli/main.py`

**Step 1: Add deprecation warning**

When `--agent-browser` flag is used:

```python
import warnings

if use_agent_browser:
    warnings.warn(
        "agent-browser is deprecated and will be removed in v0.6.0. "
        "Please migrate to --playwright for browser automation.",
        DeprecationWarning,
        stacklevel=2,
    )
```

**Step 2: Log migration suggestion**

```python
logger.warning(
    "⚠️  agent-browser is deprecated. "
    "Consider migrating to Playwright: pip install playwright && playwright install chromium"
)
```

**Step 3: Commit**

```bash
git add src/markitai/fetch.py src/markitai/cli/main.py
git commit -m "chore: add deprecation warning for agent-browser"
```

---

### Task 8: Write comprehensive tests

**Files:**
- Create: `packages/markitai/tests/unit/test_fetch_playwright.py` (expanded)
- Create: `packages/markitai/tests/integration/test_playwright_fetch.py`

**Step 1: Unit tests**

```python
# tests/unit/test_fetch_playwright.py
"""Unit tests for Playwright fetch backend."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestIsPlaywrightAvailable:
    """Tests for is_playwright_available."""

    def test_not_installed(self, monkeypatch):
        """Returns False when playwright not installed."""
        monkeypatch.setattr(
            "markitai.fetch_playwright.find_spec", lambda x: None
        )
        from markitai.fetch_playwright import is_playwright_available

        # Need to reimport to get fresh state
        import importlib
        import markitai.fetch_playwright
        importlib.reload(markitai.fetch_playwright)

        assert markitai.fetch_playwright.is_playwright_available() is False


class TestHtmlToMarkdown:
    """Tests for HTML to Markdown conversion."""

    def test_basic_conversion(self):
        """Converts basic HTML to markdown."""
        from markitai.fetch_playwright import _html_to_markdown

        html = "<h1>Title</h1><p>Paragraph</p>"
        result = _html_to_markdown(html)

        assert "Title" in result
        assert "Paragraph" in result

    def test_strips_scripts(self):
        """Removes script tags."""
        from markitai.fetch_playwright import _html_to_markdown

        html = "<p>Content</p><script>alert('xss')</script>"
        result = _html_to_markdown(html)

        assert "alert" not in result
        assert "Content" in result
```

**Step 2: Integration tests (marked as network)**

```python
# tests/integration/test_playwright_fetch.py
"""Integration tests for Playwright fetch."""

import pytest


@pytest.mark.network
@pytest.mark.slow
class TestPlaywrightFetch:
    """Integration tests requiring network and Playwright."""

    @pytest.fixture
    def skip_if_no_playwright(self):
        """Skip if Playwright not installed."""
        from markitai.fetch_playwright import is_playwright_available
        if not is_playwright_available():
            pytest.skip("Playwright not installed")

    async def test_fetch_simple_page(self, skip_if_no_playwright):
        """Fetch a simple static page."""
        from markitai.fetch_playwright import fetch_with_playwright

        result = await fetch_with_playwright(
            "https://example.com",
            timeout=30000,
        )

        assert "Example Domain" in result.content
        assert result.title is not None
```

**Step 3: Commit**

```bash
git add tests/
git commit -m "test: add Playwright fetch backend tests"
```

---

## Verification Checklist

- [ ] `uv run pytest tests/unit/test_fetch_playwright.py -v` passes
- [ ] `uv run pyright` reports no errors
- [ ] `uv run ruff check --fix && uv run ruff format` passes
- [ ] Manual test: `markitai --playwright https://example.com` works
- [ ] Manual test: `markitai --playwright https://x.com/...` renders JS content
- [ ] Setup script offers Playwright as browser option

---

## Rollback Plan

If issues arise:
1. Keep agent-browser as fallback (already coexists)
2. Set `MARKITAI_USE_AGENT_BROWSER=1` env var to force legacy behavior
3. Remove `playwright` from dependencies if causing conflicts

---

## Future Work (Out of Scope)

- Remove agent-browser support entirely (v0.6.0)
- Add `--browser-type` flag for Firefox/WebKit support
- Add browser pool for concurrent fetching
- Add cookie/auth support for protected pages
