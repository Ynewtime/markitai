from __future__ import annotations

"""Tests for Playwright browser DOM normalization (live shadow DOM flattening).

These tests use page.set_content() with local fixture HTML so they do NOT require
network access, but they DO require Playwright and Chromium to be installed.
Marked as @pytest.mark.slow.
"""

import pathlib

import pytest

FIXTURES_DIR = pathlib.Path(__file__).parent.parent / "fixtures" / "web"
SHADOW_DOM_FIXTURE = FIXTURES_DIR / "shadow_dom_page.html"


def _skip_if_no_playwright_browser() -> None:
    """Skip test if Playwright is not installed or browsers are missing."""
    from markitai.fetch_playwright import is_playwright_available

    if not is_playwright_available():
        pytest.skip("playwright not installed")

    try:
        from playwright.sync_api import sync_playwright  # type: ignore[import]

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            browser.close()
    except Exception:  # noqa: BLE001
        pytest.skip("playwright browsers not installed")


@pytest.mark.slow
async def test_playwright_normalizes_open_shadow_dom_before_extraction() -> None:
    """Browser DOM normalize should flatten live shadow roots into light DOM.

    Uses page.set_content() with a local fixture — no network required.
    """
    _skip_if_no_playwright_browser()

    from playwright.async_api import async_playwright  # type: ignore[import]

    from markitai.fetch_playwright import _build_shadow_dom_normalize_script

    fixture_html = SHADOW_DOM_FIXTURE.read_text(encoding="utf-8")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            context = await browser.new_context()
            page = await context.new_page()
            await page.set_content(fixture_html)

            # Run the shadow DOM normalize script
            script = _build_shadow_dom_normalize_script()
            await page.evaluate(script)

            html_after = await page.content()
            assert "shadow text" in html_after

            await context.close()
        finally:
            await browser.close()


@pytest.mark.slow
async def test_playwright_normalizes_inline_shadow_dom() -> None:
    """Inline shadow DOM created via JS should also be flattened."""
    _skip_if_no_playwright_browser()

    from playwright.async_api import async_playwright  # type: ignore[import]

    from markitai.fetch_playwright import _build_shadow_dom_normalize_script

    # Build a page with a live shadow root attached via JS
    html_with_js_shadow = """
    <!DOCTYPE html>
    <html><body>
    <div id="host"></div>
    <script>
      const host = document.getElementById('host');
      const shadow = host.attachShadow({mode: 'open'});
      shadow.innerHTML = '<p>live shadow text</p>';
    </script>
    </body></html>
    """

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            context = await browser.new_context()
            page = await context.new_page()
            await page.set_content(html_with_js_shadow)

            script = _build_shadow_dom_normalize_script()
            await page.evaluate(script)

            html_after = await page.content()
            assert "live shadow text" in html_after

            await context.close()
        finally:
            await browser.close()
