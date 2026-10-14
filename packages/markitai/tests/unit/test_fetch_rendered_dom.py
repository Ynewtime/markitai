"""The copy of a browser-drawn page that ``markitai serve`` keeps.

These tests use ``page.set_content()`` with invented pages, so they need no
network, but they need Playwright and Chromium (skipped otherwise).
"""

from __future__ import annotations

import pytest

CSS_IN_JS_PAGE = """<!DOCTYPE html><html><head><style id="rules"></style></head>
<body><div class="card">Lamp benchmarks, posted by a reader.</div>
<script>
  // Rules inserted through the CSSOM never show up in the style's text
  document.getElementById("rules").sheet.insertRule(
    ".card { color: rgb(1, 2, 3); border: 2px solid rgb(4, 5, 6); }", 0);
  const sheet = new CSSStyleSheet();
  sheet.replaceSync("body { margin: 7px; }");
  document.adoptedStyleSheets = [sheet];
</script></body></html>"""

CHALLENGE_PAGE = """<!DOCTYPE html><html><body>
<div class="geetest_panel"><p>请在下图依次点击：</p><img src="data:image/gif;base64,R0lGODlhAQABAAAAACw=">
<button>确认</button></div></body></html>"""


async def _rendered(html: str) -> tuple[str | None, str | None]:
    from markitai.fetch_playwright import _rendered_page, is_playwright_available

    if not is_playwright_available():
        pytest.skip("playwright not installed")
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(headless=True)
        except Exception:
            pytest.skip("playwright browsers not installed")
        try:
            page = await browser.new_page()
            await page.set_content(html)
            return await _rendered_page(page)
        finally:
            await browser.close()


@pytest.mark.slow
async def test_style_rules_built_by_script_are_kept_and_scripts_are_not() -> None:
    html, problem = await _rendered(CSS_IN_JS_PAGE)
    assert problem is None
    assert html is not None
    assert ".card { color: rgb(1, 2, 3)" in html
    assert "body { margin: 7px; }" in html
    assert "<script" not in html
    assert html.startswith("<!DOCTYPE html>")


@pytest.mark.slow
async def test_a_verification_page_is_named() -> None:
    _, problem = await _rendered(CHALLENGE_PAGE)
    assert problem == "challenge"


@pytest.mark.slow
async def test_an_empty_page_is_named() -> None:
    _, problem = await _rendered("<html><body><div id='root'></div></body></html>")
    assert problem == "blank"
