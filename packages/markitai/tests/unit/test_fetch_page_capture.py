"""What the browser fetch leaves for a caller that keeps pages (contract v1.14).

Real Chromium on invented ``data:`` pages (no network); marked ``slow`` and
skipped without Playwright's browsers, like
``test_fetch_playwright_real_browser.py``. The serve side of the same data is
``tests/unit/serve/test_workbench_v114.py``.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pytest

from markitai.config import ScreenshotConfig
from markitai.page_capture import CAPTURE, COMPARE_TILES, CaptureSink

PICTURE = (
    "data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw=="
)
LONG_PAGE = (
    "<!doctype html><html><head><title>River notes</title><style>"
    "body{margin:0;font:18px sans-serif} img{display:block;width:400px;height:120px}"
    "</style></head><body><article><h1>River notes</h1>"
    + "".join(
        f"<h2>Day {day}</h2><p>{'The gauge at the old mill read the same at dawn. ' * 6}"
        f'</p><img src="{PICTURE}" alt="Mill {day}">'
        for day in range(1, 25)
    )
    + "</article></body></html>"
)
CHALLENGE_PAGE = (
    "<!doctype html><html><head><title>Just a moment...</title></head><body>"
    '<div id="cf-chl-widget"></div><p>Verify you are human by completing the '
    "action below.</p></body></html>"
)


def _skip_if_no_playwright_browser() -> None:
    from markitai.fetch_playwright import (
        is_playwright_available,
        is_playwright_browser_installed,
    )

    if not is_playwright_available():
        pytest.skip("playwright not installed")
    if not is_playwright_browser_installed(use_cache=False):
        pytest.skip("playwright browsers not installed")


def _data_url(html: str) -> str:
    return "data:text/html;charset=utf-8," + quote(html)


async def _fetch(html: str, sink: CaptureSink | None, **kwargs: Any) -> Any:
    from markitai.fetch_playwright import PlaywrightRenderer

    token = CAPTURE.set(sink)
    try:
        async with PlaywrightRenderer() as renderer:
            return await renderer.fetch(
                _data_url(html), timeout=20000, extra_wait_ms=0, **kwargs
            )
    finally:
        CAPTURE.reset(token)


def _size(data: bytes) -> tuple[int, int]:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        assert image.format == "JPEG"
        return image.size


@pytest.mark.slow
async def test_a_page_loaded_for_its_text_gets_compare_screenshots() -> None:
    _skip_if_no_playwright_browser()
    sink = CaptureSink(tile_height=1000)
    result = await _fetch(LONG_PAGE, sink)
    view = sink.view
    assert view is not None
    assert view.screenshots == []
    assert view.quality["state"] == "ok"
    assert view.quality["title"] == "River notes"
    sizes = [_size(tile) for tile in view.compare]
    assert len(sizes) > 2
    assert all(height == 1000 for _, height in sizes[:-1])
    assert 0 < sizes[-1][1] <= 1000
    assert view.blocks is not None
    blocks = json.loads(view.blocks)
    # The boxes cover exactly the tiles: their width, their heights stacked
    assert blocks["version"] == 1
    assert blocks["width"] == sizes[0][0]
    assert blocks["height"] == sum(height for _, height in sizes)
    tags = [block["tag"] for block in blocks["blocks"]]
    assert tags[:3] == ["article", "h1", "h2"]
    pictures = [block for block in blocks["blocks"] if block["tag"] == "img"]
    assert pictures[0]["text"] == "Mill 1" and pictures[0]["src"] == PICTURE
    # Boxes in document order, from the page's top (scrolled back to it)
    heads = [block["bbox"][1] for block in blocks["blocks"] if block["tag"] == "h2"]
    assert heads == sorted(heads) and heads[0] < 1000
    assert getattr(result, "screenshot_path", None) is None


@pytest.mark.slow
async def test_compare_screenshots_stop_after_their_last_tile() -> None:
    _skip_if_no_playwright_browser()
    sink = CaptureSink(tile_height=100)
    await _fetch(LONG_PAGE, sink)
    assert sink.view is not None
    assert len(sink.view.compare) == COMPARE_TILES
    assert sink.view.blocks is not None
    blocks = json.loads(sink.view.blocks)
    assert blocks["height"] == 100 * COMPARE_TILES
    assert all(block["bbox"][1] < blocks["height"] for block in blocks["blocks"])


@pytest.mark.slow
async def test_own_screenshots_carry_the_view_and_no_compare_copy(
    tmp_path: Path,
) -> None:
    _skip_if_no_playwright_browser()
    from PIL import Image

    sink = CaptureSink()
    result = await _fetch(
        LONG_PAGE,
        sink,
        screenshot_config=ScreenshotConfig(
            enabled=True, viewport_width=1000, viewport_height=700, tile_height=1500
        ),
        output_dir=tmp_path,
    )
    tiles = getattr(result, "screenshot_tiles", [])
    assert len(tiles) > 1
    assert sink.view is not None
    assert sink.view.screenshots == tiles
    assert sink.view.compare == []
    assert sink.view.blocks is not None
    blocks = json.loads(sink.view.blocks)
    heights = []
    for tile in tiles:
        with Image.open(tile) as image:
            assert image.size[0] == blocks["width"] == 1000
            heights.append(image.size[1])
    assert blocks["height"] == sum(heights)


@pytest.mark.slow
async def test_a_verification_page_is_set_aside_not_shown() -> None:
    _skip_if_no_playwright_browser()
    sink = CaptureSink()
    result = await _fetch(CHALLENGE_PAGE, sink)
    assert sink.view is None
    assert sink.refused is not None
    html, verdict = sink.refused
    assert "cf-chl-widget" in html
    assert (verdict["state"], verdict["reason"]) == ("challenge", "challenge")
    assert verdict["title"] == "Just a moment..."
    # The page is not the result's page
    assert getattr(result, "html", None) is None


@pytest.mark.slow
async def test_without_a_sink_nothing_is_read_or_taken() -> None:
    _skip_if_no_playwright_browser()
    result = await _fetch(LONG_PAGE, None)
    assert getattr(result, "screenshot_path", None) is None
    assert getattr(result, "html", None)
