"""The browser's render wait ends when the page settles, not after a fixed 3 s.

``fetch.playwright.extra_wait_ms`` used to be slept in full on every page,
and every auto-scroll was followed by another fixed 0.8 s pause, so a page
that was complete at load (Wikipedia, example.com) still took 4.4 s of
sleeping. The wait now ends once no script/XHR/fetch request is in flight
and the DOM has stopped changing; ``extra_wait_ms`` stays its upper bound.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from markitai.fetch_playwright import _PageActivity, _wait_until_settled


class _FakePage:
    """Emits Playwright request events; answers the DOM-quiet probe."""

    def __init__(self, dom_quiet_ms: Callable[[], Any] | float = 60_000) -> None:
        self._handlers: dict[str, list[Callable[[Any], None]]] = {}
        self._dom_quiet_ms = dom_quiet_ms

    def on(self, event: str, handler: Callable[[Any], None]) -> None:
        self._handlers.setdefault(event, []).append(handler)

    def emit(self, event: str, request: Any) -> None:
        for handler in self._handlers.get(event, []):
            handler(request)

    async def evaluate(self, _script: str) -> Any:
        value = self._dom_quiet_ms
        return value() if callable(value) else value


def _request(resource_type: str) -> SimpleNamespace:
    return SimpleNamespace(resource_type=resource_type)


async def _timed(page: _FakePage, activity: _PageActivity, limit_s: float):
    started = time.monotonic()
    settled = await _wait_until_settled(page, activity, limit_s, quiet_s=0.1)
    return settled, time.monotonic() - started


class TestWaitUntilSettled:
    async def test_a_quiet_page_settles_after_the_quiet_window(self) -> None:
        page = _FakePage()
        activity = _PageActivity(page)

        settled, elapsed = await _timed(page, activity, limit_s=3.0)

        assert settled
        assert elapsed < 1.0

    async def test_a_request_in_flight_holds_it_until_it_finishes(self) -> None:
        page = _FakePage()
        activity = _PageActivity(page)
        request = _request("xhr")
        page.emit("request", request)
        asyncio.get_running_loop().call_later(
            0.4, page.emit, "requestfinished", request
        )

        settled, elapsed = await _timed(page, activity, limit_s=3.0)

        assert settled
        # Finished at 0.4 s, then quiet for 0.1 s; asyncio timers may fire a
        # clock tick early, and Windows' clock moves in 15.6 ms ticks
        tick = time.get_clock_info("monotonic").resolution
        assert elapsed >= 0.5 - 3 * tick

    async def test_a_failed_request_counts_as_finished(self) -> None:
        page = _FakePage()
        activity = _PageActivity(page)
        request = _request("fetch")
        page.emit("request", request)
        page.emit("requestfailed", request)

        settled, _ = await _timed(page, activity, limit_s=3.0)

        assert settled

    @pytest.mark.parametrize(
        "resource_type",
        ["image", "font", "media", "stylesheet", "ping", "eventsource", "websocket"],
    )
    async def test_requests_that_cannot_change_the_dom_do_not_hold_it(
        self, resource_type: str
    ) -> None:
        """A stream or a slow image never finishing must not cost the bound."""
        page = _FakePage()
        activity = _PageActivity(page)
        page.emit("request", _request(resource_type))

        settled, elapsed = await _timed(page, activity, limit_s=3.0)

        assert settled
        assert elapsed < 1.0

    async def test_a_page_that_keeps_changing_waits_the_whole_bound(self) -> None:
        page = _FakePage(dom_quiet_ms=0)
        activity = _PageActivity(page)

        settled, elapsed = await _timed(page, activity, limit_s=0.4)

        assert not settled
        assert 0.35 <= elapsed < 1.5

    async def test_a_page_that_cannot_answer_counts_as_busy(self) -> None:
        """evaluate() fails while a navigation replaces the document."""

        def navigating() -> Any:
            raise RuntimeError("Execution context was destroyed")

        page = _FakePage(dom_quiet_ms=navigating)
        activity = _PageActivity(page)

        settled, _ = await _timed(page, activity, limit_s=0.3)

        assert not settled

    async def test_a_hung_page_cannot_overrun_the_bound(self) -> None:
        page = _FakePage()
        activity = _PageActivity(page)

        async def hang(_script: str) -> Any:
            await asyncio.sleep(30)

        page.evaluate = hang  # type: ignore[method-assign]

        settled, elapsed = await _timed(page, activity, limit_s=0.3)

        assert not settled
        assert elapsed < 1.5


def _renderer_with_page() -> tuple[Any, AsyncMock]:
    from markitai.fetch_playwright import PlaywrightRenderer

    page = AsyncMock()
    page.on = MagicMock()
    page.title = AsyncMock(return_value="Title")
    page.url = "https://example.com"
    page.content = AsyncMock(
        return_value="<html><body>" + "Test content. " * 100 + "</body></html>"
    )
    context = AsyncMock()
    context.new_page = AsyncMock(return_value=page)
    browser = AsyncMock()
    browser.new_context = AsyncMock(return_value=context)
    renderer = PlaywrightRenderer()
    renderer._browser = browser
    return renderer, page


class TestPauseAfterAutoScroll:
    async def test_text_only_fetch_does_not_pause_after_scrolling(self) -> None:
        renderer, page = _renderer_with_page()

        with patch("markitai.fetch_playwright.asyncio.sleep", AsyncMock()) as sleep:
            await renderer.fetch("https://example.com", extra_wait_ms=0)

        assert any(
            "scrollTo" in str(call.args[0]) for call in page.evaluate.await_args_list
        )
        sleep.assert_not_awaited()

    async def test_a_screenshot_still_waits_for_images_to_paint(
        self, tmp_path: Path
    ) -> None:
        renderer, _page = _renderer_with_page()
        shot = tmp_path / "shot.jpg"

        with (
            patch("markitai.fetch_playwright.asyncio.sleep", AsyncMock()) as sleep,
            patch(
                "markitai.fetch_playwright._capture_screenshot",
                AsyncMock(return_value=(shot, [shot])),
            ),
        ):
            await renderer.fetch(
                "https://example.com",
                extra_wait_ms=0,
                screenshot_config=MagicMock(enabled=True),
                output_dir=tmp_path,
            )

        sleep.assert_awaited_once_with(0.8)

    async def test_the_request_tracker_is_attached_before_navigation(self) -> None:
        renderer, page = _renderer_with_page()
        events: list[str] = []
        page.on.side_effect = lambda event, _handler: events.append(event)
        page.goto.side_effect = lambda *_a, **_k: events.append("goto")

        await renderer.fetch("https://example.com", extra_wait_ms=0)

        assert events.index("request") < events.index("goto")
