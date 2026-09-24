"""Screenshot capture semantics for URL fetches (mocked browser).

Regression tests for E2E findings: the capture was taken after the DOM
cleanup stripped styles/canvas/nav, post-load page calls had no timeout (a
script loop hung the run forever), ``-s playwright`` kept only the first
tile, screenshot files ignored the query string and left stale tiles, a
missing screenshot was silent, and ``--screenshot-only`` failed whenever the
text layer could not be extracted.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from markitai.config import FetchConfig, MarkitaiConfig, ScreenshotConfig
from markitai.fetch import fetch_url
from markitai.fetch_playwright import (
    PlaywrightFetchResult,
    PlaywrightPageTimeoutError,
    PlaywrightRenderer,
    _capture_screenshot,
)
from markitai.fetch_screenshot import (
    existing_screenshot_tiles,
    remove_stale_screenshot_tiles,
)
from markitai.fetch_types import FetchError, FetchResult, FetchStrategy
from markitai.runs import Outcome
from markitai.runs import json_output as json_result

LONG_HTML = "<html><body>" + "Test content. " * 100 + "</body></html>"


def _page(calls: list[str]) -> AsyncMock:
    page = AsyncMock()
    page.url = "https://example.com/page"
    page.title = AsyncMock(return_value="Title")
    page.content = AsyncMock(return_value=LONG_HTML)
    page.goto = AsyncMock(return_value=MagicMock(status=200))

    async def evaluate(script: str):
        if "scrollTo" in script:
            calls.append("scroll")
        elif "shadowRoot" in script:
            calls.append("shadow")
        elif "querySelectorAll(sel)" in script:
            calls.append("cleanup")
        return None

    page.evaluate = AsyncMock(side_effect=evaluate)
    return page


def _renderer(page: AsyncMock) -> tuple[PlaywrightRenderer, AsyncMock]:
    context = AsyncMock()
    context.new_page = AsyncMock(return_value=page)
    browser = AsyncMock()
    browser.new_context = AsyncMock(return_value=context)
    renderer = PlaywrightRenderer()
    renderer._browser = browser
    return renderer, context


class TestCaptureBeforeDomCleanup:
    @pytest.mark.asyncio
    async def test_screenshot_is_taken_before_any_dom_mutation(self, tmp_path):
        calls: list[str] = []
        page = _page(calls)
        renderer, _ = _renderer(page)

        async def capture(*args, **kwargs):
            calls.append("screenshot")
            return tmp_path / "shot.jpg", [tmp_path / "shot.jpg"]

        with (
            patch("markitai.fetch_playwright._capture_screenshot", side_effect=capture),
            patch("markitai.fetch_playwright.asyncio.sleep", new_callable=AsyncMock),
        ):
            await renderer.fetch(
                "https://example.com/page",
                extra_wait_ms=0,
                screenshot_config=ScreenshotConfig(enabled=True),
                output_dir=tmp_path,
            )
        assert calls == ["scroll", "screenshot", "shadow", "cleanup"]


class TestPostLoadDeadline:
    @pytest.mark.asyncio
    async def test_stuck_page_times_out_and_closes_context(self):
        page = _page([])
        forever = asyncio.Event()

        async def hang(script: str):
            await forever.wait()

        page.evaluate = AsyncMock(side_effect=hang)
        renderer, context = _renderer(page)

        started = asyncio.get_running_loop().time()
        with pytest.raises(PlaywrightPageTimeoutError, match="timeout=200ms"):
            await renderer.fetch(
                "https://example.com/loop",
                timeout=200,
                extra_wait_ms=0,
                skip_auto_scroll=True,
            )
        assert asyncio.get_running_loop().time() - started < 5
        context.close.assert_awaited()

    @pytest.mark.asyncio
    async def test_timeout_is_a_fetch_error_for_the_auto_chain(self):
        assert issubclass(PlaywrightPageTimeoutError, FetchError)

    @pytest.mark.asyncio
    async def test_stuck_persistent_context_is_evicted(self):
        page = _page([])

        async def hang():
            await asyncio.Event().wait()

        page.content = AsyncMock(side_effect=hang)
        renderer, context = _renderer(page)
        renderer.enable_domain_session_cache(ttl_seconds=600, max_contexts=4)

        with pytest.raises(PlaywrightPageTimeoutError):
            await renderer.fetch(
                "https://example.com/loop",
                timeout=200,
                extra_wait_ms=0,
                skip_auto_scroll=True,
                session_key="example.com",
                persist_context=True,
            )
        assert "example.com" not in renderer._context_cache
        context.close.assert_awaited()


class TestScreenshotHasItsOwnBudget:
    """A slow or stuck screenshot costs the screenshot, never the text."""

    @pytest.mark.asyncio
    async def test_hung_screenshot_keeps_the_text(self, tmp_path):
        page = _page([])

        async def hang(**kwargs):
            await asyncio.Event().wait()

        page.screenshot = AsyncMock(side_effect=hang)
        renderer, _ = _renderer(page)

        started = asyncio.get_running_loop().time()
        result = await renderer.fetch(
            "https://example.com/page",
            timeout=300,
            extra_wait_ms=0,
            skip_auto_scroll=True,
            screenshot_config=ScreenshotConfig(enabled=True),
            output_dir=tmp_path,
        )
        assert asyncio.get_running_loop().time() - started < 5
        assert "Test content." in result.content
        assert result.screenshot_path is None
        assert "timed out" in result.metadata["screenshot_error"]

    @pytest.mark.asyncio
    async def test_slow_screenshot_leaves_the_page_budget_to_extraction(self, tmp_path):
        """A capture that takes most of fetch.playwright.timeout used to
        leave title()/content() nothing and fail the whole fetch."""
        page = _page([])
        loop = asyncio.get_running_loop()
        shot = tmp_path / "shot.jpg"

        async def slow_capture(*args, **kwargs):
            await asyncio.sleep(0.25)
            return shot, [shot]

        async def slow_title():
            await asyncio.sleep(0.1)
            return "Title"

        page.title = AsyncMock(side_effect=slow_title)
        renderer, _ = _renderer(page)
        started = loop.time()
        with patch(
            "markitai.fetch_playwright._capture_screenshot", side_effect=slow_capture
        ):
            result = await renderer.fetch(
                "https://example.com/page",
                timeout=300,
                extra_wait_ms=0,
                skip_auto_scroll=True,
                screenshot_config=ScreenshotConfig(enabled=True),
                output_dir=tmp_path,
            )
        assert loop.time() - started >= 0.35  # capture + title exceeded 300ms
        assert result.screenshot_path == shot
        assert "screenshot_error" not in result.metadata
        assert "Test content." in result.content

    @pytest.mark.asyncio
    async def test_inner_screenshot_timeout_is_shorter_than_the_outer_bound(
        self, tmp_path
    ):
        from markitai.fetch_playwright import _capture_screenshot_within

        page = AsyncMock()
        page.screenshot = AsyncMock(side_effect=TimeoutError("Timeout 800ms"))
        path, tiles, error = await _capture_screenshot_within(
            page,
            ScreenshotConfig(enabled=True),
            tmp_path,
            "https://example.com/page",
            timeout_ms=1000,
        )
        assert (path, tiles) == (None, [])
        assert error is not None and "Timeout 800ms" in error
        assert 0 < page.screenshot.await_args.kwargs["timeout"] < 1000

    @pytest.mark.asyncio
    async def test_error_page_screenshot_timeout_keeps_the_enriched_result(
        self, tmp_path
    ):
        page = _page([])
        page.goto = AsyncMock(return_value=MagicMock(status=404))

        async def hang(**kwargs):
            await asyncio.Event().wait()

        page.screenshot = AsyncMock(side_effect=hang)
        renderer, _ = _renderer(page)
        enriched = PlaywrightFetchResult(content="# Tweet\n\nText", title="Tweet")
        with patch.object(
            renderer, "_enriched_result", AsyncMock(return_value=enriched)
        ):
            result = await renderer.fetch(
                "https://x.com/user/status/1",
                timeout=300,
                extra_wait_ms=0,
                screenshot_config=ScreenshotConfig(enabled=True),
                output_dir=tmp_path,
            )
        assert result.content == "# Tweet\n\nText"
        assert "timed out" in result.metadata["screenshot_error"]


class TestTimeoutMessageNamesTheStage:
    @pytest.mark.asyncio
    async def test_stage_that_outran_the_budget_is_named(self):
        from markitai.fetch_playwright import _PageDeadline

        deadline = _PageDeadline(0.05, 50)
        with pytest.raises(PlaywrightPageTimeoutError) as exc_info:
            await deadline.run(asyncio.Event().wait(), "page.content()")
        message = str(exc_info.value)
        assert "page.content() was still running" in message
        assert "timeout=50ms" in message

    @pytest.mark.asyncio
    async def test_exhausted_budget_is_not_blamed_on_the_next_call(self):
        from markitai.fetch_playwright import _PageDeadline

        deadline = _PageDeadline(0.0, 50)
        with pytest.raises(PlaywrightPageTimeoutError) as exc_info:
            await deadline.run(asyncio.sleep(0), "page.title()")
        assert "used up before page.title() started" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_extend_gives_back_time(self):
        from markitai.fetch_playwright import _PageDeadline

        deadline = _PageDeadline(0.0, 50)
        deadline.extend(5.0)
        assert deadline.remaining() > 4
        assert await deadline.run(asyncio.sleep(0, result="ok"), "x") == "ok"


class TestRedirectedPageBaseUrl:
    @pytest.mark.asyncio
    async def test_extraction_resolves_against_the_final_url(self):
        page = _page([])
        page.url = "https://example.com/new/place/article"
        renderer, _ = _renderer(page)
        seen: list[str] = []

        def extract(html: str, base_url: str):
            seen.append(base_url)
            raise RuntimeError("stop after recording the base URL")

        with patch(
            "markitai.fetch_playwright.extract_web_content", side_effect=extract
        ):
            result = await renderer.fetch(
                "https://example.com/old",
                extra_wait_ms=0,
                skip_auto_scroll=True,
            )
        assert seen == ["https://example.com/new/place/article"]
        assert result.final_url == "https://example.com/new/place/article"


class TestTilesAndFilenames:
    @pytest.mark.asyncio
    async def test_playwright_runner_keeps_every_tile(self, tmp_path):
        from markitai.fetch_session import get_default_session
        from markitai.fetch_strategies import StrategyContext
        from markitai.fetch_strategies.playwright import PlaywrightRunner

        tiles = [tmp_path / "a.full.jpg", tmp_path / "a.full--1.jpg"]
        pw = PlaywrightFetchResult(
            content="# Page\n\n" + "words " * 50,
            title="Page",
            final_url="https://example.com/a",
            screenshot_path=tiles[0],
            screenshot_tiles=tiles,
        )
        ctx = StrategyContext(
            config=FetchConfig(),
            session=get_default_session(),
            explicit=False,
            screenshot_kwargs={},
        )
        with patch(
            "markitai.fetch_playwright.fetch_with_playwright",
            AsyncMock(return_value=pw),
        ):
            result = await PlaywrightRunner().fetch("https://example.com/a", ctx)
        assert result.screenshot_tiles == tiles

    def test_stale_tiles_are_removed_and_listed_in_order(self, tmp_path):
        tmp_path = tmp_path / "shots"
        tmp_path.mkdir()
        primary = tmp_path / "example.com_page.full.jpg"
        primary.write_bytes(b"x")
        for i in (1, 2, 10):
            (tmp_path / f"example.com_page.full--{i}.jpg").write_bytes(b"x")
        (tmp_path / "example.com_page2.full--1.jpg").write_bytes(b"x")  # other URL

        assert [p.name for p in existing_screenshot_tiles(primary)] == [
            "example.com_page.full.jpg",
            "example.com_page.full--1.jpg",
            "example.com_page.full--2.jpg",
            "example.com_page.full--10.jpg",
        ]
        remove_stale_screenshot_tiles(primary)
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "example.com_page.full.jpg",
            "example.com_page2.full--1.jpg",
        ]

    @pytest.mark.asyncio
    async def test_recapture_drops_tiles_of_a_longer_previous_page(self, tmp_path):
        from PIL import Image

        from markitai.fetch_screenshot import _url_to_screenshot_filename

        tmp_path = tmp_path / "shots"
        tmp_path.mkdir()
        url = "https://example.com/page?n=5"
        name = _url_to_screenshot_filename(url)
        stem = name.removesuffix(".jpg")
        for i in range(1, 5):
            (tmp_path / f"{stem}--{i}.jpg").write_bytes(b"old tile")

        async def screenshot(path: str, **kwargs):
            Image.new("RGB", (100, 100), "white").save(path, "JPEG")

        page = AsyncMock()
        page.screenshot = AsyncMock(side_effect=screenshot)
        primary, tiles = await _capture_screenshot(
            page, ScreenshotConfig(enabled=True), tmp_path, url
        )
        assert primary == tmp_path / name
        assert tiles == [tmp_path / name]
        assert sorted(p.name for p in tmp_path.iterdir()) == [name]


def _text_result(url: str, **kwargs) -> FetchResult:
    return FetchResult(
        content="# Page\n\n" + "words " * 50, strategy_used="static", url=url, **kwargs
    )


class TestFetchUrlScreenshotErrors:
    @pytest.mark.asyncio
    async def test_missing_playwright_is_reported(self, tmp_path):
        url = "https://example.com/a"
        with (
            patch(
                "markitai.fetch._dispatch_strategy",
                AsyncMock(return_value=(_text_result(url), None)),
            ),
            patch(
                "markitai.fetch_playwright.is_playwright_available", return_value=False
            ),
            patch("markitai.fetch._get_playwright_renderer", AsyncMock()),
        ):
            result = await fetch_url(
                url,
                FetchStrategy.AUTO,
                FetchConfig(),
                screenshot=True,
                screenshot_dir=tmp_path,
                screenshot_config=ScreenshotConfig(enabled=True),
            )
        assert "Playwright is not installed" in result.metadata["screenshot_error"]

    @pytest.mark.asyncio
    async def test_browser_refusal_is_reported(self, tmp_path):
        url = "https://example.com/a"
        with (
            patch(
                "markitai.fetch._dispatch_strategy",
                AsyncMock(return_value=(_text_result(url), None)),
            ),
            patch(
                "markitai.fetch_playwright.is_playwright_available", return_value=True
            ),
            patch(
                "markitai.fetch_playwright.fetch_with_playwright",
                AsyncMock(
                    side_effect=FetchError("Playwright navigation returned HTTP 403")
                ),
            ),
            patch("markitai.fetch._get_playwright_renderer", AsyncMock()),
        ):
            result = await fetch_url(
                url,
                FetchStrategy.AUTO,
                FetchConfig(),
                screenshot=True,
                screenshot_dir=tmp_path,
                screenshot_config=ScreenshotConfig(enabled=True),
            )
        assert result.metadata["screenshot_error"] == (
            "Playwright navigation returned HTTP 403"
        )
        assert result.screenshot_path is None

    @pytest.mark.asyncio
    async def test_cached_screenshot_from_another_directory_is_recaptured(
        self, tmp_path
    ):
        from markitai.fetch_cache import FetchCache

        url = "https://example.com/a"
        old_dir = tmp_path / "old"
        old_dir.mkdir()
        old_shot = old_dir / "example.com_a.full.jpg"
        old_shot.write_bytes(b"x")
        cache = FetchCache(tmp_path / "fetch_cache.db")
        cache.set(url, _text_result(url, screenshot_path=old_shot))

        new_dir = tmp_path / "new"
        new_dir.mkdir()
        new_shot = new_dir / "example.com_a.full.jpg"
        new_shot.write_bytes(b"x")
        pw = PlaywrightFetchResult(
            content="", screenshot_path=new_shot, screenshot_tiles=[new_shot]
        )
        with (
            patch(
                "markitai.fetch_playwright.is_playwright_available", return_value=True
            ),
            patch(
                "markitai.fetch_playwright.fetch_with_playwright",
                AsyncMock(return_value=pw),
            ) as capture,
            patch("markitai.fetch._get_playwright_renderer", AsyncMock()),
        ):
            result = await fetch_url(
                url,
                FetchStrategy.AUTO,
                FetchConfig(),
                cache=cache,
                screenshot=True,
                screenshot_dir=new_dir,
                screenshot_config=ScreenshotConfig(enabled=True),
            )
        capture.assert_awaited_once()
        assert result.cache_hit
        assert result.screenshot_path == new_shot

    @pytest.mark.asyncio
    async def test_cached_screenshot_is_dropped_when_not_requested(self, tmp_path):
        from markitai.fetch_cache import FetchCache

        url = "https://example.com/a"
        shot = tmp_path / "example.com_a.full.jpg"
        shot.write_bytes(b"x")
        cache = FetchCache(tmp_path / "fetch_cache.db")
        cache.set(url, _text_result(url, screenshot_path=shot))
        result = await fetch_url(url, FetchStrategy.AUTO, FetchConfig(), cache=cache)
        assert result.cache_hit
        assert result.screenshot_path is None


class TestScreenshotOnlyFallback:
    def _config(self) -> ScreenshotConfig:
        return ScreenshotConfig(enabled=True, screenshot_only=True)

    @pytest.mark.asyncio
    async def test_failed_extraction_completes_from_the_screenshot(self, tmp_path):
        url = "https://example.com/canvas"
        shot = tmp_path / "example.com_canvas.full.jpg"
        shot.write_bytes(b"x")
        pw = PlaywrightFetchResult(
            content="",
            title="Canvas",
            final_url=url,
            screenshot_path=shot,
            screenshot_tiles=[shot],
        )
        with (
            patch(
                "markitai.fetch._dispatch_strategy",
                AsyncMock(side_effect=FetchError("All fetch strategies failed")),
            ),
            patch(
                "markitai.fetch_playwright.is_playwright_available", return_value=True
            ),
            patch(
                "markitai.fetch_playwright.fetch_with_playwright",
                AsyncMock(return_value=pw),
            ),
            patch("markitai.fetch._get_playwright_renderer", AsyncMock()),
        ):
            result = await fetch_url(
                url,
                FetchStrategy.AUTO,
                FetchConfig(),
                screenshot=True,
                screenshot_dir=tmp_path,
                screenshot_config=self._config(),
            )
        assert result.screenshot_path == shot
        assert result.strategy_used == "playwright"
        assert result.metadata["extraction_error"] == "All fetch strategies failed"

    @pytest.mark.asyncio
    async def test_no_screenshot_either_reports_both_causes(self, tmp_path):
        url = "https://example.com/canvas"
        with (
            patch(
                "markitai.fetch._dispatch_strategy",
                AsyncMock(side_effect=FetchError("All fetch strategies failed")),
            ),
            patch(
                "markitai.fetch_playwright.is_playwright_available", return_value=True
            ),
            patch(
                "markitai.fetch_playwright.fetch_with_playwright",
                AsyncMock(side_effect=FetchError("HTTP 403")),
            ),
            patch("markitai.fetch._get_playwright_renderer", AsyncMock()),
            pytest.raises(FetchError) as excinfo,
        ):
            await fetch_url(
                url,
                FetchStrategy.AUTO,
                FetchConfig(),
                screenshot=True,
                screenshot_dir=tmp_path,
                screenshot_config=self._config(),
            )
        assert "All fetch strategies failed" in str(excinfo.value)
        assert "screenshot: HTTP 403" in str(excinfo.value)


def _mock_fetch_result(url: str, **overrides) -> MagicMock:
    result = MagicMock()
    result.content = "# Page\n\n" + "words " * 50
    result.cache_hit = False
    result.strategy_used = "playwright"
    result.screenshot_path = None
    result.screenshot_tiles = []
    result.title = "Page"
    result.final_url = url
    result.static_content = None
    result.browser_content = None
    result.metadata = {}
    for key, value in overrides.items():
        setattr(result, key, value)
    return result


class TestSingleUrlCli:
    def _cfg(self, *, only: bool) -> MarkitaiConfig:
        cfg = MarkitaiConfig()
        cfg.llm.enabled = False
        cfg.cache.enabled = False
        cfg.screenshot.enabled = True
        cfg.screenshot.screenshot_only = only
        return cfg

    @pytest.mark.asyncio
    async def test_screenshot_only_without_screenshot_fails(self, tmp_path):
        from markitai.cli.processors.url import process_url

        url = "https://example.com/a"
        fetched = _mock_fetch_result(
            url, metadata={"screenshot_error": "Playwright navigation returned 403"}
        )
        history: list[Outcome] = []
        with (
            patch("markitai.fetch.fetch_url", AsyncMock(return_value=fetched)),
            pytest.raises(SystemExit) as excinfo,
        ):
            await process_url(
                url, tmp_path, self._cfg(only=True), False, False, history=history
            )
        assert excinfo.value.code == 1
        assert history[0].status == "failed"
        assert "403" in (history[0].error or "")

    @pytest.mark.asyncio
    async def test_screenshot_only_counts_tiles_and_records_strategy(self, tmp_path):
        from markitai.cli.processors.url import process_url

        url = "https://example.com/a"
        tiles = [tmp_path / f"t{i}.jpg" for i in range(3)]
        for tile in tiles:
            tile.write_bytes(b"x")
        fetched = _mock_fetch_result(
            url,
            content="",  # canvas page: no text layer
            screenshot_path=tiles[0],
            screenshot_tiles=tiles,
            metadata={"extraction_error": "All fetch strategies failed"},
        )
        history: list[Outcome] = []
        with patch("markitai.fetch.fetch_url", AsyncMock(return_value=fetched)):
            await process_url(
                url, tmp_path, self._cfg(only=True), False, False, history=history
            )
        outcome = history[0]
        assert outcome.status == "completed"
        assert outcome.screenshots == 3
        assert outcome.fetch_strategy == "playwright"
        assert outcome.output_path == tiles[0]
        assert outcome.warnings  # text layer replaced by the screenshot

    @pytest.mark.asyncio
    async def test_missing_screenshot_is_a_visible_warning(self, tmp_path, capsys):
        from markitai.cli.processors.url import process_url

        url = "https://example.com/a"
        fetched = _mock_fetch_result(
            url, metadata={"screenshot_error": "Playwright is not installed"}
        )
        history: list[Outcome] = []
        with patch("markitai.fetch.fetch_url", AsyncMock(return_value=fetched)):
            await process_url(
                url, tmp_path, self._cfg(only=False), False, False, history=history
            )
        outcome = history[0]
        assert outcome.status == "completed"
        assert outcome.warnings == [
            "Screenshot not captured: Playwright is not installed"
        ]
        envelope = json_result.build_envelope(history)
        assert envelope["items"][0]["warnings"] == outcome.warnings


class TestBatchCli:
    @pytest.mark.asyncio
    async def test_screenshot_only_without_screenshot_fails_the_item(self, tmp_path):
        from markitai.cli.processors.url import process_url_batch

        entry = MagicMock(url="https://example.com/a", output_name=None)
        cfg = MarkitaiConfig()
        cfg.llm.enabled = False
        cfg.cache.enabled = False
        cfg.screenshot.enabled = True
        cfg.screenshot.screenshot_only = True
        history: list[Outcome] = []
        fetched = _mock_fetch_result(entry.url)
        with (
            patch("markitai.fetch.fetch_url", AsyncMock(return_value=fetched)),
            pytest.raises(SystemExit),
        ):
            await process_url_batch(
                [entry], tmp_path, cfg, False, False, history=history
            )
        assert history[0].status == "failed"
        assert "--screenshot-only" in (history[0].error or "")

    @pytest.mark.asyncio
    async def test_batch_counts_tiles_and_warns_on_missing_screenshots(self, tmp_path):
        from markitai.cli.processors.url import process_url_batch

        tiles = [tmp_path / f"t{i}.jpg" for i in range(4)]
        for tile in tiles:
            tile.write_bytes(b"x")
        with_shot = MagicMock(url="https://example.com/a", output_name="a")
        without = MagicMock(url="https://example.com/b", output_name="b")
        cfg = MarkitaiConfig()
        cfg.llm.enabled = False
        cfg.cache.enabled = False
        cfg.screenshot.enabled = True

        async def fetch(url, *args, **kwargs):
            if url.endswith("/a"):
                return _mock_fetch_result(
                    url, screenshot_path=tiles[0], screenshot_tiles=tiles
                )
            return _mock_fetch_result(
                url, metadata={"screenshot_error": "Chromium browser is missing"}
            )

        history: list[Outcome] = []
        with patch("markitai.fetch.fetch_url", side_effect=fetch):
            await process_url_batch(
                [with_shot, without], tmp_path, cfg, False, False, history=history
            )
        by_source = {o.source: o for o in history}
        assert by_source["https://example.com/a"].screenshots == 4
        assert by_source["https://example.com/a"].warnings == []
        assert by_source["https://example.com/b"].status == "completed"
        assert by_source["https://example.com/b"].warnings == [
            "Screenshot not captured: Chromium browser is missing"
        ]


def test_json_envelope_always_has_warnings() -> None:
    envelope = json_result.build_envelope(
        [Outcome(kind="url", source="https://example.com", status="completed")]
    )
    assert envelope["items"][0]["warnings"] == []
