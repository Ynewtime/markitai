"""Tests for URL batch mode screenshot parameter propagation.

Verifies that --screenshot and --screenshot-only flags work consistently
between single URL and batch URL processing paths.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from markitai.cli.processors.url import process_url_batch
from markitai.config import MarkitaiConfig


class MockUrlEntry:
    """Minimal UrlEntry substitute for tests."""

    def __init__(self, url: str, output_name: str | None = None):
        self.url = url
        self.output_name = output_name


def _make_fetch_result(
    url: str,
    screenshot_path: Path | None = None,
) -> MagicMock:
    """Create a mock FetchResult with sensible defaults."""
    result = MagicMock()
    result.content = f"# Content for {url}"
    result.cache_hit = False
    result.strategy_used = "playwright"
    result.screenshot_path = screenshot_path
    result.title = f"Title of {url}"
    result.static_content = None
    result.browser_content = None
    result.metadata = {}
    return result


class TestBatchScreenshotParamsPropagated:
    """Batch mode must forward screenshot/screenshot_dir/screenshot_config to fetch_url."""

    @pytest.mark.asyncio
    async def test_batch_passes_screenshot_params_to_fetch_url(
        self, tmp_path: Path
    ) -> None:
        """When cfg.screenshot.enabled is True, batch worker must pass
        screenshot=True, screenshot_dir, and screenshot_config to fetch_url().
        """
        entries = [MockUrlEntry("https://example.com/page1")]

        cfg = MarkitaiConfig()
        cfg.llm.enabled = False
        cfg.cache.enabled = False
        cfg.screenshot.enabled = True

        captured_kwargs: dict = {}

        async def mock_fetch(url, strategy, fetch_cfg, **kwargs):
            captured_kwargs.update(kwargs)
            return _make_fetch_result(url)

        with patch("markitai.fetch.fetch_url", side_effect=mock_fetch):
            await process_url_batch(
                url_entries=entries,
                output_dir=tmp_path,
                cfg=cfg,
                dry_run=False,
                verbose=False,
            )

        # The critical assertions: screenshot params must be forwarded
        assert captured_kwargs.get("screenshot") is True, (
            "batch worker must pass screenshot=True to fetch_url"
        )
        assert captured_kwargs.get("screenshot_dir") is not None, (
            "batch worker must pass screenshot_dir to fetch_url"
        )
        assert captured_kwargs.get("screenshot_config") is not None, (
            "batch worker must pass screenshot_config to fetch_url"
        )

    @pytest.mark.asyncio
    async def test_batch_no_screenshot_when_disabled(self, tmp_path: Path) -> None:
        """When cfg.screenshot.enabled is False, batch worker must NOT pass
        screenshot params to fetch_url().
        """
        entries = [MockUrlEntry("https://example.com/page1")]

        cfg = MarkitaiConfig()
        cfg.llm.enabled = False
        cfg.cache.enabled = False
        cfg.screenshot.enabled = False

        captured_kwargs: dict = {}

        async def mock_fetch(url, strategy, fetch_cfg, **kwargs):
            captured_kwargs.update(kwargs)
            return _make_fetch_result(url)

        with patch("markitai.fetch.fetch_url", side_effect=mock_fetch):
            await process_url_batch(
                url_entries=entries,
                output_dir=tmp_path,
                cfg=cfg,
                dry_run=False,
                verbose=False,
            )

        # screenshot should default to False / not present
        assert captured_kwargs.get("screenshot") in (False, None), (
            "batch worker must not enable screenshot when cfg.screenshot.enabled is False"
        )


class TestBatchScreenshotOnlyMode:
    """--screenshot-only in batch mode should behave like single URL mode."""

    @pytest.mark.asyncio
    async def test_batch_screenshot_only_without_llm_skips_md_output(
        self, tmp_path: Path
    ) -> None:
        """--screenshot-only without --llm: save screenshot, skip .md output."""
        entries = [MockUrlEntry("https://example.com/page1")]

        cfg = MarkitaiConfig()
        cfg.llm.enabled = False
        cfg.cache.enabled = False
        cfg.screenshot.enabled = True
        cfg.screenshot.screenshot_only = True

        # Simulate a screenshot being created by fetch_url
        screenshot_dir = tmp_path / ".markitai" / "screenshots"
        screenshot_dir.mkdir(parents=True)
        fake_screenshot = screenshot_dir / "page1.full.jpg"
        fake_screenshot.write_bytes(b"fake jpeg data")

        async def mock_fetch(url, strategy, fetch_cfg, **kwargs):
            return _make_fetch_result(url, screenshot_path=fake_screenshot)

        with patch("markitai.fetch.fetch_url", side_effect=mock_fetch):
            await process_url_batch(
                url_entries=entries,
                output_dir=tmp_path,
                cfg=cfg,
                dry_run=False,
                verbose=False,
            )

        # In screenshot-only mode without LLM, no .md file should be written
        md_files = list(tmp_path.glob("*.md"))
        assert len(md_files) == 0, (
            f"screenshot-only without LLM should not produce .md files, got {md_files}"
        )

    @pytest.mark.asyncio
    async def test_batch_screenshot_only_with_llm_extracts_from_screenshot(
        self, tmp_path: Path
    ) -> None:
        """--screenshot-only with --llm: extract content purely from screenshot."""
        entries = [MockUrlEntry("https://example.com/page1")]

        cfg = MarkitaiConfig()
        cfg.llm.enabled = True
        cfg.cache.enabled = False
        cfg.screenshot.enabled = True
        cfg.screenshot.screenshot_only = True

        # Simulate a screenshot being created by fetch_url
        screenshot_dir = tmp_path / ".markitai" / "screenshots"
        screenshot_dir.mkdir(parents=True)
        fake_screenshot = screenshot_dir / "page1.full.jpg"
        fake_screenshot.write_bytes(b"fake jpeg data")

        async def mock_fetch(url, strategy, fetch_cfg, **kwargs):
            return _make_fetch_result(url, screenshot_path=fake_screenshot)

        mock_screenshot_only = AsyncMock(
            return_value=("", 0.01, {"gpt-4-vision": {"requests": 1}})
        )

        with (
            patch("markitai.fetch.fetch_url", side_effect=mock_fetch),
            patch(
                "markitai.cli.processors.url.process_url_screenshot_only",
                mock_screenshot_only,
            ),
        ):
            await process_url_batch(
                url_entries=entries,
                output_dir=tmp_path,
                cfg=cfg,
                dry_run=False,
                verbose=False,
            )

        # process_url_screenshot_only should have been called
        mock_screenshot_only.assert_awaited_once()
        call_kwargs = mock_screenshot_only.call_args
        assert call_kwargs[1].get("url") == "https://example.com/page1" or (
            len(call_kwargs[0]) >= 2
            and call_kwargs[0][1] == "https://example.com/page1"
        )


class TestBatchScreenshotFrontmatter:
    """Batch mode should pass screenshot_path to _add_basic_frontmatter."""

    @pytest.mark.asyncio
    async def test_batch_frontmatter_includes_screenshot_path(
        self, tmp_path: Path
    ) -> None:
        """When a screenshot is captured, it should be referenced in frontmatter."""
        entries = [MockUrlEntry("https://example.com/page1")]

        cfg = MarkitaiConfig()
        cfg.llm.enabled = False
        cfg.cache.enabled = False
        cfg.screenshot.enabled = True

        screenshot_dir = tmp_path / ".markitai" / "screenshots"
        screenshot_dir.mkdir(parents=True)
        fake_screenshot = screenshot_dir / "page1.full.jpg"
        fake_screenshot.write_bytes(b"fake jpeg data")

        async def mock_fetch(url, strategy, fetch_cfg, **kwargs):
            return _make_fetch_result(url, screenshot_path=fake_screenshot)

        # We patch _add_basic_frontmatter to capture the screenshot_path arg
        with (
            patch("markitai.fetch.fetch_url", side_effect=mock_fetch),
            patch(
                "markitai.cli.processors.url._add_basic_frontmatter",
                return_value="---\nsource: test\n---\n\n# Content",
            ) as mock_frontmatter,
        ):
            await process_url_batch(
                url_entries=entries,
                output_dir=tmp_path,
                cfg=cfg,
                dry_run=False,
                verbose=False,
            )

            # Check that screenshot_path was passed
            assert mock_frontmatter.called
            call_kwargs = mock_frontmatter.call_args
            assert call_kwargs[1].get("screenshot_path") == fake_screenshot, (
                "batch worker should pass screenshot_path to _add_basic_frontmatter"
            )


class TestBatchScreenshotResultsTracking:
    """Batch mode should track screenshots count in results."""

    @pytest.mark.asyncio
    async def test_batch_results_include_screenshots_count(
        self, tmp_path: Path
    ) -> None:
        """Batch results should report screenshots count per URL."""
        import json

        entries = [MockUrlEntry("https://example.com/page1")]

        cfg = MarkitaiConfig()
        cfg.llm.enabled = False
        cfg.cache.enabled = False
        cfg.screenshot.enabled = True

        screenshot_dir = tmp_path / ".markitai" / "screenshots"
        screenshot_dir.mkdir(parents=True)
        fake_screenshot = screenshot_dir / "page1.full.jpg"
        fake_screenshot.write_bytes(b"fake jpeg data")

        async def mock_fetch(url, strategy, fetch_cfg, **kwargs):
            return _make_fetch_result(url, screenshot_path=fake_screenshot)

        with patch("markitai.fetch.fetch_url", side_effect=mock_fetch):
            await process_url_batch(
                url_entries=entries,
                output_dir=tmp_path,
                cfg=cfg,
                dry_run=False,
                verbose=False,
            )

        # Read the report (order_report transforms "urls" -> "url_sources")
        reports_dir = tmp_path / ".markitai" / "reports"
        report_files = list(reports_dir.glob("markitai.*.report.json"))
        assert len(report_files) == 1
        report = json.loads(report_files[0].read_text())

        # Navigate the hierarchical url_sources structure
        url_sources = report.get("url_sources", {})
        # Find the URL entry across all source groups
        url_result = None
        for _source, source_data in url_sources.items():
            urls = source_data.get("urls", {})
            if "https://example.com/page1" in urls:
                url_result = urls["https://example.com/page1"]
                break

        assert url_result is not None, (
            f"URL entry not found in report url_sources: {report.keys()}"
        )
        assert url_result.get("screenshots", 0) >= 1, (
            "batch results should track screenshots count"
        )
