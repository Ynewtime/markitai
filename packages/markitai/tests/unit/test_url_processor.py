"""Unit tests for URL processor module."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from markitai.cli.processors.url import (
    build_multi_source_content,
    process_url,
    process_url_batch,
    process_url_screenshot_only,
    process_url_with_vision,
)
from markitai.config import MarkitaiConfig


class TestBuildMultiSourceContent:
    """Tests for build_multi_source_content function."""

    def test_returns_fallback_when_no_sources(self) -> None:
        """Test that fallback content is returned when no other sources."""
        result = build_multi_source_content(None, None, "fallback content")
        assert result == "fallback content"

    def test_returns_fallback_with_static_content(self) -> None:
        """Test that fallback is still returned even with static content."""
        # With single-source strategy, fallback is always the best source
        result = build_multi_source_content("static content", None, "fallback content")
        assert result == "fallback content"

    def test_returns_fallback_with_browser_content(self) -> None:
        """Test that fallback is still returned even with browser content."""
        result = build_multi_source_content(None, "browser content", "fallback content")
        assert result == "fallback content"

    def test_returns_fallback_with_all_sources(self) -> None:
        """Test that fallback is returned when all sources present."""
        result = build_multi_source_content(
            "static content", "browser content", "fallback content"
        )
        assert result == "fallback content"

    def test_strips_whitespace(self) -> None:
        """Test that whitespace is stripped from result."""
        result = build_multi_source_content(None, None, "  content with spaces  ")
        assert result == "content with spaces"

    def test_empty_fallback_returns_empty(self) -> None:
        """Test that empty fallback returns empty string."""
        result = build_multi_source_content("static", "browser", "")
        assert result == ""

    def test_whitespace_only_fallback(self) -> None:
        """Test that whitespace-only fallback returns empty string."""
        result = build_multi_source_content(None, None, "   \n\t   ")
        assert result == ""


class TestProcessUrlDryRun:
    """Tests for process_url dry run functionality."""

    @pytest.mark.asyncio
    async def test_dry_run_shows_url_info(self, tmp_path: Path) -> None:
        """Test that dry run displays URL information."""
        cfg = MarkitaiConfig()
        cfg.llm.enabled = False

        with pytest.raises(SystemExit) as exc_info:
            await process_url(
                url="https://example.com/test",
                output_dir=tmp_path,
                cfg=cfg,
                dry_run=True,
                verbose=False,
            )

        assert exc_info.value.code == 0

    @pytest.mark.asyncio
    async def test_dry_run_with_llm_enabled(self, tmp_path: Path) -> None:
        """Test dry run shows LLM status when enabled."""
        cfg = MarkitaiConfig()
        cfg.llm.enabled = True

        with pytest.raises(SystemExit) as exc_info:
            await process_url(
                url="https://example.com/test",
                output_dir=tmp_path,
                cfg=cfg,
                dry_run=True,
                verbose=False,
            )

        assert exc_info.value.code == 0

    @pytest.mark.asyncio
    async def test_dry_run_with_image_options(self, tmp_path: Path) -> None:
        """Test dry run shows image options status."""
        cfg = MarkitaiConfig()
        cfg.llm.enabled = False
        cfg.image.alt_enabled = True
        cfg.image.desc_enabled = True

        with pytest.raises(SystemExit) as exc_info:
            await process_url(
                url="https://example.com/test",
                output_dir=tmp_path,
                cfg=cfg,
                dry_run=True,
                verbose=False,
            )

        assert exc_info.value.code == 0

    @pytest.mark.asyncio
    async def test_dry_run_with_screenshot_enabled(self, tmp_path: Path) -> None:
        """Test dry run shows screenshot status when enabled."""
        cfg = MarkitaiConfig()
        cfg.llm.enabled = False
        cfg.screenshot.enabled = True

        with pytest.raises(SystemExit) as exc_info:
            await process_url(
                url="https://example.com/test",
                output_dir=tmp_path,
                cfg=cfg,
                dry_run=True,
                verbose=False,
            )

        assert exc_info.value.code == 0

    @pytest.mark.asyncio
    async def test_dry_run_with_cache_disabled(self, tmp_path: Path) -> None:
        """Test dry run shows cache status when disabled."""
        cfg = MarkitaiConfig()
        cfg.cache.enabled = False

        with pytest.raises(SystemExit) as exc_info:
            await process_url(
                url="https://example.com/test",
                output_dir=tmp_path,
                cfg=cfg,
                dry_run=True,
                verbose=False,
            )

        assert exc_info.value.code == 0


class TestProcessUrlBatchDryRun:
    """Tests for process_url_batch dry run functionality."""

    @pytest.mark.asyncio
    async def test_batch_dry_run_shows_url_count(self, tmp_path: Path) -> None:
        """Test that batch dry run shows URL count."""

        # Create mock UrlEntry objects
        class MockUrlEntry:
            def __init__(self, url: str, output_name: str | None = None):
                self.url = url
                self.output_name = output_name

        entries = [
            MockUrlEntry("https://example.com/page1"),
            MockUrlEntry("https://example.com/page2"),
            MockUrlEntry("https://example.com/page3", "custom_name"),
        ]

        cfg = MarkitaiConfig()
        cfg.llm.enabled = False

        with pytest.raises(SystemExit) as exc_info:
            await process_url_batch(
                url_entries=entries,
                output_dir=tmp_path,
                cfg=cfg,
                dry_run=True,
                verbose=False,
            )

        assert exc_info.value.code == 0

    @pytest.mark.asyncio
    async def test_batch_dry_run_with_llm(self, tmp_path: Path) -> None:
        """Test batch dry run with LLM enabled."""

        class MockUrlEntry:
            def __init__(self, url: str, output_name: str | None = None):
                self.url = url
                self.output_name = output_name

        entries = [MockUrlEntry("https://example.com")]

        cfg = MarkitaiConfig()
        cfg.llm.enabled = True

        with pytest.raises(SystemExit) as exc_info:
            await process_url_batch(
                url_entries=entries,
                output_dir=tmp_path,
                cfg=cfg,
                dry_run=True,
                verbose=False,
            )

        assert exc_info.value.code == 0

    @pytest.mark.asyncio
    async def test_batch_dry_run_with_many_urls(self, tmp_path: Path) -> None:
        """Test batch dry run truncates display for many URLs."""

        class MockUrlEntry:
            def __init__(self, url: str, output_name: str | None = None):
                self.url = url
                self.output_name = output_name

        # Create more than 10 entries to trigger truncation
        entries = [MockUrlEntry(f"https://example.com/page{i}") for i in range(15)]

        cfg = MarkitaiConfig()
        cfg.llm.enabled = False

        with pytest.raises(SystemExit) as exc_info:
            await process_url_batch(
                url_entries=entries,
                output_dir=tmp_path,
                cfg=cfg,
                dry_run=True,
                verbose=False,
            )

        assert exc_info.value.code == 0


class TestProcessUrlFetchErrors:
    """Tests for process_url error handling."""

    @pytest.mark.asyncio
    async def test_empty_content_error(self, tmp_path: Path) -> None:
        """Test that empty content from fetch raises error."""
        cfg = MarkitaiConfig()
        cfg.llm.enabled = False
        cfg.cache.enabled = False

        mock_result = MagicMock()
        mock_result.content = ""
        mock_result.cache_hit = False
        mock_result.strategy_used = "static"
        mock_result.screenshot_path = None
        mock_result.title = None

        with (
            patch(
                "markitai.fetch.fetch_url",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            await process_url(
                url="https://example.com/empty",
                output_dir=tmp_path,
                cfg=cfg,
                dry_run=False,
                verbose=False,
            )

        assert exc_info.value.code == 1

    @pytest.mark.asyncio
    async def test_whitespace_only_content_error(self, tmp_path: Path) -> None:
        """Test that whitespace-only content from fetch raises error."""
        cfg = MarkitaiConfig()
        cfg.llm.enabled = False
        cfg.cache.enabled = False

        mock_result = MagicMock()
        mock_result.content = "   \n\t   "
        mock_result.cache_hit = False
        mock_result.strategy_used = "static"
        mock_result.screenshot_path = None
        mock_result.title = None

        with (
            patch(
                "markitai.fetch.fetch_url",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            await process_url(
                url="https://example.com/whitespace",
                output_dir=tmp_path,
                cfg=cfg,
                dry_run=False,
                verbose=False,
            )

        assert exc_info.value.code == 1

    @pytest.mark.asyncio
    async def test_fetch_error_handling(self, tmp_path: Path) -> None:
        """Test that FetchError is handled properly."""
        from markitai.fetch import FetchError

        cfg = MarkitaiConfig()
        cfg.llm.enabled = False
        cfg.cache.enabled = False

        with (
            patch(
                "markitai.fetch.fetch_url",
                new_callable=AsyncMock,
                side_effect=FetchError("Connection failed"),
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            await process_url(
                url="https://example.com/error",
                output_dir=tmp_path,
                cfg=cfg,
                dry_run=False,
                verbose=False,
            )

        assert exc_info.value.code == 1

    @pytest.mark.asyncio
    async def test_jina_rate_limit_error_handling(self, tmp_path: Path) -> None:
        """Test that JinaRateLimitError is handled properly."""
        from markitai.fetch import JinaRateLimitError

        cfg = MarkitaiConfig()
        cfg.llm.enabled = False
        cfg.cache.enabled = False

        with (
            patch(
                "markitai.fetch.fetch_url",
                new_callable=AsyncMock,
                side_effect=JinaRateLimitError(),
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            await process_url(
                url="https://example.com/ratelimit",
                output_dir=tmp_path,
                cfg=cfg,
                dry_run=False,
                verbose=False,
            )

        assert exc_info.value.code == 1


class TestProcessUrlWithVision:
    """Tests for process_url_with_vision function."""

    @pytest.mark.asyncio
    async def test_vision_processing_success(self, tmp_path: Path) -> None:
        """Test successful vision processing."""
        cfg = MarkitaiConfig()
        cfg.llm.enabled = True

        screenshot_path = tmp_path / ".markitai" / "screenshots" / "test.full.jpg"
        screenshot_path.parent.mkdir(parents=True)
        screenshot_path.write_bytes(b"fake image data")

        output_file = tmp_path / "output.md"
        output_file.write_text("# Original content")

        mock_processor = MagicMock()
        mock_processor.enhance_url_with_vision = AsyncMock(
            return_value=("Cleaned content", {"title": "Test Page"})
        )
        mock_processor.format_llm_output = MagicMock(
            return_value="---\ntitle: Test Page\n---\n\nCleaned content"
        )
        mock_processor.get_context_cost = MagicMock(return_value=0.001)
        mock_processor.get_context_usage = MagicMock(
            return_value={"gpt-4": {"input_tokens": 100, "output_tokens": 50}}
        )

        content, cost, usage = await process_url_with_vision(
            content="# Original content",
            screenshot_path=screenshot_path,
            url="https://example.com",
            cfg=cfg,
            output_file=output_file,
            processor=mock_processor,
        )

        assert content == "# Original content"
        assert cost == 0.001
        assert "gpt-4" in usage

        # Verify LLM output file was created
        llm_output = output_file.with_suffix(".llm.md")
        assert llm_output.exists()
        llm_content = llm_output.read_text()
        assert "Screenshot for reference" in llm_content

    @pytest.mark.asyncio
    async def test_vision_processing_success_forwards_source_metadata(
        self, tmp_path: Path
    ) -> None:
        """Successful vision processing should forward fetch metadata."""
        cfg = MarkitaiConfig()
        cfg.llm.enabled = True

        screenshot_path = tmp_path / ".markitai" / "screenshots" / "test.full.jpg"
        screenshot_path.parent.mkdir(parents=True)
        screenshot_path.write_bytes(b"fake image data")

        output_file = tmp_path / "output.md"
        output_file.write_text("# Original content")

        mock_processor = MagicMock()
        mock_processor.enhance_url_with_vision = AsyncMock(
            return_value=("Cleaned content", {"title": "Test Page"})
        )
        mock_processor.format_llm_output = MagicMock(return_value="formatted")
        mock_processor.get_context_cost = MagicMock(return_value=0.0)
        mock_processor.get_context_usage = MagicMock(return_value={})

        extra_meta = {"author": "Jane", "published": "2024-01-15"}

        await process_url_with_vision(
            content="# Original content",
            screenshot_path=screenshot_path,
            url="https://example.com",
            cfg=cfg,
            output_file=output_file,
            processor=mock_processor,
            fetch_strategy="defuddle",
            extra_meta=extra_meta,
        )

        mock_processor.enhance_url_with_vision.assert_awaited_once_with(
            "# Original content",
            screenshot_path,
            context="https://example.com",
            original_title=None,
            fetch_strategy="defuddle",
            extra_meta=extra_meta,
        )

    @pytest.mark.asyncio
    async def test_vision_processing_fallback_on_error(self, tmp_path: Path) -> None:
        """Test fallback to standard processing when vision fails."""
        cfg = MarkitaiConfig()
        cfg.llm.enabled = True

        screenshot_path = tmp_path / ".markitai" / "screenshots" / "test.full.jpg"
        screenshot_path.parent.mkdir(parents=True)
        screenshot_path.write_bytes(b"fake image data")

        output_file = tmp_path / "output.md"
        output_file.write_text("# Original content")

        mock_processor = MagicMock()
        mock_processor.enhance_url_with_vision = AsyncMock(
            side_effect=Exception("Vision API error")
        )

        # Mock the fallback process_with_llm
        with patch(
            "markitai.cli.processors.llm.process_with_llm",
            new_callable=AsyncMock,
            return_value=("fallback content", 0.002, {"model": {"requests": 1}}),
        ):
            result = await process_url_with_vision(
                content="# Original content",
                screenshot_path=screenshot_path,
                url="https://example.com",
                cfg=cfg,
                output_file=output_file,
                processor=mock_processor,
            )

            assert result[0] == "fallback content"
            assert result[1] == 0.002


class TestProcessUrlScreenshotOnly:
    """Tests for process_url_screenshot_only function."""

    @pytest.mark.asyncio
    async def test_screenshot_only_success(self, tmp_path: Path) -> None:
        """Test successful screenshot-only extraction."""
        cfg = MarkitaiConfig()
        cfg.llm.enabled = True

        screenshot_path = tmp_path / ".markitai" / "screenshots" / "test.full.jpg"
        screenshot_path.parent.mkdir(parents=True)
        screenshot_path.write_bytes(b"fake image data")

        output_file = tmp_path / "output.md"
        output_file.write_text("# Placeholder")

        mock_processor = MagicMock()
        mock_processor.extract_from_screenshot = AsyncMock(
            return_value=("Extracted content from screenshot", {"title": "Extracted"})
        )
        mock_processor.format_llm_output = MagicMock(
            return_value="---\ntitle: Extracted\n---\n\nExtracted content"
        )
        mock_processor.get_context_cost = MagicMock(return_value=0.005)
        mock_processor.get_context_usage = MagicMock(
            return_value={"gpt-4-vision": {"input_tokens": 500, "output_tokens": 200}}
        )

        content, cost, usage = await process_url_screenshot_only(
            screenshot_path=screenshot_path,
            url="https://example.com",
            cfg=cfg,
            output_file=output_file,
            processor=mock_processor,
        )

        # Returns empty string as content is extracted purely from screenshot
        assert content == ""
        assert cost == 0.005
        assert "gpt-4-vision" in usage

        # Verify LLM output file was created
        llm_output = output_file.with_suffix(".llm.md")
        assert llm_output.exists()
        llm_content = llm_output.read_text()
        assert "Screenshot for reference" in llm_content

    @pytest.mark.asyncio
    async def test_screenshot_only_with_title(self, tmp_path: Path) -> None:
        """Test screenshot-only extraction preserves original title."""
        cfg = MarkitaiConfig()
        cfg.llm.enabled = True

        screenshot_path = tmp_path / ".markitai" / "screenshots" / "test.full.jpg"
        screenshot_path.parent.mkdir(parents=True)
        screenshot_path.write_bytes(b"fake image data")

        output_file = tmp_path / "output.md"

        mock_processor = MagicMock()
        mock_processor.extract_from_screenshot = AsyncMock(
            return_value=("Content", {"title": "Preserved Title"})
        )
        mock_processor.format_llm_output = MagicMock(return_value="formatted")
        mock_processor.get_context_cost = MagicMock(return_value=0.0)
        mock_processor.get_context_usage = MagicMock(return_value={})

        await process_url_screenshot_only(
            screenshot_path=screenshot_path,
            url="https://example.com",
            cfg=cfg,
            output_file=output_file,
            processor=mock_processor,
            original_title="Original Page Title",
        )

        # Verify extract_from_screenshot was called with original_title
        mock_processor.extract_from_screenshot.assert_called_once()
        call_kwargs = mock_processor.extract_from_screenshot.call_args.kwargs
        assert call_kwargs.get("original_title") == "Original Page Title"

    @pytest.mark.asyncio
    async def test_screenshot_only_raises_on_error(self, tmp_path: Path) -> None:
        """Test that screenshot-only raises error on extraction failure."""
        cfg = MarkitaiConfig()
        cfg.llm.enabled = True

        screenshot_path = tmp_path / ".markitai" / "screenshots" / "test.full.jpg"
        screenshot_path.parent.mkdir(parents=True)
        screenshot_path.write_bytes(b"fake image data")

        output_file = tmp_path / "output.md"

        mock_processor = MagicMock()
        mock_processor.extract_from_screenshot = AsyncMock(
            side_effect=Exception("Extraction failed")
        )

        with pytest.raises(Exception, match="Extraction failed"):
            await process_url_screenshot_only(
                screenshot_path=screenshot_path,
                url="https://example.com",
                cfg=cfg,
                output_file=output_file,
                processor=mock_processor,
            )


class TestProcessUrlSuccessPath:
    """Tests for successful URL processing paths."""

    @pytest.mark.asyncio
    async def test_basic_url_conversion(self, tmp_path: Path) -> None:
        """Test basic URL conversion without LLM."""
        cfg = MarkitaiConfig()
        cfg.llm.enabled = False
        cfg.cache.enabled = False

        mock_result = MagicMock()
        mock_result.content = "# Test Page\n\nSome content here."
        mock_result.cache_hit = False
        mock_result.strategy_used = "static"
        mock_result.screenshot_path = None
        mock_result.title = "Test Page"
        mock_result.static_content = None
        mock_result.browser_content = None

        with patch(
            "markitai.fetch.fetch_url",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            # The function outputs to console and raises SystemExit(0) implicitly
            # We need to capture the output file creation
            try:
                await process_url(
                    url="https://example.com/test",
                    output_dir=tmp_path,
                    cfg=cfg,
                    dry_run=False,
                    verbose=False,
                )
            except SystemExit:
                pass  # Expected for console output mode

        # Check output file was created
        output_file = tmp_path / "test.md"
        assert output_file.exists()
        content = output_file.read_text()
        assert "Test Page" in content
        assert "source:" in content  # frontmatter

    @pytest.mark.asyncio
    async def test_single_url_report_preserves_llm_model_names(
        self, tmp_path: Path
    ) -> None:
        """Single URL reports should keep per-model usage on each URL entry."""
        cfg = MarkitaiConfig()
        cfg.llm.enabled = True
        cfg.cache.enabled = False

        mock_result = MagicMock()
        mock_result.content = "# Test Page\n\nSome content here."
        mock_result.cache_hit = False
        mock_result.strategy_used = "static"
        mock_result.screenshot_path = None
        mock_result.title = "Test Page"
        mock_result.static_content = None
        mock_result.browser_content = None
        mock_result.metadata = {}

        llm_usage = {
            "chatgpt/gpt-5.4": {
                "requests": 1,
                "input_tokens": 123,
                "output_tokens": 45,
                "cost_usd": 0.0075,
            }
        }

        with (
            patch(
                "markitai.fetch.fetch_url",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
            patch(
                "markitai.cli.processors.llm.process_with_llm",
                new_callable=AsyncMock,
                return_value=("# Cleaned", 0.0075, llm_usage),
            ),
        ):
            await process_url(
                url="https://example.com/test",
                output_dir=tmp_path,
                cfg=cfg,
                dry_run=False,
                verbose=False,
            )

        report_files = list(
            (tmp_path / ".markitai" / "reports").glob("markitai.*.report.json")
        )
        assert len(report_files) == 1
        report = json.loads(report_files[0].read_text())
        url_entry = report["url_sources"]["cli"]["urls"]["https://example.com/test"]
        assert "chatgpt/gpt-5.4" in url_entry["llm_usage"]
        assert url_entry["llm_usage"]["chatgpt/gpt-5.4"]["requests"] == 1

    @pytest.mark.asyncio
    async def test_url_with_cache_hit(self, tmp_path: Path) -> None:
        """Test URL conversion with cache hit."""
        cfg = MarkitaiConfig()
        cfg.llm.enabled = False
        cfg.cache.enabled = True

        mock_result = MagicMock()
        mock_result.content = "# Cached Content"
        mock_result.cache_hit = True  # Cache hit
        mock_result.strategy_used = "static"
        mock_result.screenshot_path = None
        mock_result.title = "Cached"
        mock_result.static_content = None
        mock_result.browser_content = None

        mock_cache = MagicMock()

        with (
            patch(
                "markitai.fetch.fetch_url",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
            patch(
                "markitai.fetch.get_fetch_cache",
                return_value=mock_cache,
            ),
        ):
            try:
                await process_url(
                    url="https://example.com/cached",
                    output_dir=tmp_path,
                    cfg=cfg,
                    dry_run=False,
                    verbose=False,
                )
            except SystemExit:
                pass

        # Output file should be created
        output_file = tmp_path / "cached.md"
        assert output_file.exists()

    @pytest.mark.asyncio
    async def test_llm_failure_cancels_parallel_image_analysis(
        self, tmp_path: Path
    ) -> None:
        """Document LLM failures should not leak the paired image-analysis task."""
        cfg = MarkitaiConfig()
        cfg.llm.enabled = True
        cfg.cache.enabled = False
        cfg.image.alt_enabled = True

        mock_result = MagicMock()
        mock_result.content = "# Test Page\n\nSome content here."
        mock_result.cache_hit = False
        mock_result.strategy_used = "static"
        mock_result.screenshot_path = None
        mock_result.title = "Test Page"
        mock_result.static_content = None
        mock_result.browser_content = None
        mock_result.metadata = {}

        download_result = MagicMock()
        download_result.updated_markdown = "# Test Page\n\nSome content here."
        download_result.downloaded_paths = [
            tmp_path / ".markitai" / "assets" / "image.png"
        ]
        download_result.failed_urls = []

        image_task_cancelled = asyncio.Event()
        image_task_ref: asyncio.Task[object] | None = None

        async def _failing_process_with_llm(*args, **kwargs):
            raise RuntimeError("doc llm failed")

        async def _waiting_image_analysis(*args, **kwargs):
            nonlocal image_task_ref
            image_task_ref = asyncio.current_task()
            llm_ready_event = kwargs["llm_ready_event"]
            try:
                await llm_ready_event.wait()
                await asyncio.Future()
            except asyncio.CancelledError:
                image_task_cancelled.set()
                raise

        with (
            patch(
                "markitai.fetch.fetch_url",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
            patch(
                "markitai.image.download_url_images",
                new_callable=AsyncMock,
                return_value=download_result,
            ),
            patch(
                "markitai.cli.processors.llm.process_with_llm",
                new_callable=AsyncMock,
                side_effect=_failing_process_with_llm,
            ),
            patch(
                "markitai.cli.processors.llm.analyze_images_with_llm",
                new_callable=AsyncMock,
                side_effect=_waiting_image_analysis,
            ),
            patch("markitai.cli.processors.url.ui.error"),
            pytest.raises(SystemExit) as exc_info,
        ):
            await process_url(
                url="https://example.com/test",
                output_dir=tmp_path,
                cfg=cfg,
                dry_run=False,
                verbose=False,
            )

        assert exc_info.value.code == 1
        assert image_task_cancelled.is_set()

        if image_task_ref is not None and not image_task_ref.done():
            image_task_ref.cancel()
            await asyncio.gather(image_task_ref, return_exceptions=True)


class TestProcessUrlBatchSuccessPath:
    """Tests for successful batch URL processing."""

    @pytest.mark.asyncio
    async def test_batch_processes_all_urls(self, tmp_path: Path) -> None:
        """Test batch processing completes for all URLs."""

        class MockUrlEntry:
            def __init__(self, url: str, output_name: str | None = None):
                self.url = url
                self.output_name = output_name

        entries = [
            MockUrlEntry("https://example.com/page1"),
            MockUrlEntry("https://example.com/page2"),
        ]

        cfg = MarkitaiConfig()
        cfg.llm.enabled = False
        cfg.cache.enabled = False

        def create_mock_result(url: str) -> MagicMock:
            result = MagicMock()
            result.content = f"# Content for {url}"
            result.cache_hit = False
            result.strategy_used = "static"
            result.screenshot_path = None
            result.title = f"Page {url}"
            return result

        async def mock_fetch(url, *args, **kwargs):
            return create_mock_result(url)

        with patch(
            "markitai.fetch.fetch_url",
            side_effect=mock_fetch,
        ):
            await process_url_batch(
                url_entries=entries,
                output_dir=tmp_path,
                cfg=cfg,
                dry_run=False,
                verbose=False,
            )

        # Check output files were created
        assert (tmp_path / "page1.md").exists()
        assert (tmp_path / "page2.md").exists()

        # Check report was created
        reports_dir = tmp_path / ".markitai" / "reports"
        assert reports_dir.exists()
        report_files = list(reports_dir.glob("markitai.*.report.json"))
        assert len(report_files) == 1

    @pytest.mark.asyncio
    async def test_batch_handles_mixed_success_failure(self, tmp_path: Path) -> None:
        """Test batch handles mix of successful and failed URLs."""
        from markitai.fetch import FetchError

        class MockUrlEntry:
            def __init__(self, url: str, output_name: str | None = None):
                self.url = url
                self.output_name = output_name

        entries = [
            MockUrlEntry("https://example.com/success"),
            MockUrlEntry("https://example.com/fail"),
        ]

        cfg = MarkitaiConfig()
        cfg.llm.enabled = False
        cfg.cache.enabled = False

        async def mock_fetch(url, *args, **kwargs):
            if "fail" in url:
                raise FetchError("Connection refused")
            result = MagicMock()
            result.content = "# Success"
            result.cache_hit = False
            result.strategy_used = "static"
            result.screenshot_path = None
            result.title = "Success"
            return result

        with patch(
            "markitai.fetch.fetch_url",
            side_effect=mock_fetch,
        ):
            await process_url_batch(
                url_entries=entries,
                output_dir=tmp_path,
                cfg=cfg,
                dry_run=False,
                verbose=False,
            )

        # Success file should exist
        assert (tmp_path / "success.md").exists()
        # Fail file should not exist
        assert not (tmp_path / "fail.md").exists()

    @pytest.mark.asyncio
    async def test_batch_with_custom_output_names(self, tmp_path: Path) -> None:
        """Test batch processing with custom output names."""

        class MockUrlEntry:
            def __init__(self, url: str, output_name: str | None = None):
                self.url = url
                self.output_name = output_name

        entries = [
            MockUrlEntry("https://example.com/page", "custom_name"),
        ]

        cfg = MarkitaiConfig()
        cfg.llm.enabled = False
        cfg.cache.enabled = False

        mock_result = MagicMock()
        mock_result.content = "# Custom"
        mock_result.cache_hit = False
        mock_result.strategy_used = "static"
        mock_result.screenshot_path = None
        mock_result.title = "Custom"

        with patch(
            "markitai.fetch.fetch_url",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            await process_url_batch(
                url_entries=entries,
                output_dir=tmp_path,
                cfg=cfg,
                dry_run=False,
                verbose=False,
            )

        # Custom name should be used
        assert (tmp_path / "custom_name.md").exists()
        assert not (tmp_path / "page.md").exists()


class TestProcessUrlOcrWarning:
    """Tests for OCR warning in URL mode."""

    @pytest.mark.asyncio
    async def test_ocr_warning_logged(self, tmp_path: Path) -> None:
        """Test that OCR enabled generates warning for URL mode.

        We mock logger.warning to verify the warning is called.
        """
        cfg = MarkitaiConfig()
        cfg.llm.enabled = False
        cfg.ocr.enabled = True  # OCR not supported for URLs
        cfg.cache.enabled = False

        mock_result = MagicMock()
        mock_result.content = "# Test"
        mock_result.cache_hit = False
        mock_result.strategy_used = "static"
        mock_result.screenshot_path = None
        mock_result.title = "Test"
        mock_result.static_content = None
        mock_result.browser_content = None

        with (
            patch(
                "markitai.fetch.fetch_url",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
            patch("markitai.cli.processors.url.logger") as mock_logger,
        ):
            try:
                await process_url(
                    url="https://example.com/test",
                    output_dir=tmp_path,
                    cfg=cfg,
                    dry_run=False,
                    verbose=True,
                )
            except SystemExit:
                pass

            # Check warning was logged
            warning_calls = mock_logger.warning.call_args_list
            assert len(warning_calls) >= 1
            # Check that at least one warning contains "ocr"
            assert any("ocr" in str(call).lower() for call in warning_calls), (
                f"Expected OCR warning, got: {warning_calls}"
            )


class TestProcessUrlOutputConflict:
    """Tests for output file conflict handling."""

    @pytest.mark.asyncio
    async def test_skip_existing_output(self, tmp_path: Path) -> None:
        """Test that skip strategy skips existing output."""
        cfg = MarkitaiConfig()
        cfg.llm.enabled = False
        cfg.cache.enabled = False
        cfg.output.on_conflict = "skip"

        # Pre-create output file
        output_file = tmp_path / "test.md"
        output_file.write_text("Existing content")

        mock_result = MagicMock()
        mock_result.content = "# New content"
        mock_result.cache_hit = False
        mock_result.strategy_used = "static"
        mock_result.screenshot_path = None
        mock_result.title = "Test"
        mock_result.static_content = None
        mock_result.browser_content = None

        with patch(
            "markitai.fetch.fetch_url",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            try:
                await process_url(
                    url="https://example.com/test",
                    output_dir=tmp_path,
                    cfg=cfg,
                    dry_run=False,
                    verbose=False,
                )
            except SystemExit:
                pass

        # Original content should be preserved
        assert output_file.read_text() == "Existing content"


class TestProcessUrlPureMode:
    """Tests for --pure/--llm/--keep-base flag handling in single-file URL processing."""

    def _make_mock_fetch_result(self) -> MagicMock:
        mock_result = MagicMock()
        mock_result.content = "# Test Page\n\nSome content here."
        mock_result.cache_hit = False
        mock_result.strategy_used = "static"
        mock_result.screenshot_path = None
        mock_result.title = "Test Page"
        mock_result.static_content = None
        mock_result.browser_content = None
        mock_result.metadata = {}
        return mock_result

    @pytest.mark.asyncio
    async def test_single_url_llm_without_keep_base_skips_base_md(
        self, tmp_path: Path
    ) -> None:
        """With --llm (no --keep-base), base .md should NOT be written."""
        cfg = MarkitaiConfig()
        cfg.llm.enabled = True
        cfg.llm.keep_base = False
        cfg.cache.enabled = False

        mock_result = self._make_mock_fetch_result()

        with (
            patch(
                "markitai.fetch.fetch_url",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
            patch(
                "markitai.cli.processors.llm.process_with_llm",
                new_callable=AsyncMock,
                return_value=("# Cleaned", 0.01, {}),
            ),
        ):
            await process_url(
                url="https://example.com/test",
                output_dir=tmp_path,
                cfg=cfg,
                dry_run=False,
                verbose=False,
            )

        # Base .md should NOT exist (only .llm.md should be produced)
        base_md = tmp_path / "test.md"
        assert not base_md.exists(), (
            f"Base .md should not be written when --llm is enabled without --keep-base, "
            f"but found: {base_md}"
        )

    @pytest.mark.asyncio
    async def test_single_url_pure_without_llm_writes_raw_markdown(
        self, tmp_path: Path
    ) -> None:
        """With --pure (no --llm), base .md should contain raw markdown without frontmatter."""
        cfg = MarkitaiConfig()
        cfg.llm.enabled = False
        cfg.llm.pure = True
        cfg.cache.enabled = False

        mock_result = self._make_mock_fetch_result()

        with patch(
            "markitai.fetch.fetch_url",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            try:
                await process_url(
                    url="https://example.com/test",
                    output_dir=tmp_path,
                    cfg=cfg,
                    dry_run=False,
                    verbose=False,
                )
            except SystemExit:
                pass

        output_file = tmp_path / "test.md"
        assert output_file.exists()
        content = output_file.read_text()
        # Should be raw markdown, no frontmatter
        assert not content.startswith("---"), (
            f"Expected raw markdown without frontmatter, but content starts with '---':\n"
            f"{content[:200]}"
        )
        assert "# Test Page" in content


class TestProcessUrlBatchPureMode:
    """Tests for --pure/--llm/--keep-base flag handling in batch URL processing."""

    @pytest.mark.asyncio
    async def test_batch_url_llm_without_keep_base_skips_base_md(
        self, tmp_path: Path
    ) -> None:
        """With --llm (no --keep-base), batch URL base .md should NOT be written."""
        cfg = MarkitaiConfig()
        cfg.llm.enabled = True
        cfg.llm.keep_base = False
        cfg.cache.enabled = False

        mock_result = MagicMock()
        mock_result.content = "# Test Page\n\nSome content."
        mock_result.cache_hit = False
        mock_result.strategy_used = "static"
        mock_result.screenshot_path = None
        mock_result.title = "Test Page"
        mock_result.static_content = None
        mock_result.browser_content = None
        mock_result.metadata = {}

        class MockUrlEntry:
            def __init__(self, url: str, output_name: str | None = None):
                self.url = url
                self.output_name = output_name

        with (
            patch(
                "markitai.fetch.fetch_url",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
            patch(
                "markitai.cli.processors.llm.process_with_llm",
                new_callable=AsyncMock,
                return_value=("# Cleaned", 0.01, {}),
            ),
        ):
            await process_url_batch(
                url_entries=[MockUrlEntry("https://example.com/test")],
                output_dir=tmp_path,
                cfg=cfg,
                dry_run=False,
                verbose=False,
            )

        base_md = tmp_path / "test.md"
        assert not base_md.exists(), (
            "Batch URL base .md should not be written when --llm is enabled without --keep-base"
        )

    @pytest.mark.asyncio
    async def test_batch_url_pure_without_llm_writes_raw_markdown(
        self, tmp_path: Path
    ) -> None:
        """With --pure (no --llm), batch URL .md should contain raw markdown without frontmatter."""
        cfg = MarkitaiConfig()
        cfg.llm.enabled = False
        cfg.llm.pure = True
        cfg.cache.enabled = False

        mock_result = MagicMock()
        mock_result.content = "# Test Page\n\nSome content."
        mock_result.cache_hit = False
        mock_result.strategy_used = "static"
        mock_result.screenshot_path = None
        mock_result.title = "Test Page"
        mock_result.static_content = None
        mock_result.browser_content = None
        mock_result.metadata = {}

        class MockUrlEntry:
            def __init__(self, url: str, output_name: str | None = None):
                self.url = url
                self.output_name = output_name

        with patch(
            "markitai.fetch.fetch_url",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            await process_url_batch(
                url_entries=[MockUrlEntry("https://example.com/test")],
                output_dir=tmp_path,
                cfg=cfg,
                dry_run=False,
                verbose=False,
            )

        output_file = tmp_path / "test.md"
        assert output_file.exists()
        content = output_file.read_text()
        assert not content.startswith("---"), (
            "Expected raw markdown without frontmatter in batch URL mode with --pure"
        )
        assert "# Test Page" in content
