"""Unit tests for cache directory configuration.

These tests verify that fetch cache uses the global cache directory
from configuration, not hardcoded relative paths.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from markitai.config import MarkitaiConfig


class TestFetchCacheUsesGlobalDir:
    """Tests verifying fetch cache uses cfg.cache.global_dir instead of hardcoded paths."""

    @pytest.mark.asyncio
    async def test_process_url_uses_global_cache_dir(self, tmp_path: Path) -> None:
        """Test that process_url initializes fetch cache with global_dir from config."""
        cfg = MarkitaiConfig()
        cfg.llm.enabled = False
        cfg.cache.enabled = True
        # Set a custom global cache directory
        custom_cache_dir = tmp_path / "custom_global_cache"
        cfg.cache.global_dir = str(custom_cache_dir)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        mock_result = MagicMock()
        mock_result.content = "# Test"
        mock_result.cache_hit = False
        mock_result.strategy_used = "static"
        mock_result.screenshot_path = None
        mock_result.title = "Test"
        mock_result.static_content = None
        mock_result.browser_content = None

        captured_cache_dir = None

        def capture_get_fetch_cache(cache_dir, max_size_bytes):
            nonlocal captured_cache_dir
            captured_cache_dir = cache_dir
            return MagicMock()

        with (
            patch(
                "markitai.fetch.fetch_url",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
            patch(
                "markitai.fetch.get_fetch_cache",
                side_effect=capture_get_fetch_cache,
            ),
        ):
            from markitai.cli.processors.url import process_url

            try:
                await process_url(
                    url="https://example.com/test",
                    output_dir=output_dir,
                    cfg=cfg,
                    dry_run=False,
                    verbose=False,
                )
            except SystemExit:
                pass

        # Verify cache was initialized with global_dir, not output_dir.parent
        assert captured_cache_dir is not None, "get_fetch_cache was not called"
        assert captured_cache_dir == custom_cache_dir, (
            f"Expected cache_dir to be {custom_cache_dir}, but got {captured_cache_dir}"
        )
        # Should NOT be output_dir.parent / ".markitai"
        wrong_path = output_dir.parent / ".markitai"
        assert captured_cache_dir != wrong_path, (
            f"Cache dir should use global_dir from config, not hardcoded {wrong_path}"
        )

    @pytest.mark.asyncio
    async def test_process_url_batch_uses_global_cache_dir(
        self, tmp_path: Path
    ) -> None:
        """Test that process_url_batch initializes fetch cache with global_dir."""
        cfg = MarkitaiConfig()
        cfg.llm.enabled = False
        cfg.cache.enabled = True
        custom_cache_dir = tmp_path / "batch_global_cache"
        cfg.cache.global_dir = str(custom_cache_dir)

        output_dir = tmp_path / "batch_output"
        output_dir.mkdir()

        class MockUrlEntry:
            def __init__(self, url: str, output_name: str | None = None):
                self.url = url
                self.output_name = output_name

        entries = [MockUrlEntry("https://example.com/page1")]

        mock_result = MagicMock()
        mock_result.content = "# Test"
        mock_result.cache_hit = False
        mock_result.strategy_used = "static"
        mock_result.screenshot_path = None
        mock_result.title = "Test"

        captured_cache_dir = None

        def capture_get_fetch_cache(cache_dir, max_size_bytes):
            nonlocal captured_cache_dir
            captured_cache_dir = cache_dir
            return MagicMock()

        with (
            patch(
                "markitai.fetch.fetch_url",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
            patch(
                "markitai.fetch.get_fetch_cache",
                side_effect=capture_get_fetch_cache,
            ),
        ):
            from markitai.cli.processors.url import process_url_batch

            await process_url_batch(
                url_entries=entries,
                output_dir=output_dir,
                cfg=cfg,
                dry_run=False,
                verbose=False,
            )

        assert captured_cache_dir is not None, "get_fetch_cache was not called"
        assert captured_cache_dir == custom_cache_dir


class TestBatchProcessorFetchCacheUsesGlobalDir:
    """Tests for batch.py fetch cache directory."""

    @pytest.mark.asyncio
    async def test_batch_processor_uses_global_cache_dir(self, tmp_path: Path) -> None:
        """Test that batch processor's URL handling uses global_dir for fetch cache.

        This test verifies batch.py:196 uses cfg.cache.global_dir
        Instead of: url_cache_dir = output_dir.parent / ".markitai"
        Should be:  url_cache_dir = Path(cfg.cache.global_dir).expanduser()
        """
        cfg = MarkitaiConfig()
        cfg.llm.enabled = False
        cfg.cache.enabled = True
        custom_cache_dir = tmp_path / "batch_processor_cache"
        cfg.cache.global_dir = str(custom_cache_dir)

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "batch_output"
        output_dir.mkdir()

        # Create a mock .urls file in input directory
        urls_file = input_dir / "test.urls"
        urls_file.write_text("https://example.com/page1\n")

        captured_cache_dir = None

        def capture_get_fetch_cache(cache_dir, max_size_bytes):
            nonlocal captured_cache_dir
            captured_cache_dir = cache_dir
            return MagicMock()

        mock_result = MagicMock()
        mock_result.content = "# Test"
        mock_result.cache_hit = False
        mock_result.strategy_used = "static"
        mock_result.screenshot_path = None
        mock_result.title = "Test"

        with (
            patch(
                "markitai.fetch.fetch_url",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
            patch(
                "markitai.fetch.get_fetch_cache",
                side_effect=capture_get_fetch_cache,
            ),
        ):
            from markitai.cli.processors.batch import process_batch

            # Call process_batch which internally handles URLs
            await process_batch(
                input_dir=input_dir,
                output_dir=output_dir,
                cfg=cfg,
                resume=False,
                dry_run=False,
                verbose=False,
            )

        assert captured_cache_dir is not None, "get_fetch_cache was not called"
        assert captured_cache_dir == custom_cache_dir, (
            f"Expected {custom_cache_dir}, got {captured_cache_dir}"
        )


class TestLLMProcessorCacheUsesGlobalDir:
    """Tests for LLMProcessor persistent cache directory."""

    def test_llm_processor_uses_custom_cache_dir(self, tmp_path: Path) -> None:
        """Test that LLMProcessor's PersistentCache uses custom global_dir."""
        from markitai.config import LiteLLMParams, LLMConfig, ModelConfig

        custom_cache_dir = tmp_path / "llm_cache"

        llm_config = LLMConfig(
            enabled=True,
            model_list=[
                ModelConfig(
                    model_name="test",
                    litellm_params=LiteLLMParams(
                        model="openai/gpt-4o-mini",
                        api_key="test-key",
                    ),
                ),
            ],
        )

        from markitai.llm import LLMProcessor

        processor = LLMProcessor(
            config=llm_config,
            cache_global_dir=custom_cache_dir,
        )

        # Verify the cache was initialized with custom directory
        assert processor._persistent_cache._global_cache is not None
        cache_path = processor._persistent_cache._global_cache._db_path
        assert str(custom_cache_dir) in str(cache_path), (
            f"Expected cache path to contain {custom_cache_dir}, got {cache_path}"
        )

    def test_create_llm_processor_passes_global_dir(self, tmp_path: Path) -> None:
        """Test that create_llm_processor passes cache.global_dir to LLMProcessor."""
        from markitai.config import (
            LiteLLMParams,
            MarkitaiConfig,
            ModelConfig,
        )

        cfg = MarkitaiConfig()
        custom_cache_dir = tmp_path / "workflow_cache"
        cfg.cache.global_dir = str(custom_cache_dir)
        cfg.llm.enabled = True
        cfg.llm.model_list = [
            ModelConfig(
                model_name="test",
                litellm_params=LiteLLMParams(
                    model="openai/gpt-4o-mini",
                    api_key="test-key",
                ),
            ),
        ]

        from markitai.workflow.helpers import create_llm_processor

        processor = create_llm_processor(cfg)

        # Verify the cache was initialized with global_dir from config
        assert processor._persistent_cache._global_cache is not None
        cache_path = processor._persistent_cache._global_cache._db_path
        assert str(custom_cache_dir) in str(cache_path), (
            f"Expected cache path to contain {custom_cache_dir}, got {cache_path}"
        )


class TestCacheDirectoryExpansion:
    """Tests for cache directory path expansion."""

    def test_global_dir_expands_tilde(self, tmp_path: Path) -> None:
        """Test that global_dir with ~ is properly expanded."""
        cfg = MarkitaiConfig()
        cfg.cache.global_dir = "~/.markitai"

        # The expanded path should be under user's home directory
        expected = Path.home() / ".markitai"
        actual = Path(cfg.cache.global_dir).expanduser()

        assert actual == expected

    def test_global_dir_handles_absolute_path(self, tmp_path: Path) -> None:
        """Test that absolute path for global_dir is used as-is."""
        cfg = MarkitaiConfig()
        custom_path = tmp_path / "absolute_cache"
        cfg.cache.global_dir = str(custom_path)

        actual = Path(cfg.cache.global_dir).expanduser()
        assert actual == custom_path
