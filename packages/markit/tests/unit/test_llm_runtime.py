"""Tests for LLMRuntime and LLMProcessor runtime integration."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from markit.llm import LLMRuntime


class TestLLMRuntime:
    """Tests for LLMRuntime class."""

    def test_basic_initialization(self) -> None:
        """Test basic runtime initialization."""
        runtime = LLMRuntime(concurrency=10)
        assert runtime.concurrency == 10
        assert runtime._semaphore is None  # Lazily created

    def test_semaphore_property(self) -> None:
        """Test semaphore is created on first access."""
        runtime = LLMRuntime(concurrency=5)

        # First access creates semaphore
        sem = runtime.semaphore
        assert isinstance(sem, asyncio.Semaphore)

        # Second access returns same semaphore
        assert runtime.semaphore is sem

    def test_different_concurrency_values(self) -> None:
        """Test runtime with different concurrency values."""
        for concurrency in [1, 5, 10, 100]:
            runtime = LLMRuntime(concurrency=concurrency)
            assert runtime.concurrency == concurrency


class TestLLMProcessorRuntimeIntegration:
    """Tests for LLMProcessor integration with LLMRuntime."""

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create a mock LLM config."""
        config = MagicMock()
        config.concurrency = 10
        config.model_list = []
        config.router_settings = MagicMock()
        config.router_settings.num_retries = 2
        return config

    def test_processor_without_runtime(self, mock_config: MagicMock) -> None:
        """Test processor creates its own semaphore without runtime."""
        from markit.llm import LLMProcessor

        processor = LLMProcessor(mock_config)
        assert processor._runtime is None

        # Accessing semaphore creates local one
        sem = processor.semaphore
        assert isinstance(sem, asyncio.Semaphore)
        assert processor._semaphore is sem

    def test_processor_with_runtime(self, mock_config: MagicMock) -> None:
        """Test processor uses runtime's semaphore when provided."""
        from markit.llm import LLMProcessor

        runtime = LLMRuntime(concurrency=5)
        processor = LLMProcessor(mock_config, runtime=runtime)

        assert processor._runtime is runtime
        assert processor._semaphore is None  # Not used

        # Accessing semaphore returns runtime's semaphore
        assert processor.semaphore is runtime.semaphore

    def test_multiple_processors_share_runtime(self, mock_config: MagicMock) -> None:
        """Test multiple processors share the same runtime semaphore."""
        from markit.llm import LLMProcessor

        runtime = LLMRuntime(concurrency=3)
        processor1 = LLMProcessor(mock_config, runtime=runtime)
        processor2 = LLMProcessor(mock_config, runtime=runtime)

        # Both processors should return the same semaphore
        assert processor1.semaphore is processor2.semaphore
        assert processor1.semaphore is runtime.semaphore


class TestLLMProcessorContextUsage:
    """Tests for per-context usage tracking in LLMProcessor."""

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create a mock LLM config."""
        config = MagicMock()
        config.concurrency = 10
        config.model_list = []
        config.router_settings = MagicMock()
        config.router_settings.num_retries = 2
        return config

    def test_context_usage_tracking(self, mock_config: MagicMock) -> None:
        """Test that usage is tracked per context."""
        from markit.llm import LLMProcessor

        processor = LLMProcessor(mock_config)

        # Manually call _track_usage with context
        processor._track_usage("model-a", 100, 50, 0.01, context="file1.pdf")
        processor._track_usage("model-a", 200, 100, 0.02, context="file1.pdf")
        processor._track_usage("model-b", 150, 75, 0.015, context="file2.pdf")

        # Check global usage
        global_usage = processor.get_usage()
        assert global_usage["model-a"]["requests"] == 2
        assert global_usage["model-a"]["input_tokens"] == 300
        assert global_usage["model-b"]["requests"] == 1

        # Check context-specific usage
        file1_usage = processor.get_context_usage("file1.pdf")
        assert file1_usage["model-a"]["requests"] == 2
        assert file1_usage["model-a"]["input_tokens"] == 300
        assert "model-b" not in file1_usage

        file2_usage = processor.get_context_usage("file2.pdf")
        assert file2_usage["model-b"]["requests"] == 1
        assert "model-a" not in file2_usage

    def test_get_context_cost(self, mock_config: MagicMock) -> None:
        """Test getting cost for specific context."""
        from markit.llm import LLMProcessor

        processor = LLMProcessor(mock_config)

        processor._track_usage("model-a", 100, 50, 0.01, context="file1.pdf")
        processor._track_usage("model-b", 100, 50, 0.02, context="file1.pdf")
        processor._track_usage("model-a", 100, 50, 0.05, context="file2.pdf")

        assert processor.get_context_cost("file1.pdf") == pytest.approx(0.03)
        assert processor.get_context_cost("file2.pdf") == pytest.approx(0.05)
        assert processor.get_context_cost("nonexistent.pdf") == 0.0

    def test_clear_context_usage(self, mock_config: MagicMock) -> None:
        """Test clearing usage for specific context."""
        from markit.llm import LLMProcessor

        processor = LLMProcessor(mock_config)

        processor._track_usage("model-a", 100, 50, 0.01, context="file1.pdf")
        processor._track_usage("model-a", 100, 50, 0.01, context="file2.pdf")

        # Clear one context
        processor.clear_context_usage("file1.pdf")

        # file1 should be gone, file2 should remain
        assert processor.get_context_usage("file1.pdf") == {}
        assert processor.get_context_usage("file2.pdf") != {}

        # Global usage should still have both
        assert processor.get_usage()["model-a"]["requests"] == 2

    def test_usage_without_context(self, mock_config: MagicMock) -> None:
        """Test that usage without context only tracks globally."""
        from markit.llm import LLMProcessor

        processor = LLMProcessor(mock_config)

        # Track without context
        processor._track_usage("model-a", 100, 50, 0.01)

        # Should be in global, but not in any context
        assert processor.get_usage()["model-a"]["requests"] == 1
        assert processor._context_usage == {}
