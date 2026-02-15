"""Unit tests for LLMProcessor utility methods and cache classes.

Focuses on non-async utility methods to increase coverage:
- Initialization and configuration
- Cache management (SQLiteCache, PersistentCache, ContentCache)
- Usage tracking methods
- Helper methods (_get_cached_image, _calculate_dynamic_max_tokens)
- Format output methods
- Router creation logic
- Module-level helper functions
"""

from __future__ import annotations

import base64
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from markitai.config import (
    LiteLLMParams,
    LLMConfig,
    ModelConfig,
    PromptsConfig,
)
from markitai.llm.cache import ContentCache, PersistentCache, SQLiteCache
from markitai.llm.models import (
    context_display_name,
    get_model_info_cached,
    get_model_max_output_tokens,
    get_response_cost,
)
from markitai.llm.processor import (
    HybridRouter,
    LLMProcessor,
    LocalProviderWrapper,
    _is_all_local_providers,
)

# =============================================================================
# Test Module-Level Helper Functions
# =============================================================================


class TestContextDisplayName:
    """Tests for context_display_name function (from models.py)."""

    def test_empty_context(self):
        """Test empty context returns empty."""
        assert context_display_name("") == ""

    def test_simple_filename(self):
        """Test simple filename."""
        assert context_display_name("file.pdf") == "file.pdf"

    def test_unix_path(self):
        """Test Unix path extracts filename."""
        assert context_display_name("/path/to/file.pdf") == "file.pdf"

    def test_windows_style_path_with_forward_slashes(self):
        """Test Windows-style path with forward slashes (normalized).

        Note: On Linux, Path("C:/path/to/file.pdf") returns "C:/path/to/file.pdf"
        because C: is not recognized as a drive letter. The function preserves
        this behavior since it uses Path().name which works correctly on Windows.
        On Windows, this would return "file.pdf".
        """
        import platform

        result = context_display_name("C:/path/to/file.pdf")
        if platform.system() == "Windows":
            assert result == "file.pdf"
        else:
            # On Linux, C: is not recognized as drive letter, so split on ":"
            assert "file.pdf" in result

    def test_path_with_suffix(self):
        """Test path with :suffix preserves suffix."""
        result = context_display_name("/path/to/file.pdf:images")
        assert result == "file.pdf:images"

    def test_windows_path_with_suffix_forward_slash(self):
        """Test Windows path with suffix using forward slashes."""
        result = context_display_name("C:/path/to/file.pdf:images")
        assert result == "file.pdf:images"

    def test_relative_path(self):
        """Test relative path extracts filename."""
        assert context_display_name("subdir/file.pdf") == "file.pdf"


class TestIsAllLocalProviders:
    """Tests for _is_all_local_providers function."""

    def test_empty_list(self):
        """Test empty list returns False."""
        assert _is_all_local_providers([]) is False

    def test_all_local_providers(self):
        """Test all local providers returns True."""
        model_list = [
            {"litellm_params": {"model": "claude-agent/sonnet"}},
            {"litellm_params": {"model": "copilot/claude-sonnet-4"}},
        ]
        with patch("markitai.providers.is_local_provider_model", return_value=True):
            assert _is_all_local_providers(model_list) is True

    def test_mixed_providers(self):
        """Test mixed providers returns False."""
        model_list = [
            {"litellm_params": {"model": "claude-agent/sonnet"}},
            {"litellm_params": {"model": "openai/gpt-4o"}},
        ]
        with patch(
            "markitai.providers.is_local_provider_model",
            side_effect=lambda x: x.startswith("claude-agent/"),
        ):
            assert _is_all_local_providers(model_list) is False

    def test_no_local_providers(self):
        """Test no local providers returns False."""
        model_list = [
            {"litellm_params": {"model": "openai/gpt-4o"}},
            {"litellm_params": {"model": "deepseek/deepseek-chat"}},
        ]
        with patch("markitai.providers.is_local_provider_model", return_value=False):
            assert _is_all_local_providers(model_list) is False


class TestGetModelInfoCached:
    """Tests for get_model_info_cached function."""

    def test_returns_cached_value(self):
        """Test that cached values are returned."""
        from markitai.llm import models

        # Clear cache first
        models._model_info_cache.clear()

        # First call - should query litellm
        with (
            patch(
                "markitai.providers.get_local_provider_model_info", return_value=None
            ),
            patch("litellm.get_model_info") as mock_get_info,
        ):
            mock_get_info.return_value = {
                "max_input_tokens": 100000,
                "max_output_tokens": 8000,
                "supports_vision": True,
            }
            result1 = get_model_info_cached("test/model")
            assert mock_get_info.called

        # Second call - should use cache
        with patch("litellm.get_model_info") as mock_get_info:
            result2 = get_model_info_cached("test/model")
            assert not mock_get_info.called
            assert result2 == result1

        # Clean up
        models._model_info_cache.clear()

    def test_local_provider_model(self):
        """Test local provider model info."""
        from markitai.llm import models

        models._model_info_cache.clear()

        local_info = {
            "max_input_tokens": 200000,
            "max_output_tokens": 16000,
            "supports_vision": True,
        }

        with patch(
            "markitai.providers.get_local_provider_model_info",
            return_value=local_info,
        ):
            result = get_model_info_cached("claude-agent/sonnet")
            assert result["max_input_tokens"] == 200000
            assert result["max_output_tokens"] == 16000
            assert result["supports_vision"] is True

        models._model_info_cache.clear()

    def test_fallback_on_exception(self):
        """Test defaults are returned when litellm fails."""
        from markitai.llm import models

        models._model_info_cache.clear()

        with (
            patch(
                "markitai.providers.get_local_provider_model_info", return_value=None
            ),
            patch("litellm.get_model_info", side_effect=Exception("API Error")),
        ):
            result = get_model_info_cached("unknown/model")
            assert result["max_input_tokens"] == 128000
            assert result["supports_vision"] is False

        models._model_info_cache.clear()


class TestGetModelMaxOutputTokens:
    """Tests for get_model_max_output_tokens function."""

    def test_returns_max_output_tokens(self):
        """Test returns max_output_tokens from model info."""
        with patch(
            "markitai.llm.models.get_model_info_cached",
            return_value={"max_output_tokens": 16384},
        ):
            assert get_model_max_output_tokens("test/model") == 16384


class TestGetResponseCost:
    """Tests for get_response_cost function."""

    def test_cost_from_hidden_params(self):
        """Test cost is extracted from _hidden_params."""
        response = MagicMock()
        response._hidden_params = {"total_cost_usd": 0.0025}

        cost = get_response_cost(response)
        assert cost == 0.0025

    def test_cost_from_litellm_completion_cost(self):
        """Test fallback to litellm.completion_cost."""
        response = MagicMock()
        response._hidden_params = {}

        with patch("markitai.llm.models.completion_cost", return_value=0.005):
            cost = get_response_cost(response)
            assert cost == 0.005

    def test_cost_zero_on_exception(self):
        """Test returns 0.0 when cost calculation fails."""
        response = MagicMock()
        response._hidden_params = {}

        with patch(
            "markitai.llm.models.completion_cost", side_effect=Exception("Error")
        ):
            cost = get_response_cost(response)
            assert cost == 0.0

    def test_no_hidden_params(self):
        """Test when _hidden_params is None."""
        response = MagicMock(spec=[])  # No _hidden_params attribute

        with patch("markitai.llm.models.completion_cost", return_value=0.003):
            cost = get_response_cost(response)
            assert cost == 0.003


# =============================================================================
# Test LocalProviderWrapper
# =============================================================================


class TestLocalProviderWrapper:
    """Tests for LocalProviderWrapper class."""

    def test_init_single_model(self):
        """Test initialization with single model."""
        model_list = [
            {
                "model_name": "default",
                "litellm_params": {"model": "claude-agent/sonnet", "weight": 1.0},
            }
        ]
        wrapper = LocalProviderWrapper(model_list)
        assert len(wrapper._model_groups["default"]) == 1

    def test_init_multiple_models(self):
        """Test initialization with multiple models."""
        model_list = [
            {
                "model_name": "default",
                "litellm_params": {"model": "claude-agent/sonnet", "weight": 2.0},
            },
            {
                "model_name": "default",
                "litellm_params": {"model": "claude-agent/haiku", "weight": 1.0},
            },
        ]
        wrapper = LocalProviderWrapper(model_list)
        assert len(wrapper._model_groups["default"]) == 2

    def test_has_images_true(self):
        """Test _has_images detects image content."""
        wrapper = LocalProviderWrapper([])
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,xxx"},
                    },
                ],
            }
        ]
        assert wrapper._has_images(messages) is True

    def test_has_images_false(self):
        """Test _has_images returns False for text-only."""
        wrapper = LocalProviderWrapper([])
        messages = [{"role": "user", "content": "Hello, world!"}]
        assert wrapper._has_images(messages) is False

    def test_is_image_capable_claude_agent(self):
        """Test claude-agent models are image capable."""
        wrapper = LocalProviderWrapper([])
        assert wrapper._is_image_capable("claude-agent/sonnet") is True
        assert wrapper._is_image_capable("claude-agent/opus") is True
        assert wrapper._is_image_capable("claude-agent/haiku") is True

    def test_is_image_capable_copilot_claude(self):
        """Test copilot Claude models are image capable."""
        wrapper = LocalProviderWrapper([])
        assert wrapper._is_image_capable("copilot/claude-sonnet-4") is True
        assert wrapper._is_image_capable("copilot/claude-3.5-sonnet") is True

    def test_is_image_capable_copilot_gpt4o(self):
        """Test copilot GPT-4o models are image capable."""
        wrapper = LocalProviderWrapper([])
        assert wrapper._is_image_capable("copilot/gpt-4o") is True
        assert wrapper._is_image_capable("copilot/gpt-4o-mini") is True

    def test_is_image_capable_non_vision_model(self):
        """Test non-vision models are not image capable."""
        wrapper = LocalProviderWrapper([])
        assert wrapper._is_image_capable("copilot/gpt-3.5-turbo") is False
        assert wrapper._is_image_capable("openai/gpt-4") is False

    def test_select_model_single(self):
        """Test model selection with single model."""
        model_list = [
            {
                "model_name": "default",
                "litellm_params": {"model": "claude-agent/sonnet", "weight": 1.0},
            }
        ]
        wrapper = LocalProviderWrapper(model_list)
        selected = wrapper._select_model("default")
        assert selected == "claude-agent/sonnet"

    def test_select_model_unknown_name(self):
        """Test selecting unknown model name returns the name."""
        wrapper = LocalProviderWrapper([])
        selected = wrapper._select_model("unknown-model")
        assert selected == "unknown-model"


# =============================================================================
# Test HybridRouter
# =============================================================================


class TestHybridRouter:
    """Tests for HybridRouter class."""

    def test_init(self):
        """Test HybridRouter initialization."""
        standard_router = MagicMock()
        standard_router.model_list = [
            {"litellm_params": {"model": "openai/gpt-4o", "weight": 1.0}}
        ]

        local_wrapper = LocalProviderWrapper(
            [
                {
                    "model_name": "default",
                    "litellm_params": {"model": "claude-agent/sonnet", "weight": 1.0},
                }
            ]
        )

        hybrid = HybridRouter(standard_router, local_wrapper)
        assert len(hybrid._all_models) == 2

    def test_model_list_property(self):
        """Test model_list property combines both routers."""
        standard_router = MagicMock()
        standard_router.model_list = [{"litellm_params": {"model": "openai/gpt-4o"}}]

        local_wrapper = LocalProviderWrapper(
            [
                {
                    "model_name": "default",
                    "litellm_params": {"model": "claude-agent/sonnet"},
                }
            ]
        )

        hybrid = HybridRouter(standard_router, local_wrapper)
        combined = hybrid.model_list
        assert len(combined) == 2

    def test_has_images(self):
        """Test _has_images detection."""
        standard_router = MagicMock()
        standard_router.model_list = []
        local_wrapper = LocalProviderWrapper([])

        hybrid = HybridRouter(standard_router, local_wrapper)

        messages_with_image = [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "x"}}],
            }
        ]
        assert hybrid._has_images(messages_with_image) is True

        messages_text = [{"role": "user", "content": "Hello"}]
        assert hybrid._has_images(messages_text) is False


# =============================================================================
# Test SQLiteCache
# =============================================================================


class TestSQLiteCache:
    """Tests for SQLiteCache class."""

    def test_init_creates_db(self, tmp_path: Path):
        """Test initialization creates database file."""
        db_path = tmp_path / "cache.db"
        _ = SQLiteCache(db_path)
        assert db_path.exists()

    def test_compute_hash_consistency(self, tmp_path: Path):
        """Test hash is consistent for same input."""
        cache = SQLiteCache(tmp_path / "cache.db")
        hash1 = cache._compute_hash("prompt", "content")
        hash2 = cache._compute_hash("prompt", "content")
        assert hash1 == hash2

    def test_compute_hash_uniqueness(self, tmp_path: Path):
        """Test different inputs produce different hashes."""
        cache = SQLiteCache(tmp_path / "cache.db")
        hash1 = cache._compute_hash("prompt1", "content")
        hash2 = cache._compute_hash("prompt2", "content")
        assert hash1 != hash2

    def test_compute_hash_uses_head_tail(self, tmp_path: Path):
        """Test hash uses head and tail for large content."""
        cache = SQLiteCache(tmp_path / "cache.db")
        # Create content longer than 25000 chars
        long_content = "a" * 50000
        hash1 = cache._compute_hash("prompt", long_content)
        # Same length, different tail
        different_tail = "a" * 25000 + "b" * 25000
        hash2 = cache._compute_hash("prompt", different_tail)
        assert hash1 != hash2

    def test_set_and_get(self, tmp_path: Path):
        """Test basic set and get operations."""
        cache = SQLiteCache(tmp_path / "cache.db")
        cache.set("prompt", "content", '{"result": "test"}', "test-model")
        result = cache.get("prompt", "content")
        assert result == '{"result": "test"}'

    def test_get_miss(self, tmp_path: Path):
        """Test get returns None for missing entry."""
        cache = SQLiteCache(tmp_path / "cache.db")
        result = cache.get("nonexistent", "content")
        assert result is None

    def test_clear(self, tmp_path: Path):
        """Test clearing cache."""
        cache = SQLiteCache(tmp_path / "cache.db")
        cache.set("p1", "c1", "r1")
        cache.set("p2", "c2", "r2")

        count = cache.clear()
        assert count == 2
        assert cache.get("p1", "c1") is None

    def test_stats(self, tmp_path: Path):
        """Test cache statistics."""
        cache = SQLiteCache(tmp_path / "cache.db")
        cache.set("prompt", "content", '{"key": "value"}')

        stats = cache.stats()
        assert stats["count"] == 1
        assert stats["size_bytes"] > 0
        assert "db_path" in stats

    def test_stats_by_model(self, tmp_path: Path):
        """Test statistics grouped by model."""
        cache = SQLiteCache(tmp_path / "cache.db")
        cache.set("p1", "c1", "r1", "model-a")
        cache.set("p2", "c2", "r2", "model-a")
        cache.set("p3", "c3", "r3", "model-b")

        stats = cache.stats_by_model()
        assert "model-a" in stats
        assert stats["model-a"]["count"] == 2
        assert "model-b" in stats
        assert stats["model-b"]["count"] == 1

    def test_list_entries(self, tmp_path: Path):
        """Test listing cache entries."""
        cache = SQLiteCache(tmp_path / "cache.db")
        cache.set("p1", "c1", '{"caption": "Test image"}', "model")

        entries = cache.list_entries(limit=10)
        assert len(entries) == 1
        assert "key" in entries[0]
        assert entries[0]["model"] == "model"

    def test_parse_value_preview_json(self, tmp_path: Path):
        """Test preview parsing for JSON values."""
        cache = SQLiteCache(tmp_path / "cache.db")

        # Image caption
        preview = cache._parse_value_preview('{"caption": "A beautiful sunset"}')
        assert preview.startswith("image:")

        # Frontmatter
        preview = cache._parse_value_preview('{"title": "My Document"}')
        assert preview.startswith("frontmatter:")

    def test_parse_value_preview_plain(self, tmp_path: Path):
        """Test preview parsing for plain text."""
        cache = SQLiteCache(tmp_path / "cache.db")
        preview = cache._parse_value_preview("Just plain text content")
        assert preview.startswith("text:")

    def test_parse_value_preview_empty(self, tmp_path: Path):
        """Test preview parsing for empty value."""
        cache = SQLiteCache(tmp_path / "cache.db")
        assert cache._parse_value_preview("") == ""
        assert cache._parse_value_preview(None) == ""


# =============================================================================
# Test PersistentCache
# =============================================================================


class TestPersistentCache:
    """Tests for PersistentCache class."""

    def test_init_disabled(self):
        """Test initialization with caching disabled."""
        cache = PersistentCache(enabled=False)
        assert cache._enabled is False
        assert cache._global_cache is None

    def test_init_enabled(self, tmp_path: Path):
        """Test initialization with caching enabled."""
        cache = PersistentCache(global_dir=tmp_path, enabled=True)
        assert cache._enabled is True
        assert cache._global_cache is not None

    def test_get_disabled(self):
        """Test get returns None when disabled."""
        cache = PersistentCache(enabled=False)
        result = cache.get("prompt", "content")
        assert result is None

    def test_get_skip_read(self, tmp_path: Path):
        """Test get returns None when skip_read is True."""
        cache = PersistentCache(global_dir=tmp_path, enabled=True, skip_read=True)
        # Set a value
        cache.set("prompt", "content", {"result": "test"})
        # Get should return None (skip_read mode)
        result = cache.get("prompt", "content")
        assert result is None

    def test_set_and_get(self, tmp_path: Path):
        """Test basic set and get."""
        cache = PersistentCache(global_dir=tmp_path, enabled=True)
        cache.set("prompt", "content", {"key": "value"}, "model")
        result = cache.get("prompt", "content")
        assert result == {"key": "value"}

    def test_set_disabled(self):
        """Test set does nothing when disabled."""
        cache = PersistentCache(enabled=False)
        cache.set("prompt", "content", {"key": "value"})
        # Should not raise, just no-op

    def test_hit_miss_tracking(self, tmp_path: Path):
        """Test hit/miss tracking."""
        cache = PersistentCache(global_dir=tmp_path, enabled=True)

        # Miss
        cache.get("prompt1", "content1")
        assert cache._misses == 1
        assert cache._hits == 0

        # Set and hit
        cache.set("prompt2", "content2", "result")
        cache.get("prompt2", "content2")
        assert cache._hits == 1
        assert cache._misses == 1

    def test_stats(self, tmp_path: Path):
        """Test cache statistics."""
        cache = PersistentCache(global_dir=tmp_path, enabled=True)
        cache.set("p", "c", "r")
        cache.get("p", "c")  # Hit
        cache.get("x", "y")  # Miss

        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 50.0

    def test_clear(self, tmp_path: Path):
        """Test clearing cache."""
        cache = PersistentCache(global_dir=tmp_path, enabled=True)
        cache.set("p1", "c1", "r1")
        cache.set("p2", "c2", "r2")

        count = cache.clear()
        assert count == 2

    def test_glob_match_standard(self, tmp_path: Path):
        """Test standard glob matching."""
        cache = PersistentCache(global_dir=tmp_path, enabled=True)
        assert cache._glob_match("file.pdf", "*.pdf") is True
        assert cache._glob_match("file.txt", "*.pdf") is False

    def test_glob_match_double_star(self, tmp_path: Path):
        """Test ** glob matching for zero-or-more directories."""
        cache = PersistentCache(global_dir=tmp_path, enabled=True)
        # Should match zero directories
        assert cache._glob_match("file.pdf", "**/*.pdf") is True
        # Should match one directory
        assert cache._glob_match("dir/file.pdf", "**/*.pdf") is True
        # Should match multiple directories
        assert cache._glob_match("a/b/c/file.pdf", "**/*.pdf") is True

    def test_extract_matchable_path_simple(self, tmp_path: Path):
        """Test extracting filename from simple path."""
        cache = PersistentCache(global_dir=tmp_path, enabled=True)
        assert cache._extract_matchable_path("file.pdf") == "file.pdf"

    def test_extract_matchable_path_with_suffix(self, tmp_path: Path):
        """Test extracting filename from path with suffix."""
        cache = PersistentCache(global_dir=tmp_path, enabled=True)
        result = cache._extract_matchable_path("/path/to/file.pdf:images")
        assert result == "file.pdf"

    def test_extract_matchable_path_windows(self, tmp_path: Path):
        """Test extracting filename from Windows path."""
        cache = PersistentCache(global_dir=tmp_path, enabled=True)
        result = cache._extract_matchable_path("C:\\Users\\test\\file.pdf")
        assert result == "file.pdf"

    def test_should_skip_cache_no_patterns(self, tmp_path: Path):
        """Test skip check with no patterns."""
        cache = PersistentCache(global_dir=tmp_path, enabled=True)
        assert cache._should_skip_cache("file.pdf") is False

    def test_should_skip_cache_matching_pattern(self, tmp_path: Path):
        """Test skip check with matching pattern."""
        cache = PersistentCache(
            global_dir=tmp_path, enabled=True, no_cache_patterns=["*.pdf"]
        )
        assert cache._should_skip_cache("file.pdf") is True
        assert cache._should_skip_cache("file.txt") is False

    def test_should_skip_cache_full_path(self, tmp_path: Path):
        """Test skip check with full path context."""
        cache = PersistentCache(
            global_dir=tmp_path, enabled=True, no_cache_patterns=["*.JPG"]
        )
        # Should match filename from full path
        assert cache._should_skip_cache("/home/user/project/photo.JPG") is True


# =============================================================================
# Test ContentCache
# =============================================================================


class TestContentCacheExtended:
    """Extended tests for ContentCache class."""

    def test_lru_eviction_order(self):
        """Test LRU eviction removes oldest accessed item."""
        cache = ContentCache(maxsize=3, ttl_seconds=300)

        cache.set("p1", "c1", "r1")
        time.sleep(0.01)
        cache.set("p2", "c2", "r2")
        time.sleep(0.01)
        cache.set("p3", "c3", "r3")

        # Access p1 to make it recently used
        cache.get("p1", "c1")
        time.sleep(0.01)

        # Add new item - should evict p2 (oldest accessed)
        cache.set("p4", "c4", "r4")

        assert cache.get("p1", "c1") == "r1"  # Still there
        assert cache.get("p2", "c2") is None  # Evicted
        assert cache.get("p3", "c3") == "r3"  # Still there
        assert cache.get("p4", "c4") == "r4"  # New item

    def test_update_moves_to_end(self):
        """Test updating existing key moves it to end."""
        cache = ContentCache(maxsize=3, ttl_seconds=300)

        cache.set("p1", "c1", "r1")
        time.sleep(0.01)
        cache.set("p2", "c2", "r2")
        time.sleep(0.01)
        cache.set("p3", "c3", "r3")

        # Update p1 - should move to end
        cache.set("p1", "c1", "r1_updated")
        time.sleep(0.01)

        # Add new item - should evict p2 (now oldest)
        cache.set("p4", "c4", "r4")

        assert cache.get("p1", "c1") == "r1_updated"
        assert cache.get("p2", "c2") is None


# =============================================================================
# Test LLMProcessor
# =============================================================================


class TestLLMProcessorInit:
    """Tests for LLMProcessor initialization."""

    def test_init_with_no_cache(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test initialization with no_cache flag."""
        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)
        assert processor._persistent_cache._skip_read is True

    def test_init_with_no_cache_patterns(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test initialization with no_cache_patterns."""
        patterns = ["*.pdf", "reports/**"]
        processor = LLMProcessor(llm_config, prompts_config, no_cache_patterns=patterns)
        assert processor._persistent_cache._no_cache_patterns == patterns

    def test_init_creates_default_usage_tracking(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test default usage tracking structures are created."""
        processor = LLMProcessor(llm_config, prompts_config)
        assert processor._usage is not None
        assert processor._context_usage is not None
        assert processor._call_counter is not None


class TestLLMProcessorUsageTracking:
    """Tests for usage tracking methods."""

    def test_track_usage_basic(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test basic usage tracking."""
        processor = LLMProcessor(llm_config, prompts_config)
        processor._track_usage("model-a", 100, 50, 0.001)

        usage = processor.get_usage()
        assert "model-a" in usage
        assert usage["model-a"]["requests"] == 1
        assert usage["model-a"]["input_tokens"] == 100
        assert usage["model-a"]["output_tokens"] == 50
        assert usage["model-a"]["cost_usd"] == pytest.approx(0.001)

    def test_track_usage_accumulates(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test usage accumulates across multiple calls."""
        processor = LLMProcessor(llm_config, prompts_config)
        processor._track_usage("model-a", 100, 50, 0.001)
        processor._track_usage("model-a", 200, 100, 0.002)

        usage = processor.get_usage()
        assert usage["model-a"]["requests"] == 2
        assert usage["model-a"]["input_tokens"] == 300
        assert usage["model-a"]["output_tokens"] == 150
        assert usage["model-a"]["cost_usd"] == pytest.approx(0.003)

    def test_track_usage_with_context(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test usage tracking with context."""
        processor = LLMProcessor(llm_config, prompts_config)
        processor._track_usage("model-a", 100, 50, 0.001, context="file.pdf")

        context_usage = processor.get_context_usage("file.pdf")
        assert "model-a" in context_usage
        assert context_usage["model-a"]["requests"] == 1

    def test_get_total_cost(self, llm_config: LLMConfig, prompts_config: PromptsConfig):
        """Test getting total cost across all models."""
        processor = LLMProcessor(llm_config, prompts_config)
        processor._track_usage("model-a", 100, 50, 0.001)
        processor._track_usage("model-b", 200, 100, 0.002)

        total = processor.get_total_cost()
        assert total == pytest.approx(0.003)

    def test_get_context_cost(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test getting cost for specific context."""
        processor = LLMProcessor(llm_config, prompts_config)
        processor._track_usage("model-a", 100, 50, 0.001, context="file.pdf")
        processor._track_usage("model-a", 200, 100, 0.002, context="file.pdf")
        processor._track_usage("model-b", 50, 25, 0.0005, context="other.pdf")

        cost = processor.get_context_cost("file.pdf")
        assert cost == pytest.approx(0.003)

    def test_get_context_cost_unknown(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test getting cost for unknown context returns 0."""
        processor = LLMProcessor(llm_config, prompts_config)
        cost = processor.get_context_cost("nonexistent")
        assert cost == 0.0

    def test_clear_context_usage(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test clearing usage for specific context."""
        processor = LLMProcessor(llm_config, prompts_config)
        processor._track_usage("model-a", 100, 50, 0.001, context="file.pdf")
        processor._call_counter["file.pdf"] = 5

        processor.clear_context_usage("file.pdf")

        assert processor.get_context_usage("file.pdf") == {}
        assert "file.pdf" not in processor._call_counter

    def test_usage_tracking_thread_safety(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test usage tracking is thread-safe."""
        processor = LLMProcessor(llm_config, prompts_config)

        def track_usage():
            for _ in range(100):
                processor._track_usage("model", 10, 5, 0.0001)

        threads = [threading.Thread(target=track_usage) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        usage = processor.get_usage()
        assert usage["model"]["requests"] == 1000


class TestLLMProcessorCallCounter:
    """Tests for call counter methods."""

    def test_get_next_call_index(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test call index increments."""
        processor = LLMProcessor(llm_config, prompts_config)

        idx1 = processor._get_next_call_index("file.pdf")
        idx2 = processor._get_next_call_index("file.pdf")
        idx3 = processor._get_next_call_index("other.pdf")

        assert idx1 == 1
        assert idx2 == 2
        assert idx3 == 1

    def test_reset_call_counter_specific(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test resetting counter for specific context."""
        processor = LLMProcessor(llm_config, prompts_config)
        processor._get_next_call_index("file.pdf")
        processor._get_next_call_index("file.pdf")
        processor._get_next_call_index("other.pdf")

        processor.reset_call_counter("file.pdf")

        # file.pdf should start fresh
        assert processor._get_next_call_index("file.pdf") == 1
        # other.pdf should continue
        assert processor._get_next_call_index("other.pdf") == 2

    def test_reset_call_counter_all(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test resetting all counters."""
        processor = LLMProcessor(llm_config, prompts_config)
        processor._get_next_call_index("file.pdf")
        processor._get_next_call_index("other.pdf")

        processor.reset_call_counter()

        assert processor._get_next_call_index("file.pdf") == 1
        assert processor._get_next_call_index("other.pdf") == 1


class TestLLMProcessorCacheManagement:
    """Tests for cache management methods."""

    def test_get_cache_stats(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test getting cache statistics."""
        processor = LLMProcessor(llm_config, prompts_config)
        processor._cache_hits = 5
        processor._cache_misses = 10

        stats = processor.get_cache_stats()
        assert stats["memory"]["hits"] == 5
        assert stats["memory"]["misses"] == 10
        assert stats["memory"]["hit_rate"] == pytest.approx(33.33, rel=0.01)

    def test_clear_cache_memory(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test clearing memory cache."""
        processor = LLMProcessor(llm_config, prompts_config)
        processor._cache.set("p", "c", "r")
        processor._cache_hits = 5
        processor._cache_misses = 10

        result = processor.clear_cache("memory")

        assert result["memory"] == 1
        assert processor._cache.size == 0
        assert processor._cache_hits == 0
        assert processor._cache_misses == 0

    def test_clear_image_cache(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test clearing image cache."""
        processor = LLMProcessor(llm_config, prompts_config)
        processor._image_cache["path"] = (b"data", "base64")
        processor._image_cache_bytes = 1000

        processor.clear_image_cache()

        assert len(processor._image_cache) == 0
        assert processor._image_cache_bytes == 0


class TestLLMProcessorImageCache:
    """Tests for _get_cached_image method."""

    def test_get_cached_image_cache_hit(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig, tmp_path: Path
    ):
        """Test image cache hit returns cached value."""
        processor = LLMProcessor(llm_config, prompts_config)

        img_path = tmp_path / "test.jpg"
        cached_data = (b"image_bytes", "base64_encoded")
        processor._image_cache[str(img_path)] = cached_data

        result = processor._get_cached_image(img_path)
        assert result == cached_data

    def test_get_cached_image_cache_miss(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        tmp_path: Path,
        sample_png_bytes: bytes,
    ):
        """Test image cache miss reads file and caches."""
        processor = LLMProcessor(llm_config, prompts_config)

        img_path = tmp_path / "test.png"
        img_path.write_bytes(sample_png_bytes)

        result = processor._get_cached_image(img_path)

        assert result[0] == sample_png_bytes
        assert result[1] == base64.b64encode(sample_png_bytes).decode()
        assert str(img_path) in processor._image_cache

    def test_get_cached_image_lru_eviction(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        tmp_path: Path,
        sample_png_bytes: bytes,
    ):
        """Test LRU eviction when cache is full."""
        processor = LLMProcessor(llm_config, prompts_config)
        processor._image_cache_max_size = 2

        # Create and cache 2 images
        for i in range(2):
            img_path = tmp_path / f"test_{i}.png"
            img_path.write_bytes(sample_png_bytes)
            processor._get_cached_image(img_path)

        assert len(processor._image_cache) == 2

        # Add third image - should evict first
        img_path_3 = tmp_path / "test_3.png"
        img_path_3.write_bytes(sample_png_bytes)
        processor._get_cached_image(img_path_3)

        assert len(processor._image_cache) == 2
        assert str(tmp_path / "test_0.png") not in processor._image_cache


class TestLLMProcessorDynamicMaxTokens:
    """Tests for _calculate_dynamic_max_tokens method."""

    def test_returns_none_without_model_info(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test returns None when model info unavailable."""
        processor = LLMProcessor(llm_config, prompts_config)

        with patch(
            "markitai.llm.processor.get_model_info_cached",
            return_value={"max_input_tokens": None, "max_output_tokens": None},
        ):
            result = processor._calculate_dynamic_max_tokens(
                [{"role": "user", "content": "test"}], "unknown/model"
            )
            assert result is None

    def test_calculates_based_on_input_size(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test calculation based on input token count."""
        processor = LLMProcessor(llm_config, prompts_config)

        with (
            patch(
                "markitai.llm.processor.get_model_info_cached",
                return_value={"max_input_tokens": 128000, "max_output_tokens": 8192},
            ),
            patch("litellm.token_counter", return_value=1000),
        ):
            result = processor._calculate_dynamic_max_tokens(
                [{"role": "user", "content": "test"}], "test/model"
            )
            # Should return a reasonable value
            assert result is not None
            assert result >= 1000  # At least minimum floor
            assert result <= 8192  # At most max_output

    def test_table_heavy_content(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test table-heavy content gets more output tokens."""
        processor = LLMProcessor(llm_config, prompts_config)

        # Content with many table rows
        table_content = "|col1|col2|\n" * 30

        with (
            patch(
                "markitai.llm.processor.get_model_info_cached",
                return_value={"max_input_tokens": 128000, "max_output_tokens": 16384},
            ),
            patch("litellm.token_counter", return_value=500),
        ):
            result = processor._calculate_dynamic_max_tokens(
                [{"role": "user", "content": table_content}], "test/model"
            )
            # Should have higher floor for table-heavy content
            assert result >= 4000

    def test_uses_router_model_limits(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test uses minimum limits across router models."""
        processor = LLMProcessor(llm_config, prompts_config)

        mock_router = MagicMock()
        mock_router.model_list = [
            {"litellm_params": {"model": "model-a"}},
            {"litellm_params": {"model": "model-b"}},
        ]

        with (
            patch(
                "markitai.llm.processor.get_model_info_cached",
                side_effect=[
                    {"max_input_tokens": 128000, "max_output_tokens": 16384},
                    {"max_input_tokens": 64000, "max_output_tokens": 4096},
                ],
            ),
            patch("litellm.token_counter", return_value=1000),
        ):
            result = processor._calculate_dynamic_max_tokens(
                [{"role": "user", "content": "test"}],
                router=mock_router,
            )
            # Should use minimum max_output (4096)
            assert result <= 4096


class TestLLMProcessorRouterHelpers:
    """Tests for router helper methods."""

    def test_get_router_primary_model(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test getting primary model from router."""
        processor = LLMProcessor(llm_config, prompts_config)

        mock_router = MagicMock()
        mock_router.model_list = [
            {"litellm_params": {"model": "openai/gpt-4o"}},
            {"litellm_params": {"model": "deepseek/deepseek-chat"}},
        ]

        result = processor._get_router_primary_model(mock_router)
        assert result == "openai/gpt-4o"

    def test_get_router_primary_model_empty(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test returns None for empty model list."""
        processor = LLMProcessor(llm_config, prompts_config)

        mock_router = MagicMock()
        mock_router.model_list = []

        result = processor._get_router_primary_model(mock_router)
        assert result is None

    def test_has_images_true(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test image detection in messages using has_images from providers.common."""
        from markitai.providers.common import has_images

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,x"},
                    },
                ],
            }
        ]

        assert has_images(messages) is True

    def test_has_images_false(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test text-only messages using has_images from providers.common."""
        from markitai.providers.common import has_images

        messages = [{"role": "user", "content": "Hello, world!"}]

        assert has_images(messages) is False


class TestLLMProcessorFormatOutput:
    """Tests for format_llm_output method."""

    def test_format_basic(self, llm_config: LLMConfig, prompts_config: PromptsConfig):
        """Test basic formatting."""
        processor = LLMProcessor(llm_config, prompts_config)

        result = processor.format_llm_output(
            "# Content", "title: Test\nsource: file.md"
        )

        assert result.startswith("---\n")
        assert "title: Test" in result
        # Content should be after frontmatter, may have trailing newline
        assert "\n\n# Content" in result

    def test_format_strips_existing_markers(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test strips existing --- markers."""
        processor = LLMProcessor(llm_config, prompts_config)

        result = processor.format_llm_output("# Content", "---\ntitle: Test\n---")

        # Should have exactly 2 markers
        assert result.count("---") == 2

    def test_format_strips_code_block(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test strips yaml code block."""
        processor = LLMProcessor(llm_config, prompts_config)

        result = processor.format_llm_output("# Content", "```yaml\ntitle: Test\n```")

        assert "```" not in result
        assert "title: Test" in result
        assert "\n\n# Content" in result


class TestLLMProcessorRouterCreation:
    """Tests for router creation logic."""

    def test_no_models_error(self, prompts_config: PromptsConfig):
        """Test error when no models configured."""
        config = LLMConfig(enabled=True, model_list=[])
        processor = LLMProcessor(config, prompts_config)

        with pytest.raises(ValueError, match="No models configured"):
            _ = processor.router

    def test_creates_local_provider_wrapper(self, prompts_config: PromptsConfig):
        """Test creates LocalProviderWrapper for all local models."""
        config = LLMConfig(
            enabled=True,
            model_list=[
                ModelConfig(
                    model_name="default",
                    litellm_params=LiteLLMParams(model="claude-agent/sonnet"),
                )
            ],
        )
        processor = LLMProcessor(config, prompts_config)

        with (
            patch("markitai.providers.is_local_provider_available", return_value=True),
            patch("markitai.providers.is_local_provider_model", return_value=True),
        ):
            router = processor.router
            assert isinstance(router, LocalProviderWrapper)

    def test_router_resolves_api_base_plain_url(self, prompts_config: PromptsConfig):
        """Test router passes plain api_base URL to litellm model list."""
        config = LLMConfig(
            enabled=True,
            model_list=[
                ModelConfig(
                    model_name="default",
                    litellm_params=LiteLLMParams(
                        model="openai/gpt-4o-mini",
                        api_key="test-key",
                        api_base="https://custom-proxy.example.com/v1",
                    ),
                )
            ],
        )
        processor = LLMProcessor(config, prompts_config)

        with patch("markitai.providers.is_local_provider_available", return_value=True):
            router = processor.router
            model_entry = router.model_list[0]
            assert (
                model_entry["litellm_params"]["api_base"]
                == "https://custom-proxy.example.com/v1"
            )

    def test_router_resolves_api_base_env_syntax(
        self, prompts_config: PromptsConfig, monkeypatch: pytest.MonkeyPatch
    ):
        """Test router resolves env:VAR_NAME in api_base before passing to litellm."""
        monkeypatch.setenv(
            "TEST_ROUTER_API_BASE", "https://env-resolved-proxy.example.com/v1"
        )
        config = LLMConfig(
            enabled=True,
            model_list=[
                ModelConfig(
                    model_name="default",
                    litellm_params=LiteLLMParams(
                        model="openai/gpt-4o-mini",
                        api_key="test-key",
                        api_base="env:TEST_ROUTER_API_BASE",
                    ),
                )
            ],
        )
        processor = LLMProcessor(config, prompts_config)

        with patch("markitai.providers.is_local_provider_available", return_value=True):
            router = processor.router
            model_entry = router.model_list[0]
            assert (
                model_entry["litellm_params"]["api_base"]
                == "https://env-resolved-proxy.example.com/v1"
            )

    def test_router_omits_api_base_when_none(self, prompts_config: PromptsConfig):
        """Test router does not include api_base when not configured."""
        config = LLMConfig(
            enabled=True,
            model_list=[
                ModelConfig(
                    model_name="default",
                    litellm_params=LiteLLMParams(
                        model="openai/gpt-4o-mini",
                        api_key="test-key",
                    ),
                )
            ],
        )
        processor = LLMProcessor(config, prompts_config)

        with patch("markitai.providers.is_local_provider_available", return_value=True):
            router = processor.router
            model_entry = router.model_list[0]
            assert "api_base" not in model_entry["litellm_params"]

    def test_semaphore_property(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test semaphore property creates semaphore."""
        processor = LLMProcessor(llm_config, prompts_config)

        sem = processor.semaphore
        assert sem is not None
        # Should return same instance
        assert processor.semaphore is sem

    def test_semaphore_uses_runtime(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test semaphore uses runtime semaphore when provided."""
        from markitai.llm.types import LLMRuntime

        runtime = LLMRuntime(concurrency=5)
        processor = LLMProcessor(llm_config, prompts_config, runtime=runtime)

        # Should use runtime's semaphore
        assert processor.semaphore is runtime.semaphore


class TestLLMProcessorVisionModel:
    """Tests for vision model detection."""

    def test_is_vision_model_config_override(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test config override for vision support."""
        processor = LLMProcessor(llm_config, prompts_config)

        model_config = MagicMock()
        model_config.litellm_params.model = "some/model"
        model_config.model_info = MagicMock()
        model_config.model_info.supports_vision = True

        assert processor._is_vision_model(model_config) is True

        model_config.model_info.supports_vision = False
        assert processor._is_vision_model(model_config) is False

    def test_is_vision_model_local_provider(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test local providers are always vision capable.

        The function imports is_local_provider_model from markitai.providers
        inside the method body. Local provider models (claude-agent/*, copilot/*)
        are recognized as vision-capable.
        """
        processor = LLMProcessor(llm_config, prompts_config)

        # Use actual local provider model ID that will be recognized
        model_config = MagicMock()
        model_config.litellm_params.model = "claude-agent/sonnet"
        model_config.model_info = None

        # The actual implementation will recognize claude-agent/ as local provider
        result = processor._is_vision_model(model_config)
        assert result is True

    def test_is_vision_model_auto_detect(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test auto-detection from litellm for standard models.

        For non-local provider models, the function uses get_model_info_cached
        to check for vision support. GPT-4o is a known vision model.
        """
        processor = LLMProcessor(llm_config, prompts_config)

        model_config = MagicMock()
        model_config.litellm_params.model = "openai/gpt-4o"
        model_config.model_info = None

        # GPT-4o is known to support vision in litellm
        # The actual get_model_info_cached will be called
        result = processor._is_vision_model(model_config)
        # GPT-4o should be detected as vision capable
        assert result is True
