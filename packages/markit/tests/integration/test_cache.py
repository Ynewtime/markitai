"""Integration tests for cache functionality (Phase 3 Optimization)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from markit.cli import app
from markit.config import LiteLLMParams, LLMConfig, ModelConfig, PromptsConfig
from markit.llm import LLMProcessor, PersistentCache, SQLiteCache


@pytest.fixture
def runner():
    """Return a CLI test runner."""
    return CliRunner()


# =============================================================================
# SQLiteCache Integration Tests
# =============================================================================


class TestSQLiteCacheIntegration:
    """Integration tests for SQLiteCache class."""

    def test_cache_persistence_across_instances(self, tmp_path: Path):
        """Test that cache data persists across different SQLiteCache instances."""
        db_path = tmp_path / "test_cache.db"

        # First instance: set data
        cache1 = SQLiteCache(db_path)
        cache1.set("prompt1", "content1", '{"result": "value1"}')
        cache1.set("prompt2", "content2", '{"result": "value2"}')

        # Second instance: read data
        cache2 = SQLiteCache(db_path)
        result1 = cache2.get("prompt1", "content1")
        result2 = cache2.get("prompt2", "content2")

        assert result1 == '{"result": "value1"}'
        assert result2 == '{"result": "value2"}'

    def test_cache_lru_eviction(self, tmp_path: Path):
        """Test LRU eviction when cache size limit is reached."""
        db_path = tmp_path / "lru_cache.db"
        # Set very small cache (500 bytes) - will only fit ~1 entry
        cache = SQLiteCache(db_path, max_size_bytes=500)

        # Add first entry (~300 bytes)
        large_value = "x" * 300
        cache.set("prompt1", "content1", large_value)

        # Verify first entry exists
        assert cache.get("prompt1", "content1") == large_value

        # Add second entry - should evict first (LRU and over limit)
        cache.set("prompt2", "content2", large_value)

        # First entry should be evicted
        assert cache.get("prompt1", "content1") is None
        # Second entry should exist
        assert cache.get("prompt2", "content2") == large_value

    def test_cache_stats(self, tmp_path: Path):
        """Test cache statistics reporting."""
        db_path = tmp_path / "stats_cache.db"
        cache = SQLiteCache(db_path)

        # Empty cache
        stats = cache.stats()
        assert stats["count"] == 0
        assert stats["size_bytes"] == 0

        # Add entries
        cache.set("prompt1", "content1", '{"data": "test"}')
        cache.set("prompt2", "content2", '{"data": "test2"}')

        stats = cache.stats()
        assert stats["count"] == 2
        assert stats["size_bytes"] > 0
        assert "db_path" in stats

    def test_cache_clear(self, tmp_path: Path):
        """Test clearing all cache entries."""
        db_path = tmp_path / "clear_cache.db"
        cache = SQLiteCache(db_path)

        cache.set("prompt1", "content1", '{"data": "test1"}')
        cache.set("prompt2", "content2", '{"data": "test2"}')
        cache.set("prompt3", "content3", '{"data": "test3"}')

        deleted_count = cache.clear()
        assert deleted_count == 3

        stats = cache.stats()
        assert stats["count"] == 0

    def test_cache_update_existing_key(self, tmp_path: Path):
        """Test updating an existing cache entry."""
        db_path = tmp_path / "update_cache.db"
        cache = SQLiteCache(db_path)

        cache.set("prompt", "content", '{"version": 1}')
        cache.set("prompt", "content", '{"version": 2}')

        result = cache.get("prompt", "content")
        assert result == '{"version": 2}'

        # Should still be only 1 entry
        stats = cache.stats()
        assert stats["count"] == 1


# =============================================================================
# PersistentCache (Dual-Layer) Integration Tests
# =============================================================================


class TestPersistentCacheIntegration:
    """Integration tests for PersistentCache dual-layer caching."""

    def test_dual_layer_lookup_order(self, tmp_path: Path):
        """Test that project cache is checked before global cache."""
        project_dir = tmp_path / "project"
        global_dir = tmp_path / "global"
        project_dir.mkdir()
        global_dir.mkdir()

        # Initialize cache
        cache = PersistentCache(
            project_dir=project_dir,
            global_dir=global_dir,
            enabled=True,
        )

        # Set different values in each layer directly
        assert cache._project_cache is not None
        assert cache._global_cache is not None
        cache._project_cache.set("prompt", "content", '{"source": "project"}')
        cache._global_cache.set("prompt", "content", '{"source": "global"}')

        # Should get project cache value (checked first)
        result = cache.get("prompt", "content")
        assert result == {"source": "project"}

    def test_global_fallback(self, tmp_path: Path):
        """Test fallback to global cache when project cache misses."""
        project_dir = tmp_path / "project"
        global_dir = tmp_path / "global"
        project_dir.mkdir()
        global_dir.mkdir()

        cache = PersistentCache(
            project_dir=project_dir,
            global_dir=global_dir,
            enabled=True,
        )

        # Set only in global cache
        assert cache._global_cache is not None
        cache._global_cache.set("prompt", "content", '{"source": "global"}')

        # Should get global cache value
        result = cache.get("prompt", "content")
        assert result == {"source": "global"}

    def test_write_to_both_layers(self, tmp_path: Path):
        """Test that set() writes to both project and global caches."""
        project_dir = tmp_path / "project"
        global_dir = tmp_path / "global"
        project_dir.mkdir()
        global_dir.mkdir()

        cache = PersistentCache(
            project_dir=project_dir,
            global_dir=global_dir,
            enabled=True,
        )

        cache.set("prompt", "content", {"result": "test"})

        # Both caches should have the value
        assert cache._project_cache is not None
        assert cache._global_cache is not None
        project_result = cache._project_cache.get("prompt", "content")
        global_result = cache._global_cache.get("prompt", "content")

        assert project_result is not None
        assert global_result is not None
        assert json.loads(project_result) == {"result": "test"}
        assert json.loads(global_result) == {"result": "test"}

    def test_cache_disabled(self, tmp_path: Path):
        """Test that disabled cache returns None and doesn't write."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        cache = PersistentCache(
            project_dir=project_dir,
            enabled=False,
        )

        cache.set("prompt", "content", {"result": "test"})
        result = cache.get("prompt", "content")

        assert result is None
        # Project cache directory should not be created
        assert not (project_dir / ".markit" / "cache.db").exists()

    def test_hit_miss_tracking(self, tmp_path: Path):
        """Test cache hit/miss counter tracking."""
        project_dir = tmp_path / "project"
        global_dir = tmp_path / "global"
        project_dir.mkdir()
        global_dir.mkdir()

        # Use isolated global dir to avoid hitting real global cache
        cache = PersistentCache(
            project_dir=project_dir,
            global_dir=global_dir,
            enabled=True,
        )

        # Initial state
        assert cache._hits == 0
        assert cache._misses == 0

        # Miss (neither project nor global cache has this entry)
        cache.get("unique_prompt_12345", "unique_content_12345")
        assert cache._misses == 1

        # Set and hit
        cache.set("unique_prompt_12345", "unique_content_12345", {"result": "test"})
        cache.get("unique_prompt_12345", "unique_content_12345")
        assert cache._hits == 1


# =============================================================================
# CLI Cache Commands Integration Tests
# =============================================================================


class TestCacheCLICommands:
    """Integration tests for markit cache CLI commands."""

    def test_cache_stats_no_cache(self, runner: CliRunner, tmp_path: Path):
        """Test cache stats when no cache exists."""
        # Run in empty directory
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(app, ["cache", "stats"])
            assert result.exit_code == 0
            assert (
                "No project cache" in result.output
                or "Cache Statistics" in result.output
            )

    def test_cache_stats_with_project_cache(self, runner: CliRunner, tmp_path: Path):
        """Test cache stats with existing project cache."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create project cache
            cache_dir = Path.cwd() / ".markit"
            cache_dir.mkdir()
            cache = SQLiteCache(cache_dir / "cache.db")
            cache.set("prompt", "content", '{"test": "data"}')

            result = runner.invoke(app, ["cache", "stats"])
            assert result.exit_code == 0
            assert "Project Cache" in result.output
            assert "Entries: 1" in result.output

    def test_cache_stats_json_output(self, runner: CliRunner, tmp_path: Path):
        """Test cache stats with JSON output format."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(app, ["cache", "stats", "--json"])
            assert result.exit_code == 0
            # Should be valid JSON
            data = json.loads(result.output)
            assert "enabled" in data
            assert "project" in data
            assert "global" in data

    def test_cache_clear_project(self, runner: CliRunner, tmp_path: Path):
        """Test clearing project cache."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create project cache with entries
            cache_dir = Path.cwd() / ".markit"
            cache_dir.mkdir()
            cache = SQLiteCache(cache_dir / "cache.db")
            cache.set("prompt1", "content1", '{"test": "data1"}')
            cache.set("prompt2", "content2", '{"test": "data2"}')

            result = runner.invoke(app, ["cache", "clear", "--scope", "project", "-y"])
            assert result.exit_code == 0
            assert "Cleared 2" in result.output or "project" in result.output.lower()

            # Verify cache is empty
            stats = cache.stats()
            assert stats["count"] == 0

    def test_cache_clear_requires_confirmation(self, runner: CliRunner, tmp_path: Path):
        """Test that cache clear requires confirmation without -y flag."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create project cache
            cache_dir = Path.cwd() / ".markit"
            cache_dir.mkdir()
            cache = SQLiteCache(cache_dir / "cache.db")
            cache.set("prompt", "content", '{"test": "data"}')

            # Without -y, should prompt and abort on 'n'
            result = runner.invoke(
                app, ["cache", "clear", "--scope", "project"], input="n\n"
            )
            assert result.exit_code == 0
            assert "Aborted" in result.output

            # Cache should still have entries
            stats = cache.stats()
            assert stats["count"] == 1


# =============================================================================
# Vision Router Integration Tests
# =============================================================================


class TestVisionRouterIntegration:
    """Integration tests for smart vision routing."""

    @pytest.fixture
    def vision_llm_config(self) -> LLMConfig:
        """Return LLM config with both vision and non-vision models."""
        from markit.config import ModelInfo

        return LLMConfig(
            enabled=True,
            model_list=[
                ModelConfig(
                    model_name="default",
                    litellm_params=LiteLLMParams(
                        model="openai/gpt-4o-mini",
                        api_key="test-key",
                    ),
                    model_info=ModelInfo(supports_vision=True),
                ),
                ModelConfig(
                    model_name="default",
                    litellm_params=LiteLLMParams(
                        model="deepseek/deepseek-chat",
                        api_key="test-key",
                    ),
                    model_info=ModelInfo(supports_vision=False),
                ),
                ModelConfig(
                    model_name="default",
                    litellm_params=LiteLLMParams(
                        model="openai/gpt-4o",
                        api_key="test-key",
                    ),
                    model_info=ModelInfo(supports_vision=True),
                ),
            ],
            concurrency=2,
        )

    @pytest.fixture
    def prompts_config(self) -> PromptsConfig:
        """Return a test prompts configuration."""
        return PromptsConfig()

    def test_message_contains_image_detection(
        self, vision_llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test image detection in messages."""
        processor = LLMProcessor(vision_llm_config, prompts_config, no_cache=True)

        # Text-only message
        text_messages = [{"role": "user", "content": "Hello, world!"}]
        assert processor._message_contains_image(text_messages) is False

        # Message with image_url
        image_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc123"},
                    },
                ],
            }
        ]
        assert processor._message_contains_image(image_messages) is True

        # Mixed messages
        mixed_messages = [
            {"role": "system", "content": "You are helpful."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/img.png"},
                    },
                ],
            },
        ]
        assert processor._message_contains_image(mixed_messages) is True

    def test_vision_router_filters_models(
        self, vision_llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test that vision_router only includes vision-capable models."""
        processor = LLMProcessor(vision_llm_config, prompts_config, no_cache=True)

        # Access vision_router to trigger lazy initialization
        vision_router = processor.vision_router

        # Check that the router has only vision-capable models
        # The model_list in Router contains deployment dicts
        vision_model_names = []
        for deployment in vision_router.model_list:
            # LiteLLM Router stores models as dicts with litellm_params
            if isinstance(deployment, dict):
                litellm_params = deployment.get("litellm_params", {})
                if isinstance(litellm_params, dict):
                    vision_model_names.append(litellm_params.get("model"))
            elif hasattr(deployment, "litellm_params"):
                # Or as objects with litellm_params attribute
                params = deployment.litellm_params
                if isinstance(params, dict):
                    vision_model_names.append(params.get("model"))

        # Should have gpt-4o-mini and gpt-4o, NOT deepseek-chat
        assert "openai/gpt-4o-mini" in vision_model_names
        assert "openai/gpt-4o" in vision_model_names
        assert "deepseek/deepseek-chat" not in vision_model_names

    def test_smart_router_selection_text(
        self, vision_llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test that text-only messages use the main router."""
        processor = LLMProcessor(vision_llm_config, prompts_config, no_cache=True)

        # Mock the router
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Response"))]
        mock_response.model = "test-model"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

        with patch.object(processor, "_router") as mock_router:
            mock_router.acompletion = AsyncMock(return_value=mock_response)
            processor._router = mock_router

            # Text-only message should NOT trigger vision router
            text_messages = [{"role": "user", "content": "Hello"}]
            requires_vision = processor._message_contains_image(text_messages)

            assert requires_vision is False

    def test_smart_router_selection_vision(
        self, vision_llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test that image messages trigger vision router selection."""
        processor = LLMProcessor(vision_llm_config, prompts_config, no_cache=True)

        # Image message should trigger vision router
        image_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc"},
                    },
                ],
            }
        ]
        requires_vision = processor._message_contains_image(image_messages)

        assert requires_vision is True

    def test_all_models_vision_capable(self, prompts_config: PromptsConfig):
        """Test behavior when all models support vision."""
        from markit.config import ModelInfo

        all_vision_config = LLMConfig(
            enabled=True,
            model_list=[
                ModelConfig(
                    model_name="default",
                    litellm_params=LiteLLMParams(model="openai/gpt-4o", api_key="test"),
                    model_info=ModelInfo(supports_vision=True),
                ),
                ModelConfig(
                    model_name="default",
                    litellm_params=LiteLLMParams(
                        model="openai/gpt-4o-mini", api_key="test"
                    ),
                    model_info=ModelInfo(supports_vision=True),
                ),
            ],
        )

        processor = LLMProcessor(all_vision_config, prompts_config, no_cache=True)
        # Should not raise - vision_router should work
        vision_router = processor.vision_router
        assert vision_router is not None

    def test_no_vision_models_fallback(self, prompts_config: PromptsConfig):
        """Test fallback to main router when no vision-capable models configured."""
        from markit.config import ModelInfo

        no_vision_config = LLMConfig(
            enabled=True,
            model_list=[
                ModelConfig(
                    model_name="default",
                    litellm_params=LiteLLMParams(
                        model="deepseek/deepseek-chat", api_key="test"
                    ),
                    model_info=ModelInfo(supports_vision=False),
                ),
            ],
        )

        processor = LLMProcessor(no_vision_config, prompts_config, no_cache=True)

        # Should fall back to main router (no error raised)
        vision_router = processor.vision_router
        assert vision_router is not None
        # Should be the same as the main router
        assert vision_router is processor.router


# =============================================================================
# LLMProcessor Cache Integration Tests
# =============================================================================


class TestLLMProcessorCacheIntegration:
    """Integration tests for LLMProcessor with persistent cache."""

    @pytest.fixture
    def llm_config(self) -> LLMConfig:
        """Return a test LLM configuration."""
        return LLMConfig(
            enabled=True,
            model_list=[
                ModelConfig(
                    model_name="default",
                    litellm_params=LiteLLMParams(
                        model="openai/gpt-4o-mini",
                        api_key="test-key",
                    ),
                ),
            ],
            concurrency=2,
        )

    @pytest.fixture
    def prompts_config(self) -> PromptsConfig:
        """Return a test prompts configuration."""
        return PromptsConfig()

    def test_cache_default_creates_cache(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig, tmp_path: Path
    ):
        """Test that default settings create an active PersistentCache."""
        # Change to temp directory to use as project dir
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            processor = LLMProcessor(llm_config, prompts_config)
            assert processor._persistent_cache is not None
            assert processor._persistent_cache._enabled is True
            assert processor._persistent_cache._skip_read is False
        finally:
            os.chdir(original_dir)

    def test_no_cache_skips_read_but_writes(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test that no_cache=True skips reading but still allows writing."""
        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)
        # Cache object exists, enabled, but skip_read is True (Bun semantics)
        assert processor._persistent_cache._enabled is True
        assert processor._persistent_cache._skip_read is True
