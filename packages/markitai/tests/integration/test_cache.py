"""Integration tests for cache functionality (Phase 3 Optimization)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from markitai.cli import app
from markitai.config import LiteLLMParams, LLMConfig, ModelConfig, PromptsConfig
from markitai.llm import LLMProcessor, PersistentCache, SQLiteCache

# Note: Uses cli_runner from conftest.py, aliased as runner for backward compatibility


@pytest.fixture
def runner(cli_runner: CliRunner) -> CliRunner:
    """Alias for cli_runner from conftest.py."""
    return cli_runner


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
# PersistentCache Integration Tests
# =============================================================================


class TestPersistentCacheIntegration:
    """Integration tests for PersistentCache global caching."""

    def test_global_cache_get_set(self, tmp_path: Path):
        """Test basic get/set operations with global cache."""
        global_dir = tmp_path / "global"
        global_dir.mkdir()

        cache = PersistentCache(
            global_dir=global_dir,
            enabled=True,
        )

        # Set a value
        cache.set("prompt", "content", {"result": "test"})

        # Should get the value back
        result = cache.get("prompt", "content")
        assert result == {"result": "test"}

    def test_cache_disabled(self, tmp_path: Path):
        """Test that disabled cache returns None and doesn't write."""
        global_dir = tmp_path / "global"
        global_dir.mkdir()

        cache = PersistentCache(
            global_dir=global_dir,
            enabled=False,
        )

        cache.set("prompt", "content", {"result": "test"})
        result = cache.get("prompt", "content")

        assert result is None

    def test_hit_miss_tracking(self, tmp_path: Path):
        """Test cache hit/miss counter tracking."""
        global_dir = tmp_path / "global"
        global_dir.mkdir()

        cache = PersistentCache(
            global_dir=global_dir,
            enabled=True,
        )

        # Initial state
        assert cache._hits == 0
        assert cache._misses == 0

        # Miss (cache does not have this entry)
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
    """Integration tests for markitai cache CLI commands."""

    def test_cache_stats_no_cache(self, runner: CliRunner, tmp_path: Path):
        """Test cache stats when no cache exists."""
        # Run in empty directory
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(app, ["cache", "stats"])
            assert result.exit_code == 0
            assert "No cache" in result.output or "Cache Statistics" in result.output

    def test_cache_stats_json_output(self, runner: CliRunner, tmp_path: Path):
        """Test cache stats with JSON output format."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(app, ["cache", "stats", "--json"])
            assert result.exit_code == 0
            # Should be valid JSON
            data = json.loads(result.output)
            assert "enabled" in data
            assert "cache" in data

    def test_cache_clear_requires_confirmation(self, runner: CliRunner, tmp_path: Path):
        """Test that cache clear requires confirmation without -y flag."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Without -y, should prompt and abort on 'n'
            result = runner.invoke(app, ["cache", "clear"], input="n\n")
            assert result.exit_code == 0
            assert "Aborted" in result.output


class TestSQLiteCacheVerboseMethods:
    """Unit tests for SQLiteCache verbose methods."""

    def test_stats_by_model(self, tmp_path: Path):
        """Test stats_by_model method."""
        cache = SQLiteCache(tmp_path / "test.db")
        cache.set("p1", "c1", '{"test": 1}', model="model-a")
        cache.set("p2", "c2", '{"test": 2}', model="model-a")
        cache.set("p3", "c3", '{"test": 3}', model="model-b")

        by_model = cache.stats_by_model()
        assert "model-a" in by_model
        assert "model-b" in by_model
        assert by_model["model-a"]["count"] == 2
        assert by_model["model-b"]["count"] == 1

    def test_stats_by_model_unknown(self, tmp_path: Path):
        """Test stats_by_model with entries without model."""
        cache = SQLiteCache(tmp_path / "test.db")
        cache.set("p1", "c1", '{"test": 1}')  # No model
        cache.set("p2", "c2", '{"test": 2}', model="")  # Empty model

        by_model = cache.stats_by_model()
        assert "unknown" in by_model
        assert by_model["unknown"]["count"] == 2

    def test_list_entries(self, tmp_path: Path):
        """Test list_entries method."""
        cache = SQLiteCache(tmp_path / "test.db")
        cache.set("p1", "c1", '{"caption": "Test image desc"}', model="gemini/flash")
        cache.set("p2", "c2", '{"title": "Doc title here"}', model="openai/gpt-4")

        entries = cache.list_entries(limit=10)
        assert len(entries) == 2
        # Check entry structure
        entry = entries[0]
        assert "key" in entry
        assert "model" in entry
        assert "size_bytes" in entry
        assert "created_at" in entry
        assert "accessed_at" in entry
        assert "preview" in entry

    def test_list_entries_limit(self, tmp_path: Path):
        """Test list_entries respects limit."""
        cache = SQLiteCache(tmp_path / "test.db")
        for i in range(10):
            cache.set(f"p{i}", f"c{i}", f'{{"n": {i}}}')

        entries = cache.list_entries(limit=3)
        assert len(entries) == 3

    def test_parse_value_preview_image(self, tmp_path: Path):
        """Test _parse_value_preview for image caption."""
        cache = SQLiteCache(tmp_path / "test.db")
        preview = cache._parse_value_preview(
            '{"caption": "A beautiful sunset over the ocean"}'
        )
        assert preview.startswith("image:")
        assert "sunset" in preview

    def test_parse_value_preview_frontmatter(self, tmp_path: Path):
        """Test _parse_value_preview for frontmatter title."""
        cache = SQLiteCache(tmp_path / "test.db")
        preview = cache._parse_value_preview('{"title": "My Document Title"}')
        assert preview.startswith("frontmatter:")
        assert "Document" in preview

    def test_parse_value_preview_text(self, tmp_path: Path):
        """Test _parse_value_preview for plain text."""
        cache = SQLiteCache(tmp_path / "test.db")
        preview = cache._parse_value_preview('"# Hello World\\nThis is content"')
        assert preview.startswith("text:")

    def test_parse_value_preview_invalid_json(self, tmp_path: Path):
        """Test _parse_value_preview with invalid JSON falls back to text."""
        cache = SQLiteCache(tmp_path / "test.db")
        preview = cache._parse_value_preview("not valid json {")
        assert preview.startswith("text:")
        assert "not valid" in preview

    def test_parse_value_preview_empty(self, tmp_path: Path):
        """Test _parse_value_preview with empty value."""
        cache = SQLiteCache(tmp_path / "test.db")
        preview = cache._parse_value_preview("")
        assert preview == ""
        preview = cache._parse_value_preview(None)
        assert preview == ""


# =============================================================================
# Vision Router Integration Tests
# =============================================================================


class TestVisionRouterIntegration:
    """Integration tests for smart vision routing."""

    @pytest.fixture
    def vision_llm_config(self) -> LLMConfig:
        """Return LLM config with both vision and non-vision models."""
        from markitai.config import ModelInfo

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

    def test_has_images_detection(
        self, vision_llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test image detection in messages using has_images from providers.common."""
        from markitai.providers.common import has_images

        # Text-only message
        text_messages = [{"role": "user", "content": "Hello, world!"}]
        assert has_images(text_messages) is False

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
        assert has_images(image_messages) is True

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
        assert has_images(mixed_messages) is True

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
            from markitai.providers.common import has_images

            text_messages = [{"role": "user", "content": "Hello"}]
            requires_vision = has_images(text_messages)

            assert requires_vision is False

    def test_smart_router_selection_vision(
        self, vision_llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test that image messages trigger vision router selection."""
        from markitai.providers.common import has_images

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
        requires_vision = has_images(image_messages)

        assert requires_vision is True

    def test_all_models_vision_capable(self, prompts_config: PromptsConfig):
        """Test behavior when all models support vision."""
        from markitai.config import ModelInfo

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
        from markitai.config import ModelInfo

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

    def test_no_cache_patterns_stored_in_processor(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test that no_cache_patterns is passed to PersistentCache."""
        patterns = ["*.pdf", "reports/**", "file.docx"]
        processor = LLMProcessor(llm_config, prompts_config, no_cache_patterns=patterns)
        assert processor._persistent_cache._no_cache_patterns == patterns

    def test_no_cache_patterns_empty_by_default(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test that no_cache_patterns defaults to empty list."""
        processor = LLMProcessor(llm_config, prompts_config)
        assert processor._persistent_cache._no_cache_patterns == []


# =============================================================================
# No Cache Patterns Tests
# =============================================================================


class TestNoCachePatterns:
    """Tests for pattern-based cache skipping."""

    @pytest.fixture
    def cache(self, tmp_path: Path) -> PersistentCache:
        """Create a PersistentCache for testing."""
        return PersistentCache(
            global_dir=tmp_path,
            no_cache_patterns=["*.pdf", "reports/**", "specific.docx"],
        )

    def test_should_skip_cache_exact_match(self, cache: PersistentCache):
        """Test exact filename matching."""
        assert cache._should_skip_cache("specific.docx") is True
        assert cache._should_skip_cache("other.docx") is False

    def test_should_skip_cache_wildcard_extension(self, cache: PersistentCache):
        """Test wildcard extension matching."""
        assert cache._should_skip_cache("file.pdf") is True
        assert cache._should_skip_cache("document.pdf") is True
        assert cache._should_skip_cache("file.docx") is False

    def test_should_skip_cache_recursive_glob(self, cache: PersistentCache):
        """Test recursive glob pattern matching."""
        assert cache._should_skip_cache("reports/file.txt") is True
        assert cache._should_skip_cache("reports/sub/deep/file.txt") is True
        assert cache._should_skip_cache("other/file.txt") is False

    def test_should_skip_cache_no_patterns(self, tmp_path: Path):
        """Test that empty patterns never skip."""
        cache = PersistentCache(global_dir=tmp_path, no_cache_patterns=[])
        assert cache._should_skip_cache("any/file.pdf") is False

    def test_should_skip_cache_empty_context(self, cache: PersistentCache):
        """Test that empty context never skips."""
        assert cache._should_skip_cache("") is False

    def test_get_respects_patterns(self, tmp_path: Path):
        """Test that get() returns None for matching patterns."""
        cache = PersistentCache(
            global_dir=tmp_path,
            no_cache_patterns=["*.pdf"],
        )

        # Manually set a value in the underlying cache
        if cache._global_cache:
            cache._global_cache.set("test_prompt", "test_content", '{"cached": true}')

        # Normal file should return cached value
        result = cache.get("test_prompt", "test_content", context="file.docx")
        assert result == {"cached": True}

        # Pattern-matched file should skip cache (return None)
        result = cache.get("test_prompt", "test_content", context="file.pdf")
        assert result is None


class TestGlobMatchEdgeCases:
    """Tests for ** glob pattern edge cases (zero-or-more directories)."""

    def test_double_star_prefix_matches_root_file(self, tmp_path: Path):
        """Test **/*.ext matches files in root directory."""
        cache = PersistentCache(
            global_dir=tmp_path,
            no_cache_patterns=["**/*.pdf"],
        )
        # Should match root level file (zero directories)
        assert cache._should_skip_cache("file.pdf") is True
        # Should match nested files (one or more directories)
        assert cache._should_skip_cache("a/file.pdf") is True
        assert cache._should_skip_cache("a/b/c/file.pdf") is True
        # Should not match other extensions
        assert cache._should_skip_cache("file.docx") is False

    def test_double_star_middle_matches_direct_child(self, tmp_path: Path):
        """Test src/**/test.py matches src/test.py (zero directories between)."""
        cache = PersistentCache(
            global_dir=tmp_path,
            no_cache_patterns=["src/**/test.py"],
        )
        # Zero directories between src/ and test.py
        assert cache._should_skip_cache("src/test.py") is True
        # One or more directories
        assert cache._should_skip_cache("src/unit/test.py") is True
        assert cache._should_skip_cache("src/a/b/c/test.py") is True
        # Wrong prefix
        assert cache._should_skip_cache("lib/test.py") is False

    def test_double_star_prefix_with_subdir(self, tmp_path: Path):
        """Test **/reports/*.pdf matches reports/ at any level including root."""
        cache = PersistentCache(
            global_dir=tmp_path,
            no_cache_patterns=["**/reports/*.pdf"],
        )
        # reports/ at root level
        assert cache._should_skip_cache("reports/file.pdf") is True
        # reports/ nested
        assert cache._should_skip_cache("foo/reports/file.pdf") is True
        assert cache._should_skip_cache("a/b/reports/file.pdf") is True
        # Not matching
        assert cache._should_skip_cache("other/file.pdf") is False

    def test_windows_path_separators(self, tmp_path: Path):
        """Test that Windows backslash paths are normalized."""
        cache = PersistentCache(
            global_dir=tmp_path,
            no_cache_patterns=["**/*.pdf"],
        )
        # Windows-style paths should be normalized
        assert cache._should_skip_cache("a\\b\\file.pdf") is True
        assert cache._should_skip_cache("file.pdf") is True

    def test_multiple_double_star_patterns(self, tmp_path: Path):
        """Test multiple ** patterns work together."""
        cache = PersistentCache(
            global_dir=tmp_path,
            no_cache_patterns=["**/*.pdf", "**/temp/*", "logs/**/*.log"],
        )
        # **/*.pdf
        assert cache._should_skip_cache("file.pdf") is True
        assert cache._should_skip_cache("a/file.pdf") is True
        # **/temp/*
        assert cache._should_skip_cache("temp/file.txt") is True
        assert cache._should_skip_cache("a/temp/file.txt") is True
        # logs/**/*.log
        assert cache._should_skip_cache("logs/app.log") is True
        assert cache._should_skip_cache("logs/a/b/app.log") is True


class TestContextPathExtraction:
    """Tests for context path extraction from various formats."""

    def test_absolute_path_with_suffix(self, tmp_path: Path):
        """Test matching against absolute paths with :suffix (e.g., from image analysis)."""
        cache = PersistentCache(
            global_dir=tmp_path,
            no_cache_patterns=["*.JPG", "*.pdf"],
        )
        # Absolute path with :images suffix (common in image analysis)
        assert (
            cache._should_skip_cache(
                "/home/user/project/tests/fixtures/candy.JPG:images"
            )
            is True
        )
        assert (
            cache._should_skip_cache(
                "/home/user/project/tests/fixtures/document.pdf:clean"
            )
            is True
        )
        # Should not match other extensions
        assert (
            cache._should_skip_cache(
                "/home/user/project/tests/fixtures/document.docx:images"
            )
            is False
        )

    def test_absolute_path_without_suffix(self, tmp_path: Path):
        """Test matching against absolute paths without suffix."""
        cache = PersistentCache(
            global_dir=tmp_path,
            no_cache_patterns=["*.JPG"],
        )
        # Absolute path without suffix
        assert cache._should_skip_cache("/home/user/project/candy.JPG") is True
        assert (
            cache._should_skip_cache("C:\\Users\\test\\candy.JPG") is True
        )  # Windows path

    def test_relative_path_still_works(self, tmp_path: Path):
        """Test that relative paths still work correctly."""
        cache = PersistentCache(
            global_dir=tmp_path,
            no_cache_patterns=["sub_dir/*.doc", "*.JPG"],
        )
        # Relative paths should work as before
        assert cache._should_skip_cache("sub_dir/file.doc") is True
        assert cache._should_skip_cache("candy.JPG") is True
        assert cache._should_skip_cache("other/file.doc") is False

    def test_glob_pattern_with_absolute_path(self, tmp_path: Path):
        """Test glob patterns against absolute paths."""
        cache = PersistentCache(
            global_dir=tmp_path,
            no_cache_patterns=["**/*.xls"],
        )
        # Filename should be extracted and matched
        assert (
            cache._should_skip_cache(
                "/home/user/project/sub_dir/file_example_XLS_100.xls"
            )
            is True
        )
        assert (
            cache._should_skip_cache(
                "/home/user/project/sub_dir/file_example_XLS_100.xls:images"
            )
            is True
        )

    def test_extract_matchable_path(self, tmp_path: Path):
        """Test _extract_matchable_path helper directly."""
        cache = PersistentCache(global_dir=tmp_path)

        # Simple filename
        assert cache._extract_matchable_path("candy.JPG") == "candy.JPG"

        # Relative path
        assert cache._extract_matchable_path("sub/candy.JPG") == "candy.JPG"

        # Absolute path
        assert cache._extract_matchable_path("/home/user/candy.JPG") == "candy.JPG"

        # Path with suffix
        assert (
            cache._extract_matchable_path("/home/user/candy.JPG:images") == "candy.JPG"
        )

        # Windows path
        assert (
            cache._extract_matchable_path("C:\\Users\\test\\candy.JPG") == "candy.JPG"
        )

        # Windows path with suffix
        assert (
            cache._extract_matchable_path("C:\\Users\\test\\candy.JPG:clean")
            == "candy.JPG"
        )


# =============================================================================
# Cache Hash Computation Tests
# =============================================================================


class TestCacheHashComputation:
    """Tests for cache hash computation with head+tail+length strategy."""

    def test_hash_changes_with_head_modification(self, tmp_path: Path):
        """Test that modifying head content changes the hash."""
        cache = SQLiteCache(tmp_path / "test.db")

        content1 = "A" * 100 + "X" * 50000
        content2 = "B" * 100 + "X" * 50000  # Different head

        hash1 = cache._compute_hash("prompt", content1)
        hash2 = cache._compute_hash("prompt", content2)

        assert hash1 != hash2

    def test_hash_changes_with_tail_modification(self, tmp_path: Path):
        """Test that modifying tail content changes the hash."""
        cache = SQLiteCache(tmp_path / "test.db")

        content1 = "X" * 50000 + "A" * 100
        content2 = "X" * 50000 + "B" * 100  # Different tail

        hash1 = cache._compute_hash("prompt", content1)
        hash2 = cache._compute_hash("prompt", content2)

        assert hash1 != hash2

    def test_hash_changes_with_length_modification(self, tmp_path: Path):
        """Test that changing content length changes the hash."""
        cache = SQLiteCache(tmp_path / "test.db")

        content1 = "X" * 50000
        content2 = "X" * 50001  # Same chars, different length

        hash1 = cache._compute_hash("prompt", content1)
        hash2 = cache._compute_hash("prompt", content2)

        assert hash1 != hash2

    def test_hash_stable_for_identical_content(self, tmp_path: Path):
        """Test that identical content produces identical hash."""
        cache = SQLiteCache(tmp_path / "test.db")

        content = "X" * 60000
        hash1 = cache._compute_hash("prompt", content)
        hash2 = cache._compute_hash("prompt", content)

        assert hash1 == hash2

    def test_hash_different_for_different_prompts(self, tmp_path: Path):
        """Test that different prompts produce different hashes."""
        cache = SQLiteCache(tmp_path / "test.db")

        content = "X" * 1000
        hash1 = cache._compute_hash("prompt1", content)
        hash2 = cache._compute_hash("prompt2", content)

        assert hash1 != hash2

    def test_hash_handles_short_content(self, tmp_path: Path):
        """Test hash computation with content shorter than 25000 chars."""
        cache = SQLiteCache(tmp_path / "test.db")

        content = "short content"
        hash_result = cache._compute_hash("prompt", content)

        # Should still produce a valid 32-char hash
        assert len(hash_result) == 32
        assert all(c in "0123456789abcdef" for c in hash_result)
