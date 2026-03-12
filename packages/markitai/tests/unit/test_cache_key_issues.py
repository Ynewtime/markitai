"""Tests for cache key correctness.

High-5: SQLiteCache._compute_hash should use full content hash, include model.
High-6: Vision cache should include document_context in key.
"""

from __future__ import annotations

from pathlib import Path

import pytest


class TestSQLiteCacheKeyFullContent:
    """High-5: Cache key must hash full content, not just head+tail."""

    def test_middle_content_change_produces_different_key(self, tmp_path: Path) -> None:
        """Content differing only in the middle (beyond head+tail window) must produce different cache keys."""
        from markitai.llm.cache import SQLiteCache

        db_path = tmp_path / "cache.db"
        cache = SQLiteCache(db_path)

        # Create two contents that differ only in the middle
        # Old implementation used 25k head + 25k tail, so middle of 60k content was invisible
        head = "A" * 30000
        tail = "Z" * 30000
        content_a = head + "MIDDLE_VERSION_A" + tail
        content_b = head + "MIDDLE_VERSION_B" + tail

        prompt = "cleaner"

        # Store with content_a
        cache.set(prompt, content_a, '{"result": "version_a"}')

        # Lookup with content_b should NOT hit the cache
        result = cache.get(prompt, content_b)
        assert result is None, (
            "Middle-only content change must produce a cache miss, "
            "but got a hit — cache key does not hash full content"
        )

    def test_model_in_cache_key_differentiates_results(self, tmp_path: Path) -> None:
        """Same prompt+content but different model should produce different cache entries."""
        from markitai.llm.cache import SQLiteCache

        db_path = tmp_path / "cache.db"
        cache = SQLiteCache(db_path)

        prompt = "cleaner"
        content = "Some markdown content"

        # Store result for model_a
        cache.set(prompt, content, '{"result": "from_model_a"}', model="gpt-4o")

        # Store result for model_b — should NOT overwrite model_a's entry
        cache.set(prompt, content, '{"result": "from_model_b"}', model="claude-sonnet")

        # We need a way to query by model. The key itself should incorporate model.
        # With model in the hash, both entries coexist.
        # Verify by checking that the cache has 2 entries, not 1.
        stats = cache.stats()
        assert stats["count"] == 2, (
            f"Expected 2 cache entries (one per model), got {stats['count']}. "
            "Cache key does not incorporate model identifier."
        )

    def test_compute_hash_uses_full_content(self, tmp_path: Path) -> None:
        """_compute_hash should hash the full content, not just head+tail."""
        from markitai.llm.cache import SQLiteCache

        db_path = tmp_path / "cache.db"
        cache = SQLiteCache(db_path)

        # Two short contents that differ
        hash_a = cache._compute_hash("prompt", "content_a")
        hash_b = cache._compute_hash("prompt", "content_b")
        assert hash_a != hash_b

        # Two long contents that differ only in the middle
        base = "X" * 30000
        tail = "Y" * 30000
        long_a = base + "DIFFER_A" + tail
        long_b = base + "DIFFER_B" + tail

        hash_long_a = cache._compute_hash("prompt", long_a)
        hash_long_b = cache._compute_hash("prompt", long_b)
        assert hash_long_a != hash_long_b, (
            "Full content hash must detect middle-only differences"
        )


class TestSQLiteCacheKeyIncludesModel:
    """High-5: _compute_hash should accept and incorporate model parameter."""

    def test_compute_hash_with_model_differs_from_without(self, tmp_path: Path) -> None:
        """Hash with model specified should differ from hash without model."""
        from markitai.llm.cache import SQLiteCache

        db_path = tmp_path / "cache.db"
        cache = SQLiteCache(db_path)

        hash_no_model = cache._compute_hash("cleaner", "content")
        hash_with_model = cache._compute_hash("cleaner", "content", model="gpt-4o")
        assert hash_no_model != hash_with_model, (
            "_compute_hash should produce different keys when model differs"
        )

    def test_compute_hash_different_models_different_keys(self, tmp_path: Path) -> None:
        """Different models should produce different hash keys."""
        from markitai.llm.cache import SQLiteCache

        db_path = tmp_path / "cache.db"
        cache = SQLiteCache(db_path)

        hash_a = cache._compute_hash("cleaner", "content", model="gpt-4o")
        hash_b = cache._compute_hash("cleaner", "content", model="claude-sonnet")
        assert hash_a != hash_b


class TestPersistentCacheModelParameter:
    """High-5: PersistentCache.get/set should pass model through to hash computation."""

    def test_get_set_with_model_isolation(self, tmp_path: Path) -> None:
        """get() with model=A should not return result stored with model=B."""
        from markitai.llm.cache import PersistentCache

        cache = PersistentCache(global_dir=tmp_path, enabled=True)

        prompt = "cleaner"
        content = "Some content"

        # Store with model A
        cache.set(prompt, content, {"result": "model_a"}, model="gpt-4o")

        # Get with model B should miss
        result = cache.get(prompt, content, model="claude-sonnet")
        assert result is None, (
            "PersistentCache.get with different model should return None"
        )

        # Get with model A should hit
        result = cache.get(prompt, content, model="gpt-4o")
        assert result is not None
        assert result["result"] == "model_a"


class TestVisionCacheIncludesDocumentContext:
    """High-6: Vision cache key must include document_context hash."""

    def test_same_image_different_context_different_cache_entries(
        self, tmp_path: Path
    ) -> None:
        """Same image analyzed with different document_context should not share cache."""
        from markitai.llm.cache import PersistentCache

        cache = PersistentCache(global_dir=tmp_path, enabled=True)

        # Simulate vision cache usage pattern
        import hashlib

        image_data = b"fake_image_bytes_for_test"
        base64_image = __import__("base64").b64encode(image_data).decode()
        image_fingerprint = hashlib.sha256(base64_image.encode()).hexdigest()

        cache_key = "image_analysis"

        # Store result for context A — include document_context in content key
        context_a = "This is a technical document about Python programming"
        content_key_a = f"{image_fingerprint}|ctx:{hashlib.sha256(context_a.encode()).hexdigest()[:16]}"
        cache.set(cache_key, content_key_a, {"caption": "Python code snippet"})

        # Lookup with context B — should NOT hit
        context_b = "This is a Japanese cooking recipe"
        content_key_b = f"{image_fingerprint}|ctx:{hashlib.sha256(context_b.encode()).hexdigest()[:16]}"
        result = cache.get(cache_key, content_key_b)

        assert result is None, (
            "Same image with different document_context should produce cache miss"
        )

        # Lookup with same context A — should hit
        result = cache.get(cache_key, content_key_a)
        assert result is not None
        assert result["caption"] == "Python code snippet"

    def test_no_document_context_still_caches(self, tmp_path: Path) -> None:
        """When document_context is empty/None, caching should still work."""
        from markitai.llm.cache import PersistentCache

        cache = PersistentCache(global_dir=tmp_path, enabled=True)

        import hashlib

        image_data = b"another_image"
        base64_image = __import__("base64").b64encode(image_data).decode()
        image_fingerprint = hashlib.sha256(base64_image.encode()).hexdigest()

        cache_key = "image_analysis"
        # No context — just fingerprint
        content_key = image_fingerprint

        cache.set(cache_key, content_key, {"caption": "generic caption"})
        result = cache.get(cache_key, content_key)

        assert result is not None
        assert result["caption"] == "generic caption"


class TestVisionCacheKeyIntegration:
    """High-6: VisionMixin.analyze_image should build cache key incorporating document_context."""

    @pytest.mark.asyncio
    async def test_analyze_image_uses_document_context_in_cache_key(
        self, tmp_path: Path
    ) -> None:
        """analyze_image called with different document_context should not share cache."""
        import asyncio
        import base64
        from unittest.mock import MagicMock

        from markitai.llm.types import ImageAnalysis
        from markitai.llm.vision import VisionMixin

        # Create a minimal mock processor
        class TestProcessor(VisionMixin):
            def __init__(self):
                self._semaphore = asyncio.Semaphore(2)
                self._persistent_cache = MagicMock()
                self._prompt_manager = MagicMock()
                self._prompt_manager.get_prompt.return_value = "mock prompt"
                self.vision_router = MagicMock()
                self.config = MagicMock()
                self.config.concurrency = 2
                self._image_cache: dict = {}

            @property
            def semaphore(self):
                return self._semaphore

            def _get_cached_image(self, image_path):
                data = b"test_image_data_xyz"
                b64 = base64.b64encode(data).decode()
                return data, b64

            def _calculate_dynamic_max_tokens(self, messages, router=None):
                return 4096

            def _track_usage(self, *args, **kwargs):
                pass

        processor = TestProcessor()

        # Create a test image
        test_image = tmp_path / "test.png"
        test_image.write_bytes(b"test_image_data_xyz")

        # Track what cache keys are used
        cache_get_calls: list[tuple] = []
        processor._persistent_cache.get.side_effect = lambda *args, **kwargs: (
            cache_get_calls.append((args, kwargs)) or None
        )
        processor._persistent_cache.set.return_value = None

        # Mock _analyze_image_with_fallback to return a result
        async def mock_analyze(*args, **kwargs):
            return ImageAnalysis(caption="test", description="test desc")

        processor._analyze_image_with_fallback = mock_analyze  # type: ignore

        # Call with context A
        await processor.analyze_image(
            test_image, context="file.pdf", document_context="Python programming guide"
        )

        # Call with context B
        await processor.analyze_image(
            test_image, context="file.pdf", document_context="Japanese cooking recipe"
        )

        # The two cache.get calls should have used DIFFERENT content keys
        assert len(cache_get_calls) == 2, (
            f"Expected 2 cache lookups, got {len(cache_get_calls)}"
        )

        # Extract the content parameter (second positional arg) from each call
        content_key_a = cache_get_calls[0][0][1]  # args[1]
        content_key_b = cache_get_calls[1][0][1]  # args[1]

        assert content_key_a != content_key_b, (
            f"Cache content keys should differ when document_context differs, "
            f"but both were: {content_key_a}"
        )

    @pytest.mark.asyncio
    async def test_analyze_image_no_context_uses_fingerprint_only(
        self, tmp_path: Path
    ) -> None:
        """analyze_image without document_context should use just the image fingerprint."""
        import asyncio
        import base64
        import hashlib
        from unittest.mock import MagicMock

        from markitai.llm.types import ImageAnalysis
        from markitai.llm.vision import VisionMixin

        class TestProcessor(VisionMixin):
            def __init__(self):
                self._semaphore = asyncio.Semaphore(2)
                self._persistent_cache = MagicMock()
                self._prompt_manager = MagicMock()
                self._prompt_manager.get_prompt.return_value = "mock prompt"
                self.vision_router = MagicMock()
                self.config = MagicMock()
                self.config.concurrency = 2

            @property
            def semaphore(self):
                return self._semaphore

            def _get_cached_image(self, image_path):
                data = b"test_image_data_xyz"
                b64 = base64.b64encode(data).decode()
                return data, b64

            def _calculate_dynamic_max_tokens(self, messages, router=None):
                return 4096

            def _track_usage(self, *args, **kwargs):
                pass

        processor = TestProcessor()

        test_image = tmp_path / "test.png"
        test_image.write_bytes(b"test_image_data_xyz")

        cache_get_calls: list[tuple] = []
        processor._persistent_cache.get.side_effect = lambda *args, **kwargs: (
            cache_get_calls.append((args, kwargs)) or None
        )

        async def mock_analyze(*args, **kwargs):
            return ImageAnalysis(caption="test", description="test desc")

        processor._analyze_image_with_fallback = mock_analyze  # type: ignore
        processor._persistent_cache.set.return_value = None

        # Call without document_context
        await processor.analyze_image(test_image, context="file.pdf")

        # Call again without document_context — should use same key
        await processor.analyze_image(test_image, context="file.pdf")

        assert len(cache_get_calls) == 2
        content_key_a = cache_get_calls[0][0][1]
        content_key_b = cache_get_calls[1][0][1]

        assert content_key_a == content_key_b, (
            "Without document_context, cache keys should be identical for same image"
        )

        # The key should be based on the image fingerprint
        expected_b64 = base64.b64encode(b"test_image_data_xyz").decode()
        expected_fingerprint = hashlib.sha256(expected_b64.encode()).hexdigest()
        assert expected_fingerprint in content_key_a, (
            "Cache content key should contain the image fingerprint"
        )
