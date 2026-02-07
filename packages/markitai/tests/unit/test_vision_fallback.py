"""Tests for vision analysis fallback mechanisms."""

import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from markitai.llm.vision import ImageAnalysis, VisionMixin


class TestVisionFallbackStrategies:
    """Test that _analyze_image_with_fallback tries strategies in order."""

    @pytest.fixture
    def mixin(self):
        """Create a minimal VisionMixin with mocked dependencies."""
        m = VisionMixin.__new__(VisionMixin)
        m.semaphore = MagicMock()
        m.semaphore.__aenter__ = AsyncMock()
        m.semaphore.__aexit__ = AsyncMock()
        return m

    @pytest.mark.asyncio
    async def test_instructor_success_skips_other_strategies(self, mixin):
        """When Instructor succeeds, JSON mode and two-call are not tried."""
        expected = ImageAnalysis(caption="instructor", description="desc")
        mixin._analyze_with_instructor = AsyncMock(return_value=expected)
        mixin._analyze_with_json_mode = AsyncMock()
        mixin._analyze_with_two_calls = AsyncMock()

        result = await mixin._analyze_image_with_fallback(
            [{"role": "user", "content": "test"}], "default", "img.jpg"
        )

        assert result.caption == "instructor"
        mixin._analyze_with_instructor.assert_awaited_once()
        mixin._analyze_with_json_mode.assert_not_awaited()
        mixin._analyze_with_two_calls.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_instructor_failure_falls_back_to_json_mode(self, mixin):
        """When Instructor fails, JSON mode is tried."""
        expected = ImageAnalysis(caption="json_mode", description="desc")
        mixin._analyze_with_instructor = AsyncMock(
            side_effect=Exception("Instructor failed")
        )
        mixin._analyze_with_json_mode = AsyncMock(return_value=expected)
        mixin._analyze_with_two_calls = AsyncMock()

        result = await mixin._analyze_image_with_fallback(
            [{"role": "user", "content": "test"}], "default", "img.jpg"
        )

        assert result.caption == "json_mode"
        mixin._analyze_with_json_mode.assert_awaited_once()
        mixin._analyze_with_two_calls.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_all_structured_fail_falls_back_to_two_calls(self, mixin):
        """When Instructor and JSON mode both fail, two-call method is used."""
        expected = ImageAnalysis(caption="two_call", description="desc")
        mixin._analyze_with_instructor = AsyncMock(
            side_effect=Exception("Instructor failed")
        )
        mixin._analyze_with_json_mode = AsyncMock(
            side_effect=Exception("JSON mode failed")
        )
        mixin._analyze_with_two_calls = AsyncMock(return_value=expected)

        result = await mixin._analyze_image_with_fallback(
            [{"role": "user", "content": "test"}], "default", "img.jpg"
        )

        assert result.caption == "two_call"
        mixin._analyze_with_two_calls.assert_awaited_once()


class TestVisionUnsupportedFormat:
    """Test unsupported image format handling."""

    @pytest.fixture
    def mixin(self):
        m = VisionMixin.__new__(VisionMixin)
        m.semaphore = MagicMock()
        m.semaphore.__aenter__ = AsyncMock()
        m.semaphore.__aexit__ = AsyncMock()
        return m

    @pytest.mark.asyncio
    async def test_svg_returns_placeholder_analysis(self, mixin):
        """SVG files should return a placeholder analysis without LLM call."""
        svg_path = Path("/tmp/test.svg")
        result = await mixin.analyze_image(svg_path)

        assert result.caption == "test"
        assert "not supported" in result.description

    @pytest.mark.asyncio
    async def test_bmp_returns_placeholder_analysis(self, mixin):
        """BMP files should return a placeholder analysis without LLM call."""
        bmp_path = Path("/tmp/test_image.bmp")
        result = await mixin.analyze_image(bmp_path)

        assert result.caption == "test_image"
        assert "not supported" in result.description


class TestVisionCacheCollision:
    """Test that vision cache handles similar images correctly."""

    def test_different_images_different_cache_keys(self):
        """Two different images should produce different SHA256 cache keys."""
        img1 = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        img2 = b"\x89PNG\r\n\x1a\n" + b"\xff" * 100

        key1 = hashlib.sha256(img1).hexdigest()
        key2 = hashlib.sha256(img2).hexdigest()

        assert key1 != key2

    def test_identical_images_same_cache_key(self):
        """Identical images should produce the same SHA256 cache key."""
        img = b"\x89PNG\r\n\x1a\n" + b"\xab" * 100

        key1 = hashlib.sha256(img).hexdigest()
        key2 = hashlib.sha256(img).hexdigest()

        assert key1 == key2

    def test_sha256_digest_length(self):
        """SHA256 hex digest should be exactly 64 characters."""
        data = b"any image data"
        assert len(hashlib.sha256(data).hexdigest()) == 64
