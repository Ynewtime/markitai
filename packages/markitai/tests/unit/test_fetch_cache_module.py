"""Tests for fetch_cache module — FetchCache + SPADomainCache extracted from fetch.py."""

from __future__ import annotations

from pathlib import Path


class TestFetchCacheImport:
    """Verify FetchCache is importable from fetch_cache module."""

    def test_fetch_cache_importable(self):
        from markitai.fetch_cache import FetchCache

        assert FetchCache is not None

    def test_spa_domain_cache_importable(self):
        from markitai.fetch_cache import SPADomainCache

        assert SPADomainCache is not None

    def test_make_json_safe_importable(self):
        from markitai.fetch_cache import _make_json_safe

        assert callable(_make_json_safe)


class TestFetchCacheBasicOps:
    """Test FetchCache basic get/set via the new module."""

    def test_set_and_get(self, tmp_path: Path):
        from markitai.fetch_cache import FetchCache
        from markitai.fetch_types import FetchResult

        cache = FetchCache(tmp_path / "test.db")
        try:
            result = FetchResult(
                content="# Hello", strategy_used="static", url="https://example.com"
            )
            cache.set("https://example.com", result)
            cached = cache.get("https://example.com")
            assert cached is not None
            assert cached.content == "# Hello"
            assert cached.strategy_used == "static"
            assert cached.cache_hit is True
        finally:
            cache.close()

    def test_get_missing_returns_none(self, tmp_path: Path):
        from markitai.fetch_cache import FetchCache

        cache = FetchCache(tmp_path / "test.db")
        try:
            assert cache.get("https://nonexistent.com") is None
        finally:
            cache.close()


class TestSPADomainCacheBasicOps:
    """Test SPADomainCache basic ops via the new module."""

    def test_record_and_check(self, tmp_path: Path):
        from markitai.fetch_cache import SPADomainCache

        cache = SPADomainCache(tmp_path / "spa.json")
        cache.record_spa_domain("https://spa-app.example.com/page")
        assert cache.is_known_spa("https://spa-app.example.com/other") is True
        assert cache.is_known_spa("https://other.com") is False

    def test_clear(self, tmp_path: Path):
        from markitai.fetch_cache import SPADomainCache

        cache = SPADomainCache(tmp_path / "spa.json")
        cache.record_spa_domain("https://example.com")
        cleared = cache.clear()
        assert cleared >= 1
        assert cache.is_known_spa("https://example.com") is False


class TestMakeJsonSafe:
    """Test _make_json_safe via the new module."""

    def test_primitives(self):
        from markitai.fetch_cache import _make_json_safe

        assert _make_json_safe(None) is None
        assert _make_json_safe("hello") == "hello"
        assert _make_json_safe(42) == 42
        assert _make_json_safe(True) is True

    def test_path_converted_to_string(self):
        from markitai.fetch_cache import _make_json_safe

        assert _make_json_safe(Path("/foo/bar")) == str(Path("/foo/bar"))

    def test_nested_dict(self):
        from markitai.fetch_cache import _make_json_safe

        result = _make_json_safe({"path": Path("/a"), "count": 1})
        assert result == {"path": str(Path("/a")), "count": 1}


class TestBackwardCompatFetchCache:
    """Verify FetchCache and SPADomainCache are still importable from markitai.fetch."""

    def test_fetch_cache_from_fetch(self):
        from markitai.fetch import FetchCache

        assert FetchCache is not None

    def test_spa_domain_cache_from_fetch(self):
        from markitai.fetch import SPADomainCache

        assert SPADomainCache is not None
